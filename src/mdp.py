"""
mdp.py – Edge-state MDP on top of an OSM road graph.

State  s_t = incoming directed edge e_{t-1}   (equivalently: node + incoming edge)
Action a_t = next outgoing edge e_t at the head of e_{t-1}
Reward R(s, a) = w^T phi(s, a)
         phi(s, a) = road features of the chosen edge  +  turn features of the
         turn between the incoming and the chosen edge.
Goal   Reaching the destination node is absorbing (value 0).

Everything is vectorised over the list of all transitions (e -> e') so soft
value iteration, exact expected feature counts and most-likely-route search
run in milliseconds on the ~4k-edge SF graph.
"""

import heapq
import numpy as np
import networkx as nx
from typing import Optional

from config import (N_FEATURES, FEATURES, FEATURES_V1, FEATURE_SET,
                    DEST_FEATURES)

NEG = -1e4          # "minus infinity" that survives arithmetic
# Graph edges always store the v1 10-vector; turn slots are filled per transition.
_B = {n: i for i, n in enumerate(FEATURES_V1)}
TURN_IDX = [_B["turn_straight"], _B["turn_slight"], _B["turn_sharp"]]


def _road_name(d: dict) -> str:
    n = d.get("name")
    if isinstance(n, (list, tuple)):
        n = n[0] if n else None
    return str(n) if n else ""


def _angle(b_in: np.ndarray, b_out: np.ndarray) -> np.ndarray:
    """Absolute turn angle in [0, 180] between two bearings (degrees)."""
    d = (b_out - b_in + 360.0) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


class RouteMDP:
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self.nodes = list(G.nodes())
        self.nid = {n: i for i, n in enumerate(self.nodes)}

        # ---- edge arrays -------------------------------------------------
        self.edges = list(G.edges(keys=True))                   # (u, v, k)
        self.eid = {e: i for i, e in enumerate(self.edges)}
        E = len(self.edges)
        self.E = E
        self.e_src = np.array([self.nid[u] for u, _, _ in self.edges])
        self.e_dst = np.array([self.nid[v] for _, v, _ in self.edges])
        F_base = np.zeros((E, len(FEATURES_V1)))
        self.bearing = np.zeros(E)
        name_ids, name_of = {}, np.full(E, -1, dtype=int)
        for i, (u, v, k) in enumerate(self.edges):
            d = G[u][v][k]
            F_base[i] = d.get("features", np.zeros(len(FEATURES_V1)))
            self.bearing[i] = float(d.get("bearing", 0.0))
            nm = _road_name(d)
            if nm:
                name_of[i] = name_ids.setdefault(nm, len(name_ids))
        self.node_xy = np.array([[G.nodes[n].get("x", 0.0), G.nodes[n].get("y", 0.0)]
                                 for n in self.nodes])
        self.node_xy[:, 0] *= np.cos(np.radians(self.node_xy[:, 1].mean()))
        self.dest_cols = [FEATURES.index(f) for f in DEST_FEATURES]
        self._D_cache = {}

        # out-edges per node
        self.out_edges = [[] for _ in self.nodes]
        for i in range(E):
            self.out_edges[self.e_src[i]].append(i)

        # ---- transitions (e -> e') --------------------------------------
        t_src, t_dst = [], []
        for e in range(E):
            for e2 in self.out_edges[self.e_dst[e]]:
                t_src.append(e)
                t_dst.append(e2)
        self.t_src = np.array(t_src)
        self.t_dst = np.array(t_dst)
        self.T = len(t_src)
        self.tid = {(int(a), int(b)): i
                    for i, (a, b) in enumerate(zip(t_src, t_dst))}

        ang = _angle(self.bearing[self.t_src], self.bearing[self.t_dst])
        T_base = F_base[self.t_dst].copy()
        T_base[:, TURN_IDX[0]] = (ang < 30).astype(float)
        T_base[:, TURN_IDX[1]] = ((ang >= 30) & (ang < 90)).astype(float)
        T_base[:, TURN_IDX[2]] = (ang >= 90).astype(float)

        fam = np.zeros(E)
        if "edge_familiarity" in FEATURES:
            import pickle
            from config import FAMILIARITY_PATH
            with open(FAMILIARITY_PATH, "rb") as fh:
                cnt = pickle.load(fh)["counts"]
            fam = np.log1p(np.array([cnt.get(e, 0) for e in self.edges], float))
            fam /= max(fam.max(), 1.0)
            self.familiarity = fam

        if FEATURE_SET == "v1":
            self.F_edge, self.T_feat = F_base, T_base
        else:
            same = ((name_of[self.t_src] >= 0) &
                    (name_of[self.t_src] == name_of[self.t_dst])).astype(float)
            self.F_edge = np.zeros((E, N_FEATURES))
            self.T_feat = np.zeros((self.T, N_FEATURES))
            for j, f in enumerate(FEATURES):
                if f == "step":
                    self.F_edge[:, j] = 1.0
                    self.T_feat[:, j] = 1.0
                elif f == "same_road_name":
                    self.T_feat[:, j] = same
                elif f == "edge_familiarity":
                    self.F_edge[:, j] = fam
                    self.T_feat[:, j] = fam[self.t_dst]
                elif f in DEST_FEATURES:
                    pass                               # filled per destination
                elif f in _B and f.startswith("turn_"):
                    self.T_feat[:, j] = T_base[:, _B[f]]
                else:
                    self.F_edge[:, j] = F_base[:, _B[f]]
                    self.T_feat[:, j] = T_base[:, _B[f]]

        # segment starts for reduceat (edges that have >=1 outgoing transition)
        counts = np.bincount(self.t_src, minlength=E)
        self.has_out = counts > 0
        self.seg_start = np.concatenate([[0], np.cumsum(counts)[:-1]])[self.has_out]

    # ------------------------------------------------------------------
    # Destination-dependent features
    # ------------------------------------------------------------------
    def dest_feat(self, dest_node: int) -> Optional[np.ndarray]:
        """(T, n_dest_features) matrix, or None when the feature set has none.
        away_from_goal[t] = 1 if the transition ends farther (straight line)
        from the destination than where the incoming edge ended."""
        if not self.dest_cols:
            return None
        D = self._D_cache.get(dest_node)
        if D is None:
            dist = np.linalg.norm(self.node_xy - self.node_xy[self.nid[dest_node]],
                                  axis=1)
            d_to = dist[self.e_dst[self.t_dst]]
            d_from = dist[self.e_dst[self.t_src]]
            D = (d_to > d_from + 1e-9).astype(float)[:, None]
            if len(self._D_cache) > 64:
                self._D_cache.clear()
            self._D_cache[dest_node] = D
        return D

    def reward_T(self, w: np.ndarray, dest_node: int) -> np.ndarray:
        """Transition rewards r(e, e') for a given destination."""
        r = self.T_feat @ w
        D = self.dest_feat(dest_node)
        if D is not None:
            r = r + D @ w[self.dest_cols]
        return r

    # ------------------------------------------------------------------
    # Soft value iteration (backward pass)
    # ------------------------------------------------------------------
    def soft_values(self, w: np.ndarray, dest_node: int,
                    n_iter: int = 150, tol: float = 1e-5,
                    r_T: Optional[np.ndarray] = None) -> np.ndarray:
        """
        V(e) = log sum_{e'} exp( r(e, e') + V(e') ),  V(e) = 0 if e enters dest.
        Returned vector is indexed by edge id (value of *having arrived via e*).
        """
        if r_T is None:
            r_T = self.reward_T(w, dest_node)
        term = self.e_dst == self.nid[dest_node]
        V = np.full(self.E, NEG)
        V[term] = 0.0
        for _ in range(n_iter):
            q = r_T + V[self.t_dst]
            Vn = np.full(self.E, NEG)
            Vn[self.has_out] = np.logaddexp.reduceat(q, self.seg_start)
            Vn[term] = 0.0
            np.maximum(Vn, NEG, out=Vn)
            if np.max(np.abs(Vn - V)) < tol:
                V = Vn
                break
            V = Vn
        return V

    def soft_values_levels(self, w: np.ndarray, dest_node: int, H: int,
                           r_T: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Finite-horizon soft values.  L[k][e] = log-sum over all continuations of
        at most k further edges after arriving via e (absorbed at the goal).
        Always finite for any w (unlike the infinite-horizon sum, which
        diverges when a cycle has non-negative reward).  Shape (H, E).
        """
        if r_T is None:
            r_T = self.reward_T(w, dest_node)
        term = self.e_dst == self.nid[dest_node]
        L = np.full((H, self.E), NEG)
        L[0, term] = 0.0
        for k in range(1, H):
            q = r_T + L[k - 1][self.t_dst]
            Vn = np.full(self.E, NEG)
            Vn[self.has_out] = np.logaddexp.reduceat(q, self.seg_start)
            Vn[term] = 0.0
            np.maximum(Vn, NEG, out=Vn)
            L[k] = Vn
        return L

    def start_logits(self, w: np.ndarray, V: np.ndarray, origin: int):
        """Log-weights of the first edge out of `origin` (no incoming edge)."""
        es = np.array(self.out_edges[self.nid[origin]], dtype=int)
        if len(es) == 0:
            return es, np.array([])
        return es, self.F_edge[es] @ w + V[es]

    # ------------------------------------------------------------------
    # Likelihood + exact expected feature counts (forward pass)
    # ------------------------------------------------------------------
    def expected_counts(self, w: np.ndarray, L: np.ndarray, origin: int,
                        dest_node: int, acc_T: np.ndarray, acc_0: np.ndarray,
                        r_T: Optional[np.ndarray] = None,
                        acc_D: Optional[np.ndarray] = None) -> float:
        """
        Exact forward pass for the finite-horizon policy.  Adds the expected
        transition / start-edge mass for one (o, d) pair to acc_T / acc_0 and
        returns log Z(o) = log sum over all <=H-edge paths from o to d.
        `acc_D` (n_dest,) accumulates expected destination-dependent features.
        """
        H = L.shape[0]
        if r_T is None:
            r_T = self.reward_T(w, dest_node)
        D_feat = self.dest_feat(dest_node) if acc_D is not None else None
        es = np.array(self.out_edges[self.nid[origin]], dtype=int)
        if len(es) == 0:
            return NEG
        lg = self.F_edge[es] @ w + L[H - 1][es]
        if lg.max() <= NEG / 2:
            return NEG
        logZ = np.logaddexp.reduce(lg)
        p0 = np.exp(lg - logZ)
        acc_0[es] += p0

        term = self.e_dst == self.nid[dest_node]
        live_src = ~term[self.t_src]
        D = np.zeros(self.E)
        D[es] = p0
        D[term] = 0.0
        for k in range(H - 1, 0, -1):            # k = steps remaining at source
            pi = np.exp(r_T + L[k - 1][self.t_dst] - L[k][self.t_src])
            pi[~live_src | (L[k][self.t_src] <= NEG / 2)] = 0.0
            m = D[self.t_src] * pi
            acc_T += m
            if D_feat is not None:
                acc_D += m @ D_feat
            D = np.bincount(self.t_dst, weights=m, minlength=self.E)
            D[term] = 0.0
            if D.sum() < 1e-8:
                break
        return float(logZ)

    # ------------------------------------------------------------------
    # Per-trajectory helpers
    # ------------------------------------------------------------------
    def traj_edge_ids(self, traj: dict) -> list[int]:
        return [self.eid[tuple(e)] for e in traj["edges"]]

    def traj_features(self, traj: dict) -> np.ndarray:
        """Sum of phi over the whole trajectory (start edge + transitions)."""
        ids = self.traj_edge_ids(traj)
        f = self.F_edge[ids[0]].copy()
        D = self.dest_feat(traj["destination"])
        for a, b in zip(ids[:-1], ids[1:]):
            t = self.tid[(a, b)]
            f += self.T_feat[t]
            if D is not None:
                f[self.dest_cols] += D[t]
        return f

    def step_log_probs(self, w: np.ndarray, traj: dict, H: int = 80):
        """
        For each step of an expert trajectory return
        (log pi of chosen edge, rank of chosen edge among alternatives, #options)
        under the finite-horizon policy used for training.
        """
        L = self.soft_values_levels(w, traj["destination"], H)
        ids = self.traj_edge_ids(traj)
        r_T = self.reward_T(w, traj["destination"])
        out = []
        es = np.array(self.out_edges[self.nid[traj["origin"]]], dtype=int)
        lg = self.F_edge[es] @ w + L[H - 1][es]
        out.append(self._lp_rank(lg, int(np.where(es == ids[0])[0][0])))
        for i, (a, b) in enumerate(zip(ids[:-1], ids[1:]), start=1):
            es2 = np.array(self.out_edges[self.e_dst[a]], dtype=int)
            k = max(H - 1 - i, 0)
            lg2 = np.array([r_T[self.tid[(a, int(e))]] + L[k][e] for e in es2])
            out.append(self._lp_rank(lg2, int(np.where(es2 == b)[0][0])))
        return out

    @staticmethod
    def _lp_rank(logits: np.ndarray, chosen: int):
        lp = logits[chosen] - np.logaddexp.reduce(logits)
        rank = int((logits > logits[chosen]).sum())
        return float(lp), rank, len(logits)

    # ------------------------------------------------------------------
    # Route generation
    # ------------------------------------------------------------------
    def max_reward_route(self, origin: int, destination: int,
                         w: np.ndarray) -> list[int]:
        """
        MAP route under the finite-horizon MaxEnt model: P(path) ~ exp(w.f(path)),
        so the most likely path is the maximum-reward one.  Dijkstra over
        edge-states with cost -r(e, e') (valid when every reward is <= 0).
        Exact and independent of soft-value convergence.
        """
        r_T = self.reward_T(w, destination)
        es = np.array(self.out_edges[self.nid[origin]], dtype=int)
        term = self.e_dst == self.nid[destination]
        dist, prev, pq = {}, {}, []
        for e in es:
            c = float(-(self.F_edge[e] @ w))
            dist[int(e)] = c
            prev[int(e)] = None
            heapq.heappush(pq, (c, int(e)))
        goal = None
        while pq:
            c, e = heapq.heappop(pq)
            if c > dist.get(e, np.inf):
                continue
            if term[e]:
                goal = e
                break
            for e2 in self.out_edges[self.e_dst[e]]:
                nc = c - float(r_T[self.tid[(e, e2)]])
                if nc < dist.get(e2, np.inf):
                    dist[e2] = nc
                    prev[e2] = e
                    heapq.heappush(pq, (nc, e2))
        if goal is None:
            return []
        chain = []
        e = goal
        while e is not None:
            chain.append(e)
            e = prev[e]
        chain.reverse()
        nodes = [self.nodes[self.e_src[chain[0]]]]
        nodes += [self.nodes[self.e_dst[e]] for e in chain]
        return nodes

    def most_likely_route(self, origin: int, destination: int,
                          w: np.ndarray) -> list[int]:
        """
        Viterbi route under pi_w: the path maximising prod pi(a|s).
        Uses the exact max-reward Dijkstra when all rewards are <= 0 (the
        infinite-horizon soft values below can diverge: with branching b the
        sum over all paths grows like (b*e^r)^k, so V need not exist).
        """
        r_T = self.reward_T(w, destination)
        if r_T.max() <= 0 and (self.F_edge @ w).max() <= 0:
            return self.max_reward_route(origin, destination, w)
        V = self.soft_values(w, destination)
        es, lg = self.start_logits(w, V, origin)
        if len(es) == 0:
            return []
        v0 = np.logaddexp.reduce(lg)
        term = self.e_dst == self.nid[destination]

        dist = {}
        prev = {}
        pq = []
        for e, l in zip(es, lg):
            if l <= NEG / 2:
                continue
            c = float(v0 - l)
            dist[int(e)] = c
            prev[int(e)] = None
            heapq.heappush(pq, (c, int(e)))
        goal = None
        while pq:
            c, e = heapq.heappop(pq)
            if c > dist.get(e, np.inf):
                continue
            if term[e]:
                goal = e
                break
            if not self.has_out[e]:
                continue
            for e2 in self.out_edges[self.e_dst[e]]:
                t = self.tid[(e, e2)]
                q = r_T[t] + V[e2]
                if q <= NEG / 2:
                    continue
                nc = c + max(0.0, float(V[e] - q))
                if nc < dist.get(e2, np.inf):
                    dist[e2] = nc
                    prev[e2] = e
                    heapq.heappush(pq, (nc, e2))
        if goal is None:
            return []
        chain = []
        e = goal
        while e is not None:
            chain.append(e)
            e = prev[e]
        chain.reverse()
        nodes = [self.nodes[self.e_src[chain[0]]]]
        nodes += [self.nodes[self.e_dst[e]] for e in chain]
        return nodes

    # kept for API / dashboard compatibility
    def sample_route(self, origin: int, destination: int, weights: np.ndarray,
                     **_) -> list[int]:
        return self.most_likely_route(origin, destination, weights)

    def sample_trajectory(self, origin: int, destination: int, w: np.ndarray,
                          rng: np.random.Generator, max_steps: int = 120):
        """Draw one trajectory from the MaxEnt policy (stochastic expert)."""
        V = self.soft_values(w, destination)
        r_T = self.reward_T(w, destination)
        es, lg = self.start_logits(w, V, origin)
        if len(es) == 0 or lg.max() <= NEG / 2:
            return None
        p = np.exp(lg - np.logaddexp.reduce(lg))
        e = int(rng.choice(es, p=p))
        chain = [e]
        term_node = self.nid[destination]
        for _ in range(max_steps):
            if self.e_dst[e] == term_node:
                return chain
            es2 = np.array(self.out_edges[self.e_dst[e]], dtype=int)
            if len(es2) == 0:
                return None
            lg2 = np.array([r_T[self.tid[(e, int(x))]] + V[x] for x in es2])
            if lg2.max() <= NEG / 2:
                return None
            p2 = np.exp(lg2 - np.logaddexp.reduce(lg2))
            e = int(rng.choice(es2, p=p2))
            chain.append(e)
        return None
