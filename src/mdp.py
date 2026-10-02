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

from config import N_FEATURES, FEATURES

NEG = -1e4          # "minus infinity" that survives arithmetic
TURN_IDX = [FEATURES.index("turn_straight"),
            FEATURES.index("turn_slight"),
            FEATURES.index("turn_sharp")]


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
        self.F_edge = np.zeros((E, N_FEATURES))
        self.bearing = np.zeros(E)
        for i, (u, v, k) in enumerate(self.edges):
            d = G[u][v][k]
            self.F_edge[i] = d.get("features", np.zeros(N_FEATURES))
            self.bearing[i] = float(d.get("bearing", 0.0))

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
        self.T_feat = self.F_edge[self.t_dst].copy()
        self.T_feat[:, TURN_IDX[0]] = (ang < 30).astype(float)
        self.T_feat[:, TURN_IDX[1]] = ((ang >= 30) & (ang < 90)).astype(float)
        self.T_feat[:, TURN_IDX[2]] = (ang >= 90).astype(float)

        # segment starts for reduceat (edges that have >=1 outgoing transition)
        counts = np.bincount(self.t_src, minlength=E)
        self.has_out = counts > 0
        self.seg_start = np.concatenate([[0], np.cumsum(counts)[:-1]])[self.has_out]

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
            r_T = self.T_feat @ w
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
            r_T = self.T_feat @ w
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
                        r_T: Optional[np.ndarray] = None) -> float:
        """
        Exact forward pass for the finite-horizon policy.  Adds the expected
        transition / start-edge mass for one (o, d) pair to acc_T / acc_0 and
        returns log Z(o) = log sum over all <=H-edge paths from o to d.
        """
        H = L.shape[0]
        if r_T is None:
            r_T = self.T_feat @ w
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
        for a, b in zip(ids[:-1], ids[1:]):
            f += self.T_feat[self.tid[(a, b)]]
        return f

    def step_log_probs(self, w: np.ndarray, traj: dict, H: int = 80):
        """
        For each step of an expert trajectory return
        (log pi of chosen edge, rank of chosen edge among alternatives, #options)
        under the finite-horizon policy used for training.
        """
        L = self.soft_values_levels(w, traj["destination"], H)
        ids = self.traj_edge_ids(traj)
        r_T = self.T_feat @ w
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
    def most_likely_route(self, origin: int, destination: int,
                          w: np.ndarray) -> list[int]:
        """
        Viterbi route under pi_w: the path maximising prod pi(a|s).
        Dijkstra over edge-states with non-negative cost V(s) - Q(s, a).
        """
        V = self.soft_values(w, destination)
        r_T = self.T_feat @ w
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
        r_T = self.T_feat @ w
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
