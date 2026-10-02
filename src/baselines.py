"""
baselines.py – Route planning baselines for comparison with IRL.

Baselines implemented:
  1. Dijkstra – shortest path (minimise distance)
  2. Dijkstra – fastest path (minimise travel time)
  3. Myopic supervised classifier (junction-level, logistic regression)
"""

import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import N_FEATURES


# ──────────────────────────────────────────────────────────────
# 1. Dijkstra — Shortest path
# ──────────────────────────────────────────────────────────────

def dijkstra_shortest(G: nx.MultiDiGraph,
                       origin: int,
                       destination: int) -> list[int]:
    """Return the node list for the shortest-distance path."""
    try:
        return nx.shortest_path(G, origin, destination, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


# ──────────────────────────────────────────────────────────────
# 2. Dijkstra — Fastest path
# ──────────────────────────────────────────────────────────────

def _travel_time_weight(u, v, d):
    """Edge weight = travel_time from features, else fallback."""
    best = min(d.values(), key=lambda ed: ed.get("length", 1e9))
    feat = best.get("features")
    if feat is not None and len(feat) >= 3:
        tt = feat[2]   # travel_time is index 2
        return max(tt, 1.0)
    length = best.get("length", 100.0)
    speed  = 30.0      # default 30 km/h
    return (length / 1000.0) / speed * 3600.0


def dijkstra_fastest(G: nx.MultiDiGraph,
                      origin: int,
                      destination: int) -> list[int]:
    """Return the node list for the fastest-time path."""
    try:
        return nx.shortest_path(G, origin, destination,
                                weight=_travel_time_weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


# ──────────────────────────────────────────────────────────────
# 3. Myopic Supervised Classifier
# ──────────────────────────────────────────────────────────────

class MyopicClassifier:
    """
    Junction-level supervised learner.

    At each intersection it sees the features of every candidate next edge
    (road features + turn from the incoming edge) and is trained with the
    expert-chosen edge as the positive example.  It has no notion of the
    destination or of long-horizon consequences -- the contrast to IRL.
    """

    def __init__(self, G: nx.MultiDiGraph = None):
        from mdp import RouteMDP
        self.mdp     = RouteMDP(G) if G is not None else None
        self.model   = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        self.scaler  = StandardScaler()
        self.trained = False

    def _candidates(self, prev_e, node_idx):
        """Candidate next edges and their features at a junction."""
        m = self.mdp
        es = np.array(m.out_edges[node_idx], dtype=int)
        if prev_e is None:
            return es, m.F_edge[es]
        return es, m.T_feat[[m.tid[(prev_e, int(e))] for e in es]]

    def train(self, trajectories: list[dict], G: nx.MultiDiGraph = None):
        if self.mdp is None:
            from mdp import RouteMDP
            self.mdp = RouteMDP(G)
        m = self.mdp
        X, y = [], []
        for traj in trajectories:
            ids = m.traj_edge_ids(traj)
            prev = None
            for e in ids:
                es, F = self._candidates(prev, int(m.e_src[e]))
                if len(es) > 1:
                    for cand, f in zip(es, F):
                        X.append(f)
                        y.append(1 if cand == e else 0)
                prev = e
        X, y = np.array(X), np.array(y)
        self.model.fit(self.scaler.fit_transform(X), y)
        self.trained = True
        print(f"[MyopicClassifier] Trained on {len(X)} samples | "
              f"train acc = {self.model.score(self.scaler.transform(X), y):.3f}")

    def scores(self, F):
        return self.model.decision_function(self.scaler.transform(F))

    def predict_route(self, G, origin, destination, max_steps: int = 300):
        """Follow the highest-scoring edge at each junction (no revisits)."""
        m = self.mdp
        dest = m.nid[destination]
        node, prev = m.nid[origin], None
        route, visited = [origin], {node}
        for _ in range(max_steps):
            if node == dest:
                break
            es, F = self._candidates(prev, node)
            if len(es) == 0:
                break
            order = np.argsort(-self.scores(F))
            nxt = None
            for j in order:
                nn = int(m.e_dst[es[j]])
                if nn not in visited or nn == dest:
                    nxt, prev, node = int(es[j]), int(es[j]), nn
                    break
            if nxt is None:
                break
            route.append(m.nodes[node])
            visited.add(node)
        return route

    def step_stats(self, trajectories, k_list=(1, 3)):
        """Average per-step log-likelihood (local softmax over scores) and top-k acc."""
        m = self.mdp
        lls, hits = [], {k: [] for k in k_list}
        for traj in trajectories:
            prev = None
            for e in m.traj_edge_ids(traj):
                es, F = self._candidates(prev, int(m.e_src[e]))
                sc = self.scores(F)
                j = int(np.where(es == e)[0][0])
                lls.append(sc[j] - np.logaddexp.reduce(sc))
                if len(es) > 1:
                    rank = int((sc > sc[j]).sum())
                    for k in k_list:
                        hits[k].append(rank < k)
                prev = e
        return float(np.mean(lls)), {k: float(np.mean(v)) for k, v in hits.items()}
