"""
evaluation.py – Evaluation metrics for route prediction quality.

Metrics:
  1. Edge-overlap Jaccard (IoU of edge sets)
  2. Edge-overlap F1     (precision + recall of edges)
  3. Path-length ratio   (predicted / expert length)
  4. Top-1 next-edge accuracy (junction-level)
  5. Top-3 next-edge accuracy
  6. Held-out log-likelihood under the learned IRL policy
"""

import math
import numpy as np
import networkx as nx
from typing import Optional

from config import N_FEATURES


# ──────────────────────────────────────────────────────────────
# Helper: node list → edge set
# ──────────────────────────────────────────────────────────────

def nodes_to_edge_set(nodes: list[int]) -> set[tuple]:
    return {(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)}


def route_length(G: nx.MultiDiGraph, nodes: list[int]) -> float:
    """Sum of 'length' attributes along node sequence."""
    total = 0.0
    for i in range(len(nodes)-1):
        u, v = nodes[i], nodes[i+1]
        if G.has_edge(u, v):
            # Minimum length over parallel edges
            edges = G[u][v]
            total += min(d.get("length", 0) for d in edges.values())
    return total


# ──────────────────────────────────────────────────────────────
# Metric functions
# ──────────────────────────────────────────────────────────────

def edge_jaccard(pred_nodes: list[int], expert_nodes: list[int]) -> float:
    """Jaccard similarity of edge sets."""
    pred_edges   = nodes_to_edge_set(pred_nodes)
    expert_edges = nodes_to_edge_set(expert_nodes)
    if not pred_edges and not expert_edges:
        return 1.0
    inter = pred_edges & expert_edges
    union = pred_edges | expert_edges
    return len(inter) / len(union)


def edge_f1(pred_nodes: list[int], expert_nodes: list[int]) -> float:
    """F1 score of edge sets."""
    pred_edges   = nodes_to_edge_set(pred_nodes)
    expert_edges = nodes_to_edge_set(expert_nodes)
    if not pred_edges or not expert_edges:
        return 0.0
    tp = len(pred_edges & expert_edges)
    precision = tp / len(pred_edges)
    recall    = tp / len(expert_edges)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def path_length_ratio(G: nx.MultiDiGraph,
                       pred_nodes: list[int],
                       expert_nodes: list[int]) -> float:
    """Ratio of predicted route length to expert route length."""
    pred_len   = route_length(G, pred_nodes)
    expert_len = route_length(G, expert_nodes)
    if expert_len == 0:
        return 1.0
    return pred_len / expert_len


def policy_step_stats(G, trajectories, weights, mdp=None, ks=(1, 3)):
    """
    Destination-aware per-step stats under the MaxEnt policy pi_w:
      average log pi(a_t|s_t), and top-k accuracy of the expert's choice
      (only at junctions that offer a real choice).
    """
    from mdp import RouteMDP
    mdp = mdp or RouteMDP(G)
    lls, hits = [], {k: [] for k in ks}
    for traj in trajectories:
        for lp, rank, n_opt in mdp.step_log_probs(weights, traj):
            lls.append(lp)
            if n_opt > 1:
                for k in ks:
                    hits[k].append(rank < k)
    return float(np.mean(lls)), {k: float(np.mean(v)) for k, v in hits.items()}


# ──────────────────────────────────────────────────────────────
# Full evaluation suite
# ──────────────────────────────────────────────────────────────

def evaluate_all(G: nx.MultiDiGraph,
                  test_trajectories: list[dict],
                  irl_weights: np.ndarray,
                  myopic_clf=None,
                  n_eval: int = None,
                  oracle_weights: np.ndarray = None) -> dict:
    """
    Route-level metrics for IRL and baselines on held-out trajectories, plus
    junction-level metrics (log-likelihood, top-k).  `oracle_weights`, if given,
    adds the simulator's true reward as an upper bound.
    """
    from baselines import dijkstra_shortest, dijkstra_fastest
    from mdp import RouteMDP

    mdp = RouteMDP(G)
    test = test_trajectories if n_eval is None else test_trajectories[:n_eval]
    print(f"[Evaluating {len(test)} test trajectories ...]")

    methods = {
        "IRL (MaxEnt)":      lambda o, d: mdp.most_likely_route(o, d, irl_weights),
        "Dijkstra-Shortest": lambda o, d: dijkstra_shortest(G, o, d),
        "Dijkstra-Fastest":  lambda o, d: dijkstra_fastest(G, o, d),
        "Myopic-Classifier": lambda o, d: myopic_clf.predict_route(G, o, d),
    }
    if oracle_weights is not None:
        methods["Oracle (true reward)"] = lambda o, d: mdp.most_likely_route(
            o, d, oracle_weights)

    results = {}
    for name, route_fn in methods.items():
        J, F, L = [], [], []
        for traj in test:
            route = route_fn(traj["origin"], traj["destination"])
            if len(route) < 2:
                J.append(0.0); F.append(0.0); L.append(np.nan)
                continue
            J.append(edge_jaccard(route, traj["nodes"]))
            F.append(edge_f1(route, traj["nodes"]))
            L.append(path_length_ratio(G, route, traj["nodes"]))
        results[name] = {
            "Edge Jaccard": float(np.mean(J)),
            "Edge F1": float(np.mean(F)),
            "Path Length Ratio": float(np.nanmean(L)),
        }

    print("[Evaluating junction-level metrics ...]")
    ll, topk = policy_step_stats(G, test, irl_weights, mdp)
    results["IRL (MaxEnt)"].update({
        "Top-1 Accuracy": topk[1], "Top-3 Accuracy": topk[3], "Log-Likelihood": ll})
    if oracle_weights is not None:
        ll, topk = policy_step_stats(G, test, oracle_weights, mdp)
        results["Oracle (true reward)"].update({
            "Top-1 Accuracy": topk[1], "Top-3 Accuracy": topk[3], "Log-Likelihood": ll})
    if myopic_clf is not None and myopic_clf.trained:
        ll, topk = myopic_clf.step_stats(test)
        results["Myopic-Classifier"].update({
            "Top-1 Accuracy": topk[1], "Top-3 Accuracy": topk[3], "Log-Likelihood": ll})
    return results


def print_results_table(results: dict):
    metrics = ["Edge Jaccard", "Edge F1", "Path Length Ratio",
               "Top-1 Accuracy", "Top-3 Accuracy", "Log-Likelihood"]
    header = f"{'Method':<22} " + " ".join(f"{m:>16}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for method, scores in results.items():
        row = f"{method:<22} "
        for m in metrics:
            val = scores.get(m, "-")
            row += f"{val:>16.4f} " if isinstance(val, float) else f"{str(val):>16} "
        print(row)
    print("=" * len(header))
