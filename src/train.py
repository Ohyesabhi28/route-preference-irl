"""
train.py – End-to-end pipeline: load graph + demonstrations, train MaxEnt IRL,
train the myopic baseline, evaluate, save.

Region is chosen with the IRL_REGION environment variable (default: porto).
"""

import os
import sys
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    WEIGHTS_SAVE_PATH, RESULTS_DIR, FEATURES, TRUE_WEIGHTS,
    REGION_NAME, SYNTHETIC
)
import region
from maxent_irl import MaxEntIRL
from baselines import MyopicClassifier
from evaluation import evaluate_all, print_results_table
from metrics import summarize_per_trip


def main():
    print("=" * 60)
    print(f"  IRL Route Preference Learning — {REGION_NAME}"
          f" ({'synthetic' if SYNTHETIC else 'real GPS'} demonstrations)")
    print("=" * 60)

    print("\n[1/5] Loading road network ...")
    G = region.load_graph()
    print(f"      Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    print("\n[2/5] Loading demonstrations ...")
    data = region.load_data(G)
    train_traj, test_traj = data["train"], data["test"]
    print(f"      Train: {len(train_traj)}  Test: {len(test_traj)}")

    print("\n[3/5] Training MaxEnt IRL ...")
    if os.path.exists(WEIGHTS_SAVE_PATH):
        print("      Loading cached IRL weights ...")
        irl = MaxEntIRL.load(G)
    else:
        irl = MaxEntIRL(G)
        irl.train(train_traj)

    oracle = None
    if SYNTHETIC:
        true_w = np.array(TRUE_WEIGHTS)
        oracle = true_w
        print("\n      Feature weights comparison:")
        print(f"      {'Feature':<20} {'True':>10} {'Learned':>10}")
        print("      " + "-" * 42)
        for feat, tw, lw in zip(FEATURES, true_w, irl.weights):
            print(f"      {feat:<20} {tw:>10.4f} {lw:>10.4f}")
        cos = float(true_w @ irl.weights /
                    (np.linalg.norm(true_w) * np.linalg.norm(irl.weights) + 1e-12))
        print(f"      cosine(true, learned) = {cos:.4f}")
    else:
        print("\n      Learned reward weights (real drivers):")
        for feat, lw in zip(FEATURES, irl.weights):
            print(f"      {feat:<20} {lw:>+10.4f}")

    print("\n[4/5] Training Myopic Classifier ...")
    clf = MyopicClassifier(G)
    clf.train(train_traj)

    print("\n[5/5] Evaluating ...")
    results, per_trip = evaluate_all(G, test_traj, irl.weights, clf,
                                     oracle_weights=oracle, return_per_trip=True)
    print_results_table(results)

    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n      Results saved -> {results_path}")

    # per-trip arrays + bootstrap CIs + paired tests vs Dijkstra-Shortest
    ci = summarize_per_trip(per_trip, reference="Dijkstra-Shortest")
    ci_path = os.path.join(RESULTS_DIR, "evaluation_ci.json")
    with open(ci_path, "w") as f:
        json.dump(ci, f, indent=2)
    flat = {f"{m}|{k}": v for m, d in per_trip.items() for k, v in d.items()}
    np.savez(os.path.join(RESULTS_DIR, "per_trip_metrics.npz"), **flat)
    print(f"      Per-trip metrics + CIs -> {ci_path}")
    for m in ("Edge Jaccard", "Edge F1"):
        s = ci["IRL (MaxEnt)"][m]
        print(f"      IRL {m}: {s['mean']:.4f}  95% CI [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]"
              f"  (n={s['n']})  vs Dijkstra-Shortest p="
              f"{ci['IRL (MaxEnt)'][m].get('vs_Dijkstra-Shortest', {}).get('p_value', float('nan')):.3g}")
    with open(os.path.join(RESULTS_DIR, "irl_history.pkl"), "wb") as f:
        pickle.dump(irl.history, f)

    irl_j = results["IRL (MaxEnt)"]["Edge Jaccard"]
    dks_j = results["Dijkstra-Shortest"]["Edge Jaccard"]
    print("\n" + "=" * 60)
    print(f"  IRL Jaccard: {irl_j:.4f}  vs  Dijkstra-Shortest: {dks_j:.4f}  "
          f"({(irl_j - dks_j) / (dks_j + 1e-9) * 100:+.1f}%)")
    print("=" * 60)
    return results, irl, G, data


if __name__ == "__main__":
    main()
