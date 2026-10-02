"""
train.py – End-to-end training pipeline for IRL Route Preference Learning.

Steps:
  1. Build / load the SF OSM graph
  2. Generate / load synthetic expert trajectories
  3. Train MaxEnt IRL
  4. Train myopic baseline classifier
  5. Evaluate all methods
  6. Save results
"""

import os
import json
import pickle
import numpy as np
import sys

# Make src importable when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAJ_SAVE_PATH, WEIGHTS_SAVE_PATH, RESULTS_DIR,
    N_TRAJECTORIES, FEATURES, TRUE_WEIGHTS
)
from graph_builder import build_graph
from trajectory_generator import generate_all_trajectories, load_trajectories
from maxent_irl import MaxEntIRL
from baselines import MyopicClassifier
from evaluation import evaluate_all, print_results_table


def main():
    print("=" * 60)
    print("  IRL Route Preference Learning — San Francisco")
    print("=" * 60)

    # ── Step 1: Graph ────────────────────────────────────────
    print("\n[1/5] Building road network …")
    G = build_graph()
    print(f"      Nodes: {G.number_of_nodes()}  "
          f"Edges: {G.number_of_edges()}")

    # ── Step 2: Trajectories ─────────────────────────────────
    print("\n[2/5] Generating expert trajectories …")
    if os.path.exists(TRAJ_SAVE_PATH):
        print("      Loading cached trajectories …")
        data = load_trajectories()
    else:
        data = generate_all_trajectories(G)

    train_traj = data["train"]
    test_traj  = data["test"]
    print(f"      Train: {len(train_traj)}  Test: {len(test_traj)}")

    # ── Step 3: MaxEnt IRL ────────────────────────────────────
    print("\n[3/5] Training MaxEnt IRL …")
    if os.path.exists(WEIGHTS_SAVE_PATH):
        print("      Loading cached IRL weights …")
        irl = MaxEntIRL.load(G)
    else:
        irl = MaxEntIRL(G)
        irl.train(train_traj)

    print(f"      Learned weights: {irl.weights.round(4)}")

    # Weight comparison table
    print("\n      Feature weights comparison:")
    print(f"      {'Feature':<20} {'True':>10} {'Learned':>10}")
    print("      " + "-" * 42)
    true_w = np.array(TRUE_WEIGHTS)
    for feat, tw, lw in zip(FEATURES, true_w, irl.weights):
        print(f"      {feat:<20} {tw:>10.4f} {lw:>10.4f}")
    cos = float(true_w @ irl.weights /
                (np.linalg.norm(true_w) * np.linalg.norm(irl.weights) + 1e-12))
    print(f"      cosine(true, learned) = {cos:.4f}")

    # ── Step 4: Myopic Classifier ─────────────────────────────
    print("\n[4/5] Training Myopic Classifier …")
    clf = MyopicClassifier(G)
    clf.train(train_traj)

    # ── Step 5: Evaluation ────────────────────────────────────
    print("\n[5/5] Evaluating …")
    results = evaluate_all(G, test_traj, irl.weights, clf,
                           oracle_weights=np.array(TRUE_WEIGHTS))
    print_results_table(results)

    # Save results
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n      Results saved -> {results_path}")

    # Save training history
    hist_path = os.path.join(RESULTS_DIR, "irl_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(irl.history, f)

    # ── Summary ───────────────────────────────────────────────
    irl_j  = results["IRL (MaxEnt)"]["Edge Jaccard"]
    dks_j  = results["Dijkstra-Shortest"]["Edge Jaccard"]
    gain   = (irl_j - dks_j) / (dks_j + 1e-9) * 100

    print("\n" + "=" * 60)
    print(f"  IRL Jaccard: {irl_j:.4f}  vs  "
          f"Dijkstra-Shortest: {dks_j:.4f}  "
          f"({gain:+.1f}%)")
    print("=" * 60)

    return results, irl, G, data


if __name__ == "__main__":
    main()
