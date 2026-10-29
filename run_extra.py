"""
run_extra.py – RL-planner baseline + Frechet distance on the existing v3 CV folds.

No training: re-uses the stored per-fold weights / alpha / pop beta from
results/porto/cv/v3_s<seed>_f<fold>.json and the same fold split as run_cv.py.

New method
  RL-Handcrafted : soft value iteration + route extraction on a FIXED,
                   hand-designed reward (no learning from data):
                   -1.0 / 100 m, -1.0 sharp turn, -0.3 slight turn.
                   This is "RL without inverse RL" – what IRL adds is the learned reward.
New metric
  Frechet (m)    : discrete Frechet distance between predicted and recorded
                   node polylines (lower is better).

    python run_extra.py --seeds 0

Output: results/porto/cv/extra_s<seed>_f<fold>.json
"""
import os
import sys
import json
import argparse
import numpy as np

os.environ["IRL_FEATURES"] = "v3"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config
from config import FEATURES, BASE_DIR
import region
from mdp import RouteMDP
from baselines import dijkstra_shortest, dijkstra_fastest
from evaluation import edge_jaccard, edge_f1, path_length_ratio
import run_cv

CV_DIR = run_cv.CV_DIR
HAND = {"length": -1.0, "turn_sharp": -1.0, "turn_slight": -0.3}


def xy_metres(G, nodes):
    lat0 = np.radians(41.16)
    a = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes], float)
    return np.c_[a[:, 0] * 111320.0 * np.cos(lat0), a[:, 1] * 110540.0]


def frechet(P, Q):
    n, m = len(P), len(Q)
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    C = np.full((n, m), np.inf)
    C[0, 0] = D[0, 0]
    for i in range(1, n):
        C[i, 0] = max(C[i - 1, 0], D[i, 0])
    for j in range(1, m):
        C[0, j] = max(C[0, j - 1], D[0, j])
    for i in range(1, n):
        for j in range(1, m):
            C[i, j] = max(min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]), D[i, j])
    return float(C[-1, -1])


def evaluate(G, trips, route_fn):
    J, F, L, Fr = [], [], [], []
    for t in trips:
        r = route_fn(t["origin"], t["destination"])
        if len(r) < 2:
            J.append(0.0); F.append(0.0); L.append(np.nan); Fr.append(np.nan)
            continue
        J.append(edge_jaccard(r, t["nodes"]))
        F.append(edge_f1(r, t["nodes"]))
        L.append(path_length_ratio(G, r, t["nodes"]))
        Fr.append(frechet(xy_metres(G, r), xy_metres(G, t["nodes"])))
    return {"Edge Jaccard": np.array(J), "Edge F1": np.array(F),
            "Path Length Ratio": np.array(L), "Frechet (m)": np.array(Fr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    G = region.load_graph()
    trips = region.load_data(G)["all"]
    mdp = RouteMDP(G)
    w_hand = np.zeros(len(FEATURES))
    for k, v in HAND.items():
        w_hand[FEATURES.index(k)] = v
    li = FEATURES.index("length")

    for seed in args.seeds:
        for k in range(args.folds):
            tag = f"s{seed}_f{k}"
            out = os.path.join(CV_DIR, f"extra_{tag}.json")
            src = os.path.join(CV_DIR, f"v3_{tag}.json")
            if os.path.exists(out) or not os.path.exists(src):
                continue
            rec = json.load(open(src))
            w = np.array([rec["weights"][f] for f in FEATURES])
            w_cal = w.copy()
            w_cal[li] -= rec["alpha"]
            _, _, test_i = run_cv.split(trips, seed, args.folds, k)
            test = [trips[i] for i in test_i]
            methods = {
                "Dijkstra-Shortest": lambda o, d: dijkstra_shortest(G, o, d),
                "Dijkstra-Fastest": lambda o, d: dijkstra_fastest(G, o, d),
                "Pop-Shortest": run_cv.pop_shortest_fn(G, mdp, rec["pop_beta"]),
                "RL-Handcrafted": lambda o, d: mdp.most_likely_route(o, d, w_hand),
                "IRL-v3 (raw)": lambda o, d: mdp.most_likely_route(o, d, w),
                "IRL-v3 (calibrated)": lambda o, d: mdp.most_likely_route(o, d, w_cal),
            }
            arrs = {m: evaluate(G, test, fn) for m, fn in methods.items()}
            np.savez(os.path.join(CV_DIR, f"extra_{tag}.npz"), test_idx=test_i,
                     **{f"{m}|{k_}": v for m, a in arrs.items() for k_, v in a.items()})
            summ = {m: {k_: float(np.nanmean(v)) for k_, v in a.items()}
                    for m, a in arrs.items()}
            json.dump({"seed": seed, "fold": k, "summary": summ}, open(out, "w"), indent=2)
            print(f"[extra] {tag}", flush=True)
            for m, s in summ.items():
                print(f"   {m:<22} J={s['Edge Jaccard']:.3f} Frechet={s['Frechet (m)']:.0f}m", flush=True)


if __name__ == "__main__":
    main()
