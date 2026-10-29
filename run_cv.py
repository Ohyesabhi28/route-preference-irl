"""
run_cv.py – K-fold cross-validation (over all 3 000 matched trips), repeated
over partition seeds, for ONE feature set (set IRL_FEATURES=v1|v2|v3).

    set IRL_FEATURES=v3 && python run_cv.py --folds 5 --seeds 0
    set IRL_FEATURES=v2 && python run_cv.py --folds 5 --seeds 0

Per fold:  fit 2040 / val 360 / test 600 trips (val carved from the training
part; test fold never touched for training or for choosing alpha / beta).
  * MaxEnt IRL trained from w = 0 (cold start: no information from other folds)
  * alpha (extra length penalty at decision time) chosen on val Jaccard
  * baselines: Dijkstra shortest / fastest (all runs), myopic classifier (v1
    run), popularity-aware shortest path (v3 run; independent familiarity pool,
    beta chosen on val)
Outputs  results/porto/cv/<fs>_s<seed>_f<fold>.{json,npz}  (resumable).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config
from config import FEATURES, FEATURE_SET, BASE_DIR
import region
import maxent_irl as M
from maxent_irl import MaxEntIRL
from baselines import dijkstra_shortest, dijkstra_fastest, MyopicClassifier
from evaluation import (edge_jaccard, edge_f1, path_length_ratio,
                        policy_step_stats_per_trip)

CV_DIR = os.environ.get("IRL_CV_DIR",
                        os.path.join(BASE_DIR, "results", "porto", "cv"))
SPLIT_BY = os.environ.get("IRL_CV_SPLIT", "trip")      # trip | taxi
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
BETAS = (0.25, 0.5, 0.75, 0.9)
os.makedirs(CV_DIR, exist_ok=True)


def split(trips, seed, K, k):
    rng = np.random.default_rng(seed)
    if SPLIT_BY == "taxi":              # whole taxis held out together
        taxis = np.array([t["taxi_id"] for t in trips])
        ut = rng.permutation(np.unique(taxis))
        tfolds = np.array_split(ut, K)
        folds = [np.where(np.isin(taxis, tf))[0] for tf in tfolds]
    else:
        order = rng.permutation(len(trips))
        folds = np.array_split(order, K)
    train_idx = np.concatenate([folds[i] for i in range(K) if i != k])
    n_val = int(0.15 * len(train_idx))
    return train_idx[:-n_val], train_idx[-n_val:], folds[k]


def route_metrics(G, trips, route_fn):
    J, F, L = [], [], []
    for t in trips:
        r = route_fn(t["origin"], t["destination"])
        if len(r) < 2:
            J.append(0.0); F.append(0.0); L.append(np.nan)
            continue
        J.append(edge_jaccard(r, t["nodes"]))
        F.append(edge_f1(r, t["nodes"]))
        L.append(path_length_ratio(G, r, t["nodes"]))
    return {"Edge Jaccard": np.array(J), "Edge F1": np.array(F),
            "Path Length Ratio": np.array(L)}


def pop_shortest_fn(G, mdp, beta):
    fam, eid = mdp.familiarity, mdp.eid

    def wfun(u, v, d):
        return min(dd.get("length", 1e9) * (1.0 - beta * fam[eid[(u, v, k)]])
                   for k, dd in d.items())

    def route(o, d):
        try:
            return nx.shortest_path(G, o, d, weight=wfun)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    return route


def run_fold(G, trips, seed, K, k, iters, verbose):
    tag = f"{FEATURE_SET}_s{seed}_f{k}"
    out_json = os.path.join(CV_DIR, tag + ".json")
    if os.path.exists(out_json):
        print(f"[cv] {tag} exists, skipping")
        return
    fit_i, val_i, test_i = split(trips, seed, K, k)
    fit = [trips[i] for i in fit_i]
    val = [trips[i] for i in val_i]
    test = [trips[i] for i in test_i]
    print(f"\n[cv] {tag}: fit {len(fit)} val {len(val)} test {len(test)}", flush=True)

    # keep the full-data weights file untouched
    M.WEIGHTS_SAVE_PATH = os.path.join(CV_DIR, tag + "_weights.npy")
    t0 = time.time()
    irl = MaxEntIRL(G, n_iter=iters)
    irl.train(fit, verbose=verbose)
    w, mdp = irl.weights, irl.mdp
    train_s = time.time() - t0

    # ---- alpha on validation -----------------------------------------
    li = FEATURES.index("length")
    val_j = {}
    for a in ALPHAS:
        v = w.copy()
        v[li] -= a
        val_j[a] = float(route_metrics(
            G, val, lambda o, d: mdp.most_likely_route(o, d, v))["Edge Jaccard"].mean())
    alpha = max(val_j, key=val_j.get)
    w_cal = w.copy()
    w_cal[li] -= alpha

    # ---- test ----------------------------------------------------------
    arrs = {}
    arrs["Dijkstra-Shortest"] = route_metrics(
        G, test, lambda o, d: dijkstra_shortest(G, o, d))
    arrs["Dijkstra-Fastest"] = route_metrics(
        G, test, lambda o, d: dijkstra_fastest(G, o, d))
    ll = policy_step_stats_per_trip(G, test, w, mdp)
    raw = route_metrics(G, test, lambda o, d: mdp.most_likely_route(o, d, w))
    raw.update(ll)
    cal = route_metrics(G, test, lambda o, d: mdp.most_likely_route(o, d, w_cal))
    cal.update(ll)                       # likelihood is of the estimated model
    arrs[f"IRL-{FEATURE_SET} (raw)"] = raw
    arrs[f"IRL-{FEATURE_SET} (calibrated)"] = cal

    extra = {}
    if FEATURE_SET == "v1":
        clf = MyopicClassifier(G)
        clf.train(fit)
        arrs["Myopic-Classifier"] = route_metrics(
            G, test, lambda o, d: clf.predict_route(G, o, d))
        mll, mtopk = clf.step_stats(test)
        extra["Myopic step stats"] = {"Log-Likelihood": mll,
                                      "Top-1": mtopk[1], "Top-3": mtopk[3]}
    if FEATURE_SET == "v3":
        bj = {b: float(route_metrics(G, val, pop_shortest_fn(G, mdp, b))
                       ["Edge Jaccard"].mean()) for b in BETAS}
        beta = max(bj, key=bj.get)
        arrs["Pop-Shortest"] = route_metrics(G, test, pop_shortest_fn(G, mdp, beta))
        extra["pop_beta"] = beta
        extra["pop_val_jaccard"] = {str(b): v for b, v in bj.items()}

    summary = {m: {k_: float(np.nanmean(v)) for k_, v in a.items()}
               for m, a in arrs.items()}
    np.savez(os.path.join(CV_DIR, tag + ".npz"), test_idx=test_i,
             **{f"{m}|{k_}": v for m, a in arrs.items() for k_, v in a.items()})
    rec = {"feature_set": FEATURE_SET, "seed": seed, "fold": k, "K": K,
           "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
           "alpha": alpha, "alpha_val_jaccard": {str(a): v for a, v in val_j.items()},
           "weights": dict(zip(FEATURES, map(float, w))),
           "history": irl.history, "train_seconds": train_s,
           "summary": summary, **extra}
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[cv] {tag} done in {time.time()-t0:.0f}s  alpha={alpha}")
    for m, s in summary.items():
        print(f"      {m:<26} J={s['Edge Jaccard']:.4f} F1={s['Edge F1']:.4f}"
              + (f" LL={s['Log-Likelihood']:.4f}" if "Log-Likelihood" in s else ""),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    G = region.load_graph()
    trips = region.load_data(G)["all"]
    print(f"[cv] {FEATURE_SET}: {len(trips)} trips, {args.folds} folds, "
          f"seeds {args.seeds}, {args.iters} iters")
    for seed in args.seeds:
        for k in range(args.folds):
            if args.only_fold is not None and k != args.only_fold:
                continue
            run_fold(G, trips, seed, args.folds, k, args.iters, not args.quiet)


if __name__ == "__main__":
    main()
