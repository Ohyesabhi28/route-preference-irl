"""
run_b1.py – B1: identifiable v2 feature set, validation-based calibration.

    set IRL_FEATURES=v2 && python run_b1.py [--retrain]

1. train MaxEnt IRL on data['fit']  (cached weights reused unless --retrain)
2. choose an extra length-penalty alpha on data['val']  (never on test)
3. evaluate raw and calibrated reward on data['test'] with per-trip CIs and
   paired tests vs Dijkstra-Shortest and vs the v1 per-trip results
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config
from config import FEATURES, RESULTS_DIR, WEIGHTS_SAVE_PATH, FEATURE_SET
import region
from maxent_irl import MaxEntIRL
from baselines import MyopicClassifier
from evaluation import evaluate_all, edge_jaccard
from mdp import RouteMDP
from metrics import summarize_per_trip, paired_wilcoxon

ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def val_jaccard(mdp, trips, w):
    J = []
    for t in trips:
        r = mdp.most_likely_route(t["origin"], t["destination"], w)
        J.append(edge_jaccard(r, t["nodes"]) if len(r) >= 2 else 0.0)
    return float(np.mean(J))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    assert FEATURE_SET in ("v2", "v3"), "run with IRL_FEATURES=v2 or v3"
    tag = f"IRL-{FEATURE_SET}"

    G = region.load_graph()
    data = region.load_data(G)
    fit, val, test = data["fit"], data["val"], data["test"]
    print(f"fit {len(fit)}  val {len(val)}  test {len(test)}")

    if os.path.exists(WEIGHTS_SAVE_PATH) and not args.retrain:
        irl = MaxEntIRL.load(G)
    else:
        irl = MaxEntIRL(G)
        w0 = None
        v2w = WEIGHTS_SAVE_PATH.replace("_v3.npy", "_v2.npy")
        if FEATURE_SET == "v3" and not config.ABLATION and os.path.exists(v2w):
            w0 = np.append(np.load(v2w), 0.0)          # warm start from v2
        irl.train(fit, w0=w0)
    w = irl.weights
    mdp = irl.mdp

    # ---- calibration on validation -----------------------------------
    li = FEATURES.index("length")
    scores = {}
    for a in ALPHAS:
        v = w.copy()
        v[li] -= a
        scores[a] = val_jaccard(mdp, val, v)
        print(f"  val Jaccard  alpha={a:<5} {scores[a]:.4f}")
    best = max(scores, key=scores.get)
    w_cal = w.copy()
    w_cal[li] -= best
    print(f"selected alpha = {best}")

    # ---- test ---------------------------------------------------------
    clf = MyopicClassifier(G)
    clf.train(fit)
    res_raw, pt_raw = evaluate_all(G, test, w, clf, return_per_trip=True)
    res_cal, pt_cal = evaluate_all(G, test, w_cal, clf, return_per_trip=True)

    per_trip = dict(pt_raw)
    per_trip[f"{tag} (raw)"] = per_trip.pop("IRL (MaxEnt)")
    cal = dict(pt_cal["IRL (MaxEnt)"])
    # likelihood is a property of the estimated model, not of the decision-time
    # length adjustment: report the raw-model LL / NLL for the calibrated row
    for k in ("Log-Likelihood", "Trip NLL"):
        cal[k] = per_trip[f"{tag} (raw)"][k]
    per_trip[f"{tag} (calibrated)"] = cal
    np.savez(os.path.join(RESULTS_DIR, "b1_per_trip.npz"),
             **{f"{n}|{k}": v for n, a in per_trip.items() for k, v in a.items()})

    # earlier runs on the same test trips (same order)
    v1_path = os.path.join(os.path.dirname(RESULTS_DIR), "per_trip_metrics.npz")
    if os.path.exists(v1_path):
        z = np.load(v1_path)
        per_trip["IRL-v1"] = {k.split("|")[1]: z[k] for k in z.files
                              if k.startswith("IRL (MaxEnt)|")}
    if FEATURE_SET == "v3":
        v2_path = os.path.join(os.path.dirname(RESULTS_DIR), "v2", "b1_per_trip.npz")
        if os.path.exists(v2_path):
            z = np.load(v2_path)
            for n in ("IRL-v2 (raw)", "IRL-v2 (calibrated)"):
                per_trip[n] = {k.split("|")[1]: z[k] for k in z.files
                               if k.startswith(n + "|")}

    ci = summarize_per_trip(per_trip, reference="Dijkstra-Shortest")
    extra = {}
    mets = ("Edge Jaccard", "Edge F1", "Log-Likelihood", "Trip NLL")
    refs = [r for r in ("IRL-v1", "IRL-v2 (raw)", "IRL-v2 (calibrated)")
            if r in per_trip and not r.startswith(tag)]
    for name in (f"{tag} (raw)", f"{tag} (calibrated)"):
        for ref in refs:
            # compare like with like: raw vs raw/v1, calibrated vs calibrated/v1
            if ref not in ("IRL-v1",) and ("calibrated" in name) != ("calibrated" in ref):
                continue
            extra[f"{name} vs {ref}"] = {
                m: paired_wilcoxon(per_trip[name][m], per_trip[ref][m])
                for m in mets if m in per_trip[name] and m in per_trip[ref]}

    out = {"features": FEATURES, "weights": dict(zip(FEATURES, map(float, w))),
           "alpha_val_scores": {str(k): v for k, v in scores.items()},
           "alpha_selected": best,
           "summary": ci, "paired_vs_v1": extra}
    path = os.path.join(RESULTS_DIR, "b1_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nLearned {FEATURE_SET} weights")
    for f_, v in zip(FEATURES, w):
        print(f"  {f_:<16} {v:+.4f}")
    print(f"\n{'method':<22} {'Jaccard':>8} {'95% CI':>16} {'F1':>7} "
          f"{'LL/step':>8} {'p vs SP':>8}")
    for m in ("Dijkstra-Shortest", "Dijkstra-Fastest", "IRL-v1",
              "IRL-v2 (raw)", "IRL-v2 (calibrated)",
              f"{tag} (raw)", f"{tag} (calibrated)"):
        if m not in ci:
            continue
        j, f1 = ci[m]["Edge Jaccard"], ci[m]["Edge F1"]
        ll = ci[m].get("Log-Likelihood", {}).get("mean", float("nan"))
        p = j.get("vs_Dijkstra-Shortest", {}).get("p_value", float("nan"))
        print(f"{m:<22} {j['mean']:>8.4f} [{j['ci_lo']:.3f},{j['ci_hi']:.3f}] "
              f"{f1['mean']:>7.4f} {ll:>8.4f} {p:>8.3g}")
    print("\npaired tests (diff = first - second):")
    for k, v in extra.items():
        for met, r in v.items():
            print(f"  {k} | {met}: {r['mean_diff']:+.4f}  p={r['p_value']:.3g}")
    print("saved ->", path)


if __name__ == "__main__":
    main()
