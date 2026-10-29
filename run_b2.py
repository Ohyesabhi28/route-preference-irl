"""
run_b2.py – B2: context-conditioned reward (peak / night / weekend vs off-peak).

    set IRL_FEATURES=v2 && python run_b2.py [--retrain]

Model selection uses data['val'] only:
  * lambda  (shrinkage of the context deltas)      by held-out log-likelihood
  * alpha   (extra length penalty, as in run_b1)   by edge Jaccard
Final numbers are on data['test'] with per-trip CIs and paired tests.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import FEATURES, RESULTS_DIR, WEIGHTS_SAVE_PATH, FEATURE_SET, N_FEATURES
import region
from baselines import dijkstra_shortest
from context_irl import (ContextMaxEntIRL, CONTEXTS, N_CTX, context_of,
                         effective_weights)
from evaluation import edge_jaccard, edge_f1, policy_step_stats_per_trip
from mdp import RouteMDP
from metrics import summarize, paired_wilcoxon

CTX_PATH = WEIGHTS_SAVE_PATH.replace(".npy", "_ctx.npy")
LI = FEATURES.index("length")
LAMBDAS = (0.0, 0.5, 1.0)
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def per_trip(G, mdp, trips, W, alpha=0.0):
    """Per-trip Jaccard / F1 / LL / NLL under context weights W (N_CTX, N)."""
    n = len(trips)
    out = {k: np.full(n, np.nan) for k in
           ("Edge Jaccard", "Edge F1", "Log-Likelihood", "Trip NLL")}
    ctx = np.array([context_of(t) for t in trips])
    for c in range(N_CTX):
        idx = np.where(ctx == c)[0]
        if len(idx) == 0:
            continue
        w_route = W[c].copy()
        w_route[LI] -= alpha
        for i in idx:
            t = trips[i]
            r = mdp.most_likely_route(t["origin"], t["destination"], w_route)
            if len(r) >= 2:
                out["Edge Jaccard"][i] = edge_jaccard(r, t["nodes"])
                out["Edge F1"][i] = edge_f1(r, t["nodes"])
            else:
                out["Edge Jaccard"][i] = out["Edge F1"][i] = 0.0
        ll = policy_step_stats_per_trip(G, [trips[i] for i in idx], W[c], mdp,
                                        ks=(1,))
        out["Log-Likelihood"][idx] = ll["Log-Likelihood"]
        out["Trip NLL"][idx] = ll["Trip NLL"]
    return out, ctx


def pick_alpha(G, mdp, val, W):
    best, bs = None, -1
    for a in ALPHAS:
        j = np.nanmean(per_trip_route_only(mdp, val, W, a))
        print(f"    val Jaccard alpha={a:<4} {j:.4f}")
        if j > bs:
            best, bs = a, j
    return best


def per_trip_route_only(mdp, trips, W, alpha):
    J = []
    for t in trips:
        w = W[context_of(t)].copy()
        w[LI] -= alpha
        r = mdp.most_likely_route(t["origin"], t["destination"], w)
        J.append(edge_jaccard(r, t["nodes"]) if len(r) >= 2 else 0.0)
    return np.array(J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()
    assert FEATURE_SET == "v2", "run with IRL_FEATURES=v2"

    G = region.load_graph()
    data = region.load_data(G)
    fit, val, test = data["fit"], data["val"], data["test"]
    w_glob = np.load(WEIGHTS_SAVE_PATH)

    m = ContextMaxEntIRL(G, n_iter=args.iters)
    mdp = m.mdp
    if os.path.exists(CTX_PATH) and not args.retrain:
        theta = np.load(CTX_PATH)
    else:
        theta = m.fit(fit, w0=w_glob)
        np.save(CTX_PATH, theta)
    th = theta.reshape(N_CTX, N_FEATURES)

    # ---- lambda on validation (held-out likelihood) -------------------
    print("selecting shrinkage lambda on validation LL")
    lam_scores = {}
    for lam in LAMBDAS:
        W = effective_weights(theta, lam)
        pt, _ = per_trip(G, mdp, val, W)
        lam_scores[lam] = float(np.nanmean(pt["Log-Likelihood"]))
        print(f"  lambda={lam:<4} val LL/step {lam_scores[lam]:.4f}")
    lam = max(lam_scores, key=lam_scores.get)
    W_ctx = effective_weights(theta, lam)
    W_glob = np.tile(w_glob, (N_CTX, 1))

    print("selecting alpha on validation (global model)")
    a_glob = pick_alpha(G, mdp, val, W_glob)
    print("selecting alpha on validation (context model)")
    a_ctx = pick_alpha(G, mdp, val, W_ctx)
    print(f"lambda={lam}  alpha_global={a_glob}  alpha_ctx={a_ctx}")

    # ---- test ---------------------------------------------------------
    models = {
        "Global v2 (raw)": (W_glob, 0.0),
        "Global v2 (calibrated)": (W_glob, a_glob),
        "Context v2 (raw)": (W_ctx, 0.0),
        "Context v2 (calibrated)": (W_ctx, a_ctx),
    }
    arrs, ctx_test = {}, None
    for name, (W, a) in models.items():
        print("evaluating", name, flush=True)
        arrs[name], ctx_test = per_trip(G, mdp, test, W, a)
    sp = np.array([edge_jaccard(dijkstra_shortest(G, t["origin"], t["destination"]),
                                t["nodes"]) for t in test])
    arrs["Dijkstra-Shortest"] = {"Edge Jaccard": sp}

    summary, tests = {}, {}
    for name, a in arrs.items():
        summary[name] = {k: summarize(v) for k, v in a.items()}
    for name in ("Context v2 (raw)", "Context v2 (calibrated)"):
        ref = "Global v2 (raw)" if "raw" in name else "Global v2 (calibrated)"
        tests[f"{name} vs {ref}"] = {
            k: paired_wilcoxon(arrs[name][k], arrs[ref][k])
            for k in ("Edge Jaccard", "Edge F1", "Log-Likelihood")}
    tests["Context v2 (calibrated) vs Dijkstra-Shortest"] = {
        "Edge Jaccard": paired_wilcoxon(arrs["Context v2 (calibrated)"]["Edge Jaccard"], sp)}

    by_ctx = {}
    for c, cname in enumerate(CONTEXTS):
        msk = ctx_test == c
        by_ctx[cname] = {"n": int(msk.sum())}
        for name in ("Global v2 (raw)", "Context v2 (raw)",
                     "Global v2 (calibrated)", "Context v2 (calibrated)"):
            by_ctx[cname][name] = {
                "Edge Jaccard": float(np.nanmean(arrs[name]["Edge Jaccard"][msk])),
                "Log-Likelihood": float(np.nanmean(arrs[name]["Log-Likelihood"][msk]))}
        by_ctx[cname]["Dijkstra-Shortest"] = float(sp[msk].mean())

    out = {
        "lambda": lam, "lambda_val_LL": {str(k): v for k, v in lam_scores.items()},
        "alpha_global": a_glob, "alpha_context": a_ctx,
        "w0": dict(zip(FEATURES, map(float, th[0]))),
        "deltas": {c: dict(zip(FEATURES, map(float, th[i])))
                   for i, c in enumerate(CONTEXTS) if i > 0},
        "effective_weights": {c: dict(zip(FEATURES, map(float, W_ctx[i])))
                              for i, c in enumerate(CONTEXTS)},
        "summary": summary, "paired_tests": tests, "by_context_test": by_ctx,
        "history": m.history,
    }
    path = os.path.join(RESULTS_DIR, "b2_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(RESULTS_DIR, "b2_per_trip.npz"),
             ctx=ctx_test, **{f"{n}|{k}": v for n, a in arrs.items()
                              for k, v in a.items()})

    print("\nContext deltas (relative to off-peak):")
    print(f"  {'feature':<16}" + "".join(f"{c:>10}" for c in CONTEXTS[1:]))
    for j, f_ in enumerate(FEATURES):
        print(f"  {f_:<16}" + "".join(f"{lam * th[i, j]:>+10.3f}" for i in range(1, N_CTX)))
    print(f"\n{'model':<26} {'Jaccard':>8} {'95% CI':>16} {'F1':>7} {'LL/step':>9}")
    for name, a in summary.items():
        j = a["Edge Jaccard"]
        f1 = a.get("Edge F1", {"mean": float('nan')})["mean"]
        ll = a.get("Log-Likelihood", {"mean": float('nan')})["mean"]
        print(f"{name:<26} {j['mean']:>8.4f} [{j['ci_lo']:.3f},{j['ci_hi']:.3f}] "
              f"{f1:>7.4f} {ll:>9.4f}")
    print("\npaired tests:")
    for k, v in tests.items():
        for met, r in v.items():
            print(f"  {k} | {met}: diff {r['mean_diff']:+.4f}  p={r['p_value']:.3g}")
    print("\nsaved ->", path)


if __name__ == "__main__":
    main()
