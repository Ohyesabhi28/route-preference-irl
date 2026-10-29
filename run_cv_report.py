"""
run_cv_report.py – aggregate results/porto/cv/*.{json,npz} into
  * results/porto/cv/cv_summary.json / cv_table.md
  * figures with variance bands in results/porto/cv/figures/

Fold-level mean +- std (n = folds x seeds) and trip-level pooled bootstrap CIs
and paired Wilcoxon tests (every trip is a test trip exactly once per seed).
"""
import os
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from metrics import summarize, paired_wilcoxon

CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "results", "porto", "cv")
FIG_DIR = os.path.join(CV_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ORDER = ["Dijkstra-Fastest", "Dijkstra-Shortest", "Myopic-Classifier",
         "Pop-Shortest", "IRL-v1 (raw)", "IRL-v1 (calibrated)",
         "IRL-v2 (raw)", "IRL-v2 (calibrated)",
         "IRL-v3 (raw)", "IRL-v3 (calibrated)"]
METRICS = ["Edge Jaccard", "Edge F1", "Path Length Ratio",
           "Log-Likelihood", "Trip NLL", "Top-1 Accuracy"]
COL = {"Dijkstra": "#8a8a8a", "Myopic": "#b08a6a", "Pop": "#c9a227",
       "v1": "#7fa6d6", "v2": "#4a7fbd", "v3": "#1d4f91"}


def color(m):
    for k, c in COL.items():
        if k in m:
            return c
    return "#444"


def load():
    recs = []
    for jp in sorted(glob.glob(os.path.join(CV_DIR, "*_s*_f*.json"))):
        r = json.load(open(jp))
        z = np.load(jp.replace(".json", ".npz"))
        r["arrays"] = {k: z[k] for k in z.files}
        recs.append(r)
    return recs


def main():
    recs = load()
    if not recs:
        print("no CV results yet")
        return
    seeds = sorted({r["seed"] for r in recs})
    n_all = max(int(r["arrays"]["test_idx"].max()) for r in recs) + 1

    # ---- fold-level stats ------------------------------------------------
    fold_vals = {}                      # method -> metric -> list over (seed, fold)
    for r in recs:
        for m, s in r["summary"].items():
            for k, v in s.items():
                fold_vals.setdefault(m, {}).setdefault(k, []).append(v)
    methods = [m for m in ORDER if m in fold_vals]

    # ---- pooled per-trip arrays per seed, aligned by trip index ----------
    pooled = {}                         # (seed, method, metric) -> array(n_all)
    for r in recs:
        idx = r["arrays"]["test_idx"]
        for key, a in r["arrays"].items():
            if key == "test_idx":
                continue
            m, k = key.split("|")
            arr = pooled.setdefault((r["seed"], m, k), np.full(n_all, np.nan))
            arr[idx] = a
    def trip_arr(m, k):                 # concatenate seeds
        return np.concatenate([pooled[(s, m, k)] for s in seeds
                               if (s, m, k) in pooled])

    summary = {}
    for m in methods:
        summary[m] = {}
        for k in METRICS:
            if k not in fold_vals[m]:
                continue
            v = np.array(fold_vals[m][k])
            pooled_s = summarize(trip_arr(m, k))
            summary[m][k] = {"fold_mean": float(v.mean()),
                             "fold_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                             "n_folds": int(len(v)),
                             "trip_ci_lo": pooled_s["ci_lo"],
                             "trip_ci_hi": pooled_s["ci_hi"]}
    tests = {}
    pairs = [("IRL-v3 (raw)", "Dijkstra-Shortest"), ("IRL-v3 (calibrated)", "Dijkstra-Shortest"),
             ("IRL-v3 (raw)", "Pop-Shortest"), ("IRL-v3 (calibrated)", "Pop-Shortest"),
             ("IRL-v3 (raw)", "IRL-v2 (raw)"), ("IRL-v2 (raw)", "IRL-v1 (raw)"),
             ("IRL-v2 (calibrated)", "Dijkstra-Shortest"),
             ("IRL-v1 (raw)", "Dijkstra-Shortest")]
    for a, b in pairs:
        if a in methods and b in methods:
            tests[f"{a} vs {b}"] = {
                k: paired_wilcoxon(trip_arr(a, k), trip_arr(b, k))
                for k in ("Edge Jaccard", "Edge F1")}
    with open(os.path.join(CV_DIR, "cv_summary.json"), "w") as f:
        json.dump({"seeds": seeds, "n_runs": len(recs), "summary": summary,
                   "paired_tests": tests}, f, indent=2)

    # ---- markdown table -------------------------------------------------
    lines = ["| Method | Edge Jaccard | Edge F1 | Length ratio | LL / step | trip NLL |",
             "|---|---|---|---|---|---|"]
    for m in methods:
        row = [m]
        for k in ("Edge Jaccard", "Edge F1", "Path Length Ratio",
                  "Log-Likelihood", "Trip NLL"):
            s = summary[m].get(k)
            row.append(f"{s['fold_mean']:.3f} ± {s['fold_std']:.3f}" if s else "–")
        lines.append("| " + " | ".join(row) + " |")
    n_f = len(recs) // max(1, len({r["feature_set"] for r in recs}))
    lines.append(f"\nmean ± std over folds (≈{n_f} runs per feature set; seeds {seeds}).")
    with open(os.path.join(CV_DIR, "cv_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print()
    for k, v in tests.items():
        for met, r in v.items():
            print(f"{k} | {met}: diff {r['mean_diff']:+.4f}  p={r['p_value']:.3g}")

    # ---- figures ---------------------------------------------------------
    # 1. bars with fold std + trip-level CI
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, k, title in zip(axes, ("Edge Jaccard", "Edge F1", "Log-Likelihood"),
                            ("Edge Jaccard (↑)", "Edge F1 (↑)",
                             "Held-out log-likelihood / step (↑)")):
        ms = [m for m in methods if k in summary[m]]
        mu = [summary[m][k]["fold_mean"] for m in ms]
        sd = [summary[m][k]["fold_std"] for m in ms]
        ax.bar(range(len(ms)), mu, yerr=sd, capsize=3, color=[color(m) for m in ms])
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([m.replace(" (", "\n(") for m in ms], rotation=60,
                           ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        if k != "Log-Likelihood":
            ax.set_ylim(min(mu) * 0.8, max(mu) * 1.08)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("5-fold CV on Porto taxi trips: mean ± std over folds", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cv_methods_bars.png"), dpi=160)
    plt.close(fig)

    # 2. training curves with band (NLL per trajectory vs iteration)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for fs in ("v1", "v2", "v3"):
        hs = [r["history"] for r in recs if r["feature_set"] == fs and r["history"]]
        if not hs:
            continue
        L = max(len(h) for h in hs)
        M = np.array([[h[min(i, len(h) - 1)][1] for i in range(L)] for h in hs])
        x = np.arange(L)
        mu, sd = M.mean(0), M.std(0)
        ax.plot(x, mu, color=COL[fs], label=f"{fs} (n={len(hs)})")
        ax.fill_between(x, mu - sd, mu + sd, color=COL[fs], alpha=.25)
    ax.set_xlabel("L-BFGS iteration")
    ax.set_ylabel("training NLL / trajectory")
    ax.set_title("MaxEnt IRL convergence (mean ± std over folds)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cv_training_curves.png"), dpi=160)
    plt.close(fig)

    # 3. Jaccard by trip-length tercile with bootstrap CI band
    lens = None
    try:
        import pickle
        d = pickle.load(open(os.path.join(os.path.dirname(CV_DIR), "..", "..",
                                          "data", "porto_trajectories.pkl"), "rb"))
        lens = np.array([len(t["edges"]) for t in d["all"]])
    except Exception:
        pass
    if lens is not None:
        edges = np.quantile(lens, [0, .25, .5, .75, 1.0])
        mid = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for m in ("Dijkstra-Shortest", "Pop-Shortest", "IRL-v1 (raw)",
                  "IRL-v2 (calibrated)", "IRL-v3 (raw)", "IRL-v3 (calibrated)"):
            if m not in methods:
                continue
            a = trip_arr(m, "Edge Jaccard")
            L = np.tile(lens, len(seeds))[: len(a)]
            mus, los, his = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (L >= lo) & (L <= hi)
                s = summarize(a[sel])
                mus.append(s["mean"]); los.append(s["ci_lo"]); his.append(s["ci_hi"])
            ax.plot(mid, mus, marker="o", color=color(m), label=m)
            ax.fill_between(mid, los, his, color=color(m), alpha=.15)
        ax.set_xlabel("trip length (edges, quartile bins)")
        ax.set_ylabel("Edge Jaccard")
        ax.set_title("Overlap vs trip length (95% bootstrap bands)")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "cv_jaccard_by_length.png"), dpi=160)
        plt.close(fig)
    print("figures ->", FIG_DIR)


if __name__ == "__main__":
    main()
