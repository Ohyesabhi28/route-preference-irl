"""
make_report_figs.py - report figures with a colour-blind-safe palette (Okabe-Ito),
distinct markers / hatches so methods stay separable in greyscale print.

Reads results/porto/cv (seed 0): cv_summary.json, v2/v3 json (history),
extra_s0_f*.{json,npz} (baselines, RL-Handcrafted, IRL-v3 per trip).
Writes figures into report/figures and report_ieeeaccess/figures.
"""
import os
import sys
import glob
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))
from metrics import summarize

CV = os.path.join(BASE, "results", "porto", "cv")
OUTS = [os.path.join(BASE, "report", "figures"),
        os.path.join(BASE, "report_ieeeaccess", "figures")]

STYLE = {   # colour, marker, linestyle
    "Dijkstra-Fastest":    ("#7f7f7f", "v", ":"),
    "Dijkstra-Shortest":   ("#000000", "s", "-"),
    "RL-Handcrafted":      ("#CC79A7", "D", "--"),
    "Pop-Shortest":        ("#E69F00", "^", "-"),
    "IRL-v2 (raw)":        ("#56B4E9", "o", "--"),
    "IRL-v2 (calibrated)": ("#009E73", "o", "-"),
    "IRL-v3 (raw)":        ("#0072B2", "o", "--"),
    "IRL-v3 (calibrated)": ("#D55E00", "o", "-"),
}
HATCH = {"IRL-v2 (raw)": "", "IRL-v2 (calibrated)": "//",
         "IRL-v3 (raw)": "", "IRL-v3 (calibrated)": "//"}
ORDER = list(STYLE)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    for d in OUTS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fold_stats():
    summ = json.load(open(os.path.join(CV, "cv_summary.json")))["summary"]
    ex = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CV, "extra_s0_f?.json")))]
    out = {}
    for m in ORDER:
        for k in ("Edge Jaccard", "Edge F1", "Log-Likelihood"):
            if m.startswith("IRL-v2") or (k == "Log-Likelihood" and m in summ):
                if m in summ and k in summ[m]:
                    out.setdefault(m, {})[k] = (summ[m][k]["fold_mean"], summ[m][k]["fold_std"])
                    continue
            vals = [e["summary"][m][k] for e in ex if m in e["summary"] and k in e["summary"][m]]
            if vals:
                out.setdefault(m, {})[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def bars():
    st = fold_stats()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, k, title in zip(axes, ("Edge Jaccard", "Edge F1", "Log-Likelihood"),
                            ("Edge Jaccard (higher better)", "Edge F1 (higher better)",
                             "Held-out LL per step (higher better)")):
        ms = [m for m in ORDER if k in st.get(m, {})]
        mu = [st[m][k][0] for m in ms]
        sd = [st[m][k][1] for m in ms]
        ax.bar(range(len(ms)), mu, yerr=sd, capsize=3,
               color=[STYLE[m][0] for m in ms], edgecolor="black", linewidth=.6,
               hatch=None)
        for i, m in enumerate(ms):
            if HATCH.get(m):
                ax.patches[i].set_hatch(HATCH[m])
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([m.replace(" (", "\n(") for m in ms], rotation=60, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        if k != "Log-Likelihood":
            ax.set_ylim(min(mu) * 0.8, max(mu) * 1.06)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    save(fig, "cv_methods_bars.png")


def curves():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for fs, col, ls in (("v2", "#009E73", "--"), ("v3", "#D55E00", "-")):
        hs = []
        for f in sorted(glob.glob(os.path.join(CV, f"{fs}_s0_f?.json"))):
            h = json.load(open(f)).get("history")
            if h:
                hs.append([x[1] for x in h])
        if not hs:
            continue
        L = max(map(len, hs))
        M = np.array([[h[min(i, len(h) - 1)] for i in range(L)] for h in hs])
        mu, sd = M.mean(0), M.std(0)
        ax.plot(mu, color=col, ls=ls, lw=1.8, label=f"{fs} (n={len(hs)} folds)")
        ax.fill_between(range(L), mu - sd, mu + sd, color=col, alpha=.2)
    ax.set_xlabel("L-BFGS iteration")
    ax.set_ylabel("training NLL / trajectory")
    ax.legend(frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "cv_training_curves.png")


def by_length():
    files = sorted(glob.glob(os.path.join(CV, "extra_s0_f?.npz")))
    d = pickle.load(open(os.path.join(BASE, "data", "porto_trajectories.pkl"), "rb"))
    lens = np.array([len(t["edges"]) for t in d["all"]])
    n = len(lens)
    arr = {}
    for f in files:
        z = np.load(f)
        idx = z["test_idx"]
        for key in z.files:
            if key.endswith("|Edge Jaccard"):
                m = key.split("|")[0]
                arr.setdefault(m, np.full(n, np.nan))[idx] = z[key]
    edges = np.quantile(lens, [0, .25, .5, .75, 1.0])
    mid = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for m in ("Dijkstra-Shortest", "RL-Handcrafted", "Pop-Shortest", "IRL-v3 (raw)", "IRL-v3 (calibrated)"):
        col, mk, ls = STYLE[m]
        mus, los, his = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (lens >= lo) & (lens <= hi) & ~np.isnan(arr[m])
            s = summarize(arr[m][sel])
            mus.append(s["mean"]); los.append(s["ci_lo"]); his.append(s["ci_hi"])
        ax.plot(mid, mus, marker=mk, ls=ls, color=col, lw=1.6, ms=5, label=m)
        ax.fill_between(mid, los, his, color=col, alpha=.10)
    ax.set_xlabel("trip length (edges, quartile bins)")
    ax.set_ylabel("Edge Jaccard")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "cv_jaccard_by_length.png")


if __name__ == "__main__":
    bars()
    curves()
    by_length()
    print("ok")
