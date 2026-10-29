"""
run_ablation.py – feature-group ablations of the v3 model on the Porto split.

    python run_ablation.py            # runs every group sequentially (~20 min each)
    python run_ablation.py no_turn    # runs one group
    python run_ablation.py --report   # only rebuild results/porto/v3/ablation_summary.json

Each run is `run_b1.py` with IRL_FEATURES=v3 and IRL_DROP=<group>, which retrains
from zero (no warm start), picks the length penalty alpha on the validation
slice, and evaluates on the same 600 test trips as the full v3 model.
"""
import os
import sys
import json
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.join(BASE, "results", "porto", "v3")

GROUPS = {
    "no_turn":        "turn_slight,turn_sharp",
    "no_familiarity": "edge_familiarity",
    "no_goal":        "away_from_goal",
    "no_roadclass":   "road_motorway,road_primary",
    "no_step":        "step",
}


def run(tag):
    env = dict(os.environ, IRL_FEATURES="v3", IRL_DROP=GROUPS[tag], IRL_ABLATION_TAG=tag)
    print(f"\n=== ablation {tag}: drop {GROUPS[tag]} ===", flush=True)
    subprocess.run([sys.executable, os.path.join(BASE, "run_b1.py"), "--retrain"],
                   env=env, check=True, cwd=BASE)


def report():
    out = {}
    full = os.path.join(V3_DIR, "b1_results.json")
    with open(full) as f:
        r = json.load(f)
    out["full v3"] = {"dropped": [], "alpha": r["alpha_selected"],
                      **{m: r["summary"]["IRL-v3 (calibrated)"][m]["mean"]
                         for m in ("Edge F1", "Edge Jaccard", "Trip NLL", "Top-1 Accuracy")}}
    ref = r["summary"]["Dijkstra-Shortest"]["Edge F1"]["mean"]
    for tag, drop in GROUPS.items():
        p = os.path.join(V3_DIR, f"ablate_{tag}", "b1_results.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            a = json.load(f)
        s = a["summary"]["IRL-v3 (calibrated)"]
        out[tag] = {"dropped": drop.split(","), "alpha": a["alpha_selected"],
                    **{m: s[m]["mean"] for m in ("Edge F1", "Edge Jaccard", "Trip NLL", "Top-1 Accuracy")}}
        out[tag]["F1 change vs full"] = out[tag]["Edge F1"] - out["full v3"]["Edge F1"]
        out[tag]["NLL change vs full"] = out[tag]["Trip NLL"] - out["full v3"]["Trip NLL"]
    out["Dijkstra-Shortest F1"] = ref
    with open(os.path.join(V3_DIR, "ablation_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"{'model':<16}{'F1':>8}{'dF1':>8}{'NLL':>8}{'dNLL':>8}{'Top-1':>8}")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:<16}{v['Edge F1']:>8.3f}{v.get('F1 change vs full', 0):>+8.3f}"
                  f"{v['Trip NLL']:>8.2f}{v.get('NLL change vs full', 0):>+8.2f}{v['Top-1 Accuracy']:>8.3f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--report" not in sys.argv:
        for tag in (args or GROUPS):
            run(tag)
    report()
