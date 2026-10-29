"""
run_diagnostics.py – why does the learned reward not beat shortest-path on
route overlap?  Tests the usual suspects and writes results/<region>/diagnostics.json.

  A. Reward sign / convergence sanity (positive-reward transitions, V converged?)
  B. Where does IRL lose?  overlap by trip length, agreement with shortest path
  C. How far are experts from shortest path at all (ceiling for any length-ish cost)?
  D. Weight surgery: which learned components help / hurt overlap?
  E. Calibration sweep: scale the whole reward, push extra weight on length
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import FEATURES, RESULTS_DIR, WEIGHTS_SAVE_PATH
import region
from mdp import RouteMDP, NEG
from baselines import dijkstra_shortest
from evaluation import edge_jaccard, edge_f1, route_length

N_TRIPS = int(os.environ.get("DIAG_N", 300))
IDX = {f: i for i, f in enumerate(FEATURES)}


def mean_jaccard(mdp, trips, w, cache=None):
    J = []
    for t in trips:
        r = mdp.most_likely_route(t["origin"], t["destination"], w)
        J.append(edge_jaccard(r, t["nodes"]) if len(r) >= 2 else 0.0)
    return float(np.mean(J))


def main():
    G = region.load_graph()
    data = region.load_data(G)
    w = np.load(WEIGHTS_SAVE_PATH)
    mdp = RouteMDP(G)
    trips = data["test"][:N_TRIPS]
    out = {"n_trips": len(trips), "weights": dict(zip(FEATURES, map(float, w)))}

    # ---- A. sanity -----------------------------------------------------
    r_T = mdp.T_feat @ w
    out["A_reward"] = {
        "transitions": int(mdp.T),
        "frac_positive_reward": float((r_T > 0).mean()),
        "reward_min": float(r_T.min()), "reward_max": float(r_T.max()),
        "reward_mean": float(r_T.mean()),
    }
    # feature scale & collinearity of per-edge features actually used
    F = mdp.F_edge
    out["A_feature_stats"] = {f: {"mean": float(F[:, i].mean()),
                                  "std": float(F[:, i].std()),
                                  "max": float(F[:, i].max())}
                              for f, i in IDX.items()}
    corr = np.corrcoef(F.T + 1e-12 * np.random.default_rng(0).standard_normal(F.T.shape))
    pairs = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            if abs(corr[i, j]) > 0.8:
                pairs.append((FEATURES[i], FEATURES[j], float(corr[i, j])))
    out["A_collinear_pairs_gt_0.8"] = pairs

    d0 = trips[0]["destination"]
    V150 = mdp.soft_values(w, d0, n_iter=150)
    V600 = mdp.soft_values(w, d0, n_iter=600, tol=1e-9)
    ok = V600 > NEG / 2
    out["A_value_iter_gap_150_vs_600"] = float(np.max(np.abs(V150[ok] - V600[ok])))

    # ---- B/C. per-trip comparison -------------------------------------
    rows = []
    for t in trips:
        o, d = t["origin"], t["destination"]
        irl = mdp.most_likely_route(o, d, w)
        sp = dijkstra_shortest(G, o, d)
        if len(irl) < 2 or len(sp) < 2:
            continue
        rows.append({
            "n_edges": len(t["nodes"]) - 1,
            "J_irl": edge_jaccard(irl, t["nodes"]),
            "J_sp": edge_jaccard(sp, t["nodes"]),
            "J_irl_sp": edge_jaccard(irl, sp),
            "same_as_sp": float(irl == sp),
            "exp_over_sp_len": route_length(G, t["nodes"]) / max(route_length(G, sp), 1e-9),
            "irl_over_sp_len": route_length(G, irl) / max(route_length(G, sp), 1e-9),
        })
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["B_overall"] = {k: float(v.mean()) for k, v in a.items() if k != "n_edges"}
    qs = np.quantile(a["n_edges"], [0, 1 / 3, 2 / 3, 1])
    buckets = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (a["n_edges"] >= lo) & (a["n_edges"] <= hi)
        buckets.append({"edges": [float(lo), float(hi)], "n": int(m.sum()),
                        "J_irl": float(a["J_irl"][m].mean()),
                        "J_sp": float(a["J_sp"][m].mean()),
                        "same_as_sp": float(a["same_as_sp"][m].mean())})
    out["B_by_length_tercile"] = buckets
    d = a["J_irl"] - a["J_sp"]
    out["B_irl_wins_ties_losses"] = {"win": float((d > 1e-9).mean()),
                                     "tie": float((np.abs(d) <= 1e-9).mean()),
                                     "loss": float((d < -1e-9).mean())}
    # ceiling: for each trip, best of {irl, sp}
    out["C_oracle_best_of_irl_sp"] = float(np.maximum(a["J_irl"], a["J_sp"]).mean())
    out["C_experts_longer_than_sp_by"] = float(a["exp_over_sp_len"].mean() - 1)

    # ---- D. weight surgery --------------------------------------------
    def with_zero(names):
        v = w.copy()
        for n in names:
            v[IDX[n]] = 0.0
        return v

    groups = {
        "full_learned": w,
        "no_road_class": with_zero(["road_motorway", "road_primary", "road_residential"]),
        "no_turns": with_zero(["turn_straight", "turn_slight", "turn_sharp"]),
        "no_speed_lanes": with_zero(["speed_kph", "num_lanes"]),
        "no_travel_time": with_zero(["travel_time"]),
        "length_only": np.eye(len(w))[IDX["length"]] * -1.0,
        "time_only": np.eye(len(w))[IDX["travel_time"]] * -1.0,
    }
    out["D_weight_surgery_jaccard"] = {k: mean_jaccard(mdp, trips, v)
                                       for k, v in groups.items()}
    # unit test: pure-length reward must reproduce Dijkstra-shortest routes
    lo = groups["length_only"]
    agree = [edge_jaccard(mdp.most_likely_route(t["origin"], t["destination"], lo),
                          dijkstra_shortest(G, t["origin"], t["destination"]))
             for t in trips[:100]]
    out["D_length_only_vs_dijkstra_jaccard"] = float(np.mean(agree))

    # ---- E. calibration sweep -----------------------------------------
    sweep = {}
    for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
        v = w.copy()
        v[IDX["length"]] -= alpha
        sweep[f"extra_length_penalty_{alpha}"] = mean_jaccard(mdp, trips, v)
    out["E_length_sweep_jaccard"] = sweep

    path = os.path.join(RESULTS_DIR, "diagnostics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("saved ->", path)


if __name__ == "__main__":
    main()
