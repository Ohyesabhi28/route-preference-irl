"""
Finite-difference check of the MaxEnt IRL objective gradient (incl. the
destination-dependent away_from_goal feature when IRL_FEATURES=v2).

    set IRL_FEATURES=v2 && python tests/test_gradient.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import region
from config import N_FEATURES, FEATURES
from maxent_irl import MaxEntIRL


def main(n_trips=25, eps=1e-5):
    G = region.load_graph()
    trajs = region.load_data(G)["fit"][:n_trips]
    irl = MaxEntIRL(G, n_workers=1, horizon=40)
    mdp = irl.mdp
    f_exp = np.sum([mdp.traj_features(t) for t in trajs], axis=0)
    by_dest = {}
    for i, t in enumerate(trajs):
        by_dest.setdefault(t["destination"], []).append(i)

    rng = np.random.default_rng(0)
    w = -0.3 * np.abs(rng.standard_normal(N_FEATURES)) - 0.05
    _, g = irl._objective(w, trajs, f_exp, by_dest)
    worst = 0.0
    for j in range(N_FEATURES):
        e = np.zeros(N_FEATURES)
        e[j] = eps
        fp, _ = irl._objective(w + e, trajs, f_exp, by_dest)
        fm, _ = irl._objective(w - e, trajs, f_exp, by_dest)
        num = (fp - fm) / (2 * eps)
        err = abs(num - g[j]) / max(1e-8, abs(num) + abs(g[j]))
        worst = max(worst, err)
        print(f"{FEATURES[j]:<16} analytic {g[j]:+.6f}  numeric {num:+.6f}  rel.err {err:.1e}")
    print("max relative error:", f"{worst:.2e}")
    assert worst < 1e-4, "gradient mismatch"
    print("OK")


if __name__ == "__main__":
    main()
