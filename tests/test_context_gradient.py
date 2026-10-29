"""Finite-difference check of the context-IRL objective (IRL_FEATURES=v2)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import region
from config import N_FEATURES
from context_irl import ContextMaxEntIRL, N_CTX


def main(n_trips=60, eps=1e-5, n_check=12):
    G = region.load_graph()
    trajs = region.load_data(G)["fit"][:n_trips]
    m = ContextMaxEntIRL(G, n_iter=0, n_workers=1, horizon=40)
    m.fit(trajs, verbose=False)
    fun = m._objective
    rng = np.random.default_rng(1)
    th = rng.standard_normal(N_CTX * N_FEATURES) * 0.1
    th[:N_FEATURES] = -0.3 * np.abs(rng.standard_normal(N_FEATURES)) - 0.05
    _, g = fun(th)
    worst = 0.0
    for j in rng.choice(len(th), n_check, replace=False):
        e = np.zeros_like(th)
        e[j] = eps
        num = (fun(th + e)[0] - fun(th - e)[0]) / (2 * eps)
        err = abs(num - g[j]) / max(1e-8, abs(num) + abs(g[j]))
        worst = max(worst, err)
        print(f"theta[{j:2d}] analytic {g[j]:+.6f} numeric {num:+.6f} rel.err {err:.1e}")
    print("max relative error:", f"{worst:.2e}")
    assert worst < 1e-4
    print("OK")


if __name__ == "__main__":
    main()
