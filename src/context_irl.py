"""
context_irl.py – B2: context-conditioned MaxEnt IRL.

    w_c = w_0 + delta_c        (delta_offpeak = 0, reference context)

Contexts come from the trip's timestamp (night / weekend / weekday peak /
off-peak).  The model is MaxEnt IRL with block-structured parameters
theta = [w_0, delta_peak, delta_night, delta_weekend]; the objective re-uses
the exact per-destination forward/backward passes of `maxent_irl`.

L2 acts on w_0 (l2) and, more strongly, on the deltas (l2_delta), which
shrinks every context towards the shared reward.
"""
import time
import numpy as np
from scipy.optimize import minimize

import maxent_irl as M
from config import N_FEATURES, MAX_HORIZON, REGULARIZATION_L2
from mdp import RouteMDP

CONTEXTS = ["offpeak", "peak", "night", "weekend"]
N_CTX = len(CONTEXTS)


def context_of(trip: dict) -> int:
    h, wd = trip["hour"], trip["weekday"]
    if h >= 22 or h < 6:
        return CONTEXTS.index("night")
    if wd >= 5:
        return CONTEXTS.index("weekend")
    if 7 <= h <= 9 or 16 <= h <= 19:
        return CONTEXTS.index("peak")
    return CONTEXTS.index("offpeak")


def effective_weights(theta: np.ndarray, shrink: float = 1.0) -> np.ndarray:
    """(N_CTX, N_FEATURES) context weights; `shrink` scales every delta."""
    th = np.asarray(theta).reshape(N_CTX, N_FEATURES)
    W = np.tile(th[0], (N_CTX, 1))
    W[1:] += shrink * th[1:]
    return W


class ContextMaxEntIRL:
    def __init__(self, G, n_iter=40, l2=REGULARIZATION_L2, l2_delta=0.02,
                 horizon=MAX_HORIZON, n_workers=None):
        self.G, self.n_iter, self.l2, self.l2_delta = G, n_iter, l2, l2_delta
        self.horizon, self.n_workers = horizon, n_workers
        self.mdp = RouteMDP(G)
        self.theta = np.zeros(N_CTX * N_FEATURES)
        self.history = []

    def fit(self, trajs, w0=None, verbose=True):
        mdp, H = self.mdp, self.horizon
        by_ctx = [[] for _ in range(N_CTX)]
        for t in trajs:
            by_ctx[context_of(t)].append(t)
        f_exp = np.array([np.sum([mdp.traj_features(t) for t in ts], axis=0)
                          if ts else np.zeros(N_FEATURES) for ts in by_ctx])
        n = len(trajs)
        items = []
        for ts in by_ctx:
            by_dest = {}
            for t in ts:
                by_dest.setdefault(t["destination"], []).append(t["origin"])
            items.append(list(by_dest.items()))
        print("[CtxIRL] trips per context:",
              {c: len(ts) for c, ts in zip(CONTEXTS, by_ctx)})

        import os
        nw = self.n_workers or max(1, (os.cpu_count() or 2) - 1)
        pool = None
        if nw > 1:
            import multiprocessing as mp
            pool = mp.Pool(nw, initializer=M._worker_init, initargs=(self.G,))
        else:
            M._W["mdp"] = mdp
        chunks = []
        for c in range(N_CTX):
            k = max(1, min(nw * 2, len(items[c])))
            chunks += [(c, items[c][i::k]) for i in range(k) if items[c][i::k]]

        l2v = np.zeros((N_CTX, N_FEATURES))
        l2v[0], l2v[1:] = self.l2, self.l2_delta
        t0 = time.time()
        state = {"it": 0}
        last = {}

        def fun(theta):
            th = theta.reshape(N_CTX, N_FEATURES)
            W = effective_weights(theta)
            tasks = [(W[c], H, ch) for c, ch in chunks]
            res = pool.map(M._worker_task, tasks) if pool else \
                [M._worker_task(tk) for tk in tasks]
            logZ = np.zeros(N_CTX)
            f_pi = np.zeros((N_CTX, N_FEATURES))
            for (c, _), (lz, f) in zip(chunks, res):
                logZ[c] += lz
                f_pi[c] += f
            nll = sum(logZ[c] - W[c] @ f_exp[c] for c in range(N_CTX)) / n
            nll += 0.5 * float((l2v * th * th).sum())
            g_c = (f_pi - f_exp) / n                     # grad wrt W_c
            g = np.zeros_like(th)
            g[0] = g_c.sum(axis=0)
            g[1:] = g_c[1:]
            g += l2v * th
            last["nll"], last["g"] = nll, g.ravel()
            return nll, g.ravel()

        def cb(_):
            self.history.append((state["it"], float(last["nll"]),
                                 float(np.linalg.norm(last["g"]))))
            if verbose and state["it"] % 5 == 0:
                print(f"[CtxIRL] iter {state['it']:3d}  NLL/traj = "
                      f"{last['nll']:8.4f}  |grad| = {np.linalg.norm(last['g']):.5f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            state["it"] += 1

        self._objective = fun                    # exposed for gradient tests
        th0 = np.zeros((N_CTX, N_FEATURES))
        if w0 is not None:
            th0[0] = w0
        try:
            res = minimize(fun, th0.ravel(), jac=True, method="L-BFGS-B",
                           callback=cb,
                           options={"maxiter": self.n_iter, "gtol": 1e-5})
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        self.theta = res.x
        print(f"[CtxIRL] done in {time.time()-t0:.0f}s – {res.message}")
        return self.theta
