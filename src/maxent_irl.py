"""
maxent_irl.py – Maximum Entropy Inverse Reinforcement Learning

Algorithm (Ziebart et al., 2008), exact version:
  For the current w:
    a. backward pass  – soft value iteration per destination  (V)
    b. forward pass   – expected feature counts E_pi[phi] for each expert (o, d)
    c. gradient of the negative log-likelihood = E_pi[phi] - phi_expert
  and optimise w with L-BFGS (L2-regularised).

Both feature expectations are *per trajectory sums* averaged over the
demonstrations, so expert and policy counts are on the same scale.
"""

import os
import time
import pickle
import numpy as np
import networkx as nx
from scipy.optimize import minimize

from config import (
    N_FEATURES, IRL_N_ITERATIONS, WEIGHTS_SAVE_PATH, REGULARIZATION_L2, FEATURES,
    MAX_HORIZON
)
from mdp import RouteMDP


# ---- multiprocessing workers (one MDP copy per process) ---------------------
_W = {}


def _worker_init(G):
    _W["mdp"] = RouteMDP(G)


def _worker_task(args):
    """Backward + forward pass for a chunk of destinations."""
    w, horizon, items = args
    mdp = _W["mdp"]
    acc_T = np.zeros(mdp.T)
    acc_0 = np.zeros(mdp.E)
    acc_D = np.zeros(len(mdp.dest_cols)) if mdp.dest_cols else None
    logZ = 0.0
    for d, origins in items:
        r_T = mdp.reward_T(w, d)
        L = mdp.soft_values_levels(w, d, horizon, r_T=r_T)
        for o in origins:
            logZ += mdp.expected_counts(w, L, o, d, acc_T, acc_0, r_T=r_T,
                                        acc_D=acc_D)
    f = acc_T @ mdp.T_feat + acc_0 @ mdp.F_edge
    if acc_D is not None:
        f[mdp.dest_cols] += acc_D
    return logZ, f


class MaxEntIRL:
    def __init__(self,
                 G: nx.MultiDiGraph,
                 n_iter: int = IRL_N_ITERATIONS,
                 l2: float = REGULARIZATION_L2,
                 horizon: int = MAX_HORIZON,
                 n_workers: int = None):
        self.G = G
        self.n_iter = n_iter
        self.l2 = l2
        self.horizon = horizon
        self.n_workers = n_workers
        self._pool = None
        self.mdp = RouteMDP(G)
        self.weights = np.zeros(N_FEATURES)
        self.history = []          # (iter, nll, grad_norm)

    # ------------------------------------------------------------------
    def _objective(self, w, trajs, f_exp, by_dest):
        """Average negative log-likelihood and its gradient."""
        mdp = self.mdp
        if self._pool is not None:
            chunks = self._chunks
            res = self._pool.map(_worker_task,
                                 [(w, self.horizon, c) for c in chunks])
            logZ = sum(r[0] for r in res)
            f_pi = np.sum([r[1] for r in res], axis=0)
        else:
            acc_T = np.zeros(mdp.T)
            acc_0 = np.zeros(mdp.E)
            acc_D = np.zeros(len(mdp.dest_cols)) if mdp.dest_cols else None
            logZ = 0.0
            for d, idx in by_dest.items():
                r_T = mdp.reward_T(w, d)
                L = mdp.soft_values_levels(w, d, self.horizon, r_T=r_T)
                for i in idx:
                    logZ += mdp.expected_counts(w, L, trajs[i]["origin"], d,
                                                acc_T, acc_0, r_T=r_T,
                                                acc_D=acc_D)
            f_pi = acc_T @ mdp.T_feat + acc_0 @ mdp.F_edge
            if acc_D is not None:
                f_pi[mdp.dest_cols] += acc_D
        n = len(trajs)
        ll = (float(w @ f_exp) - logZ) / n
        grad = -(f_exp - f_pi) / n + self.l2 * w
        nll = -ll + 0.5 * self.l2 * float(w @ w)
        return nll, grad

    def train(self, train_trajectories: list[dict], verbose: bool = True,
              w0: np.ndarray = None) -> np.ndarray:
        mdp = self.mdp
        trajs = train_trajectories
        f_exp = np.sum([mdp.traj_features(t) for t in trajs], axis=0)
        by_dest = {}
        for i, t in enumerate(trajs):
            by_dest.setdefault(t["destination"], []).append(i)

        print(f"[MaxEntIRL] {len(trajs)} demos, {len(by_dest)} destinations, "
              f"mu_expert/traj = {(f_exp / len(trajs)).round(3)}")

        w_init = np.zeros(N_FEATURES) if w0 is None else w0.copy()

        import os as _os
        nw = self.n_workers if self.n_workers is not None else max(1, (_os.cpu_count() or 2) - 1)
        if nw > 1 and len(by_dest) > 50:
            import multiprocessing as mp
            items = [(d, [trajs[i]["origin"] for i in idx]) for d, idx in by_dest.items()]
            n_chunks = nw * 4
            self._chunks = [items[k::n_chunks] for k in range(n_chunks)]
            self._pool = mp.Pool(nw, initializer=_worker_init, initargs=(self.G,))
            print(f"[MaxEntIRL] parallel: {nw} workers, {n_chunks} chunks")
        self.history = []
        t0 = time.time()
        state = {"it": 0}

        last = {}

        def fun(w):
            nll, g = self._objective(w, trajs, f_exp, by_dest)
            last["nll"], last["g"] = nll, g
            return nll, g

        def cb(w):
            nll, g = last["nll"], last["g"]
            self.history.append((state["it"], float(nll), float(np.linalg.norm(g))))
            if verbose and state["it"] % 5 == 0:
                print(f"[MaxEntIRL] iter {state['it']:3d}  NLL/traj = {nll:8.4f}  "
                      f"|grad| = {np.linalg.norm(g):.5f}  ({time.time()-t0:.0f}s)")
            state["it"] += 1

        try:
            res = minimize(fun, w_init, jac=True, method="L-BFGS-B", callback=cb,
                           options={"maxiter": self.n_iter, "gtol": 1e-5})
        finally:
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None
        self.weights = res.x
        print(f"[MaxEntIRL] Done in {time.time()-t0:.0f}s – {res.message}")
        print("[MaxEntIRL] Learned weights:")
        for f, v in zip(FEATURES, self.weights):
            print(f"    {f:<18} {v:+.4f}")
        self.save()
        return self.weights

    # ------------------------------------------------------------------
    def save(self):
        np.save(WEIGHTS_SAVE_PATH, self.weights)
        hist_path = WEIGHTS_SAVE_PATH.replace(".npy", "_history.pkl")
        with open(hist_path, "wb") as f:
            pickle.dump(self.history, f)
        print(f"[MaxEntIRL] Saved weights -> {WEIGHTS_SAVE_PATH}")

    @classmethod
    def load(cls, G: nx.MultiDiGraph) -> "MaxEntIRL":
        obj = cls(G)
        obj.weights = np.load(WEIGHTS_SAVE_PATH)
        hist_path = WEIGHTS_SAVE_PATH.replace(".npy", "_history.pkl")
        if os.path.exists(hist_path):
            with open(hist_path, "rb") as f:
                obj.history = pickle.load(f)
        return obj
