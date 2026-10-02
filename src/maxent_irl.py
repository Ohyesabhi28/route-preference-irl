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
    N_FEATURES, IRL_N_ITERATIONS, WEIGHTS_SAVE_PATH, REGULARIZATION_L2, FEATURES
)
from mdp import RouteMDP


class MaxEntIRL:
    def __init__(self,
                 G: nx.MultiDiGraph,
                 n_iter: int = IRL_N_ITERATIONS,
                 l2: float = REGULARIZATION_L2,
                 horizon: int = 80):
        self.G = G
        self.n_iter = n_iter
        self.l2 = l2
        self.horizon = horizon
        self.mdp = RouteMDP(G)
        self.weights = np.zeros(N_FEATURES)
        self.history = []          # (iter, nll, grad_norm)

    # ------------------------------------------------------------------
    def _objective(self, w, trajs, f_exp, by_dest):
        """Average negative log-likelihood and its gradient."""
        mdp = self.mdp
        r_T = mdp.T_feat @ w
        acc_T = np.zeros(mdp.T)
        acc_0 = np.zeros(mdp.E)
        logZ = 0.0
        for d, idx in by_dest.items():
            L = mdp.soft_values_levels(w, d, self.horizon, r_T=r_T)
            for i in idx:
                logZ += mdp.expected_counts(w, L, trajs[i]["origin"], d,
                                            acc_T, acc_0, r_T=r_T)
        f_pi = acc_T @ mdp.T_feat + acc_0 @ mdp.F_edge
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

        res = minimize(fun, w_init, jac=True, method="L-BFGS-B", callback=cb,
                       options={"maxiter": self.n_iter, "gtol": 1e-5})
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
