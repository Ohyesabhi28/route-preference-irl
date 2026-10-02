"""
trajectory_generator.py – Simulate expert demonstrations.

Each demonstration is a draw from the MaxEnt policy of a *simulated driver*
whose reward weights are TRUE_WEIGHTS (stochastic, but with known preferences).
This lets us check that IRL recovers the weights, before moving to real GPS data.

  1. Sample an origin-destination pair on the road graph.
  2. Roll out the soft-optimal policy for the true weights.
  3. Record the edge sequence as one trajectory.
  4. Split into train / test by OD pair.
"""

import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm

from config import (
    TRAJ_SAVE_PATH, TRUE_WEIGHTS, N_TRAJECTORIES, TRAIN_FRAC, RANDOM_SEED
)
from mdp import RouteMDP


def generate_all_trajectories(G: nx.MultiDiGraph,
                              n: int = N_TRAJECTORIES,
                              weights: np.ndarray = None,
                              save: bool = True,
                              min_hops: int = 5,
                              max_hops: int = 40) -> dict:
    """Generate `n` demonstrations and split into train/test."""
    if weights is None:
        weights = np.array(TRUE_WEIGHTS, dtype=np.float64)
    rng = np.random.default_rng(RANDOM_SEED)
    mdp = RouteMDP(G)
    nodes = list(G.nodes())

    trajectories, attempts = [], 0
    pbar = tqdm(total=n, desc="Generating trajectories")
    while len(trajectories) < n and attempts < n * 50:
        attempts += 1
        o, d = (nodes[i] for i in rng.choice(len(nodes), 2, replace=False))
        try:
            hops = nx.shortest_path_length(G, o, d)
        except nx.NetworkXNoPath:
            continue
        if not (min_hops <= hops <= max_hops):
            continue
        chain = mdp.sample_trajectory(o, d, weights, rng)
        if chain is None or len(chain) < 3:
            continue
        edges = [mdp.edges[e] for e in chain]
        trajectories.append({
            "origin": o,
            "destination": d,
            "nodes": [edges[0][0]] + [e[1] for e in edges],
            "edges": edges,                       # (u, v, key)
        })
        pbar.update(1)
    pbar.close()
    print(f"[trajectory_gen] Generated {len(trajectories)} trajectories "
          f"({attempts} attempts)")

    order = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in order]
    split = int(len(trajectories) * TRAIN_FRAC)
    result = {
        "train": trajectories[:split],
        "test":  trajectories[split:],
        "all":   trajectories,
        "true_weights": weights,
    }
    if save:
        with open(TRAJ_SAVE_PATH, "wb") as f:
            pickle.dump(result, f)
        print(f"[trajectory_gen] Saved -> {TRAJ_SAVE_PATH}")
    return result


def load_trajectories() -> dict:
    with open(TRAJ_SAVE_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from graph_builder import build_graph
    G = build_graph()
    data = generate_all_trajectories(G)
    print(f"Train: {len(data['train'])}  Test: {len(data['test'])}")
    ex = data["train"][0]
    print(f"Example trajectory: {len(ex['edges'])} edges, "
          f"O={ex['origin']}, D={ex['destination']}")
