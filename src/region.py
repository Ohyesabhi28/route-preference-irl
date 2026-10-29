"""
region.py – one place that knows how to load the graph and demonstrations for
the active region (Porto = real taxi GPS, SF = synthetic demo).
"""
import os
from config import SYNTHETIC, TRAJ_SAVE_PATH, PORTO_TRAJ_PATH


def load_graph():
    if SYNTHETIC:
        from graph_builder import build_graph
        return build_graph()
    from porto_data import load_porto_graph
    return load_porto_graph()


def data_exists() -> bool:
    return os.path.exists(TRAJ_SAVE_PATH if SYNTHETIC else PORTO_TRAJ_PATH)


def load_data(G=None):
    """Return {"train": [...], "test": [...]} of demonstrations."""
    if SYNTHETIC:
        from trajectory_generator import generate_all_trajectories, load_trajectories
        if os.path.exists(TRAJ_SAVE_PATH):
            return load_trajectories()
        return generate_all_trajectories(G)
    from porto_data import load_porto_trajectories
    if not os.path.exists(PORTO_TRAJ_PATH):
        raise FileNotFoundError(
            "Porto trips not found. Run:  python src/porto_data.py --n 3000")
    return with_validation(load_porto_trajectories())


def with_validation(data: dict, val_frac: float = 0.15) -> dict:
    """Add data['fit'] / data['val'] carved out of data['train'] (test untouched).
    Hyper-parameters (e.g. calibration) are chosen on 'val', never on 'test'."""
    n_val = int(len(data["train"]) * val_frac)
    data["fit"] = data["train"][:-n_val] if n_val else data["train"]
    data["val"] = data["train"][-n_val:] if n_val else []
    return data
