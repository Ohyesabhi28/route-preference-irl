"""
porto_data.py – Real taxi demonstrations from the Porto ECML/PKDD-2015 dataset.

Pipeline
  1. Build the central-Porto drive graph (largest strongly-connected component).
  2. Stream train.csv, keep trips that lie inside the bounding box and are long
     enough to contain a real routing decision.
  3. HMM map-match every GPS polyline onto the graph (Leuven MapMatching).
  4. Store each matched trip as an edge sequence  (u, v, key)  plus context
     (taxi id, hour, weekday, call type) for context-dependent IRL.

Run:  python src/porto_data.py --n 3000
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from math import radians, sin, cos, asin, sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (PORTO_BBOX, PORTO_GRAPH_PATH, PORTO_RAW_CSV,
                    PORTO_TRAJ_PATH, DATA_DIR, RANDOM_SEED, TRAIN_FRAC)

CAND_PATH = os.path.join(DATA_DIR, "porto_candidates.pkl")


# ──────────────────────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────────────────────
def load_porto_graph() -> nx.MultiDiGraph:
    import osmnx as ox
    from graph_builder import build_graph
    G = build_graph(PORTO_BBOX, PORTO_GRAPH_PATH)
    if not nx.is_strongly_connected(G):
        n0 = G.number_of_nodes()
        G = ox.truncate.largest_component(G, strongly=True)
        print(f"[porto] largest strongly-connected component: "
              f"{G.number_of_nodes()} / {n0} nodes")
    return G


def _haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * asin(sqrt(a))


# ──────────────────────────────────────────────────────────────
# Candidate trips from the raw CSV
# ──────────────────────────────────────────────────────────────
def load_candidates(max_candidates: int = 12000, min_pts: int = 15,
                    max_pts: int = 120, min_disp_m: float = 800.0) -> list[dict]:
    if os.path.exists(CAND_PATH):
        with open(CAND_PATH, "rb") as f:
            cands = pickle.load(f)
        print(f"[porto] loaded {len(cands)} cached candidate trips")
        return cands

    b = PORTO_BBOX
    cands = []
    t0 = time.time()
    usecols = ["TRIP_ID", "CALL_TYPE", "TAXI_ID", "TIMESTAMP", "DAY_TYPE",
               "MISSING_DATA", "POLYLINE"]
    for chunk in pd.read_csv(PORTO_RAW_CSV, usecols=usecols, chunksize=100_000):
        chunk = chunk[(chunk.MISSING_DATA == False) & (chunk.POLYLINE.str.len() > 2 * min_pts * 18 // 2)]
        for row in chunk.itertuples(index=False):
            pts = json.loads(row.POLYLINE)
            n = len(pts)
            if not (min_pts <= n <= max_pts):
                continue
            a = np.asarray(pts)
            if (a[:, 0].min() < b["west"] or a[:, 0].max() > b["east"] or
                    a[:, 1].min() < b["south"] or a[:, 1].max() > b["north"]):
                continue
            if _haversine(*a[0], *a[-1]) < min_disp_m:
                continue
            cands.append({
                "trip_id": row.TRIP_ID, "taxi_id": int(row.TAXI_ID),
                "timestamp": int(row.TIMESTAMP), "call_type": row.CALL_TYPE,
                "day_type": row.DAY_TYPE, "gps": a,
            })
        print(f"[porto]   scanned chunk, {len(cands)} candidates "
              f"({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(RANDOM_SEED)
    if len(cands) > max_candidates:
        keep = rng.choice(len(cands), max_candidates, replace=False)
        cands = [cands[i] for i in sorted(keep)]
    with open(CAND_PATH, "wb") as f:
        pickle.dump(cands, f)
    return cands


# ──────────────────────────────────────────────────────────────
# Map matching
# ──────────────────────────────────────────────────────────────
_STATE = {}


def _init_worker():
    from leuvenmapmatching.map.inmem import InMemMap
    G = load_porto_graph()
    mp = InMemMap("porto", use_latlon=True, use_rtree=True, index_edges=True)
    for n, d in G.nodes(data=True):
        mp.add_node(n, (d["y"], d["x"]))
    for u, v in set((u, v) for u, v, _ in G.edges(keys=True)):
        mp.add_edge(u, v)
    _STATE["map"] = mp
    _STATE["G"] = G


def _match_one(c: dict):
    from leuvenmapmatching.matcher.distance import DistanceMatcher
    G = _STATE["G"]
    matcher = DistanceMatcher(_STATE["map"], max_dist=60, max_dist_init=80,
                              min_prob_norm=0.001, obs_noise=20, obs_noise_ne=30,
                              non_emitting_states=True, max_lattice_width=8)
    path = [(lat, lon) for lon, lat in c["gps"]]
    try:
        _, last = matcher.match(path)
    except Exception:
        return None
    if last != len(path) - 1:
        return None
    nodes = list(matcher.path_pred_onlynodes)
    nodes = [n for i, n in enumerate(nodes) if i == 0 or n != nodes[i - 1]]
    if len(nodes) < 6:
        return None
    edges = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            return None                      # matcher produced a gap -> discard
        k = min(G[u][v], key=lambda kk: G[u][v][kk].get("length", 1e9))
        edges.append((u, v, k))
    hour = pd.to_datetime(c["timestamp"], unit="s").hour
    wd = pd.to_datetime(c["timestamp"], unit="s").weekday()
    return {
        "origin": nodes[0], "destination": nodes[-1],
        "nodes": nodes, "edges": edges,
        "trip_id": c["trip_id"], "taxi_id": c["taxi_id"],
        "timestamp": c["timestamp"], "hour": int(hour), "weekday": int(wd),
        "call_type": c["call_type"], "n_gps": len(path),
    }


def match_trips(cands: list[dict], n_target: int, workers: int = None) -> list[dict]:
    import multiprocessing as mp
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    out, tried, t0 = [], 0, time.time()
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for res in pool.imap(_match_one, cands, chunksize=8):
            tried += 1
            if res is not None:
                out.append(res)
            if tried % 200 == 0:
                print(f"[porto]   matched {len(out)}/{tried} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if len(out) >= n_target:
                pool.terminate()
                break
    print(f"[porto] matched {len(out)} of {tried} tried "
          f"({100*len(out)/max(tried,1):.0f}% success)")
    return out


def build_dataset(n_target: int = 3000, max_edges: int = 75) -> dict:
    cands = load_candidates()
    trips = match_trips(cands, int(n_target * 1.15))
    trips = [t for t in trips if 5 <= len(t["edges"]) <= max_edges][:n_target]
    rng = np.random.default_rng(RANDOM_SEED)
    # split by (origin, destination) hash is overkill here; split by taxi-agnostic shuffle
    order = rng.permutation(len(trips))
    trips = [trips[i] for i in order]
    split = int(len(trips) * TRAIN_FRAC)
    data = {"train": trips[:split], "test": trips[split:], "all": trips}
    with open(PORTO_TRAJ_PATH, "wb") as f:
        pickle.dump(data, f)
    L = [len(t["edges"]) for t in trips]
    print(f"[porto] saved {len(trips)} trips -> {PORTO_TRAJ_PATH}  "
          f"(train {split}, test {len(trips)-split}; edges/trip median {int(np.median(L))})")
    return data


def load_porto_trajectories() -> dict:
    with open(PORTO_TRAJ_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    build_dataset(args.n)
