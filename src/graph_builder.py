"""
graph_builder.py – Download the SF drive graph from OSM, compute edge
features, and save it to disk.

Features computed per directed edge (u, v, key):
  length, speed_kph, travel_time,
  road_motorway, road_primary, road_residential,
  turn_straight, turn_slight, turn_sharp,
  num_lanes
"""

import os
import math
import pickle
import numpy as np
import networkx as nx

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

from config import (
    SF_BBOX, GRAPH_SAVE_PATH, FEATURES, DATA_DIR
)

# ──────────────────────────────────────────────────────────────
# Speed defaults (km/h) for OSM highway types
# ──────────────────────────────────────────────────────────────
DEFAULT_SPEED = {
    "motorway": 100, "motorway_link": 80,
    "trunk": 80,     "trunk_link": 60,
    "primary": 60,   "primary_link": 50,
    "secondary": 50, "secondary_link": 40,
    "tertiary": 40,  "tertiary_link": 30,
    "residential": 30, "living_street": 15,
    "unclassified": 30, "road": 30,
    "service": 15,
}

LENGTH_SCALE = 100.0   # metres
SPEED_SCALE  = 50.0    # km/h
TIME_SCALE   = 60.0    # seconds
LANES_SCALE  = 2.0


def _parse_speed(val) -> float:
    """Convert maxspeed tag to float km/h."""
    if val is None:
        return 30.0
    if isinstance(val, list):
        val = val[0]
    s = str(val).replace("mph", "").replace("km/h", "").strip()
    try:
        kmh = float(s)
        if "mph" in str(val):
            kmh *= 1.60934
        return max(5.0, min(kmh, 130.0))
    except ValueError:
        return 30.0


def _parse_lanes(val) -> int:
    if val is None:
        return 1
    if isinstance(val, list):
        val = val[0]
    try:
        return max(1, int(str(val).strip()))
    except ValueError:
        return 1


def _highway_type(hw) -> str:
    if isinstance(hw, list):
        hw = hw[0]
    hw = str(hw).strip()
    if hw.startswith("["):           # list serialised by GraphML
        hw = hw.strip("[]").split(",")[0].strip(" '\"")
    return hw


def _bearing_diff(b1: float, b2: float) -> float:
    """Signed angular difference in [−180, 180]."""
    d = (b2 - b1 + 360) % 360
    return d - 360 if d > 180 else d


def build_graph() -> nx.MultiDiGraph:
    """
    Download or load the SF drive graph, enrich edges with features,
    and return a NetworkX MultiDiGraph.
    """
    # ── Load if cached ─────────────────────────────────────────
    if os.path.exists(GRAPH_SAVE_PATH) and OSMNX_AVAILABLE:
        print("[graph_builder] Loading cached graph …")
        G = ox.load_graphml(GRAPH_SAVE_PATH)
        G = _ensure_features(G)
        return G

    if OSMNX_AVAILABLE:
        print("[graph_builder] Downloading SF OSM graph …")
        try:
            # OSMnx 2.x API: bbox=(left, bottom, right, top)
            G = ox.graph_from_bbox(
                bbox=(
                    SF_BBOX["west"], SF_BBOX["south"],
                    SF_BBOX["east"], SF_BBOX["north"],
                ),
                network_type="drive",
            )
        except TypeError:
            # OSMnx 1.x fallback
            G = ox.graph_from_bbox(
                north=SF_BBOX["north"], south=SF_BBOX["south"],
                east=SF_BBOX["east"],  west=SF_BBOX["west"],
                network_type="drive",
            )
        try:
            G = ox.add_edge_bearings(G)
        except Exception:
            pass  # bearings computed manually in _ensure_features
        ox.save_graphml(G, GRAPH_SAVE_PATH)
        print(f"[graph_builder] Saved graph → {GRAPH_SAVE_PATH}")
    else:
        print("[graph_builder] osmnx not available – building toy SF grid …")
        G = _build_toy_sf_grid()

    G = _ensure_features(G)
    return G


# ──────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────

def _ensure_features(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add / normalise feature columns on every directed edge."""
    # Compute bearings if missing
    for u, v, k, d in G.edges(data=True, keys=True):
        if "bearing" not in d:
            try:
                u_d, v_d = G.nodes[u], G.nodes[v]
                bearing = math.degrees(math.atan2(
                    v_d["x"] - u_d["x"],
                    v_d["y"] - u_d["y"]
                )) % 360
                G[u][v][k]["bearing"] = bearing
            except Exception:
                G[u][v][k]["bearing"] = 0.0

    # Per-edge feature vector.
    # Features are scaled to roughly unit magnitude so gradient-based IRL is
    # well conditioned.  The three turn features are left at 0 here: a turn
    # depends on the (incoming edge, outgoing edge) pair, so RouteMDP fills
    # them in per transition.
    for u, v, k, d in G.edges(data=True, keys=True):
        hw = _highway_type(d.get("highway", "residential"))
        length = float(d.get("length", 100.0))
        if d.get("maxspeed") is None:
            speed = DEFAULT_SPEED.get(hw, 30.0)
        else:
            speed = _parse_speed(d.get("maxspeed"))
        travel_time = (length / 1000.0) / speed * 3600.0  # seconds

        # Road class one-hot
        if hw in ("motorway", "trunk", "motorway_link", "trunk_link"):
            rm, rp, rr = 1, 0, 0
        elif hw in ("primary", "secondary", "primary_link", "secondary_link"):
            rm, rp, rr = 0, 1, 0
        else:
            rm, rp, rr = 0, 0, 1

        lanes = _parse_lanes(d.get("lanes"))

        feat = np.array([
            length / LENGTH_SCALE,
            speed / SPEED_SCALE,
            travel_time / TIME_SCALE,
            rm, rp, rr,
            0.0, 0.0, 0.0,
            lanes / LANES_SCALE,
        ], dtype=np.float64)

        G[u][v][k]["features"] = feat
        G[u][v][k]["bearing"] = float(d.get("bearing", 0.0))
        G[u][v][k]["length"] = length
        G[u][v][k]["travel_time"] = travel_time

    return G


# ──────────────────────────────────────────────────────────────
# Toy grid fallback (used when osmnx unavailable)
# ──────────────────────────────────────────────────────────────

def _build_toy_sf_grid(rows: int = 20, cols: int = 20) -> nx.MultiDiGraph:
    """
    Build a synthetic grid graph mimicking an urban block layout.
    Nodes represent intersections; edges are directed road segments.
    """
    G = nx.MultiDiGraph()
    import random
    rng = random.Random(42)

    lat0, lon0 = 37.760, -122.430
    dlat, dlon = 0.002, 0.003  # ~200 m blocks

    node_id = 0
    node_map = {}
    for r in range(rows):
        for c in range(cols):
            nid = node_id
            G.add_node(nid,
                       y=lat0 + r * dlat,
                       x=lon0 + c * dlon,
                       osmid=nid)
            node_map[(r, c)] = nid
            node_id += 1

    hw_choices = (["residential"] * 8 +
                  ["primary"] * 3 +
                  ["secondary"] * 3 +
                  ["motorway"])

    for r in range(rows):
        for c in range(cols):
            u = node_map[(r, c)]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    v = node_map[(nr, nc)]
                    hw = rng.choice(hw_choices)
                    length = rng.uniform(80, 250)
                    G.add_edge(u, v, key=0,
                               highway=hw,
                               length=length,
                               maxspeed=DEFAULT_SPEED.get(hw, 30),
                               lanes=rng.randint(1, 3),
                               bearing=rng.uniform(0, 360))
    print(f"[graph_builder] Toy grid: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


if __name__ == "__main__":
    G = build_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # Show feature shape
    sample_edge = next(iter(G.edges(data=True, keys=True)))
    print(f"Feature vector shape: {sample_edge[3]['features'].shape}")
    print(f"Features: {FEATURES}")
    print(f"Sample feature vector: {sample_edge[3]['features']}")
