"""
api.py – Flask REST API that serves the IRL pipeline results
         and supports interactive route queries from the web dashboard.

Endpoints:
  GET  /api/status            → pipeline status (trained / not trained)
  GET  /api/graph/stats       → node/edge counts, bbox
  GET  /api/weights           → learned weight vector + feature names
  GET  /api/history           → IRL training history (loss, grad_norm)
  GET  /api/evaluation        → evaluation results table
  POST /api/route             → route for a given O/D pair + method
  GET  /api/trajectories      → sample trajectories for visualisation
  POST /api/run_pipeline      → trigger full pipeline training
"""

import os
import sys
import json
import pickle
import threading
import traceback
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (
    BASE_DIR, WEIGHTS_SAVE_PATH, RESULTS_DIR, TRAJ_SAVE_PATH,
    FEATURES, TRUE_WEIGHTS, REGION_BBOX, REGION_NAME, SYNTHETIC, FEATURE_SET
)
import region

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")

app = Flask(__name__, static_folder=DASHBOARD_DIR, static_url_path="/static-dashboard")
CORS(app)


@app.route("/")
def dashboard():
    """Serve the dashboard over http:// (map tile servers reject file:// pages)."""
    from flask import send_from_directory
    return send_from_directory(DASHBOARD_DIR, "index.html")

# ──────────────────────────────────────────────────────────────
# Global state (populated lazily)
# ──────────────────────────────────────────────────────────────
_state = {
    "G":        None,
    "irl":      None,
    "data":     None,
    "clf":      None,
    "results":  None,
    "training": False,
    "error":    None,
    "log":      [],
}


def _log(msg: str):
    _state["log"].append(msg)
    print(msg)


def _load_assets():
    """Load graph, weights, trajectories, and results if they exist."""
    if _state["G"] is None:
        _log("Loading graph …")
        _state["G"] = region.load_graph()

    if _state["data"] is None and region.data_exists():
        _state["data"] = region.load_data(_state["G"])

    if _state["irl"] is None and os.path.exists(WEIGHTS_SAVE_PATH):
        from maxent_irl import MaxEntIRL
        _state["irl"] = MaxEntIRL.load(_state["G"])

    if _state["clf"] is None and _state["data"] is not None:
        from baselines import MyopicClassifier
        clf = MyopicClassifier()
        if _state["G"] is not None:
            clf.train(_state["data"]["train"], _state["G"])
            _state["clf"] = clf

    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    b1_path = os.path.join(RESULTS_DIR, "b1_results.json")
    if _state["results"] is None and os.path.exists(results_path):
        with open(results_path) as f:
            _state["results"] = json.load(f)
    elif _state["results"] is None and os.path.exists(b1_path):
        _state["results"] = _results_from_b1(b1_path)


def _results_from_b1(path):
    """Flatten run_b1.py output into the {method: {metric: mean}} table the
    dashboard expects; the calibrated v2/v3 model is shown as 'IRL (MaxEnt)'."""
    with open(path) as f:
        summ = json.load(f)["summary"]
    cal = [k for k in summ if k == f"IRL-{FEATURE_SET} (calibrated)"]
    out = {}
    for name, m in summ.items():
        if name.startswith("IRL") and name not in cal:
            continue
        label = "IRL (MaxEnt)" if name in cal else name
        out[label] = {k: v["mean"] for k, v in m.items()}
    return out


def _route_weights(irl):
    """Weights used for route generation: the raw weights plus the extra
    length penalty alpha chosen on validation data by run_b1.py (if present)."""
    w = irl.weights.copy()
    b1_path = os.path.join(RESULTS_DIR, "b1_results.json")
    if os.path.exists(b1_path) and "length" in FEATURES:
        with open(b1_path) as f:
            alpha = json.load(f).get("alpha_selected", 0.0)
        w[FEATURES.index("length")] -= alpha
    return w


def _run_pipeline_bg():
    """Run the full training pipeline in a background thread."""
    try:
        _state["training"] = True
        _state["error"]    = None
        _log("Pipeline started …")
        from train import main
        results, irl, G, data = main()
        _state["G"]       = G
        _state["irl"]     = irl
        _state["data"]    = data
        _state["results"] = results
        _log("Pipeline complete.")
    except Exception as e:
        _state["error"] = str(e)
        _log(f"Pipeline error: {e}")
        traceback.print_exc()
    finally:
        _state["training"] = False


# ──────────────────────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    _load_assets()
    return jsonify({
        "graph_loaded":    _state["G"] is not None,
        "irl_trained":     _state["irl"] is not None,
        "trajectories":    _state["data"] is not None,
        "evaluation_done": _state["results"] is not None,
        "training":        _state["training"],
        "irl_weights_ready": _state["irl"] is not None,
        "region":          REGION_NAME,
        "synthetic":       SYNTHETIC,
        "n_train":         len(_state["data"]["train"]) if _state["data"] else None,
        "n_test":          len(_state["data"]["test"]) if _state["data"] else None,
        "error":           _state["error"],
        "log":             _state["log"][-20:],
    })


@app.route("/api/graph/stats")
def graph_stats():
    _load_assets()
    G = _state["G"]
    if G is None:
        return jsonify({"error": "Graph not loaded"}), 503

    # Sample nodes for frontend visualisation (max 500)
    nodes_list = list(G.nodes(data=True))
    import random
    if len(nodes_list) > 500:
        nodes_list = random.sample(nodes_list, 500)

    nodes_out = [{"id": n,
                  "lat": d.get("y", 0),
                  "lon": d.get("x", 0)}
                 for n, d in nodes_list]

    return jsonify({
        "n_nodes":   G.number_of_nodes(),
        "n_edges":   G.number_of_edges(),
        "bbox":      REGION_BBOX,
        "region":    REGION_NAME,
        "synthetic": SYNTHETIC,
        "sample_nodes": nodes_out,
    })


@app.route("/api/weights")
def weights():
    _load_assets()
    irl = _state["irl"]
    if irl is None:
        return jsonify({"error": "IRL not trained"}), 503

    true_w = np.array(TRUE_WEIGHTS, dtype=np.float64).tolist() if SYNTHETIC else None

    return jsonify({
        "features":     FEATURES,
        "learned":      irl.weights.tolist(),
        "true":         true_w,
        "synthetic":    SYNTHETIC,
    })


@app.route("/api/history")
def history():
    hist_path = os.path.join(RESULTS_DIR, "irl_history.pkl")
    # Also try weights history
    alt_path  = WEIGHTS_SAVE_PATH.replace(".npy", "_history.pkl")

    path = hist_path if os.path.exists(hist_path) else alt_path
    if not os.path.exists(path):
        return jsonify({"error": "No training history"}), 404

    with open(path, "rb") as f:
        hist = pickle.load(f)

    if _state["irl"] is not None and _state["irl"].history:
        hist = _state["irl"].history

    return jsonify({
        "iterations": [h[0] for h in hist],
        "loss":       [h[1] for h in hist],
        "grad_norm":  [h[2] for h in hist],
    })


@app.route("/api/evaluation")
def evaluation():
    _load_assets()
    if _state["results"] is None:
        return jsonify({"error": "No evaluation results"}), 404
    return jsonify(_state["results"])


@app.route("/api/trajectories")
def trajectories():
    _load_assets()
    data = _state["data"]
    G    = _state["G"]
    if data is None or G is None:
        return jsonify({"trajectories": []})

    import random
    samples = random.sample(data["test"][:50],
                            min(5, len(data["test"])))

    out = []
    for traj in samples:
        coords = []
        for n in traj["nodes"]:
            nd = G.nodes.get(n, {})
            coords.append({
                "lat": nd.get("y", 0),
                "lon": nd.get("x", 0),
            })
        out.append({
            "origin":      traj["origin"],
            "destination": traj["destination"],
            "n_edges":     len(traj["edges"]),
            "coords":      coords,
        })

    return jsonify({"trajectories": out})


@app.route("/api/route", methods=["POST"])
def route():
    _load_assets()
    body = request.get_json() or {}
    method = body.get("method", "irl")
    origin = body.get("origin")
    dest   = body.get("destination")

    G   = _state["G"]
    irl = _state["irl"]

    if G is None:
        return jsonify({"error": "Graph not loaded"}), 503

    # If no OD given, sample a random test pair
    if origin is None or dest is None:
        if _state["data"]:
            import random
            pool   = [t for t in _state["data"]["test"] if len(t["edges"]) >= 12] or _state["data"]["test"]
            traj   = random.choice(pool)
            origin = traj["origin"]
            dest   = traj["destination"]
            expert_nodes = traj["nodes"]
        else:
            nodes  = list(G.nodes())
            import random
            origin = random.choice(nodes)
            dest   = random.choice(nodes)
            expert_nodes = []
    else:
        expert_nodes = []

    # Generate route
    try:
        if method == "irl" and irl is not None:
            if _state.get("mdp") is None:
                from mdp import RouteMDP
                _state["mdp"] = RouteMDP(G)
            nodes = _state["mdp"].most_likely_route(origin, dest, _route_weights(irl))
        elif method == "shortest":
            from baselines import dijkstra_shortest
            nodes = dijkstra_shortest(G, origin, dest)
        elif method == "fastest":
            from baselines import dijkstra_fastest
            nodes = dijkstra_fastest(G, origin, dest)
        elif method == "myopic" and _state["clf"]:
            nodes = _state["clf"].predict_route(G, origin, dest)
        else:
            from baselines import dijkstra_shortest
            nodes = dijkstra_shortest(G, origin, dest)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Convert to lat/lon
    def to_coords(node_list):
        out = []
        for n in node_list:
            nd = G.nodes.get(n, {})
            out.append({"lat": nd.get("y", 0), "lon": nd.get("x", 0)})
        return out

    return jsonify({
        "method":   method,
        "origin":   origin,
        "destination": dest,
        "route":    to_coords(nodes),
        "n_edges":  max(0, len(nodes)-1),
        "expert":   to_coords(expert_nodes),
    })


# ──────────────────────────────────────────────────────────────
# Maps-style endpoints: all route options for one trip in a single call
# ──────────────────────────────────────────────────────────────
def _edge_between(G, u, v):
    k = min(G[u][v], key=lambda kk: G[u][v][kk].get("length", 1e9))
    return G[u][v][k]


def _route_stats(G, nodes):
    """Distance, free-flow time (speed-limit based), turns and road mix of a node path."""
    if len(nodes) < 2:
        return None
    dist = tsec = resid = 0.0
    turns = sharp = 0
    prev_b = None
    for u, v in zip(nodes[:-1], nodes[1:]):
        d = _edge_between(G, u, v)
        L = float(d.get("length", 0.0))
        dist += L
        tsec += float(d.get("travel_time", L / 8.3))
        f = d.get("features")
        if f is not None and len(f) > 5 and f[5] > 0.5:
            resid += L
        b = float(d.get("bearing", 0.0))
        if prev_b is not None:
            ang = abs(((b - prev_b + 180.0) % 360.0) - 180.0)
            turns += ang >= 30
            sharp += ang >= 90
        prev_b = b
    return {
        "distance_m": round(dist),
        "time_s": round(tsec),
        "turns": int(turns),
        "sharp_turns": int(sharp),
        "pct_residential": round(100.0 * resid / dist) if dist else 0,
        "n_edges": len(nodes) - 1,
    }


def _coords(G, nodes):
    return [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in nodes]


def _nearest_node(G, lat, lon):
    if _state.get("xy") is None:
        ids = list(G.nodes())
        _state["xy"] = (ids, np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in ids]))
    ids, xy = _state["xy"]
    dlat = xy[:, 0] - lat
    dlon = (xy[:, 1] - lon) * np.cos(np.radians(lat))
    return ids[int(np.argmin(dlat * dlat + dlon * dlon))]



# ── Place names (offline: street names stored in the OSM graph) ──
import unicodedata


def _norm(t):
    return "".join(c for c in unicodedata.normalize("NFKD", str(t).lower()) if not unicodedata.combining(c))


def _street_index(G):
    """name -> list of node ids on that street; node -> set of street names."""
    if _state.get("streets") is None:
        by_name, by_node = {}, {}
        for u, v, d in G.edges(data=True):
            nm = d.get("name")
            if isinstance(nm, list):
                nm = nm[0] if nm else None
            if not nm:
                continue
            by_name.setdefault(nm, set()).update((u, v))
            by_node.setdefault(u, set()).add(nm)
            by_node.setdefault(v, set()).add(nm)
        _state["streets"] = (by_name, by_node, {n: _norm(n) for n in by_name})
    return _state["streets"]


@app.route("/api/places")
def places():
    """Autocomplete street names. ?q=boavista -> [{name, lat, lon}]"""
    _load_assets()
    G = _state["G"]
    q = _norm(request.args.get("q", "")).strip()
    if G is None or len(q) < 2:
        return jsonify({"places": []})
    by_name, _, normed = _street_index(G)
    hits = [(0 if normed[n].startswith(q) or any(w.startswith(q) for w in normed[n].split()) else 1, -len(by_name[n]), n)
            for n in by_name if q in normed[n]]
    hits.sort()
    out = []
    for _, _, n in hits[:8]:
        nodes = list(by_name[n])
        cy = np.mean([G.nodes[x]["y"] for x in nodes]); cx = np.mean([G.nodes[x]["x"] for x in nodes])
        best = min(nodes, key=lambda x: (G.nodes[x]["y"] - cy) ** 2 + (G.nodes[x]["x"] - cx) ** 2)
        out.append({"name": n, "lat": G.nodes[best]["y"], "lon": G.nodes[best]["x"]})
    return jsonify({"places": out})


@app.route("/api/place_at")
def place_at():
    """Name for a coordinate: the street(s) at the nearest junction. ?lat=&lon="""
    _load_assets()
    G = _state["G"]
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        return jsonify({"name": None}), 400
    node = _nearest_node(G, lat, lon)
    _, by_node, _ = _street_index(G)
    names = sorted(by_node.get(node, []))
    if len(names) >= 2:
        label = " & ".join(names[:2])
    elif names:
        label = names[0]
    else:
        label = None
        # unnamed junction: use the closest named one within ~250 m
        if _state.get("named_xy") is None:
            ids = list(by_node)
            _state["named_xy"] = (ids, np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in ids]))
        ids, xy = _state["named_xy"]
        if len(ids):
            dlat = (xy[:, 0] - lat) * 111320.0
            dlon = (xy[:, 1] - lon) * 111320.0 * np.cos(np.radians(lat))
            d2 = dlat * dlat + dlon * dlon
            j = int(np.argmin(d2))
            if d2[j] <= 250.0 ** 2:
                near = sorted(by_node[ids[j]])
                label = "Near " + " & ".join(near[:2])
    return jsonify({"name": label})


def _jaccard_nodes(a, b):
    ea = set(zip(a[:-1], a[1:]))
    eb = set(zip(b[:-1], b[1:]))
    return len(ea & eb) / len(ea | eb) if (ea | eb) else 0.0


@app.route("/api/trips")
def trips():
    """A small, fixed set of real test trips the user can replay."""
    _load_assets()
    data, G = _state["data"], _state["G"]
    if data is None or G is None:
        return jsonify({"trips": []})
    import random
    rnd = random.Random(7)
    idx = [i for i, t in enumerate(data["test"]) if len(t["edges"]) >= 15]
    rnd.shuffle(idx)
    out = []
    for i in idx[:12]:
        t = data["test"][i]
        st = _route_stats(G, t["nodes"])
        out.append({"index": i, "distance_m": st["distance_m"], "time_s": st["time_s"],
                    "hour": t.get("hour"), "weekday": t.get("weekday"),
                    "from": _coords(G, t["nodes"][:1])[0], "to": _coords(G, t["nodes"][-1:])[0]})
    out.sort(key=lambda x: x["distance_m"])
    return jsonify({"trips": out})



# ── Explainability: exact per-feature decomposition of a route's reward ──
_FEATURE_TEXT = {
    # feature: (label, unit scale shown to the user, unit, good-if-more?)
    "length":           ("distance",                    100.0, "m"),
    "travel_time":      ("free-flow travel time",       60.0,  "s"),
    "step":             ("number of road segments",     1.0,   ""),
    "road_motorway":    ("motorway/trunk segments",     1.0,   ""),
    "road_primary":     ("primary/secondary roads",     1.0,   ""),
    "turn_slight":      ("slight turns",                1.0,   ""),
    "turn_sharp":       ("sharp turns / U-turns",       1.0,   ""),
    "num_lanes":        ("lane count",                  2.0,   ""),
    "same_road_name":   ("continuing on the same road", 1.0,   ""),
    "away_from_goal":   ("steps moving away from the destination", 1.0, ""),
    "edge_familiarity": ("use of roads taxis travel often", 1.0, ""),
}


def _route_edges(G, nodes):
    edges = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        k = min(G[u][v], key=lambda kk: G[u][v][kk].get("length", 1e9))
        edges.append((u, v, k))
    return edges


def _explain_route(mdp, G, nodes, w):
    """Total reward R = w.phi and each feature's contribution w_i * phi_i."""
    try:
        f = mdp.traj_features({"edges": _route_edges(G, nodes), "destination": nodes[-1]})
    except Exception:
        return None
    contrib = f * w
    return {"total": float(contrib.sum()),
            "features": {n: {"count": float(f[i]), "contrib": float(contrib[i])}
                         for i, n in enumerate(FEATURES)}}


def _why(exp, ref, ref_name):
    """Plain-language reasons why `exp` scores higher/lower than `ref` under the learned reward."""
    diff = []
    for n in FEATURES:
        d = exp["features"][n]["contrib"] - ref["features"][n]["contrib"]
        diff.append((abs(d), d, n))
    diff.sort(reverse=True)
    lines = []
    for _, d, n in diff[:4]:
        if abs(d) < 0.05:
            continue
        label, scale, unit = _FEATURE_TEXT[n]
        a, b = exp["features"][n]["count"] * scale, ref["features"][n]["count"] * scale
        fmt = (lambda x: f"{x:.0f}{unit}") if unit else (lambda x: f"{x:.1f}" if n == "edge_familiarity" else f"{x:.0f}")
        lines.append({"feature": n, "delta": round(d, 2), "text":
                      f"{label}: {fmt(a)} vs {fmt(b)} on the {ref_name} route"})
    return {"vs": ref_name, "delta_total": round(exp["total"] - ref["total"], 2), "reasons": lines}


@app.route("/api/routes", methods=["POST"])
def routes():
    """All route options for one trip. Body: {trip_index} | {origin_ll, dest_ll} | {} (random)."""
    _load_assets()
    G, irl, data = _state["G"], _state["irl"], _state["data"]
    if G is None or irl is None:
        return jsonify({"error": "Model not ready"}), 503
    body = request.get_json(silent=True) or {}
    expert = None
    if body.get("trip_index") is not None and data:
        t = data["test"][int(body["trip_index"])]
        origin, dest, expert = t["origin"], t["destination"], t["nodes"]
    elif body.get("origin_ll") and body.get("dest_ll"):
        origin = _nearest_node(G, *body["origin_ll"])
        dest   = _nearest_node(G, *body["dest_ll"])
    else:
        import random
        pool = [t for t in (data["test"] if data else []) if len(t["edges"]) >= 12]
        if pool:
            t = random.choice(pool)
            origin, dest, expert = t["origin"], t["destination"], t["nodes"]
        else:
            ids = list(G.nodes()); origin, dest = random.sample(ids, 2)
    if origin == dest:
        return jsonify({"error": "Origin and destination snap to the same junction"}), 400

    if _state.get("mdp") is None:
        from mdp import RouteMDP
        _state["mdp"] = RouteMDP(G)
    from baselines import dijkstra_shortest, dijkstra_fastest
    paths = {
        "irl":      lambda: _state["mdp"].most_likely_route(origin, dest, _route_weights(irl)),
        "shortest": lambda: dijkstra_shortest(G, origin, dest),
        "fastest":  lambda: dijkstra_fastest(G, origin, dest),
    }
    if _state["clf"]:
        paths["myopic"] = lambda: _state["clf"].predict_route(G, origin, dest)
    out = {}
    for name, fn in paths.items():
        try:
            nodes = fn()
        except Exception:
            nodes = []
        if len(nodes) < 2 or nodes[-1] != dest:
            continue
        st = _route_stats(G, nodes)
        if expert:
            st["overlap_pct"] = round(100 * _jaccard_nodes(nodes, expert))
        st["coords"] = _coords(G, nodes)
        st["sig"] = hash(tuple(nodes)) & 0xffffffff
        st["_nodes"] = nodes
        out[name] = st
    # explanations: reward breakdown of every route; each route compared with "shortest"
    w_exp = _route_weights(irl)
    expl = {k: _explain_route(_state["mdp"], G, v["_nodes"], w_exp) for k, v in out.items()}
    for k, v in out.items():
        e = expl[k]
        if e is not None:
            v["explain"] = e
            if k != "shortest" and expl.get("shortest") is not None and v["sig"] != out["shortest"]["sig"]:
                v["why"] = _why(e, expl["shortest"], "shortest")
        del v["_nodes"]
    resp = {
        "origin": {"lat": G.nodes[origin]["y"], "lon": G.nodes[origin]["x"]},
        "destination": {"lat": G.nodes[dest]["y"], "lon": G.nodes[dest]["x"]},
        "routes": out,
        "driver": None,
    }
    if expert:
        st = _route_stats(G, expert)
        st["coords"] = _coords(G, expert)
        try:
            st["explain"] = _explain_route(_state["mdp"], G, expert, w_exp)
        except Exception:
            pass
        resp["driver"] = st
    return jsonify(resp)


# ──────────────────────────────────────────────────────────────
# Live weather along a route (Open-Meteo forecast API, no key needed)
# ──────────────────────────────────────────────────────────────
import time as _time

_WX_CACHE = {}          # key -> (fetched_at, payload)
_WX_TTL = 300           # seconds
_WX_CODES = {
    0: ("Clear", "☀️"), 1: ("Mostly clear", "🌤️"), 2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"), 45: ("Fog", "🌫️"), 48: ("Freezing fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy drizzle", "🌧️"),
    61: ("Light rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"), 67: ("Freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "🌨️"),
    77: ("Snow grains", "🌨️"), 80: ("Light showers", "🌦️"), 81: ("Showers", "🌧️"),
    82: ("Heavy showers", "⛈️"), 85: ("Snow showers", "🌨️"), 86: ("Snow showers", "🌨️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm, hail", "⛈️"), 99: ("Thunderstorm, hail", "⛈️"),
}


def _wx_desc(code):
    return _WX_CODES.get(int(code), ("Unknown", "❓"))


def _sample_points(coords, n=4):
    """n points spread evenly by index along the route (start and end included)."""
    if len(coords) <= n:
        return [list(c) for c in coords]
    idx = [round(i * (len(coords) - 1) / (n - 1)) for i in range(n)]
    return [list(coords[i]) for i in idx]


def _fetch_weather(points):
    import requests
    key = tuple((round(a, 2), round(b, 2)) for a, b in points)
    hit = _WX_CACHE.get(key)
    if hit and _time.time() - hit[0] < _WX_TTL:
        return hit[1]
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=8, params={
        "latitude": ",".join(f"{p[0]:.4f}" for p in points),
        "longitude": ",".join(f"{p[1]:.4f}" for p in points),
        "current": "temperature_2m,precipitation,weather_code,wind_speed_10m,is_day",
        "hourly": "temperature_2m,precipitation,precipitation_probability,weather_code,wind_speed_10m",
        "forecast_days": 2, "timezone": "GMT", "timeformat": "unixtime", "wind_speed_unit": "kmh",
    })
    r.raise_for_status()
    data = r.json()
    data = data if isinstance(data, list) else [data]
    _WX_CACHE[key] = (_time.time(), data)
    return data


@app.route("/api/route_weather", methods=["POST"])
def route_weather():
    """Live weather at points along a route. Body: {coords:[[lat,lon],...], time_s: ETA seconds}.

    For each sample point: conditions now, and the hourly forecast at the moment the
    traveller is expected to be there (departure now + share of the trip time).
    """
    body = request.get_json(silent=True) or {}
    coords = body.get("coords") or []
    if len(coords) < 2:
        return jsonify({"error": "coords required"}), 400
    total = float(body.get("time_s") or 0)
    pts = _sample_points(coords)
    try:
        data = _fetch_weather(pts)
    except Exception as e:
        return jsonify({"error": f"Weather service unavailable: {e}"}), 502
    now = _time.time()
    out = []
    for i, (pt, d) in enumerate(zip(pts, data)):
        frac = i / max(1, len(pts) - 1)
        eta = now + frac * total
        cur = d.get("current", {})
        h = d.get("hourly", {})
        times = h.get("time", [])
        j = min(range(len(times)), key=lambda k: abs(times[k] - eta)) if times else None
        at = None
        if j is not None:
            at = {k: h[k][j] for k in ("temperature_2m", "precipitation", "precipitation_probability",
                                       "weather_code", "wind_speed_10m")}
            at["label"], at["icon"] = _wx_desc(at["weather_code"])
        c = {"temp_c": cur.get("temperature_2m"), "precip_mm": cur.get("precipitation"),
             "wind_kmh": cur.get("wind_speed_10m"), "is_day": cur.get("is_day"),
             "code": cur.get("weather_code", 0)}
        c["label"], c["icon"] = _wx_desc(c["code"])
        out.append({"lat": pt[0], "lon": pt[1], "frac": round(frac, 2), "now": c, "at_arrival": at})
    # worst conditions expected along the way
    def sev(p):
        a = p["at_arrival"] or {}
        return (a.get("precipitation") or 0, a.get("weather_code") or 0)
    worst = max(out, key=sev)
    wet = (sev(worst)[0] >= 0.2) or (sev(worst)[1] >= 51)
    return jsonify({
        "points": out,
        "alert": (f"{(worst['at_arrival'] or {}).get('label', 'Rain')} expected near "
                  f"{int(worst['frac'] * 100)}% of the way: roads may be wet") if wet else None,
        "updated": int(now),
        "source": "Open-Meteo",
    })


# ── Map tiles: proxied + cached so OSM sees an identified app (tile usage policy) ──
_TILE_DIR = os.path.join(BASE_DIR, "cache", "tiles")


@app.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def tile(z, x, y):
    from flask import send_file, Response
    import requests
    if not (0 <= z <= 19):
        return Response(status=404)
    path = os.path.join(_TILE_DIR, str(z), str(x), f"{y}.png")
    if not os.path.exists(path):
        try:
            r = requests.get(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", timeout=8, headers={
                "User-Agent": "PortoRoutesIRL/1.0 (academic route-preference research; local dashboard)",
                "Referer": "http://localhost:5000/"})
        except Exception:
            return Response(status=502)
        if r.status_code != 200:
            return Response(status=r.status_code)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(r.content)
    resp = send_file(path, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/api/run_pipeline", methods=["POST"])
def run_pipeline():
    if _state["training"]:
        return jsonify({"message": "Already training", "started": False})

    # Reset cached weights so pipeline re-trains
    if os.path.exists(WEIGHTS_SAVE_PATH):
        os.remove(WEIGHTS_SAVE_PATH)
    if SYNTHETIC and os.path.exists(TRAJ_SAVE_PATH):
        os.remove(TRAJ_SAVE_PATH)          # real (Porto) trips are never deleted
    _state["irl"]     = None
    _state["data"]    = None
    _state["results"] = None
    _state["log"]     = []

    t = threading.Thread(target=_run_pipeline_bg, daemon=True)
    t.start()
    return jsonify({"message": "Pipeline started", "started": True})


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[API] Initialising …")
    _load_assets()
    print("[API] Dashboard: http://localhost:5000   (open this, not the .html file)")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
