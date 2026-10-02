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
    WEIGHTS_SAVE_PATH, RESULTS_DIR, TRAJ_SAVE_PATH,
    FEATURES, TRUE_WEIGHTS, SF_BBOX
)

app = Flask(__name__)
CORS(app)

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
    from graph_builder import build_graph

    if _state["G"] is None:
        _log("Loading graph …")
        _state["G"] = build_graph()

    if _state["data"] is None and os.path.exists(TRAJ_SAVE_PATH):
        from trajectory_generator import load_trajectories
        _state["data"] = load_trajectories()

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
    if _state["results"] is None and os.path.exists(results_path):
        with open(results_path) as f:
            _state["results"] = json.load(f)


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
        "bbox":      SF_BBOX,
        "sample_nodes": nodes_out,
    })


@app.route("/api/weights")
def weights():
    _load_assets()
    irl = _state["irl"]
    if irl is None:
        return jsonify({"error": "IRL not trained"}), 503

    true_w = np.array(TRUE_WEIGHTS, dtype=np.float64)
    true_w = true_w / (np.linalg.norm(true_w) + 1e-9)

    return jsonify({
        "features":     FEATURES,
        "learned":      irl.weights.tolist(),
        "true":         true_w.tolist(),
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
            traj   = random.choice(_state["data"]["test"])
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
            from mdp import RouteMDP
            mdp   = RouteMDP(G)
            nodes = mdp.sample_route(origin, dest, irl.weights)
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


@app.route("/api/run_pipeline", methods=["POST"])
def run_pipeline():
    if _state["training"]:
        return jsonify({"message": "Already training", "started": False})

    # Reset cached weights so pipeline re-trains
    if os.path.exists(WEIGHTS_SAVE_PATH):
        os.remove(WEIGHTS_SAVE_PATH)
    if os.path.exists(TRAJ_SAVE_PATH):
        os.remove(TRAJ_SAVE_PATH)
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
    print("[API] Starting Flask server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
