"""
visualise.py – Generate static matplotlib figures for the report.

Figures produced:
  1. Training convergence (loss + grad_norm vs iteration)
  2. Feature weight comparison (true vs learned bar chart)
  3. Route comparison map (IRL vs Dijkstra on a subgraph)
  4. Evaluation metrics radar chart
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RESULTS_DIR, FIGURES_DIR, FEATURES, TRUE_WEIGHTS,
    WEIGHTS_SAVE_PATH
)

# Styling
plt.rcParams.update({
    "figure.facecolor":  "#060b18",
    "axes.facecolor":    "#0a0e1a",
    "axes.edgecolor":    "#1a3050",
    "axes.labelcolor":   "#7aa8cc",
    "xtick.color":       "#4a7090",
    "ytick.color":       "#4a7090",
    "text.color":        "#e8f4ff",
    "grid.color":        "#0f1e35",
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
    "figure.dpi":        150,
})

CYAN   = "#00d4ff"
VIOLET = "#9b59f5"
AMBER  = "#f5c542"
RED    = "#ff4d6d"
GREEN  = "#39ff7e"


# ── 1. Convergence plot ──────────────────────────────────────
def plot_convergence(history: list, save: bool = True):
    iters     = [h[0] for h in history]
    losses    = [h[1] for h in history]
    gradnorms = [h[2] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                   sharex=True, tight_layout=True)
    fig.suptitle("MaxEnt IRL Training Convergence", color="#e8f4ff",
                 fontsize=14, fontweight="bold")

    ax1.plot(iters, losses, color=CYAN, lw=2, label="Feature-count MSE")
    ax1.fill_between(iters, losses, alpha=0.15, color=CYAN)
    ax1.set_ylabel("Loss (‖μ_expert − μ_π‖²)")
    ax1.legend(framealpha=0.2, facecolor="#060b18")
    ax1.grid(True)

    ax2.plot(iters, gradnorms, color=VIOLET, lw=2, label="Gradient Norm")
    ax2.fill_between(iters, gradnorms, alpha=0.15, color=VIOLET)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("‖∇L‖")
    ax2.legend(framealpha=0.2, facecolor="#060b18")
    ax2.grid(True)

    if save:
        path = os.path.join(FIGURES_DIR, "convergence.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── 2. Feature weights bar chart ─────────────────────────────
def plot_weights(learned: np.ndarray, true_w: np.ndarray,
                 save: bool = True):
    labels = [f.replace("road_", "").replace("turn_", "") for f in FEATURES]
    x      = np.arange(len(labels))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    fig.suptitle("Learned vs Ground-Truth Feature Weights",
                 color="#e8f4ff", fontsize=14, fontweight="bold")

    bars1 = ax.bar(x - width/2, learned, width, color=CYAN,
                   alpha=0.85, label="Learned (IRL)", zorder=3)
    bars2 = ax.bar(x + width/2, true_w,  width, color=VIOLET,
                   alpha=0.65, label="Ground Truth", zorder=3)

    ax.axhline(0, color="#1a3050", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Weight Value")
    ax.legend(framealpha=0.2, facecolor="#060b18")
    ax.grid(True, axis="y", zorder=0)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=7, color=CYAN)

    if save:
        path = os.path.join(FIGURES_DIR, "feature_weights.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── 3. Evaluation metrics bar chart ──────────────────────────
def plot_evaluation(results: dict, save: bool = True):
    metrics = ["Edge Jaccard", "Edge F1", "Top-1 Accuracy", "Top-3 Accuracy"]
    methods = list(results.keys())
    colors  = [CYAN, AMBER, RED, VIOLET]

    x     = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    fig.suptitle("Evaluation Metrics Comparison",
                 color="#e8f4ff", fontsize=14, fontweight="bold")

    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [results[method].get(m, 0) or 0 for m in metrics]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, color=color,
               alpha=0.85, label=method, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(framealpha=0.2, facecolor="#060b18", fontsize=9,
              loc="upper right")
    ax.grid(True, axis="y", zorder=0)

    if save:
        path = os.path.join(FIGURES_DIR, "evaluation_metrics.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── 4. Route visualisation on subgraph ───────────────────────
def plot_routes(G: nx.MultiDiGraph,
                irl_route:      list[int],
                shortest_route: list[int],
                fastest_route:  list[int],
                expert_route:   list[int],
                save: bool = True):
    """Draw all routes on a subgraph layout."""
    # Union of all nodes
    all_nodes = set(irl_route + shortest_route +
                    fastest_route + expert_route)
    subG = G.subgraph(all_nodes)

    fig, ax = plt.subplots(figsize=(10, 9), tight_layout=True)
    ax.set_facecolor("#060b18")
    fig.patch.set_facecolor("#060b18")
    ax.set_title("Route Comparison on SF Road Network",
                 color="#e8f4ff", fontsize=14, fontweight="bold")

    # Position: use lat/lon
    pos = {n: (subG.nodes[n].get("x", 0),
                subG.nodes[n].get("y", 0)) for n in subG.nodes()}

    # Draw background edges
    nx.draw_networkx_edges(subG, pos, ax=ax,
                           edge_color="#0f1e35", width=0.5, alpha=0.5,
                           arrows=False)

    def _draw_path(route, color, label, lw=2, style="-", alpha=0.9):
        if len(route) < 2:
            return
        edges = [(route[i], route[i+1]) for i in range(len(route)-1)
                 if subG.has_edge(route[i], route[i+1])]
        if not edges:
            return
        nx.draw_networkx_edges(subG, pos, edgelist=edges, ax=ax,
                               edge_color=color, width=lw,
                               alpha=alpha, arrows=False,
                               style=style)

    _draw_path(expert_route,   GREEN,  "Expert",           lw=3, style="dashed")
    _draw_path(shortest_route, AMBER,  "Dijkstra Shortest", lw=2)
    _draw_path(fastest_route,  RED,    "Dijkstra Fastest",  lw=2)
    _draw_path(irl_route,      CYAN,   "IRL (MaxEnt)",      lw=4)

    # Markers
    if irl_route:
        o = irl_route[0];  d = irl_route[-1]
        ax.scatter([pos[o][0]], [pos[o][1]], s=120, c="#00d4ff",
                   zorder=10, edgecolors="white", linewidths=1.5, label="Origin")
        ax.scatter([pos[d][0]], [pos[d][1]], s=120, c="#ff4d6d",
                   zorder=10, edgecolors="white", linewidths=1.5, label="Destination")

    patches = [
        mpatches.Patch(color=CYAN,  label="IRL (MaxEnt)"),
        mpatches.Patch(color=AMBER, label="Dijkstra Shortest"),
        mpatches.Patch(color=RED,   label="Dijkstra Fastest"),
        mpatches.Patch(color=GREEN, label="Expert Trajectory"),
    ]
    ax.legend(handles=patches, framealpha=0.25,
              facecolor="#060b18", fontsize=9, loc="lower left")
    ax.axis("off")

    if save:
        path = os.path.join(FIGURES_DIR, "route_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Main ─────────────────────────────────────────────────────
def generate_all_figures():
    import json

    # Load training history
    hist_path = WEIGHTS_SAVE_PATH.replace(".npy", "_history.pkl")
    if os.path.exists(hist_path):
        with open(hist_path, "rb") as f:
            hist = pickle.load(f)
        plot_convergence(hist)
    else:
        print("No training history found — skip convergence plot.")

    # Load weights
    if os.path.exists(WEIGHTS_SAVE_PATH):
        learned = np.load(WEIGHTS_SAVE_PATH)
        true_w  = np.array(TRUE_WEIGHTS)
        plot_weights(learned, true_w)
    else:
        print("No weights found — skip weights plot.")

    # Load evaluation
    res_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)
        plot_evaluation(results)
    else:
        print("No evaluation results found — skip metrics plot.")

    # Route visualisation
    from graph_builder import build_graph
    from trajectory_generator import load_trajectories
    from mdp import RouteMDP
    from baselines import dijkstra_shortest, dijkstra_fastest
    import random

    G    = build_graph()
    data = load_trajectories()
    irl  = np.load(WEIGHTS_SAVE_PATH)
    mdp  = RouteMDP(G)

    traj = random.choice(data["test"])
    o, d = traj["origin"], traj["destination"]

    irl_r  = mdp.sample_route(o, d, irl)
    dks_r  = dijkstra_shortest(G, o, d)
    dkf_r  = dijkstra_fastest(G, o, d)
    exp_r  = traj["nodes"]
    plot_routes(G, irl_r, dks_r, dkf_r, exp_r)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
