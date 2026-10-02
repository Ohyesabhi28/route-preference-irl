# IRL Route Preference Learning — San Francisco

**22AIE401 · Reinforcement Learning | Amrita Vishwa Vidyapeetham, Amritapuri**

| Name | Roll No |
|------|---------|
| Abhinav Nair | AM.SC.U4AIE23005 |
| Advaith K | AM.SC.U4AIE23010 |
| Akash A | AM.SC.U4AIE23011 |

---

## Overview

This project implements **Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)** for route preference learning on the **San Francisco OpenStreetMap road network**. The system learns latent driver preferences from historical GPS trajectories and uses them to generate human-aligned routes.

```
GPS Trajectories (Expert Demos)
         ↓
  Map-Matched Edge Sequences
         ↓
  MaxEnt IRL  (learn w* s.t. w*ᵀφ explains expert behaviour)
         ↓
  Soft Value Iteration  →  Stochastic Policy π_w
         ↓
  Route Generation  +  Evaluation vs Dijkstra baselines
```

---

## Project Structure

```
new/
├── src/
│   ├── config.py               # All hyperparameters & paths
│   ├── graph_builder.py        # OSM graph download & feature engineering
│   ├── trajectory_generator.py # Synthetic expert trajectory simulation
│   ├── mdp.py                  # MDP + soft value iteration + MaxEnt policy
│   ├── maxent_irl.py           # MaxEnt IRL gradient ascent trainer
│   ├── baselines.py            # Dijkstra (shortest/fastest) + Myopic classifier
│   ├── evaluation.py           # Jaccard, F1, path-ratio, top-k, log-likelihood
│   ├── train.py                # End-to-end training pipeline
│   └── api.py                  # Flask REST API
├── dashboard/
│   └── index.html              # Interactive web dashboard
├── data/                       # Auto-created: graph, trajectories
├── models/                     # Auto-created: learned weights
├── results/                    # Auto-created: evaluation JSON
├── run_pipeline.py             # Step 1: train
├── run_api.py                  # Step 2: serve API
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the training pipeline
```bash
python run_pipeline.py
```

This will:
- Download the SF drive graph from OpenStreetMap (via OSMnx)
- Generate 300 synthetic expert trajectories using ground-truth preference weights
- Train MaxEnt IRL for 150 gradient-ascent iterations
- Train the myopic logistic-regression baseline
- Evaluate all methods and save results to `results/`

> **Note**: First run downloads the OSM graph (~30 s). Subsequent runs load from cache.

### 3. Start the API server
```bash
python run_api.py
```

### 4. Open the dashboard
Open `dashboard/index.html` in your browser (or serve it with any static file server).

---

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** `s` | `(current_node, incoming_edge)` — avoids non-Markovian turn ambiguity |
| **Action** `a` | Choice of outgoing road segment at current intersection |
| **Reward** `R(s,a)` | `wᵀ · φ(s,a)` — linear in 10 edge features |
| **Policy** `π` | `π(a\|s) ∝ exp(Q_soft(s,a))` — MaxEnt stochastic policy |
| **Objective** | Maximise log-likelihood of expert trajectories |

### Feature Vector φ (10 dimensions)

| # | Feature | Description |
|---|---------|-------------|
| 0 | `length` | Segment length (m) |
| 1 | `speed_kph` | Speed limit (km/h) |
| 2 | `travel_time` | Estimated travel time (s) |
| 3 | `road_motorway` | Motorway/trunk indicator |
| 4 | `road_primary` | Primary/secondary road indicator |
| 5 | `road_residential` | Residential/service road indicator |
| 6 | `turn_straight` | Near-straight (< 30°) |
| 7 | `turn_slight` | Slight turn (30–90°) |
| 8 | `turn_sharp` | Sharp/U-turn (≥ 90°) |
| 9 | `num_lanes` | Number of lanes |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Edge Jaccard** | IoU of edge sets between predicted and expert route |
| **Edge F1** | Precision + recall of edge overlap |
| **Path Length Ratio** | `len(predicted) / len(expert)` — closer to 1.0 is better |
| **Top-1 Accuracy** | Expert edge is the model's highest-scored choice at each junction |
| **Top-3 Accuracy** | Expert edge is in the model's top-3 choices |
| **Held-out Log-Likelihood** | `log π(τ\|w)` on unseen trajectories |

---

## Baselines

| Method | Description |
|--------|-------------|
| **Dijkstra-Shortest** | Minimises total edge length only |
| **Dijkstra-Fastest** | Minimises estimated travel time |
| **Myopic Classifier** | Logistic regression on edge features — black-box, no reward interpretation |
| **MaxEnt IRL** | Learns interpretable reward weights from expert demonstrations |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Pipeline status, training log |
| GET | `/api/graph/stats` | Node/edge counts, sample nodes |
| GET | `/api/weights` | Learned + true feature weights |
| GET | `/api/history` | IRL training loss/gradient history |
| GET | `/api/evaluation` | Full evaluation results table |
| GET | `/api/trajectories` | Sample test trajectories (lat/lon) |
| POST | `/api/route` | Generate route `{method, origin, destination}` |
| POST | `/api/run_pipeline` | Trigger full training pipeline |

---

## References

1. Ziebart et al. (2008). *Maximum Entropy Inverse Reinforcement Learning*. AAAI.
2. Liu et al. (2020). *Integrating Dijkstra into Deep IRL for Food Delivery*. Transp. Res. Part E.
3. Zhao & Liang (2023). *Deep IRL for Route Choice Modeling*. Transp. Res. Part C.
4. Barnes et al. (2024). *Massively Scalable IRL in Google Maps*. ICLR.
5. Zhang et al. (2024). *Recursive Logit-Based Meta-IRL for Route Planning*. Transp. Res. Part E.
6. Pitombeira-Neto et al. (2024). *Trajectory Modeling via Random Utility IRL*. Information Sciences.
