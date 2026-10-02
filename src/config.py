"""
config.py – Central configuration for the IRL Route Preference Learning project
Environment: San Francisco, CA
"""

import os

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for _d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# San Francisco OSM bounding box  (inner Tenderloin / SOMA –
# dense enough to be interesting, small enough to be tractable)
# ──────────────────────────────────────────────────────────────
SF_PLACE = "San Francisco, California, USA"

# Sub-region bounding box: roughly Mission / SOMA / Castro
SF_BBOX = {
    "north": 37.7850,
    "south": 37.7550,
    "east":  -122.3900,
    "west":  -122.4400,
}

GRAPH_SAVE_PATH   = os.path.join(DATA_DIR, "sf_drive_graph.graphml")
TRAJ_SAVE_PATH    = os.path.join(DATA_DIR, "synthetic_trajectories.pkl")
WEIGHTS_SAVE_PATH = os.path.join(MODELS_DIR, "irl_weights.npy")

# ──────────────────────────────────────────────────────────────
# MDP / Feature settings
# ──────────────────────────────────────────────────────────────
FEATURES = [
    "length",          # segment length (m)
    "speed_kph",       # OSM maxspeed or inferred (km/h)
    "travel_time",     # length / speed  (seconds)
    "road_motorway",   # one-hot: motorway / trunk
    "road_primary",    # one-hot: primary / secondary
    "road_residential",# one-hot: residential / living_street / unclassified
    "turn_straight",   # 0/1 – roughly straight (< 30°)
    "turn_slight",     # 0/1 – slight turn (30–90°)
    "turn_sharp",      # 0/1 – sharp / U-turn  (> 90°)
    "num_lanes",       # number of lanes (default 1)
]

N_FEATURES = len(FEATURES)

# ──────────────────────────────────────────────────────────────
# Synthetic trajectory generation
# ──────────────────────────────────────────────────────────────
N_TRAJECTORIES     = 300   # total synthetic expert trajectories
TRAIN_FRAC         = 0.80
RANDOM_SEED        = 42

# "Ground-truth" driver preference weights used to simulate demonstrations.
# They act on the *scaled* features (length/100 m, speed/50 km/h, time/60 s,
# lanes/2).  Order matches FEATURES above.
TRUE_WEIGHTS = [
    -1.5,   # length          – penalise distance (per 100 m)
    +0.1,   # speed_kph       – mild preference for faster roads
    -0.5,   # travel_time     – penalise slow segments
     0.0,   # road_motorway   – reference class (no penalty)
    -0.1,   # road_primary    – arterials almost as good
    -0.6,   # road_residential – avoid residential streets
     0.0,   # turn_straight   – reference turn type
    -0.3,   # turn_slight     – small penalty for turning
    -1.0,   # turn_sharp      – avoid sharp turns / U-turns
    +0.05,  # num_lanes       – mild preference for wider roads
]
# NOTE: rewards are kept (almost) non-positive so that the soft-optimal policy
# has no reward-positive cycles; otherwise a MaxEnt driver would loop forever.

# ──────────────────────────────────────────────────────────────
# MaxEnt IRL training
# ──────────────────────────────────────────────────────────────
IRL_N_ITERATIONS     = 80     # L-BFGS iterations
IRL_DISCOUNT         = 0.99
SOFT_VI_ITERATIONS   = 200   # soft value-iteration steps per update
REGULARIZATION_L2    = 1e-3
