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

# Porto (real taxi GPS, ECML/PKDD 2015) -- central Porto
PORTO_BBOX = {
    "north": 41.1850,
    "south": 41.1400,
    "east":  -8.5800,
    "west":  -8.6500,
}
PORTO_RAW_CSV      = os.path.join(DATA_DIR, "porto", "train.csv")
PORTO_GRAPH_PATH   = os.path.join(DATA_DIR, "porto_drive_graph.graphml")
PORTO_TRAJ_PATH    = os.path.join(DATA_DIR, "porto_trajectories.pkl")

GRAPH_SAVE_PATH   = os.path.join(DATA_DIR, "sf_drive_graph.graphml")
TRAJ_SAVE_PATH    = os.path.join(DATA_DIR, "synthetic_trajectories.pkl")
WEIGHTS_SAVE_PATH = os.path.join(MODELS_DIR, "irl_weights.npy")

# ──────────────────────────────────────────────────────────────
# MDP / Feature settings
# ──────────────────────────────────────────────────────────────
FEATURES_V1 = [
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

# v2 (IRL_FEATURES=v2): identifiable parameterisation.
#  - road class and turn one-hots each sum to 1, so with an explicit per-edge
#    `step` constant one reference level of each group is dropped
#    (residential and straight are the references);
#  - speed_kph dropped (length / travel_time already encode it);
#  - same_road_name: next edge continues the same named road;
#  - away_from_goal: destination-dependent, 1 if the step increases the
#    straight-line distance to the destination.  (A signed progress term
#    would telescope to a constant per trip and carry no information.)
FEATURES_V2 = [
    "length", "travel_time", "step",
    "road_motorway", "road_primary",
    "turn_slight", "turn_sharp",
    "num_lanes", "same_road_name", "away_from_goal",
]
DEST_FEATURES_V2 = ["away_from_goal"]

# v3 = v2 + data-driven edge familiarity: log(1 + #trips that traversed the
# edge) in an INDEPENDENT pool of map-matched taxi trips (disjoint from
# train/val/test; see build_familiarity.py), scaled to [0, 1].
FEATURES_V3 = FEATURES_V2 + ["edge_familiarity"]
FAMILIARITY_PATH = os.environ.get("IRL_FAM_PATH",
                                  os.path.join(DATA_DIR, "porto_edge_counts.pkl"))

FEATURE_SET = os.environ.get("IRL_FEATURES", "v1").lower()
if FEATURE_SET == "v3":
    FEATURES = FEATURES_V3
    DEST_FEATURES = DEST_FEATURES_V2
elif FEATURE_SET == "v2":
    FEATURES = FEATURES_V2
    DEST_FEATURES = DEST_FEATURES_V2
else:
    FEATURE_SET = "v1"
    FEATURES = FEATURES_V1
    DEST_FEATURES = []

# Feature-group ablation: IRL_DROP="turn_slight,turn_sharp" removes those columns
# (used by run_ablation.py).  Results/weights are written to a separate tag.
ABLATION = [f for f in os.environ.get("IRL_DROP", "").split(",") if f]
if ABLATION:
    _unknown = [f for f in ABLATION if f not in FEATURES]
    assert not _unknown, f"IRL_DROP has unknown features: {_unknown}"
    FEATURES = [f for f in FEATURES if f not in ABLATION]
    DEST_FEATURES = [f for f in DEST_FEATURES if f not in ABLATION]
ABLATION_TAG = os.environ.get("IRL_ABLATION_TAG", "")

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
REGULARIZATION_L2    = 1e-3
# Episodic task: gamma = 1, terminal at the destination, soft value iteration
# runs for a finite horizon (RouteMDP.soft_values_levels, MaxEntIRL horizon=80).
# (The former IRL_DISCOUNT / SOFT_VI_ITERATIONS constants were never read.)
MAX_HORIZON          = 80


# ──────────────────────────────────────────────────────────────
# Region switch:  IRL_REGION=porto (default, real taxi GPS)  |  sf (synthetic demo)
# ──────────────────────────────────────────────────────────────
REGION = os.environ.get("IRL_REGION", "porto").lower()
if REGION == "porto":
    REGION_NAME   = "Porto"
    REGION_BBOX   = PORTO_BBOX
    SYNTHETIC     = False
    WEIGHTS_SAVE_PATH = os.path.join(MODELS_DIR, "irl_weights_porto.npy")
    RESULTS_DIR   = os.path.join(BASE_DIR, "results", "porto")
    FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
else:
    REGION        = "sf"
    REGION_NAME   = "San Francisco"
    REGION_BBOX   = SF_BBOX
    SYNTHETIC     = True

if FEATURE_SET != "v1":
    WEIGHTS_SAVE_PATH = WEIGHTS_SAVE_PATH.replace(".npy", f"_{FEATURE_SET}.npy")
    RESULTS_DIR = os.path.join(RESULTS_DIR, FEATURE_SET)
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

if ABLATION:
    _tag = ABLATION_TAG or "drop_" + "_".join(ABLATION)
    WEIGHTS_SAVE_PATH = WEIGHTS_SAVE_PATH.replace(".npy", f"_abl_{_tag}.npy")
    RESULTS_DIR = os.path.join(RESULTS_DIR, f"ablate_{_tag}")
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for _d in [RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(_d, exist_ok=True)
