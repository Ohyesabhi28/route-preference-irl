"""
build_familiarity.py – edge-traversal counts from an INDEPENDENT pool of taxi
trips (candidates not in the train/val/test dataset), used for the
`edge_familiarity` feature (IRL_FEATURES=v3).

Independence matters: counting the training trips themselves would make
"drivers use familiar edges" true by construction and would not transfer
to unseen trips.

    python build_familiarity.py --n 3000
"""
import os
import sys
import pickle
import argparse
import collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import porto_data as P
from config import FAMILIARITY_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    used = {t["trip_id"] for t in P.load_porto_trajectories()["all"]}
    cands = [c for c in P.load_candidates() if c["trip_id"] not in used]
    print(f"[fam] {len(cands)} candidates outside the dataset")
    trips = P.match_trips(cands, args.n)
    trips = [t for t in trips if t["trip_id"] not in used]
    counts = collections.Counter()
    for t in trips:
        for e in set(map(tuple, t["edges"])):          # count each trip once
            counts[e] += 1
    with open(FAMILIARITY_PATH, "wb") as f:
        pickle.dump({"counts": dict(counts), "n_trips": len(trips)}, f)
    print(f"[fam] {len(trips)} trips, {len(counts)} distinct edges, "
          f"max count {max(counts.values())} -> {FAMILIARITY_PATH}")


if __name__ == "__main__":
    main()
