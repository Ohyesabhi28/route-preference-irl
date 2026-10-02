# Innovation plan — what we add beyond "MaxEnt IRL on a road graph"

## Honest positioning
Plain MaxEnt IRL for route choice is not new (Ziebart 2008). Three papers in our own
literature table already cover close ideas:

| Idea | Already done by | Their method |
|---|---|---|
| Context-dependent rewards | Zhao & Liang 2023 | deep IRL |
| Personalised / meta preferences | Zhang et al. 2024, Liu et al. 2025 | deep / meta-IRL |
| Large-scale road IRL | Barnes et al. 2024 (Google) | proprietary, huge |

So we should **not** claim "first to do context/personalised IRL". Our defensible angle:

> **An exact, fully interpretable linear-reward MaxEnt IRL on open data (Porto) that
> reports *what* drivers trade off and *how that changes with context*, with a
> reproducible pipeline** — not another black-box deep model.

## Extensions (ranked by value ÷ effort)

### E1 — Familiarity feature  (core, ~½ day)
Add `log(1 + n_train_trips_using_edge)` as a reward feature: drivers favour roads they
know. Computed from **training trips only** (no leakage).
- Test: with vs without the feature (held-out log-lik, edge F1, top-1).
- Risk: popularity partly encodes "main road"; report weights of both.

### E2 — Context-dependent preferences  (core, ~1 day)
`w(c) = w0 + Δ_peak·1[peak] + Δ_weekend·1[weekend] + Δ_night·1[night]` (L2 shrinkage on Δ).
Porto gives timestamp, so contexts are free. The gradient is the same feature-count
difference, computed per context group — our exact trainer extends directly.
- Test: held-out log-lik vs the single global `w`; likelihood-ratio test; plot which
  weights move (e.g. peak → larger time penalty, fewer turns).
- Risk: small effect sizes; use bootstrap CIs so we report only what is significant.

### E3 — Value-of-time analysis  (core, ~½ day, mostly free)
`VOT = w_time / w_length` → "metres of detour a driver accepts to save one minute",
overall, per context, with bootstrap CIs. This is the headline *interpretable* result a
neural policy cannot give, and ties the model to transport economics.

### E4 — Driver-type clusters  (stretch, ~1–1.5 days)
Latent-class MaxEnt IRL, K = 2–3 reward vectors fitted by EM over taxi IDs
(3,000 trips / 433 taxis ≈ 7 trips each, so per-driver `w` is too noisy — cluster instead).
- Test: held-out log-lik per K (pick K by validation), cluster sizes, weight profiles.
- Risk: clusters may just encode shift/area. Only keep if validation log-lik improves.

### E5 — Data-driven travel times  (stretch, ~1 day)
Replace OSM speed limits with per-edge, per-hour speeds estimated from the GPS traces
(training split). A congestion-aware `travel_time` feature.
- Risk: sparse coverage on small streets; fall back to the OSM value.

## What we commit to
**E1 + E2 + E3** are the core contribution. E4/E5 only if time remains. Frame as:
"interpretable contextual MaxEnt IRL with familiarity and value-of-time analysis".

## Evaluation protocol (needed to make any claim credible)
- Split by **origin–destination pair** (no OD in both train and test) and report a
  split by **taxi** as a robustness check.
- 3 random seeds / 200-sample bootstrap → mean ± CI for every table cell.
- Baselines unchanged: Dijkstra shortest/fastest, myopic classifier. Add
  **popularity-weighted Dijkstra** (cost = length / familiarity) — a strong
  non-learned baseline, so a win over it actually means something.
- Keep the synthetic run as the correctness check (cosine 0.987 weight recovery).

## Engineering needed first
1. Porto map-matched dataset (running: 56 % match, route/GPS length ratio median 1.09).
2. Scale the trainer: 3,000 trips × 6.5k-edge graph is ~10× the SF cost. Fix by
   multiprocessing over destinations and a smaller horizon (H≈60); else subsample to ~1,000.
3. Context labels per trip (already stored: hour, weekday, call type).

## Report changes
- §3 / §11: GeoLife/Beijing → Porto taxi dataset (ECML/PKDD 2015), map-matching step.
- §13 Expected Novelty: replace with the positioning above.
- Fig. 1 labels ("GeoLife", "Beijing") → "Porto taxi GPS".
- Add the ablation tables from E1–E3.
