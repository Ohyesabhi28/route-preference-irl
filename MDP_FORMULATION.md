# MDP Formulation – Route Preference IRL (Porto taxi GPS)

Structured after the course reference (`MDP-pdf.pdf`: Markov Process → MRP → MDP,
tuple **(S, A, P, R, γ)**, Markov property, policy, V/Q, Bellman equations,
value iteration, policy extraction). Implementation: `src/mdp.py` (`RouteMDP`).

---

## 1. The problem in one paragraph

A taxi driver moves through the Porto road network from an origin to a destination.
At every intersection they choose which road to take next. We observe many such
routes (map-matched GPS) but **not** the reward the driver is optimising.
The MDP is the model of the *driving decision process*; MaxEnt IRL then
recovers the reward weights **w** that make the observed routes (soft-)optimal.

Reference-deck mapping: this is a **fully observable, deterministic-transition,
goal-terminated MDP** whose *reward is unknown* (that is the IRL part).

---

## 2. The MDP tuple

### 2.1 State space S — "the road you just arrived on"

```
s_t = e_{t-1} = (u → v, k)      a directed edge of the OSM graph
```

Goal-conditioned: the destination `d` is a fixed parameter of each episode, so
formally `S_aug = {(e, d)}`; in code one value function is computed per `d`.

**Why this state (deck: "smallest representation that retains all
decision-relevant information")**

| Candidate | Verdict | Reason |
|---|---|---|
| Node only | Too small | Turn angle depends on the *incoming* bearing; a node cannot tell you whether the next move is straight or a U-turn → reward is not a function of state. Information loss. |
| **Incoming edge** | **Chosen** | Fixes the node (head of edge) *and* the heading. Next options and turn cost are fully determined → Markov. |
| Full path history | Too large | Not needed for any reward term; state space explodes. |

Size: |S| = E ≈ number of directed edges in the central-Porto graph
(≈ 4k-order in the SF graph; same code path).

**Markov check:** `P(e_{t+1} | e_t, a_t, e_{t-1}, …) = P(e_{t+1} | e_t, a_t)` — the
available roads and the turn feature depend only on `e_t`.

**Initial state:** no incoming edge exists at the origin, so the first action
chooses the start edge directly from `out_edges(origin)` with reward
`φ_edge(e_0)` (`start_logits`). Equivalent to a dummy start state.

**Terminal state:** any edge whose head node is the destination. Absorbing,
`V = 0` (the deck's terminal states ±2, here reward-free).

### 2.2 Action space A(s) — "which outgoing road next"

```
A(e) = { e' : tail(e') = head(e) }       state-dependent, ~2–4 actions
```

- Includes the reverse edge (U-turn) when it exists in the graph; it is
  penalised through `turn_sharp`, not forbidden.
- Dead ends with no out-edge are not given a value (`has_out` mask, `NEG`).

### 2.3 Transition model P(s' | s, a) — deterministic

```
P(s' = a | s, a) = 1
```

Unlike the 3×4 grid world in the deck (0.6 / 0.2 / 0.2 slip), choosing a road
*is* arriving on it. All randomness lives in the **policy** (driver variability),
not in the environment. So the Bellman sum over s' collapses to a single term.

### 2.4 Reward R(s, a) — linear, *unknown*, learned

```
R(s, a) = wᵀ φ(s, a)        φ(s, a) = [ road features of chosen edge ‖ turn features of (e → e') ]
```

| Group | Features (`config.FEATURES`) | Scaling |
|---|---|---|
| Distance / time | `length`, `speed_kph`, `travel_time` | /100 m, /50 kph, /60 s |
| Road class | `road_motorway`, `road_primary`, `road_residential` | one-hot |
| Turn (depends on s *and* a) | `turn_straight` <30°, `turn_slight` 30–90°, `turn_sharp` >90° | one-hot, from bearings |
| Capacity | `num_lanes` | /2 |

- Road features depend only on `a`; turn features depend on the pair `(s, a)` —
  this is exactly why the state must carry the heading.
- Per-step cost structure ("−0.5 per move" in the deck's grid) is learned here:
  negative `length` / `travel_time` weights make the agent want short, fast routes.
- Ground truth (`TRUE_WEIGHTS`) exists only for the synthetic SF demo; for Porto
  the weights are recovered purely from data.

### 2.5 Discount γ and horizon

- Conceptually this is a **stochastic-shortest-path / episodic** task with
  γ = 1: episodes end at the destination, so no discounting is needed.
- ⚠ `config.IRL_DISCOUNT = 0.99` and `SOFT_VI_ITERATIONS` are **not used** by
  `mdp.py` (grep confirms no reader). The effective setting is γ = 1 with a
  finite horizon `H = 80` edges (`soft_values_levels`), which keeps values
  finite even if a learned cycle has non-negative reward (speed / lanes terms
  are positive). Either wire γ in or delete the dead constant.
- Demonstrations are filtered to 5–75 edges (`porto_data.build_dataset`),
  consistent with `H = 80`.

---

## 3. Policy, value functions, Bellman equations

Deck notation → this project:

| Deck | Standard MDP | Ours (MaxEnt / soft) |
|---|---|---|
| Policy | π(a\|s) or π(s) | `π(a\|s) = exp(Q(s,a) − V(s))` (Boltzmann over road choices) |
| Action value | `Q(s,a) = R + γ Σ P V(s')` | `Q(e,e') = wᵀφ(e,e') + V(e')` (P deterministic) |
| State value | `V(s) = max_a Q(s,a)` | `V(e) = log Σ_{e'} exp Q(e,e')` (soft-max) |
| Terminal | V = 0 | `V(e)=0` if head(e) = destination |
| Solve | Value iteration | `soft_values` (backward, per destination); `soft_values_levels` (H-level) |
| Extract policy | `π*(s)=argmax_a …` | `most_likely_route` (Viterbi/Dijkstra on `V(s)−Q(s,a)`) |

**Why soft-max instead of max.** Real drivers are noisy-optimal. The
maximum-entropy principle (Ziebart 2008) gives the least-committal policy
matching expert feature counts; `max` is recovered as the temperature → 0 and
is what `most_likely_route` uses for a single predicted path.

**Trajectory probability** (what IRL maximises):

```
P(ξ | w, d) = ∏_t π(a_t | s_t) = exp( wᵀ f(ξ) ) / Z_d(o)
f(ξ) = Σ_t φ(s_t, a_t)           (feature counts of a route)
```

---

## 4. How MDP and IRL fit together (deck: *Prediction vs Control*)

```
 outer loop  (L-BFGS on w)
   └─ inner loop per destination d
        1. backward pass  – soft value iteration  → V_d(e)     [DP "control"]
        2. forward  pass  – state-visitation mass → E_π[f]     [DP "prediction"]
        3. gradient = E_π[f] − f_expert  (+ L2)
```

Reward is **given** in the deck's MDP; here it is the **unknown** and the MDP
(S, A, P, γ) is the fixed known model. IRL = search over R such that the
planner's behaviour matches the demonstrations.

---

## 5. Assumptions and limitations (be explicit in the report)

1. **Single reward for all drivers/trips** – no time-of-day, no per-driver
   heterogeneity (a contextual / mixture reward would extend S or R).
2. **Fully observable** – GPS is map-matched to edges; matching errors are
   treated as ground truth (deck: fully vs partially observable).
3. **Destination known** to the agent from the start (goal-conditioned).
4. **Deterministic dynamics** – no traffic / signal stochasticity; congestion
   is not in the state. Adding a time-bucket or congestion level to S would be
   the natural extension.
5. **Linear reward** over 10 hand-built features.
6. **Finite horizon H = 80** truncation (γ = 1), see §2.5.
7. Tail wording for slides: *reward weights are statistically identifiable only
   up to the information in the chosen features.*

---

## 6. Formal summary (slide-ready)

> **M = ⟨S, A, P, R, γ⟩** with
> S = directed road edges (arrival edge) × fixed destination;
> A(e) = outgoing edges at head(e);
> P(e' | e, a=e') = 1;
> R(e, e') = wᵀ[φ_road(e'), φ_turn(e, e')], w unknown;
> γ = 1, terminal when head(e) = destination, horizon H = 80.
> Policy π_w(a|s) ∝ exp Q_w(s,a), obtained by soft value iteration;
> w learned by maximum-entropy IRL from map-matched Porto taxi routes.

---

## 7. Worked example (diagram: `mdp_diagram.svg`, also on slide 6 of `presentation.html`)

A toy road MDP with the same structure as the real one, using the learned v3 weights
(`length −0.24, travel_time +0.16, step −0.67, motorway +0.23, primary −0.11,
turn_slight −0.52, turn_sharp −1.46, lanes −0.01, same_road −0.04, away_from_goal −0.29,
familiarity +0.87`). Style follows the course sample MDPs: circles = states, diamonds = actions,
double circles = terminals (green = positive, red = negative). Edge features are illustrative;
the topology is a valid road graph (an edge only follows an edge whose head is its tail).

**States** — S1 start edge; S2 arterial edge and S3 side-street edge (both start at S1's junction);
S4 reverse edge (after a U-turn); S5 arterial edge reached from S3's junction;
terminals **GOAL** (head(e) = d, V = 0) and **DEAD END** (no out-edge, V = −∞).

**GOAL bonus** — entering GOAL adds +5 so the positive terminal looks like the course examples.
Every trip ends at GOAL exactly once, so this shifts every route's total reward by the same
constant: the policy π is unchanged (the real model has no bonus).

| State | Action | Meaning | Next | P | R | π |
|---|---|---|---|---|---|---|
| S1 | A1 | straight onto arterial | S2 | 1 | −0.40 | 0.68 |
| S1 | A2 | slight turn, side street | S3 | 1 | −1.10 | 0.31 |
| S1 | A3 | U-turn | S4 | 1 | −2.34 | 0.01 |
| S2 | B1 | straight to destination | GOAL | 1 | −0.38 + 5 = +4.62 | 1.00 |
| S2 | B2 | enter dead-end stub | DEAD END | 1 | −1.53 | 0 (never) |
| S3 | C1 | straight, long and quiet | GOAL | 1 | −0.96 + 5 = +4.04 | 0.62 |
| S3 | C2 | slight turn onto arterial | S5 | 1 | −0.93 | 0.38 |
| S5 | E1 | straight to destination | GOAL | 1 | −0.51 + 5 = +4.49 | 1.00 |
| S4 | D1 | U-turn again | S1 | 1 | −2.04 | 1.00 |

Soft values (γ = 1, `V(e) = log Σ exp[R + V(e′)]`): V(S1) = 4.61, V(S2) = 4.62,
V(S3) = 4.52, V(S4) = 2.57, V(S5) = 4.49, V(GOAL) = 0.

Reading it: the destination is the best ending (the "+" terminal); the dead end is never chosen
(the "−" terminal); the sharp-turn U-turn is nearly never taken (π = 0.01) because of the −1.46
turn weight. All transitions are deterministic (P = 1), unlike the stochastic sample MDPs — the
randomness is in π. Regenerate with `python make_mdp_diagram.py`.
