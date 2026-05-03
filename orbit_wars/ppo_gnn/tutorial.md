# GNN-PPO Agent for Orbit Wars — Tutorial

## Why a GNN?

Orbit Wars is fundamentally about relationships between planets: who owns what, how far apart they are, which paths cross the sun. A standard MLP flattens all this into a vector and loses the structure. A Graph Neural Network (GNN) treats each planet as a node and each planet-pair as an edge, letting the model reason about these relationships directly.

The key advantage: a GNN is **permutation-invariant** (planet ordering doesn't matter) and handles **variable planet counts** (20-40 in real Kaggle games) without padding to a fixed size.

## Architecture Overview

```
Planet observations       Edge features (distance, sun, angle)
       |                              |
  Node Encoder (MLP)            Edge Encoder (MLP)
       |                              |
       +--------- GAT Layer 1 --------+
       |                              |
       +--------- GAT Layer 2 --------+
       |                              |
  Node Embeddings              Global Context (mean pool)
       |                              |
       +--- Source Head (which planet to launch from, or noop)
       |
       +--- Target Head (where to send fleet, conditioned on source)
       |
       +--- Fraction Head (how many ships: 25/50/75/100%)
       |
       +--- Value Head (how good is this state?)
```

## The Factored Action Distribution

This is the non-obvious part. Instead of a flat distribution over all possible (source, target, fraction) combinations (which grows as N^2 * 4 — that's 3,136 for 28 planets), we factor the action into three sequential decisions:

### Step 1: Pick source planet (or noop)
- One logit per planet + one noop logit
- Masked to planets you own with ships > 0
- Softmax → sample or argmax

### Step 2: Pick target planet (given source)
- For the chosen source, compute a score for every other planet
- Score depends on: source embedding, target embedding, edge features between them, global context
- Self-loops masked (can't send to yourself)
- Optionally: sun-blocked paths masked
- Softmax → sample or argmax

### Step 3: Pick fraction (given source and target)
- Choose from {25%, 50%, 75%, 100%} of ships on source planet
- Depends on source embedding, target embedding, global context

### Why factor?

**Credit assignment.** If the agent sends 50 ships from planet 3 to planet 7 and it goes badly, the flat model has to figure out which part was wrong — was planet 3 a bad source? Planet 7 a bad target? 50% too many ships? The factored model gets separate gradients for each decision.

**Exploration efficiency.** The flat model has to explore a 3136-dim space. The factored model explores three smaller spaces (28 + 28 + 4) that compose.

### Joint log-probability

For PPO's ratio computation, we need log p(action). The factored decomposition gives:

```
log p(action) = log p(source) + log p(target | source) + log p(fraction | source, target)
```

For noop, only log p(source=noop) applies (target and fraction terms are 0).

## Sun Avoidance

The sun at (50, 50) with radius 10 destroys fleets that cross it. We handle this two ways:

1. **Edge feature** (always on): For every planet pair, we compute:
   - `sun_intersects`: binary — does the straight-line path cross within radius + safety margin?
   - `sun_clearance`: continuous — how close does the path get to the sun? (0 = through center, 1 = far away)

   These are part of the 6-dim edge features the GNN uses during message passing and target scoring.

2. **Hard mask** (optional, `--mask-sun` flag): Targets where the path crosses the sun are excluded from the target distribution entirely. This prevents the agent from ever selecting sun-blocked paths, but removes the ability to learn that sometimes a longer route around the sun is worth taking.

## Three-Phase Training Pipeline

### Phase 1: Behavioral Cloning (BC)

**What:** Imitate winning players from Kaggle replays.

**Data:** 346 2P + 280 4P replay files → ~600K+ state-action pairs. Filtered to rank-1 (winning) players only.

**Loss:** Sum of three cross-entropies:
```
L = CE(source_logits, actual_source) + CE(target_logits, actual_target) + CE(fraction_logits, actual_fraction)
```

**Run:**
```bash
# Parse replays and train 2P model
python -m ppo_gnn.train_bc --mode 2p --replay-dir kaggle_replays --epochs 50

# Train 4P model
python -m ppo_gnn.train_bc --mode 4p --replay-dir kaggle_replays --epochs 50
```

First run parses all replays and caches to `ppo_gnn/cache/transitions_{2p,4p}.pt`. Subsequent runs load from cache.

**Output:** `ppo_gnn/cache/checkpoint_bc_{2p,4p}.pt`

### Phase 2: Offline RL (Advantage-Weighted Regression)

**What:** Improve beyond imitation by learning from outcomes.

**Key insight:** We use ALL transitions (winners AND losers). The value head learns to predict game outcomes, and the advantage `A = return - V(s)` tells us which actions were better/worse than expected. Winning players' actions get upweighted; losing players' actions get downweighted.

**This is the counterfactual step.** The agent learns: "in this board state, the loser attacked planet X and lost. The winner defended planet Y and won. I should defend Y."

**Temperature annealing:** Controls how aggressively the policy deviates from replay behavior. Starts at 1.0 (mild), ends at 0.1 (aggressive).

**Run:**
```bash
python -m ppo_gnn.train_offline_rl --mode 2p --bc-checkpoint ppo_gnn/cache/checkpoint_bc_2p.pt
python -m ppo_gnn.train_offline_rl --mode 4p --bc-checkpoint ppo_gnn/cache/checkpoint_bc_4p.pt
```

**Output:** `ppo_gnn/cache/checkpoint_awr_{2p,4p}.pt`

### Phase 3: Online PPO Fine-Tuning

**What:** Polish the agent against live opponents.

**Why needed:** Offline RL can only learn from states in the replay data. Online PPO explores new states and adapts to specific opponents.

**Setup:** Requires a live environment. The training loop is implemented in `train_ppo.py` — you need to provide an env wrapper that implements `get_observation(player)` and `step(player, action)`.

```bash
python -m ppo_gnn.train_ppo --mode 2p --checkpoint ppo_gnn/cache/checkpoint_awr_2p.pt
```

## Using the Trained Agent

To use in a Kaggle submission, convert the model's output back to the expected action format:

```python
from ppo_gnn.gnn_policy import OrbitWarsGNNPolicy, FRACTION_BUCKETS
from ppo_gnn.replay_parser import _build_node_features
import math, torch

model = OrbitWarsGNNPolicy(hidden_dim=64)
model.load_state_dict(torch.load("ppo_gnn/cache/checkpoint_awr_2p.pt", weights_only=True))
model.eval()

def agent(obs, config):
    planets = obs["planets"]
    fleets = obs.get("fleets", [])
    player = obs["player"]
    num_players = 2  # or detect from context

    nf, pos, owned = _build_node_features(planets, fleets, player, num_players)
    src, tgt, frac, is_noop, _, _ = model.sample_action(nf, pos, owned, deterministic=True)

    if is_noop:
        return []

    src_planet = planets[src]
    tgt_planet = planets[tgt]
    angle = math.atan2(
        float(tgt_planet[3]) - float(src_planet[3]),
        float(tgt_planet[2]) - float(src_planet[2]),
    )
    ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])
    return [[int(src_planet[0]), angle, ships]]
```

## GAT vs GraphSAGE

Both are available behind a flag (`--use-sage`). GAT uses multi-head attention to learn which planet relationships matter most in each state. GraphSAGE uses a simpler mean-aggregation with learned edge gates.

In practice, GAT typically performs better for heterogeneous graphs (mixed owned/enemy/neutral planets with varying importance), but GraphSAGE trains faster and is more stable early on.

**Recommendation:** Start with GAT (default). If training is unstable, try GraphSAGE.

## Parameter Budget

| Component | Params (dim=64) | Params (dim=128) |
|-----------|-----------------|------------------|
| Node encoder | ~8.5k | ~33k |
| Edge encoder | ~1.2k | ~4.4k |
| GAT layers (x2) | ~33k | ~132k |
| Global context | ~4.2k | ~16.5k |
| Action heads | ~27k | ~107k |
| Value head | ~8.4k | ~33k |
| **Total** | **~83k** | **~326k** |

Both are well under the 500k budget. Use dim=64 to start; increase to 128 if the model underfits.

## Evaluating

Run the trained agent in the existing tournament framework:

1. Create a submission file that wraps the model (see "Using the Trained Agent" above)
2. Drop it into `submission/` alongside your heuristic agents
3. Run tournaments via `scripts/` to compare
