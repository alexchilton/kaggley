# Orbit Wars Agent

Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition agent.

## Architecture

A single-file decision engine (`orbit_wars_agent.py`) that each turn:

1. **Builds a projected world** — tracks every in-flight fleet, predicts arrivals, and simulates per-planet timelines with full combat resolution
2. **Generates candidate missions** — defense, snipe, comet capture, expand (neutrals), attack (hostiles), multi-source swarm, crash exploit, reinforce, evacuate
3. **Scores and ranks globally** — each mission gets a value/cost score incorporating production profit, static planet bonus, race-time advantage, game phase, and opponent priority
4. **Commits greedily with lookahead** — picks the best mission considering what follow-up missions it enables or blocks

### Key subsystems

| Component | Purpose |
|-----------|---------|
| `PositionPredictor` | Predicts future positions for orbiting planets and comets |
| `InterceptSolver` | Solves edge-to-edge intercepts against moving targets |
| `ProjectedWorld` | Simulates planet timelines under planned + visible fleet arrivals |
| `SurplusCalculator` | Determines safely sendable ships while preserving planet ownership |
| `ForwardSimulator` | Approximates future ship totals for phase heuristics |
| `DecisionLogic` | Main engine: builds missions, scores, commits |

## Version History

### v16 (current) — Shun 2p Quality Filter

Analysed replays from top player "Shun" to understand their 2-player opening edge. Key findings:

1. **Less opening spam** — Shun doesn't fire every turn early; the opening is selective
2. **Neutral-first** — early fleets go to neutral captures, not hostile fights
3. **Static > orbiting** — bias toward non-orbiting planets (easier to secure and value)
4. **High-production first** — early targets skew toward prod 3-5, not nearest cheap planets
5. **Enough ships, not random pressure** — confident captures, not speculative probes
6. **Accuracy over tempo** — fewer bad openings matters more than speed

**What changed:** Kept our aggression level (no launch caps for 2p) but applied Shun's target quality filter:

- 2p early expand sort now quality-aware: static first, safe neutrals preferred, high production prioritised, then distance/efficiency
- Early attacks and snipes suppressed in 2p when 3+ good neutrals (prod >= 2) remain
- Orbiting neutrals require prod >= 3 during early phase (all player counts)
- Confidence scoring boosted for high-prod static targets (+0.22), penalises low-prod orbiting more (-0.16)

### v15 — Shun-style opening (4p focused)

Added Shun opening mode for 4+ player games: launch cap of 2 per turn, no early hostile attacks, orbiting neutrals require prod >= 3, aggressive quality sorting. Improved 4p play but 2p regressed.

### v8 — Best 2p version pre-Shun

Strong 2p performance (15-1 vs v5), decent 4p (15/32 top finishes). Established the baseline target for 2p quality.

### v5 — Baseline architecture

Unified mission system, projected world with full timeline simulation, opening confidence filter. Foundation for all subsequent versions.

## Benchmark Results (overnight suite, 2p + 4p)

### 2-player (vs baseline_agent and v5)

| Version | vs baseline | vs v5 | Notes |
|---------|-------------|-------|-------|
| v8 | 3-13 | 15-1 | Best 2p pre-Shun |
| v12 | 6-10 | 6-4-6D | Opening filter tuning |
| v13 | 6-10 | 9-5-2D | Confidence + margin tuning |
| v6 | 4-12 | 12-4 | First post-v5 iteration |
| v15 | 3-11 | 3-11 | Shun 4p focus, 2p regressed |

### 4-player (top finishes out of 32 games)

| Version | Rank 1 | Rate |
|---------|--------|------|
| v8 | 15/32 | 47% |
| v12 | 14/32 | 44% |
| v13 | 14/32 | 44% |
| v6 | 13/32 | 41% |
| v11 | 13/32 | 41% |

Baseline agent remains significantly stronger in 2p — closing this gap is the primary goal.

## Files

- `orbit_wars_agent.py` — the competition agent (also copied to `submission/main.py` for Kaggle)
- `snapshots/` — frozen versions for benchmarking (v5 through v15)
- `test_agent.py` / `test_orbit_wars_agent.py` — unit tests
- `run_overnight_benchmark.py` — automated benchmark runner
- `shun/` — replay files from top player analysis (local only)
- `possible_todo.md` — future improvement ideas

## Running

```python
# The agent entry point
from orbit_wars_agent import agent
actions = agent(obs, config)
```

Each action is `[planet_id, angle, num_ships]` — launch `num_ships` from `planet_id` at `angle` radians.
