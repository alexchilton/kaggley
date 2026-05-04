"""Phase 3: Online PPO fine-tuning against heuristic opponents.

Initializes from Phase 2 (AWR) checkpoint and fine-tunes with standard PPO
using the factored action distribution. Uses kaggle_environments for live play
against a rotating pool of opponents.

The factored log-prob is used for the PPO ratio:
    ratio = exp(new_log_prob - old_log_prob)
where log_prob = log p(source) + log p(target|source) + log p(fraction|source,target)

Usage:
    python -m ppo_gnn.train_ppo --mode 2p --checkpoint ppo_gnn/cache/checkpoint_awr_2p.pt
    python -m ppo_gnn.train_ppo --mode 4p --checkpoint ppo_gnn/cache/checkpoint_awr_4p.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from .gnn_policy import FRACTION_BUCKETS, OrbitWarsGNNPolicy
from .replay_parser import _build_node_features
from .sun_geometry import sun_intersects_path

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RolloutStep:
    node_features: torch.Tensor   # (N, 10)
    positions: torch.Tensor       # (N, 2)
    owned_mask: torch.Tensor      # (N,)
    source: int
    target: int
    fraction: int
    is_noop: bool
    log_prob: float
    value: float
    reward: float
    done: bool


# ---------------------------------------------------------------------------
# Opponent loading
# ---------------------------------------------------------------------------

def load_agent_from_file(path: str) -> Callable:
    """Load an agent(obs, config) callable from a .py file."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Agent file not found: {p}")
    module_name = f"_agent_{p.stem}_{id(p)}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"No top-level agent() function in {p}")
    return mod.agent


def build_opponent_pool(mode: str) -> list[tuple[str, Callable | str]]:
    """Build the pool of heuristic opponents.

    All heuristics are strong — curriculum controls the random:heuristic mix ratio.
    """
    root = Path(__file__).parent.parent
    heuristics: list[tuple[str, Callable | str]] = []

    candidates = [
        ("shun_combined", "submission/main_s2_shun_combined.py"),
        ("rc_v3", "submission/main_release_candidate_v3_antidogpile_position.py"),
        ("rc_v2", "submission/main_release_candidate_v2.py"),
    ]

    if mode == "4p":
        candidates.extend([
            ("4p_antidogpile", "submission/main_s2_4p_antidogpile.py"),
            ("4p_earlyaggro", "submission/main_s2_4p_earlyaggro.py"),
        ])

    for name, rel_path in candidates:
        full_path = root / rel_path
        if full_path.exists():
            try:
                agent_fn = load_agent_from_file(str(full_path))
                heuristics.append((name, agent_fn))
            except Exception as e:
                print(f"  Warning: failed to load {name}: {e}")

    return heuristics


def make_self_play_agent(
    model: OrbitWarsGNNPolicy,
    num_players: int,
    noop_penalty: float = 4.0,
    device: torch.device = torch.device("cpu"),
) -> Callable:
    """Create a frozen copy of the model wrapped as a kaggle agent function.

    The copy's weights are snapshotted at call time — the training model
    can keep updating without affecting the opponent.
    """
    shadow = OrbitWarsGNNPolicy(
        hidden_dim=model.hidden_dim,
        use_gat=model.use_gat,
    )
    shadow.load_state_dict(model.state_dict())
    shadow.to(device)
    shadow.eval()

    def _agent(obs, config):
        planets = obs.get("planets", [])
        if not planets:
            return []
        player = obs.get("player", 0)
        fleets = obs.get("fleets", [])

        has_planets = any(int(p[1]) == player for p in planets)
        if not has_planets:
            return []

        nf, pos, owned = _build_node_features(planets, fleets, player, num_players)
        nf = nf.to(device)
        pos = pos.to(device)
        owned = owned.to(device)

        with torch.no_grad():
            nf_b = nf.unsqueeze(0)
            pos_b = pos.unsqueeze(0)
            om_b = owned.unsqueeze(0)
            N = nf_b.shape[1]

            source_logits, all_target_logits, all_fraction_logits = shadow(
                nf_b, pos_b, om_b,
            )
            source_logits[0, N] -= noop_penalty

            src = torch.distributions.Categorical(logits=source_logits.squeeze(0)).sample().item()
            if src == N:
                return []

            tgt = torch.distributions.Categorical(logits=all_target_logits[0, src]).sample().item()
            frac = torch.distributions.Categorical(logits=all_fraction_logits[0, src, tgt]).sample().item()

        src_planet = planets[src]
        tgt_planet = planets[tgt]
        sx, sy = float(src_planet[2]), float(src_planet[3])
        tx, ty = float(tgt_planet[2]), float(tgt_planet[3])
        if sun_intersects_path(sx, sy, tx, ty):
            return []
        angle = math.atan2(ty - sy, tx - sx)
        ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])
        if ships < 1:
            return []
        return [[int(src_planet[0]), angle, ships]]

    return _agent


# Curriculum: win-rate thresholds and opponent mix fractions.
# (random_frac, self_play_frac, heuristic_frac) must sum to 1.0.
# As win rate improves, we advance to harder stages.
# min_games: require this many heuristic games before considering advancement.
# Higher thresholds + larger sample = no premature promotion.
CURRICULUM_STAGES = [
    # Stage 0: mostly self-play — learn fundamentals before facing heuristics
    {"name": "warmup", "advance_wr": 0.25, "min_games": 50, "random": 0.00, "self_play": 0.85, "heuristic": 0.15},
    # Stage 1: introduce more heuristic exposure once agent can win ~25% vs heuristics
    {"name": "developing", "advance_wr": 0.35, "min_games": 50, "random": 0.00, "self_play": 0.65, "heuristic": 0.35},
    # Stage 2: balanced mix
    {"name": "intermediate", "advance_wr": 0.45, "min_games": 50, "random": 0.00, "self_play": 0.45, "heuristic": 0.55},
    # Stage 3: heuristic-dominant — terminal stage
    {"name": "advanced", "advance_wr": 1.01, "min_games": 50, "random": 0.00, "self_play": 0.30, "heuristic": 0.70},
]

# How often to refresh the self-play shadow weights (episodes)
SELF_PLAY_REFRESH_INTERVAL = 20
# How often to refresh the "2000 episodes ago" lagging reference (episodes)
LAGGING_REFRESH_INTERVAL = 2000

# Performance-gated horizon schedule: (heuristic_wr_threshold, max_steps)
# Horizon only advances when the agent proves it can beat heuristics at
# the current horizon. No time pressure — it stays until it's ready.
HORIZON_STAGES = [
    (0.0, 50),     # start: 50 steps
    (0.75, 100),   # need 75% heur wr to unlock 100 steps
    (0.75, 200),   # need 75% heur wr at 100 steps to unlock 200
    (0.75, 500),   # need 75% heur wr at 200 steps to unlock full game
]


def pick_curriculum_opponent(
    heuristics: list[tuple[str, Callable | str]],
    self_play_agent: Callable | None,
    stage: int,
) -> tuple[str, Callable | str]:
    """Pick an opponent based on the current curriculum stage."""
    cfg = CURRICULUM_STAGES[min(stage, len(CURRICULUM_STAGES) - 1)]
    r = random.random()
    if r < cfg["random"]:
        return ("random", "random")
    if r < cfg["random"] + cfg["self_play"] and self_play_agent is not None:
        return ("self_play", self_play_agent)
    if heuristics:
        return random.choice(heuristics)
    # fallback
    if self_play_agent is not None:
        return ("self_play", self_play_agent)
    return ("random", "random")


# ---------------------------------------------------------------------------
# GAE and PPO update (unchanged from Phase 3 template)
# ---------------------------------------------------------------------------

def compute_gae(
    rollout: List[RolloutStep],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns from a rollout."""
    T = len(rollout)
    rewards = torch.tensor([s.reward for s in rollout])
    values = torch.tensor([s.value for s in rollout])
    dones = torch.tensor([1.0 if s.done else 0.0 for s in rollout])

    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: OrbitWarsGNNPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: List[RolloutStep],
    max_planets: int = 48,
    epochs: int = 4,
    mini_batch_size: int = 64,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.015,
    kl_coef: float = 0.1,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Run PPO update on a collected rollout."""
    advantages, returns = compute_gae(rollout, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = len(rollout)
    M = max_planets
    nf_all = torch.zeros(T, M, 10)
    pos_all = torch.zeros(T, M, 2)
    owned_all = torch.zeros(T, M)
    sources = torch.zeros(T, dtype=torch.long)
    targets = torch.zeros(T, dtype=torch.long)
    fractions = torch.zeros(T, dtype=torch.long)
    is_noops = torch.zeros(T)
    old_log_probs = torch.zeros(T)

    for i, step in enumerate(rollout):
        n = step.node_features.shape[0]
        n = min(n, M)
        nf_all[i, :n] = step.node_features[:n]
        pos_all[i, :n] = step.positions[:n]
        owned_all[i, :n] = step.owned_mask[:n]
        sources[i] = step.source if not step.is_noop else M
        targets[i] = step.target if not step.is_noop else 0
        fractions[i] = step.fraction if not step.is_noop else 0
        is_noops[i] = 1.0 if step.is_noop else 0.0
        old_log_probs[i] = step.log_prob

    nf_all = nf_all.to(device)
    pos_all = pos_all.to(device)
    owned_all = owned_all.to(device)
    sources = sources.to(device)
    targets = targets.to(device)
    fractions = fractions.to(device)
    is_noops = is_noops.to(device)
    old_log_probs = old_log_probs.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "kl": 0.0}
    num_updates = 0

    for _ in range(epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, mini_batch_size):
            idx = perm[start : start + mini_batch_size]

            log_prob, entropy, value = model.evaluate_action(
                nf_all[idx], pos_all[idx], owned_all[idx],
                sources[idx], targets[idx], fractions[idx], is_noops[idx],
            )

            ratio = torch.exp(log_prob - old_log_probs[idx])
            adv = advantages[idx]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, returns[idx])
            entropy_loss = -entropy.mean()

            # KL anchor: penalise large deviations from the old policy.
            # Approximates KL(old || new) ≈ (log_prob_old - log_prob_new).mean()
            # This prevents the 4k→6k regression where the policy drifted too far.
            kl_approx = (old_log_probs[idx] - log_prob).mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss + kl_coef * kl_approx

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > clip_epsilon).float().mean().item()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += -entropy_loss.item()
            stats["clip_frac"] += clip_frac
            stats["kl"] += kl_approx.item()
            num_updates += 1

    for k in stats:
        stats[k] /= max(num_updates, 1)
    return stats


# ---------------------------------------------------------------------------
# Rollout collection using kaggle_environments
# ---------------------------------------------------------------------------

def play_episode(
    model: OrbitWarsGNNPolicy,
    opponent: Callable | str,
    mode: str = "2p",
    noop_penalty: float = 4.0,
    idle_penalty: float = 0.02,
    step_penalty: float = 0.002,
    max_steps: int = 500,
    win_bonus: float = 10.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[List[RolloutStep], float]:
    """Play one full episode against an opponent using kaggle_environments.

    Args:
        max_steps: Horizon cap. If < 500, episode is cut short and a partial
            terminal reward is assigned based on fleet+production advantage.
        win_bonus: Terminal reward for a win. Default 10.0, use higher for
            heuristic opponents to give stronger learning signal.

    Returns:
        (rollout, final_reward) where final_reward is +1 (win) or -1 (loss).
    """
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    num_players = 2 if mode == "2p" else 4

    # Set up agents: our model in slot 0, opponent(s) in remaining slots
    if mode == "2p":
        trainer = env.train([None, opponent])
    else:
        # For 4P, fill remaining slots with the opponent
        trainer = env.train([None, opponent, opponent, opponent])

    obs = trainer.reset()
    model.eval()
    rollout: List[RolloutStep] = []
    # Dense reward shaping: track fleet+production advantage each step.
    # 0.01 is the shaping scale; 10.0 is the terminal bonus (see end of function).
    prev_shaped_score = 0.0
    prev_enemy_total = 0.0  # track opponent fleet+prod for delta-v signal

    for step_idx in range(max_steps):
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        player = obs.get("player", 0)

        # Skip empty observations (step 0)
        if not planets:
            obs, reward, done, info = trainer.step([])
            if done:
                break
            continue

        # Check if we're still alive
        has_planets = any(int(p[1]) == player for p in planets)
        if not has_planets:
            obs, reward, done, info = trainer.step([])
            if done:
                break
            continue

        nf, pos, owned = _build_node_features(planets, fleets, player, num_players)
        nf = nf.to(device)
        pos = pos.to(device)
        owned = owned.to(device)

        with torch.no_grad():
            nf_b = nf.unsqueeze(0)
            pos_b = pos.unsqueeze(0)
            om_b = owned.unsqueeze(0)
            N = nf_b.shape[1]

            source_logits, all_target_logits, all_fraction_logits = model(
                nf_b, pos_b, om_b,
            )
            h, _, g = model._encode(nf_b, pos_b)
            value = model.value_head(g).squeeze(-1).item()

            # Apply noop penalty
            source_logits[0, N] -= noop_penalty

            source_dist = torch.distributions.Categorical(logits=source_logits.squeeze(0))
            src = source_dist.sample().item()
            log_p_src = source_dist.log_prob(torch.tensor(src)).item()

            if src == N:  # noop
                action = []
                is_noop = True
                tgt, frac = 0, 0
                log_prob = log_p_src
            else:
                is_noop = False
                tgt_logits = all_target_logits[0, src]
                target_dist = torch.distributions.Categorical(logits=tgt_logits)
                tgt = target_dist.sample().item()
                log_p_tgt = target_dist.log_prob(torch.tensor(tgt)).item()

                frac_logits = all_fraction_logits[0, src, tgt]
                frac_dist = torch.distributions.Categorical(logits=frac_logits)
                frac = frac_dist.sample().item()
                log_p_frac = frac_dist.log_prob(torch.tensor(frac)).item()

                log_prob = log_p_src + log_p_tgt + log_p_frac

                # Convert to Kaggle action
                src_planet = planets[src]
                tgt_planet = planets[tgt]
                sx, sy = float(src_planet[2]), float(src_planet[3])
                tx, ty = float(tgt_planet[2]), float(tgt_planet[3])

                # Sun safety check
                if sun_intersects_path(sx, sy, tx, ty):
                    action = []
                    is_noop = True
                    log_prob = log_p_src  # treat as noop
                else:
                    angle = math.atan2(ty - sy, tx - sx)
                    ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])
                    if ships < 1:
                        action = []
                        is_noop = True
                        log_prob = log_p_src
                    else:
                        action = [[int(src_planet[0]), angle, ships]]

        # Step the environment
        obs, reward, done, info = trainer.step(action)

        # Dense reward: fleet + production advantage delta with opponent delta-v.
        # Three signals: own fleet/prod growth, opponent fleet/prod change, and
        # the relative advantage delta.
        my_fleet = 0.0
        my_prod = 0.0
        enemy_fleet = 0.0
        enemy_prod = 0.0
        for p in planets:
            owner = int(p[1])
            ships = float(p[5])
            prod = float(p[6])
            if owner == player:
                my_fleet += ships
                my_prod += prod
            elif owner >= 0 and owner != player:
                enemy_fleet += ships
                enemy_prod += prod

        # Relative advantage: production weighted 3x fleet.
        # Production compounds over time, fleet is a one-time asset.
        # Without this weighting, random opponents score high just by
        # grabbing neutrals (fleet-heavy) which generates misleading loss signals.
        PROD_WEIGHT = 3.0
        score = (my_fleet + PROD_WEIGHT * my_prod) - (enemy_fleet + PROD_WEIGHT * enemy_prod)
        delta = score - prev_shaped_score
        prev_shaped_score = score

        # Delta-v: penalise opponent growth, reward opponent losses.
        # Track enemy total separately to get a pure opponent-change signal.
        enemy_total = enemy_fleet + PROD_WEIGHT * enemy_prod
        enemy_delta = enemy_total - prev_enemy_total
        prev_enemy_total = enemy_total

        # Combined dense reward:
        #   delta * 0.01  — own advantage growth (prod-weighted)
        #  -enemy_delta * 0.005 — opponent growth penalty / shrink bonus
        shaped_reward = delta * 0.01 - enemy_delta * 0.005 - step_penalty

        if is_noop and has_planets and my_fleet > 50:
            shaped_reward -= idle_penalty

        rollout.append(RolloutStep(
            node_features=nf.cpu(),
            positions=pos.cpu(),
            owned_mask=owned.cpu(),
            source=src if not is_noop else 0,
            target=tgt,
            fraction=frac,
            is_noop=is_noop,
            log_prob=log_prob,
            value=value,
            reward=shaped_reward,
            done=done,
        ))

        if done:
            break

    # Terminal reward: ±10.0 to dominate over dense shaping (which uses 0.01 scale)
    raw_reward = reward if reward is not None else 0.0
    hit_horizon = (not done) and (len(rollout) >= max_steps - 1)

    # Compute final fleet/prod snapshot (needed for horizon-cutoff metric and ep_stats)
    my_fleet_final = 0.0
    my_prod_final = 0.0
    enemy_fleet_final = 0.0
    enemy_prod_final = 0.0
    if obs and "planets" in obs:
        for p in obs["planets"]:
            owner = int(p[1])
            ships = float(p[5])
            prod = float(p[6])
            if owner == player:
                my_fleet_final += ships
                my_prod_final += prod
            elif owner >= 0 and owner != player:
                enemy_fleet_final += ships
                enemy_prod_final += prod

    if hit_horizon:
        # Differential horizon metric with time-value weighting and tanh smoothing.
        # Avoids the binary 450× problem: instead normalises by expected game-scale
        # values so proportional differences produce proportional signals.
        #
        # At step 50 (horizon cutoff), time_value ≈ 0.9, so:
        #   advantage ≈ 0.1 × fleet_diff + 1.8 × prod_diff
        # Production is weighted heavily because it compounds over remaining steps,
        # but tanh prevents saturation from dominating the gradient.
        FULL_GAME_LENGTH = 500
        step = len(rollout)
        remaining_steps = FULL_GAME_LENGTH - step
        time_value = remaining_steps / FULL_GAME_LENGTH

        # Expected game-scale at this step (normalising constants, not hard targets)
        expected_fleet = max(50 + step * 3, 1.0)   # ~200 at step 50
        expected_prod  = max(5  + step * 0.2, 1.0)  # ~15 at step 50

        fleet_diff = (my_fleet_final - enemy_fleet_final) / expected_fleet
        prod_diff  = (my_prod_final  - enemy_prod_final)  / expected_prod

        advantage = (1 - time_value) * fleet_diff + time_value * 1.5 * prod_diff
        advantage_signal = math.tanh(advantage)
        final_reward = advantage_signal * (win_bonus / 2.0)
    elif raw_reward > 0:
        final_reward = win_bonus
    elif raw_reward < 0:
        final_reward = -win_bonus
        # Survival bonus: lasting longer softens the loss penalty
        if rollout:
            survival_frac = len(rollout) / max_steps
            final_reward = final_reward * (1.0 - 0.5 * survival_frac)
    else:
        final_reward = 0.0

    if rollout and final_reward != 0:
        # Assign final reward to last step, let GAE propagate it back
        rollout[-1] = RolloutStep(
            **{**rollout[-1].__dict__, "reward": final_reward, "done": True},
        )

    ep_stats = {
        "my_fleet": my_fleet_final,
        "my_prod": my_prod_final,
        "enemy_fleet": enemy_fleet_final,
        "enemy_prod": enemy_prod_final,
    }

    return rollout, final_reward, ep_stats


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Online PPO")
    parser.add_argument("--mode", choices=["2p", "4p"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Phase 2 (AWR) checkpoint")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--num-episodes", type=int, default=1000000,
                        help="Total episodes to train on (runs until performance gates are met)")
    parser.add_argument("--episodes-per-update", type=int, default=4,
                        help="Episodes to collect before each PPO update")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--use-sage", action="store_true")
    parser.add_argument("--mask-sun", action="store_true")
    parser.add_argument("--max-planets", type=int, default=48)
    parser.add_argument("--noop-penalty", type=float, default=4.0)
    parser.add_argument("--idle-penalty", type=float, default=0.02)
    parser.add_argument("--step-penalty", type=float, default=0.002,
                        help="Per-step negative reward to encourage fast wins")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Evaluate against full pool every N episodes")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum (uniform random opponent)")
    parser.add_argument("--progressive-horizon", action="store_true",
                        help="Enable progressive horizon (50→100→200→500 steps)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Fixed max steps per episode (ignored if --progressive-horizon)")
    parser.add_argument("--device", default="cpu",
                        help="Device for rollout inference (cpu recommended)")
    parser.add_argument("--update-device", default=None,
                        help="Device for PPO gradient updates (default: same as --device, "
                             "use 'mps' to accelerate updates while keeping rollouts on cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    update_device = torch.device(args.update_device) if args.update_device else device
    cache_dir = Path(args.cache_dir)

    # Load model
    model = OrbitWarsGNNPolicy(
        hidden_dim=args.hidden_dim,
        use_gat=not args.use_sage,
        mask_sun_targets=args.mask_sun,
    )
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = str(cache_dir / f"checkpoint_ppo_{args.mode}.pt")

    # Build opponent pool
    heuristics = build_opponent_pool(args.mode)
    print(f"Heuristic opponents ({len(heuristics)}): {[name for name, _ in heuristics]}")

    use_curriculum = not args.no_curriculum
    num_players = 2 if args.mode == "2p" else 4

    # Create initial self-play agent (frozen copy of model)
    self_play_fn = make_self_play_agent(model, num_players, args.noop_penalty, device)

    # Reference opponents for tracking absolute progress:
    # 1. "baseline" — frozen copy of the starting checkpoint (never updated)
    baseline_fn = make_self_play_agent(model, num_players, args.noop_penalty, device)
    # 2. "lagging" — snapshot from ~2000 episodes ago (refreshed periodically)
    lagging_fn = make_self_play_agent(model, num_players, args.noop_penalty, device)
    baseline_results: list[float] = []   # rolling window vs baseline
    lagging_results: list[float] = []    # rolling window vs lagging

    # Training state
    total_wins = 0
    total_episodes = 0
    best_heuristic_wr = 0.0
    window_results: list[float] = []       # rolling window (all opponents)
    heuristic_results: list[float] = []    # rolling window (heuristic-only)
    curriculum_stage = 0
    horizon_stage = 0  # index into HORIZON_STAGES

    use_progressive = args.progressive_horizon

    print(f"\nStarting PPO training (runs until done, max {args.num_episodes} episodes)")
    print(f"  Mode: {args.mode}, LR: {args.lr}, Noop penalty: {args.noop_penalty}")
    print(f"  Device: rollout={device}, update={update_device}")
    if use_curriculum:
        print(f"  Curriculum: ON (stage 0 = {CURRICULUM_STAGES[0]['name']})")
    else:
        print(f"  Curriculum: OFF (uniform random from all opponents)")
    if use_progressive:
        print(f"  Progressive horizon: ON, performance-gated "
              f"({' → '.join(str(s) for _, s in HORIZON_STAGES)} steps, "
              f"advances at 75% heur wr)")
    else:
        print(f"  Max steps: {args.max_steps}")
    print()

    for ep_start in range(0, args.num_episodes, args.episodes_per_update):
        # Collect episodes
        combined_rollout: List[RolloutStep] = []
        ep_rewards = []

        for ep_offset in range(args.episodes_per_update):
            ep_num = ep_start + ep_offset
            if ep_num >= args.num_episodes:
                break

            # Refresh self-play shadow periodically
            if ep_num > 0 and ep_num % SELF_PLAY_REFRESH_INTERVAL == 0:
                self_play_fn = make_self_play_agent(
                    model, num_players, args.noop_penalty, device,
                )

            # Refresh lagging reference periodically
            if ep_num > 0 and ep_num % LAGGING_REFRESH_INTERVAL == 0:
                lagging_fn = make_self_play_agent(
                    model, num_players, args.noop_penalty, device,
                )
                print(f"  --- Refreshed lagging reference at ep {ep_num} ---")

            # Pick opponent: 5% baseline, 5% lagging, 90% normal curriculum
            ref_roll = random.random()
            if ref_roll < 0.05:
                opp_name, opp_fn = "baseline", baseline_fn
            elif ref_roll < 0.10:
                opp_name, opp_fn = "lagging", lagging_fn
            elif use_curriculum:
                opp_name, opp_fn = pick_curriculum_opponent(
                    heuristics, self_play_fn, curriculum_stage,
                )
            else:
                all_pool = [("random", "random")] + heuristics
                opp_name, opp_fn = random.choice(all_pool)

            # Determine horizon for this episode
            if use_progressive:
                horizon = HORIZON_STAGES[horizon_stage][1]
            else:
                horizon = args.max_steps

            # Higher reward for heuristic wins to give stronger learning signal
            is_heuristic = opp_name not in ("random", "self_play", "baseline", "lagging")
            ep_win_bonus = 15.0 if is_heuristic else 10.0

            t0 = time.time()
            rollout, final_reward, ep_stats = play_episode(
                model, opp_fn, mode=args.mode,
                noop_penalty=args.noop_penalty,
                idle_penalty=args.idle_penalty,
                step_penalty=args.step_penalty,
                max_steps=horizon,
                win_bonus=ep_win_bonus,
                device=device,
            )
            elapsed = time.time() - t0

            combined_rollout.extend(rollout)
            ep_rewards.append(final_reward)
            window_results.append(final_reward)
            if len(window_results) > 50:
                window_results.pop(0)

            # Track heuristic-only win rate for curriculum advancement
            is_heuristic = opp_name not in ("random", "self_play", "baseline", "lagging")
            if is_heuristic:
                heuristic_results.append(final_reward)
                if len(heuristic_results) > 100:
                    heuristic_results.pop(0)

            # Track reference opponent results
            if opp_name == "baseline":
                baseline_results.append(final_reward)
                if len(baseline_results) > 50:
                    baseline_results.pop(0)
            elif opp_name == "lagging":
                lagging_results.append(final_reward)
                if len(lagging_results) > 50:
                    lagging_results.pop(0)

            total_episodes += 1
            if final_reward > 0:
                total_wins += 1

            result = "WIN" if final_reward > 0 else ("LOSS" if final_reward < 0 else "DRAW")
            noop_count = sum(1 for s in rollout if s.is_noop)
            launch_count = len(rollout) - noop_count
            horizon_str = f"/{horizon}" if use_progressive else ""
            mf = ep_stats["my_fleet"]
            mp = ep_stats["my_prod"]
            ef = ep_stats["enemy_fleet"]
            ep_ = ep_stats["enemy_prod"]
            print(
                f"  Ep {ep_num+1:>5}/{args.num_episodes} vs {opp_name:<16} "
                f"{result:>4}  steps={len(rollout):>3}{horizon_str}  "
                f"fleet={mf:.0f}v{ef:.0f}  prod={mp:.1f}v{ep_:.1f}  "
                f"launch={launch_count:>3}  noop={noop_count:>3}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

        if not combined_rollout:
            continue

        # PPO update (optionally on a different device for acceleration)
        if update_device != device:
            model = model.to(update_device)
        model.train()
        stats = ppo_update(
            model, optimizer, combined_rollout,
            max_planets=args.max_planets,
            device=update_device,
        )
        model.eval()
        if update_device != device:
            model = model.to(device)

        # Stats
        win_rate_window = sum(1 for r in window_results if r > 0) / max(len(window_results), 1)
        heuristic_wr = (
            sum(1 for r in heuristic_results if r > 0) / max(len(heuristic_results), 1)
            if heuristic_results else 0.0
        )
        avg_reward = sum(ep_rewards) / len(ep_rewards)
        baseline_wr = (
            sum(1 for r in baseline_results if r > 0) / max(len(baseline_results), 1)
            if baseline_results else 0.0
        )
        lagging_wr = (
            sum(1 for r in lagging_results if r > 0) / max(len(lagging_results), 1)
            if lagging_results else 0.0
        )

        stage_name = CURRICULUM_STAGES[min(curriculum_stage, len(CURRICULUM_STAGES) - 1)]["name"]
        horizon_now = HORIZON_STAGES[horizon_stage][1] if use_progressive else args.max_steps
        print(
            f"  Update: policy={stats['policy_loss']:.4f} "
            f"value={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.3f} "
            f"kl={stats['kl']:.4f} "
            f"clip={stats['clip_frac']:.2f} "
            f"| wr(50)={win_rate_window:.1%} "
            f"heur_wr={heuristic_wr:.1%} "
            f"avg_r={avg_reward:+.2f} "
            f"base_wr={baseline_wr:.0%} "
            f"lag_wr={lagging_wr:.0%} "
            f"stage={stage_name} "
            f"horizon={horizon_now}",
            flush=True,
        )

        # Sanity summary every 1000 episodes
        if total_episodes > 0 and total_episodes % 1000 == 0:
            overall_wr = total_wins / total_episodes
            print(
                f"\n  === Ep {total_episodes} summary ===\n"
                f"    Overall wr: {overall_wr:.1%} ({total_wins}/{total_episodes})\n"
                f"    Heuristic wr (last 100): {heuristic_wr:.1%} ({len(heuristic_results)} games)\n"
                f"    vs Baseline (last 50): {baseline_wr:.0%} ({len(baseline_results)} games)\n"
                f"    vs Lagging (last 50): {lagging_wr:.0%} ({len(lagging_results)} games)\n"
                f"    Curriculum: {stage_name}, Horizon: {horizon_now}\n"
                f"    Entropy: {stats['entropy']:.3f}, Value loss: {stats['value_loss']:.4f}\n",
                flush=True,
            )

        # Curriculum advancement — gate on HEURISTIC win rate, not overall.
        # Each stage defines its own min_games threshold for advancement.
        if use_curriculum:
            stage_cfg = CURRICULUM_STAGES[min(curriculum_stage, len(CURRICULUM_STAGES) - 1)]
            min_games = stage_cfg.get("min_games", 50)
            advance_wr = stage_cfg["advance_wr"]
            if len(heuristic_results) >= min_games and heuristic_wr >= advance_wr and curriculum_stage < len(CURRICULUM_STAGES) - 1:
                curriculum_stage += 1
                new_stage = CURRICULUM_STAGES[curriculum_stage]
                print(f"  >>> Curriculum advanced to stage {curriculum_stage}: "
                      f"{new_stage['name']} "
                      f"(random={new_stage['random']:.0%} "
                      f"self_play={new_stage['self_play']:.0%} "
                      f"heuristic={new_stage['heuristic']:.0%})")

        # Horizon advancement — performance-gated, not time-based.
        # Need 75% heur wr over min_games to unlock the next horizon.
        if use_progressive and horizon_stage < len(HORIZON_STAGES) - 1:
            next_wr_threshold = HORIZON_STAGES[horizon_stage + 1][0]
            horizon_min_games = 50
            if len(heuristic_results) >= horizon_min_games and heuristic_wr >= next_wr_threshold:
                horizon_stage += 1
                new_horizon = HORIZON_STAGES[horizon_stage][1]
                print(f"  >>> Horizon advanced to {new_horizon} steps "
                      f"(heur_wr={heuristic_wr:.1%} >= {next_wr_threshold:.0%})")
                # Reset heuristic window so it must re-prove at the new horizon
                heuristic_results.clear()

        # Save best checkpoint based on heuristic win rate
        if heuristic_wr > best_heuristic_wr and len(heuristic_results) >= 30:
            best_heuristic_wr = heuristic_wr
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint (heuristic_wr={heuristic_wr:.1%})")

        # Periodic save
        if (ep_start + args.episodes_per_update) % 20 == 0:
            latest_path = str(cache_dir / f"checkpoint_ppo_{args.mode}_latest.pt")
            torch.save(model.state_dict(), latest_path)

        # Periodic eval: every eval_every episodes, play a mini-match vs each heuristic
        # Uses the CURRENT horizon so results are comparable to training games
        if total_episodes % args.eval_every == 0 and heuristics:
            eval_horizon = HORIZON_STAGES[horizon_stage][1] if use_progressive else args.max_steps
            print(f"\n  --- Eval checkpoint (ep {total_episodes}, horizon={eval_horizon}) ---")
            eval_wins = 0
            eval_total = 0
            for h_name, h_fn in heuristics:
                h_wins = 0
                for _ in range(4):  # 4 games per heuristic for better signal
                    _, r, es = play_episode(
                        model, h_fn, mode=args.mode,
                        noop_penalty=args.noop_penalty,
                        idle_penalty=args.idle_penalty,
                        max_steps=eval_horizon,
                        device=device,
                    )
                    eval_total += 1
                    if r > 0:
                        eval_wins += 1
                        h_wins += 1
                print(f"    vs {h_name}: {h_wins}/4")
            eval_wr = eval_wins / max(eval_total, 1)
            print(f"  Eval total: {eval_wins}/{eval_total} ({eval_wr:.1%})")
            print()

    # Final save
    latest_path = str(cache_dir / f"checkpoint_ppo_{args.mode}_latest.pt")
    torch.save(model.state_dict(), latest_path)
    overall_wr = total_wins / max(total_episodes, 1)
    print(f"\nPPO training complete. {total_episodes} episodes, "
          f"overall win rate: {overall_wr:.1%}")
    print(f"Best heuristic win rate: {best_heuristic_wr:.1%}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Latest checkpoint: {latest_path}")


if __name__ == "__main__":
    main()
