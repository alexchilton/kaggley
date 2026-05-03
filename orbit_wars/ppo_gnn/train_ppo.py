"""Phase 3: Online PPO fine-tuning against heuristic opponents.

Initializes from Phase 2 (AWR) checkpoint and fine-tunes with standard PPO
using the factored action distribution. Designed to work with the existing
SimplifiedPlanetEnv or a compatible environment.

The factored log-prob is used for the PPO ratio:
    ratio = exp(new_log_prob - old_log_prob)
where log_prob = log p(source) + log p(target|source) + log p(fraction|source,target)

Usage:
    python -m ppo_gnn.train_ppo --mode 2p --checkpoint ppo_gnn/cache/checkpoint_awr_2p.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from .gnn_policy import FRACTION_BUCKETS, OrbitWarsGNNPolicy


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
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Run PPO update on a collected rollout."""
    advantages, returns = compute_gae(rollout, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Prepare batched tensors
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

    # Move to device
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

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
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

            # Clipped surrogate
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, returns[idx])
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

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
            num_updates += 1

    for k in stats:
        stats[k] /= max(num_updates, 1)
    return stats


def collect_rollout_from_env(
    model: OrbitWarsGNNPolicy,
    env,
    player: int,
    num_steps: int = 128,
    device: torch.device = torch.device("cpu"),
) -> List[RolloutStep]:
    """Collect a rollout by playing in an environment.

    This is a template — the actual env interface depends on your setup.
    Expects env to provide:
        - env.get_observation(player) -> dict with 'planets', 'fleets'
        - env.step(player, action) -> reward, done
        - env.reset() -> None
    """
    from .replay_parser import _build_node_features

    model.eval()
    rollout = []

    for _ in range(num_steps):
        obs = env.get_observation(player)
        planets = obs["planets"]
        fleets = obs.get("fleets", [])
        num_players = obs.get("num_players", 2)

        nf, pos, owned = _build_node_features(planets, fleets, player, num_players)
        nf = nf.to(device)
        pos = pos.to(device)
        owned = owned.to(device)

        src, tgt, frac, is_noop, log_prob, value = model.sample_action(nf, pos, owned)

        # Convert to env action format
        if is_noop:
            action = []
        else:
            src_planet = planets[src]
            tgt_planet = planets[tgt]
            angle = math.atan2(
                float(tgt_planet[3]) - float(src_planet[3]),
                float(tgt_planet[2]) - float(src_planet[2]),
            )
            ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])
            action = [[int(src_planet[0]), angle, ships]]

        reward, done = env.step(player, action)

        rollout.append(RolloutStep(
            node_features=nf.cpu(),
            positions=pos.cpu(),
            owned_mask=owned.cpu(),
            source=src,
            target=tgt,
            fraction=frac,
            is_noop=is_noop,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
        ))

        if done:
            break

    return rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Online PPO")
    parser.add_argument("--mode", choices=["2p", "4p"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Phase 2 (AWR) checkpoint")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--use-sage", action="store_true")
    parser.add_argument("--mask-sun", action="store_true")
    parser.add_argument("--max-planets", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)

    model = OrbitWarsGNNPolicy(
        hidden_dim=args.hidden_dim,
        use_gat=not args.use_sage,
        mask_sun_targets=args.mask_sun,
    )
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # NOTE: You need to provide an env that implements get_observation/step/reset.
    # This is a placeholder showing the training loop structure.
    print("Phase 3 requires a live environment. The training loop is structured as:")
    print("  1. Collect rollout via collect_rollout_from_env()")
    print("  2. Run ppo_update() on the rollout")
    print("  3. Repeat for --num-updates iterations")
    print()
    print("To use with SimplifiedPlanetEnv, wrap it to provide get_observation/step.")
    print("To use with the full Kaggle env, provide a compatible wrapper.")
    print()
    print(f"Model ready for fine-tuning. Would save to: {cache_dir}/checkpoint_ppo_{args.mode}.pt")


if __name__ == "__main__":
    main()
