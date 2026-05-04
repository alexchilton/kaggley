"""Parse Kaggle Orbit Wars replay JSON files into training datasets.

Each replay contains 500 steps x N players, with full observations (planets,
fleets) and actions [[source_id, angle_radians, num_ships], ...].

This module converts those into (graph_obs, factored_action, discounted_return)
tuples suitable for behavioral cloning and offline RL.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .gnn_policy import FRACTION_BUCKETS, NODE_DIM

BOARD_SIZE = 100.0
SUN_X, SUN_Y = 50.0, 50.0
DEFAULT_SHIP_SPEED = 6.0


@dataclass
class Transition:
    node_features: torch.Tensor   # (N, 10)
    positions: torch.Tensor       # (N, 2)
    owned_mask: torch.Tensor      # (N,)
    source_idx: int               # planet index or -1 for noop
    target_idx: int               # planet index or -1 for noop
    fraction_idx: int             # 0-3 or -1 for noop
    is_noop: bool
    discounted_return: float
    player_rank: int              # 1-based rank (1 = winner)
    mode: str                     # "2p" or "4p"


def _angle_to_target(
    source_x: float, source_y: float, angle: float,
    planets: List[List], source_id: int,
    cone_tolerance: float = 0.3,
) -> Optional[int]:
    """Convert an angle-based action to the target planet index.

    Finds the planet closest to the ray from the source at the given angle.
    Uses perpendicular distance with a forward-only constraint.
    """
    best_idx = None
    best_score = float("inf")

    for i, p in enumerate(planets):
        pid = int(p[0])
        if pid == source_id:
            continue
        px, py = float(p[2]), float(p[3])
        dx = px - source_x
        dy = py - source_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            continue

        # Angle from source to this planet
        planet_angle = math.atan2(dy, dx)
        # Angular difference (wrapped to [-pi, pi])
        diff = (planet_angle - angle + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > cone_tolerance:
            continue

        # Score: distance * angular deviation (prefer close + on-target)
        score = dist * (1.0 + abs(diff))
        if score < best_score:
            best_score = score
            best_idx = i

    return best_idx


def _fraction_to_bucket(ships_sent: float, ships_available: float) -> int:
    """Snap ship fraction to nearest bucket index."""
    if ships_available < 1:
        return 0
    frac = ships_sent / ships_available
    best_idx = 0
    best_diff = abs(frac - FRACTION_BUCKETS[0])
    for i, b in enumerate(FRACTION_BUCKETS[1:], 1):
        d = abs(frac - b)
        if d < best_diff:
            best_diff = d
            best_idx = i
    return best_idx


def _build_node_features(
    planets: List[List], fleets: List[List], player_id: int, num_players: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build node features, positions, and owned_mask from observation data.

    Returns:
        node_features: (N, 10)
        positions: (N, 2)
        owned_mask: (N,) — 1.0 for owned planets with ships > 0
    """
    N = len(planets)
    nf = torch.zeros(N, NODE_DIM)
    pos = torch.zeros(N, 2)
    owned = torch.zeros(N)

    # Pre-compute inbound fleets per planet
    inbound_friendly = [0.0] * N
    inbound_enemy = [0.0] * N
    # Build planet_id -> index mapping
    pid_to_idx = {}
    for i, p in enumerate(planets):
        pid_to_idx[int(p[0])] = i

    for fleet in fleets:
        # fleet: [id, owner, x, y, angle, from_planet_id, ships]
        f_owner = int(fleet[1])
        f_ships = float(fleet[6])
        # Determine target by finding closest planet in fleet's direction
        # For simplicity, attribute inbound to the nearest planet ahead
        fx, fy = float(fleet[2]), float(fleet[3])
        f_angle = float(fleet[4])
        best_dist = float("inf")
        best_planet_idx = None
        for i, p in enumerate(planets):
            px, py = float(p[2]), float(p[3])
            dx, dy = px - fx, py - fy
            dist = math.sqrt(dx * dx + dy * dy)
            # Check if planet is roughly in front of fleet
            to_planet_angle = math.atan2(dy, dx)
            angle_diff = abs((to_planet_angle - f_angle + math.pi) % (2 * math.pi) - math.pi)
            if angle_diff < 0.5 and dist < best_dist:
                best_dist = dist
                best_planet_idx = i
        if best_planet_idx is not None:
            if f_owner == player_id:
                inbound_friendly[best_planet_idx] += f_ships
            else:
                inbound_enemy[best_planet_idx] += f_ships

    for i, p in enumerate(planets):
        px, py = float(p[2]), float(p[3])
        p_owner = int(p[1])
        p_ships = float(p[5])
        p_prod = float(p[6])

        pos[i, 0] = px / BOARD_SIZE
        pos[i, 1] = py / BOARD_SIZE

        is_mine = 1.0 if p_owner == player_id else 0.0
        is_enemy = 1.0 if p_owner >= 0 and p_owner != player_id else 0.0
        is_neutral = 1.0 if p_owner < 0 else 0.0

        nf[i, 0] = px / BOARD_SIZE
        nf[i, 1] = py / BOARD_SIZE
        nf[i, 2] = is_mine
        nf[i, 3] = is_enemy
        nf[i, 4] = is_neutral
        nf[i, 5] = math.log1p(p_ships) / 6.5  # normalized: log1p(500)~6.2
        nf[i, 6] = math.log1p(p_prod) / 4.5    # normalized: log1p(60)~4.1
        nf[i, 7] = 1.0 if is_mine and p_ships > 0 else 0.0  # ship_ratio (simplified)
        nf[i, 8] = math.log1p(inbound_friendly[i]) / 6.5
        nf[i, 9] = math.log1p(inbound_enemy[i]) / 6.5

        if is_mine > 0 and p_ships > 0:
            owned[i] = 1.0

    return nf, pos, owned


def parse_replay(filepath: str, gamma: float = 0.999) -> List[Transition]:
    """Parse a single replay file into transitions.

    Args:
        filepath: Path to the replay JSON.
        gamma: Discount factor for computing returns.

    Returns:
        List of Transition objects, one per (step, player) with valid observation.
    """
    with open(filepath) as f:
        data = json.load(f)

    if "steps" not in data:
        return []

    steps = data["steps"]
    num_players = len(steps[0])
    mode = "2p" if num_players == 2 else "4p"
    rewards = data.get("rewards", [0] * num_players)

    # Compute player ranks from rewards
    sorted_rewards = sorted(enumerate(rewards), key=lambda x: -(x[1] or -999))
    rank_map = {}
    for rank_idx, (pid, _) in enumerate(sorted_rewards):
        rank_map[pid] = rank_idx + 1

    # Collect raw transitions per player
    player_transitions: Dict[int, List[dict]] = {p: [] for p in range(num_players)}

    for step_idx, step in enumerate(steps):
        for player_id in range(num_players):
            agent_data = step[player_id]
            obs = agent_data.get("observation")
            action = agent_data.get("action", [])

            if obs is None or "planets" not in obs:
                continue

            planets = obs["planets"]
            fleets = obs.get("fleets", [])

            if not planets:
                continue

            # Check if player is still alive (owns at least one planet)
            has_planets = any(int(p[1]) == player_id for p in planets)
            if not has_planets:
                continue

            node_features, positions, owned_mask = _build_node_features(
                planets, fleets, player_id, num_players,
            )

            # Build planet_id -> index mapping for this step
            pid_to_idx = {int(p[0]): i for i, p in enumerate(planets)}

            # Convert actions
            if not action:
                player_transitions[player_id].append({
                    "nf": node_features, "pos": positions, "om": owned_mask,
                    "source": -1, "target": -1, "fraction": -1, "is_noop": True,
                    "step": step_idx,
                })
            else:
                # Take the first action (multi-launch: just use first for now)
                act = action[0]
                src_planet_id = int(act[0])
                angle = float(act[1])
                ships_sent = float(act[2])

                if src_planet_id not in pid_to_idx:
                    continue

                src_idx = pid_to_idx[src_planet_id]
                src_planet = planets[src_idx]
                src_x, src_y = float(src_planet[2]), float(src_planet[3])
                src_ships = float(src_planet[5])

                tgt_idx = _angle_to_target(src_x, src_y, angle, planets, src_planet_id)
                if tgt_idx is None:
                    continue

                frac_idx = _fraction_to_bucket(ships_sent, src_ships)

                player_transitions[player_id].append({
                    "nf": node_features, "pos": positions, "om": owned_mask,
                    "source": src_idx, "target": tgt_idx, "fraction": frac_idx,
                    "is_noop": False, "step": step_idx,
                })

    # Compute discounted returns
    all_transitions = []
    for player_id, trans_list in player_transitions.items():
        if not trans_list:
            continue
        final_reward = rewards[player_id] if rewards[player_id] is not None else 0.0
        rank = rank_map.get(player_id, num_players)

        # Assign discounted return: final_reward * gamma^(T - t)
        T = len(steps) - 1
        for t_data in trans_list:
            steps_remaining = T - t_data["step"]
            disc_return = final_reward * (gamma ** steps_remaining)

            all_transitions.append(Transition(
                node_features=t_data["nf"],
                positions=t_data["pos"],
                owned_mask=t_data["om"],
                source_idx=t_data["source"],
                target_idx=t_data["target"],
                fraction_idx=t_data["fraction"],
                is_noop=t_data["is_noop"],
                discounted_return=disc_return,
                player_rank=rank,
                mode=mode,
            ))

    return all_transitions


def parse_all_replays(
    replay_dir: str,
    gamma: float = 0.999,
    verbose: bool = True,
) -> Tuple[List[Transition], List[Transition]]:
    """Parse all replay files, split into 2P and 4P datasets.

    Returns:
        (transitions_2p, transitions_4p)
    """
    trans_2p: List[Transition] = []
    trans_4p: List[Transition] = []
    total_files = 0
    errors = 0

    for root, _dirs, files in os.walk(replay_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            total_files += 1
            try:
                transitions = parse_replay(os.path.join(root, fname), gamma)
                for t in transitions:
                    if t.mode == "2p":
                        trans_2p.append(t)
                    else:
                        trans_4p.append(t)
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  Error parsing {fname}: {e}")

    if verbose:
        print(f"Parsed {total_files} files ({errors} errors)")
        print(f"  2P transitions: {len(trans_2p)}")
        print(f"  4P transitions: {len(trans_4p)}")

    return trans_2p, trans_4p


class ReplayDataset(Dataset):
    """PyTorch Dataset wrapping a list of Transitions.

    Pads/truncates to max_planets for batching. Planets beyond the actual
    count are zero-padded with owned_mask = 0.
    """

    def __init__(
        self,
        transitions: List[Transition],
        max_planets: int = 48,
        winners_only: bool = False,
    ):
        if winners_only:
            transitions = [t for t in transitions if t.player_rank == 1]
        self.transitions = transitions
        self.max_planets = max_planets

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.transitions[idx]
        N = t.node_features.shape[0]
        M = self.max_planets

        # Pad to max_planets
        nf = torch.zeros(M, NODE_DIM)
        pos = torch.zeros(M, 2)
        owned = torch.zeros(M)
        valid = torch.zeros(M)

        n = min(N, M)
        nf[:n] = t.node_features[:n]
        pos[:n] = t.positions[:n]
        owned[:n] = t.owned_mask[:n]
        valid[:n] = 1.0

        source = t.source_idx if not t.is_noop else M  # noop = M (the N+1 slot after padding)
        target = t.target_idx if not t.is_noop else 0
        fraction = t.fraction_idx if not t.is_noop else 0

        return {
            "node_features": nf,
            "positions": pos,
            "owned_mask": owned,
            "valid_mask": valid,
            "source": torch.tensor(source, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "fraction": torch.tensor(fraction, dtype=torch.long),
            "is_noop": torch.tensor(1.0 if t.is_noop else 0.0),
            "discounted_return": torch.tensor(t.discounted_return, dtype=torch.float32),
            "player_rank": torch.tensor(t.player_rank, dtype=torch.long),
            "num_planets": torch.tensor(n, dtype=torch.long),
        }


def save_dataset(transitions: List[Transition], path: str) -> None:
    """Save transitions to disk as a .pt file."""
    torch.save(transitions, path)
    print(f"Saved {len(transitions)} transitions to {path}")


def load_dataset(path: str) -> List[Transition]:
    """Load transitions from a .pt file."""
    return torch.load(path, weights_only=False)


def prebatch_and_save(
    transitions: List[Transition],
    path: str,
    max_planets: int = 40,
    winners_only: bool = False,
) -> None:
    """Convert transitions to pre-padded tensors and save as a single dict.

    Much faster to load than a list of Transition objects.
    """
    if winners_only:
        transitions = [t for t in transitions if t.player_rank == 1]

    N = len(transitions)
    M = max_planets

    nf = torch.zeros(N, M, NODE_DIM)
    pos = torch.zeros(N, M, 2)
    owned = torch.zeros(N, M)
    sources = torch.zeros(N, dtype=torch.long)
    targets = torch.zeros(N, dtype=torch.long)
    fractions = torch.zeros(N, dtype=torch.long)
    is_noops = torch.zeros(N)
    returns = torch.zeros(N)
    ranks = torch.zeros(N, dtype=torch.long)
    num_planets = torch.zeros(N, dtype=torch.long)

    # Filter out samples where action references invalid planets
    valid_transitions = []
    skipped_oob = 0
    skipped_unowned = 0
    for t in transitions:
        if not t.is_noop:
            if t.source_idx >= M or t.target_idx >= M:
                skipped_oob += 1
                continue  # action references a planet we'd truncate
            # Skip if source planet isn't owned by this player in the observation
            if t.owned_mask[t.source_idx].item() == 0:
                skipped_unowned += 1
                continue
        valid_transitions.append(t)

    if skipped_oob > 0:
        print(f"  Skipped {skipped_oob} transitions with actions beyond max_planets={M}")
    if skipped_unowned > 0:
        print(f"  Skipped {skipped_unowned} transitions with unowned source planet")
    transitions = valid_transitions
    N = len(transitions)

    # Re-allocate with correct size
    nf = torch.zeros(N, M, NODE_DIM)
    pos = torch.zeros(N, M, 2)
    owned = torch.zeros(N, M)
    sources = torch.zeros(N, dtype=torch.long)
    targets = torch.zeros(N, dtype=torch.long)
    fractions = torch.zeros(N, dtype=torch.long)
    is_noops = torch.zeros(N)
    returns = torch.zeros(N)
    ranks = torch.zeros(N, dtype=torch.long)
    num_planets = torch.zeros(N, dtype=torch.long)

    for i, t in enumerate(transitions):
        n = min(t.node_features.shape[0], M)
        nf[i, :n] = t.node_features[:n]
        pos[i, :n] = t.positions[:n]
        owned[i, :n] = t.owned_mask[:n]
        sources[i] = t.source_idx if not t.is_noop else M
        targets[i] = t.target_idx if not t.is_noop else 0
        fractions[i] = t.fraction_idx if not t.is_noop else 0
        is_noops[i] = 1.0 if t.is_noop else 0.0
        returns[i] = t.discounted_return
        ranks[i] = t.player_rank
        num_planets[i] = n

    data = {
        "node_features": nf, "positions": pos, "owned_mask": owned,
        "source": sources, "target": targets, "fraction": fractions,
        "is_noop": is_noops, "discounted_return": returns,
        "player_rank": ranks, "num_planets": num_planets,
    }
    torch.save(data, path)
    print(f"Saved {N} pre-batched transitions to {path}")


class PreBatchedDataset(Dataset):
    """Fast dataset from pre-tensorized data."""

    def __init__(self, path: str):
        self.data = torch.load(path, weights_only=True)
        self._len = self.data["source"].shape[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.data.items()}
