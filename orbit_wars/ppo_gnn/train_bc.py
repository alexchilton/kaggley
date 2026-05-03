"""Phase 1: Behavioral Cloning from Kaggle replay data.

Trains the GNN policy to imitate winning players' actions using a factored
cross-entropy loss: CE(source) + CE(target|source) + CE(fraction|source,target).

Usage:
    python -m ppo_gnn.train_bc --mode 2p --replay-dir kaggle_replays --epochs 50
    python -m ppo_gnn.train_bc --mode 4p --replay-dir kaggle_replays --epochs 50
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .gnn_policy import OrbitWarsGNNPolicy
from .replay_parser import (
    ReplayDataset,
    load_dataset,
    parse_all_replays,
    save_dataset,
)


def bc_loss(
    model: OrbitWarsGNNPolicy,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute factored BC loss for a batch.

    Loss = CE(source) + CE(target|source) + CE(fraction|source,target)
    For noop actions, only the source CE applies.
    """
    nf = batch["node_features"].to(device)
    pos = batch["positions"].to(device)
    owned = batch["owned_mask"].to(device)
    source = batch["source"].to(device)
    target = batch["target"].to(device)
    fraction = batch["fraction"].to(device)
    is_noop = batch["is_noop"].to(device)
    num_planets = batch["num_planets"].to(device)

    B = nf.shape[0]
    M = nf.shape[1]  # max_planets (padded)

    source_logits, all_target_logits, all_fraction_logits = model(nf, pos, owned)

    # Source CE: noop maps to index M (the noop slot = last in source_logits)
    # source_logits is (B, M+1) — M planet slots + 1 noop slot
    # For noop: source target is M. For launch: source target is the planet index.
    source_target = torch.where(is_noop.bool(), torch.full_like(source, M), source)
    loss_source = F.cross_entropy(source_logits, source_target, reduction="mean")

    # Target CE and Fraction CE: only for non-noop
    launch_mask = (1.0 - is_noop)
    if launch_mask.sum() > 0:
        launch_idx = launch_mask.bool()

        # Target logits for the selected source
        src_clamped = source.clamp(0, M - 1)
        tgt_logits = all_target_logits[torch.arange(B, device=device), src_clamped]  # (B, M)
        loss_target_all = F.cross_entropy(tgt_logits, target.clamp(0, M - 1), reduction="none")
        loss_target = (loss_target_all * launch_mask).sum() / launch_mask.sum()

        # Fraction logits
        tgt_clamped = target.clamp(0, M - 1)
        frac_logits = all_fraction_logits[torch.arange(B, device=device), src_clamped, tgt_clamped]
        loss_frac_all = F.cross_entropy(frac_logits, fraction.clamp(0, 3), reduction="none")
        loss_frac = (loss_frac_all * launch_mask).sum() / launch_mask.sum()
    else:
        loss_target = torch.tensor(0.0, device=device)
        loss_frac = torch.tensor(0.0, device=device)

    return loss_source + loss_target + loss_frac


def train_bc(
    model: OrbitWarsGNNPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = "checkpoint_bc.pt",
) -> OrbitWarsGNNPolicy:
    """Train the model with behavioral cloning."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch in train_loader:
            loss = bc_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = bc_loss(model, batch, device)
                val_loss_sum += loss.item()
                val_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Behavioral Cloning")
    parser.add_argument("--mode", choices=["2p", "4p"], required=True)
    parser.add_argument("--replay-dir", default="kaggle_replays")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--use-sage", action="store_true", help="Use GraphSAGE instead of GAT")
    parser.add_argument("--mask-sun", action="store_true", help="Hard-mask sun-blocked targets")
    parser.add_argument("--max-planets", type=int, default=48)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load or parse replay data
    cache_file = cache_dir / f"transitions_{args.mode}.pt"
    if cache_file.exists():
        print(f"Loading cached transitions from {cache_file}")
        transitions = load_dataset(str(cache_file))
    else:
        print(f"Parsing replays from {args.replay_dir}...")
        trans_2p, trans_4p = parse_all_replays(args.replay_dir)
        save_dataset(trans_2p, str(cache_dir / "transitions_2p.pt"))
        save_dataset(trans_4p, str(cache_dir / "transitions_4p.pt"))
        transitions = trans_2p if args.mode == "2p" else trans_4p

    print(f"Total {args.mode} transitions: {len(transitions)}")

    # Build dataset (winners only for BC)
    dataset = ReplayDataset(transitions, max_planets=args.max_planets, winners_only=True)
    print(f"Winners-only dataset: {len(dataset)} samples")

    # Split 90/10
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = OrbitWarsGNNPolicy(
        hidden_dim=args.hidden_dim,
        use_gat=not args.use_sage,
        mask_sun_targets=args.mask_sun,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    checkpoint_path = str(cache_dir / f"checkpoint_bc_{args.mode}.pt")
    model = train_bc(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=device, checkpoint_path=checkpoint_path,
    )
    print(f"BC training complete. Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
