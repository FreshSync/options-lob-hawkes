"""
Train TLOB on SPX options sequences.

Reads:  data/sequences/spx_smoke_2025-06-17_h50/sequences.npz
Writes: checkpoints/tlob_options_<timestamp>.pt
        logs/tlob_options_<timestamp>.txt
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score

from models.tlob import TLOB


def load_data(npz_path: str):
    """Load sequences and split into train/val/test by contract-day (temporal)."""
    data = np.load(npz_path)
    sequences = data["sequences"]  # (N, 128, 4)
    labels = data["labels"]         # (N,)
    tte_days = data["tte_days"]     # (N,)
    contract_ids = data["contract_ids"]  # (N,)

    # Temporal split: since the data is 1 hour, we can't do a proper day-based split.
    # For this smoke test, just do 70/15/15 on the sequence order (which is already
    # sorted by time within each contract, then concatenated across contracts).
    # This isn't a "correct" split for a paper but is fine for the smoke test.
    n = len(sequences)
    idx = np.arange(n)
    np.random.default_rng(seed=42).shuffle(idx)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return {
        "train": (sequences[train_idx], labels[train_idx]),
        "val": (sequences[val_idx], labels[val_idx]),
        "test": (sequences[test_idx], labels[test_idx]),
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config (matching TLOB defaults where possible)
    npz_path = "data/sequences/spx_smoke_2025-06-17_h50/sequences.npz"
    batch_size = 256
    lr = 1e-4
    num_epochs = 10
    seq_size = 128
    num_features = 4
    hidden_dim = 40
    num_layers = 4
    num_heads = 1

    # Load
    print("Loading data...")
    splits = load_data(npz_path)
    print(f"Train: {len(splits['train'][0]):,} | "
          f"Val: {len(splits['val'][0]):,} | "
          f"Test: {len(splits['test'][0]):,}")

    def make_loader(sequences, labels, shuffle):
        ds = TensorDataset(
            torch.from_numpy(sequences).float(),
            torch.from_numpy(labels).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    train_loader = make_loader(*splits["train"], shuffle=True)
    val_loader = make_loader(*splits["val"], shuffle=False)
    test_loader = make_loader(*splits["test"], shuffle=False)

    # Model
    print("Building model...")
    model = TLOB(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_size=seq_size,
        num_features=num_features,
        num_heads=num_heads,
        is_sin_emb=True,
        dataset_type="OPTIONS",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    print("\nTraining...")
    best_val_f1 = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = ckpt_dir / f"tlob_options_{timestamp}.pt"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                val_preds.append(logits.argmax(dim=1).cpu().numpy())
                val_labels.append(y.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_acc = (val_preds == val_labels).mean()

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_f1={val_f1:.4f} val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint (best val_f1 so far)")

    # Test
    print(f"\nLoading best checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            test_preds.append(logits.argmax(dim=1).cpu().numpy())
            test_labels.append(y.cpu().numpy())
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)

    print("\n" + "="*60)
    print("Test set results:")
    print(classification_report(test_labels, test_preds, digits=4,
                                target_names=["down", "stable", "up"]))
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"Macro F1: {test_f1:.4f}")
    print(f"Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()