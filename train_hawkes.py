"""
Train TLOB-Transformer-Hawkes hybrid with TTE-conditioned additive bias.

Reads:  data/sequences/spx_smoke_2025-06-17_hawkes/sequences.npz
Writes: checkpoints/tlob_hawkes_<timestamp>.pt

Joint loss: direction cross-entropy + Hawkes NLL on next-event prediction.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score

from models.tlob_hawkes import TLOBHawkes, hawkes_nll_loss


def load_data(npz_path: str):
    """Load all six arrays and split 70/15/15 (random)."""
    data = np.load(npz_path)
    sequences = data["sequences"]
    labels = data["labels"]
    tte_days = data["tte_days"]
    contract_ids = data["contract_ids"]
    next_event_type = data["next_event_type"]
    next_event_dt = data["next_event_dt"]

    n = len(sequences)
    idx = np.arange(n)
    np.random.default_rng(seed=42).shuffle(idx)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    def take(i):
        return (
            sequences[i], labels[i], tte_days[i],
            next_event_type[i], next_event_dt[i],
        )

    return {
        "train": take(train_idx),
        "val": take(val_idx),
        "test": take(test_idx),
    }


def make_loader(sequences, labels, tte, next_type, next_dt, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(sequences).float(),
        torch.from_numpy(labels).long(),
        torch.from_numpy(tte).float(),
        torch.from_numpy(next_type).long(),
        torch.from_numpy(next_dt).float(),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=2, pin_memory=True,
    )


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    npz_path = "data/sequences/spx_smoke_2025-06-17_hawkes/sequences.npz"
    batch_size = 256
    lr = 1e-4
    num_epochs = 10
    seq_size = 128
    num_features = 5  # 4 LOB + 1 log-TTE
    hidden_dim = 40
    num_layers = 4
    num_heads = 1
    num_event_types = 3
    hawkes_layers = 2

    # Load
    print("Loading data...")
    splits = load_data(npz_path)
    print(f"Train: {len(splits['train'][0]):,} | "
          f"Val: {len(splits['val'][0]):,} | "
          f"Test: {len(splits['test'][0]):,}")

    train_loader = make_loader(*splits["train"], batch_size=batch_size, shuffle=True)
    val_loader = make_loader(*splits["val"], batch_size=batch_size, shuffle=False)
    test_loader = make_loader(*splits["test"], batch_size=batch_size, shuffle=False)

    # Model
    print("Building model...")
    model = TLOBHawkes(
        num_features=num_features,
        seq_size=seq_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_event_types=num_event_types,
        hawkes_layers=hawkes_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    print("\nTraining...")
    best_val_f1 = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = ckpt_dir / f"tlob_hawkes_{timestamp}.pt"

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_ce_sum = 0.0
        train_hawkes_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y, tte, nxt_type, nxt_dt in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tte = tte.to(device, non_blocking=True)
            nxt_type = nxt_type.to(device, non_blocking=True)
            nxt_dt = nxt_dt.to(device, non_blocking=True)

            optimizer.zero_grad()
            log_intensity, dir_logits = model(x, tte)

            ce = ce_loss_fn(dir_logits, y)
            hnll = hawkes_nll_loss(log_intensity, nxt_type, nxt_dt)

            # Equal weighting (as per design decision)
            hawkes_weight = 0.00
            loss = ce + hawkes_weight * hnll
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_ce_sum += ce.item() * bs
            train_hawkes_sum += hnll.item() * bs
            train_correct += (dir_logits.argmax(dim=1) == y).sum().item()
            train_total += bs

        train_ce = train_ce_sum / train_total
        train_hawkes = train_hawkes_sum / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_ce_sum = 0.0
        val_hawkes_sum = 0.0
        val_total = 0

        with torch.no_grad():
            for x, y, tte, nxt_type, nxt_dt in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                tte = tte.to(device, non_blocking=True)
                nxt_type = nxt_type.to(device, non_blocking=True)
                nxt_dt = nxt_dt.to(device, non_blocking=True)

                log_intensity, dir_logits = model(x, tte)
                ce = ce_loss_fn(dir_logits, y)
                hnll = hawkes_nll_loss(log_intensity, nxt_type, nxt_dt)

                bs = x.size(0)
                val_ce_sum += ce.item() * bs
                val_hawkes_sum += hnll.item() * bs
                val_total += bs
                val_preds.append(dir_logits.argmax(dim=1).cpu().numpy())
                val_labels.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_ce = val_ce_sum / val_total
        val_hawkes = val_hawkes_sum / val_total
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_acc = (val_preds == val_labels).mean()

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"train: ce={train_ce:.4f} hnll={train_hawkes:.4f} acc={train_acc:.4f} | "
              f"val: ce={val_ce:.4f} hnll={val_hawkes:.4f} f1={val_f1:.4f} acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint")

    # Test
    print(f"\nLoading best checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for x, y, tte, nxt_type, nxt_dt in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tte = tte.to(device, non_blocking=True)
            log_intensity, dir_logits = model(x, tte)
            test_preds.append(dir_logits.argmax(dim=1).cpu().numpy())
            test_labels.append(y.cpu().numpy())
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)

    print("\n" + "=" * 60)
    print("Test set results:")
    print(classification_report(test_labels, test_preds, digits=4,
                                target_names=["down", "stable", "up"]))
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"Macro F1: {test_f1:.4f}")
    print(f"Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()