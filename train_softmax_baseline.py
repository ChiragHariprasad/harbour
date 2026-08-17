"""
Softmax-CNN Baseline: Training Script
Trains a plain CNN classifier (cross-entropy loss) using the same CNNEmbedder backbone.
This serves as the MSP (Maximum Softmax Probability) OOD baseline.
"""
import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from Bio import SeqIO
import gzip

# Import shared components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import encode_sequence, VOCAB_SIZE, DEVICE


class CNNClassifier(nn.Module):
    """Same CNN backbone as CNNEmbedder, but with a classification head instead of L2-normalized projection."""
    def __init__(self, num_classes, embed_dim=128, conv_channels=128):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE + 1, 64, padding_idx=VOCAB_SIZE)
        self.conv1 = nn.Conv1d(64, conv_channels, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(conv_channels, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        features = torch.relu(self.fc(x))
        logits = self.classifier(features)
        return logits


class eDNADataset(Dataset):
    def __init__(self, fasta_file: str):
        self.sequences = []
        self.labels = []
        self.label_map = {}

        fasta_path = Path(fasta_file)
        open_fn = gzip.open if fasta_path.suffix == '.gz' else open

        if fasta_path.exists():
            with open_fn(fasta_file, "rt") as f:
                for record in SeqIO.parse(f, "fasta"):
                    parts = record.id.split("|")
                    label_str = parts[-1] if len(parts) > 1 else "Unknown"

                    if label_str not in self.label_map:
                        self.label_map[label_str] = len(self.label_map)

                    self.sequences.append(str(record.seq))
                    self.labels.append(self.label_map[label_str])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = encode_sequence(self.sequences[idx]).squeeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def train_single_config(rank, seed, base_dir, epochs=3, batch_size=64, lr=1e-3):
    """Train a single softmax-CNN classifier for one rank/seed configuration."""
    cache_dir = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}"
    output_dir = f"{base_dir}/assets/HTRP_models/softmax_baseline/16S/{rank}/seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)

    train_fasta = f"{cache_dir}/train_fit.fasta"
    if not os.path.exists(train_fasta):
        train_fasta = f"{cache_dir}/train.fasta"

    print(f"[{rank}/{seed}] Loading dataset from {train_fasta}...")
    dataset = eDNADataset(train_fasta)
    num_classes = len(dataset.label_map)
    print(f"[{rank}/{seed}] {len(dataset)} sequences, {num_classes} classes")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = CNNClassifier(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    metrics_history = []
    total_train_time = 0

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(dataloader)
        acc = correct / total if total > 0 else 0
        epoch_time = time.time() - t0
        total_train_time += epoch_time

        metrics_history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_accuracy": acc,
            "epoch_time_s": epoch_time
        })
        print(f"[{rank}/{seed}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Time: {epoch_time:.1f}s")

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "classifier.pt"))

    # Save label map (needed for inference)
    reverse_map = {v: k for k, v in dataset.label_map.items()}
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"label_to_idx": dataset.label_map, "idx_to_label": reverse_map, "num_classes": num_classes}, f, indent=2)

    # Save training metrics
    metrics = {
        "model_type": "CNNClassifier_Softmax",
        "loss_type": "cross_entropy",
        "rank": rank,
        "seed": seed,
        "epochs": epochs,
        "num_classes": num_classes,
        "total_train_time_s": total_train_time,
        "history": metrics_history
    }
    with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{rank}/{seed}] Training complete. Model saved to {output_dir}")
    return {"rank": rank, "seed": seed, "status": "OK", "num_classes": num_classes, "train_time_s": total_train_time}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_dir", type=str, default="E:/HTRP")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    result = train_single_config(args.rank, args.seed, args.base_dir, args.epochs, args.batch_size, args.lr)
    print(json.dumps(result))
