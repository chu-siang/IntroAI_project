import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, sosfiltfilt


def parse_args():
    parser = argparse.ArgumentParser(description="Train Task 2 ensemble EEG classifier.")

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2_train"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/task2_train/task2_model.pt"))

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--seeds", type=str, default="42,7,2025")

    # bandpass filter
    parser.add_argument("--fs", type=float, default=250.0)
    parser.add_argument("--lowcut", type=float, default=4.0)
    parser.add_argument("--highcut", type=float, default=40.0)
    parser.add_argument("--filter-order", type=int, default=4)

    return parser.parse_args()


def parse_seeds(seed_string: str):
    return [int(s.strip()) for s in seed_string.split(",") if s.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_task2_train_data(data_dir: Path):
    x_list = []
    y_list = []

    for i in range(1, 11):
        file_path = data_dir / f"subject{i:02d}.npz"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing training file: {file_path}")

        data = np.load(file_path)
        x_list.append(data["x"])
        y_list.append(data["y"])

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return x_all, y_all


def bandpass_filter(
    x: np.ndarray,
    fs: float = 250.0,
    lowcut: float = 4.0,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    nyquist = fs / 2.0

    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Invalid bandpass range: lowcut={lowcut}, highcut={highcut}, fs={fs}"
        )

    sos = butter(
        N=order,
        Wn=[lowcut / nyquist, highcut / nyquist],
        btype="bandpass",
        output="sos",
    )

    x_filtered = sosfiltfilt(sos, x, axis=2)
    return x_filtered.astype(np.float32)


class EEGDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EEGClassifier(nn.Module):
    def __init__(self, num_channels, input_length, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += len(x)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


def train_single_model(seed, x, y, args, device, num_channels, input_length):
    set_seed(seed)

    dataset = EEGDataset(x, y)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = EEGClassifier(
        num_channels=num_channels,
        input_length=input_length,
        num_classes=4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"\n========== Training seed {seed} ==========")

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Seed {seed} | Epoch {epoch:03d} | loss={loss:.4f} | train_acc={acc:.4f}")

    return model.cpu().state_dict()


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)
    print(f"Using device: {device}")

    seeds = parse_seeds(args.seeds)
    print("Using seeds:", seeds)

    x, y = load_task2_train_data(args.data)

    print("Raw x shape:", x.shape)
    print("Raw y shape:", y.shape)
    print("Labels:", np.unique(y, return_counts=True))

    # 1. bandpass filter
    x = bandpass_filter(
        x,
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        order=args.filter_order,
    )

    print(f"Applied bandpass filter: {args.lowcut}-{args.highcut} Hz, fs={args.fs}")

    # 2. global normalization
    mean = x.mean()
    std = x.std()

    x = (x - mean) / (std + 1e-8)
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    print("Global mean after filtering:", float(mean))
    print("Global std after filtering:", float(std))
    print("Preprocessed x mean:", float(x.mean()))
    print("Preprocessed x std:", float(x.std()))

    num_channels = x.shape[1]
    input_length = x.shape[2]

    model_state_dicts = []

    for seed in seeds:
        state_dict = train_single_model(
            seed=seed,
            x=x,
            y=y,
            args=args,
            device=device,
            num_channels=num_channels,
            input_length=input_length,
        )
        model_state_dicts.append(state_dict)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "ensemble": True,
        "model_state_dicts": model_state_dicts,
        "seeds": seeds,
        "model_config": {
            "num_channels": num_channels,
            "input_length": input_length,
            "num_classes": 4,
        },
        "preprocess": {
            "filter": {
                "type": "butter_bandpass",
                "fs": float(args.fs),
                "lowcut": float(args.lowcut),
                "highcut": float(args.highcut),
                "order": int(args.filter_order),
            },
            "normalization": {
                "type": "global_zscore",
                "mean": float(mean),
                "std": float(std),
            },
        },
    }

    torch.save(checkpoint, args.checkpoint)
    print(f"\nSaved ensemble checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()