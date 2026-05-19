import argparse
from pathlib import Path
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, sosfiltfilt


def parse_args():
    parser = argparse.ArgumentParser(description="Train Task 2 final ensemble EEG classifier.")

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2_train"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/task2_train/task2_model.pt"))

    # final training
    parser.add_argument("--epochs", type=int, default=85)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    # ensemble seeds
    parser.add_argument("--seeds", type=str, default="42,7,2025,88,123,999,314")

    # bandpass
    parser.add_argument("--fs", type=float, default=250.0)
    parser.add_argument("--lowcut", type=float, default=8.0)
    parser.add_argument("--highcut", type=float, default=30.0)
    parser.add_argument("--filter-order", type=int, default=4)

    return parser.parse_args()


def parse_seeds(seed_string: str):
    return [int(s.strip()) for s in seed_string.split(",") if s.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_task2_train_data(data_dir: Path):
    x_list, y_list = [], []

    for i in range(1, 11):
        file_path = data_dir / f"subject{i:02d}.npz"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing training file: {file_path}")

        data = np.load(file_path)
        x_list.append(data["x"])
        y_list.append(data["y"])

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return x, y


def bandpass_filter(
    x: np.ndarray,
    fs: float = 250.0,
    lowcut: float = 8.0,
    highcut: float = 30.0,
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
    def __init__(self, x, y, is_train=True, noise_level=0.02):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_train = is_train
        self.noise_level = noise_level

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]

        if self.is_train and self.noise_level > 0:
            noise = torch.randn_like(x_data) * self.noise_level
            x_data = x_data + noise

        return x_data, self.y[idx]


class EEGClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        input_length,
        num_classes=4,
        F1=16,
        D=2,
        F2=32,
        dropout=0.35,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False,
            ),
            nn.BatchNorm2d(F1),

            nn.Conv2d(
                F1,
                F1 * D,
                kernel_size=(num_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                F1 * D,
                F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                F1 * D,
                F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        logits = self.classifier(x)
        return logits


def train_single_model_final(seed, x, y, args, device, num_channels, input_length):
    set_seed(seed)

    train_dataset = EEGDataset(
        x,
        y,
        is_train=True,
        noise_level=0.02,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = EEGClassifier(
        num_channels=num_channels,
        input_length=input_length,
        num_classes=4,
        F1=16,
        D=2,
        F2=32,
        dropout=0.35,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    print(f"\n========== Final training seed {seed} ==========")

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()

            logits = model(bx)
            loss = criterion(logits, by)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(bx)
            train_correct += (torch.argmax(logits, dim=1) == by).sum().item()
            train_total += len(bx)

        scheduler.step()

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Seed {seed} | "
            f"Ep {epoch:03d} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Tr_Acc: {train_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

    final_model_state = copy.deepcopy(
        {k: v.cpu() for k, v in model.state_dict().items()}
    )

    return final_model_state


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    seeds = parse_seeds(args.seeds)
    print(f"Seeds: {seeds}")

    x, y = load_task2_train_data(args.data)

    print(f"Raw x shape: {x.shape}")
    print(f"Raw y shape: {y.shape}")

    x = bandpass_filter(
        x,
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        order=args.filter_order,
    )

    trial_means = x.mean(axis=-1, keepdims=True)
    trial_stds = x.std(axis=-1, keepdims=True)
    x = (x - trial_means) / (trial_stds + 1e-8)

    x = x.astype(np.float32)
    y = y.astype(np.int64)

    num_channels = x.shape[1]
    input_length = x.shape[2]

    print(f"Processed x shape: {x.shape}")
    print(f"Num channels: {num_channels}")
    print(f"Input length: {input_length}")
    print(f"Bandpass: {args.lowcut}-{args.highcut} Hz")
    print("Training mode: FINAL, using all training data. No validation split.")

    model_state_dicts = []

    for seed in seeds:
        state_dict = train_single_model_final(
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
        "training_mode": "final_all_data_no_validation",
        "model_config": {
            "model_type": "eegnet_like",
            "num_channels": num_channels,
            "input_length": input_length,
            "num_classes": 4,
            "F1": 16,
            "D": 2,
            "F2": 32,
            "dropout": 0.35,
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
                "type": "trial_wise_zscore",
            },
        },
    }

    torch.save(checkpoint, args.checkpoint)

    print("\n========== Finished ==========")
    print(f"Saved final ensemble checkpoint to {args.checkpoint}")
    print(f"Number of models in ensemble: {len(model_state_dicts)}")


if __name__ == "__main__":
    main()