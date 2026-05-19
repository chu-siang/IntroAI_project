import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 2 inference and write a Kaggle submission CSV.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the task2 data directory or directly to test.npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to your saved model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path. Defaults to ./submission.csv",
    )
    return parser.parse_args()


def resolve_test_file(data_path: Path) -> Path:
    if data_path.is_file():
        return data_path

    candidates = [
        data_path / "test.npz",
        data_path / "task2" / "test.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate Task 2 test data. "
        "If you run from part1/task2, use --data data or --data data/test.npz. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def load_test_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    test_file = resolve_test_file(data_path)
    arrays = np.load(test_file, allow_pickle=True)

    if "x" not in arrays:
        raise KeyError(f"Missing 'x' in {test_file}")

    x = arrays["x"]
    ids = arrays["id"] if "id" in arrays else np.arange(len(x), dtype=np.int64)
    return x, ids.astype(np.int64)


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


class EEGClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        input_length,
        num_classes=4,
        F1=8,
        D=2,
        F2=16,
        dropout=0.25,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
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
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
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
        x = x.unsqueeze(1)       # (B, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)         # (B, F2, 1, 1)
        x = x.flatten(1)         # (B, F2)
        x = self.classifier(x)   # (B, 4)
        return x


def load_checkpoint(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_config" not in checkpoint:
        raise KeyError("Missing key 'model_config' in checkpoint")
    if "preprocess" not in checkpoint:
        raise KeyError("Missing key 'preprocess' in checkpoint")
    if "model_state_dicts" not in checkpoint and "model_state_dict" not in checkpoint:
        raise KeyError("Missing model weights in checkpoint")

    return checkpoint


def preprocess_for_inference(x: np.ndarray, checkpoint) -> np.ndarray:
    preprocess = checkpoint["preprocess"]

    if "filter" in preprocess:
        filter_config = preprocess["filter"]
        if filter_config.get("type") == "butter_bandpass":
            x = bandpass_filter(
                x,
                fs=filter_config["fs"],
                lowcut=filter_config["lowcut"],
                highcut=filter_config["highcut"],
                order=filter_config["order"],
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_config}")

    if "normalization" in preprocess:
        norm_config = preprocess["normalization"]

        if norm_config.get("type") == "global_zscore":
            mean = norm_config["mean"]
            std = norm_config["std"]
            x = (x - mean) / (std + 1e-8)
            return x.astype(np.float32)

        elif norm_config.get("type") == "trial_wise_zscore":
            trial_means = x.mean(axis=-1, keepdims=True)
            trial_stds = x.std(axis=-1, keepdims=True)
            x = (x - trial_means) / (trial_stds + 1e-8)
            return x.astype(np.float32)

        raise ValueError(f"Unknown normalization type: {norm_config}")

    if "mean" in preprocess and "std" in preprocess:
        mean = preprocess["mean"]
        std = preprocess["std"]
        x = (x - mean) / (std + 1e-8)
        return x.astype(np.float32)

    raise ValueError(f"Unknown preprocess format: {preprocess}")


def build_one_model(checkpoint, state_dict):
    config = checkpoint["model_config"]
    model = EEGClassifier(
        num_channels=config["num_channels"],
        input_length=config["input_length"],
        num_classes=config["num_classes"],
        F1=config.get("F1", 8),
        D=config.get("D", 2),
        F2=config.get("F2", 16),
        dropout=config.get("dropout", 0.25), # 會自動讀取 checkpoint 中的 0.50
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_models(checkpoint):
    if "model_state_dicts" in checkpoint:
        return [
            build_one_model(checkpoint, state_dict)
            for state_dict in checkpoint["model_state_dicts"]
        ]
    return [build_one_model(checkpoint, checkpoint["model_state_dict"])]


def predict(models, x: np.ndarray) -> np.ndarray:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    batch_size = 64
    predictions = []

    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            batch_x = x_tensor[start:start + batch_size]
            logits_sum = None

            for model in models:
                logits = model(batch_x)
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits

            logits_avg = logits_sum / len(models)
            batch_pred = torch.argmax(logits_avg, dim=1)
            predictions.append(batch_pred.cpu().numpy())

    pred = np.concatenate(predictions, axis=0)
    return pred.astype(np.int64)


def validate_predictions(pred: np.ndarray, num_examples: int) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.shape != (num_examples,):
        raise ValueError(f"Expected predictions with shape ({num_examples},), got {pred.shape}")
    if not np.issubdtype(pred.dtype, np.integer):
        raise TypeError(f"Predictions must be integers, got dtype {pred.dtype}")
    if np.any((pred < 0) | (pred > 3)):
        raise ValueError("Predicted labels must be integers in {0, 1, 2, 3}")
    return pred.astype(np.int64)


def write_submission(rows: Iterable[Tuple[int, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    x_test, ids = load_test_data(args.data)
    checkpoint = load_checkpoint(args.checkpoint)

    x_test = preprocess_for_inference(x_test, checkpoint)
    models = build_models(checkpoint)

    print(f"Loaded {len(models)} model(s) for inference.")
    pred = predict(models, x_test)
    pred = validate_predictions(pred, len(ids))

    rows = [(int(sample_id), int(label)) for sample_id, label in zip(ids, pred)]
    write_submission(rows, args.output)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()