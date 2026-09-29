"""Train the Cover Type classifier and export all inference artifacts.

Refactor of ``notebooks/covtype_training_and_evaluation.ipynb`` with identical:
  * data source      : sklearn.datasets.fetch_covtype(as_frame=True)
  * target encoding  : Cover_Type (1..7) -> Cover_Type - 1 (0..6)
  * split            : 80 / 10 / 10, stratified, random_state=11
  * preprocessing    : ColumnTransformer(StandardScaler on numeric, passthrough on one-hot), fit on train only
  * class weights    : 1 / sqrt(count), normalized by mean
  * model            : CoverTypeClassifier (see covtype/model.py)
  * optimization     : Adam(lr=1e-3), weighted CrossEntropyLoss, 100 epochs, batch 1024, seed 11
  * final weights    : last epoch (use --restore-best to keep the best-val-loss epoch instead)

Exports (default ./artifacts):
  * covtype_model.pth       — model state_dict
  * preprocessor.joblib     — fitted ColumnTransformer
  * metadata.json           — feature order, label mapping, class names, feature stats, config,
                              metrics, and one representative test row per class (UI presets)

Usage:
    python train_and_export.py
    python train_and_export.py --epochs 30 --restore-best
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_covtype
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from covtype.constants import (
    BINARY_FEATURES,
    CLASS_NAMES,
    FEATURE_NAMES,
    NUM_CLASSES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_OFFSET,
)
from covtype.model import CoverTypeClassifier

MODEL_FILENAME = "covtype_model.pth"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 1024
    lr: float = 1e-3
    seed: int = RANDOM_STATE
    split_seed: int = RANDOM_STATE
    restore_best: bool = False


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    """Load Covertype and shift labels 1..7 -> 0..6 (notebook cells 1 & 7)."""
    bunch = fetch_covtype(as_frame=True)
    df = pd.concat([bunch["data"], bunch["target"]], axis=1)
    df[TARGET_COLUMN] = df[TARGET_COLUMN] - TARGET_OFFSET

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    if feature_cols != FEATURE_NAMES:
        raise RuntimeError("Feature columns differ from covtype.constants.FEATURE_NAMES")
    return df


def split_dataset(df: pd.DataFrame, seed: int) -> DataSplits:
    """80/10/10 stratified split (notebook cell 9)."""
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=seed, stratify=y_val_test
    )
    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def compute_class_weights(y_train: pd.Series) -> torch.Tensor:
    """1/sqrt(count), normalized by the mean (notebook cell 10)."""
    counts = np.bincount(y_train.to_numpy(), minlength=NUM_CLASSES)
    weights = 1.0 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """StandardScaler on continuous features, passthrough on one-hot (notebook cell 11)."""
    binary_cols = [c for c in columns if c.startswith(("Wilderness_Area", "Soil_Type"))]
    numeric_cols = [c for c in columns if c not in binary_cols]
    if numeric_cols != NUMERIC_FEATURES or binary_cols != BINARY_FEATURES:
        raise RuntimeError("Column grouping differs from covtype.constants")

    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", binary_cols),
        ],
        verbose_feature_names_out=False,
    )


def to_tensors(
    preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series
) -> tuple[torch.Tensor, torch.Tensor]:
    X_np = np.asarray(preprocessor.transform(X), dtype=np.float32)
    y_np = y.to_numpy().astype(np.int64)  # copy -> writable array
    return torch.from_numpy(X_np), torch.from_numpy(y_np)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Full-batch-on-device training loop (notebook cell 14)."""
    n = X_train.shape[0]
    train_acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)
    val_acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(cfg.epochs):
        start = time.perf_counter()

        # ---- Train ----
        model.train()
        train_acc_metric.reset()
        running_loss = torch.zeros((), device=device)

        perm = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            X_batch, y_batch = X_train[idx], y_train[idx]

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach() * y_batch.size(0)
            train_acc_metric.update(logits.detach().argmax(dim=1), y_batch)

        train_loss = (running_loss / n).item()
        train_acc = train_acc_metric.compute().item()

        # ---- Validation ----
        model.eval()
        val_acc_metric.reset()
        with torch.inference_mode():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()
            val_acc_metric.update(val_logits.argmax(dim=1), y_val)
            val_acc = val_acc_metric.compute().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            if cfg.restore_best:
                best_state = copy.deepcopy(model.state_dict())

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
            f"{elapsed:.2f}s"
        )

    if cfg.restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best-validation-loss weights from epoch {best_epoch} ({best_val_loss:.4f})")

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_epoch,
        "exported_epoch": best_epoch if cfg.restore_best else cfg.epochs,
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_proba(
    model: nn.Module, X: torch.Tensor, device: torch.device, batch_size: int = 16_384
) -> np.ndarray:
    """Batched inference -> softmax probabilities (notebook cell 17)."""
    model.eval()
    probs: list[torch.Tensor] = []
    for i in range(0, X.size(0), batch_size):
        logits = model(X[i : i + batch_size].to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def evaluate(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    y_pred = y_proba.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(NUM_CLASSES), zero_division=0
    )
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": [
            {
                "class_index": k,
                "class_name": CLASS_NAMES[k],
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k in range(NUM_CLASSES)
        ],
    }


def select_examples(
    X_raw: pd.DataFrame, X_scaled: np.ndarray, y_true: np.ndarray, y_proba: np.ndarray
) -> list[dict[str, Any]]:
    """Pick one representative, correctly classified test row per class.

    Representative = closest (in scaled numeric space) to the per-class median.
    Used by the web UI as "load sample" presets.
    """
    y_pred = y_proba.argmax(axis=1)
    n_num = len(NUMERIC_FEATURES)
    examples: list[dict[str, Any]] = []
    for k in range(NUM_CLASSES):
        mask = (y_true == k) & (y_pred == k)
        if not mask.any():
            mask = y_true == k
        idx = np.flatnonzero(mask)
        Z = X_scaled[idx, :n_num]
        center = np.median(Z, axis=0)
        best = int(idx[np.argmin(((Z - center) ** 2).sum(axis=1))])
        row = X_raw.iloc[best]
        examples.append(
            {
                "class_index": k,
                "cover_type": k + TARGET_OFFSET,
                "class_name": CLASS_NAMES[k],
                "model_confidence": float(y_proba[best, k]),
                "features": {c: float(row[c]) for c in FEATURE_NAMES},
            }
        )
    return examples


def feature_stats(X_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Training-split ranges for numeric features (used for UI defaults / sanity checks)."""
    desc = X_train[NUMERIC_FEATURES].describe().T
    return {
        name: {
            "min": float(row["min"]),
            "max": float(row["max"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "median": float(row["50%"]),
        }
        for name, row in desc.iterrows()
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_artifacts(
    output_dir: Path,
    model: nn.Module,
    preprocessor: ColumnTransformer,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, output_dir / MODEL_FILENAME)
    joblib.dump(preprocessor, output_dir / PREPROCESSOR_FILENAME)
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nArtifacts written to {output_dir.resolve()}:")
    for name in (MODEL_FILENAME, PREPROCESSOR_FILENAME, METADATA_FILENAME):
        size_kb = (output_dir / name).stat().st_size / 1024
        print(f"  - {name:<22} {size_kb:8.1f} KB")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed, help="torch seed (notebook: 11)")
    parser.add_argument("--restore-best", action="store_true",
                        help="Export best-val-loss weights instead of last-epoch weights")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        restore_best=args.restore_best,
    )
    device = resolve_device(args.device)
    print(f"Device: {device} | torch {torch.__version__} | sklearn {sklearn.__version__}")

    # ---- Data ----
    df = load_dataset()
    splits = split_dataset(df, cfg.split_seed)
    print(f"Split sizes | train {len(splits.X_train):,} | val {len(splits.X_val):,} | test {len(splits.X_test):,}")

    class_weights = compute_class_weights(splits.y_train)
    print(f"Class weights: {[round(float(w), 4) for w in class_weights]}")

    preprocessor = build_preprocessor(list(splits.X_train.columns))
    preprocessor.fit(splits.X_train)  # fit on train only

    X_train_t, y_train_t = to_tensors(preprocessor, splits.X_train, splits.y_train)
    X_val_t, y_val_t = to_tensors(preprocessor, splits.X_val, splits.y_val)
    X_test_t, y_test_t = to_tensors(preprocessor, splits.X_test, splits.y_test)

    # ---- Model (notebook cell 13) ----
    torch.manual_seed(cfg.seed)
    input_size = X_train_t.size(1)

    X_train_t, y_train_t = X_train_t.to(device), y_train_t.to(device)
    X_val_t, y_val_t = X_val_t.to(device), y_val_t.to(device)

    model = CoverTypeClassifier(input_size, NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ---- Train ----
    train_out = train(model, X_train_t, y_train_t, X_val_t, y_val_t, loss_fn, optimizer, cfg, device)

    # ---- Evaluate ----
    test_proba = predict_proba(model, X_test_t, device)
    y_test_np = y_test_t.numpy()
    metrics = {
        "validation": evaluate(y_val_t.cpu().numpy(), predict_proba(model, X_val_t, device)),
        "test": evaluate(y_test_np, test_proba),
    }
    examples = select_examples(splits.X_test, X_test_t.numpy(), y_test_np, test_proba)
    for split_name, m in metrics.items():
        print(
            f"{split_name:<10} | acc {m['accuracy']:.4f} | bal_acc {m['balanced_accuracy']:.4f} | "
            f"macro_f1 {m['macro_f1']:.4f} | mcc {m['mcc']:.4f}"
        )

    # ---- Export ----
    metadata: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {"torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "model": {
            "class": "CoverTypeClassifier",
            "input_size": int(input_size),
            "num_classes": NUM_CLASSES,
            "hidden_layers": [128, 128, 128, 64, 32],
            "weights_file": MODEL_FILENAME,
        },
        "preprocessing": {
            "preprocessor_file": PREPROCESSOR_FILENAME,
            "feature_names": FEATURE_NAMES,
            "numeric_features": NUMERIC_FEATURES,
            "binary_features": BINARY_FEATURES,
            "numeric_transform": "StandardScaler (fit on train split)",
            "binary_transform": "passthrough",
            "output_feature_order": list(preprocessor.get_feature_names_out()),
        },
        "target": {
            "column": TARGET_COLUMN,
            "encoding": "class_index = Cover_Type - 1",
            "offset": TARGET_OFFSET,
            "classes": [
                {"class_index": k, "cover_type": k + TARGET_OFFSET, "class_name": CLASS_NAMES[k]}
                for k in range(NUM_CLASSES)
            ],
        },
        "feature_stats": feature_stats(splits.X_train),
        "class_weights": [float(w) for w in class_weights],
        "training": {
            **asdict(cfg),
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss(weight=1/sqrt(count), mean-normalized)",
            "device": str(device),
            "best_val_loss": train_out["best_val_loss"],
            "best_val_loss_epoch": train_out["best_val_loss_epoch"],
            "exported_epoch": train_out["exported_epoch"],
        },
        "history": train_out["history"],
        "metrics": metrics,
        "examples": examples,
    }
    export_artifacts(args.output_dir, model, preprocessor, metadata)


if __name__ == "__main__":
    main()
