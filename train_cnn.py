#!/usr/bin/env python3
"""Train and evaluate character-level CNN detectors on URL datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from data_paths import resolve_dataset_dir

MAX_URL_LENGTH = 200
EMBEDDING_DIM = 32
VALID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;="
CHAR_TO_INT = {char: idx + 1 for idx, char in enumerate(VALID_CHARS)}
VOCAB_SIZE = len(CHAR_TO_INT) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the CharCNN on URL datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=resolve_dataset_dir("dataset1_binary"),
        help="Directory containing split_train.csv, split_val.csv, and split_test.csv.",
    )
    parser.add_argument(
        "--task",
        choices=["auto", "binary", "multiclass"],
        default="auto",
        help="Training task type. Defaults to auto-detect from labels.",
    )
    parser.add_argument(
        "--url-column",
        default="auto",
        help="URL column to use for training. Defaults to auto, which prefers url_norm, then url_raw, then url_canon.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("outputs/char_cnn_model.pth"),
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("outputs/cnn_metrics.json"),
        help="Path to save JSON metrics.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        return torch.device("mps")
    if choice == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_url(url: str) -> list[int]:
    encoded = [CHAR_TO_INT.get(char, 0) for char in str(url)[:MAX_URL_LENGTH]]
    if len(encoded) < MAX_URL_LENGTH:
        encoded.extend([0] * (MAX_URL_LENGTH - len(encoded)))
    return encoded


def resolve_task_metadata(data_dir: Path, requested_task: str) -> tuple[str, dict[str, int], list[str]]:
    train_labels = pd.read_csv(data_dir / "split_train.csv", usecols=["label"])["label"]
    inferred_binary = pd.api.types.is_numeric_dtype(train_labels) and set(train_labels.dropna().unique()).issubset({0, 1, 0.0, 1.0})

    if requested_task == "auto":
        task_type = "binary" if inferred_binary else "multiclass"
    else:
        task_type = requested_task

    if task_type == "binary":
        return task_type, {}, ["0", "1"]

    mapping_path = data_dir / "label_mapping.json"
    if mapping_path.exists():
        raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        label_to_index = {str(label): int(index) for label, index in raw_mapping.items()}
        class_names = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
        return task_type, label_to_index, class_names

    class_names = sorted(str(value) for value in train_labels.dropna().unique())
    label_to_index = {label: index for index, label in enumerate(class_names)}
    return task_type, label_to_index, class_names


def resolve_url_column(data_dir: Path, requested_column: str) -> str:
    train_columns = pd.read_csv(data_dir / "split_train.csv", nrows=0).columns.tolist()
    if requested_column != "auto":
        if requested_column not in train_columns:
            raise ValueError(f"Requested URL column '{requested_column}' was not found in {data_dir / 'split_train.csv'}.")
        return requested_column

    for candidate in ("url_norm", "url_raw", "url_canon"):
        if candidate in train_columns:
            return candidate
    raise ValueError("No supported URL column found. Expected one of: url_norm, url_raw, url_canon.")


def auto_adjust_output_paths(args: argparse.Namespace, task_type: str) -> None:
    if task_type != "multiclass":
        return
    if args.model_out == Path("outputs/char_cnn_model.pth"):
        args.model_out = Path("outputs/char_cnn_multiclass_model.pth")
    if args.metrics_out == Path("outputs/cnn_metrics.json"):
        args.metrics_out = Path("outputs/cnn_multiclass_metrics.json")


class URLDataset(Dataset):
    def __init__(self, csv_file: Path, task_type: str, label_to_index: dict[str, int], url_column: str):
        self.url_column = url_column
        self.data = pd.read_csv(csv_file, usecols=[url_column, "label"])
        self.task_type = task_type
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        tokens = tokenize_url(str(row[self.url_column]))
        raw_label = row["label"]
        if self.task_type == "binary":
            label = float(raw_label)
            return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

        label = int(self.label_to_index[str(raw_label)])
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class CharCNN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=0,
        )
        self.conv1 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1 if num_classes == 1 else num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        if self.num_classes == 1:
            return torch.sigmoid(x).squeeze(-1)
        return x


def positive_report(report: dict) -> dict:
    result = report.get("1") or report.get("1.0")
    if result is not None:
        return result
    label_keys = [key for key in report if key not in {"accuracy", "macro avg", "weighted avg"}]
    if not label_keys:
        raise ValueError("Classification report did not contain class-specific metrics.")
    return report[sorted(label_keys, key=lambda value: float(value))[-1]]


def summarize_binary_metrics(labels: list[float], scores: list[float]) -> dict:
    binary_preds = [1 if score >= 0.5 else 0 for score in scores]
    report = classification_report(labels, binary_preds, output_dict=True, zero_division=0)
    pos = positive_report(report)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, binary_preds)),
        "precision_1": float(pos["precision"]),
        "recall_1": float(pos["recall"]),
        "f1_1": float(pos["f1-score"]),
        "support_1": int(pos["support"]),
        "report": report,
    }


def summarize_multiclass_metrics(labels: list[int], score_rows: list[list[float]], class_names: list[str]) -> dict:
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(score_rows, dtype=float)
    y_pred = np.argmax(y_score, axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    try:
        auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None

    return {
        "roc_auc_ovr_macro": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "report": report,
    }


def evaluate_model(
    model: CharCNN,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
    class_names: list[str],
) -> dict:
    model.eval()
    if task_type == "binary":
        all_scores: list[float] = []
        all_labels: list[float] = []
        with torch.no_grad():
            for urls, labels in loader:
                urls = urls.to(device)
                preds = model(urls).detach().cpu().numpy().tolist()
                all_scores.extend(float(value) for value in preds)
                all_labels.extend(float(value) for value in labels.numpy().tolist())
        return summarize_binary_metrics(all_labels, all_scores)

    all_score_rows: list[list[float]] = []
    all_labels_int: list[int] = []
    with torch.no_grad():
        for urls, labels in loader:
            urls = urls.to(device)
            logits = model(urls)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().tolist()
            all_score_rows.extend([[float(value) for value in row] for row in probs])
            all_labels_int.extend(int(value) for value in labels.numpy().tolist())
    return summarize_multiclass_metrics(all_labels_int, all_score_rows, class_names)


def build_loader(
    csv_path: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    task_type: str,
    label_to_index: dict[str, int],
    url_column: str,
) -> tuple[URLDataset, DataLoader]:
    dataset = URLDataset(csv_path, task_type=task_type, label_to_index=label_to_index, url_column=url_column)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, loader


def primary_val_metric(metrics: dict, task_type: str) -> float:
    if task_type == "binary":
        return float(metrics["roc_auc"])
    auc = metrics.get("roc_auc_ovr_macro")
    if auc is not None:
        return float(auc)
    return float(metrics["accuracy"])


def main() -> None:
    args = parse_args()
    set_seeds(args.random_state)
    device = select_device(args.device)
    data_dir = args.data_dir

    task_type, label_to_index, class_names = resolve_task_metadata(data_dir, args.task)
    url_column = resolve_url_column(data_dir, args.url_column)
    auto_adjust_output_paths(args, task_type)

    train_dataset, train_loader = build_loader(
        data_dir / "split_train.csv",
        args.batch_size,
        True,
        args.num_workers,
        task_type,
        label_to_index,
        url_column,
    )
    val_dataset, val_loader = build_loader(
        data_dir / "split_val.csv",
        args.batch_size,
        False,
        args.num_workers,
        task_type,
        label_to_index,
        url_column,
    )
    test_dataset, test_loader = build_loader(
        data_dir / "split_test.csv",
        args.batch_size,
        False,
        args.num_workers,
        task_type,
        label_to_index,
        url_column,
    )

    num_classes = 1 if task_type == "binary" else len(class_names)
    model = CharCNN(num_classes=num_classes).to(device)
    criterion: nn.Module
    if task_type == "binary":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []

    print(f"Training on {device} with dataset: {data_dir} ({task_type}) using {url_column}")
    if task_type == "multiclass":
        print(f"Classes: {class_names}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for urls, labels in train_loader:
            urls = urls.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(urls)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(urls.size(0))

        train_loss = total_loss / len(train_dataset)
        val_metrics = evaluate_model(model, val_loader, device, task_type, class_names)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
        }
        if task_type == "binary":
            epoch_record["val_roc_auc"] = val_metrics["roc_auc"]
            epoch_record["val_f1_1"] = val_metrics["f1_1"]
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Loss={train_loss:.4f} | "
                f"Val ROC-AUC={val_metrics['roc_auc']:.4f} | "
                f"Val Acc={val_metrics['accuracy']:.4f} | "
                f"Val F1={val_metrics['f1_1']:.4f}"
            )
        else:
            epoch_record["val_roc_auc_ovr_macro"] = val_metrics["roc_auc_ovr_macro"]
            epoch_record["val_f1_macro"] = val_metrics["f1_macro"]
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Loss={train_loss:.4f} | "
                f"Val ROC-AUC(OVR Macro)={val_metrics['roc_auc_ovr_macro']:.4f} | "
                f"Val Acc={val_metrics['accuracy']:.4f} | "
                f"Val F1(Macro)={val_metrics['f1_macro']:.4f}"
            )
        history.append(epoch_record)

        val_score = primary_val_metric(val_metrics, task_type)
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    validation_metrics = evaluate_model(model, val_loader, device, task_type, class_names)
    test_metrics = evaluate_model(model, test_loader, device, task_type, class_names)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"Best model saved to: {args.model_out}")

    payload = {
        "model": "char_cnn",
        "task_type": task_type,
        "num_classes": num_classes,
        "class_names": class_names,
        "label_to_index": label_to_index if task_type == "multiclass" else None,
        "data_dir": str(data_dir),
        "url_column": url_column,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "random_state": args.random_state,
        "split_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "best_validation_metric": best_val_score,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if task_type == "binary":
        print(
            "Test: "
            f"ROC-AUC={test_metrics['roc_auc']:.4f}, "
            f"Acc={test_metrics['accuracy']:.4f}, "
            f"P1={test_metrics['precision_1']:.4f}, "
            f"R1={test_metrics['recall_1']:.4f}, "
            f"F1_1={test_metrics['f1_1']:.4f}"
        )
    else:
        auc_text = f"{test_metrics['roc_auc_ovr_macro']:.4f}" if test_metrics["roc_auc_ovr_macro"] is not None else "n/a"
        print(
            "Test: "
            f"ROC-AUC(OVR Macro)={auc_text}, "
            f"Acc={test_metrics['accuracy']:.4f}, "
            f"F1(Macro)={test_metrics['f1_macro']:.4f}, "
            f"F1(Weighted)={test_metrics['f1_weighted']:.4f}"
        )
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
