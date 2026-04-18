#!/usr/bin/env python3
"""
Train and evaluate baseline phishing URL classifiers using prepared lexical features.

Example:
  .venv/bin/python train_baseline.py --model logreg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_paths import resolve_dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline model(s) on dataset1_binary lexical features."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=resolve_dataset_dir("dataset1_binary"),
        help="Directory containing split_*.csv and lexical_features.csv.",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "random_forest", "svm"],
        default="logreg",
        help="Baseline model to train.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Max iterations for logistic regression and SVM.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees for random forest.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fraction of each split to use (0 < f <= 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and model reproducibility.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("outputs/baseline_metrics.json"),
        help="Where to write JSON metrics.",
    )
    return parser.parse_args()


def load_splits(data_dir: Path, sample_frac: float, random_state: int) -> dict[str, pd.DataFrame]:
    lexical_path = data_dir / "lexical_features.csv"
    if not lexical_path.exists():
        raise FileNotFoundError(f"Missing required file: {lexical_path}")

    lexical = pd.read_csv(lexical_path)
    feature_cols = [
        c for c in lexical.columns if c not in {"label", "label_raw", "url_norm"}
    ]

    splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_path = data_dir / f"split_{split_name}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing required file: {split_path}")

        split_df = pd.read_csv(split_path, usecols=["url_norm", "label"])
        if sample_frac < 1.0:
            split_df = split_df.sample(frac=sample_frac, random_state=random_state)

        merged = split_df.merge(
            lexical[["url_norm"] + feature_cols], on="url_norm", how="left"
        )
        missing_count = int(merged[feature_cols].isna().sum().sum())
        if missing_count > 0:
            raise ValueError(
                f"Split '{split_name}' has {missing_count} missing feature values after merge."
            )

        splits[split_name] = merged

    return {"splits": splits, "feature_cols": feature_cols}


def build_model(
    model_name: str,
    max_iter: int,
    n_estimators: int,
    random_state: int,
) -> Pipeline:
    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=max_iter,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_name == "svm":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")


def extract_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    raise ValueError("Model does not expose predict_proba or decision_function.")


def evaluate_split(
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    y_pred = model.predict(x)
    y_score = extract_scores(model, x)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    positive_report = report.get("1") or report.get("1.0")
    if positive_report is None:
        label_keys = [key for key in report.keys() if key not in {"accuracy", "macro avg", "weighted avg"}]
        if not label_keys:
            raise ValueError("Classification report did not contain class-specific metrics.")
        positive_report = report[sorted(label_keys, key=lambda value: float(value))[-1]]
    auc = float(roc_auc_score(y, y_score))
    return {
        "roc_auc": auc,
        "accuracy": float(report["accuracy"]),
        "precision_1": float(positive_report["precision"]),
        "recall_1": float(positive_report["recall"]),
        "f1_1": float(positive_report["f1-score"]),
        "support_1": int(positive_report["support"]),
        "report": report,
    }


def main() -> None:
    args = parse_args()
    if not (0.0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0, 1].")

    loaded = load_splits(args.data_dir, args.sample_frac, args.random_state)
    splits = loaded["splits"]
    feature_cols = loaded["feature_cols"]

    x_train = splits["train"][feature_cols].to_numpy()
    y_train = splits["train"]["label"].to_numpy()
    x_val = splits["val"][feature_cols].to_numpy()
    y_val = splits["val"]["label"].to_numpy()
    x_test = splits["test"][feature_cols].to_numpy()
    y_test = splits["test"]["label"].to_numpy()

    model = build_model(args.model, args.max_iter, args.n_estimators, args.random_state)
    model.fit(x_train, y_train)

    val_metrics = evaluate_split(model, x_val, y_val)
    test_metrics = evaluate_split(model, x_test, y_test)

    print(f"\nModel: {args.model}")
    print(
        "Val: "
        f"ROC-AUC={val_metrics['roc_auc']:.4f}, "
        f"Acc={val_metrics['accuracy']:.4f}, "
        f"P1={val_metrics['precision_1']:.4f}, "
        f"R1={val_metrics['recall_1']:.4f}, "
        f"F1_1={val_metrics['f1_1']:.4f}"
    )
    print(
        "Test: "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}, "
        f"Acc={test_metrics['accuracy']:.4f}, "
        f"P1={test_metrics['precision_1']:.4f}, "
        f"R1={test_metrics['recall_1']:.4f}, "
        f"F1_1={test_metrics['f1_1']:.4f}"
    )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "data_dir": str(args.data_dir),
        "sample_frac": args.sample_frac,
        "random_state": args.random_state,
        "feature_count": len(feature_cols),
        "split_sizes": {
            k: int(len(v)) for k, v in splits.items()
        },
        "validation": val_metrics,
        "test": test_metrics,
    }
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
