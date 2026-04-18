#!/usr/bin/env python3
"""
Evaluate the CNN model trained on dataset 1 against unseen datasets (2 and 3)
to measure cross-dataset generalization and distribution shift.
"""

import json
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from pathlib import Path

from data_paths import resolve_dataset_file

# --- 1. CONFIGURATION (Must match train_cnn.py exactly) ---
MAX_URL_LENGTH = 200
EMBEDDING_DIM = 32
VALID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;="
CHAR_TO_INT = {char: idx + 1 for idx, char in enumerate(VALID_CHARS)}
VOCAB_SIZE = len(CHAR_TO_INT) + 1 

# --- 2. MODEL DEFINITION ---
class CharCNN(nn.Module):
    def __init__(self):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(-1)

# --- 3. EVALUATION HELPER ---
def tokenize_url(url: str) -> list:
    encoded = [CHAR_TO_INT.get(c, 0) for c in str(url)[:MAX_URL_LENGTH]]
    if len(encoded) < MAX_URL_LENGTH:
        encoded += [0] * (MAX_URL_LENGTH - len(encoded))
    return encoded

def summarize_metrics(labels, scores):
    binary_preds = [1 if score >= 0.5 else 0 for score in scores]
    report = classification_report(labels, binary_preds, output_dict=True, zero_division=0)
    positive_report = report.get("1") or report.get("1.0")
    if positive_report is None:
        label_keys = [key for key in report.keys() if key not in {"accuracy", "macro avg", "weighted avg"}]
        if not label_keys:
            raise ValueError("Classification report did not contain class-specific metrics.")
        positive_report = report[sorted(label_keys, key=lambda value: float(value))[-1]]
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, binary_preds)),
        "precision_1": float(positive_report["precision"]),
        "recall_1": float(positive_report["recall"]),
        "f1_1": float(positive_report["f1-score"]),
        "support_1": int(positive_report["support"]),
    }


def evaluate_on_dataset(model, csv_path):
    """Runs inference on a given CSV file and returns summary metrics."""
    if not Path(csv_path).exists():
        print(f"  [!] Could not find {csv_path}. Skipping.")
        return None

    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    if 'url_raw' not in df.columns or 'label' not in df.columns:
        print(f"  [!] {csv_path} is missing 'url_raw' or 'label' columns. Skipping.")
        return None

    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for _, row in df.iterrows():
            url = str(row['url_raw'])
            label = float(row['label'])
            
            input_tensor = torch.tensor([tokenize_url(url)], dtype=torch.long)
            pred_prob = model(input_tensor).item()
            
            all_preds.append(pred_prob)
            all_labels.append(label)
            
    return summarize_metrics(all_labels, all_preds)

# --- 4. MAIN EXECUTION ---
def main():
    model_path = Path('outputs/char_cnn_model.pth')
    if not model_path.exists():
        print(f"Error: Trained model not found at {model_path}.")
        print("Please run 'python train_cnn.py' first to generate the model weights.")
        return

    print("Loading PyTorch CNN Model (Trained on Dataset 1)...")
    model = CharCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Define the datasets to test against. 
    # Adjust the filename ('split_test.csv' or 'data.csv') based on what is actually in your folders!
    target_datasets = {
        "Dataset 2 (Binary)": resolve_dataset_file("dataset2_binary", "split_test.csv"),
        "Dataset 3 (Binary)": resolve_dataset_file("dataset3_binary", "split_test.csv"),
    }
    results = {}

    print("\n--- RUNNING CROSS-DATASET EVALUATION ---")
    for name, path in target_datasets.items():
        print(f"\nEvaluating on {name}...")
        metrics = evaluate_on_dataset(model, path)
        
        if metrics is not None:
            results[name] = {
                "path": str(path),
                **metrics,
            }
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision_1']:.4f}")
            print(f"  Recall:    {metrics['recall_1']:.4f}")
            print(f"  F1:        {metrics['f1_1']:.4f}")

    if results:
        output_path = Path("outputs/cross_dataset_metrics.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved cross-dataset metrics to {output_path}")

if __name__ == "__main__":
    main()
