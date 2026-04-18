import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from data_paths import resolve_dataset_file

# --- 1. CONFIGURATION (Must match your trained model) ---
MAX_URL_LENGTH = 200
EMBEDDING_DIM = 32
VALID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;="
CHAR_TO_INT = {char: idx + 1 for idx, char in enumerate(VALID_CHARS)}
VOCAB_SIZE = len(CHAR_TO_INT) + 1 
STRESS_RANDOM_SEED = 42

# --- 2. OBFUSCATION FUNCTIONS (The "Stress" in Stress Test) ---
def inject_subdomains(url: str) -> str:
    """Simulates attackers adding trusted keywords as subdomains."""
    subdomains = ["secure.", "verify.", "account-update.", "login.", "auth."]
    if "://" in url:
        parts = url.split("://", 1)
        return f"{parts[0]}://{random.choice(subdomains)}{parts[1]}"
    return f"{random.choice(subdomains)}{url}"

def inject_homoglyphs(url: str) -> str:
    """Simulates visual character spoofing (e.g., 'o' to '0')."""
    substitutions = {'a': '@', 'o': '0', 'l': '1', 'i': '!', 'e': '3', 's': '5'}
    mutated_url = ""
    for char in url:
        # 30% chance to substitute a character if it has a homoglyph
        if char.lower() in substitutions and random.random() < 0.3:
            mutated_url += substitutions[char.lower()]
        else:
            mutated_url += char
    return mutated_url

def inject_delimiters(url: str) -> str:
    """Simulates dash-heavy domains used to evade simple keyword blocks."""
    if "://" in url:
        scheme, rest = url.split("://", 1)
        domain = rest.split("/")[0]
        # Insert random dashes into the domain
        mutated_domain = "-".join(list(domain)) 
        return url.replace(domain, mutated_domain, 1)
    return url


def inject_redirect_wrapper(url: str) -> str:
    """Simulates shortener/redirect wrappers that bury the destination in a parameter."""
    redirectors = [
        "https://t.co/redirect?url=",
        "https://bit.ly/out?target=",
        "https://social.example/redirect?next=",
    ]
    return f"{random.choice(redirectors)}{url}"


def apply_random_obfuscation(url: str) -> str:
    """Randomly applies 1 or 2 obfuscation techniques to a URL."""
    techniques = [
        inject_subdomains,
        inject_homoglyphs,
        inject_delimiters,
        inject_redirect_wrapper,
    ]
    # Pick 1 or 2 random techniques to apply
    num_techniques = random.choice([1, 2])
    chosen_techniques = random.sample(techniques, num_techniques)
    
    mutated_url = url
    for tech in chosen_techniques:
        mutated_url = tech(mutated_url)
    return mutated_url

# --- 3. MODEL DEFINITION ---
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

# --- 4. EVALUATION PIPELINE ---
def tokenize_url(url: str) -> list:
    encoded = [CHAR_TO_INT.get(c, 0) for c in url[:MAX_URL_LENGTH]]
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


def evaluate_model(model, df, obfuscate=False):
    """Runs inference on a dataframe and returns summary metrics."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for _, row in df.iterrows():
            url = str(row['url_raw'])
            label = float(row['label'])
            
            # Apply perturbations if testing robustness
            if obfuscate:
                url = apply_random_obfuscation(url)
                
            input_tensor = torch.tensor([tokenize_url(url)], dtype=torch.long)
            pred_prob = model(input_tensor).item()
            
            all_preds.append(pred_prob)
            all_labels.append(label)
            
    return summarize_metrics(all_labels, all_preds)

def run_stress_test():
    print("Loading PyTorch CNN Model...")
    model = CharCNN()
    model.load_state_dict(torch.load('outputs/char_cnn_model.pth', weights_only=True))
    
    print("Loading unseen Test Dataset (split_test.csv)...")
    test_df = pd.read_csv(resolve_dataset_file('dataset1_binary', 'split_test.csv'))
    
    print("\n--- RUNNING BASELINE (CLEAN) TEST ---")
    clean_metrics = evaluate_model(model, test_df, obfuscate=False)
    print(f"Clean ROC-AUC:  {clean_metrics['roc_auc']:.4f}")
    print(f"Clean Accuracy: {clean_metrics['accuracy']:.4f}")
    print(f"Clean Precision: {clean_metrics['precision_1']:.4f}")
    print(f"Clean Recall:    {clean_metrics['recall_1']:.4f}")
    print(f"Clean F1:        {clean_metrics['f1_1']:.4f}")
    
    print("\n--- RUNNING OBFUSCATION STRESS TEST ---")
    print("Applying synthetic perturbations (homoglyphs, subdomains, delimiters, redirect wrappers)...")
    random.seed(STRESS_RANDOM_SEED)
    stress_metrics = evaluate_model(model, test_df, obfuscate=True)
    print(f"Stress ROC-AUC:  {stress_metrics['roc_auc']:.4f}")
    print(f"Stress Accuracy: {stress_metrics['accuracy']:.4f}")
    print(f"Stress Precision: {stress_metrics['precision_1']:.4f}")
    print(f"Stress Recall:    {stress_metrics['recall_1']:.4f}")
    print(f"Stress F1:        {stress_metrics['f1_1']:.4f}")
    
    print("\n--- ROBUSTNESS DEGRADATION ---")
    print(f"ROC-AUC Drop:  {clean_metrics['roc_auc'] - stress_metrics['roc_auc']:.4f}")
    print(f"Accuracy Drop: {clean_metrics['accuracy'] - stress_metrics['accuracy']:.4f}")
    print(f"F1 Drop:       {clean_metrics['f1_1'] - stress_metrics['f1_1']:.4f}")

    # --- NEW: Save metrics for the plotting script ---
    metrics_file = Path('outputs/stress_test_metrics.json')
    metrics_data = {
        "clean_auc": clean_metrics["roc_auc"],
        "clean_acc": clean_metrics["accuracy"],
        "stress_auc": stress_metrics["roc_auc"],
        "stress_acc": stress_metrics["accuracy"],
        "clean": clean_metrics,
        "stress": stress_metrics,
    }
    
    with metrics_file.open('w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"\n[+] Metrics saved to {metrics_file}")


if __name__ == "__main__":
    run_stress_test()
