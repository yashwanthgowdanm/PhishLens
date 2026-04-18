#!/usr/bin/env python3
"""Serve the phishing UI and a PyTorch CNN-backed /api/classify endpoint."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent
BINARY_CNN_MODEL_PATH = ROOT_DIR / "outputs" / "char_cnn_model.pth"
MULTICLASS_CNN_MODEL_PATH = ROOT_DIR / "outputs" / "char_cnn_multiclass_model.pth"
MULTICLASS_LABEL_MAPPING_PATH = ROOT_DIR / "dataset1_multiclass" / "label_mapping.json"
DEFAULT_MULTICLASS_CLASS_NAMES = ["benign", "defacement", "malware", "phishing"]

# --- CNN CONFIGURATION (Must match training script exactly) ---
MAX_URL_LENGTH = 200
EMBEDDING_DIM = 32
VALID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;="
CHAR_TO_INT = {char: idx + 1 for idx, char in enumerate(VALID_CHARS)}
VOCAB_SIZE = len(CHAR_TO_INT) + 1 

SUSPICIOUS_TOKENS = (
    "login", "verify", "update", "secure", "bank", "signin",
    "password", "confirm", "invoice", "support", "auth", "wallet",
    "recovery", "unlock", "pay",
)

FEATURE_DESCRIPTIONS = {
    "url_len": "Long URL structure increases ambiguity.",
    "host_len": "Unusually long host value.",
    "path_len": "Deep path resembles obfuscated routing.",
    "query_len": "Large query payload can hide redirects/tokens.",
    "num_digits": "High digit density is common in malicious links.",
    "num_specials": "Heavy symbol usage can obscure intent.",
    "ratio_digits": "Digit-heavy composition raises risk.",
    "ratio_specials": "Special-character ratio is elevated.",
    "count_dot": "Many dotted segments can mimic trusted domains.",
    "count_dash": "Dash-heavy hosts are frequently abused.",
    "count_at": "@ can hide the real destination host.",
    "count_qmark": "Multiple query markers are suspicious.",
    "count_amp": "Many parameters increase obfuscation surface.",
    "count_eq": "Many key/value parameters in URL.",
    "subdomain_count": "Multiple subdomains may imitate trusted brands.",
    "is_https": "No HTTPS encryption.",
    "is_domain_ip": "Raw IP host instead of a registered domain.",
    "suspicious_token_hits": "Contains phishing-associated keywords.",
    "entropy": "High character entropy resembles random strings.",
}

RISKY_TLDS = {"zip", "mov", "top", "xyz", "work", "support", "click", "country", "gq", "cf", "tk", "ml"}
TRUSTED_DOMAIN_SUFFIXES = {
    "asu.edu",
    "ebay.com",
    "google.com",
    "mpb.com",
    "villasinudaipur.co.in",
    "youtube.com",
}
COMMON_SAFE_TLDS = {"com", "org", "net", "edu", "gov", "us", "in", "co"}
DEFACEMENT_HINT_TOKENS = ("hacked", "deface", "defaced", "owned", "mirror")
MALWARE_HINT_TOKENS = (
    "download", "install", "setup", "payload", "patch", "driver", "codec",
    "launcher", "crack", "exe", "scr", "msi", "apk", "dll",
)
PHISHING_HINT_TOKENS = (
    "login", "verify", "signin", "account", "auth", "secure", "bank",
    "password", "confirm", "recovery", "session", "wallet", "pay",
)

MODEL_BUNDLES: dict[str, dict | None] = {"binary": None, "multiclass": None}

# --- PYTORCH MODEL DEFINITION ---
class CharCNN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super(CharCNN, self).__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1 if num_classes == 1 else num_classes)

    def forward(self, x):
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


def load_multiclass_class_names() -> list[str]:
    if MULTICLASS_LABEL_MAPPING_PATH.exists():
        payload = json.loads(MULTICLASS_LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
        return [label for label, _ in sorted(payload.items(), key=lambda item: int(item[1]))]
    return list(DEFAULT_MULTICLASS_CLASS_NAMES)


def load_cnn_model() -> dict:
    """Loads the pre-trained PyTorch weights."""
    model = CharCNN()
    if BINARY_CNN_MODEL_PATH.exists():
        model.load_state_dict(torch.load(BINARY_CNN_MODEL_PATH, weights_only=True))
        model.eval() # Set to evaluation mode
        print("Successfully loaded PyTorch CNN weights.")
    else:
        print(f"WARNING: Model weights not found at {BINARY_CNN_MODEL_PATH}. Using untrained random weights!")
        model.eval()

    return {
        "model": model,
        "threshold": 0.50, # Standard threshold for Sigmoid output
        "model_name": "PyTorch CharCNN Track",
    }


def load_multiclass_cnn_model() -> dict:
    class_names = load_multiclass_class_names()
    model = CharCNN(num_classes=len(class_names))
    if MULTICLASS_CNN_MODEL_PATH.exists():
        model.load_state_dict(torch.load(MULTICLASS_CNN_MODEL_PATH, weights_only=True))
        model.eval()
        print("Successfully loaded PyTorch multiclass CNN weights.")
    else:
        raise FileNotFoundError(f"Multiclass model weights not found at {MULTICLASS_CNN_MODEL_PATH}")

    return {
        "model": model,
        "class_names": class_names,
        "model_name": "PyTorch CharCNN Multiclass Track",
    }


def normalize_url(raw_url: str) -> str:
    raw = raw_url.strip()
    if not raw:
        return raw
    if "://" not in raw:
        return f"http://{raw}"
    return raw


def is_ip_host(host: str) -> int:
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def encode_url_tensor(normalized_url: str) -> torch.Tensor:
    encoded_url = [CHAR_TO_INT.get(c, 0) for c in normalized_url[:MAX_URL_LENGTH]]
    if len(encoded_url) < MAX_URL_LENGTH:
        encoded_url += [0] * (MAX_URL_LENGTH - len(encoded_url))
    return torch.tensor([encoded_url], dtype=torch.long)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def extract_lexical_features(url: str) -> dict[str, float]:
    """Kept for UI Explainability and heuristic fallback mixing."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    url_lower = url.lower()

    host_parts = [part for part in host.split(".") if part]
    url_len = len(url)
    num_digits = sum(ch.isdigit() for ch in url)
    num_letters = sum(ch.isalpha() for ch in url)
    num_specials = sum(not ch.isalnum() for ch in url)

    features: dict[str, float] = {
        "url_len": float(url_len),
        "host_len": float(len(host)),
        "path_len": float(len(path)),
        "query_len": float(len(query)),
        "num_digits": float(num_digits),
        "num_letters": float(num_letters),
        "num_specials": float(num_specials),
        "ratio_digits": float(num_digits / url_len) if url_len else 0.0,
        "ratio_specials": float(num_specials / url_len) if url_len else 0.0,
        "count_dot": float(url.count(".")),
        "count_slash": float(url.count("/")),
        "count_dash": float(url.count("-")),
        "count_underscore": float(url.count("_")),
        "count_at": float(url.count("@")),
        "count_qmark": float(url.count("?")),
        "count_amp": float(url.count("&")),
        "count_eq": float(url.count("=")),
        "subdomain_count": float(max(0, len(host_parts) - 2)),
        "tld_len": float(len(host_parts[-1])) if host_parts else 0.0,
        "is_https": float(parsed.scheme.lower() == "https"),
        "is_domain_ip": float(is_ip_host(host)),
        "suspicious_token_hits": float(sum(url_lower.count(token) for token in SUSPICIOUS_TOKENS)),
        "entropy": float(shannon_entropy(url_lower)),
    }
    return features


def compute_rule_risk(url: str, features: dict[str, float]) -> int:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    tld = host.split(".")[-1] if "." in host else ""
    score = 0

    if features["is_domain_ip"] > 0: score += 22
    if features["count_at"] > 0: score += 15
    if "xn--" in host: score += 18
    if features["is_https"] == 0: score += 8
    if features["url_len"] > 80: score += 10
    if features["subdomain_count"] >= 2: score += 12
    if tld in RISKY_TLDS: score += 10
    
    rule_token_hits = suspicious_hits_in_path_or_query(url)
    if rule_token_hits > 0: score += min(24, int(6 * rule_token_hits))
    if (features["count_dash"] + features["count_underscore"]) > 6: score += 9
    if features["query_len"] > 30: score += 6
    if features["entropy"] >= 4.2: score += 8

    return max(0, min(100, int(score)))


def suspicious_hits_in_path_or_query(url: str) -> int:
    parsed = urlparse(url)
    text = f"{parsed.path} {parsed.query}".lower()
    return int(sum(text.count(token) for token in SUSPICIOUS_TOKENS))


def humanize_label(label: str) -> str:
    return str(label).replace("_", " ").title()


def match_trusted_suffix(host: str) -> str | None:
    host = host.lower().strip(".")
    for suffix in sorted(TRUSTED_DOMAIN_SUFFIXES, key=len, reverse=True):
        if host == suffix or host.endswith(f".{suffix}"):
            return suffix
    return None


def is_trusted_https_context(url: str, features: dict[str, float], rule_token_hits: int) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    trusted_suffix = match_trusted_suffix(host)
    if trusted_suffix is None:
        return None
    if parsed.scheme.lower() != "https":
        return None
    if features["is_domain_ip"] > 0 or features["count_at"] > 0:
        return None
    if "xn--" in host:
        return None
    if features["subdomain_count"] >= 3:
        return None
    if rule_token_hits >= 3:
        return None
    return trusted_suffix


def collect_feature_signals(
    features: dict[str, float],
    rule_token_hits: int,
    trusted_suffix: str | None = None,
) -> list[dict]:
    signals: list[dict] = []

    if features["is_domain_ip"] > 0:
        signals.append({"title": "Raw IP host", "detail": FEATURE_DESCRIPTIONS["is_domain_ip"]})
    if features["count_at"] > 0:
        signals.append({"title": "@ symbol present", "detail": FEATURE_DESCRIPTIONS["count_at"]})
    if features["is_https"] == 0:
        signals.append({"title": "HTTP detected", "detail": FEATURE_DESCRIPTIONS["is_https"]})
    if rule_token_hits > 0:
        signals.append({"title": "Sensitive keywords", "detail": f"{rule_token_hits} suspicious keyword hits in path/query."})
    if features["count_dash"] >= 3:
        signals.append({"title": "Dash-heavy URL", "detail": FEATURE_DESCRIPTIONS["count_dash"]})
    if features["entropy"] >= 4.2:
        signals.append({"title": "High entropy", "detail": FEATURE_DESCRIPTIONS["entropy"]})
    if features["query_len"] >= 30:
        signals.append({"title": "Large query payload", "detail": FEATURE_DESCRIPTIONS["query_len"]})
    if features["subdomain_count"] >= 2:
        signals.append({"title": "Nested subdomains", "detail": FEATURE_DESCRIPTIONS["subdomain_count"]})
    if trusted_suffix:
        signals.append(
            {
                "title": "Trusted domain context",
                "detail": (
                    f"This host matches {trusted_suffix}, so the final score was softened to avoid an obvious false positive."
                ),
            }
        )
    return signals


def count_token_hits(text: str, tokens: tuple[str, ...]) -> int:
    return int(sum(text.count(token) for token in tokens))


def has_executable_extension(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith((".exe", ".scr", ".msi", ".apk", ".jar", ".bat", ".cmd", ".dll"))


def calibrate_multiclass_probabilities(
    url: str,
    class_names: list[str],
    probabilities: list[float],
    features: dict[str, float],
    rule_token_hits: int,
    trusted_suffix: str | None,
) -> list[float]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path_text = f"{parsed.path} {parsed.query}".lower()
    full_text = f"{host} {path_text}".lower()
    tld = host.split(".")[-1] if "." in host else ""
    executable_like = has_executable_extension(url)
    phishing_hits = count_token_hits(full_text, PHISHING_HINT_TOKENS)
    malware_hits = count_token_hits(full_text, MALWARE_HINT_TOKENS)
    defacement_hits = count_token_hits(full_text, DEFACEMENT_HINT_TOKENS)

    probs = {class_name: float(probability) for class_name, probability in zip(class_names, probabilities)}

    safe_profile = (
        features["is_https"] == 1
        and features["is_domain_ip"] == 0
        and features["count_at"] == 0
        and "xn--" not in host
        and features["subdomain_count"] <= 1
        and features["count_dash"] <= 2
        and tld in COMMON_SAFE_TLDS
        and features["query_len"] < 70
        and phishing_hits == 0
        and malware_hits == 0
        and defacement_hits == 0
        and not executable_like
        and rule_token_hits == 0
    )
    trusted_safe_profile = (
        trusted_suffix is not None
        and features["is_https"] == 1
        and features["is_domain_ip"] == 0
        and features["count_at"] == 0
        and "xn--" not in host
        and features["subdomain_count"] <= 2
        and tld in COMMON_SAFE_TLDS
        and features["query_len"] < 160
        and malware_hits == 0
        and defacement_hits == 0
        and not executable_like
        and phishing_hits <= 2
        and rule_token_hits <= 2
    )

    if trusted_safe_profile:
        probs["benign"] = max(probs.get("benign", 0.0), 0.90 if phishing_hits else 0.96)
        for class_name in probs:
            if class_name != "benign":
                probs[class_name] *= 0.10 if phishing_hits else 0.08
    elif safe_profile:
        probs["benign"] = max(probs.get("benign", 0.0), 0.80)
        probs["phishing"] *= 0.25
        probs["malware"] *= 0.30
        probs["defacement"] *= 0.45

    if defacement_hits > 0 and not executable_like and phishing_hits == 0:
        probs["defacement"] = max(probs.get("defacement", 0.0), 0.70 if defacement_hits >= 2 else 0.58)
        probs["phishing"] *= 0.45
        probs["malware"] *= 0.55

    if executable_like or malware_hits >= 2:
        probs["malware"] = max(probs.get("malware", 0.0), 0.82 if executable_like else 0.68)
        probs["benign"] *= 0.08
        if phishing_hits <= 1:
            probs["phishing"] *= 0.50

    phishing_profile = (
        phishing_hits >= 2
        or (rule_token_hits > 0 and (tld in RISKY_TLDS or features["is_https"] == 0))
        or features["is_domain_ip"] > 0
        or features["count_at"] > 0
        or "xn--" in host
    )
    if phishing_profile and not executable_like and not trusted_safe_profile:
        probs["phishing"] = max(probs.get("phishing", 0.0), 0.76 if phishing_hits >= 2 else 0.68)
        probs["benign"] *= 0.08
        if malware_hits <= 1:
            probs["malware"] *= 0.55
        if defacement_hits == 0:
            probs["defacement"] *= 0.70

    total = sum(max(value, 0.0) for value in probs.values())
    if total <= 0:
        return probabilities
    return [max(probs.get(class_name, 0.0), 0.0) / total for class_name in class_names]


def build_signals(
    features: dict[str, float],
    model_score: int,
    rule_score: int,
    rule_token_hits: int,
    trusted_suffix: str | None = None,
) -> list[dict]:
    signals = collect_feature_signals(features, rule_token_hits, trusted_suffix)
    signals.append(
        {
            "title": "Model score",
            "detail": (
                f"CNN score {model_score}/100. "
                f"Lexical score {rule_score}/100 is shown only as extra context."
            ),
        }
    )

    if not signals:
        signals.append({"title": "Low-risk structure", "detail": "No high-impact phishing indicators were triggered."})
    return signals[:6]


def build_multiclass_signals(
    features: dict[str, float],
    class_probabilities: list[dict[str, float | str]],
    rule_token_hits: int,
    trusted_suffix: str | None = None,
) -> list[dict]:
    signals = collect_feature_signals(features, rule_token_hits, trusted_suffix)
    top_classes = ", ".join(
        f"{item['label']} {int(round(float(item['probability'])))}%"
        for item in class_probabilities[:3]
    )
    signals.append(
        {
            "title": "Top class probabilities",
            "detail": top_classes or "No class scores available.",
        }
    )
    if not signals:
        signals.append({"title": "Low-risk structure", "detail": "No high-impact indicators were triggered."})
    return signals[:6]


def build_mitigation(label: str, score: int) -> dict:
    if label == "Likely phishing":
        return {
            "severity": "high",
            "action": "block_and_report",
            "title": "Block and escalate",
            "detail": (
                "Treat this link as malicious. Block it, hide previews, and send it for review."
            ),
        }
    if label == "Suspicious":
        return {
            "severity": "medium",
            "action": "quarantine_for_review",
            "title": "Hold for review",
            "detail": (
                "Keep this behind a warning and review it before letting it spread."
            ),
        }
    return {
        "severity": "low",
        "action": "allow_and_monitor",
        "title": "Allow with monitoring",
        "detail": (
            f"This looks safe for now. Allow it, but keep a record because the model score is {score}/100."
        ),
    }


def build_multiclass_mitigation(class_name: str, confidence: int) -> dict:
    if class_name == "phishing":
        return {
            "severity": "high",
            "action": "block_and_report",
            "title": "Block phishing workflow",
            "detail": (
                f"The model leans phishing at {confidence}/100. Block the link and send it for review."
            ),
        }
    if class_name == "malware":
        return {
            "severity": "high",
            "action": "block_and_isolate",
            "title": "Block and isolate",
            "detail": (
                f"The model leans malware at {confidence}/100. Block the link and isolate the case for follow-up."
            ),
        }
    if class_name == "defacement":
        return {
            "severity": "medium",
            "action": "review_for_tampering",
            "title": "Review for tampering",
            "detail": (
                f"The model leans defacement at {confidence}/100. Hold it for review and check the destination page."
            ),
        }
    return {
        "severity": "low",
        "action": "allow_and_monitor",
        "title": "Allow with monitoring",
        "detail": (
            f"The model leans benign at {confidence}/100. Allow it for now and keep the run in the audit trail."
        ),
    }


def classify(url_input: str) -> dict:
    if MODEL_BUNDLES["binary"] is None:
        MODEL_BUNDLES["binary"] = load_cnn_model()

    normalized_url = normalize_url(url_input)
    if not normalized_url:
        raise ValueError("URL is required.")

    model = MODEL_BUNDLES["binary"]["model"]
    threshold = float(MODEL_BUNDLES["binary"]["threshold"])

    # 1. Tokenize URL for PyTorch
    input_tensor = encode_url_tensor(normalized_url)
    
    # 2. Get Neural Network Prediction
    with torch.no_grad():
        score = float(model(input_tensor).item())
        
    model_score = int(round(score * 100))

    # 3. Extract features for UI Explainability
    features = extract_lexical_features(normalized_url)
    rule_score = compute_rule_risk(normalized_url, features)
    rule_token_hits = suspicious_hits_in_path_or_query(normalized_url)
    trusted_suffix = is_trusted_https_context(normalized_url, features, rule_token_hits)

    # Blend the CNN with lexical risk so a high model score alone does not
    # hard-block otherwise clean HTTPS URLs.
    low_risk_profile = (
        features["is_domain_ip"] == 0
        and features["count_at"] == 0
        and rule_token_hits == 0
        and features["subdomain_count"] <= 1
        and features["query_len"] == 0
        and features["url_len"] < 70
        and features["count_dash"] <= 1
        and features["is_https"] == 1
    )

    hard_block_profile = (
        (features["is_domain_ip"] > 0 and (features["is_https"] == 0 or rule_token_hits > 0))
        or features["count_at"] > 0
    )

    if trusted_suffix:
        final_score = int(round((0.25 * model_score) + (0.20 * rule_score)))
        final_score = max(0, min(34, final_score))
    elif low_risk_profile and rule_score < 20:
        final_score = min(34, int(round(model_score * 0.35)))
    else:
        final_score = int(round((0.55 * model_score) + (0.45 * rule_score)))
        if model_score >= 60 and rule_score >= 30:
            final_score = max(final_score, 67)
        if hard_block_profile:
            final_score = max(final_score, 67)
        final_score = max(0, min(100, final_score))

    # Verdict Logic
    if final_score >= 65: label = "Likely phishing"
    elif final_score >= 35: label = "Suspicious"
    else: label = "Likely benign"

    # Confidence calculation
    margin = abs(score - threshold)
    scale = max(threshold, 1.0 - threshold)
    confidence = int(round(min(99, max(45, 50 + (margin / (scale + 1e-9)) * 45 + abs(final_score - 50) * 0.12))))

    signals = build_signals(
        features,
        model_score=model_score,
        rule_score=rule_score,
        rule_token_hits=rule_token_hits,
        trusted_suffix=trusted_suffix,
    )
    model_tag = f"{MODEL_BUNDLES['binary'].get('model_name')} · thr={threshold:.2f}"
    mitigation = build_mitigation(label, final_score)

    return {
        "label": label,
        "score": final_score,
        "confidence": confidence,
        "signals": signals,
        "model": model_tag,
        "mitigation": mitigation,
        "task_type": "binary",
    }


def classify_multiclass(url_input: str) -> dict:
    if MODEL_BUNDLES["multiclass"] is None:
        MODEL_BUNDLES["multiclass"] = load_multiclass_cnn_model()

    normalized_url = normalize_url(url_input)
    if not normalized_url:
        raise ValueError("URL is required.")

    bundle = MODEL_BUNDLES["multiclass"]
    model = bundle["model"]
    class_names = list(bundle["class_names"])
    input_tensor = encode_url_tensor(normalized_url)
    features = extract_lexical_features(normalized_url)
    rule_token_hits = suspicious_hits_in_path_or_query(normalized_url)
    trusted_suffix = is_trusted_https_context(normalized_url, features, rule_token_hits)

    with torch.no_grad():
        logits = model(input_tensor)
        raw_probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    probabilities = calibrate_multiclass_probabilities(
        normalized_url,
        class_names,
        [float(value) for value in raw_probabilities],
        features,
        rule_token_hits,
        trusted_suffix,
    )

    scored_classes = sorted(
        (
            {
                "label": humanize_label(class_name),
                "class_name": class_name,
                "probability": round(float(probability) * 100, 2),
            }
            for class_name, probability in zip(class_names, probabilities)
        ),
        key=lambda item: float(item["probability"]),
        reverse=True,
    )
    top_class = scored_classes[0]
    top_confidence = int(round(float(top_class["probability"])))
    signals = build_multiclass_signals(
        features,
        class_probabilities=scored_classes,
        rule_token_hits=rule_token_hits,
        trusted_suffix=trusted_suffix,
    )
    mitigation = build_multiclass_mitigation(str(top_class["class_name"]), top_confidence)

    return {
        "label": str(top_class["label"]),
        "class_name": str(top_class["class_name"]),
        "score": top_confidence,
        "confidence": top_confidence,
        "signals": signals,
        "model": f"{bundle.get('model_name')} · {len(class_names)} classes",
        "mitigation": mitigation,
        "task_type": "multiclass",
        "class_probabilities": scored_classes,
    }


class ApiHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self) -> None:
        route = self.path.rstrip("/")
        if route not in {"/api/classify", "/api/classify-multiclass"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})
            return

        url_value = str(payload.get("url", "")).strip()
        if not url_value:
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "Field 'url' is required."})
            return

        try:
            if route == "/api/classify":
                result = classify(url_value)
            else:
                result = classify_multiclass(url_value)
        except Exception as exc: 
            self.respond_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Classification failed: {exc}"})
            return

        self.respond_json(HTTPStatus.OK, result)

    def respond_json(self, status: HTTPStatus, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UI + API phishing detection server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=5173, type=int, help="Bind port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Warm up model
    MODEL_BUNDLES["binary"] = load_cnn_model()
    print(f"Binary model ready: {MODEL_BUNDLES['binary']['model_name']} (threshold={MODEL_BUNDLES['binary']['threshold']:.2f})")
    if MULTICLASS_CNN_MODEL_PATH.exists():
        try:
            MODEL_BUNDLES["multiclass"] = load_multiclass_cnn_model()
            print(
                "Multiclass model ready: "
                f"{MODEL_BUNDLES['multiclass']['model_name']} ({len(MODEL_BUNDLES['multiclass']['class_names'])} classes)"
            )
        except Exception as exc:
            print(f"WARNING: Failed to warm up multiclass model: {exc}")
    else:
        print(f"WARNING: Multiclass model weights not found at {MULTICLASS_CNN_MODEL_PATH}")

    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"Serving http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
