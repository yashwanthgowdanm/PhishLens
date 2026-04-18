#!/usr/bin/env python3
"""Clean, audit, and merge adversarial URL data into dataset1_binary."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from data_paths import resolve_dataset_dir

SUSPICIOUS_TOKENS = (
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "bank",
    "signin",
    "password",
    "confirm",
    "invoice",
    "support",
    "auth",
    "wallet",
    "recovery",
    "unlock",
    "pay",
)

POSITIVE_LABELS = {"1", "1.0", "bad", "phishing", "malicious", "malware"}
NEGATIVE_LABELS = {"0", "0.0", "benign", "good", "safe", "legitimate"}
URL_COLUMN_CANDIDATES = ("url_raw", "URL", "url", "Url")
LABEL_COLUMN_CANDIDATES = ("label_raw", "Label", "label", "Label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize, audit, deduplicate, and merge adversarial URL data."
    )
    parser.add_argument(
        "--new-data",
        type=Path,
        default=Path("new_dataset.csv"),
        help="CSV containing new URLs to inject into the training split.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=resolve_dataset_dir("dataset1_binary"),
        help="Directory containing split_train.csv and lexical_features.csv.",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("outputs/data_merge_audit.json"),
        help="Where to write the merge audit JSON report.",
    )
    return parser.parse_args()


def is_ip_host(host: str) -> int:
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return float(entropy)


def normalize_url(raw_url: object) -> str:
    text = "" if pd.isna(raw_url) else str(raw_url).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    if "://" not in text:
        text = f"http://{text}"
    return text.lower()


def build_url_key(raw_url: object) -> str:
    normalized = normalize_url(raw_url)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    path = parsed.path or ""
    params = f";{parsed.params}" if parsed.params else ""
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.netloc.lower()}{path}{params}{query}"


def coerce_label(raw_label: object) -> float | None:
    if pd.isna(raw_label):
        return None
    normalized = str(raw_label).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1.0
    if normalized in NEGATIVE_LABELS:
        return 0.0
    return None


def extract_lexical_features(url: str) -> dict[str, float]:
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

    return {
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


def canonicalize_existing_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "url_norm" not in frame.columns:
        raise ValueError("Expected column 'url_norm' in existing dataset.")
    frame["url_norm"] = frame["url_norm"].map(normalize_url)
    frame["url_key"] = frame["url_norm"].map(build_url_key)
    return frame


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    url_column = next((column for column in URL_COLUMN_CANDIDATES if column in df.columns), None)
    label_column = next(
        (column for column in LABEL_COLUMN_CANDIDATES if column in df.columns),
        None,
    )
    if url_column is None or label_column is None:
        raise ValueError(
            "Could not find URL/label columns. "
            f"Accepted URL columns: {URL_COLUMN_CANDIDATES}; "
            f"accepted label columns: {LABEL_COLUMN_CANDIDATES}."
        )

    renamed = df.rename(columns={url_column: "url_raw", label_column: "label_raw"}).copy()
    renamed["url_raw"] = renamed["url_raw"].astype(str).str.strip()
    renamed["url_norm"] = renamed["url_raw"].map(normalize_url)
    renamed["url_key"] = renamed["url_raw"].map(build_url_key)
    renamed["label"] = renamed["label_raw"].map(coerce_label)
    return renamed


def summarize_label_conflicts(df: pd.DataFrame) -> list[str]:
    grouped = df.groupby("url_key")["label"].nunique()
    return sorted(grouped[grouped > 1].index.tolist())


def dedupe_by_url(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="url_key", keep="first").reset_index(drop=True)


def build_new_feature_rows(rows: pd.DataFrame, lexical_columns: list[str]) -> pd.DataFrame:
    feature_rows = []
    for _, row in rows.iterrows():
        features = extract_lexical_features(row["url_raw"])
        features["url_norm"] = row["url_norm"]
        features["url_key"] = row["url_key"]
        features["label"] = row["label"]
        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)
    for column in lexical_columns:
        if column not in feature_df.columns:
            feature_df[column] = pd.NA
    return feature_df[lexical_columns]


def merge_data(args: argparse.Namespace) -> dict:
    train_file = args.data_dir / "split_train.csv"
    lex_file = args.data_dir / "lexical_features.csv"

    if not args.new_data.exists():
        raise FileNotFoundError(f"Missing new data file: {args.new_data}")
    if not train_file.exists():
        raise FileNotFoundError(f"Missing training split: {train_file}")
    if not lex_file.exists():
        raise FileNotFoundError(f"Missing lexical features file: {lex_file}")

    raw_new_df = pd.read_csv(args.new_data)
    existing_train_df = canonicalize_existing_frame(pd.read_csv(train_file))
    existing_lex_df = canonicalize_existing_frame(pd.read_csv(lex_file))

    prepared_new_df = resolve_columns(raw_new_df)
    invalid_url_df = prepared_new_df[prepared_new_df["url_norm"] == ""]
    invalid_label_df = prepared_new_df[prepared_new_df["label"].isna()]

    cleaned_new_df = prepared_new_df[
        (prepared_new_df["url_norm"] != "") & prepared_new_df["label"].notna()
    ][["url_raw", "url_norm", "url_key", "label", "label_raw"]].copy()

    conflicting_new_urls = summarize_label_conflicts(cleaned_new_df)
    cleaned_new_df = cleaned_new_df[~cleaned_new_df["url_key"].isin(conflicting_new_urls)]

    duplicate_new_rows = int(cleaned_new_df.duplicated(subset="url_key").sum())
    cleaned_new_df = dedupe_by_url(cleaned_new_df)

    stable_existing_train = existing_train_df.copy()
    existing_conflicting_urls = summarize_label_conflicts(
        stable_existing_train[["url_key", "label"]]
    )
    stable_existing_train = stable_existing_train[
        ~stable_existing_train["url_key"].isin(existing_conflicting_urls)
    ].copy()
    stable_existing_train = dedupe_by_url(stable_existing_train)

    stable_existing_lex = existing_lex_df[
        ~existing_lex_df["url_key"].isin(existing_conflicting_urls)
    ].copy()
    stable_existing_lex = dedupe_by_url(stable_existing_lex)

    existing_label_by_url = stable_existing_train.set_index("url_key")["label"].to_dict()
    duplicate_against_train = 0
    conflicting_against_train_urls: list[str] = []
    rows_to_insert = []

    for _, row in cleaned_new_df.iterrows():
        url_key = row["url_key"]
        existing_label = existing_label_by_url.get(url_key)
        if existing_label is None:
            rows_to_insert.append(row)
            continue
        if float(existing_label) == float(row["label"]):
            duplicate_against_train += 1
        else:
            conflicting_against_train_urls.append(url_key)

    insert_df = pd.DataFrame(rows_to_insert, columns=cleaned_new_df.columns)

    combined_train = pd.concat(
        [stable_existing_train, insert_df[["url_raw", "url_norm", "url_key", "label"]]],
        ignore_index=True,
    )
    combined_train = dedupe_by_url(combined_train)

    lexical_columns = list(existing_lex_df.columns)
    new_feature_rows = build_new_feature_rows(insert_df, lexical_columns)
    combined_lex = pd.concat([stable_existing_lex, new_feature_rows], ignore_index=True)
    combined_lex = dedupe_by_url(combined_lex)

    combined_train.drop(columns=["url_key"], errors="ignore").to_csv(train_file, index=False)
    combined_lex.drop(columns=["url_key"], errors="ignore").to_csv(lex_file, index=False)

    audit = {
        "new_data_file": str(args.new_data),
        "data_dir": str(args.data_dir),
        "new_rows_total": int(len(raw_new_df)),
        "invalid_url_rows_dropped": int(len(invalid_url_df)),
        "invalid_label_rows_dropped": int(len(invalid_label_df)),
        "conflicting_new_label_urls_dropped": len(conflicting_new_urls),
        "duplicate_new_rows_dropped": duplicate_new_rows,
        "existing_conflicting_label_urls_removed": len(existing_conflicting_urls),
        "duplicates_against_existing_train_skipped": duplicate_against_train,
        "conflicts_against_existing_train_skipped": len(conflicting_against_train_urls),
        "rows_inserted": int(len(insert_df)),
        "train_rows_before": int(len(existing_train_df)),
        "train_rows_after": int(len(combined_train)),
        "train_unique_urls_after": int(combined_train["url_key"].nunique()),
        "lexical_rows_before": int(len(existing_lex_df)),
        "lexical_rows_after": int(len(combined_lex)),
        "lexical_unique_urls_after": int(combined_lex["url_key"].nunique()),
        "conflicting_new_label_examples": conflicting_new_urls[:10],
        "conflicts_against_existing_train_examples": sorted(conflicting_against_train_urls)[:10],
    }
    return audit


def main() -> None:
    args = parse_args()
    audit = merge_data(args)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_out.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print("Data merge completed.")
    print(f"Inserted rows: {audit['rows_inserted']}")
    print(f"Dropped invalid URLs: {audit['invalid_url_rows_dropped']}")
    print(f"Dropped invalid labels: {audit['invalid_label_rows_dropped']}")
    print(f"Skipped duplicates already in train: {audit['duplicates_against_existing_train_skipped']}")
    print(f"Audit report written to: {args.audit_out}")


if __name__ == "__main__":
    main()
