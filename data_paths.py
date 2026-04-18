#!/usr/bin/env python3
"""Helpers for resolving dataset directories in this repo."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def dataset_dir_candidates(dataset_name: str) -> list[Path]:
    return [
        ROOT_DIR / dataset_name,
        ROOT_DIR / "processed_data" / dataset_name,
    ]


def resolve_dataset_dir(dataset_name: str) -> Path:
    for candidate in dataset_dir_candidates(dataset_name):
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in dataset_dir_candidates(dataset_name))
    raise FileNotFoundError(
        f"Could not find dataset directory '{dataset_name}'. Searched: {searched}"
    )


def resolve_dataset_file(dataset_name: str, relative_path: str) -> Path:
    return resolve_dataset_dir(dataset_name) / relative_path
