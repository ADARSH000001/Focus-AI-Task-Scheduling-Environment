#!/usr/bin/env python3
"""Lightweight validator-safe checks for the local project artifact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "Dockerfile",
    "inference.py",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "env.py",
    "models.py",
    "reward_and_tasks.py",
    "server/app.py",
    "baseline_scores.json",
]


def assert_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")


def assert_score_ranges() -> None:
    data = json.loads((ROOT / "baseline_scores.json").read_text(encoding="utf-8"))
    for item in data.get("difficulties", []):
        score = float(item["score"])
        if not (0.0 < score < 1.0):
            raise SystemExit(
                f"Baseline score out of range for {item['difficulty']}: {score}"
            )


def main() -> None:
    assert_required_files()
    assert_score_ranges()
    print("artifact_ok")


if __name__ == "__main__":
    main()
