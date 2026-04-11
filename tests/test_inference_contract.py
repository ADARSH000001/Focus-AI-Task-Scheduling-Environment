from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_inference_emits_required_markers():
    baseline = ROOT / "baseline_scores.json"
    original = baseline.read_text(encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        baseline.write_text(original, encoding="utf-8")

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert any(line.startswith("[START]") for line in lines)
    assert any(line.startswith("[STEP]") for line in lines)
    assert any(line.startswith("[END]") for line in lines)
