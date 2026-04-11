#!/usr/bin/env python3
"""Run a hidden-test-style quality gate against the project."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import FocusEnv, smart_agent
from models import Observation, Reward
from reward_and_tasks import GRADERS, safe_score
from scripts.check_core_artifact import assert_required_files, assert_score_ranges


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_safe_score_edges() -> None:
    for raw in (-1.0, 0.0, 0.1, 0.5, 1.0, 2.0):
        score = safe_score(raw)
        assert_true(0.0 < score < 1.0, f"safe_score failed for raw={raw}: {score}")


def run_seed_matrix() -> None:
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(20):
            env = FocusEnv(difficulty=difficulty)
            obs = env.reset(seed=seed)
            assert_true(isinstance(obs, Observation), "reset() did not return Observation")
            score = None
            for _ in range(20):
                action = smart_agent(obs)
                obs, reward, done, info = env.step(action)
                assert_true(isinstance(obs, Observation), "step() obs type mismatch")
                assert_true(isinstance(reward, Reward), "step() reward type mismatch")
                assert_true("metrics" in info, "step() missing metrics")
                assert_true(0.0 <= reward.reward <= 20.0 or -20.0 <= reward.reward <= 0.0, "reward out of clamp range")
                if done:
                    score = info.get("score")
                    break
            if score is None:
                score = GRADERS[difficulty](env.metrics)
            assert_true(0.0 < float(score) < 1.0, f"final score out of range: {difficulty} seed={seed} score={score}")


def check_invalid_action_termination() -> None:
    env = FocusEnv(difficulty="easy")
    env.reset(seed=0)
    done = False
    final_score = None
    for _ in range(5):
        _, reward, done, info = env.step("this_is_invalid")
        assert_true(reward.reward == -3.0, "invalid action penalty must be flat -3")
        final_score = info.get("score")
    assert_true(done, "environment did not terminate after repeated invalid actions")
    assert_true(final_score is not None, "final score missing after stagnation termination")
    assert_true(0.0 < float(final_score) < 1.0, "stagnation score out of range")


def check_server_contract() -> None:
    from server.app import ResetRequest, StepRequest, health, reset, root, validate

    root_payload = root()
    assert_true(root_payload.get("status") == "ready", "root() payload mismatch")
    assert_true(health().get("status") == "healthy", "health() payload mismatch")

    reset_payload = reset(ResetRequest(difficulty="easy"))
    assert_true("observation" in reset_payload, "/reset missing observation")
    assert_true("observation_text" in reset_payload, "/reset missing observation_text")
    tasks = reset_payload["observation"]["tasks"]
    assert_true(bool(tasks), "/reset returned no tasks")
    first_task_id = tasks[0]["id"]

    from server.app import step as step_endpoint

    step_payload = step_endpoint(StepRequest(action=f"start_task('{first_task_id}')"))
    assert_true("reward" in step_payload, "/step missing reward")
    assert_true("info" in step_payload, "/step missing info")

    validate_payload = validate()
    assert_true(validate_payload.get("valid") is True, "/validate did not return valid=true")


def check_inference_contract() -> None:
    baseline_path = ROOT / "baseline_scores.json"
    original = baseline_path.read_text(encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=True,
        )
    finally:
        baseline_path.write_text(original, encoding="utf-8")

    output_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert_true(any(line.startswith("[START]") for line in output_lines), "inference missing [START] log")
    assert_true(any(line.startswith("[STEP]") for line in output_lines), "inference missing [STEP] log")
    assert_true(any(line.startswith("[END]") for line in output_lines), "inference missing [END] log")


def check_pyproject_entrypoint() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert_true('server = "server.app:main"' in text, "pyproject missing server entrypoint")


def main() -> None:
    logging.disable(logging.CRITICAL)
    checks = [
        ("required_files", assert_required_files),
        ("baseline_score_ranges", assert_score_ranges),
        ("safe_score_edges", check_safe_score_edges),
        ("seed_matrix", run_seed_matrix),
        ("invalid_action_termination", check_invalid_action_termination),
        ("server_contract", check_server_contract),
        ("inference_contract", check_inference_contract),
        ("pyproject_entrypoint", check_pyproject_entrypoint),
    ]

    results = []
    for name, fn in checks:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        results.append({"check": name, "status": "ok"})

    print(json.dumps({"status": "ok", "checks": results}, indent=2))


if __name__ == "__main__":
    main()
