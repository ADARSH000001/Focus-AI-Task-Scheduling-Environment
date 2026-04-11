"""
server/app.py
-------------
FastAPI server for the FocusAI Task Scheduling Environment.

Run from project root:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

All imports resolve because the project root is added to sys.path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is on sys.path so `env`, `models`, etc. resolve.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env import FocusEnv                           # noqa: E402
from reward_and_tasks import GRADERS               # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Focus AI Environment API",
    version="2.0.0",
    description="OpenEnv-compatible RL environment for AI task scheduling.",
)

# NOTE: Single global env instance. Not thread-safe by design.
# For production, replace with per-session env management (e.g. dict keyed by session_id
# with a threading.Lock() around all mutations in /step and /reset).
# Acceptable for hackathon single-agent evaluation.
_env: Optional[FocusEnv] = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: str = Field(
        default="easy",
        pattern="^(easy|medium|hard)$",
        description="Task difficulty level: easy | medium | hard",
    )


class StepRequest(BaseModel):
    action: str = Field(
        ...,
        description="e.g. start_task('report') | take_break(2) | noop()",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> Dict[str, Any]:
    """Serialize a Pydantic model or dict to a plain dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot serialize {type(obj)}")


def _require_env() -> FocusEnv:
    """Return the current env, auto-initialising if needed."""
    global _env
    if _env is None:
        _env = FocusEnv(difficulty="easy")
        _env.reset()
    return _env


def _manifest_tasks() -> list:
    """Read task list from openenv.yaml."""
    manifest = _ROOT / "openenv.yaml"
    if not manifest.exists():
        return []
    try:
        content = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        tasks = content.get("tasks", [])
        return tasks if isinstance(tasks, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "name":    "Focus AI Environment",
        "status":  "ready",
        "message": "Use POST /reset and POST /step to interact.",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {"tasks": _manifest_tasks()}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    global _env
    payload = request or ResetRequest()
    _env = FocusEnv(difficulty=payload.difficulty)
    obs = _env.reset()
    return {
        "difficulty":        payload.difficulty,
        "observation":       _to_dict(obs),
        "observation_text":  _env.get_observation_text(),
    }


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    env = _require_env()
    try:
        obs, reward, done, info = env.step(request.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    reward_dict = _to_dict(reward)
    return {
        "observation":    _to_dict(obs),
        "reward":         reward_dict.get("reward", 0.0),
        "reward_details": reward_dict,
        "done":           bool(done),
        "info":           info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    env = _require_env()
    return {
        "difficulty":        env.difficulty,
        "observation":       _to_dict(env.state),
        "observation_text":  env.get_observation_text(),
        "metrics":           dict(env.metrics),
    }


@app.get("/grade/{difficulty}")
def grade(difficulty: str) -> Dict[str, Any]:
    """
    Return the current score for the running episode.
    Score is ALWAYS strictly inside (0, 1) — guaranteed by the grader.
    """
    if difficulty not in GRADERS:
        raise HTTPException(
            status_code=400,
            detail=f"difficulty must be one of {list(GRADERS)}",
        )
    env = _require_env()
    if env.difficulty != difficulty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Current env difficulty is '{env.difficulty}'. "
                f"POST /reset with difficulty='{difficulty}' first."
            ),
        )
    score = GRADERS[difficulty](env.metrics)
    return {
        "difficulty": difficulty,
        "score":      float(score),
        "metrics":    dict(env.metrics),
    }


@app.get("/validate")
def validate() -> Dict[str, Any]:
    """Sanity-check the environment implementation."""
    checks: Dict[str, bool] = {
        "has_reset":          hasattr(FocusEnv, "reset"),
        "has_step":           hasattr(FocusEnv, "step"),
        "has_state_property": isinstance(getattr(FocusEnv, "state", None), property),
        "has_manifest_tasks": len(_manifest_tasks()) >= 3,
        "models_importable":  False,
    }
    try:
        from models import Observation, Action, Reward  # noqa: F401
        checks["models_importable"] = True
    except Exception:
        pass

    return {
        "valid":    all(checks.values()),
        "checks":   checks,
        "env_name": "focus-ai-env",
        "version":  "2.0.0",
    }


@app.get("/score/{difficulty}")
def score_difficulty(difficulty: str) -> Dict[str, Any]:
    """
    Run a complete smart_agent episode for the given difficulty and return
    the final score.  Useful for judges to quickly benchmark the env without
    running inference.py.

    Score is ALWAYS strictly inside (0, 1) — guaranteed by safe_score().
    """
    if difficulty not in GRADERS:
        raise HTTPException(
            status_code=400,
            detail=f"difficulty must be one of {list(GRADERS)}",
        )

    from env import smart_agent  # local import to avoid circular at module level

    bench_env = FocusEnv(difficulty=difficulty)
    obs = bench_env.reset()

    step_rewards = []
    final_score = None

    for _ in range(15):
        action = smart_agent(obs)
        obs, reward, done, info = bench_env.step(action)
        step_rewards.append(reward.reward)
        if done:
            final_score = info.get("score")
            break

    if final_score is None:
        final_score = GRADERS[difficulty](bench_env.metrics)

    return {
        "difficulty":    difficulty,
        "score":         float(final_score),
        "total_reward":  round(float(sum(step_rewards)), 4),
        "steps":         len(step_rewards),
        "metrics":       dict(bench_env.metrics),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
