"""
reward_and_tasks.py
-------------------
Task definitions, reward shaping, and deterministic graders for FocusAI.

SCORE CONTRACT  (HIGHEST PRIORITY)
----------------------------------
Every public grader returns a float STRICTLY inside the open interval (0, 1).

    safe_score(raw) = LOWER + (UPPER - LOWER) * clamp(raw, 0, 1)

    where LOWER = 0.01, UPPER = 0.99

This guarantees:
    raw = 0.0  ->  0.01   (> 0, never equals 0)
    raw = 1.0  ->  0.99   (< 1, never equals 1)
    raw = 0.5  ->  0.50
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENERGY_LEVELS = ("low", "medium", "high")
PRIORITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}

_SCORE_LOWER = 0.01
_SCORE_UPPER = 0.99


# ---------------------------------------------------------------------------
# Score safety
# ---------------------------------------------------------------------------

def safe_score(raw: float) -> float:
    raw = float(raw)
    if raw < 0.0:
        raw = 0.0
    elif raw > 1.0:
        raw = 1.0
    result = _SCORE_LOWER + (_SCORE_UPPER - _SCORE_LOWER) * raw
    result = round(result, 6)
    assert 0.0 < result < 1.0, (
        f"safe_score VIOLATION: raw={raw!r} produced result={result!r} "
        f"which is not strictly inside (0, 1)"
    )
    return result


# ---------------------------------------------------------------------------
# Energy helpers
# ---------------------------------------------------------------------------

def numeric_to_level(energy: int) -> str:
    if energy > 70:
        return "high"
    if energy > 40:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Task Pool
# ---------------------------------------------------------------------------

TASK_POOL = [
    # High priority
    {"id": "incident",    "name": "Fix production incident",       "priority": "high",   "category": "engineering"},
    {"id": "security",    "name": "Apply security patch",          "priority": "high",   "category": "engineering"},
    {"id": "deploy",      "name": "Deploy critical release",       "priority": "high",   "category": "engineering"},
    {"id": "client_demo", "name": "Prepare client demo",           "priority": "high",   "category": "communication"},
    # Medium priority
    {"id": "pr_review",   "name": "Review pull request",           "priority": "medium", "category": "engineering"},
    {"id": "meeting",     "name": "Team sync meeting",             "priority": "medium", "category": "communication"},
    {"id": "slides",      "name": "Prepare presentation slides",   "priority": "medium", "category": "communication"},
    {"id": "budget",      "name": "Submit budget request",         "priority": "medium", "category": "finance"},
    # Low priority
    {"id": "docs",        "name": "Update documentation",          "priority": "low",    "category": "writing"},
    {"id": "email",       "name": "Clear inbox",                   "priority": "low",    "category": "communication"},
    {"id": "cleanup",     "name": "Code cleanup/refactor",         "priority": "low",    "category": "engineering"},
    {"id": "retro",       "name": "Write retrospective notes",     "priority": "low",    "category": "process"},
]


# ---------------------------------------------------------------------------
# Randomized Task Generator
# ---------------------------------------------------------------------------

def get_random_task(difficulty: str, seed: int | None = None) -> dict:
    """
    Generate a randomized task scenario for the given difficulty.

    Parameters
    ----------
    difficulty : str
        One of 'easy', 'medium', 'hard'.
    seed : int or None
        - None (default): truly random each call — good for training so the
          agent cannot memorise a fixed scenario.
        - Fixed integer (e.g. seed=42): deterministic output — use this in
          evaluation / debugging to reproduce an exact episode.

    Returns
    -------
    dict with keys: energy, energy_level, tasks
    """
    rng = random.Random(seed)

    config = {
        "easy":   {"n": 2, "energy": 80, "deadline_range": (12, 20)},
        "medium": {"n": 3, "energy": 60, "deadline_range": (10, 18)},
        "hard":   {"n": 6, "energy": 55, "deadline_range": (10, 16)},
    }[difficulty]

    assert len(TASK_POOL) >= config["n"], (
        f"TASK_POOL too small for difficulty={difficulty!r}: "
        f"need {config['n']} tasks but pool only has {len(TASK_POOL)}"
    )

    selected = rng.sample(TASK_POOL, config["n"])
    tasks = []
    current_time = 9

    for t in selected:
        duration = rng.randint(1, 2 if difficulty == "easy" else 3)
        deadline = rng.randint(*config["deadline_range"])
        tasks.append({
            "id":         t["id"],
            "name":       t["name"],
            "priority":   t["priority"],
            "deadline":   max(current_time + duration, deadline),
            "difficulty": duration,
            "category":   t["category"],
        })

    energy = config["energy"] + rng.randint(-10, 10)
    energy = max(30, min(100, energy))

    return {
        "energy":       energy,
        "energy_level": numeric_to_level(energy),
        "tasks":        tasks,
    }


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def calculate_reward(state: dict, action: str, result: dict) -> float:
    reward = 0.0

    raw_energy = state.get("energy", "medium")
    if isinstance(raw_energy, int):
        energy = numeric_to_level(raw_energy)
    elif raw_energy in ENERGY_LEVELS:
        energy = raw_energy
    else:
        energy = "medium"

    if result.get("task_completed"):
        priority = result.get("priority", "low")
        base_reward = {"high": 15, "medium": 12, "low": 10}.get(priority, 10)
        reward += base_reward

        if action.startswith("start_task"):
            if priority == "high":
                reward += 2
            elif priority == "low":
                reward -= 2

        # NOTE: before_deadline bonus is handled once below (+5).
        # Do NOT add a second bonus here.

    if result.get("before_deadline"):
        reward += 5
    elif result.get("missed_deadline"):
        reward -= 10

    if action.startswith("start_task"):
        energy_delta = {"high": 3, "medium": 1, "low": -5}.get(energy, 0)
        reward += energy_delta

    elif action.startswith("take_break"):
        energy_delta = {"low": 4, "medium": 1, "high": -3}.get(energy, 0)
        reward += energy_delta

    if action.startswith("noop"):
        reward -= 3

    return float(max(-20.0, min(20.0, reward)))


# ---------------------------------------------------------------------------
# Task format conversion
# ---------------------------------------------------------------------------

def build_env_tasks(raw_tasks: list) -> list:
    return [
        {
            "id":        t["id"],
            "name":      t.get("name", t["id"]),
            "priority":  t["priority"],
            "deadline":  t["deadline"],
            "duration":  t.get("difficulty", 1),
            "category":  t.get("category", "general"),
            "completed": False,
        }
        for t in raw_tasks
    ]


# ---------------------------------------------------------------------------
# Static task generators (kept for debugging / deterministic testing)
# ---------------------------------------------------------------------------

def get_easy_task() -> dict:
    return {
        "energy": 80, "energy_level": "high",
        "tasks": [
            {"id": "report",  "name": "Write Q3 status report", "priority": "high",  "deadline": 12, "difficulty": 1, "category": "writing"},
            {"id": "inbox",   "name": "Clear email inbox",       "priority": "low",   "deadline": 18, "difficulty": 1, "category": "communication"},
        ],
    }


def get_medium_task() -> dict:
    return {
        "energy": 60, "energy_level": "medium",
        "tasks": [
            {"id": "pr_review",   "name": "Review pull request blocking deploy", "priority": "high",   "deadline": 11, "difficulty": 2, "category": "engineering"},
            {"id": "client_call", "name": "Prepare slides for client",           "priority": "medium", "deadline": 14, "difficulty": 2, "category": "communication"},
            {"id": "docs",        "name": "Update internal documentation",       "priority": "low",    "deadline": 18, "difficulty": 1, "category": "writing"},
        ],
    }


def get_hard_task() -> dict:
    return {
        "energy": 55, "energy_level": "medium",
        "tasks": [
            {"id": "incident",   "name": "Respond to production incident",     "priority": "high",   "deadline": 12, "difficulty": 2, "category": "engineering"},
            {"id": "security",   "name": "Apply critical security patch",      "priority": "high",   "deadline": 13, "difficulty": 3, "category": "engineering"},
            {"id": "interview",  "name": "Complete interview feedback form",   "priority": "medium", "deadline": 15, "difficulty": 2, "category": "hr"},
            {"id": "budget",     "name": "Submit team budget request",         "priority": "medium", "deadline": 16, "difficulty": 2, "category": "finance"},
            {"id": "onboarding", "name": "Set up new hire workstation",        "priority": "low",    "deadline": 20, "difficulty": 1, "category": "hr"},
            {"id": "retro",      "name": "Write sprint retrospective notes",   "priority": "low",    "deadline": 22, "difficulty": 1, "category": "process"},
        ],
    }


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_easy(metrics: dict) -> float:
    total = max(1, metrics.get("total_tasks", 1))
    raw = (
        0.60 * _safe_ratio(metrics.get("completed_tasks", 0), total)
        + 0.40 * _safe_ratio(metrics.get("on_time", 0), total)
    )
    return safe_score(raw)


def grade_medium(metrics: dict) -> float:
    total = max(1, metrics.get("total_tasks", 1))
    steps = max(1, metrics.get("total_steps", 1))
    raw = (
        0.40 * _safe_ratio(metrics.get("completed_tasks", 0), total)
        + 0.35 * _safe_ratio(metrics.get("on_time", 0), total)
        + 0.25 * _safe_ratio(metrics.get("good_energy_usage", 0), steps)
    )
    return safe_score(raw)


def grade_hard(metrics: dict) -> float:
    total = max(1, metrics.get("total_tasks", 1))
    steps = max(1, metrics.get("total_steps", 1))
    completed = max(1, metrics.get("completed_tasks", 1))
    raw = (
        0.35 * _safe_ratio(metrics.get("completed_tasks", 0), total)
        + 0.30 * _safe_ratio(metrics.get("on_time", 0), total)
        + 0.20 * _safe_ratio(metrics.get("good_energy_usage", 0), steps)
        + 0.15 * _safe_ratio(metrics.get("high_priority_choices", 0), completed)
    )
    return safe_score(raw)


def grade_performance(metrics: dict) -> float:
    """Aggregate grader used for cross-difficulty scoring in inference.py."""
    total = max(1, metrics.get("total_tasks", 1))
    steps = max(1, metrics.get("total_steps", 1))
    raw = (
        0.40 * _safe_ratio(metrics.get("completed_tasks", 0), total)
        + 0.30 * _safe_ratio(metrics.get("on_time", 0), total)
        + 0.20 * _safe_ratio(metrics.get("good_energy_usage", 0), steps)
        + 0.10 * _safe_ratio(metrics.get("high_priority_choices", 0), steps)
    )
    return safe_score(raw)


# ---------------------------------------------------------------------------
# Registry maps
# ---------------------------------------------------------------------------

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

# Randomized loaders — prevents LLM from memorizing fixed scenarios
TASK_LOADERS = {
    "easy":   lambda: get_random_task("easy"),
    "medium": lambda: get_random_task("medium"),
    "hard":   lambda: get_random_task("hard"),
}