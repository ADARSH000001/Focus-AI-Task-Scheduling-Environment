"""
env.py
------
FocusAI Task Scheduling RL Environment.

Models a knowledge worker's day: the agent must complete tasks before their
deadlines while managing cognitive energy to avoid burnout.

OpenEnv interface
-----------------
    reset()  ->  Observation
    step()   ->  (Observation, Reward, bool, dict)
    state    ->  Observation  (@property, read-only)

Actions
-------
    start_task('<id>')   — work on a pending task  (costs time + energy)
    take_break(<hours>)  — recover energy          (costs time)
    switch_task('<id>')  — change active focus      (0.5h context-switch cost)
    noop()               — do nothing              (always penalised -3)

Episode terminates when
-----------------------
    - All tasks are completed
    - Energy reaches 0  (burnout)
    - Time reaches 24h  (end of day)
    - STAGNATION_LIMIT consecutive invalid / unproductive steps
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from models import Observation, Reward, Task
from reward_and_tasks import (
    GRADERS,
    TASK_LOADERS,
    build_env_tasks,
    calculate_reward,
    numeric_to_level,
    safe_score,
)

logger = logging.getLogger(__name__)

# Episode ends after this many consecutive invalid / noop steps.
STAGNATION_LIMIT = 5


class FocusEnv:
    """
    FocusAI Task Scheduling RL Environment.

    Implements the OpenEnv interface: reset(), step(), and the state property.

    Usage
    -----
        env = FocusEnv(difficulty="medium")
        obs = env.reset()
        obs, reward, done, info = env.step("start_task('pr_review')")
    """

    def __init__(self, difficulty: str = "easy") -> None:
        if difficulty not in GRADERS:
            raise ValueError(
                f"difficulty must be one of {list(GRADERS)}; got {difficulty!r}"
            )
        self.difficulty: str = difficulty
        self._state: Optional[Dict[str, Any]] = None
        self.history: list = []
        self.metrics: Dict[str, Any] = {}
        self._stagnation: int = 0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Observation:
        """Start a fresh episode and return the initial Observation."""
        from reward_and_tasks import get_random_task
        if seed is not None:
            scenario = get_random_task(self.difficulty, seed=seed)
        else:
            scenario = TASK_LOADERS[self.difficulty]()
        tasks = build_env_tasks(scenario["tasks"])

        self._state = {
            "time":         9.0,
            "energy":       scenario["energy"],
            "energy_level": scenario["energy_level"],
            "tasks":        tasks,
            "current_task": None,
        }
        self.history = []
        self._stagnation = 0
        self.metrics = {
            "completed_tasks":       0,
            "total_tasks":           len(tasks),
            "on_time":               0,
            "good_energy_usage":     0,
            "high_priority_choices": 0,
            "total_steps":           0,
        }

        logger.info(
            "reset | difficulty=%s  tasks=%d  energy=%d (%s)",
            self.difficulty, len(tasks),
            self._state["energy"], self._state["energy_level"],
        )
        return self._build_obs()

    @property
    def state(self) -> Observation:
        """Current observation (OpenEnv @property requirement)."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state.")
        return self._build_obs()

    def step(
        self,
        action: str,
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply one action and advance the environment.

        Returns (observation, reward, done, info).
        info always contains 'metrics'; 'score' is set when done=True.

        Invalid actions receive a flat -3 reward without mutating state.
        calculate_reward() is never called for invalid actions — this
        prevents energy-delta shaping from stacking with the penalty.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        if self._is_done():
            return (
                self._build_obs(),
                Reward(reward=0.0),
                True,
                {"error": "Episode already finished."},
            )

        logger.debug(
            "step | action=%r  energy=%d  time=%.1f",
            action, self._state["energy"], self._state["time"],
        )

        name, args = self._parse_action(action)

        # ---- Unparseable action ------------------------------------------
        if name == "invalid":
            logger.warning("step | unparseable action: %r", action)
            return self._penalise(f"Unparseable action: {action!r}")

        # Snapshot energy BEFORE state mutation (used for reward shaping)
        energy_before = self._state["energy_level"]

        self.history.append(action)

        # ---- Dispatch ----------------------------------------------------
        if name == "start_task":
            result = self._do_start_task(args[0]) if args else {"invalid": True}
        elif name == "take_break":
            result = self._do_take_break(int(args[0])) if args else {"invalid": True}
        elif name == "switch_task":
            result = self._do_switch_task(args[0]) if args else {"invalid": True}
        elif name == "noop":
            result = self._empty_result()
        else:
            result = {"invalid": True}

        # Keep energy_level string in sync with numeric value
        self._state["energy_level"] = numeric_to_level(self._state["energy"])

        # ---- Invalid result (e.g. unknown task id) -----------------------
        if result.get("invalid"):
            logger.warning("step | invalid result for action %r", action)
            return self._penalise(f"Invalid action: {action!r}")

        # ---- Stagnation counter ------------------------------------------
        if name == "noop":
            self._stagnation += 1
        else:
            self._stagnation = 0

        # ---- Reward (only for valid, state-mutating actions) -------------
        reward_value = calculate_reward(
            {"energy": energy_before}, action, result,
        )

        # Extra burnout penalty if energy is now depleted
        if self._state["energy"] <= 0:
            reward_value = max(-20.0, reward_value - 10.0)

        # ---- Metrics update ----------------------------------------------
        self.metrics["total_steps"] += 1

        if result.get("task_completed"):
            self.metrics["completed_tasks"] += 1
            if result.get("before_deadline"):
                self.metrics["on_time"] += 1
            if result.get("priority") == "high":
                self.metrics["high_priority_choices"] += 1

        if name == "start_task" and energy_before in ("medium", "high"):
            self.metrics["good_energy_usage"] += 1
        elif name == "take_break" and energy_before == "low":
            self.metrics["good_energy_usage"] += 1

        # ---- Build output ------------------------------------------------
        done = self._is_done()
        obs = self._build_obs()
        reward_obj = Reward(
            reward=reward_value,
            task_completed=result.get("task_completed", False),
            before_deadline=result.get("before_deadline", False),
            missed_deadline=result.get("missed_deadline", False),
            priority=result.get("priority"),
        )

        info: Dict[str, Any] = {"metrics": dict(self.metrics), "score": None}
        if done:
            info["score"] = GRADERS[self.difficulty](self.metrics)
            logger.info("step | episode done  score=%.6f", info["score"])

        logger.debug(
            "step | reward=%+.1f  done=%s  time=%.1fh  energy=%d (%s)",
            reward_value, done,
            self._state["time"], self._state["energy"],
            self._state["energy_level"],
        )
        return obs, reward_obj, done, info

    # ------------------------------------------------------------------
    # Done condition  (single authoritative location)
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        if self._state is None:
            return False
        if all(t["completed"] for t in self._state["tasks"]):
            logger.debug("done | all tasks completed")
            return True
        if self._state["energy"] <= 0:
            logger.warning("done | burnout — energy depleted")
            return True
        if self._state["time"] >= 24.0:
            logger.debug("done | end of day reached")
            return True
        if self._stagnation >= STAGNATION_LIMIT:
            logger.warning("done | stagnation limit %d reached", STAGNATION_LIMIT)
            return True
        return False

    # ------------------------------------------------------------------
    # Penalty helper  (flat -3, no calculate_reward call)
    # ------------------------------------------------------------------

    def _penalise(self, reason: str) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Handle invalid actions.

        Applies a flat -3 reward without mutating environment state.
        Increments the stagnation counter so repeated invalid actions
        eventually terminate the episode.
        """
        self._stagnation += 1
        self.metrics["total_steps"] += 1

        done = self._is_done()
        obs = self._build_obs()
        info: Dict[str, Any] = {
            "metrics": dict(self.metrics),
            "score":   None,
            "error":   reason,
        }
        if done:
            info["score"] = GRADERS[self.difficulty](self.metrics)

        return obs, Reward(reward=-3.0), done, info

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _do_start_task(self, task_id: str) -> dict:
        for task in self._state["tasks"]:
            if task["id"] == task_id:
                if task["completed"]:
                    return {"invalid": True}
                self._state["current_task"] = task_id
                self._state["time"]   = min(24.0, self._state["time"] + task["duration"])
                self._state["energy"] = max(0,    self._state["energy"] - task["duration"] * 10)
                task["completed"] = True
                on_time = self._state["time"] <= task["deadline"]
                return {
                    "task_completed":  True,
                    "before_deadline": on_time,
                    "missed_deadline": not on_time,
                    "priority":        task["priority"],
                }
        return {"invalid": True}

    def _do_take_break(self, hours: int) -> dict:
        if hours <= 0:
            return {"invalid": True}
        self._state["energy"] = min(100, self._state["energy"] + hours * 10)
        self._state["time"]   = min(24.0, self._state["time"] + hours)
        logger.debug("take_break | %dh  energy→%d", hours, self._state["energy"])
        return self._empty_result()

    def _do_switch_task(self, task_id: str) -> dict:
        for task in self._state["tasks"]:
            if task["id"] == task_id:
                if task["completed"]:
                    return {"invalid": True}
                if self._state["current_task"] == task_id:
                    return {"invalid": True}
                self._state["current_task"] = task_id
                self._state["time"] = min(24.0, self._state["time"] + 0.5)
                return self._empty_result()
        return {"invalid": True}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        return {
            "task_completed":  False,
            "before_deadline": False,
            "missed_deadline": False,
            "priority":        None,
        }

    @staticmethod
    def _parse_action(action: str):
        """
        Parse an action string into (name, args).

        Examples:
            "start_task('report')"  →  ("start_task", ["report"])
            "take_break(2)"         →  ("take_break", ["2"])
            "noop()"                →  ("noop", [])
            "garbage"               →  ("invalid", [])
        """
        try:
            action = action.strip()
            if "(" not in action or not action.endswith(")"):
                return "invalid", []
            name = action.split("(")[0].strip()
            args_str = action[action.index("(") + 1 : action.rindex(")")]
            if not args_str.strip():
                return name, []
            args = [a.strip().strip("'\"") for a in args_str.split(",")]
            return name, args
        except Exception as exc:
            logger.error("parse_action failed on %r: %s", action, exc)
            return "invalid", []

    def _build_legal_actions(self) -> list:
        """Dynamically compute the list of valid actions for the current state."""
        actions = []
        pending = [t for t in self._state["tasks"] if not t["completed"]]

        for t in pending:
            actions.append(f"start_task('{t['id']}')")

        if self._state["current_task"] is not None:
            for t in pending:
                if t["id"] != self._state["current_task"]:
                    actions.append(f"switch_task('{t['id']}')")

        actions += ["take_break(1)", "take_break(2)", "noop()"]
        return actions

    def _build_obs(self) -> Observation:
        return Observation(
            time=self._state["time"],
            energy=self._state["energy"],
            energy_level=self._state["energy_level"],
            tasks=[
                Task(**{k: v for k, v in t.items()
                        if k in ("id", "name", "priority", "deadline",
                                 "duration", "category", "completed")})
                for t in self._state["tasks"]
            ],
            current_task=self._state["current_task"],
            recent_actions=list(self.history[-3:]),
            goal="Complete all tasks before their deadlines while managing energy to avoid burnout.",
            legal_actions=self._build_legal_actions(),
        )

    def get_observation_text(self) -> str:
        """Human-readable observation string optimised for LLM prompts."""
        time   = self._state["time"]
        energy = self._state["energy"]
        level  = self._state["energy_level"]

        advice = {
            "high":   "Energy HIGH — ideal for difficult tasks.",
            "medium": "Energy MEDIUM — manageable, consider task order.",
            "low":    "Energy LOW — burnout risk! Take a short break first.",
        }[level]

        completed_tasks = [t for t in self._state["tasks"] if t["completed"]]
        pending_tasks   = [t for t in self._state["tasks"] if not t["completed"]]

        completed_lines = []
        for t in completed_tasks:
            completed_lines.append(f"  [DONE] {t['id']}")

        pending_lines = []
        for t in pending_tasks:
            time_left = t["deadline"] - time
            urgency = "⚠ URGENT" if time_left <= t["duration"] + 1 else "OK"
            pending_lines.append(
                f"  [TODO] {t['id']:14s}  {t['name']}\n"
                f"           priority={t['priority']:6s}  "
                f"deadline={t['deadline']}h  "
                f"duration={t['duration']}h  "
                f"time_left={time_left:.1f}h  {urgency}"
            )

        history_str = ", ".join(self.history[-3:]) or "none"
        legal = self._build_legal_actions()

        completed_section = (
            "\nCOMPLETED TASKS (DO NOT REPEAT):\n"
            + ("\n".join(completed_lines) if completed_lines else "  (none yet)")
        )
        pending_section = (
            "\nPENDING TASKS:\n"
            + ("\n".join(pending_lines) if pending_lines else "  (all done!)")
        )

        return (
            f"=== FOCUS AI — TASK SCHEDULER ===\n"
            f"Time    : {time}h / 24h\n"
            f"Energy  : {energy}/100 ({level.upper()})\n"
            f"Advice  : {advice}\n"
            f"Pending : {len(pending_tasks)}/{len(self._state['tasks'])} tasks\n"
            + completed_section + "\n"
            + "WARNING: Calling start_task on a DONE task wastes your step and counts against you.\n"
            + pending_section + "\n\n"
            f"Recent actions : {history_str}\n\n"
            f"VALID ACTIONS RIGHT NOW:\n"
            + "\n".join(f"  {a}" for a in legal) + "\n\n"
            f"GOAL: Complete all pending tasks before their deadlines.\n"
        )


# ---------------------------------------------------------------------------
# Smart baseline agent  (deterministic, no LLM required)
# ---------------------------------------------------------------------------

def smart_agent(observation):
    energy = observation.energy
    time = observation.time

    pending = [t for t in observation.tasks if not t.completed]

    if not pending:
        return "noop()"

    # Priority weights
    pri = {"high": 3, "medium": 2, "low": 1}

    # -------------------------------
    # STEP 1 — TASK SCORING
    # -------------------------------
    def score_task(t):
        slack = t.deadline - time - t.duration

        # Urgency (less slack = more important)
        urgency_score = max(0, 10 - slack)

        # Priority weight
        priority_score = pri[t.priority] * 15

        # Energy feasibility
        energy_penalty = 0
        if energy < t.duration * 10:
            energy_penalty = -20

        # Deadline violation penalty
        deadline_penalty = 0
        if time + t.duration > t.deadline:
            deadline_penalty = -50

        return (
            priority_score
            + urgency_score
            + energy_penalty
            + deadline_penalty
        )

    tasks = sorted(pending, key=lambda t: -score_task(t))

    # -------------------------------
    # STEP 2 — CRITICAL TASK OVERRIDE
    # -------------------------------
    for t in tasks:
        if t.priority == "high":
            # If delaying this task risks missing it, do it now
            if time + t.duration >= t.deadline:
                return f"start_task('{t.id}')"

    # -------------------------------
    # STEP 3 — ENERGY MANAGEMENT
    # -------------------------------
    if energy < 30:
        for t in tasks:
            if t.priority == "high":
                if time + 2 + t.duration > t.deadline:
                    return f"start_task('{t.id}')"
        return "take_break(2)"

    # -------------------------------
    # STEP 4 — LOOKAHEAD FILTER
    # -------------------------------
    def safe_to_do(task):
        future_time = time + task.duration

        for other in tasks:
            if other.id == task.id:
                continue

            if future_time + other.duration > other.deadline:
                if other.priority == "high":
                    return False

        return True

    # -------------------------------
    # STEP 5 — PICK BEST SAFE TASK
    # -------------------------------
    for t in tasks:
        if time + t.duration <= t.deadline:
            if energy >= t.duration * 10:
                if safe_to_do(t):
                    return f"start_task('{t.id}')"

    # -------------------------------
    # STEP 6 — SECOND PASS
    # -------------------------------
    for t in tasks:
        if time + t.duration <= t.deadline:
            if energy >= t.duration * 10:
                return f"start_task('{t.id}')"

    # -------------------------------
    # STEP 7 — SMART BREAK
    # -------------------------------
    best = tasks[0]
    needed_energy = best.duration * 10

    if energy < needed_energy:
        needed = (needed_energy - energy + 9) // 10
        needed = max(1, min(2, needed))

        for t in tasks:
            if t.priority == "high":
                if time + needed + t.duration > t.deadline:
                    return f"start_task('{t.id}')"

        return f"take_break({needed})"

    # -------------------------------
    # STEP 8 — LAST RESORT
    # -------------------------------
    return f"start_task('{best.id}')"# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    all_passed = True
    for diff in ("easy", "medium", "hard"):
        print(f"\n{'='*60}")
        print(f"  {diff.upper()}")
        print("=" * 60)

        env = FocusEnv(difficulty=diff)
        obs = env.reset()
        print(env.get_observation_text())
        _ = env.state  # verify @property works

        score = None
        for step_n in range(20):
            action = smart_agent(obs)
            obs, rew, done, info = env.step(action)
            print(
                f"  Step {step_n+1:2d} | {action:<35} "
                f"| reward={rew.reward:+.1f} | done={done}"
            )
            if done:
                score = info["score"]
                print(f"\n  Score ({diff}): {score}")
                if not (0 < score < 1):
                    print(f"  ❌ SCORE OUT OF RANGE: {score}")
                    all_passed = False
                else:
                    print(f"  ✓  score strictly in (0, 1)")
                break

    print("\n" + ("✓  All scores in (0, 1)." if all_passed else "❌  FAILURES DETECTED."))
    sys.exit(0 if all_passed else 1)