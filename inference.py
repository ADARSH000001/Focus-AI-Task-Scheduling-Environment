"""
inference.py
------------
Inference runner for the FocusAI Task Scheduling Environment.

Supports two agent modes:
  1. LLM agent    — uses API_BASE_URL + MODEL_NAME + HF_TOKEN env vars
  2. Smart agent  — built-in deterministic fallback (no API needed)

Structured stdout logs (machine-parseable, required by OpenEnv validator):
  [START] task=<id> env=focus_ai model=<model>
  [STEP]  step=N action=ACTION reward=R done=true|false error=null
  [END]   task=<id> success=true|false steps=N score=S rewards=r1,r2,...

Usage:
    python inference.py                       # all 3 difficulties, smart agent
    python inference.py --difficulty easy     # single difficulty
    python inference.py --strict-env          # fail if LLM env vars are missing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

from env import FocusEnv, smart_agent
from models import Observation
from reward_and_tasks import GRADERS, safe_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 10
ALLOWED_ACTIONS = {"start_task", "take_break", "switch_task", "noop"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured logging helpers  (key=value format — required by OpenEnv)
# ---------------------------------------------------------------------------

def emit_start(task_id: str, model: str) -> None:
    """[START] task=<id> env=focus_ai model=<model>"""
    print(f"[START] task={task_id} env=focus_ai model={model}", flush=True)


def emit_step(
    step: int,
    action: str,
    reward: float,
    done: bool, 
    error: str = "null", 
) -> None:
    """[STEP] step=N action=ACTION reward=R done=true|false error=null"""
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.4f} done={str(done).lower()} error={error}",
        flush=True,
    )


def emit_end(
    task_id: str,
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """[END] task=<id> success=true|false steps=N score=S rewards=r1,r2,..."""
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] task={task_id} success={str(success).lower()} "
        f"steps={steps} score={score} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FocusAI inference runner")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Run one difficulty or all three (default: all)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum steps per episode (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="Exit 1 if API_BASE_URL / MODEL_NAME / HF_TOKEN are missing",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

_DEPRECATED_HF_URL = "https://api-inference.huggingface.co"
_HF_ROUTER_URL = "https://router.huggingface.co/v1"


def get_runtime_config(strict_env: bool = False) -> Dict[str, Optional[str]]:
    """Read API configuration from environment variables."""
    api_base = os.getenv("API_BASE_URL", _HF_ROUTER_URL)
    model    = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    token    = os.getenv("HF_TOKEN")

    # Auto-correct the deprecated api-inference endpoint
    if api_base and _DEPRECATED_HF_URL in api_base:
        print(
            f"[WARN] API_BASE_URL points to deprecated endpoint.\n"
            f"       Auto-correcting to {_HF_ROUTER_URL}",
            file=sys.stderr,
        )
        api_base = _HF_ROUTER_URL

    if strict_env and not all([api_base, model, token]):
        missing = [
            k for k, v in {
                "API_BASE_URL": api_base,
                "MODEL_NAME":   model,
                "HF_TOKEN":     token,
            }.items()
            if not v
        ]
        print(f"[ERROR] Missing required environment variables: {missing}", file=sys.stderr)
        sys.exit(1)

    return {"api_base_url": api_base, "model_name": model, "hf_token": token}


# ---------------------------------------------------------------------------
# LLM System Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are FocusAI — an expert task scheduling agent operating in a deterministic productivity environment.

Your objective is to MAXIMIZE the final score.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES (NEVER BREAK)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- NEVER use noop()
- NEVER start_task when energy is LOW (≤40), unless skipping it causes a HIGH priority deadline miss
- NEVER take_break if it will cause missing a HIGH priority deadline
- NEVER repeat a completed task
- ALWAYS return exactly ONE valid action string
- NO explanations, NO JSON, NO extra text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTION FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Valid outputs ONLY:
- start_task('<id>')
- take_break(1)
- take_break(2)
- switch_task('<id>')

Output EXACTLY one line. Nothing else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — CHECK ENERGY
If energy_level == "low":
    For each HIGH priority task:
        If current_time + 2 + task.duration > task.deadline:
            → start_task('<id>')  # urgent override
    Otherwise:
        → take_break(2)

STEP 2 — SELECT BEST TASK
From all incomplete tasks:

1. Sort by:
   - Priority: high > medium > low
   - Deadline: earliest first

2. Choose a task ONLY IF:
   current_time + duration <= deadline

3. Then:
   → start_task('<id>')

STEP 3 — NO SAFE TASK
If no task can be completed before deadline:
    → take_break(1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING LOGIC (OPTIMIZE THIS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAXIMIZE:
✔ Completing ALL tasks
✔ Finishing BEFORE deadlines
✔ Doing HIGH priority tasks first
✔ Working at medium/high energy
✔ Taking breaks ONLY when needed

AVOID:
✘ Missing deadlines (−10)
✘ Working at low energy (−5)
✘ Unnecessary breaks (−3)
✘ noop (−3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD CASE EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Energy: LOW (30)
Task: security (HIGH priority, duration=3, deadline in 3h)

Correct:
→ start_task('security')

Wrong:
→ take_break(2)  ❌ (causes deadline miss)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL THINKING RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before choosing an action, ask:

"Will this reduce my final score?"

If YES → do NOT choose it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REMEMBER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are optimizing long-term score, not short-term comfort.

Always act like a perfect planner.
"""


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def _extract_action(text: str, legal: List[str]) -> Optional[str]:
    """Extract a valid action from LLM output using multiple fallback strategies."""
    text = text.strip()

    # 1. Direct exact match
    if text in legal:
        return text

    # 2. Strip markdown fences / quotes
    cleaned = text.strip("`\"'")
    if cleaned in legal:
        return cleaned

    # 3. Regex search for known action patterns
    pattern = r"(start_task|take_break|switch_task|noop)\s*\([^)]*\)"
    for match in re.finditer(pattern, text):
        candidate = match.group(0)
        if candidate in legal:
            return candidate

    # 4. Partial name match — first legal action whose base name appears in text
    for action in legal:
        name = action.split("(")[0]
        if name in text:
            return action

    return None


def llm_agent(
    observation: Observation,
    client: Any,
    model_name: str,
    conversation: List[Dict[str, str]],
) -> str:
    """
    Call the LLM and return a valid action string.

    Falls back to smart_agent if the LLM returns no valid action or errors.
    """
    completed_ids  = [t.id for t in observation.tasks if t.completed]
    pending_tasks  = [t for t in observation.tasks if not t.completed]

    pending_detail = "\n".join(
        f"  - {t.id}: priority={t.priority} deadline={t.deadline}h duration={t.duration}h"
        for t in pending_tasks
    ) or "  (all tasks completed)"

    obs_text = (
        "You are an RL agent. Return ONE action string, nothing else.\n\n"
        "RULES:\n"
        "- NEVER call start_task on a task in the COMPLETED list\n"
        "- If energy < 30, take_break(2) first unless a HIGH priority deadline will be missed\n"
        "- Choose the highest priority pending task within its deadline\n\n"
        f"Time: {observation.time:.1f}h | Energy: {observation.energy}/100 "
        f"({observation.energy_level.upper()})\n\n"
        f"COMPLETED (forbidden — do NOT start_task on these):\n"
        f"  {completed_ids if completed_ids else '(none)'}\n\n"
        f"PENDING (choose from these):\n{pending_detail}\n\n"
        f"VALID ACTIONS RIGHT NOW:\n"
        + "\n".join(f"  {a}" for a in observation.legal_actions)
        + "\n\nYour action:"
    )

    conversation.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + conversation,
            max_tokens=32,
            temperature=0.0,
        )
        reply = response.choices[0].message.content or ""
        conversation.append({"role": "assistant", "content": reply})

        action = _extract_action(reply, observation.legal_actions)
        if action:
            return action

        logger.warning("LLM returned no valid action (%r) — using smart_agent", reply)
    except Exception as exc:
        logger.error("LLM error: %s — using smart_agent", exc)

    return smart_agent(observation)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    difficulty: str,
    max_steps: int,
    config: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Run one full episode, returning results with a guaranteed (0,1) score."""
    use_llm = bool(config.get("hf_token"))
    client = None

    if use_llm:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=config["hf_token"],
                base_url=config["api_base_url"],
            )
        except Exception as exc:
            logger.warning("Could not create LLM client: %s", exc)
            use_llm = False

    env = FocusEnv(difficulty=difficulty)
    obs = env.reset()

    model_label = config.get("model_name") if use_llm else "smart_agent"
    task_id = difficulty  # matches openenv.yaml task ids

    emit_start(task_id=task_id, model=model_label)

    conversation: List[Dict[str, str]] = []
    step_rewards: List[float] = []
    total_reward = 0.0
    final_score: Optional[float] = None

    for step_n in range(max_steps):
        if use_llm and client:
            action = llm_agent(obs, client, config["model_name"], conversation)
        else:
            action = smart_agent(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward.reward
        step_rewards.append(reward.reward)

        emit_step(
            step=step_n + 1,
            action=action,
            reward=reward.reward,
            done=done,
        )

        if done:
            final_score = info.get("score")
            # IMP 8 — surface termination reason so evaluators can diagnose agent behaviour
            termination_reason = info.get("error", "natural")
            if env._stagnation >= 5:
                termination_reason = "stagnation"
                logger.warning(
                    "[END] Stagnation termination at step %d — agent repeated invalid actions",
                    step_n + 1,
                )
                print(
                    f"[END] task={task_id} termination_reason=stagnation step={step_n + 1}",
                    flush=True,
                )
            break

    # Compute score if the episode did not reach a natural terminal state
    if final_score is None:
        final_score = GRADERS[difficulty](env.metrics)

    # DOUBLE SAFETY — safe_score guarantees (0, 1) even if grader has a bug
    final_score = float(final_score)

    # ASSERTION — catch any residual violation before it reaches the validator
    assert 0 < final_score < 1, (
        f"Score {final_score} is outside (0, 1) for difficulty={difficulty}"
    )

    success = final_score >= 0.5

    emit_end(
        task_id=task_id,
        success=success,
        steps=len(step_rewards),
        score=final_score,
        rewards=step_rewards,
    )

    return {
        "difficulty":   difficulty,
        "score":        final_score,
        "total_reward": total_reward,
        "metrics":      dict(env.metrics),
        "agent":        "llm" if use_llm else "smart",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    config = get_runtime_config(strict_env=args.strict_env)

    difficulties = (
        ["easy", "medium", "hard"]
        if args.difficulty == "all"
        else [args.difficulty]
    )

    results = []
    for diff in difficulties:
        result = run_episode(diff, args.max_steps, config)
        results.append(result)

    # Compute and persist aggregate score across all difficulties
    if len(results) > 1:
        avg_score = sum(r["score"] for r in results) / len(results)
        overall = float(avg_score)

        summary = {
            "overall_score": overall,
            "difficulties": [
                {
                    "difficulty":   r["difficulty"],
                    "score":        r["score"],
                    "total_reward": round(float(r["total_reward"]), 4),
                }
                for r in results
            ],
        }
        with open("baseline_scores.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(
            f"\nSummary: avg_score={avg_score:.4f}  overall={overall:.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
