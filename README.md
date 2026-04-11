---
title: Focus AI Task Scheduling Environment
sdk: docker
app_port: 7860
---

# 🧠 Focus AI — Task Scheduling RL Environment

> A real-world reinforcement learning environment where an AI agent manages tasks, deadlines, and energy to maximize productivity while avoiding burnout.

---

## 1. Project Overview

Focus AI is an **OpenEnv-compatible RL environment** where an agent must:

- Complete tasks before their deadlines
- Manage energy levels to avoid burnout
- Prioritize high-importance work
- Recover efficiently when exhausted

The environment simulates a real knowledge-worker's day — making it directly useful for agent benchmarking, policy evaluation, and productivity research.

---

## 2. Problem Statement

Knowledge workers constantly balance urgent tasks, deadlines, context switching, and cognitive fatigue. Focus AI models that decision process as a structured RL environment where every action affects future productivity and wellbeing.

---

## 3. Environment Design

### Action Space

String-based actions passed as plain text:

| Action | Description |
|---|---|
| `start_task('<id>')` | Work on a task (costs time + energy) |
| `take_break(<hours>)` | Rest to recover energy (costs time) |
| `switch_task('<id>')` | Change current focus (0.5h overhead) |
| `noop()` | Do nothing (penalized -3) |

### Observation Space

Typed Pydantic `Observation` model returned from `reset()` and `step()`:

| Field | Type | Description |
|---|---|---|
| `time` | float | Current hour (starts at 9, ends at 24) |
| `energy` | int | Energy level 0–100 |
| `energy_level` | str | `low` (≤40) / `medium` (41–70) / `high` (>70) |
| `tasks` | List[Task] | All tasks with status |
| `current_task` | str \| None | Active task id |
| `recent_actions` | List[str] | Last 3 actions taken |
| `goal` | str | Episode objective |
| `legal_actions` | List[str] | Valid actions at this state |

---

## 4. Reward Function

Reward is calculated per step and clamped to `[-20, +20]`.

| Condition | Reward |
|---|---|
| Complete high-priority task | +15 base (+2 priority bonus) |
| Complete medium-priority task | +12 base |
| Complete low-priority task | +10 base (-2 priority penalty) |
| Finish before deadline | +7 total (+5 deadline + +2 early) |
| Miss deadline | -10 |
| Work at high energy | +3 |
| Work at medium energy | +1 |
| Work at low energy | -11 (-8 delta -3 extra) |
| Smart break at low energy | +7 (+5 delta +2 extra) |
| Unnecessary break at high energy | -3 |
| `noop()` | -3 |
| Invalid action | -3 (flat) |
| Burnout (energy ≤ 0) | -10 on top of step reward |

---

## 5. Task Difficulty Levels

### 🟢 Easy
- 2 tasks, high starting energy (80/100)
- `report` (high priority, deadline 12h, 1h) + `inbox` (low, 18h, 1h)
- Scoring: 60% task completion + 40% on-time delivery

### 🟡 Medium
- 3 tasks, medium energy (60/100)
- `pr_review` (high, 11h, 2h) + `client_call` (medium, 14h, 2h) + `docs` (low, 18h, 1h)
- Scoring: 40% completion + 35% on-time + 25% energy efficiency

### 🔴 Hard
- 6 tasks, medium-low energy (55/100)
- 2 critical blockers (`incident` + `security`) + 4 mixed-priority tasks
- Requires genuine multi-step planning under resource constraints
- Scoring: 35% completion + 30% on-time + 20% energy + 15% priority quality

> **Note:** All difficulties use randomized task generation by default (`get_random_task()`), preventing LLMs from memorizing fixed scenarios. Deterministic versions (`get_easy_task()`, etc.) are available for debugging.

---

## 6. Scoring System

Each difficulty has a deterministic grader. Raw scores are mapped to **strictly inside (0, 1)** via `safe_score()`:

```
safe_score(raw) = 0.01 + 0.98 × raw,    raw ∈ [0, 1]
```

- **Minimum possible score**: 0.01 (nothing done)
- **Maximum possible score**: 0.99 (perfect episode)
- **Never 0.0 or 1.0** — guaranteed by linear mapping + runtime assertion

| Grader | Weights |
|---|---|
| `grade_easy` | 60% completion + 40% on-time |
| `grade_medium` | 40% completion + 35% on-time + 25% energy efficiency |
| `grade_hard` | 35% completion + 30% on-time + 20% energy + 15% priority quality |

---

## 7. Agent Modes

### Smart Agent (default)
A built-in deterministic baseline. No API keys required. Follows priority + deadline ordering with energy-aware break decisions.

### LLM Agent
Uses any OpenAI-compatible API. Configured via environment variables. Falls back to smart agent automatically on API errors or invalid outputs.

The LLM receives a structured system prompt covering:
- Energy-first decision rules
- Priority + deadline sorting strategy
- Reward penalties to avoid
- Hard override cases (e.g., work through low energy if deadline is imminent)

---

## 8. Episode Termination

An episode ends when ANY of these is true:

1. All tasks completed
2. Energy reaches 0 (burnout)
3. Time reaches 24h (end of day)
4. 5 consecutive invalid or noop steps (stagnation guard)

Maximum steps per episode: **10**

---

## 9. OpenEnv Compliance

| Requirement | Status |
|---|---|
| `reset()` returns typed `Observation` | ✅ |
| `step()` returns `(obs, reward, done, info)` | ✅ |
| `state` is a `@property` returning `Observation` | ✅ |
| Typed Pydantic models (`Observation`, `Action`, `Reward`) | ✅ |
| `legal_actions` field in `Observation` | ✅ |
| `openenv.yaml` with `entry_point: env:FocusEnv` | ✅ |
| 3 tasks with graders returning strictly `(0, 1)` | ✅ |
| `inference.py` with OpenAI client | ✅ |
| Structured logs `[START]` `[STEP]` `[END]` in `key=value` format | ✅ |
| Dockerfile builds and runs | ✅ |
| HF Space deployable | ✅ |

---

## 10. How to Run

### Local — Smart Agent (no API key needed)
```bash
pip install -r requirements.txt
python inference.py
```

### Local — Single Difficulty
```bash
python inference.py --difficulty easy     # or medium / hard
python inference.py --difficulty all      # runs all three (default)
```

### Local — LLM Agent
```bash
pip install -r requirements.txt
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_xxx"
python inference.py --strict-env
```

### Docker
```bash
docker build -t focus-ai .
docker run --rm -p 7860:7860 focus-ai
```

### Self-test (env only)
```bash
python env.py
```

### With uv (recommended for reproducible installs)
```bash
uv sync          # installs from uv.lock — exact versions
uv run python inference.py
```

---

## 11. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | No | `meta-llama/Llama-3.1-8B-Instruct` | Model to use |
| `HF_TOKEN` | For LLM mode | — | Hugging Face API token |

> If `HF_TOKEN` is not set, the runner falls back to the smart agent silently. Use `--strict-env` to fail loudly instead.

---

## 12. Structured Log Format

All episode output follows `key=value` format for machine parsing:

```
[START] task=easy env=focus_ai model=smart_agent
[STEP]  step=1 action=start_task('report') reward=22.0000 done=false error=null
[STEP]  step=2 action=start_task('inbox') reward=12.0000 done=true error=null
[END]   task=easy success=true steps=2 score=0.9702 rewards=22.0000,12.0000
```

---

## 13. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Status check |
| GET | `/health` | Health check |
| GET | `/tasks` | List task definitions |
| POST | `/reset` | Start new episode |
| POST | `/step` | Apply action |
| GET | `/state` | Current observation + metrics |
| GET | `/grade/{difficulty}` | Current episode score |
| GET | `/validate` | OpenEnv compliance check |

---

## 14. Project Structure

```
├── env.py                  # FocusEnv class + smart_agent baseline
├── reward_and_tasks.py     # Reward function, task generators, graders, safe_score
├── inference.py            # Inference runner (LLM + smart agent)
├── models.py               # Pydantic models (Observation, Action, Reward, Task)
├── server/
│   └── app.py              # FastAPI server
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Project metadata + entry points
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies (pip)
├── uv.lock                 # Locked dependency tree (uv)
└── README.md
```

---

## 15. HF Space Deployment

1. Create HF Space with SDK type **Docker**
2. Upload this folder
3. Set secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`
4. Verify: `GET /health` → `{"status":"healthy"}`

---

## 16. Scoring Output

After running all difficulties, results are written to `baseline_scores.json`:

```json
{
  "overall_score": 0.7412,
  "difficulties": [
    { "difficulty": "easy",   "score": 0.9702, "total_reward": 34.0 },
    { "difficulty": "medium", "score": 0.7105, "total_reward": 21.0 },
    { "difficulty": "hard",   "score": 0.5429, "total_reward": 12.0 }
  ]
}
```