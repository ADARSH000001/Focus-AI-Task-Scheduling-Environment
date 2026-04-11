"""
models.py
---------
Pydantic v2 data models for the FocusAI Task Scheduling Environment.

Four models:
  Task        — a single schedulable work item
  Observation — full agent-visible state at each step
  Action      — thin wrapper around a raw action string
  Reward      — step reward with diagnostic breakdown
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single schedulable task in the agent's workday."""

    id: str = Field(description="Unique identifier used in action strings, e.g. 'report'")
    name: str = Field(description="Human-readable task description")
    priority: str = Field(description="'high' | 'medium' | 'low'")
    deadline: int = Field(description="Hour of day by which the task must be completed")
    duration: int = Field(description="Hours of focused work required to finish")
    category: str = Field(default="general", description="Task domain, e.g. 'engineering', 'writing'")
    completed: bool = Field(default=False, description="True once the task has been finished")


class Observation(BaseModel):
    """Everything the agent can perceive at one timestep."""

    time: float = Field(description="Current hour of day (9.0 → 24.0)")
    energy: int = Field(description="Cognitive energy 0–100; reaching 0 causes burnout")
    energy_level: str = Field(description="Qualitative band: 'low' | 'medium' | 'high'")
    tasks: List[Task] = Field(description="All tasks with their current completion status")
    current_task: Optional[str] = Field(default=None, description="ID of the currently active task")
    recent_actions: List[str] = Field(default_factory=list, description="Last 3 actions taken")
    goal: str = Field(default="", description="Plain-text episode objective for the agent")
    legal_actions: List[str] = Field(default_factory=list, description="All valid actions right now")


class Action(BaseModel):
    """Wrapper for a single raw action string."""

    action: str = Field(description="e.g. start_task('report') | take_break(2) | noop()")


class Reward(BaseModel):
    """Step reward with a diagnostic breakdown."""

    reward: float = Field(description="Numeric reward for this step, clamped to [-20, +20]")
    task_completed: bool = Field(default=False, description="A task was finished this step")
    before_deadline: bool = Field(default=False, description="The finished task beat its deadline")
    missed_deadline: bool = Field(default=False, description="The finished task missed its deadline")
    priority: Optional[str] = Field(default=None, description="Priority of the completed task, if any")
