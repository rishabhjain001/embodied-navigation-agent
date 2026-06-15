from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    step: int
    action: str  # "move_forward" | "turn_left" | "turn_right" | "stop"
    reward: float
    agent_pos: tuple[int, int]  # (row, col)


@dataclass
class EvaluationMetrics:
    path_length: int              # count of move_forward actions
    steps_to_goal: int | None     # None until terminal
    collision_count: int
    navigation_efficiency: float | None  # None until terminal success
    success_rate: float | None    # None until >= 1 session complete


@dataclass
class Session:
    session_id: str               # UUID4 string
    instruction: str
    goal: dict                    # parsed Goal dict
    env: Any                      # gym.Env — typed loosely to avoid circular imports
    policy: Any                   # SB3 PPO
    step_count: int
    status: str                   # "active" | "success" | "failure"
    failure_reason: str | None    # "timeout" | "incorrect_stop" | None
    step_log: list                # list[StepRecord]
    metrics: EvaluationMetrics
    agent_start_pos: tuple[int, int]
    target_pos: tuple[int, int]
    collision_count: int
    last_obs_image: Any = None    # most recent obs["image"] for CLIP encoding
    _counted: bool = False        # whether this session was added to experiment stats


_sessions: dict[str, Session] = {}
_experiment_stats: dict[str, int] = {"total": 0, "successes": 0}


def create_session(instruction: str, goal: dict, env: Any, policy: Any) -> Session:
    session_id = str(uuid.uuid4())
    metrics = EvaluationMetrics(
        path_length=0,
        steps_to_goal=None,
        collision_count=0,
        navigation_efficiency=None,
        success_rate=None,
    )
    session = Session(
        session_id=session_id,
        instruction=instruction,
        goal=goal,
        env=env,
        policy=policy,
        step_count=0,
        status="active",
        failure_reason=None,
        step_log=[],
        metrics=metrics,
        agent_start_pos=(0, 0),
        target_pos=(0, 0),
        collision_count=0,
    )
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)


def update_session(session: Session) -> None:
    # Count terminal sessions once. The session is stored by reference, so we
    # use a flag rather than comparing previous status (which is already mutated).
    if not session._counted and session.status in ("success", "failure"):
        _experiment_stats["total"] += 1
        if session.status == "success":
            _experiment_stats["successes"] += 1
        session._counted = True
    _sessions[session.session_id] = session


def get_success_rate() -> float | None:
    total = _experiment_stats["total"]
    if total == 0:
        return None
    return _experiment_stats["successes"] / total
