from pydantic import BaseModel, Field
from typing import Optional


class StartSessionRequest(BaseModel):
    instruction: str = Field(max_length=500)


class StartSessionResponse(BaseModel):
    session_id: str
    pov_frame: str  # base64-encoded PNG
    goal: dict
    status: str


class EvaluationMetricsModel(BaseModel):
    path_length: int
    steps_to_goal: int | None
    collision_count: int
    navigation_efficiency: float | None
    success_rate: float | None


class StepRecordModel(BaseModel):
    step: int
    action: str
    reward: float
    agent_pos: list[int]  # [row, col]


class StepResponse(BaseModel):
    step: int
    action: str
    reward: float
    agent_pos: list[int]  # [row, col]
    pov_frame: str  # base64-encoded PNG
    done: bool
    status: str
    metrics: EvaluationMetricsModel


class StatusResponse(BaseModel):
    session_id: str
    status: str
    failure_reason: str | None
    step_log: list[StepRecordModel]
    metrics: EvaluationMetricsModel


class RunResponse(BaseModel):
    # Structurally identical to StatusResponse — kept separate for endpoint clarity
    session_id: str
    status: str
    failure_reason: str | None
    step_log: list[StepRecordModel]
    metrics: EvaluationMetricsModel


class ConfigResponse(BaseModel):
    step_budget: int
    grid_size: int
    object_count: int
    model_path: str
