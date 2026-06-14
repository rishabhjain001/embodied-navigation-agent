import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from api.models import (
    ConfigResponse,
    EvaluationMetricsModel,
    StartSessionRequest,
    StepRecordModel,
)

# Feature: embodied-navigation-agent, Property 18: Request validation returns 422 with field details


# P18a — Missing required fields
def test_start_session_missing_instruction_raises():
    """StartSessionRequest with no instruction raises ValidationError (FastAPI → 422)."""
    with pytest.raises(ValidationError):
        StartSessionRequest()


# P18b — Instruction exceeds max_length=500
@given(st.text(min_size=501, max_size=1000))
@settings(max_examples=100)
def test_start_session_instruction_too_long_raises(text):
    # Feature: embodied-navigation-agent, Property 18: Request validation returns 422 with field details
    """Instructions longer than 500 chars are rejected with ValidationError."""
    with pytest.raises(ValidationError):
        StartSessionRequest(instruction=text)


# P18c — Instruction at or below max_length is valid
@given(st.text(min_size=0, max_size=500))
@settings(max_examples=100)
def test_start_session_instruction_at_limit_is_valid(text):
    # Feature: embodied-navigation-agent, Property 18: Request validation returns 422 with field details
    """Instructions of ≤500 chars are always accepted without raising."""
    model = StartSessionRequest(instruction=text)
    assert model.instruction == text


# P18d — Valid request creates model with correct instruction value
@given(st.text(min_size=1, max_size=100))
@settings(max_examples=100)
def test_start_session_stores_instruction(text):
    # Feature: embodied-navigation-agent, Property 18: Request validation returns 422 with field details
    """Model stores instruction exactly as provided."""
    model = StartSessionRequest(instruction=text)
    assert model.instruction == text


# Additional model validation tests


def test_evaluation_metrics_nullable_fields_valid():
    """EvaluationMetricsModel accepts None for nullable fields."""
    model = EvaluationMetricsModel(
        path_length=5,
        steps_to_goal=None,
        collision_count=2,
        navigation_efficiency=None,
        success_rate=None,
    )
    assert model.steps_to_goal is None
    assert model.navigation_efficiency is None
    assert model.success_rate is None


def test_step_record_valid_int_list_agent_pos():
    """StepRecordModel accepts a valid [row, col] int list for agent_pos."""
    model = StepRecordModel(step=1, action="move_forward", reward=0.1, agent_pos=[3, 4])
    assert model.agent_pos == [3, 4]


def test_config_response_requires_all_fields():
    """ConfigResponse raises ValidationError when any required field is missing."""
    with pytest.raises(ValidationError):
        ConfigResponse(step_budget=200, grid_size=10, object_count=5)  # missing model_path

    with pytest.raises(ValidationError):
        ConfigResponse(step_budget=200, grid_size=10, model_path="models/policy.zip")  # missing object_count

    # All four fields present — should succeed
    model = ConfigResponse(
        step_budget=200,
        grid_size=10,
        object_count=5,
        model_path="models/policy.zip",
    )
    assert model.step_budget == 200
    assert model.model_path == "models/policy.zip"


# ---------------------------------------------------------------------------
# P16 & P17 — HTTP session error responses
# ---------------------------------------------------------------------------
# These tests bypass the heavy lifespan startup (CLIP + PPO + env vars) by
# patching the lifespan to a no-op and setting module-level state directly.

import uuid as _uuid_mod
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

import api.main as _main_module
import api.session as _session_module
from api.session import EvaluationMetrics, Session


def _make_client() -> TestClient:
    """Return a TestClient with startup bypassed."""
    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    # Patch the app's lifespan so TestClient never runs the real startup.
    with patch.object(_main_module.app.router, "lifespan_context", _noop_lifespan):
        _main_module._config = {
            "step_budget": 200,
            "grid_size": 10,
            "object_count": 5,
            "model_path": "models/policy.zip",
        }
        _main_module._policy = MagicMock()
        client = TestClient(_main_module.app, raise_server_exceptions=False)
    return client


def _blank_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        path_length=0,
        steps_to_goal=None,
        collision_count=0,
        navigation_efficiency=None,
        success_rate=None,
    )


def _terminal_session(session_id: str, status: str) -> Session:
    return Session(
        session_id=session_id,
        instruction="go to the chair",
        goal={"target": "chair", "attribute": None, "context": None, "error": None},
        env=MagicMock(),
        policy=MagicMock(),
        step_count=5,
        status=status,
        failure_reason=None if status == "success" else "timeout",
        step_log=[],
        metrics=_blank_metrics(),
        agent_start_pos=(0, 0),
        target_pos=(1, 1),
        collision_count=0,
    )


# Feature: embodied-navigation-agent, Property 16: 404 for unknown session IDs

@given(st.uuids().map(str))
@settings(max_examples=50)
def test_step_unknown_session_returns_404(session_id):
    # Validates: Requirements 6.2, 6.5
    _session_module._sessions.clear()
    client = _make_client()
    resp = client.post(f"/session/{session_id}/step")
    assert resp.status_code == 404, (
        f"Expected 404 for unknown session {session_id!r}, got {resp.status_code}"
    )


@given(st.uuids().map(str))
@settings(max_examples=50)
def test_status_unknown_session_returns_404(session_id):
    # Validates: Requirements 6.2, 6.5
    _session_module._sessions.clear()
    client = _make_client()
    resp = client.get(f"/session/{session_id}/status")
    assert resp.status_code == 404, (
        f"Expected 404 for unknown session {session_id!r}, got {resp.status_code}"
    )


@given(st.uuids().map(str))
@settings(max_examples=50)
def test_run_unknown_session_returns_404(session_id):
    # Validates: Requirements 6.2, 6.5
    _session_module._sessions.clear()
    client = _make_client()
    resp = client.post(f"/session/{session_id}/run")
    assert resp.status_code == 404, (
        f"Expected 404 for unknown session {session_id!r}, got {resp.status_code}"
    )


# Feature: embodied-navigation-agent, Property 17: 409 for terminated sessions

@given(st.sampled_from(["success", "failure"]))
@settings(max_examples=50)
def test_step_terminal_session_returns_409(terminal_status):
    # Validates: Requirements 6.2, 6.5
    session_id = str(_uuid_mod.uuid4())
    _session_module._sessions.clear()
    _session_module._sessions[session_id] = _terminal_session(session_id, terminal_status)
    client = _make_client()
    resp = client.post(f"/session/{session_id}/step")
    assert resp.status_code == 409, (
        f"Expected 409 for terminal ({terminal_status!r}) session, got {resp.status_code}"
    )
