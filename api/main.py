import base64
import io
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image as PILImage

import gymnasium as gym

from agent.env import make_env, get_object_positions
from agent.language import parse_instruction
from agent.policy import load_policy, build_obs_space, predict_action, make_observation
from agent.reward import compute_reward, manhattan_distance
from agent.vision import encode_frame
from api.models import (
    ConfigResponse,
    EvaluationMetricsModel,
    StartSessionRequest,
    StartSessionResponse,
    StatusResponse,
    StepRecordModel,
    StepResponse,
)
from api.session import (
    Session,
    StepRecord,
    create_session,
    get_session,
    get_success_rate,
    update_session,
)

logging.basicConfig(level=logging.INFO)

_config: dict | None = None
_policy = None  # loaded PPO

ACTION_NAMES = {0: "move_forward", 1: "turn_left", 2: "turn_right", 3: "stop"}


class _DummyEnv(gym.Env):
    """Minimal env with matching obs/action spaces for PPO.load()."""
    observation_space = build_obs_space()
    action_space = gym.spaces.Discrete(4)

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _policy

    load_dotenv()

    REQUIRED = [
        "GEMINI_API_KEY",
        "MODEL_PATH",
        "GRID_SIZE",
        "STEP_BUDGET",
        "OBJECT_COUNT",
        "REWARD_STEP_CLOSER",
        "REWARD_STEP_AWAY",
        "REWARD_COLLISION",
        "REWARD_SUCCESS",
        "REWARD_TIMEOUT",
    ]
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        for k in missing:
            logging.error(f"missing required env var: {k}")
        sys.exit(1)

    grid_size = int(os.getenv("GRID_SIZE"))
    step_budget = int(os.getenv("STEP_BUDGET"))
    object_count = int(os.getenv("OBJECT_COUNT"))
    model_path = os.getenv("MODEL_PATH")

    float(os.getenv("REWARD_STEP_CLOSER"))
    float(os.getenv("REWARD_STEP_AWAY"))
    float(os.getenv("REWARD_COLLISION"))
    float(os.getenv("REWARD_SUCCESS"))
    float(os.getenv("REWARD_TIMEOUT"))

    try:
        import agent.vision as _vis  # noqa: F401
        # fast pixel projection matches Colab training — no CLIP needed
        if os.getenv("USE_FAST_PROJECTION", "false").lower() == "true":
            _vis.USE_FAST_PROJECTION = True
            logging.info("using fast pixel projection (no CLIP)")
    except ImportError as exc:
        logging.error(f"failed to import vision module: {exc}")
        sys.exit(1)

    try:
        _policy = load_policy(model_path, _DummyEnv())
    except RuntimeError as exc:
        logging.error(str(exc))
        sys.exit(1)

    _config = {
        "step_budget": step_budget,
        "grid_size": grid_size,
        "object_count": object_count,
        "model_path": model_path,
    }

    logging.info("startup complete — config loaded, CLIP and policy ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    if _config is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "configuration not loaded"},
        )
    return ConfigResponse(**_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_frame_b64(frame: np.ndarray) -> str:
    """uint8 RGB array → base64-encoded PNG string."""
    img = PILImage.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _unwrap_base_env(env):
    """Walk wrapper chain to reach the underlying _NavigationEnv."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _metrics_model(session: Session) -> EvaluationMetricsModel:
    return EvaluationMetricsModel(
        path_length=session.metrics.path_length,
        steps_to_goal=session.metrics.steps_to_goal,
        collision_count=session.metrics.collision_count,
        navigation_efficiency=session.metrics.navigation_efficiency,
        success_rate=session.metrics.success_rate,
    )


def _step_log_model(session: Session) -> list[StepRecordModel]:
    return [
        StepRecordModel(
            step=r.step,
            action=r.action,
            reward=r.reward,
            agent_pos=list(r.agent_pos),
        )
        for r in session.step_log
    ]


def _execute_step(session: Session, step_budget: int) -> bool:
    """Run one step on the session in-place. Returns done flag.

    Mutates session.step_count, step_log, metrics, status, failure_reason,
    collision_count, and last_obs_image. Caller must call update_session().
    """
    base_env = _unwrap_base_env(session.env)
    prev_pos = (int(base_env.agent_pos[1]), int(base_env.agent_pos[0]))  # (row, col)

    # Use the frame stored from the previous step/reset for CLIP encoding
    obs_image: np.ndarray = session.last_obs_image
    frame_embedding = encode_frame(obs_image)
    obs_vector = make_observation(frame_embedding, session.goal)
    action = predict_action(session.policy, obs_vector)

    next_obs, _, terminated, truncated, info = session.env.step(action)
    curr_pos = (int(base_env.agent_pos[1]), int(base_env.agent_pos[0]))

    done_stop = (action == 3)
    reward, done = compute_reward(
        action=action,
        prev_agent_pos=prev_pos,
        curr_agent_pos=curr_pos,
        target_pos=session.target_pos,
        step_count=session.step_count + 1,
        step_budget=step_budget,
        done_stop=done_stop,
    )

    session.step_count += 1
    action_name = ACTION_NAMES[action]

    session.step_log.append(StepRecord(
        step=session.step_count,
        action=action_name,
        reward=reward,
        agent_pos=curr_pos,
    ))

    if action == 0:  # move_forward
        session.metrics.path_length += 1

    # collision_count in info is cumulative from the env; store it directly
    session.collision_count = info.get("collision_count", 0)
    session.metrics.collision_count = session.collision_count

    session.last_obs_image = next_obs["image"]

    if done:
        session.metrics.steps_to_goal = session.step_count

        if action == 3 and manhattan_distance(curr_pos, session.target_pos) <= 1:
            session.status = "success"
        elif session.step_count >= step_budget:
            session.status = "failure"
            session.failure_reason = "timeout"
        else:
            session.status = "failure"
            session.failure_reason = "incorrect_stop"

        if session.status == "success":
            dist = manhattan_distance(session.agent_start_pos, session.target_pos)
            pl = session.metrics.path_length
            session.metrics.navigation_efficiency = (
                dist / pl if pl > 0 and dist > 0 else 1.0
            )

        session.metrics.success_rate = get_success_rate()

    return done


# ---------------------------------------------------------------------------
# POST /session/start
# ---------------------------------------------------------------------------

@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    try:
        env = make_env(
            grid_size=_config["grid_size"],
            object_count=_config["object_count"],
        )
        obs, _ = env.reset()

        goal = parse_instruction(request.instruction, env.object_types)

        positions = get_object_positions(env)
        target_key = next(
            (k for k in positions if k.startswith(goal.get("target") or "")),
            None,
        )
        target_pos = positions.get(target_key, (0, 0)) if target_key else (0, 0)

        session = create_session(request.instruction, goal, env, _policy)

        base_env = _unwrap_base_env(env)
        agent_pos = (int(base_env.agent_pos[1]), int(base_env.agent_pos[0]))

        session.agent_start_pos = agent_pos
        session.target_pos = target_pos
        # store initial frame so the first _execute_step call has something to encode
        session.last_obs_image = obs["image"]
        update_session(session)

        return StartSessionResponse(
            session_id=session.session_id,
            pov_frame=_encode_frame_b64(obs["image"]),
            goal=goal,
            status=session.status,
        )
    except Exception as exc:
        logging.error(f"start_session failed: {exc}")
        raise HTTPException(status_code=500, detail="Failed to initialize session")


# ---------------------------------------------------------------------------
# POST /session/{session_id}/step
# ---------------------------------------------------------------------------

@app.post("/session/{session_id}/step", response_model=StepResponse)
async def step_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    if session.status != "active":
        raise HTTPException(status_code=409, detail="session is already complete")

    done = _execute_step(session, _config["step_budget"])
    update_session(session)

    return StepResponse(
        step=session.step_count,
        action=session.step_log[-1].action,
        reward=session.step_log[-1].reward,
        agent_pos=list(session.step_log[-1].agent_pos),
        pov_frame=_encode_frame_b64(session.last_obs_image),
        done=done,
        status=session.status,
        metrics=_metrics_model(session),
    )


# ---------------------------------------------------------------------------
# GET /session/{session_id}/status
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/status", response_model=StatusResponse)
async def session_status(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")

    return StatusResponse(
        session_id=session.session_id,
        status=session.status,
        failure_reason=session.failure_reason,
        step_log=_step_log_model(session),
        metrics=_metrics_model(session),
    )


# ---------------------------------------------------------------------------
# POST /session/{session_id}/run
# ---------------------------------------------------------------------------

@app.post("/session/{session_id}/run", response_model=StatusResponse)
async def run_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    if session.status != "active":
        raise HTTPException(status_code=409, detail="session is already complete")

    run_error: list[str] = []

    def _run():
        try:
            step_budget = _config["step_budget"]
            while session.status == "active":
                done = _execute_step(session, step_budget)
                update_session(session)
                if done:
                    break
        except Exception as exc:
            run_error.append(str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=60)

    if thread.is_alive():
        # thread exceeded 60-second wall-clock limit
        session.status = "failure"
        session.failure_reason = "timeout"
        session.metrics.steps_to_goal = session.step_count
        session.metrics.success_rate = get_success_rate()
        update_session(session)

    if run_error:
        logging.error(f"run_session failed: {run_error[0]}")
        raise HTTPException(status_code=500, detail="run failed")

    return StatusResponse(
        session_id=session.session_id,
        status=session.status,
        failure_reason=session.failure_reason,
        step_log=_step_log_model(session),
        metrics=_metrics_model(session),
    )
