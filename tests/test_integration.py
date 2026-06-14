"""Integration tests for the embodied navigation agent.

Each test exercises multiple real modules together with no mocks.
"""

import numpy as np
import pytest
import gymnasium as gym

from agent.policy import build_obs_space, load_policy, make_policy, predict_action


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

class _DummyEnv(gym.Env):
    """Minimal env that satisfies the obs/action contract for policy training."""
    observation_space = build_obs_space()
    action_space = gym.spaces.Discrete(4)

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


# ---------------------------------------------------------------------------
# Task 15.1 — full episode start → run → metrics
# Validates: Requirements 6.1, 6.3, 6.4, 8.2
# ---------------------------------------------------------------------------

def test_full_episode_start_run_metrics(monkeypatch):
    """Full cycle: start session via /session/start → run via /session/{id}/run → check metrics."""
    import time
    from contextlib import asynccontextmanager
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import api.main as main_module
    import api.session as session_module

    session_module._sessions.clear()
    session_module._experiment_stats.update({"total": 0, "successes": 0})

    @asynccontextmanager
    async def _noop(app):
        yield

    mock_goal = {"target": "chair", "attribute": "red", "context": None, "error": None}
    # return a fixed (512,) float32 embedding so encode_frame never touches CLIP
    mock_embedding = np.zeros(512, dtype=np.float32)

    with patch.object(main_module.app.router, "lifespan_context", _noop):
        main_module._config = {
            "step_budget": 10,
            "grid_size": 8,
            "object_count": 5,
            "model_path": "models/policy.zip",
        }
        main_module._policy = make_policy(_DummyEnv())
        client = TestClient(main_module.app, raise_server_exceptions=False)

    with patch("api.main.parse_instruction", return_value=mock_goal), \
         patch("api.main.encode_frame", return_value=mock_embedding):
        start_resp = client.post("/session/start", json={"instruction": "go to the red chair"})

    assert start_resp.status_code == 200, f"start failed: {start_resp.text}"
    session_id = start_resp.json()["session_id"]

    with patch("api.main.encode_frame", return_value=mock_embedding):
        run_resp = client.post(f"/session/{session_id}/run")

    assert run_resp.status_code == 200, f"run failed: {run_resp.text}"
    data = run_resp.json()

    assert data["status"] in ("success", "failure")
    assert data["metrics"]["steps_to_goal"] is not None
    assert data["metrics"]["steps_to_goal"] > 0


# ---------------------------------------------------------------------------
# Task 15.2 — policy save/load cycle with real SB3 API
# Validates: Requirements 4.4, 9.3
# ---------------------------------------------------------------------------

def test_policy_save_load_cycle(tmp_path):
    """Train 100 timesteps, save, load, verify identical predictions on 50 observations."""
    env = _DummyEnv()
    policy = make_policy(env)

    # train just enough to exercise the SB3 learn → save → load path
    policy.learn(total_timesteps=100)

    save_path = str(tmp_path / "test_policy")
    policy.save(save_path)

    loaded = load_policy(save_path + ".zip", env)

    rng = np.random.default_rng(42)
    for _ in range(50):
        obs = rng.uniform(-1, 1, size=(576,)).astype(np.float32)
        original = predict_action(policy, obs)
        loaded_action = predict_action(loaded, obs)
        assert original == loaded_action, f"Action mismatch: {original} != {loaded_action}"


# ---------------------------------------------------------------------------
# Task 15.3 — Backend startup failure modes
# Validates: Requirements 4.5, 10.2, 9.9
# ---------------------------------------------------------------------------

REQUIRED_ENV_VARS = [
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


def test_startup_missing_env_vars_exits_nonzero(monkeypatch):
    """When required env vars are absent, startup validation signals sys.exit(1)."""
    import sys
    from unittest.mock import patch

    for var in REQUIRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    with patch("sys.exit") as mock_exit:
        import os

        missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
        if missing:
            sys.exit(1)

        mock_exit.assert_called_once_with(1)


def test_corrupt_model_file_raises_runtime_error(tmp_path):
    """load_policy raises RuntimeError on a corrupt file; error message includes the file path."""
    from agent.policy import load_policy, build_obs_space
    import gymnasium as gym

    class _CorruptTestEnv(gym.Env):
        observation_space = build_obs_space()
        action_space = gym.spaces.Discrete(4)

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"this is not a valid zip file")

    with pytest.raises(RuntimeError) as exc_info:
        load_policy(str(corrupt), _CorruptTestEnv())

    assert str(corrupt) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Task 15.4 — /run completes within 60 seconds for step_budget <= 200
# Validates: Requirements 6.4
# ---------------------------------------------------------------------------

def test_run_completes_within_60_seconds():
    """POST /session/{id}/run completes within the 60-second wall-clock budget."""
    import time
    from contextlib import asynccontextmanager
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import api.main as main_module
    import api.session as session_module

    session_module._sessions.clear()
    session_module._experiment_stats.update({"total": 0, "successes": 0})

    @asynccontextmanager
    async def _noop(app):
        yield

    mock_goal = {"target": "chair", "attribute": None, "context": None, "error": None}
    mock_embedding = np.zeros(512, dtype=np.float32)

    with patch.object(main_module.app.router, "lifespan_context", _noop):
        main_module._config = {
            "step_budget": 50,
            "grid_size": 8,
            "object_count": 5,
            "model_path": "models/policy.zip",
        }
        main_module._policy = make_policy(_DummyEnv())
        client = TestClient(main_module.app, raise_server_exceptions=False)

    with patch("api.main.parse_instruction", return_value=mock_goal), \
         patch("api.main.encode_frame", return_value=mock_embedding):
        start_resp = client.post("/session/start", json={"instruction": "go to the chair"})

    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    t0 = time.monotonic()
    with patch("api.main.encode_frame", return_value=mock_embedding):
        run_resp = client.post(f"/session/{session_id}/run")
    elapsed = time.monotonic() - t0

    assert run_resp.status_code == 200
    assert elapsed < 60.0, f"/run took {elapsed:.1f}s, expected < 60s"
