"""Property-based tests for agent/policy.py (Properties P11–P13)."""

import numpy as np
import pytest
from gymnasium import Env
from gymnasium import spaces
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent.policy import (
    GOAL_ENCODING_DIM,
    OBS_DIM,
    build_obs_space,
    encode_goal,
    load_policy,
    make_policy,
    predict_action,
)


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

class _DummyEnv(Env):
    """Minimal env with the right obs/action spaces for policy construction."""
    observation_space = build_obs_space()
    action_space = spaces.Discrete(4)

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, False, False, {}


# Create once at module level — policy construction is expensive.
_DUMMY_ENV = _DummyEnv()
_POLICY = make_policy(_DUMMY_ENV)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def float32_obs(draw):
    """Draw a random float32 (576,) observation vector."""
    values = draw(
        st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1e6,
                max_value=1e6,
            ),
            min_size=OBS_DIM,
            max_size=OBS_DIM,
        )
    )
    return np.array(values, dtype=np.float32)


@st.composite
def goal_dict(draw):
    """Draw a Goal dict with any combination of present/absent fields."""
    text_or_none = st.one_of(
        st.none(),
        st.text(min_size=1, max_size=50).filter(str.strip),
    )
    return {
        "target":    draw(text_or_none),
        "attribute": draw(text_or_none),
        "context":   draw(st.one_of(st.none(), st.text(min_size=1, max_size=100).filter(str.strip))),
        "error":     draw(text_or_none),
    }


# ---------------------------------------------------------------------------
# Property 11: Policy predict returns valid action for any observation
# ---------------------------------------------------------------------------

# Feature: embodied-navigation-agent, Property 11: Policy predict returns valid action for any observation
@given(obs=float32_obs())
@settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
def test_p11_predict_returns_valid_action(obs):
    """predict_action returns an integer in {0, 1, 2, 3} for any float32 (576,) input."""
    action = predict_action(_POLICY, obs)
    assert isinstance(action, int)
    assert action in {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Property 12: Policy serialization round trip
# ---------------------------------------------------------------------------

# Feature: embodied-navigation-agent, Property 12: Policy serialization round trip
def test_p12_serialization_round_trip(tmp_path):
    """Saving then loading a policy produces identical predictions for a range of observations."""
    save_path = str(tmp_path / "test_policy")
    _POLICY.save(save_path)

    loaded = load_policy(save_path + ".zip", _DUMMY_ENV)

    rng = np.random.default_rng(42)
    for _ in range(50):
        obs = rng.uniform(-1.0, 1.0, size=(OBS_DIM,)).astype(np.float32)
        original_action = predict_action(_POLICY, obs)
        loaded_action = predict_action(loaded, obs)
        assert original_action == loaded_action, (
            f"Action mismatch after round trip: {original_action} vs {loaded_action}"
        )


# ---------------------------------------------------------------------------
# Property 13: Goal encoding is consistent and fixed-length
# ---------------------------------------------------------------------------

# Feature: embodied-navigation-agent, Property 13: Goal encoding is consistent and fixed-length
@given(goal=goal_dict())
@settings(max_examples=100)
def test_p13_goal_encoding_shape_and_dtype(goal):
    """encode_goal returns float32 (64,) for any Goal dict including null fields."""
    result = encode_goal(goal)
    assert isinstance(result, np.ndarray)
    assert result.shape == (GOAL_ENCODING_DIM,)
    assert result.dtype == np.float32


# Feature: embodied-navigation-agent, Property 13: Goal encoding is consistent and fixed-length
@given(goal=goal_dict())
@settings(max_examples=100)
def test_p13_goal_encoding_is_deterministic(goal):
    """Calling encode_goal twice with the same input returns element-wise equal arrays."""
    result1 = encode_goal(goal)
    result2 = encode_goal(goal)
    assert np.array_equal(result1, result2), (
        f"encode_goal is not deterministic for goal={goal}"
    )


def test_p13_goal_encoding_all_none():
    """encode_goal with all None fields returns float32 (64,) without error."""
    goal = {"target": None, "attribute": None, "context": None, "error": None}
    result = encode_goal(goal)
    assert result.shape == (GOAL_ENCODING_DIM,)
    assert result.dtype == np.float32


def test_p13_goal_encoding_all_present():
    """encode_goal with all fields populated returns float32 (64,)."""
    goal = {"target": "chair", "attribute": "red", "context": "near the window", "error": None}
    result = encode_goal(goal)
    assert result.shape == (GOAL_ENCODING_DIM,)
    assert result.dtype == np.float32
