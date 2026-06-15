"""Property-based tests for agent/policy.py (Properties P11–P13)."""

import numpy as np
import pytest
from gymnasium import Env
from gymnasium import spaces
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.policy import (
    OBS_DIM,
    build_obs_space,
    build_nav_obs,
    load_policy,
    make_policy,
    predict_action,
)


class _DummyEnv(Env):
    """Minimal env with the right obs/action spaces for policy construction."""
    observation_space = build_obs_space()
    action_space = spaces.Discrete(4)

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


# Create once at module level — policy construction is expensive.
_DUMMY_ENV = _DummyEnv()
_POLICY = make_policy(_DUMMY_ENV)


@st.composite
def float32_obs(draw):
    """Draw a random float32 (OBS_DIM,) observation vector in [-1, 1]."""
    values = draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1.0, max_value=1.0),
            min_size=OBS_DIM,
            max_size=OBS_DIM,
        )
    )
    return np.array(values, dtype=np.float32)


pos = st.tuples(st.integers(min_value=0, max_value=32), st.integers(min_value=0, max_value=32))
walls = st.tuples(
    st.sampled_from([0.0, 1.0]),
    st.sampled_from([0.0, 1.0]),
    st.sampled_from([0.0, 1.0]),
)


# Feature: embodied-navigation-agent, Property 11: Policy predict returns valid action for any observation
@given(obs=float32_obs())
@settings(max_examples=100)
def test_p11_predict_returns_valid_action(obs):
    """predict_action returns an integer in {0, 1, 2, 3} for any observation."""
    action = predict_action(_POLICY, obs)
    assert isinstance(action, int)
    assert action in {0, 1, 2, 3}


# Feature: embodied-navigation-agent, Property 12: Policy serialization round trip
def test_p12_serialization_round_trip(tmp_path):
    """Saving then loading a policy produces identical predictions for any observation."""
    save_path = str(tmp_path / "test_policy")
    _POLICY.save(save_path)
    loaded = load_policy(save_path + ".zip", _DUMMY_ENV)

    rng = np.random.default_rng(42)
    for _ in range(50):
        obs = rng.uniform(-1.0, 1.0, size=(OBS_DIM,)).astype(np.float32)
        assert predict_action(_POLICY, obs) == predict_action(loaded, obs)


# Feature: embodied-navigation-agent, Property 13: Observation encoding is consistent and fixed-length
@given(agent=pos, target=pos, direction=st.integers(0, 3), w=walls)
@settings(max_examples=100)
def test_p13_obs_shape_and_dtype(agent, target, direction, w):
    """build_nav_obs returns float32 (OBS_DIM,) for any agent/target/direction/walls."""
    obs = build_nav_obs(agent, direction, target, grid_size=10, walls=w)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


# Feature: embodied-navigation-agent, Property 13: Observation encoding is consistent and fixed-length
@given(agent=pos, target=pos, direction=st.integers(0, 3), w=walls)
@settings(max_examples=100)
def test_p13_obs_is_deterministic(agent, target, direction, w):
    """Calling build_nav_obs twice with the same input returns equal arrays."""
    o1 = build_nav_obs(agent, direction, target, grid_size=10, walls=w)
    o2 = build_nav_obs(agent, direction, target, grid_size=10, walls=w)
    assert np.array_equal(o1, o2)


# Feature: embodied-navigation-agent, Property 13: Observation encoding is consistent and fixed-length
def test_p13_obs_direction_signal():
    """Target below-and-right of agent yields positive row/col deltas."""
    obs = build_nav_obs((2, 2), 0, (8, 6), grid_size=10, walls=(0.0, 0.0, 0.0))
    assert obs[0] > 0  # delta_row positive (target below)
    assert obs[1] > 0  # delta_col positive (target right)
