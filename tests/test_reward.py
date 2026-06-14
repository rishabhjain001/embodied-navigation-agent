import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.reward import compute_reward, euclidean_distance, manhattan_distance

# Reusable position strategy
pos = st.tuples(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100))


# Feature: embodied-navigation-agent, Property 14: Stop action termination condition
@given(agent_pos=pos, target_pos=pos)
@settings(max_examples=100)
def test_stop_always_sets_done(agent_pos, target_pos):
    """done is always True when action=3 (stop), regardless of position."""
    reward, done = compute_reward(
        action=3,
        prev_agent_pos=agent_pos,
        curr_agent_pos=agent_pos,
        target_pos=target_pos,
        step_count=0,
        step_budget=200,
        done_stop=True,
    )
    assert done is True


# Feature: embodied-navigation-agent, Property 14: Stop action termination condition
@given(agent_pos=pos, target_pos=pos)
@settings(max_examples=100)
def test_stop_reward_by_proximity(agent_pos, target_pos):
    """Stop within manhattan distance <= 1 gives +10.0; otherwise 0.0."""
    reward, done = compute_reward(
        action=3,
        prev_agent_pos=agent_pos,
        curr_agent_pos=agent_pos,
        target_pos=target_pos,
        step_count=0,
        step_budget=200,
        done_stop=True,
    )
    if manhattan_distance(agent_pos, target_pos) <= 1:
        assert reward == pytest.approx(10.0)
    else:
        assert reward == pytest.approx(0.0)


# Feature: embodied-navigation-agent, Property 15: Reward signal correctness — move_forward closer
@given(
    prev_pos=pos,
    target_pos=pos,
)
@settings(max_examples=100)
def test_move_forward_closer_gives_positive_reward(prev_pos, target_pos):
    """Moving closer to target (euclidean distance decreases) contributes +0.1."""
    # Build a curr_pos that is strictly closer to target than prev_pos.
    # Use target itself as curr_pos — guaranteed closer unless prev_pos == target.
    curr_pos = target_pos
    prev_dist = euclidean_distance(prev_pos, target_pos)
    curr_dist = euclidean_distance(curr_pos, target_pos)  # 0.0

    if prev_dist <= curr_dist:
        # prev_pos == target_pos; distance is 0 either way — skip
        return

    reward, done = compute_reward(
        action=0,
        prev_agent_pos=prev_pos,
        curr_agent_pos=curr_pos,
        target_pos=target_pos,
        step_count=0,
        step_budget=200,
        done_stop=False,
    )
    assert reward == pytest.approx(0.1)
    assert done is False


# Feature: embodied-navigation-agent, Property 15: Reward signal correctness — move_forward farther
@given(
    curr_pos=pos,
    target_pos=pos,
)
@settings(max_examples=100)
def test_move_forward_farther_gives_negative_reward(curr_pos, target_pos):
    """Moving away from target (euclidean distance increases) contributes -0.1."""
    # Use target itself as prev_pos — guaranteed farther unless curr_pos == target.
    prev_pos = target_pos
    prev_dist = euclidean_distance(prev_pos, target_pos)  # 0.0
    curr_dist = euclidean_distance(curr_pos, target_pos)

    if curr_dist <= prev_dist:
        # curr_pos == target_pos; same distance — skip
        return

    reward, done = compute_reward(
        action=0,
        prev_agent_pos=prev_pos,
        curr_agent_pos=curr_pos,
        target_pos=target_pos,
        step_count=0,
        step_budget=200,
        done_stop=False,
    )
    assert reward == pytest.approx(-0.1)
    assert done is False


# Feature: embodied-navigation-agent, Property 15: Reward signal correctness — collision penalty
@given(agent_pos=pos, target_pos=pos)
@settings(max_examples=100)
def test_move_forward_collision_adds_penalty(agent_pos, target_pos):
    """When curr_pos == prev_pos (wall block), -0.5 is added on top of distance component."""
    # Distance is 0 (same position), so distance reward is 0.0; only penalty applies.
    reward, done = compute_reward(
        action=0,
        prev_agent_pos=agent_pos,
        curr_agent_pos=agent_pos,  # position unchanged — collision
        target_pos=target_pos,
        step_count=0,
        step_budget=200,
        done_stop=False,
    )
    # distance component is 0 (prev_dist == curr_dist), plus collision -0.5
    assert reward == pytest.approx(-0.5)
    assert done is False


# Feature: embodied-navigation-agent, Property 15: Reward signal correctness — timeout
@given(
    agent_pos=pos,
    target_pos=pos,
    step_budget=st.integers(min_value=1, max_value=500),
)
@settings(max_examples=100)
def test_timeout_gives_penalty_and_done(agent_pos, target_pos, step_budget):
    """When step_count >= step_budget, -5.0 is added and done=True (any action except stop)."""
    step_count = step_budget  # exactly at budget triggers timeout

    # Use action=1 (turn_left) to isolate timeout logic from stop/move_forward rewards.
    reward, done = compute_reward(
        action=1,
        prev_agent_pos=agent_pos,
        curr_agent_pos=agent_pos,
        target_pos=target_pos,
        step_count=step_count,
        step_budget=step_budget,
        done_stop=False,
    )
    assert reward == pytest.approx(-5.0)
    assert done is True


# Feature: embodied-navigation-agent, Property 15: Reward signal correctness — no timeout before budget
@given(
    agent_pos=pos,
    target_pos=pos,
    step_budget=st.integers(min_value=2, max_value=500),
)
@settings(max_examples=100)
def test_no_timeout_before_budget(agent_pos, target_pos, step_budget):
    """When step_count < step_budget, no timeout penalty and done stays False (turn actions)."""
    step_count = step_budget - 1

    reward, done = compute_reward(
        action=1,
        prev_agent_pos=agent_pos,
        curr_agent_pos=agent_pos,
        target_pos=target_pos,
        step_count=step_count,
        step_budget=step_budget,
        done_stop=False,
    )
    assert reward == pytest.approx(0.0)
    assert done is False
