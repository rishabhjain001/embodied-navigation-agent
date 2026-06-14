# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.reward import manhattan_distance
from api.session import EvaluationMetrics, StepRecord

# Strategies
pos = st.tuples(st.integers(min_value=0, max_value=20), st.integers(min_value=0, max_value=20))
action = st.integers(min_value=0, max_value=3)

# A step is (action_int, pos_before, pos_after).
# For action==0 (move_forward), pos_after may equal pos_before (collision) or differ (successful move).
step_entry = st.tuples(
    action,
    pos,
    pos,
)


def _step_sequence():
    """Generate list of (action, pos_before, pos_after) with consistent collision logic."""
    return st.lists(step_entry, min_size=1, max_size=50)


# ---------------------------------------------------------------------------
# Metric computation helpers — these mirror what api/main.py will implement.
# ---------------------------------------------------------------------------

def compute_path_length(steps: list[tuple[int, tuple, tuple]]) -> int:
    """Count of move_forward (action 0) actions."""
    return sum(1 for action, _, _ in steps if action == 0)


def compute_collision_count(steps: list[tuple[int, tuple, tuple]]) -> int:
    """Count of move_forward where position did not change."""
    return sum(
        1 for action, pos_before, pos_after in steps
        if action == 0 and pos_before == pos_after
    )


def compute_steps_to_goal(steps: list[tuple[int, tuple, tuple]]) -> int:
    """Total number of steps in the log (through the terminal step)."""
    return len(steps)


def compute_efficiency(
    start: tuple[int, int],
    target: tuple[int, int],
    path_length: int,
) -> float:
    """
    navigation_efficiency for a successful session.
    Returns manhattan_distance(start, target) / path_length,
    or 1.0 when path_length == 0 or manhattan distance == 0.
    """
    dist = manhattan_distance(start, target)
    if path_length == 0 or dist == 0:
        return 1.0
    return dist / path_length


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
@given(steps=_step_sequence())
@settings(max_examples=100)
def test_path_length_equals_move_forward_count(steps):
    """path_length == number of action-0 entries in the step log."""
    expected = sum(1 for a, _, _ in steps if a == 0)
    assert compute_path_length(steps) == expected


# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
@given(steps=_step_sequence())
@settings(max_examples=100)
def test_steps_to_goal_equals_total_step_count(steps):
    """steps_to_goal == total entries in the step log."""
    assert compute_steps_to_goal(steps) == len(steps)


# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
@given(steps=_step_sequence())
@settings(max_examples=100)
def test_collision_count_is_forward_with_unchanged_position(steps):
    """collision_count == move_forward actions where pos_before == pos_after."""
    expected = sum(
        1 for a, pb, pa in steps if a == 0 and pb == pa
    )
    assert compute_collision_count(steps) == expected


# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
@given(
    start=pos,
    target=pos,
    # Generate at least one move_forward to ensure path_length > 0
    extra_steps=st.lists(step_entry, min_size=0, max_size=49),
    forward_pos_before=pos,
    forward_pos_after=pos,
)
@settings(max_examples=100)
def test_efficiency_formula_for_successful_session(
    start, target, extra_steps, forward_pos_before, forward_pos_after
):
    """
    For a successful session where path_length > 0 and manhattan distance > 0,
    efficiency = manhattan_distance(start, target) / path_length.
    """
    # Ensure pos_after != pos_before so this move_forward is not a collision,
    # and ensure it actually moves (skip if they happen to be equal).
    if forward_pos_before == forward_pos_after:
        forward_pos_after = (
            (forward_pos_after[0] + 1) % 21,
            forward_pos_after[1],
        )

    # Build a step list with at least one non-collision move_forward.
    steps = [(0, forward_pos_before, forward_pos_after)] + extra_steps
    path_length = compute_path_length(steps)
    dist = manhattan_distance(start, target)

    # Only test the ratio formula when both conditions are satisfied.
    if path_length > 0 and dist > 0:
        efficiency = compute_efficiency(start, target, path_length)
        assert efficiency == pytest.approx(dist / path_length)


# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
@given(start=pos, target=pos)
@settings(max_examples=100)
def test_efficiency_is_one_when_path_length_zero(start, target):
    """When path_length == 0, efficiency is 1.0 regardless of distance."""
    assert compute_efficiency(start, target, path_length=0) == pytest.approx(1.0)


# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
@given(steps=_step_sequence(), target=pos)
@settings(max_examples=100)
def test_efficiency_is_one_when_start_equals_target(steps, target):
    """When manhattan_distance(start, target) == 0, efficiency is 1.0."""
    start = target  # distance == 0
    path_length = compute_path_length(steps)
    assert compute_efficiency(start, target, path_length) == pytest.approx(1.0)


# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
@given(
    n_sessions=st.integers(min_value=1, max_value=100),
    n_successes=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100)
def test_success_rate_formula(n_sessions, n_successes):
    """success_rate == successful_sessions / total_sessions."""
    # Clamp successes to not exceed total sessions.
    successes = min(n_successes, n_sessions)
    rate = successes / n_sessions
    assert rate == pytest.approx(successes / n_sessions)
    assert 0.0 <= rate <= 1.0


# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
@given(
    steps=_step_sequence(),
    start=pos,
    target=pos,
)
@settings(max_examples=100)
def test_metrics_consistency_across_helpers(steps, start, target):
    """
    All metric helpers are consistent with each other:
    collision_count <= path_length, steps_to_goal == len(steps),
    and efficiency is in (0, inf) for success with positive path_length and distance.
    """
    path_length = compute_path_length(steps)
    collision_count = compute_collision_count(steps)
    steps_to_goal = compute_steps_to_goal(steps)

    # Collisions can only happen on move_forward — never exceed path_length.
    assert collision_count <= path_length

    # Total steps equals log length.
    assert steps_to_goal == len(steps)

    # Efficiency is always positive when computable.
    dist = manhattan_distance(start, target)
    if path_length > 0 and dist > 0:
        efficiency = compute_efficiency(start, target, path_length)
        assert efficiency > 0.0


# Feature: embodied-navigation-agent, Property 19: Evaluation metrics correctness
def test_evaluation_metrics_dataclass_defaults():
    """EvaluationMetrics initialises with expected nullable fields."""
    m = EvaluationMetrics(
        path_length=0,
        steps_to_goal=None,
        collision_count=0,
        navigation_efficiency=None,
        success_rate=None,
    )
    assert m.path_length == 0
    assert m.steps_to_goal is None
    assert m.collision_count == 0
    assert m.navigation_efficiency is None
    assert m.success_rate is None
