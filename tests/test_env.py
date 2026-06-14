"""Property-based tests for agent/env.py (Properties P1–P4)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.env import make_env, get_object_positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(env):
    """Walk wrapper chain to reach the base _NavigationEnv."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _forward_cell(base_env):
    """Return the grid cell directly in front of the agent (may be None)."""
    # MiniGrid directions: 0=right(+x), 1=down(+y), 2=left(-x), 3=up(-y)
    dir_to_delta = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    dx, dy = dir_to_delta[base_env.agent_dir]
    fx = base_env.agent_pos[0] + dx
    fy = base_env.agent_pos[1] + dy
    return base_env.grid.get(fx, fy)


def _face_wall(env, base_env):
    """
    Turn the agent until it faces a wall, then return True.
    Returns False if no wall-facing direction is found after 4 turns
    (shouldn't happen in a bounded room, but guards against infinite loops).
    """
    for _ in range(4):
        cell = _forward_cell(base_env)
        if cell is not None and cell.type == "wall":
            return True
        # turn right (action 2)
        env.step(2)
    return False


# ---------------------------------------------------------------------------
# Property 1: Environment layout invariants on reset
# ---------------------------------------------------------------------------

# Feature: embodied-navigation-agent, Property 1: Environment layout invariants on reset
@settings(max_examples=100)
@given(
    grid_size=st.integers(min_value=8, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_p1_layout_invariants_on_reset(grid_size, seed):
    """
    After reset, env must have exactly 5 objects at distinct cells with
    distinct (type, color) pairs; agent at an interior non-object cell with a
    cardinal orientation.
    """
    env = make_env(grid_size=grid_size, object_count=5, seed=seed)
    # make_env already calls reset internally; call again with explicit seed
    # so hypothesis controls the layout.
    obs, info = env.reset(seed=seed)
    base = _unwrap(env)

    positions = get_object_positions(env)

    # Exactly 5 objects
    assert len(positions) == 5, f"Expected 5 objects, got {len(positions)}"

    # All at distinct cells
    cells = list(positions.values())
    assert len(cells) == len(set(cells)), "Objects share cells"

    # Distinct (type, color) combos — keys are already "{type}_{color}" so
    # uniqueness of keys guarantees this.
    assert len(set(positions.keys())) == 5, "Duplicate (type, color) combos"

    # Agent is at an interior cell (not on the border wall ring)
    agent_col, agent_row = base.agent_pos   # MiniGrid: (x=col, y=row)
    assert 1 <= agent_row <= grid_size - 2, "Agent on border row"
    assert 1 <= agent_col <= grid_size - 2, "Agent on border col"

    # Agent cell is not occupied by an object
    object_cells = set(positions.values())  # stored as (row, col)
    assert (agent_row, agent_col) not in object_cells, "Agent spawned on object"

    # Cardinal orientation: MiniGrid directions 0–3 are all cardinal
    assert base.agent_dir in {0, 1, 2, 3}, f"Non-cardinal direction: {base.agent_dir}"

    env.close()


# ---------------------------------------------------------------------------
# Property 2: Consecutive sessions differ in layout
# ---------------------------------------------------------------------------

# Feature: embodied-navigation-agent, Property 2: Consecutive sessions differ in layout
@settings(max_examples=100)
@given(
    grid_size=st.integers(min_value=8, max_value=32),
    seed_a=st.integers(min_value=0, max_value=2**31 - 1),
    seed_b=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_p2_consecutive_sessions_differ(grid_size, seed_a, seed_b):
    """
    Two resets with different seeds must produce different object layouts.
    We use distinct seeds to make the check deterministic and reliable.
    """
    # Different seeds → different layouts (by design of the RNG seeding).
    # Equal seeds → same layout; skip those cases.
    if seed_a == seed_b:
        return

    env = make_env(grid_size=grid_size, object_count=5, seed=seed_a)

    env.reset(seed=seed_a)
    positions_a = get_object_positions(env)

    env.reset(seed=seed_b)
    positions_b = get_object_positions(env)

    # Positions recorded as {type_color: (row, col)}.  With different seeds
    # it is astronomically unlikely (and by spec, undesired) for both full
    # configurations to be identical — assert they differ.
    assert positions_a != positions_b, (
        f"Layouts identical for seeds {seed_a} and {seed_b} on grid {grid_size}"
    )

    env.close()


# ---------------------------------------------------------------------------
# Property 3: Collision keeps position and increments count
# ---------------------------------------------------------------------------

# Feature: embodied-navigation-agent, Property 3: Collision keeps position and increments count
@settings(max_examples=100)
@given(
    grid_size=st.integers(min_value=8, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_p3_collision_keeps_position(grid_size, seed):
    """
    When the agent's forward cell is a wall, move_forward must leave (row, col)
    and orientation unchanged and increment collision_count by 1.
    """
    env = make_env(grid_size=grid_size, object_count=5, seed=seed)
    env.reset(seed=seed)
    base = _unwrap(env)

    # Manoeuvre agent so it faces a wall.
    faced_wall = _face_wall(env, base)
    if not faced_wall:
        # Couldn't engineer wall-facing state; skip this example.
        env.close()
        return

    # Snapshot state before the collision attempt.
    pos_before = tuple(base.agent_pos)      # (col, row) in MiniGrid coords
    dir_before = base.agent_dir

    # Retrieve current collision count from the last step's info, or 0.
    # The cleanest way is to inspect the private counter directly.
    count_before = base._collision_count

    # Issue move_forward (action 0).
    obs, reward, terminated, truncated, info = env.step(0)

    pos_after = tuple(base.agent_pos)
    dir_after = base.agent_dir
    count_after = info["collision_count"]

    assert pos_after == pos_before, (
        f"Position changed on wall collision: {pos_before} → {pos_after}"
    )
    assert dir_after == dir_before, (
        f"Direction changed on wall collision: {dir_before} → {dir_after}"
    )
    assert count_after == count_before + 1, (
        f"collision_count not incremented: {count_before} → {count_after}"
    )

    env.close()


# ---------------------------------------------------------------------------
# Property 4: POV frame has valid shape and type on every step
# ---------------------------------------------------------------------------

# Feature: embodied-navigation-agent, Property 4: POV frame has valid shape and type on every step
@settings(max_examples=100)
@given(
    grid_size=st.integers(min_value=8, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    action=st.integers(min_value=0, max_value=3),
)
def test_p4_pov_frame_shape_and_dtype(grid_size, seed, action):
    """
    obs["image"] from env.reset() and env.step() must be uint8 with shape
    (H, W, 3) for positive H and W.
    """
    env = make_env(grid_size=grid_size, object_count=5, seed=seed)
    reset_obs, _ = env.reset(seed=seed)

    # Check frame from reset.
    img_reset = reset_obs["image"]
    assert isinstance(img_reset, np.ndarray), "image is not ndarray after reset"
    assert img_reset.dtype == np.uint8, f"reset image dtype {img_reset.dtype} != uint8"
    assert img_reset.ndim == 3, f"reset image ndim {img_reset.ndim} != 3"
    H, W, C = img_reset.shape
    assert C == 3, f"reset image channel count {C} != 3"
    assert H > 0 and W > 0, f"reset image has zero spatial dimension: {H}x{W}"

    # Check frame from step.
    step_obs, _, terminated, truncated, _ = env.step(action)

    if not (terminated or truncated):
        img_step = step_obs["image"]
        assert isinstance(img_step, np.ndarray), "image is not ndarray after step"
        assert img_step.dtype == np.uint8, f"step image dtype {img_step.dtype} != uint8"
        assert img_step.ndim == 3, f"step image ndim {img_step.ndim} != 3"
        H2, W2, C2 = img_step.shape
        assert C2 == 3, f"step image channel count {C2} != 3"
        assert H2 > 0 and W2 > 0, f"step image has zero spatial dimension: {H2}x{W2}"

    env.close()
