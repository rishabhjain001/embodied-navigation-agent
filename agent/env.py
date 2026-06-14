from __future__ import annotations

from itertools import product

import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Ball, Box, Door, Key, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper


# MiniGrid only knows a fixed set of object types; our semantic labels need
# to map to renderable MiniGrid types so the grid encoder doesn't break.
_SEMANTIC_TO_NATIVE = {
    "chair": "ball",
    "table": "key",
    "lamp": "goal",
    "door": "door",
    "box": "box",
}

OBJECT_TYPES = frozenset(_SEMANTIC_TO_NATIVE.keys())


class _SemanticObj(WorldObj):
    """WorldObj subclass that carries a semantic type label (e.g. 'chair')
    while using the corresponding native MiniGrid type for grid encoding and
    rendering so the rest of MiniGrid infrastructure stays intact."""

    def __init__(self, semantic_type: str, color: str):
        native = _SEMANTIC_TO_NATIVE[semantic_type]
        # Bypass the type-string assertion by going to object.__init__ first,
        # then setting the fields WorldObj.__init__ would have set.
        object.__init__(self)
        assert native in OBJECT_TO_IDX
        assert color in COLOR_TO_IDX
        self._native_type = native
        self.type = semantic_type   # exposed externally as "chair", etc.
        self.color = color
        self.contains = None
        self.init_pos = None
        self.cur_pos = None

    def encode(self):
        # Use the native type index so Grid.encode() and the observation
        # encoder produce valid integer triples.
        return (OBJECT_TO_IDX[self._native_type], COLOR_TO_IDX[self.color], 0)

    def render(self, img):
        # Delegate rendering to a temporary native object of the same color.
        _NATIVE_CLASS[self._native_type](self.color).render(img)


# Mapping used by _SemanticObj.render to instantiate the correct native class.
_NATIVE_CLASS = {
    "ball": Ball,
    "key": Key,
    "goal": lambda c: Goal(),   # Goal ignores color in its render
    "door": lambda c: Door(c),
    "box": Box,
}


class _NavigationEnv(MiniGridEnv):
    """Single rectangular room populated with exactly `object_count` objects.

    Actions are remapped to our 4-action interface:
        0 = move_forward
        1 = turn_left
        2 = turn_right
        3 = stop  (handled before super().step; not a native MiniGrid action)
    """

    def __init__(self, grid_size: int, object_count: int, seed: int | None):
        mission_space = MissionSpace(mission_func=lambda: "navigate to the target")
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=10_000,       # high cap; episode end is handled externally
            see_through_walls=False,
            render_mode=None,
        )

        self._object_count = object_count
        self._collision_count = 0

        # Override to our 4-action discrete space; native MiniGrid has 7.
        self.action_space = spaces.Discrete(4)

        # Expose the set of semantic object type strings.
        self.object_types = OBJECT_TYPES

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Choose object_count distinct (type, color) combos for this episode.
        all_combos = list(product(sorted(OBJECT_TYPES), COLOR_NAMES))
        chosen = self.np_random.choice(len(all_combos), size=self._object_count, replace=False)
        type_color_pairs = [all_combos[i] for i in chosen]

        for sem_type, color in type_color_pairs:
            obj = _SemanticObj(sem_type, color)
            # place_obj picks a random empty interior cell.
            self.place_obj(obj, top=(1, 1), size=(width - 2, height - 2))

        # Spawn agent at a random empty interior cell with a random direction.
        self.place_agent(top=(1, 1), size=(width - 2, height - 2), rand_dir=True)
        self.mission = "navigate to the target"

    def reset(self, **kwargs):
        self._collision_count = 0
        return super().reset(**kwargs)

    def step(self, action: int):
        # Intercept the stop action (3) before MiniGrid sees it.
        if action == 3:
            obs = self.gen_obs()
            return obs, 0.0, True, False, {"collision_count": self._collision_count}

        # Translate our action indices to MiniGrid's Actions enum values.
        #   Our 0 (forward) → MiniGrid forward (2)
        #   Our 1 (left)    → MiniGrid left    (0)
        #   Our 2 (right)   → MiniGrid right   (1)
        _ACTION_MAP = {0: 2, 1: 0, 2: 1}
        native_action = _ACTION_MAP[action]

        pos_before = tuple(self.agent_pos)
        obs, reward, terminated, truncated, info = super().step(native_action)
        pos_after = tuple(self.agent_pos)

        if action == 0 and pos_after == pos_before:
            self._collision_count += 1

        info["collision_count"] = self._collision_count
        return obs, reward, terminated, truncated, info


def make_env(grid_size: int, object_count: int, seed: int | None = None):
    """Returns a wrapped MiniGrid env with our 4-action interface and RGB
    partial observations.  object_types is accessible on the unwrapped env
    and propagated to the wrapper via gymnasium's standard unwrapping."""
    base = _NavigationEnv(grid_size=grid_size, object_count=object_count, seed=seed)
    env = RGBImgPartialObsWrapper(base)

    # Seed immediately so the first reset produces a deterministic layout when
    # a seed is given, or a random one otherwise.
    env.reset(seed=seed)

    # Surface the frozenset from the base env for callers that wrap with
    # RGBImgPartialObsWrapper and still need to read it.
    env.object_types = base.object_types

    return env


def get_object_positions(env) -> dict[str, tuple[int, int]]:
    """Returns {'{type}_{color}': (row, col)} for every placed object.

    Unwraps gymnasium wrappers to reach the base _NavigationEnv so this works
    regardless of how many wrappers are stacked on top."""
    base = env
    while hasattr(base, "env"):
        base = base.env

    positions = {}
    for col in range(base.grid.width):
        for row in range(base.grid.height):
            cell = base.grid.get(col, row)
            # Skip walls, empty cells, and any non-semantic objects.
            if cell is None or not isinstance(cell, _SemanticObj):
                continue
            key = f"{cell.type}_{cell.color}"
            # MiniGrid stores positions as (x=col, y=row); we return (row, col)
            # to match the standard matrix convention used by reward.py.
            positions[key] = (row, col)

    return positions
