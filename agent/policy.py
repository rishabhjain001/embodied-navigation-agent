import numpy as np
import gymnasium as gym
import yaml
from stable_baselines3 import PPO

# Low-dimensional observation for fast, reliable navigation learning.
# Layout (7 floats):
#   [0] delta_row to target (normalized by grid size)
#   [1] delta_col to target (normalized by grid size)
#   [2] agent direction sin
#   [3] agent direction cos
#   [4] wall directly ahead   (0 or 1)
#   [5] wall to the left      (0 or 1)
#   [6] wall to the right     (0 or 1)
OBS_DIM = 7

# MiniGrid direction → (dcol, drow) forward / left / right cell offsets
# dir: 0=right(+col), 1=down(+row), 2=left(-col), 3=up(-row)
_FORWARD = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
_LEFT = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
_RIGHT = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}


def build_obs_space() -> gym.spaces.Box:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)


def _is_wall(base_env, col: int, row: int) -> float:
    """1.0 if the cell at (col,row) is a wall or out of bounds, else 0.0."""
    if col < 0 or row < 0 or col >= base_env.grid.width or row >= base_env.grid.height:
        return 1.0
    cell = base_env.grid.get(col, row)
    if cell is not None and getattr(cell, "type", None) == "wall":
        return 1.0
    return 0.0


def detect_walls(base_env):
    """Returns (ahead, left, right) wall flags relative to agent facing."""
    col, row = int(base_env.agent_pos[0]), int(base_env.agent_pos[1])
    d = int(base_env.agent_dir)
    fa = _FORWARD[d]
    le = _LEFT[d]
    ri = _RIGHT[d]
    return (
        _is_wall(base_env, col + fa[0], row + fa[1]),
        _is_wall(base_env, col + le[0], row + le[1]),
        _is_wall(base_env, col + ri[0], row + ri[1]),
    )


def build_nav_obs(agent_rc, agent_dir, target_rc, grid_size, walls) -> np.ndarray:
    """Build the 7-dim observation. agent_rc/target_rc are (row, col)."""
    dr = (target_rc[0] - agent_rc[0]) / max(grid_size, 1)
    dc = (target_rc[1] - agent_rc[1]) / max(grid_size, 1)
    angle = agent_dir * (np.pi / 2.0)
    return np.array(
        [dr, dc, np.sin(angle), np.cos(angle), walls[0], walls[1], walls[2]],
        dtype=np.float32,
    )


def make_policy(env) -> PPO:
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)["ppo"]
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
    )


def load_policy(model_path: str, env) -> PPO:
    try:
        return PPO.load(model_path, env=env)
    except Exception as exc:
        raise RuntimeError(f"failed to load policy from {model_path}: {exc}") from exc


def predict_action(policy: PPO, obs: np.ndarray) -> int:
    action, _ = policy.predict(obs, deterministic=True)
    return int(action)
