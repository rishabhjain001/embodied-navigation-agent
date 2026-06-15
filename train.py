import argparse
import logging
import math
import os
import sys

import numpy as np
import gymnasium as gym
import yaml
from dotenv import load_dotenv
from stable_baselines3.common.callbacks import BaseCallback

from agent.env import _NavigationEnv, get_object_positions
from agent.policy import make_policy, load_policy, build_obs_space, build_nav_obs, detect_walls


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    result = {}
    result.update(cfg.get("ppo", {}))
    result.update(cfg.get("env", {}))
    result.update(cfg.get("reward", {}))

    overrides = {
        "grid_size": os.getenv("GRID_SIZE"),
        "step_budget": os.getenv("STEP_BUDGET"),
        "object_count": os.getenv("OBJECT_COUNT"),
        "total_timesteps": os.getenv("TOTAL_TIMESTEPS"),
    }
    for key, val in overrides.items():
        if val is not None:
            if key in ("grid_size", "step_budget", "object_count", "total_timesteps"):
                result[key] = int(val)
            else:
                result[key] = float(val)
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Train the navigation policy")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model-path", default=None)
    p.add_argument("--force-retrain", action="store_true")
    return p.parse_args()


class NavTrainEnv(gym.Env):
    """Single clean training env: 7-dim obs + shaped reward, no tangled wrappers.

    Wraps a raw _NavigationEnv. Picks a random target object each episode and
    rewards the agent for navigating toward it.
    """

    def __init__(self, grid_size: int, object_count: int, step_budget: int, cfg: dict):
        super().__init__()
        self._env = _NavigationEnv(grid_size=grid_size, object_count=object_count, seed=None)
        self._grid_size = grid_size
        self._budget = step_budget
        self._cfg = cfg
        self.observation_space = build_obs_space()
        self.action_space = gym.spaces.Discrete(4)
        self._target = None
        self._prev_dist = None
        self._steps = 0

    def _agent_rc(self):
        return (int(self._env.agent_pos[1]), int(self._env.agent_pos[0]))

    def _obs(self):
        walls = detect_walls(self._env)
        return build_nav_obs(self._agent_rc(), int(self._env.agent_dir),
                             self._target, self._grid_size, walls)

    def reset(self, *, seed=None, options=None):
        self._env.reset(seed=seed)
        self._steps = 0
        positions = get_object_positions(self._env)
        if positions:
            keys = list(positions.keys())
            idx = self._env.np_random.integers(0, len(keys))
            self._target = positions[keys[idx]]
        else:
            self._target = (1, 1)
        self._prev_dist = math.dist(self._agent_rc(), self._target)
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        prev_rc = self._agent_rc()
        self._env.step(action)
        self._steps += 1
        curr_rc = self._agent_rc()
        curr_dist = math.dist(curr_rc, self._target)
        man = abs(curr_rc[0] - self._target[0]) + abs(curr_rc[1] - self._target[1])

        # dense shaping: reward reduction in distance, small time penalty so
        # spinning/standing still is actively punished (drives forward motion)
        reward = (self._prev_dist - curr_dist) - 0.05
        terminated = False

        if action == 0 and curr_rc == prev_rc:  # blocked forward = collision
            reward += self._cfg.get("collision", -0.5)

        if action == 3:  # stop
            if man <= 1:
                reward += self._cfg.get("success", 10.0)
            else:
                reward += -2.0  # penalize stopping in the wrong place
            terminated = True

        # auto-success: reaching the target cell ends the episode positively
        # (the policy still learns to stop, but this gives a denser success signal)
        if man <= 1 and not terminated:
            reward += self._cfg.get("success", 10.0)
            terminated = True

        truncated = False
        if self._steps >= self._budget:
            reward += self._cfg.get("timeout", -5.0)
            truncated = True

        self._prev_dist = curr_dist
        return self._obs(), reward, terminated, truncated, {}


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._rew, self._lens, self._succ = [], [], []
        self._cr, self._cl = 0.0, 0

    def _on_step(self) -> bool:
        r = self.locals.get("rewards", [0])
        d = self.locals.get("dones", [False])
        self._cr += float(r[0])
        self._cl += 1
        if d[0]:
            self._rew.append(self._cr)
            self._lens.append(self._cl)
            self._succ.append(self._cr >= 9.0)  # success terminal reward is +10
            self._cr, self._cl = 0.0, 0
        return True

    def get_final_stats(self) -> dict:
        lr = self._rew[-100:] or [0]
        ll = self._lens[-100:] or [0]
        ls = self._succ[-100:] or [False]
        return {
            "mean_reward": sum(lr) / len(lr),
            "mean_episode_length": sum(ll) / len(ll),
            "success_rate": sum(ls) / len(ls),
            "total_episodes": len(self._rew),
        }


def main():
    load_dotenv()
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logging.error(f"failed to load config from {args.config}: {exc}")
        sys.exit(1)

    model_path = args.model_path or os.getenv("MODEL_PATH") or "models/policy.zip"

    try:
        train_env = NavTrainEnv(
            grid_size=cfg["grid_size"],
            object_count=cfg["object_count"],
            step_budget=cfg["step_budget"],
            cfg=cfg,
        )
    except Exception as exc:
        logging.error(f"failed to initialize environment: {exc}")
        sys.exit(1)

    policy = None
    if not args.force_retrain and os.path.exists(model_path):
        try:
            policy = load_policy(model_path, train_env)
            logging.info(f"loaded existing policy from {model_path}")
        except RuntimeError as exc:
            logging.error(str(exc))
            sys.exit(1)

    if policy is None:
        try:
            policy = make_policy(train_env)
            logging.info("initialized new policy from scratch")
        except Exception as exc:
            logging.error(f"failed to initialize policy: {exc}")
            sys.exit(1)

    callback = LoggingCallback()
    try:
        policy.learn(total_timesteps=cfg["total_timesteps"], callback=callback)
    except Exception as exc:
        logging.error(f"training failed: {exc}")
        sys.exit(1)

    try:
        policy.save(model_path)
        logging.info(f"saved policy to {model_path}")
    except Exception as exc:
        logging.error(f"failed to save policy to {model_path}: {exc}")
        sys.exit(1)

    stats = callback.get_final_stats()
    logging.info(
        f"training complete — "
        f"mean_reward={stats['mean_reward']:.3f}  "
        f"mean_episode_length={stats['mean_episode_length']:.1f}  "
        f"success_rate={stats['success_rate']:.3f}  "
        f"total_episodes={stats['total_episodes']}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
