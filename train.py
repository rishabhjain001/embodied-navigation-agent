import argparse
import logging
import os
import sys

import yaml
from dotenv import load_dotenv
from stable_baselines3.common.callbacks import BaseCallback

from agent.env import make_env
from agent.policy import make_policy, load_policy, wrap_for_training


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    result = {}
    result.update(cfg.get("ppo", {}))
    result.update(cfg.get("env", {}))
    result.update(cfg.get("reward", {}))

    # env vars take precedence over YAML values
    overrides = {
        "grid_size": os.getenv("GRID_SIZE"),
        "step_budget": os.getenv("STEP_BUDGET"),
        "object_count": os.getenv("OBJECT_COUNT"),
        "learning_rate": os.getenv("LEARNING_RATE"),
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
    parser = argparse.ArgumentParser(description="Train the navigation policy")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--model-path", default=None, help="Path to save/load model weights")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore existing model, train from scratch")
    return parser.parse_args()


class LoggingCallback(BaseCallback):
    """Tracks per-episode metrics; call get_final_stats() after training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_successes: list[bool] = []
        self._current_episode_reward = 0.0
        self._current_episode_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0])
        dones = self.locals.get("dones", [False])

        self._current_episode_reward += float(rewards[0])
        self._current_episode_length += 1

        if dones[0]:
            self._episode_rewards.append(self._current_episode_reward)
            self._episode_lengths.append(self._current_episode_length)
            # success terminal reward is +10.0; any episode reaching that threshold counts
            self._episode_successes.append(self._current_episode_reward >= 10.0)
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

        return True

    def get_final_stats(self) -> dict:
        last = self._episode_rewards[-100:]
        last_lens = self._episode_lengths[-100:]
        last_success = self._episode_successes[-100:]
        return {
            "mean_reward": sum(last) / len(last) if last else 0.0,
            "mean_episode_length": sum(last_lens) / len(last_lens) if last_lens else 0.0,
            "success_rate": sum(last_success) / len(last_success) if last_success else 0.0,
            "total_episodes": len(self._episode_rewards),
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
        env = make_env(
            grid_size=cfg["grid_size"],
            object_count=cfg["object_count"],
        )
        # SB3 requires a flat Box obs space — wrap the MiniGrid Dict obs
        train_env = wrap_for_training(env)
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
        policy.learn(
            total_timesteps=cfg["total_timesteps"],
            callback=callback,
        )
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
