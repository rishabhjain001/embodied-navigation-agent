import hashlib

import numpy as np
import gymnasium as gym
import yaml
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO

GOAL_ENCODING_DIM = 64
OBS_DIM = 512 + GOAL_ENCODING_DIM  # 576

# Default dummy goal used during training (no instruction available)
_DUMMY_GOAL = {"target": None, "attribute": None, "context": None, "error": None}


def build_obs_space() -> gym.spaces.Box:
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)


def _field_vector(value: str | None) -> np.ndarray:
    # use md5 for a stable, cross-process hash — Python's built-in hash() is not stable
    text = value if value is not None else "null"
    digest = hashlib.md5(text.encode()).digest()
    seed = int.from_bytes(digest[:4], "little") % (2**31)
    return np.random.default_rng(seed).random(16).astype(np.float32)


def encode_goal(goal: dict) -> np.ndarray:
    parts = [
        _field_vector(goal.get("target")),
        _field_vector(goal.get("attribute")),
        _field_vector(goal.get("context")),
        _field_vector(goal.get("error")),
    ]
    return np.concatenate(parts)  # (64,)


def make_observation(frame_embedding: np.ndarray, goal: dict) -> np.ndarray:
    # layout: [0:512] frame, [512:576] goal encoding — matches design doc observation spec
    return np.concatenate([frame_embedding, encode_goal(goal)])  # (576,)


class FlatObsWrapper(ObservationWrapper):
    """Converts a MiniGrid Dict obs to a flat (576,) float32 vector for SB3.

    Uses real CLIP embeddings so the policy trains on the same observation
    space it will see at inference time via the backend.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = build_obs_space()
        # Import vision here to avoid loading CLIP unless this wrapper is used
        from agent.vision import encode_frame as _encode
        self._encode = _encode

    def observation(self, obs):
        frame_emb = self._encode(obs["image"])  # float32 (512,), L2-normalized
        return make_observation(frame_emb, _DUMMY_GOAL)  # (576,)


def wrap_for_training(env) -> gym.Env:
    """Wrap a MiniGrid env so it has the flat (576,) obs space SB3 expects."""
    return FlatObsWrapper(env)


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
