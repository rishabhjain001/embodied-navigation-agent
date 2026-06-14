#!/usr/bin/env python3
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from agent.env import make_env
from agent.policy import load_policy, predict_action, make_observation
from agent.vision import encode_frame
from train import load_config

logging.basicConfig(level=logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained navigation policy")
    parser.add_argument("--model-path", required=True, help="Path to saved PPO weights (.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    return parser.parse_args()


# dummy goal used when no instruction is available in standalone eval
_DUMMY_GOAL = {"target": None, "attribute": None, "context": None, "error": None}


def run_episode_tracked(env, policy, step_budget: int, seed: int | None) -> dict:
    """Run one episode and return per-episode metrics."""
    obs, _ = env.reset(seed=seed)
    frame = obs["image"]

    steps = 0
    forward_moves = 0
    done = False
    info = {}

    while not done and steps < step_budget:
        embedding = encode_frame(frame)
        observation = make_observation(embedding, _DUMMY_GOAL)
        action = predict_action(policy, observation)

        obs, _reward, terminated, truncated, info = env.step(action)
        frame = obs["image"]
        steps += 1

        if action == 0:
            forward_moves += 1

        if action == 3 or terminated or truncated:
            done = True

    collisions = info.get("collision_count", 0)
    # path_length counts all forward-move actions (including blocked ones mirrors
    # what the API session tracker does: every move_forward issued = one step along path)
    path_length = forward_moves

    # no real target in standalone eval; success is always False
    success = False

    # navigation efficiency is undefined without a real target
    efficiency = 0.0

    return {
        "success": success,
        "steps": steps,
        "path_length": path_length,
        "collisions": collisions,
        "efficiency": efficiency,
    }


def print_table(results: list[dict]) -> None:
    header = f"{'Episode':>7}  {'Result':<9}  {'Steps':>5}  {'PathLen':>7}  {'Collisions':>10}  {'Efficiency':>10}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, start=1):
        result_str = "success" if r["success"] else "failure"
        print(
            f"{i:>7}  {result_str:<9}  {r['steps']:>5}  {r['path_length']:>7}  "
            f"{r['collisions']:>10}  {r['efficiency']:>10.3f}"
        )

    print("-" * len(header))

    n = len(results)
    success_rate = sum(r["success"] for r in results) / n
    mean_steps = sum(r["steps"] for r in results) / n
    mean_path = sum(r["path_length"] for r in results) / n
    mean_coll = sum(r["collisions"] for r in results) / n
    mean_eff = sum(r["efficiency"] for r in results) / n

    print(
        f"{'AGGREGATE':>7}  {'-':<9}  {mean_steps:>5.1f}  {mean_path:>7.1f}  "
        f"{mean_coll:>10.1f}  {mean_eff:>10.3f}"
    )
    print(f"\nSuccess rate: {success_rate:.1%}  ({int(success_rate * n)}/{n})")


def main():
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logging.error(f"failed to load config from {args.config}: {exc}")
        sys.exit(1)

    step_budget = cfg.get("step_budget", 200)
    grid_size = cfg.get("grid_size", 10)
    object_count = cfg.get("object_count", 5)

    try:
        env = make_env(grid_size=grid_size, object_count=object_count)
    except Exception as exc:
        logging.error(f"failed to initialize environment: {exc}")
        sys.exit(1)

    try:
        policy = load_policy(args.model_path, env)
    except RuntimeError as exc:
        logging.error(str(exc))
        sys.exit(1)

    print(f"Evaluating {args.model_path} for {args.episodes} episode(s)  (step_budget={step_budget})\n")

    results = []
    for i in range(args.episodes):
        episode_seed = (args.seed + i) if args.seed is not None else None
        result = run_episode_tracked(env, policy, step_budget, episode_seed)
        results.append(result)

    print_table(results)


if __name__ == "__main__":
    main()
