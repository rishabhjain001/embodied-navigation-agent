#!/usr/bin/env python3
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from agent.policy import load_policy, predict_action
from train import load_config, NavTrainEnv

logging.basicConfig(level=logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained navigation policy")
    parser.add_argument("--model-path", required=True, help="Path to saved PPO weights (.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    return parser.parse_args()


def run_episode(env, policy, step_budget: int, seed: int | None) -> dict:
    obs, _ = env.reset(seed=seed)
    steps = 0
    forward_moves = 0
    collisions = 0
    success = False

    while steps < step_budget:
        action = predict_action(policy, obs)
        prev_rc = env._agent_rc()
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        if action == 0:
            forward_moves += 1
            if env._agent_rc() == prev_rc:
                collisions += 1
        if terminated:
            # terminated (not truncated) means success in NavTrainEnv
            success = reward > 0
            break
        if truncated:
            break

    start = env._env.agent_pos
    dist = abs(env._target[0] - start[1]) + abs(env._target[1] - start[0])
    efficiency = (dist / forward_moves) if (success and forward_moves > 0 and dist > 0) else (1.0 if success else 0.0)

    return {
        "success": success,
        "steps": steps,
        "path_length": forward_moves,
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

    try:
        env = NavTrainEnv(
            grid_size=cfg["grid_size"],
            object_count=cfg["object_count"],
            step_budget=step_budget,
            cfg=cfg,
        )
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
        results.append(run_episode(env, policy, step_budget, episode_seed))

    print_table(results)


if __name__ == "__main__":
    main()
