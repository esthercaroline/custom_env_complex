"""Evaluation that filters out unsolvable random layouts.

The GridWorldCPPEnv places obstacles uniformly at random, with no guarantee
that the free cells form a single connected component. A small fraction of
random seeds therefore produce layouts where 100% coverage is mathematically
impossible (some free cells are walled off from the agent's spawn).

This script does two things:

1. `count`: samples N random layouts and reports how many are unsolvable.
2. `eval`: same as `train_grid_world_cpp.test`, but reports both raw and
   solvable-only metrics.

Usage:
    # Quick: how many seeds are unsolvable on 10x10?
    python eval_solvable.py count --size 10 --obs 12 --n 1000

    # Full eval, filtered:
    python eval_solvable.py eval --model data/recppo_stage10_<ts> \\
        --size 10 --obs 12 --max-steps 400 --episodes 200
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from sb3_contrib import RecurrentPPO

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv, FREE, WALL


def is_solvable(env: GridWorldCPPEnv) -> bool:
    """BFS from agent spawn over free cells. True if every free cell is
    reachable from spawn (i.e. 100% coverage is possible on this layout)."""
    grid = env._grid
    size = env.size
    spawn = (int(env._agent[0]), int(env._agent[1]))

    if grid[spawn[0], spawn[1]] == WALL:
        return False

    total_free = int((grid == FREE).sum())
    seen = {spawn}
    q = deque([spawn])

    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and grid[nx, ny] == FREE:
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))

    return len(seen) == total_free


def cmd_count(args: argparse.Namespace) -> None:
    env = GridWorldCPPEnv(
        size=args.size, obs_quantity=args.obs,
        max_steps=100, local_view_size=5,
    )
    unsolvable = 0
    for i in range(args.n):
        env.reset(seed=args.seed + i if args.seed is not None else None)
        if not is_solvable(env):
            unsolvable += 1
    rate = unsolvable / args.n
    print(f"\n{args.size}x{args.size} grid, {args.obs} obstacles, {args.n} samples:")
    print(f"  unsolvable layouts: {unsolvable}/{args.n} = {rate:.2%}")
    print(f"  solvable layouts:   {args.n - unsolvable}/{args.n} = {1 - rate:.2%}")
    print(f"  -> realistic max full-coverage rate: ~{(1 - rate) * 100:.1f}/100\n")


def cmd_eval(args: argparse.Namespace) -> None:
    print(f"Loading {args.model} ...")
    model = RecurrentPPO.load(args.model, device="auto")

    env = GridWorldCPPEnv(
        size=args.size, obs_quantity=args.obs,
        max_steps=args.max_steps, local_view_size=5,
    )

    n_total = 0
    n_solvable = 0
    full_total = 0
    full_solvable = 0
    coverages_all = []
    coverages_solvable = []

    for i in range(args.episodes):
        obs, info = env.reset()
        solvable = is_solvable(env)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        truncated = False
        steps = 0
        while not (done or truncated):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts,
                deterministic=args.deterministic,
            )
            obs, reward, done, truncated, info = env.step(action.item())
            episode_starts = np.array([False], dtype=bool)
            steps += 1

        n_total += 1
        coverages_all.append(info["coverage"])
        full = (done and not truncated)
        if full:
            full_total += 1
        if solvable:
            n_solvable += 1
            coverages_solvable.append(info["coverage"])
            if full:
                full_solvable += 1

        if args.verbose:
            tag = "[solv]  " if solvable else "[UNSOLV]"
            outcome = "FULL" if full else f"{info['coverage']:.1%}"
            print(f"Ep {i+1:3d} {tag} {outcome:>6s} in {steps} steps")

    print(f"\n=== {args.size}x{args.size} obs={args.obs} max_steps={args.max_steps} ===")
    print(f"Episodes:           {n_total}")
    print(f"Solvable layouts:   {n_solvable}/{n_total} ({n_solvable/n_total:.1%})")
    print()
    print(f"--- Raw (all episodes) ---")
    print(f"Full coverage rate: {full_total}/{n_total} = {full_total/n_total:.2%}")
    print(f"Avg coverage:       {np.mean(coverages_all):.2%} "
          f"(std {np.std(coverages_all):.2%})")
    print()
    if n_solvable > 0:
        print(f"--- Solvable layouts only ---")
        print(f"Full coverage rate: {full_solvable}/{n_solvable} = "
              f"{full_solvable/n_solvable:.2%}")
        print(f"Avg coverage:       {np.mean(coverages_solvable):.2%} "
              f"(std {np.std(coverages_solvable):.2%})")
    print()


def build_parser():
    p = argparse.ArgumentParser(description="Connectivity-aware evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("count", help="How many random layouts are unsolvable?")
    pc.add_argument("--size", type=int, required=True)
    pc.add_argument("--obs", type=int, required=True)
    pc.add_argument("--n", type=int, default=1000)
    pc.add_argument("--seed", type=int, default=None)

    pe = sub.add_parser("eval", help="Evaluate model with solvable filter")
    pe.add_argument("--model", required=True)
    pe.add_argument("--size", type=int, required=True)
    pe.add_argument("--obs", type=int, required=True)
    pe.add_argument("--max-steps", dest="max_steps", type=int, required=True)
    pe.add_argument("--episodes", type=int, default=200)
    pe.add_argument("--deterministic", action="store_true")
    pe.add_argument("--verbose", action="store_true")

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "count":
        cmd_count(args)
    elif args.cmd == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()