"""Training and evaluation script for GridWorld CPP.

Strategy:
    - Custom Dict observation with a fixed-shape local view (5x5) so the
      same policy weights apply on 5x5, 10x10 and 20x20 grids.
    - RecurrentPPO (sb3-contrib) with an LSTM head, since the task is
      partially observable: the agent needs memory of where it has been
      beyond the current local patch.
    - Curriculum learning across three stages, with transfer learning
      between stages (load the previous checkpoint, keep training).

Usage:
    # Stage 1: from scratch on 5x5
    python train_grid_world_cpp.py train --stage 5

    # Stage 2: continue on 10x10, starting from a stage-5 checkpoint
    python train_grid_world_cpp.py train --stage 10 --from data/recppo_stage5_<ts>

    # Stage 3 (optional, for the extra credit): continue on 20x20
    python train_grid_world_cpp.py train --stage 20 --from data/recppo_stage10_<ts>

    # Evaluate
    python train_grid_world_cpp.py test --model data/recppo_stage10_<ts> \\
        --size 10 --obs 12 --max-steps 400
    python train_grid_world_cpp.py eval_all --model data/recppo_stage10_<ts>

    # Watch a single rollout
    python train_grid_world_cpp.py run --model data/recppo_stage10_<ts> \\
        --size 10 --obs 12 --max-steps 400
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sb3_contrib import RecurrentPPO

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from gymnasium_env.cpp_features_extractor import CPPFeaturesExtractor


# --------------------------------------------------------------------- config

LOCAL_VIEW = 5  # 5x5 patch -> same shape on every grid stage

# Per-stage curriculum config. Keys are grid sizes (as strings) so the CLI
# can reference them. Numbers were chosen so total training time across all
# stages on a CPU is in the few-hour range.
STAGE_CONFIGS = {
    "5": {
        "size": 5,
        "obstacles": 3,
        "max_steps": 200,
        "timesteps": 1_500_000,
        "lr_schedule": (3.0e-4, 1.0e-4),  # linear from-to
        "ent_coef": 0.05,                 # high exploration on 5x5
        "n_envs": 8,
    },
    "10": {
        "size": 10,
        "obstacles": 12,
        "max_steps": 400,
        "timesteps": 2_000_000,
        "lr_schedule": (1.5e-4, 5.0e-5),
        "ent_coef": 0.02,                 # less exploration once we transfer in
        "n_envs": 8,
    },
    "20": {
        "size": 20,
        "obstacles": 48,
        "max_steps": 800,
        "timesteps": 3_000_000,
        "lr_schedule": (1.0e-4, 3.0e-5),
        "ent_coef": 0.01,
        "n_envs": 8,
    },
}


# --------------------------------------------------------------------- utils

def print_action(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")


def linear_schedule(initial_value: float, final_value: float):
    """Linear LR schedule used by SB3: callable of progress_remaining."""
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func


def normalize_model_path(path: Optional[str]) -> Optional[str]:
    """SB3 saves/loads ``*.zip``; CLI may omit the suffix."""
    if path is None:
        return None
    return path if path.endswith(".zip") else f"{path}.zip"


def env_kwargs_from_stage(stage: dict, render_mode: Optional[str] = None) -> dict:
    return {
        "size": stage["size"],
        "obs_quantity": stage["obstacles"],
        "max_steps": stage["max_steps"],
        "local_view_size": LOCAL_VIEW,
        "render_mode": render_mode,
    }


def make_train_envs(stage: dict, n_envs: int) -> gym.Env:
    env_kwargs = env_kwargs_from_stage(stage, render_mode=None)
    if n_envs <= 1:
        return make_vec_env(
            GridWorldCPPEnv, n_envs=1, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv
        )
    try:
        return make_vec_env(
            GridWorldCPPEnv, n_envs=n_envs, env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
        )
    except Exception as exc:
        # Subproc can fail on Windows / inside notebooks; fall back gracefully.
        print(f"[WARN] SubprocVecEnv failed ({exc}); falling back to DummyVecEnv")
        return make_vec_env(
            GridWorldCPPEnv, n_envs=n_envs, env_kwargs=env_kwargs,
            vec_env_cls=DummyVecEnv,
        )


def make_single_env(stage: dict, render_mode: Optional[str] = None) -> gym.Env:
    return GridWorldCPPEnv(**env_kwargs_from_stage(stage, render_mode=render_mode))


def policy_kwargs_factory() -> dict:
    """Policy config: custom CNN extractor + LSTM + small MLP heads."""
    return dict(
        features_extractor_class=CPPFeaturesExtractor,
        features_extractor_kwargs=dict(cnn_features_dim=64, vec_features_dim=32),
        lstm_hidden_size=128,
        n_lstm_layers=1,
        enable_critic_lstm=True,
        net_arch=dict(pi=[64], vf=[64]),
        share_features_extractor=True,
        normalize_images=False,
    )


def build_or_load_model(
    env, stage: dict, log_dir: str, from_path: Optional[str]
) -> RecurrentPPO:
    lr = linear_schedule(*stage["lr_schedule"])
    ent_coef = stage["ent_coef"]

    common_kwargs = dict(
        verbose=1,
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        learning_rate=lr,
        ent_coef=ent_coef,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir,
        device="auto",
    )

    if from_path is None:
        return RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            policy_kwargs=policy_kwargs_factory(),
            **common_kwargs,
        )

    print(f"[INFO] Loading checkpoint from {normalize_model_path(from_path)}")
    model = RecurrentPPO.load(
        normalize_model_path(from_path),
        env=env,
        device="auto",
        # Override the saved schedule with the new stage's schedule.
        custom_objects={
            "learning_rate": lr,
            "lr_schedule": lr,
            "clip_range": 0.2,
        },
    )
    model.ent_coef = ent_coef
    model.tensorboard_log = log_dir
    return model


# ---------------------------------------------------------------- subcommands

def cmd_train(args: argparse.Namespace) -> None:
    if args.stage not in STAGE_CONFIGS:
        raise SystemExit(f"Unknown stage {args.stage}. Choose one of {list(STAGE_CONFIGS)}")

    stage_key = args.stage
    stage = dict(STAGE_CONFIGS[stage_key])
    if args.timesteps:
        stage["timesteps"] = args.timesteps
    if args.n_envs:
        stage["n_envs"] = args.n_envs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"log/recppo_stage{stage_key}_{timestamp}"
    os.makedirs("data", exist_ok=True)

    print(f"--- Training stage {stage_key} ---")
    print(f"    size={stage['size']}  obstacles={stage['obstacles']}  max_steps={stage['max_steps']}")
    print(f"    timesteps={stage['timesteps']:,}  n_envs={stage['n_envs']}")
    print(f"    lr={stage['lr_schedule']}  ent_coef={stage['ent_coef']}")
    if args.from_path:
        print(f"    transfer-learning from: {normalize_model_path(args.from_path)}")

    if args.check_env:
        check_env(make_single_env(stage))

    env = make_train_envs(stage, n_envs=stage["n_envs"])
    model = build_or_load_model(env, stage, log_dir, args.from_path)

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting learn() for {stage['timesteps']:,} timesteps...")
    model.learn(
        total_timesteps=stage["timesteps"],
        reset_num_timesteps=args.from_path is None,
    )

    model_path = f"data/recppo_stage{stage_key}_{timestamp}"
    model.save(model_path)
    print(f"Saved model to {model_path}.zip")
    print(f"Logs at {log_dir}")


def _evaluate_recurrent(
    model: RecurrentPPO,
    env: gym.Env,
    num_episodes: int = 100,
    deterministic: bool = False,
    verbose: bool = True,
) -> dict:
    """Run num_episodes and aggregate coverage/step statistics."""
    full_coverage_count = 0
    coverages = []
    steps_list = []

    for i in range(num_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        truncated = False
        steps = 0
        while not (done or truncated):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            obs, reward, done, truncated, info = env.step(action.item())
            episode_starts = np.array([False], dtype=bool)
            steps += 1

        coverages.append(info["coverage"])
        steps_list.append(steps)
        if done and not truncated:
            full_coverage_count += 1
            if verbose:
                print(f"Episode {i + 1:3d}: full coverage in {steps} steps")
        elif verbose:
            print(f"Episode {i + 1:3d}: coverage {info['coverage']:.1%} in {steps} steps")

    return {
        "num_episodes": num_episodes,
        "full_coverage": full_coverage_count,
        "full_coverage_rate": full_coverage_count / num_episodes,
        "avg_coverage": float(np.mean(coverages)),
        "std_coverage": float(np.std(coverages)),
        "min_coverage": float(np.min(coverages)),
        "max_coverage": float(np.max(coverages)),
        "avg_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "min_steps": int(np.min(steps_list)),
        "max_steps": int(np.max(steps_list)),
    }


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    print(
        f"Full Coverage Rate: {metrics['full_coverage_rate']:.2%} "
        f"({metrics['full_coverage']}/{metrics['num_episodes']})"
    )
    print(
        f"Avg Coverage:       {metrics['avg_coverage']:.2%} "
        f"(std {metrics['std_coverage']:.2%}, "
        f"min {metrics['min_coverage']:.2%}, max {metrics['max_coverage']:.2%})"
    )
    print(
        f"Avg Steps:          {metrics['avg_steps']:.1f} "
        f"(std {metrics['std_steps']:.1f}, "
        f"min {metrics['min_steps']}, max {metrics['max_steps']})"
    )


def cmd_test(args: argparse.Namespace) -> None:
    model_path = normalize_model_path(args.model)
    print(f"--- Loading model {model_path} ---")
    model = RecurrentPPO.load(model_path, device="auto")

    if args.size is None or args.obs is None or args.max_steps is None:
        raise SystemExit("test requires --size, --obs and --max-steps")

    stage = {"size": args.size, "obstacles": args.obs, "max_steps": args.max_steps}
    env = make_single_env(stage)

    metrics = _evaluate_recurrent(
        model, env, num_episodes=args.episodes,
        deterministic=args.deterministic, verbose=True,
    )
    _print_metrics(
        f"Test {args.size}x{args.size} obs={args.obs} max_steps={args.max_steps}",
        metrics,
    )


def cmd_eval_all(args: argparse.Namespace) -> None:
    model_path = normalize_model_path(args.model)
    print(f"--- Loading model {model_path} ---")
    model = RecurrentPPO.load(model_path, device="auto")

    grids = [(5, 3, 200), (10, 12, 400)]
    if args.include_20:
        grids.append((20, 48, 800))

    summary = []
    for size, obs_q, max_steps in grids:
        env = make_single_env({"size": size, "obstacles": obs_q, "max_steps": max_steps})
        metrics = _evaluate_recurrent(
            model, env, num_episodes=args.episodes,
            deterministic=args.deterministic, verbose=False,
        )
        metrics["grid"] = f"{size}x{size}"
        metrics["obstacles"] = obs_q
        metrics["max_steps_cap"] = max_steps
        summary.append(metrics)
        _print_metrics(f"{size}x{size} (obs={obs_q}, max_steps={max_steps})", metrics)

    print("\n=== Summary ===")
    print(
        f"{'grid':>6} | {'obs':>3} | {'cap':>4} | {'full%':>7} | "
        f"{'avg cov%':>8} | {'std cov%':>8} | {'avg steps':>9}"
    )
    print("-" * 70)
    for m in summary:
        print(
            f"{m['grid']:>6} | {m['obstacles']:>3} | {m['max_steps_cap']:>4} | "
            f"{m['full_coverage_rate'] * 100:>6.1f}% | "
            f"{m['avg_coverage'] * 100:>7.2f}% | "
            f"{m['std_coverage'] * 100:>7.2f}% | "
            f"{m['avg_steps']:>9.1f}"
        )


def cmd_run(args: argparse.Namespace) -> None:
    model_path = normalize_model_path(args.model)
    print(f"--- Loading model {model_path} ---")
    model = RecurrentPPO.load(model_path, device="auto")

    if args.size is None or args.obs is None or args.max_steps is None:
        raise SystemExit("run requires --size, --obs and --max-steps")

    stage = {"size": args.size, "obstacles": args.obs, "max_steps": args.max_steps}
    env = make_single_env(stage, render_mode="human")

    obs, info = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    done = False
    truncated = False
    steps = 0
    total_reward = 0.0
    while not (done or truncated):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=args.deterministic,
        )
        obs, reward, done, truncated, info = env.step(action.item())
        episode_starts = np.array([False], dtype=bool)
        steps += 1
        total_reward += float(reward)
        print(
            f"Step {steps:4d} | action={print_action(action.item()):5s} | "
            f"reward={reward:+5.2f} | coverage={info['coverage']:.1%}"
        )
    print(
        f"--- Run finished --- total reward: {total_reward:.2f}, "
        f"coverage: {info['coverage']:.1%}, steps: {steps}, "
        f"full coverage: {done and not truncated}"
    )


# ----------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GridWorld CPP training/evaluation with RecurrentPPO + curriculum"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train an agent for a given stage")
    p_train.add_argument("--stage", default="5", choices=list(STAGE_CONFIGS),
                         help="Curriculum stage (grid size as key)")
    p_train.add_argument("--from", dest="from_path", default=None,
                         help="Optional checkpoint to start from (transfer learning)")
    p_train.add_argument("--timesteps", type=int, default=None,
                         help="Override total_timesteps for this run")
    p_train.add_argument("--n-envs", type=int, default=None,
                         help="Override number of vectorized envs")
    p_train.add_argument("--check-env", action="store_true",
                         help="Run stable-baselines3 env_checker before training")

    p_test = sub.add_parser("test", help="Evaluate a saved model on a single grid")
    p_test.add_argument("--model", required=True, help="Path (no .zip) to saved model")
    p_test.add_argument("--size", type=int, required=True)
    p_test.add_argument("--obs", type=int, required=True)
    p_test.add_argument("--max-steps", dest="max_steps", type=int, required=True)
    p_test.add_argument("--episodes", type=int, default=100)
    p_test.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions (default: stochastic)")

    p_eval = sub.add_parser("eval_all", help="Evaluate a model on 5x5, 10x10 (and 20x20)")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--episodes", type=int, default=100)
    p_eval.add_argument("--include-20", action="store_true",
                        help="Also evaluate on 20x20 (extra credit grid)")
    p_eval.add_argument("--deterministic", action="store_true")

    p_run = sub.add_parser("run", help="Render a single rollout from a saved model")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--size", type=int, required=True)
    p_run.add_argument("--obs", type=int, required=True)
    p_run.add_argument("--max-steps", dest="max_steps", type=int, required=True)
    p_run.add_argument("--deterministic", action="store_true")

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "test":
        cmd_test(args)
    elif args.cmd == "eval_all":
        cmd_eval_all(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()