#
# python train_grid_world_cpp.py <train|test|run|curriculum> dim obstacles max_steps total_timesteps
#

import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from datetime import datetime
import sys

def print_action(action: int) -> str:
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")

if sys.argv[1] not in ['train', 'test', 'run', 'curriculum']:
    print("Usage: python train_grid_world_cpp.py <train|test|run|curriculum> dim obstacles max_steps total_timesteps")
    sys.exit(1)
elif sys.argv[1] in ['train','curriculum']:
    if len(sys.argv) != 6:
        print("Usage for training: python train_grid_world_cpp.py train|curriculum dim obstacles max_steps total_timesteps")
        sys.exit(1)
elif sys.argv[1] in ['test', 'run']:
    if len(sys.argv) != 4:
        print("Usage for testing/running: python train_grid_world_cpp.py test|run dim obstacles")
        sys.exit(1)

# --- Hyperparameters ---
mode = sys.argv[1]
DIM = int(sys.argv[2]) # 5, 10, 20
OBSTACLES = int(sys.argv[3]) # 3, 12, 48

if mode in ['train', 'curriculum']:
    MAX_STEPS = int(sys.argv[4]) # 200, 500, 1000
    TOTAL_TIMESTEPS = int(sys.argv[5]) # 500_000
else:
    # test / run: derive a sensible default cap on episode length from the grid size
    # so the agent has enough steps to attempt full coverage. Same heuristic used during training.
    MAX_STEPS = 200 if DIM <= 5 else 400 if DIM <= 10 else 800
    TOTAL_TIMESTEPS = 0  # unused in test/run

ENTROPY_COEF = 0.05
# -----------------------

try:
    gym.register(
        id="gymnasium_env/GridWorldCPP-v0",
        entry_point=GridWorldCPPEnv,
    )
except Exception:
    pass



if mode == 'train':
    print("--- Starting CPP Training ---")
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )
    check_env(env)

    model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=ENTROPY_COEF, device="cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'log/ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}'
    model_path = f'data/ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}.zip'

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting learning with {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == 'curriculum':

    print("--- Starting CPP Curriculum Learning Training ---")

    model_name = input("Enter source model filename (e.g., ppo_cpp_5_3_200_0.05_20260324_100000): ")
    source_model_path = f'data/{model_name}.zip'

    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )

    # Load the previously trained policy onto the new (larger) env. The 7x7
    # local view + normalized agent vector make the observation space identical
    # across grid sizes, so the loaded weights are directly compatible.
    model = PPO.load(source_model_path, env=env, device="cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'log/ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}_curriculum'
    new_model_path = f'data/ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}_curriculum.zip'

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Continuing learning for {TOTAL_TIMESTEPS} timesteps (reset_num_timesteps=False)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
    model.save(new_model_path)
    print(f"Model trained and saved to {new_model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == 'run':
    model_name = input("Enter model filename (e.g., ppo_cpp_5_3_200_0.05_20260324_100000): ")
    model_path = f'data/{model_name}.zip'
    print(f'--- Loading model from {model_path} for a run ---')

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="human"
    )

    (obs, info) = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action.item())
        total_reward += reward
        steps += 1
        print(f"Step: {steps}, Action: {print_action(action.item())}, "
              f"Reward: {reward:.2f}, Coverage: {info['coverage']:.1%}, "
              f"Done: {done}, Truncated: {truncated}")
    print(f"--- Run Finished --- Total reward: {total_reward:.2f}, Coverage: {info['coverage']:.1%}")

elif mode == 'test':
    model_name = input("Enter model filename (e.g., ppo_cpp_5_3_200_0.05_20260324_100000): ")
    model_path = f'data/{model_name}.zip'
    print(f'--- Loading model from {model_path} for testing ---')

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )

    num_episodes = 100
    full_coverage_count = 0
    total_coverages = []
    total_steps_list = []

    for i in range(num_episodes):
        (obs, info) = env.reset()
        done = False
        truncated = False
        steps = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action.item())
            steps += 1

        total_coverages.append(info['coverage'])
        total_steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1
            print(f"Episode {i+1}: Full coverage in {steps} steps.")
        else:
            print(f"Episode {i+1}: Coverage {info['coverage']:.1%} in {steps} steps.")

    import numpy as np
    full_coverage_rate = (full_coverage_count / num_episodes) * 100
    avg_coverage = np.mean(total_coverages) * 100
    standard_deviation = np.std(total_coverages) * 100
    avg_steps = np.mean(total_steps_list)
    standard_deviation_steps = np.std(total_steps_list)
    print(f"\n--- Test Finished ---")
    print(f"Full Coverage Rate: {full_coverage_rate:.2f}% ({full_coverage_count}/{num_episodes})")
    print(f"Average Coverage: {avg_coverage:.2f}% Standard Deviation: {standard_deviation:.2f}% Min Coverage: {np.min(total_coverages)*100:.2f}% Max Coverage: {np.max(total_coverages)*100:.2f}%")
    print(f"Average Steps: {avg_steps:.1f} Standard Deviation: {standard_deviation_steps:.1f} Min Steps: {np.min(total_steps_list)} Max Steps: {np.max(total_steps_list)}")
