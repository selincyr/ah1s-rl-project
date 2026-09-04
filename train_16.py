import os
import random

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from helicopter_env_16 import HelicopterEnv


# ==========================================================
# V16-A CONTROLLED EXPERIMENT SETTINGS
# ==========================================================

SEED = 42

MODEL_DIR = "models/task16b_vertical300"
LOG_DIR = "logs/task16b_vertical300"

# First experiment is intentionally short.
# We only want to see whether the normalized reward learns
# a physically meaningful trajectory before spending 400k steps.
TOTAL_TIMESTEPS = 75_000


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(seed):
    env = HelicopterEnv()
    env.reset(seed=seed)
    return Monitor(env)


def main():
    set_global_seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/best", exist_ok=True)

    train_env = make_env(SEED)
    eval_env = make_env(SEED + 1)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_ah1s_vertical300_v16b",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/best",
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    print("=" * 80)
    print("V16-A CONTROLLED REWARD EXPERIMENT")
    print("Reward: physics-normalized baseline")
    print("Seed:", SEED)
    print("Observation size: 15")
    print("Timesteps:", TOTAL_TIMESTEPS)
    print("=" * 80)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            checkpoint_callback,
            eval_callback,
        ],
        progress_bar=True,
    )

    final_model_path = (
        f"{MODEL_DIR}/"
        "ppo_ah1s_vertical300_v16a_final"
    )

    model.save(final_model_path)

    train_env.close()
    eval_env.close()

    print("=" * 80)
    print("V16-A TRAINING COMPLETED")
    print("BEST :", f"{MODEL_DIR}/best/best_model")
    print("FINAL:", final_model_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
