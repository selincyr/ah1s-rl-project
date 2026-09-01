import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback
)

from helicopter_env_3_v4 import HelicopterEnv


def main():

    print("AH-1S V4 training baslatiliyor...")

    os.makedirs(
        "models/task3_trim_v4",
        exist_ok=True
    )

    os.makedirs(
        "logs/task3_trim_v4",
        exist_ok=True
    )

    train_env = Monitor(
        HelicopterEnv()
    )

    eval_env = Monitor(
        HelicopterEnv()
    )

    print(
        "Observation space:",
        train_env.observation_space.shape
    )

    print(
        "Action space:",
        train_env.action_space.shape
    )

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

        verbose=1,

        tensorboard_log=(
            "logs/task3_trim_v4/"
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,

        save_path=(
            "models/task3_trim_v4/"
        ),

        name_prefix="checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,

        best_model_save_path=(
            "models/task3_trim_v4/best/"
        ),

        log_path=(
            "logs/task3_trim_v4/eval/"
        ),

        eval_freq=20_000,

        n_eval_episodes=3,

        deterministic=True,

        render=False
    )

    model.learn(
        total_timesteps=300_000,

        callback=[
            checkpoint_callback,
            eval_callback
        ]
    )

    model.save(
        "models/task3_trim_v4/"
        "ppo_ah1s_task3_trim_v4_final"
    )

    train_env.close()
    eval_env.close()

    print()
    print("V4 training tamamlandi.")


if __name__ == "__main__":
    main()
