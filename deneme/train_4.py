import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback
)

from helicopter_env_4 import HelicopterEnv


def main():

    print("AH-1S stable-hover curriculum training baslatiliyor...")

    # =====================================================
    # FOLDERS
    # =====================================================

    os.makedirs(
        "models/task4_hover",
        exist_ok=True
    )

    os.makedirs(
        "logs/task4_hover",
        exist_ok=True
    )

    # =====================================================
    # ENVIRONMENTS
    # =====================================================

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

    # =====================================================
    # PPO MODEL
    # =====================================================

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
            "logs/task4_hover/"
        )
    )

    # =====================================================
    # CHECKPOINT CALLBACK
    # =====================================================

    checkpoint_callback = CheckpointCallback(

        save_freq=50_000,

        save_path=(
            "models/task4_hover/"
        ),

        name_prefix="checkpoint"
    )

    # =====================================================
    # EVALUATION CALLBACK
    # =====================================================

    eval_callback = EvalCallback(

        eval_env,

        best_model_save_path=(
            "models/task4_hover/best/"
        ),

        log_path=(
            "logs/task4_hover/eval/"
        ),

        eval_freq=20_000,

        # Environment su an deterministik oldugu icin
        # ayni testi 3 kez tekrarlamaya gerek yok.
        n_eval_episodes=1,

        deterministic=True,

        render=False
    )

    # =====================================================
    # TRAINING
    # =====================================================

    model.learn(

        total_timesteps=300_000,

        callback=[
            checkpoint_callback,
            eval_callback
        ]
    )

    # =====================================================
    # FINAL MODEL
    # =====================================================

    model.save(
        "models/task4_hover/"
        "ppo_ah1s_hover_final"
    )

    train_env.close()
    eval_env.close()

    print()
    print("Training tamamlandi.")

    print(
        "Final model:"
    )

    print(
        "models/task4_hover/"
        "ppo_ah1s_hover_final.zip"
    )


if __name__ == "__main__":
    main()
