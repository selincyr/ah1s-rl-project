import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback
)

from helicopter_env_8 import HelicopterEnv


MODEL_DIR = "models/task8_hover300"
LOG_DIR = "logs/task8_hover300"


def main():

    # -----------------------------------------------------
    # KLASORLER
    # -----------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        LOG_DIR,
        exist_ok=True
    )

    os.makedirs(
        f"{MODEL_DIR}/best",
        exist_ok=True
    )


    # -----------------------------------------------------
    # ENVIRONMENTS
    # -----------------------------------------------------

    train_env = Monitor(
        HelicopterEnv()
    )

    eval_env = Monitor(
        HelicopterEnv()
    )


    # -----------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------

    checkpoint_callback = CheckpointCallback(

        save_freq=50000,

        save_path=MODEL_DIR,

        name_prefix=(
            "ppo_ah1s_hover300_v8"
        )
    )


    eval_callback = EvalCallback(

        eval_env,

        best_model_save_path=(
            f"{MODEL_DIR}/best"
        ),

        log_path=LOG_DIR,

        eval_freq=20000,

        n_eval_episodes=1,

        deterministic=True,

        render=False
    )


    # -----------------------------------------------------
    # PPO MODEL
    # -----------------------------------------------------

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

        seed=42,

        tensorboard_log=LOG_DIR
    )


    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("AH-1S PPO V8 TRAINING")
    print("Target altitude: 300 ft")
    print("Goal: stable hover")
    print("=" * 70)
    print()


    model.learn(

        total_timesteps=300_000,

        callback=[
            checkpoint_callback,
            eval_callback
        ],

        progress_bar=True
    )


    # -----------------------------------------------------
    # FINAL MODEL SAVE
    # -----------------------------------------------------

    final_model_path = (
        f"{MODEL_DIR}/"
        "ppo_ah1s_hover300_v8_final"
    )


    model.save(
        final_model_path
    )


    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("V8 TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Final model:",
        final_model_path + ".zip"
    )

    print(
        "Best model:",
        f"{MODEL_DIR}/best/best_model.zip"
    )


    # -----------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
