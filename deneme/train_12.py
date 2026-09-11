import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback
)

from helicopter_env_12 import HelicopterEnv


# ==============================================================
# PATHS
# ==============================================================

V10_FINAL_MODEL = (
    "models/task10_hover300/"
    "ppo_ah1s_hover300_v10_final"
)

MODEL_DIR = (
    "models/task12_hover300"
)

LOG_DIR = (
    "logs/task12_hover300"
)


# ==============================================================
# TRAINING SETTINGS
# ==============================================================

TOTAL_TIMESTEPS = 150_000

LEARNING_RATE = 5e-5


def main():

    # ==========================================================
    # DIRECTORIES
    # ==========================================================

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

    # ==========================================================
    # ENVIRONMENTS
    # ==========================================================

    train_env = Monitor(
        HelicopterEnv()
    )

    eval_env = Monitor(
        HelicopterEnv()
    )

    # ==========================================================
    # LOAD V10 FINAL
    # ==========================================================
    #
    # V10 FINAL bizim en değerli temelimiz:
    #
    # - 300 ft bölgesine geliyor
    # - level-off yapabiliyor
    # - VS'yi sıfıra yaklaştırabiliyor
    #
    # V12 burada:
    #
    # - forward drift
    # - lateral drift
    # - heading
    # - roll stability
    #
    # taraflarını iyileştirecek.
    # ==========================================================

    print()
    print("=" * 80)
    print("LOADING V10 FINAL MODEL")
    print("=" * 80)

    print(
        "Model:",
        V10_FINAL_MODEL
    )

    model = PPO.load(
        V10_FINAL_MODEL,
        env=train_env,
        tensorboard_log=LOG_DIR
    )

    # ==========================================================
    # FINE-TUNING LEARNING RATE
    # ==========================================================
    #
    # V10 davranışını mümkün olduğunca korumak için
    # V11'den de daha düşük learning rate kullanıyoruz.
    # ==========================================================

    model.learning_rate = (
        LEARNING_RATE
    )

    model.lr_schedule = (
        lambda progress_remaining:
        LEARNING_RATE
    )

    for param_group in (
        model.policy.optimizer.param_groups
    ):

        param_group["lr"] = (
            LEARNING_RATE
        )

    # ==========================================================
    # CHECKPOINT CALLBACK
    # ==========================================================

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=25_000,
            save_path=MODEL_DIR,
            name_prefix=(
                "ppo_ah1s_hover300_v12"
            )
        )
    )

    # ==========================================================
    # EVALUATION CALLBACK
    # ==========================================================

    eval_callback = (
        EvalCallback(
            eval_env,

            best_model_save_path=(
                f"{MODEL_DIR}/best"
            ),

            log_path=LOG_DIR,

            eval_freq=5_000,

            n_eval_episodes=1,

            deterministic=True,

            render=False
        )
    )

    # ==========================================================
    # TRAIN
    # ==========================================================

    print()
    print("=" * 80)
    print("V12 FINE-TUNING STARTED")
    print("=" * 80)

    print(
        "Total timesteps:",
        TOTAL_TIMESTEPS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,

        callback=[
            checkpoint_callback,
            eval_callback
        ],

        reset_num_timesteps=False,

        progress_bar=True
    )

    # ==========================================================
    # FINAL MODEL
    # ==========================================================

    final_model_path = (
        f"{MODEL_DIR}/"
        "ppo_ah1s_hover300_v12_final"
    )

    model.save(
        final_model_path
    )

    # ==========================================================
    # CLOSE
    # ==========================================================

    train_env.close()
    eval_env.close()

    # ==========================================================
    # FINISH
    # ==========================================================

    print()
    print("=" * 80)
    print("V12 TRAINING COMPLETED")
    print("=" * 80)

    print()
    print(
        "Final model:"
    )

    print(
        final_model_path
    )

    print()

    print(
        "Best model:"
    )

    print(
        f"{MODEL_DIR}/best/best_model"
    )


if __name__ == "__main__":
    main()
