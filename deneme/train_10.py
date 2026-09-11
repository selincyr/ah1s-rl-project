import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback
)

from helicopter_env_10 import HelicopterEnv


# ==============================================================
# PATHS
# ==============================================================

V9_BEST_MODEL = (
    "models/task9_hover300/best/best_model"
)

MODEL_DIR = (
    "models/task10_hover300"
)

LOG_DIR = (
    "logs/task10_hover300"
)


# ==============================================================
# TRAINING SETTINGS
# ==============================================================

TOTAL_TIMESTEPS = 150_000

LEARNING_RATE = 1e-4


def main():

    # ==========================================================
    # CREATE DIRECTORIES
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
    # LOAD V9 BEST
    # ==========================================================
    #
    # V10 sıfırdan başlamıyor.
    #
    # V9 BEST zaten:
    #
    # - yerden kalkabiliyor
    # - 200 ft
    # - 230 ft
    # - 260 ft
    # - 280 ft
    # - 295 ft
    # - 300 ft
    #
    # seviyelerini geçebiliyor.
    #
    # V10 bu policy'yi level-off ve hover için geliştirecek.
    # ==========================================================

    print()
    print("=" * 80)
    print("LOADING V9 BEST MODEL")
    print("=" * 80)

    print(
        "Model:",
        V9_BEST_MODEL
    )

    model = PPO.load(
        V9_BEST_MODEL,
        env=train_env,
        tensorboard_log=LOG_DIR
    )

    # ==========================================================
    # LOWER LEARNING RATE FOR FINE-TUNING
    # ==========================================================
    #
    # V9'daki davranışı tamamen bozmak istemiyoruz.
    # Bu yüzden 3e-4 yerine 1e-4 kullanıyoruz.
    # ==========================================================

    model.learning_rate = (
        LEARNING_RATE
    )

    model.lr_schedule = (
        lambda progress_remaining:
        LEARNING_RATE
    )

    # Optimizer'ın mevcut learning rate'ini de güncelle.
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
                "ppo_ah1s_hover300_v10"
            )
        )
    )

    # ==========================================================
    # EVALUATION CALLBACK
    # ==========================================================
    #
    # Fine-tune olduğumuz için daha sık kontrol ediyoruz.
    #
    # Her 5k step:
    # deterministic evaluation
    #
    # En iyi model:
    #
    # models/task10_hover300/best/best_model.zip
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
    print("V10 FINE-TUNING STARTED")
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
        "ppo_ah1s_hover300_v10_final"
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
    # FINISH INFO
    # ==========================================================

    print()
    print("=" * 80)
    print("V10 TRAINING COMPLETED")
    print("=" * 80)

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
