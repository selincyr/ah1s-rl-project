from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)

from helicopter_env_stage2 import HelicopterEnvStage2


# ============================================================
# PATHS
# ============================================================

STAGE1_MODEL = Path(
    "models_v2/AH1S_STAGE1_SUCCESS.zip"
)

MODEL_DIR = Path("models_stage2")
LOG_DIR = Path("logs_stage2")

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

BEST_SUCCESS_MODEL = (
    MODEL_DIR / "AH1S_STAGE2_BEST_SUCCESS"
)


# ============================================================
# CHECK STAGE 1 MODEL
# ============================================================

if not STAGE1_MODEL.exists():
    raise FileNotFoundError(
        f"Stage 1 model bulunamadı: {STAGE1_MODEL}"
    )

print(
    "Stage 1 model bulundu:",
    STAGE1_MODEL
)


# ============================================================
# ENVIRONMENTS
# ============================================================

print("\nTraining environment oluşturuluyor...")

train_env = Monitor(
    HelicopterEnvStage2(),
    filename=str(
        LOG_DIR / "train_monitor.csv"
    )
)

print(
    "Evaluation environment oluşturuluyor..."
)

eval_env = HelicopterEnvStage2()


# ============================================================
# SUCCESS-BASED EVALUATION CALLBACK
# ============================================================

class Stage2SuccessEvalCallback(BaseCallback):

    def __init__(
        self,
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=3,
        verbose=1,
    ):

        super().__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.best_success_count = -1
        self.best_mean_success_time = float("inf")
        self.best_mean_reward = -float("inf")

    def _on_step(self):

        if (
            self.num_timesteps
            %
            self.eval_freq
            != 0
        ):
            return True

        print("\n")
        print("=" * 75)
        print(
            f"STAGE 2 EVALUATION "
            f"@ {self.num_timesteps} STEPS"
        )
        print("=" * 75)

        success_count = 0

        rewards = []
        times = []
        success_times = []
        final_distances = []

        for episode in range(
            self.n_eval_episodes
        ):

            obs, info = (
                self.eval_env.reset()
            )

            total_reward = 0.0

            for step in range(
                self.eval_env.max_steps
            ):

                action, _ = (
                    self.model.predict(
                        obs,
                        deterministic=True
                    )
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(
                    action
                )

                total_reward += reward

                if (
                    terminated
                    or
                    truncated
                ):
                    break

            flight_time = (
                (step + 1)
                *
                self.eval_env.CONTROL_DT
            )

            success = bool(
                info["success"]
            )

            if success:
                success_count += 1
                success_times.append(
                    flight_time
                )

            rewards.append(
                total_reward
            )

            times.append(
                flight_time
            )

            final_distances.append(
                info[
                    "forward_distance"
                ]
            )

            print(
                f"Episode {episode + 1}: "
                f"SUCCESS={success} | "
                f"TIME={flight_time:.2f}s | "
                f"PHASE={info['phase']} | "
                f"DIST={info['forward_distance']:.2f} ft | "
                f"ALT={info['altitude']:.2f} ft | "
                f"VS={info['vertical_speed']:.2f} ft/s | "
                f"FWD={info['forward_velocity']:.2f} ft/s | "
                f"ELE={info['elevator']:.4f}"
            )

        mean_reward = float(
            np.mean(rewards)
        )

        mean_distance = float(
            np.mean(final_distances)
        )

        if success_times:

            mean_success_time = float(
                np.mean(
                    success_times
                )
            )

        else:

            mean_success_time = (
                float("inf")
            )

        print("\nSUMMARY")
        print(
            "Success:",
            f"{success_count}/"
            f"{self.n_eval_episodes}"
        )

        print(
            "Mean reward:",
            mean_reward
        )

        print(
            "Mean distance:",
            mean_distance
        )

        if success_times:

            print(
                "Mean success time:",
                mean_success_time
            )

        # ----------------------------------------------------
        # MODEL SELECTION
        # ----------------------------------------------------
        #
        # Öncelik:
        # 1) Daha fazla gerçek success
        # 2) Aynı success sayısında daha hızlı görev
        # 3) Henüz success yoksa reward
        #
        # Böylece Stage 1'de yaşadığımız
        # "yüksek reward ama fail" problemi tekrarlanmaz.
        # ----------------------------------------------------

        better = False

        if (
            success_count
            >
            self.best_success_count
        ):

            better = True

        elif (
            success_count
            ==
            self.best_success_count
            and
            success_count > 0
            and
            mean_success_time
            <
            self.best_mean_success_time
        ):

            better = True

        elif (
            success_count == 0
            and
            self.best_success_count == 0
            and
            mean_reward
            >
            self.best_mean_reward
        ):

            better = True

        if better:

            self.best_success_count = (
                success_count
            )

            self.best_mean_success_time = (
                mean_success_time
            )

            self.best_mean_reward = (
                mean_reward
            )

            self.model.save(
                str(
                    BEST_SUCCESS_MODEL
                )
            )

            print(
                "\n✅ NEW BEST STAGE 2 MODEL SAVED"
            )

            print(
                str(
                    BEST_SUCCESS_MODEL
                )
                +
                ".zip"
            )

        print("=" * 75)

        return True


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_callback = (
    CheckpointCallback(
        save_freq=10_000,
        save_path=str(
            MODEL_DIR
        ),
        name_prefix=(
            "ppo_ah1s_stage2_checkpoint"
        ),
    )
)

success_eval_callback = (
    Stage2SuccessEvalCallback(
        eval_env=eval_env,
        eval_freq=10_000,
        n_eval_episodes=3,
        verbose=1,
    )
)

callbacks = CallbackList(
    [
        checkpoint_callback,
        success_eval_callback,
    ]
)


# ============================================================
# LOAD STAGE 1 MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING STAGE 1 SUCCESS MODEL")
print("=" * 75)

model = PPO.load(
    str(STAGE1_MODEL),
    env=train_env,
    device="auto",
)

print(
    "Stage 1 weights loaded."
)

# ------------------------------------------------------------
# Fine tuning:
# Stage 1'deki politikayı hızlı bozmayalım.
# ------------------------------------------------------------

NEW_LEARNING_RATE = 1e-4

model.learning_rate = (
    NEW_LEARNING_RATE
)

model.lr_schedule = (
    lambda progress_remaining:
        NEW_LEARNING_RATE
)

print(
    "Fine-tune learning rate:",
    NEW_LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

TOTAL_TIMESTEPS = 150_000

print("\n" + "=" * 75)
print("AH-1S PPO STAGE 2 TRAINING")
print("=" * 75)

print(
    "Stage 1:"
)

print(
    "Motor -> Takeoff -> 300 ft -> Hover"
)

print(
    "\nStage 2:"
)

print(
    "300 ft korunurken -> Forward Flight"
)

print(
    "Target forward distance:",
    HelicopterEnvStage2.TARGET_FORWARD_DISTANCE,
    "ft"
)

print(
    "\nTraining timesteps:",
    TOTAL_TIMESTEPS
)

print("=" * 75 + "\n")


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    progress_bar=True,
    reset_num_timesteps=True,
    tb_log_name="AH1S_STAGE2",
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL = (
    MODEL_DIR /
    "AH1S_STAGE2_FINAL"
)

model.save(
    str(FINAL_MODEL)
)

print("\n" + "=" * 75)
print("STAGE 2 TRAINING COMPLETE")
print("=" * 75)

print(
    "Final model:",
    str(FINAL_MODEL)
    +
    ".zip"
)

print(
    "Best success-based model:",
    str(BEST_SUCCESS_MODEL)
    +
    ".zip"
)

print("=" * 75)


# ============================================================
# CLEANUP
# ============================================================

train_env.close()
eval_env.close()
