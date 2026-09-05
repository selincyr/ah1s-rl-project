
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)

from helicopter_env_stage2_refine import (
    HelicopterEnvStage2Refine
)


# ============================================================
# PATHS
# ============================================================

START_MODEL = Path(
    "models_stage2/"
    "AH1S_STAGE2_NEAR_SUCCESS_40K.zip"
)

MODEL_DIR = Path(
    "models_stage2_refine"
)

LOG_DIR = Path(
    "logs_stage2_refine"
)

MODEL_DIR.mkdir(
    exist_ok=True
)

LOG_DIR.mkdir(
    exist_ok=True
)

SUCCESS_MODEL = (
    MODEL_DIR /
    "AH1S_STAGE2_REFINE_SUCCESS"
)

BEST_CANDIDATE = (
    MODEL_DIR /
    "AH1S_STAGE2_REFINE_BEST_CANDIDATE"
)


# ============================================================
# CHECK START MODEL
# ============================================================

if not START_MODEL.exists():

    raise FileNotFoundError(
        f"40K model bulunamadı: {START_MODEL}"
    )

print(
    "Başlangıç modeli:",
    START_MODEL
)


# ============================================================
# ENVIRONMENTS
# ============================================================

train_env = Monitor(
    HelicopterEnvStage2Refine(),
    filename=str(
        LOG_DIR /
        "train_monitor.csv"
    )
)

eval_env = (
    HelicopterEnvStage2Refine()
)


# ============================================================
# CUSTOM EVALUATION CALLBACK
# ============================================================

class RefineEvalCallback(
    BaseCallback
):

    def __init__(
        self,
        eval_env,
        eval_freq=5000,
        verbose=1
    ):

        super().__init__(verbose)

        self.eval_env = eval_env

        self.eval_freq = (
            eval_freq
        )

        self.best_candidate_score = (
            float("inf")
        )

        self.success_found = False


    def _on_step(self):

        if (
            self.num_timesteps
            %
            self.eval_freq
            != 0
        ):
            return True


        print("\n")
        print("=" * 90)

        print(
            f"STAGE 2 REFINE EVALUATION "
            f"@ {self.num_timesteps} STEPS"
        )

        print("=" * 90)


        obs, info = (
            self.eval_env.reset()
        )

        total_reward = 0.0

        crossing = None

        max_alt = (
            info["altitude"]
        )

        min_alt = (
            info["altitude"]
        )


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
                info
            ) = self.eval_env.step(
                action
            )

            total_reward += (
                reward
            )

            max_alt = max(
                max_alt,
                info["altitude"]
            )

            min_alt = min(
                min_alt,
                info["altitude"]
            )

            # İlk kez 300 ft mesafeye
            # ulaştığı andaki durumu kaydet.
            if (
                crossing is None
                and
                info[
                    "forward_distance"
                ] >= 300.0
            ):

                crossing = {

                    "altitude":
                        info[
                            "altitude"
                        ],

                    "altitude_error":
                        info[
                            "altitude_error"
                        ],

                    "vertical_speed":
                        info[
                            "vertical_speed"
                        ],

                    "forward_velocity":
                        info[
                            "forward_velocity"
                        ],

                    "lateral_velocity":
                        info[
                            "lateral_velocity"
                        ],

                    "collective":
                        info[
                            "collective"
                        ],

                    "elevator":
                        info[
                            "elevator"
                        ],

                    "time":
                        (
                            (step + 1)
                            *
                            self.eval_env.CONTROL_DT
                        ),
                }


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


        print(
            f"SUCCESS : "
            f"{info['success']}"
        )

        print(
            f"TIME    : "
            f"{flight_time:.2f} s"
        )

        print(
            f"DIST    : "
            f"{info['forward_distance']:.2f} ft"
        )

        print(
            f"ALT     : "
            f"{info['altitude']:.2f} ft"
        )

        print(
            f"ALT ERR : "
            f"{info['altitude_error']:.2f} ft"
        )

        print(
            f"VS      : "
            f"{info['vertical_speed']:.2f} ft/s"
        )

        print(
            f"FWD     : "
            f"{info['forward_velocity']:.2f} ft/s"
        )

        print(
            f"LAT     : "
            f"{info['lateral_velocity']:.2f} ft/s"
        )

        print(
            f"COL     : "
            f"{info['collective']:.4f}"
        )

        print(
            f"ELE     : "
            f"{info['elevator']:.4f}"
        )

        print(
            f"MIN ALT : "
            f"{min_alt:.2f}"
        )

        print(
            f"MAX ALT : "
            f"{max_alt:.2f}"
        )

        print(
            f"REWARD  : "
            f"{total_reward:.2f}"
        )


        # ====================================================
        # 300 FT CROSSING INFO
        # ====================================================

        if crossing is not None:

            print("\n--- 300 FT DISTANCE CROSSING ---")

            print(
                f"Time    : "
                f"{crossing['time']:.2f}s"
            )

            print(
                f"Altitude: "
                f"{crossing['altitude']:.2f} ft"
            )

            print(
                f"Alt err : "
                f"{crossing['altitude_error']:.2f} ft"
            )

            print(
                f"VS      : "
                f"{crossing['vertical_speed']:.2f}"
            )

            print(
                f"FWD     : "
                f"{crossing['forward_velocity']:.2f}"
            )

            print(
                f"COL     : "
                f"{crossing['collective']:.4f}"
            )

            print(
                f"ELE     : "
                f"{crossing['elevator']:.4f}"
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        if info["success"]:

            self.model.save(
                str(
                    SUCCESS_MODEL
                )
            )

            self.success_found = True

            print("\n")
            print(
                "🏆 STAGE 2 SUCCESS FOUND"
            )

            print(
                "Model saved:"
            )

            print(
                str(
                    SUCCESS_MODEL
                )
                + ".zip"
            )

            print(
                "\nTraining otomatik "
                "durduruluyor."
            )

            print("=" * 90)

            # Eğitim burada durur.
            return False


        # ====================================================
        # BEST NON-SUCCESS CANDIDATE
        # ====================================================

        if crossing is not None:

            # 300 ft ileri mesafeye ulaştığı
            # andaki görev hatasını ölç.

            candidate_score = (

                crossing[
                    "altitude_error"
                ]

                +

                2.0
                *
                abs(
                    crossing[
                        "vertical_speed"
                    ]
                )

                +

                0.25
                *
                abs(
                    crossing[
                        "lateral_velocity"
                    ]
                )
            )

        else:

            # Henüz 300 ft gidemiyorsa
            # mesafe açığını cezalandır.

            candidate_score = (

                1000.0

                +

                max(
                    0.0,
                    300.0
                    -
                    info[
                        "forward_distance"
                    ]
                )
            )


        print(
            "\nCandidate score:",
            candidate_score
        )


        if (
            candidate_score
            <
            self.best_candidate_score
        ):

            self.best_candidate_score = (
                candidate_score
            )

            self.model.save(
                str(
                    BEST_CANDIDATE
                )
            )

            print(
                "✅ New best refine "
                "candidate saved."
            )


        print("=" * 90)

        return True


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_callback = (
    CheckpointCallback(

        save_freq=5000,

        save_path=str(
            MODEL_DIR
        ),

        name_prefix=(
            "stage2_refine_checkpoint"
        ),
    )
)


eval_callback = (
    RefineEvalCallback(

        eval_env=eval_env,

        eval_freq=5000,

        verbose=1,
    )
)


callbacks = CallbackList(
    [
        checkpoint_callback,
        eval_callback,
    ]
)


# ============================================================
# LOAD 40K MODEL
# ============================================================

print("\n" + "=" * 90)

print(
    "LOADING STAGE 2 "
    "NEAR-SUCCESS 40K MODEL"
)

print("=" * 90)


model = PPO.load(

    str(
        START_MODEL
    ),

    env=train_env,

    device="auto",
)


# ============================================================
# FINE-TUNE SETTINGS
# ============================================================

# Çok düşük LR:
# çalışan forward-flight davranışını
# bozmak istemiyoruz.

NEW_LR = 3e-5

model.learning_rate = (
    NEW_LR
)

model.lr_schedule = (
    lambda progress_remaining:
        NEW_LR
)


# Artık keşiften çok
# hassas kontrol istiyoruz.

model.ent_coef = 0.0

# Büyük politika sıçramalarını engelle.
model.target_kl = 0.015


# TensorBoard'u ayrı klasöre yaz.

model.tensorboard_log = str(
    LOG_DIR /
    "tensorboard"
)


print(
    "Learning rate :",
    NEW_LR
)

print(
    "Entropy coef  :",
    model.ent_coef
)

print(
    "Target KL     :",
    model.target_kl
)


# ============================================================
# TRAIN
# ============================================================

TOTAL_TIMESTEPS = 60_000


print("\n" + "=" * 90)

print(
    "AH-1S STAGE 2 "
    "ALTITUDE-HOLD REFINEMENT"
)

print("=" * 90)

print(
    "Starting state : "
    "~300 ft stable hover"
)

print(
    "Forward target : "
    "300 ft"
)

print(
    "Altitude target: "
    "300 ft"
)

print(
    "Timesteps      :",
    TOTAL_TIMESTEPS
)

print(
    "Evaluation     : "
    "every 5,000 steps"
)

print(
    "Auto stop      : "
    "first SUCCESS=True"
)

print("=" * 90 + "\n")


model.learn(

    total_timesteps=TOTAL_TIMESTEPS,

    callback=callbacks,

    progress_bar=True,

    reset_num_timesteps=True,

    tb_log_name=(
        "AH1S_STAGE2_REFINE"
    ),
)


# ============================================================
# SAVE END STATE
# ============================================================

FINAL_MODEL = (

    MODEL_DIR /

    "AH1S_STAGE2_REFINE_FINAL"
)


model.save(
    str(
        FINAL_MODEL
    )
)


print("\n" + "=" * 90)

print(
    "REFINEMENT FINISHED"
)

print("=" * 90)


if eval_callback.success_found:

    print(
        "✅ SUCCESS MODEL:"
    )

    print(
        str(
            SUCCESS_MODEL
        )
        + ".zip"
    )

else:

    print(
        "⚠️ Henüz tam success yok."
    )

    print(
        "Best candidate:"
    )

    print(
        str(
            BEST_CANDIDATE
        )
        + ".zip"
    )


print(
    "\nFinal model:"
)

print(
    str(
        FINAL_MODEL
    )
    + ".zip"
)

print("=" * 90)


train_env.close()
eval_env.close()
