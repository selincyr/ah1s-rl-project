import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_stage1_straight import (
    HelicopterEnvStage1Straight
)


SOURCE_MODEL = (
    "models_v2/"
    "AH1S_STAGE1_SUCCESS.zip"
)

OUTPUT_DIR = (
    "models_stage1_straight"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ENV
# ============================================================

train_env = Monitor(
    HelicopterEnvStage1Straight()
)


# ============================================================
# OLD MODEL
# ============================================================

print("=" * 100)
print("LOADING ORIGINAL STAGE 1 MODEL")
print("=" * 100)

source_model = PPO.load(
    SOURCE_MODEL
)

print("✅ Source model loaded")


# ============================================================
# NEW 14-OBS MODEL
# ============================================================

model = PPO(
    "MlpPolicy",

    train_env,

    learning_rate=3e-5,

    n_steps=2048,

    batch_size=64,

    n_epochs=10,

    gamma=0.995,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.0,

    vf_coef=0.5,

    max_grad_norm=0.5,

    target_kl=0.015,

    policy_kwargs=dict(
        net_arch=[
            128,
            128
        ]
    ),

    verbose=1,

    tensorboard_log=(
        "logs_stage1_straight/"
        "tensorboard/"
    ),
)


# ============================================================
# TRANSFER WEIGHTS
# ============================================================

print("\n")
print("=" * 100)
print("TRANSFERRING STAGE 1 WEIGHTS")
print("=" * 100)


source_state = (
    source_model
    .policy
    .state_dict()
)

target_state = (
    model
    .policy
    .state_dict()
)


copied_exact = 0
copied_expanded = 0


for key, source_tensor in (
    source_state.items()
):

    if key not in target_state:
        continue

    target_tensor = (
        target_state[key]
    )


    # Exact same shape
    if (
        source_tensor.shape
        ==
        target_tensor.shape
    ):

        target_state[key] = (
            source_tensor.clone()
        )

        copied_exact += 1


    # First layer:
    # 12 input -> 14 input
    elif (
        source_tensor.ndim == 2
        and
        target_tensor.ndim == 2
        and
        source_tensor.shape[0]
        ==
        target_tensor.shape[0]
        and
        source_tensor.shape[1]
        <
        target_tensor.shape[1]
    ):

        new_tensor = (
            target_tensor.clone()
        )

        old_inputs = (
            source_tensor.shape[1]
        )

        new_tensor[
            :,
            :old_inputs
        ] = source_tensor

        # Yeni position girişleri başlangıçta
        # eski politikayı bozmasın
        new_tensor[
            :,
            old_inputs:
        ] = 0.0

        target_state[key] = (
            new_tensor
        )

        copied_expanded += 1


model.policy.load_state_dict(
    target_state
)


print(
    "Exact tensors copied:",
    copied_exact
)

print(
    "Expanded input layers:",
    copied_expanded
)

print(
    "✅ Transfer complete"
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    label
):

    env = (
        HelicopterEnvStage1Straight()
    )

    obs, info = env.reset()

    total_reward = 0.0

    next_print = 0.0


    for step in range(
        env.max_steps_straight
    ):

        action, _ = (
            model.predict(
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
        ) = env.step(
            action
        )

        total_reward += reward

        t = (
            (step + 1)
            *
            env.control_dt
        )


        if t >= next_print:

            print(
                f"{label} | "
                f"t={t:6.1f}s | "
                f"ALT={info['altitude']:7.2f} | "
                f"DRIFT={info['horizontal_drift']:6.2f} | "
                f"MAX={info['max_horizontal_drift']:6.2f} | "
                f"N={info['north_position']:7.2f} | "
                f"E={info['east_position']:7.2f} | "
                f"VS={info['vertical_speed']:6.2f}"
            )

            next_print += 10.0


        if terminated or truncated:
            break


    result = {

        "success":
            info["success"],

        "time":
            t,

        "altitude":
            info["altitude"],

        "drift":
            info[
                "horizontal_drift"
            ],

        "max_drift":
            info[
                "max_horizontal_drift"
            ],

        "path":
            info[
                "horizontal_path"
            ],

        "north":
            info[
                "north_position"
            ],

        "east":
            info[
                "east_position"
            ],

        "reward":
            total_reward,
    }


    env.close()

    return result


# ============================================================
# BEFORE TRAINING
# ============================================================

print("\n")
print("=" * 100)
print("TRANSFER MODEL BASELINE")
print("=" * 100)

baseline = evaluate(
    model,
    "BASE"
)

print(
    baseline
)


# ============================================================
# TRAIN IN CHUNKS
# ============================================================

TOTAL_LIMIT = 100_000

CHUNK = 5_000

trained = 0

success_found = False


while trained < TOTAL_LIMIT:

    print("\n")
    print("=" * 100)

    print(
        f"TRAINING "
        f"{trained:,} -> "
        f"{trained + CHUNK:,}"
    )

    print("=" * 100)


    model.learn(
        total_timesteps=CHUNK,

        reset_num_timesteps=False,

        progress_bar=True
    )


    trained += CHUNK


    checkpoint_path = (
        f"{OUTPUT_DIR}/"
        f"stage1_straight_"
        f"{trained}_steps"
    )

    model.save(
        checkpoint_path
    )


    print("\n")
    print("=" * 100)

    print(
        f"STRAIGHT TAKEOFF "
        f"EVALUATION @ "
        f"{trained} STEPS"
    )

    print("=" * 100)


    result = evaluate(
        model,
        f"{trained // 1000}K"
    )


    print("\n")
    print(
        "SUCCESS   :",
        result["success"]
    )

    print(
        "ALT       :",
        round(
            result["altitude"],
            2
        ),
        "ft"
    )

    print(
        "MAX DRIFT :",
        round(
            result["max_drift"],
            2
        ),
        "ft"
    )

    print(
        "FINAL     :",
        round(
            result["drift"],
            2
        ),
        "ft"
    )

    print(
        "PATH      :",
        round(
            result["path"],
            2
        ),
        "ft"
    )

    print(
        "NORTH     :",
        round(
            result["north"],
            2
        ),
        "ft"
    )

    print(
        "EAST      :",
        round(
            result["east"],
            2
        ),
        "ft"
    )


    if result["success"]:

        success_path = (
            f"{OUTPUT_DIR}/"
            "AH1S_STAGE1_STRAIGHT_SUCCESS"
        )

        model.save(
            success_path
        )

        success_found = True

        print("\n")
        print(
            "🏆 STRAIGHT TAKEOFF SUCCESS"
        )

        print(
            "Model saved:"
        )

        print(
            success_path
            +
            ".zip"
        )

        print(
            "\nTraining automatically stopped."
        )

        break


# ============================================================
# FINAL
# ============================================================

model.save(
    f"{OUTPUT_DIR}/"
    "AH1S_STAGE1_STRAIGHT_FINAL"
)

print("\n")
print("=" * 100)
print("STAGE 1 STRAIGHT REFINEMENT FINISHED")
print("=" * 100)

if success_found:

    print(
        "✅ SUCCESS MODEL:"
    )

    print(
        f"{OUTPUT_DIR}/"
        "AH1S_STAGE1_STRAIGHT_SUCCESS.zip"
    )

else:

    print(
        "⚠️ No strict straight-takeoff "
        "success yet."
    )

print("=" * 100)
