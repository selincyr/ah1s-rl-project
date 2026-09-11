import os

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_stage1_4action import (
    HelicopterEnvStage1FourAction
)


# ============================================================
# PATHS
# ============================================================

SOURCE_MODEL = (
    "models_v2/"
    "AH1S_STAGE1_SUCCESS.zip"
)

OUTPUT_DIR = (
    "models_stage1_4action"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ENVIRONMENT
# ============================================================

train_env = Monitor(
    HelicopterEnvStage1FourAction(
        teacher_model_path=SOURCE_MODEL,
        use_teacher_reward=True
    )
)


# ============================================================
# SOURCE STAGE 1
# ============================================================

print("=" * 110)
print("LOADING ORIGINAL STAGE 1 PPO")
print("=" * 110)

source_model = PPO.load(
    SOURCE_MODEL
)

print("✅ Original Stage 1 loaded")


# ============================================================
# NEW 4-ACTION PPO
# ============================================================

model = PPO(
    "MlpPolicy",

    train_env,

    learning_rate=5e-5,

    n_steps=1024,

    batch_size=64,

    n_epochs=10,

    gamma=0.995,

    gae_lambda=0.95,

    clip_range=0.15,

    ent_coef=0.001,

    vf_coef=0.5,

    max_grad_norm=0.5,

    target_kl=0.015,

    policy_kwargs=dict(
        net_arch=[
            128,
            128
        ],

        log_std_init=-1.5
    ),

    verbose=1,

    tensorboard_log=(
        "logs_stage1_4action/"
        "tensorboard/"
    ),
)


# ============================================================
# SELECTIVE WEIGHT TRANSFER
# ============================================================

print("\n")
print("=" * 110)
print("SELECTIVE WEIGHT TRANSFER")
print("=" * 110)


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


exact_copied = 0
expanded_copied = 0


for key in target_state.keys():

    if key not in source_state:

        continue


    source_tensor = (
        source_state[key]
    )

    target_tensor = (
        target_state[key]
    )


    # ========================================================
    # ACTION NET WEIGHT
    #
    # ONLY ROW 0 = COLLECTIVE
    # ========================================================

    if key == "action_net.weight":

        new_tensor = (
            target_tensor.clone()
        )

        # Collective output
        new_tensor[
            0,
            :
        ] = source_tensor[
            0,
            :
        ]

        # Elevator / aileron / rudder:
        # start neutral.
        new_tensor[
            1:,
            :
        ] = 0.0

        target_state[key] = (
            new_tensor
        )

        print(
            "✅ action_net.weight:"
            " collective copied,"
            " cyclic/pedal zeroed"
        )

        continue


    # ========================================================
    # ACTION NET BIAS
    # ========================================================

    if key == "action_net.bias":

        new_tensor = (
            target_tensor.clone()
        )

        new_tensor[0] = (
            source_tensor[0]
        )

        new_tensor[1:] = 0.0

        target_state[key] = (
            new_tensor
        )

        print(
            "✅ action_net.bias:"
            " collective copied,"
            " cyclic/pedal zeroed"
        )

        continue


    # ========================================================
    # LOG STD
    #
    # DO NOT COPY OLD UNUSED ACTION STD VALUES HERE.
    # ========================================================

    if key == "log_std":

        new_tensor = (
            target_tensor.clone()
        )

        # Preserve old collective exploration
        new_tensor[0] = (
            source_tensor[0]
        )

        # New control axes:
        # relatively small exploration
        new_tensor[1] = -1.5
        new_tensor[2] = -1.5
        new_tensor[3] = -1.8

        target_state[key] = (
            new_tensor
        )

        print(
            "✅ log_std:"
            " collective preserved,"
            " new axes initialized safely"
        )

        continue


    # ========================================================
    # EXACT SHAPE COPY
    # ========================================================

    if (
        source_tensor.shape
        ==
        target_tensor.shape
    ):

        target_state[key] = (
            source_tensor.clone()
        )

        exact_copied += 1

        continue


    # ========================================================
    # FIRST LAYERS
    #
    # old input = 12
    # new input = 18
    #
    # Copy first 12 columns.
    #
    # New:
    # N/E/VN/VE/heading/yaw
    #
    # start at zero influence.
    # ========================================================

    if (
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


        new_tensor[
            :,
            old_inputs:
        ] = 0.0


        target_state[key] = (
            new_tensor
        )

        expanded_copied += 1


# ============================================================
# LOAD TRANSFERRED PARAMETERS
# ============================================================

model.policy.load_state_dict(
    target_state
)


print(
    "Exact tensors copied:",
    exact_copied
)

print(
    "Expanded layers copied:",
    expanded_copied
)

print(
    "✅ Selective transfer complete"
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    label
):

    # Evaluation teacher reward kapalı.
    #
    # Model gerçekten tek başına uçuyor.
    env = (
        HelicopterEnvStage1FourAction(
            teacher_model_path=None,
            use_teacher_reward=False
        )
    )


    obs, info = env.reset()

    total_reward = 0.0

    next_print = 0.0


    for step in range(
        env.max_steps
    ):

        action, _ = model.predict(
            obs,
            deterministic=True
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
            env.dt
        )


        if t >= next_print:

            heading_deg = (
                info[
                    "heading_error"
                ]
                *
                180.0
                /
                np.pi
            )


            print(
                f"{label} | "
                f"t={t:6.1f}s | "
                f"ALT={info['altitude']:7.2f} | "
                f"DRIFT={info['drift']:6.2f} | "
                f"MAX={info['max_drift']:6.2f} | "
                f"PATH={info['path']:6.2f} | "
                f"N={info['north']:6.2f} | "
                f"E={info['east']:6.2f} | "
                f"HDG={heading_deg:6.2f}° | "
                f"COL={info['collective']:.4f} | "
                f"ELE={info['elevator']:.5f} | "
                f"AIL={info['aileron']:.5f} | "
                f"RUD={info['rudder']:.5f}"
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

        "altitude_error":
            info["altitude_error"],

        "vertical_speed":
            info["vertical_speed"],

        "drift":
            info["drift"],

        "max_drift":
            info["max_drift"],

        "path":
            info["path"],

        "north":
            info["north"],

        "east":
            info["east"],

        "heading_error_deg":
            (
                info[
                    "heading_error"
                ]
                *
                180.0
                /
                np.pi
            ),

        "reward":
            total_reward,
    }


    env.close()

    return result


# ============================================================
# BASELINE
# ============================================================

print("\n")
print("=" * 110)
print("4-ACTION TRANSFER BASELINE")
print("=" * 110)


baseline = evaluate(
    model,
    "BASE"
)


print("\n")
print("BASELINE RESULT")
print("-" * 60)

print(
    "SUCCESS   :",
    baseline["success"]
)

print(
    "ALT       :",
    round(
        baseline["altitude"],
        2
    ),
    "ft"
)

print(
    "MAX DRIFT :",
    round(
        baseline["max_drift"],
        2
    ),
    "ft"
)

print(
    "FINAL     :",
    round(
        baseline["drift"],
        2
    ),
    "ft"
)

print(
    "PATH      :",
    round(
        baseline["path"],
        2
    ),
    "ft"
)

print(
    "HEADING   :",
    round(
        baseline[
            "heading_error_deg"
        ],
        2
    ),
    "deg"
)


# ============================================================
# TRAINING
# ============================================================

CHUNK = 5_120

TOTAL_LIMIT = 153_600

trained = 0

success_found = False

best_score = float(
    "inf"
)


while trained < TOTAL_LIMIT:

    print("\n")
    print("=" * 110)

    print(
        f"TRAINING "
        f"{trained:,} -> "
        f"{trained + CHUNK:,}"
    )

    print("=" * 110)


    model.learn(
        total_timesteps=CHUNK,

        reset_num_timesteps=False,

        progress_bar=True
    )


    trained += CHUNK


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_path = (
        f"{OUTPUT_DIR}/"
        f"stage1_4action_"
        f"{trained}_steps"
    )


    model.save(
        checkpoint_path
    )


    # ========================================================
    # EVALUATE
    # ========================================================

    print("\n")
    print("=" * 110)

    print(
        "4-ACTION STRAIGHT TAKEOFF "
        f"EVALUATION @ {trained:,} STEPS"
    )

    print("=" * 110)


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
        "ALT ERROR :",
        round(
            result["altitude_error"],
            2
        ),
        "ft"
    )

    print(
        "VS        :",
        round(
            result["vertical_speed"],
            2
        ),
        "ft/s"
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

    print(
        "HEADING   :",
        round(
            result[
                "heading_error_deg"
            ],
            2
        ),
        "deg"
    )

    print(
        "REWARD    :",
        round(
            result["reward"],
            2
        )
    )


    # ========================================================
    # BEST MODEL SCORE
    #
    # Do NOT reward a model that simply stays on ground.
    # ========================================================

    altitude_shortfall = max(
        0.0,
        290.0
        -
        result["altitude"]
    )


    score = (

        10.0
        *
        result["max_drift"]

        +

        1.0
        *
        result["path"]

        +

        2.0
        *
        result["drift"]

        +

        5.0
        *
        altitude_shortfall

        +

        1.0
        *
        abs(
            result[
                "heading_error_deg"
            ]
        )
    )


    if score < best_score:

        best_score = score

        best_path = (
            f"{OUTPUT_DIR}/"
            "AH1S_STAGE1_4ACTION_BEST"
        )

        model.save(
            best_path
        )

        print(
            "✅ New best 4-action candidate saved."
        )


    # ========================================================
    # STRICT SUCCESS
    # ========================================================

    if result["success"]:

        success_path = (
            f"{OUTPUT_DIR}/"
            "AH1S_STAGE1_4ACTION_SUCCESS"
        )

        model.save(
            success_path
        )

        success_found = True


        print("\n")

        print(
            "🏆 STRICT 4-ACTION "
            "STRAIGHT TAKEOFF SUCCESS"
        )

        print(
            "Model:"
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
# FINAL SAVE
# ============================================================

model.save(
    f"{OUTPUT_DIR}/"
    "AH1S_STAGE1_4ACTION_FINAL"
)


print("\n")
print("=" * 110)
print("STAGE 1 4-ACTION TRAINING FINISHED")
print("=" * 110)


if success_found:

    print(
        "✅ SUCCESS MODEL:"
    )

    print(
        f"{OUTPUT_DIR}/"
        "AH1S_STAGE1_4ACTION_SUCCESS.zip"
    )

else:

    print(
        "⚠️ Strict straight-takeoff "
        "success not found."
    )

    print(
        "Best candidate:"
    )

    print(
        f"{OUTPUT_DIR}/"
        "AH1S_STAGE1_4ACTION_BEST.zip"
    )


print("=" * 110)
