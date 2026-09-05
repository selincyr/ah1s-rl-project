import os
import copy
import numpy as np
import torch

from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill
)


# ============================================================
# PATHS
# ============================================================

SOURCE_MODEL = (
    "models_stage1_distill/"
    "AH1S_STAGE1_DISTILL_SUCCESS.zip"
)

OUTPUT_DIR = (
    "models_stage1_final"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TARGET
# ============================================================

TARGET_ALTITUDE = 300.0


# ============================================================
# BIAS GATE
#
# Below 275 ft:
# zero change.
#
# 275-295:
# smoothly introduce correction.
#
# Above 295:
# full correction.
# ============================================================

GATE_START = 275.0
GATE_FULL = 295.0


def altitude_gate(altitude):

    if altitude <= GATE_START:
        return 0.0

    if altitude >= GATE_FULL:
        return 1.0

    return float(
        (altitude - GATE_START)
        /
        (GATE_FULL - GATE_START)
    )


# ============================================================
# PHYSICAL COLLECTIVE BIAS CANDIDATES
# ============================================================

BIAS_CANDIDATES = [
     0.0000,
    -0.0002,
    -0.0004,
    -0.0006,
    -0.0008,
    -0.0010,
    -0.0012,
    -0.0014,
    -0.0016,
    -0.0018,
    -0.0020,
]


# ============================================================
# ORIGINAL SUCCESSFUL MODEL
# ============================================================

print("=" * 120)
print("STAGE 1 ALTITUDE HEAD REFINEMENT")
print("=" * 120)

print("\nLoading:")

print(
    SOURCE_MODEL
)


base_model = PPO.load(
    SOURCE_MODEL
)


print(
    "✅ Successful single 4-action PPO loaded"
)


# ============================================================
# CALIBRATION RUN
#
# Original model remains untouched.
# We inject candidate correction only at runtime.
# ============================================================

def test_bias(
    physical_bias,
    detailed=False
):

    env = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False
        )
    )

    obs, info = env.reset()

    dt = env.dt

    max_time = 90.0

    max_steps = int(
        max_time / dt
    )


    alt_final_window = []

    next_print = 0.0

    failed = False


    for step in range(
        max_steps
    ):

        t = (
            step
            *
            dt
        )


        # ====================================================
        # ORIGINAL SINGLE PPO
        # ====================================================

        action, _ = base_model.predict(
            obs,
            deterministic=True
        )

        action = np.asarray(
            action,
            dtype=np.float32
        ).copy()


        # ====================================================
        # CURRENT ALTITUDE
        # ====================================================

        current_altitude = float(
            env._state()[
                "altitude"
            ]
        )


        gate = altitude_gate(
            current_altitude
        )


        # ====================================================
        # collective =
        # 0.620 + 0.030 * action[0]
        #
        # therefore normalized action correction:
        #
        # physical_bias / 0.030
        # ====================================================

        action_bias = (
            physical_bias
            /
            0.030
        )


        action[0] += (
            gate
            *
            action_bias
        )


        action[0] = float(
            np.clip(
                action[0],
                -1.0,
                1.0
            )
        )


        # ====================================================
        # STEP
        # ====================================================

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )


        t = (
            (step + 1)
            *
            dt
        )


        altitude = float(
            info[
                "altitude"
            ]
        )


        if 70.0 <= t <= 85.0:

            alt_final_window.append(
                altitude
            )


        if detailed and t >= next_print:

            print(
                f"t={t:6.1f}s | "
                f"ALT={altitude:7.2f} | "
                f"VS={info['vertical_speed']:6.2f} | "
                f"DRIFT={info['drift']:5.2f} | "
                f"MAX={info['max_drift']:5.2f} | "
                f"PATH={info['path']:5.2f} | "
                f"COL={info['collective']:.5f} | "
                f"GATE={gate:.2f}"
            )

            next_print += 5.0


        # Parent success is NOT failure.
        if (
            terminated
            and
            not info[
                "success"
            ]
        ):

            failed = True
            break


        if altitude > 360.0:

            failed = True
            break


        if info["max_drift"] > 12.0:

            failed = True
            break


    env.close()


    if len(
        alt_final_window
    ) > 0:

        mean_altitude = float(
            np.mean(
                alt_final_window
            )
        )

        std_altitude = float(
            np.std(
                alt_final_window
            )
        )

    else:

        mean_altitude = altitude
        std_altitude = 999.0


    return {
        "bias":
            physical_bias,

        "failed":
            failed,

        "final_alt":
            altitude,

        "mean_alt":
            mean_altitude,

        "std_alt":
            std_altitude,

        "vs":
            float(
                info[
                    "vertical_speed"
                ]
            ),

        "max_drift":
            float(
                info[
                    "max_drift"
                ]
            ),

        "final_drift":
            float(
                info[
                    "drift"
                ]
            ),

        "path":
            float(
                info[
                    "path"
                ]
            ),
    }


# ============================================================
# AUTOMATIC BIAS CALIBRATION
# ============================================================

print("\n")
print("=" * 120)
print("AUTOMATIC COLLECTIVE BIAS CALIBRATION")
print("=" * 120)


bias_results = []


for bias in BIAS_CANDIDATES:

    r = test_bias(
        bias,
        detailed=False
    )

    bias_results.append(
        r
    )


    print(
        f"BIAS={bias:+.4f} | "
        f"MEAN_ALT={r['mean_alt']:7.2f} | "
        f"FINAL={r['final_alt']:7.2f} | "
        f"STD={r['std_alt']:5.2f} | "
        f"VS={r['vs']:6.2f} | "
        f"MAX={r['max_drift']:5.2f} | "
        f"PATH={r['path']:5.2f}"
    )


# ============================================================
# CALIBRATION SCORE
#
# Main goal:
# mean altitude = 300
#
# Preserve XY at same time.
# ============================================================

def calibration_score(r):

    if r["failed"]:

        return 1e9

    return (

        20.0
        *
        abs(
            r["mean_alt"]
            -
            TARGET_ALTITUDE
        )

        +

        5.0
        *
        r["std_alt"]

        +

        3.0
        *
        r["max_drift"]

        +

        0.5
        *
        r["path"]
    )


bias_results.sort(
    key=calibration_score
)


best_bias_result = (
    bias_results[0]
)

BEST_BIAS = float(
    best_bias_result[
        "bias"
    ]
)


print("\n")
print("=" * 120)
print("BEST CALIBRATED BIAS")
print("=" * 120)

print(
    "BIAS       :",
    BEST_BIAS
)

print(
    "MEAN ALT   :",
    round(
        best_bias_result[
            "mean_alt"
        ],
        3
    ),
    "ft"
)

print(
    "FINAL ALT  :",
    round(
        best_bias_result[
            "final_alt"
        ],
        3
    ),
    "ft"
)

print(
    "MAX DRIFT  :",
    round(
        best_bias_result[
            "max_drift"
        ],
        3
    ),
    "ft"
)

print(
    "PATH       :",
    round(
        best_bias_result[
            "path"
        ],
        3
    ),
    "ft"
)


# ============================================================
# DETAIL CALIBRATED TEACHER
# ============================================================

print("\n")
print("=" * 120)
print("DETAILED CALIBRATED TEACHER RUN")
print("=" * 120)


test_bias(
    BEST_BIAS,
    detailed=True
)


# ============================================================
# COLLECT SUPERVISED DATA
#
# teacher target:
#
# original successful 4-action PPO
#
# +
#
# ONLY gated action[0] correction
#
# actions 1,2,3 remain original.
# ============================================================

print("\n")
print("=" * 120)
print("COLLECTING ALTITUDE TEACHER DATA")
print("=" * 120)


observations = []
targets_a0 = []


# Small XY state variations.
# We are NOT teaching XY again.
# They only help collective head remain robust.
OFFSETS = [

    (0.0, 0.0),

    (1.5, 0.0),

    (-1.5, 0.0),

    (0.0, 1.5),

    (0.0, -1.5),

    (1.0, 1.0),

    (-1.0, -1.0),
]


for episode_index, (
    offset_n,
    offset_e
) in enumerate(
    OFFSETS
):

    env = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False
        )
    )


    obs, info = env.reset()


    env.north = (
        offset_n
    )

    env.east = (
        offset_e
    )

    obs = env._get_obs()


    samples = 0


    for step in range(
        env.max_steps
    ):

        # ====================================================
        # ORIGINAL PPO ACTION
        # ====================================================

        original_action, _ = (
            base_model.predict(
                obs,
                deterministic=True
            )
        )


        original_action = (
            np.asarray(
                original_action,
                dtype=np.float32
            )
            .copy()
        )


        # ====================================================
        # TARGET COLLECTIVE ACTION
        # ====================================================

        altitude = float(
            env._state()[
                "altitude"
            ]
        )


        gate = altitude_gate(
            altitude
        )


        target_action0 = float(
            original_action[0]
            +
            gate
            *
            (
                BEST_BIAS
                /
                0.030
            )
        )


        target_action0 = float(
            np.clip(
                target_action0,
                -1.0,
                1.0
            )
        )


        # ====================================================
        # STORE
        # ====================================================

        observations.append(
            obs.copy()
        )


        targets_a0.append(
            target_action0
        )


        # ====================================================
        # EXECUTE CALIBRATED TEACHER
        # ====================================================

        execution_action = (
            original_action.copy()
        )


        execution_action[0] = (
            target_action0
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            execution_action
        )


        samples += 1


        if (
            terminated
            and
            not info[
                "success"
            ]
        ):

            break


        # Enough data after hover established.
        if (
            (step + 1)
            *
            env.dt
            >=
            85.0
        ):

            break


    env.close()


    print(
        f"Episode "
        f"{episode_index + 1}: "
        f"{samples} samples"
    )


observations = np.asarray(
    observations,
    dtype=np.float32
)


targets_a0 = np.asarray(
    targets_a0,
    dtype=np.float32
)


print(
    "\nObservations:",
    observations.shape
)

print(
    "Collective targets:",
    targets_a0.shape
)


# ============================================================
# LOAD FRESH COPY OF SUCCESS MODEL
# ============================================================

refined_model = PPO.load(
    SOURCE_MODEL
)


# ============================================================
# FREEZE EVERYTHING
# ============================================================

for param in (
    refined_model
    .policy
    .parameters()
):

    param.requires_grad = False


# ============================================================
# UNFREEZE ACTION HEAD ONLY
#
# Loss uses ONLY output[0].
#
# Therefore:
#
# action_net row 0 gets gradients.
#
# rows 1/2/3 receive zero gradient.
#
# Shared network remains frozen.
# ============================================================

refined_model.policy.action_net.weight.requires_grad = True

refined_model.policy.action_net.bias.requires_grad = True


trainable_parameters = [
    refined_model
    .policy
    .action_net
    .weight,

    refined_model
    .policy
    .action_net
    .bias,
]


optimizer = torch.optim.Adam(
    trainable_parameters,
    lr=2e-4
)


device = (
    refined_model.device
)


obs_tensor = torch.as_tensor(
    observations,
    dtype=torch.float32,
    device=device
)


target_tensor = torch.as_tensor(
    targets_a0,
    dtype=torch.float32,
    device=device
)


# ============================================================
# SAVE ORIGINAL NON-COLLECTIVE ROWS
#
# Additional hard guarantee that actions 1-3
# never change.
# ============================================================

with torch.no_grad():

    frozen_rows = (
        refined_model
        .policy
        .action_net
        .weight[
            1:
        ]
        .clone()
    )

    frozen_biases = (
        refined_model
        .policy
        .action_net
        .bias[
            1:
        ]
        .clone()
    )


# ============================================================
# SUPERVISED COLLECTIVE HEAD TRAINING
# ============================================================

print("\n")
print("=" * 120)
print("TRAINING ONLY COLLECTIVE OUTPUT HEAD")
print("=" * 120)


EPOCHS = 150
BATCH_SIZE = 128


n = (
    observations.shape[0]
)


best_loss = float(
    "inf"
)


best_actor_state = None


for epoch in range(
    EPOCHS
):

    permutation = (
        np.random.permutation(
            n
        )
    )


    total_loss = 0.0

    batches = 0


    for start in range(
        0,
        n,
        BATCH_SIZE
    ):

        idx = (
            permutation[
                start:
                start
                +
                BATCH_SIZE
            ]
        )


        batch_obs = (
            obs_tensor[
                idx
            ]
        )


        batch_target = (
            target_tensor[
                idx
            ]
        )


        distribution = (
            refined_model
            .policy
            .get_distribution(
                batch_obs
            )
        )


        predicted_mean = (
            distribution
            .distribution
            .mean
        )


        predicted_a0 = (
            predicted_mean[
                :,
                0
            ]
        )


        loss = torch.mean(
            (
                predicted_a0
                -
                batch_target
            )
            ** 2
        )


        optimizer.zero_grad()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            1.0
        )


        optimizer.step()


        # ====================================================
        # HARD-RESTORE ROWS 1-3
        # ====================================================

        with torch.no_grad():

            refined_model
            .policy
            .action_net
            .weight[
                1:
            ].copy_(
                frozen_rows
            )


            refined_model
            .policy
            .action_net
            .bias[
                1:
            ].copy_(
                frozen_biases
            )


        total_loss += float(
            loss.item()
        )

        batches += 1


    avg_loss = (
        total_loss
        /
        batches
    )


    if avg_loss < best_loss:

        best_loss = (
            avg_loss
        )


        best_actor_state = {
            "weight":
                refined_model
                .policy
                .action_net
                .weight
                .detach()
                .clone(),

            "bias":
                refined_model
                .policy
                .action_net
                .bias
                .detach()
                .clone(),
        }


    if (
        epoch == 0
        or
        (epoch + 1) % 10 == 0
    ):

        print(
            f"Epoch "
            f"{epoch + 1:3d}/"
            f"{EPOCHS} | "
            f"loss="
            f"{avg_loss:.10f}"
        )


# ============================================================
# RESTORE BEST HEAD
# ============================================================

with torch.no_grad():

    refined_model
    .policy
    .action_net
    .weight
    .copy_(
        best_actor_state[
            "weight"
        ]
    )


    refined_model
    .policy
    .action_net
    .bias
    .copy_(
        best_actor_state[
            "bias"
        ]
    )


# ============================================================
# SAVE CANDIDATE
# ============================================================

candidate_path = (
    f"{OUTPUT_DIR}/"
    "AH1S_STAGE1_300FT_CANDIDATE"
)


refined_model.save(
    candidate_path
)


print(
    "\n✅ Candidate saved:"
)

print(
    candidate_path
    +
    ".zip"
)


# ============================================================
# STRICT TEACHER-OFF EVALUATION
# ============================================================

def evaluate_refined(
    model,
    detailed=True
):

    env = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False
        )
    )


    obs, info = env.reset()


    dt = env.dt


    max_time = 100.0

    max_steps = int(
        max_time
        /
        dt
    )


    stable_steps = 0


    required_steps = int(
        10.0
        /
        dt
    )


    strict_success = False


    next_print = 0.0


    for step in range(
        max_steps
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


        t = (
            (step + 1)
            *
            dt
        )


        altitude = float(
            info[
                "altitude"
            ]
        )


        altitude_error = abs(
            TARGET_ALTITUDE
            -
            altitude
        )


        vertical_speed = float(
            info[
                "vertical_speed"
            ]
        )


        horizontal_speed = float(
            np.hypot(
                info[
                    "vn"
                ],
                info[
                    "ve"
                ]
            )
        )


        strict_stable = (

            altitude_error
            <=
            5.0

            and

            abs(
                vertical_speed
            )
            <=
            0.75

            and

            info[
                "drift"
            ]
            <=
            5.0

            and

            info[
                "max_drift"
            ]
            <=
            8.0

            and

            info[
                "path"
            ]
            <=
            25.0

            and

            horizontal_speed
            <=
            1.5
        )


        if strict_stable:

            stable_steps += 1

        else:

            stable_steps = 0


        if (
            stable_steps
            >=
            required_steps
        ):

            strict_success = True


        if (
            detailed
            and
            t >= next_print
        ):

            print(
                f"t={t:6.1f}s | "
                f"ALT={altitude:7.2f} | "
                f"ERR={altitude_error:5.2f} | "
                f"VS={vertical_speed:6.2f} | "
                f"DRIFT={info['drift']:5.2f} | "
                f"MAX={info['max_drift']:5.2f} | "
                f"PATH={info['path']:5.2f} | "
                f"N={info['north']:6.2f} | "
                f"E={info['east']:6.2f} | "
                f"COL={info['collective']:.5f}"
            )

            next_print += 5.0


        if strict_success:

            break


        if (
            terminated
            and
            not info[
                "success"
            ]
        ):

            break


        if altitude > 350.0:

            break


        if info[
            "max_drift"
        ] > 12.0:

            break


    result = {

        "success":
            strict_success,

        "time":
            t,

        "altitude":
            altitude,

        "error":
            altitude_error,

        "vs":
            vertical_speed,

        "max_drift":
            float(
                info[
                    "max_drift"
                ]
            ),

        "final_drift":
            float(
                info[
                    "drift"
                ]
            ),

        "path":
            float(
                info[
                    "path"
                ]
            ),

        "north":
            float(
                info[
                    "north"
                ]
            ),

        "east":
            float(
                info[
                    "east"
                ]
            ),
    }


    env.close()

    return result


# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("=" * 120)
print("FINAL STRICT TEACHER-OFF TEST")
print("=" * 120)

print(
    "Teacher: OFF"
)

print(
    "Classical XY controller: OFF"
)

print(
    "Runtime collective bias: OFF"
)

print(
    "Only single 4-action PPO."
)

print()


result = evaluate_refined(
    refined_model,
    detailed=True
)


print("\n")
print("=" * 120)
print("FINAL RESULT")
print("=" * 120)


print(
    "STRICT SUCCESS :",
    result[
        "success"
    ]
)


print(
    "TIME           :",
    round(
        result[
            "time"
        ],
        2
    ),
    "s"
)


print(
    "ALTITUDE       :",
    round(
        result[
            "altitude"
        ],
        3
    ),
    "ft"
)


print(
    "ALT ERROR      :",
    round(
        result[
            "error"
        ],
        3
    ),
    "ft"
)


print(
    "VERTICAL SPEED :",
    round(
        result[
            "vs"
        ],
        3
    ),
    "ft/s"
)


print(
    "MAX DRIFT      :",
    round(
        result[
            "max_drift"
        ],
        3
    ),
    "ft"
)


print(
    "FINAL DRIFT    :",
    round(
        result[
            "final_drift"
        ],
        3
    ),
    "ft"
)


print(
    "PATH           :",
    round(
        result[
            "path"
        ],
        3
    ),
    "ft"
)


print(
    "NORTH          :",
    round(
        result[
            "north"
        ],
        3
    ),
    "ft"
)


print(
    "EAST           :",
    round(
        result[
            "east"
        ],
        3
    ),
    "ft"
)


# ============================================================
# SAVE ONLY IF STRICT
# ============================================================

if result["success"]:

    final_path = (
        f"{OUTPUT_DIR}/"
        "AH1S_STAGE1_FINAL_300FT"
    )


    refined_model.save(
        final_path
    )


    print("\n")
    print(
        "🏆🏆🏆 STAGE 1 FINAL SUCCESS"
    )

    print(
        "Single PPO"
    )

    print(
        "4 actions"
    )

    print(
        "Teacher OFF"
    )

    print(
        "Classical controller OFF"
    )

    print(
        "Runtime bias OFF"
    )

    print("\nMODEL:")

    print(
        final_path
        +
        ".zip"
    )

else:

    print("\n")
    print(
        "⚠️ Strict 300 ft criterion "
        "not achieved."
    )

    print(
        "Original successful model "
        "remains untouched."
    )


print("=" * 120)
