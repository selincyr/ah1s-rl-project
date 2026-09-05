import os
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
    "models_stage1_final/"
    "AH1S_STAGE1_FINAL_300FT.zip"
)

OUTPUT_DIR = (
    "models_stage1_final_distilled"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FINAL PD TEACHER
#
# Calibrated from 120 second sweep:
#
# B  = -0.0021
# KP =  0.00050
# KD =  0.0017
# ============================================================

TARGET_ALT = 300.0

BASE_BIAS = -0.0021
KP = 0.00050
KD = 0.0017


GATE_START = 285.0
GATE_FULL = 298.0


MAX_NEGATIVE_CORRECTION = -0.0060
MAX_POSITIVE_CORRECTION = +0.0030


# ============================================================
# VALIDATION
# ============================================================

TOTAL_TIME = 120.0
HOVER_START = 60.0


# ============================================================
# LOAD SOURCE
# ============================================================

print("=" * 120)
print("STAGE 1 HOVER PD -> PPO DISTILLATION")
print("=" * 120)

print("\nSource:")
print(SOURCE_MODEL)


base_model = PPO.load(
    SOURCE_MODEL
)


print("\n✅ Source single 4-action PPO loaded")

print("\nTeacher:")
print(f"BASE_BIAS = {BASE_BIAS}")
print(f"KP        = {KP}")
print(f"KD        = {KD}")


# ============================================================
# GATE
# ============================================================

def gate_value(altitude):

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
# PD CORRECTION
# ============================================================

def get_pd_correction(
    altitude,
    vertical_speed
):

    signed_error = (
        TARGET_ALT
        -
        altitude
    )


    correction = (
        BASE_BIAS
        +
        KP
        *
        signed_error
        -
        KD
        *
        vertical_speed
    )


    correction = float(
        np.clip(
            correction,
            MAX_NEGATIVE_CORRECTION,
            MAX_POSITIVE_CORRECTION
        )
    )


    correction *= gate_value(
        altitude
    )


    return correction


# ============================================================
# COLLECT TEACHER TRAJECTORIES
#
# Teacher:
#
# existing single PPO
# +
# PD correction ONLY on action[0]
#
# Actions 1/2/3 are exactly original PPO outputs.
# ============================================================

print("\n")
print("=" * 120)
print("COLLECTING PD TEACHER DATA")
print("=" * 120)


observations = []
target_actions = []


# Small coordinate variations help prevent collective
# output from accidentally depending too much on XY.
OFFSETS = [

    (0.0, 0.0),

    (+1.0, 0.0),
    (-1.0, 0.0),

    (0.0, +1.0),
    (0.0, -1.0),

    (+1.0, +1.0),
    (-1.0, -1.0),
]


for episode_index, (
    offset_n,
    offset_e
) in enumerate(
    OFFSETS,
    start=1
):

    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False
    )


    obs, info = env.reset()


    env.north = offset_n
    env.east = offset_e

    obs = env._get_obs()


    samples = 0


    max_steps = int(
        TOTAL_TIME
        /
        env.dt
    )


    for step in range(
        max_steps
    ):

        # ====================================================
        # ORIGINAL SINGLE PPO
        # ====================================================

        original_action, _ = (
            base_model.predict(
                obs,
                deterministic=True
            )
        )


        original_action = np.asarray(
            original_action,
            dtype=np.float32
        ).copy()


        # ====================================================
        # CURRENT STATE
        # ====================================================

        state = env._state()


        altitude = float(
            state["altitude"]
        )


        vertical_speed = float(
            state["vertical_speed"]
        )


        # ====================================================
        # PD TEACHER
        # ====================================================

        correction = get_pd_correction(
            altitude,
            vertical_speed
        )


        teacher_action = (
            original_action.copy()
        )


        # collective =
        # 0.620 + 0.030 * action[0]
        #
        # therefore:
        #
        # delta_a0 = physical correction / 0.030
        teacher_action[0] += (
            correction
            /
            0.030
        )


        teacher_action[0] = float(
            np.clip(
                teacher_action[0],
                -1.0,
                1.0
            )
        )


        # ====================================================
        # STORE FULL TARGET
        #
        # a1/a2/a3 remain original.
        # ====================================================

        observations.append(
            obs.copy()
        )


        target_actions.append(
            teacher_action.copy()
        )


        # ====================================================
        # TEACHER FLIES
        # ====================================================

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            teacher_action
        )


        samples += 1


        # Parent success should NOT end the 120 sec dataset.
        if (
            terminated
            and
            not info["success"]
        ):

            print(
                f"⚠️ Episode {episode_index} "
                f"physical failure."
            )

            break


    env.close()


    print(
        f"Episode {episode_index}: "
        f"{samples} samples"
    )


observations = np.asarray(
    observations,
    dtype=np.float32
)


target_actions = np.asarray(
    target_actions,
    dtype=np.float32
)


print("\nDataset:")
print(
    "Observations:",
    observations.shape
)

print(
    "Actions     :",
    target_actions.shape
)


# ============================================================
# EXTRACT FROZEN ACTOR LATENT FEATURES
#
# We are NOT modifying:
#
# - observation processing
# - actor hidden layers
# - elevator output
# - aileron output
# - rudder output
#
# Only collective output row will change.
# ============================================================

print("\n")
print("=" * 120)
print("EXTRACTING FROZEN PPO FEATURES")
print("=" * 120)


device = base_model.device


obs_tensor = torch.as_tensor(
    observations,
    dtype=torch.float32,
    device=device
)


latent_batches = []

BATCH = 512


with torch.no_grad():

    for start in range(
        0,
        len(observations),
        BATCH
    ):

        batch_obs = obs_tensor[
            start:
            start + BATCH
        ]


        features = (
            base_model
            .policy
            .extract_features(
                batch_obs
            )
        )


        latent_pi = (
            base_model
            .policy
            .mlp_extractor
            .forward_actor(
                features
            )
        )


        latent_batches.append(
            latent_pi
            .detach()
            .cpu()
            .numpy()
        )


latent = np.concatenate(
    latent_batches,
    axis=0
)


print(
    "Actor latent:",
    latent.shape
)


# ============================================================
# DESIGN MATRIX
#
# action0 =
#
# latent @ weight
# +
# bias
# ============================================================

ones = np.ones(
    (
        latent.shape[0],
        1
    ),
    dtype=np.float64
)


X = np.concatenate(
    [
        latent.astype(
            np.float64
        ),
        ones
    ],
    axis=1
)


y = target_actions[
    :,
    0
].astype(
    np.float64
)


# ============================================================
# ORIGINAL COLLECTIVE HEAD PARAMETERS
# ============================================================

original_weight = (
    base_model
    .policy
    .action_net
    .weight[
        0
    ]
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float64
    )
)


original_bias = float(
    base_model
    .policy
    .action_net
    .bias[
        0
    ]
    .detach()
    .cpu()
    .numpy()
)


original_params = np.concatenate(
    [
        original_weight,
        np.array(
            [original_bias],
            dtype=np.float64
        )
    ]
)


print(
    "Collective head parameters:",
    original_params.shape
)


# ============================================================
# RIDGE AROUND ORIGINAL PPO
#
# We solve:
#
# min ||X w - y||²
#     + lambda ||w - w_original||²
#
# Large lambda:
# preserve original PPO more.
#
# Small lambda:
# imitate PD teacher more.
# ============================================================

RIDGE_VALUES = [

    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    500.0,
    1000.0,
]


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    detailed=False
):

    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False
    )


    obs, info = env.reset()


    dt = env.dt


    max_steps = int(
        TOTAL_TIME
        /
        dt
    )


    hover_altitudes = []
    hover_vertical_speeds = []


    altitude_violations = 0
    vs_violations = 0


    physical_failure = False

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
            info["altitude"]
        )


        vertical_speed = float(
            info["vertical_speed"]
        )


        altitude_error = abs(
            TARGET_ALT
            -
            altitude
        )


        if t >= HOVER_START:

            hover_altitudes.append(
                altitude
            )


            hover_vertical_speeds.append(
                vertical_speed
            )


            if altitude_error > 5.0:

                altitude_violations += 1


            if abs(
                vertical_speed
            ) > 0.75:

                vs_violations += 1


        if (
            detailed
            and
            t >= next_print
        ):

            print(
                f"t={t:6.1f}s | "
                f"ALT={altitude:7.2f} | "
                f"ERR={altitude_error:5.2f} | "
                f"VS={vertical_speed:6.3f} | "
                f"DRIFT={info['drift']:5.2f} | "
                f"MAX={info['max_drift']:5.2f} | "
                f"PATH={info['path']:5.2f} | "
                f"N={info['north']:6.2f} | "
                f"E={info['east']:6.2f} | "
                f"COL={info['collective']:.5f}"
            )


            next_print += 5.0


        if (
            terminated
            and
            not info["success"]
        ):

            physical_failure = True
            break


        if altitude > 340.0:

            physical_failure = True
            break


        if (
            altitude < 270.0
            and
            t > 70.0
        ):

            physical_failure = True
            break


        if info[
            "max_drift"
        ] > 12.0:

            physical_failure = True
            break


    env.close()


    hover_altitudes = np.asarray(
        hover_altitudes,
        dtype=np.float64
    )


    hover_vertical_speeds = np.asarray(
        hover_vertical_speeds,
        dtype=np.float64
    )


    if len(
        hover_altitudes
    ) == 0:

        return {
            "success":
                False,

            "score":
                1e12
        }


    mean_alt = float(
        np.mean(
            hover_altitudes
        )
    )


    std_alt = float(
        np.std(
            hover_altitudes
        )
    )


    min_alt = float(
        np.min(
            hover_altitudes
        )
    )


    max_alt = float(
        np.max(
            hover_altitudes
        )
    )


    max_error = float(
        np.max(
            np.abs(
                hover_altitudes
                -
                TARGET_ALT
            )
        )
    )


    max_abs_vs = float(
        np.max(
            np.abs(
                hover_vertical_speeds
            )
        )
    )


    max_drift = float(
        info[
            "max_drift"
        ]
    )


    final_drift = float(
        info[
            "drift"
        ]
    )


    path = float(
        info[
            "path"
        ]
    )


    success = bool(

        not physical_failure

        and

        altitude_violations == 0

        and

        vs_violations == 0

        and

        max_drift <= 8.0

        and

        final_drift <= 5.0

        and

        path <= 25.0
    )


    score = (

        1_000_000
        *
        altitude_violations

        +

        1_000_000
        *
        vs_violations

        +

        100.0
        *
        max_error

        +

        30.0
        *
        abs(
            mean_alt
            -
            TARGET_ALT
        )

        +

        20.0
        *
        std_alt

        +

        5.0
        *
        max_drift

        +

        path
    )


    return {

        "success":
            success,

        "score":
            score,

        "mean_alt":
            mean_alt,

        "std_alt":
            std_alt,

        "min_alt":
            min_alt,

        "max_alt":
            max_alt,

        "max_error":
            max_error,

        "max_abs_vs":
            max_abs_vs,

        "alt_violations":
            altitude_violations,

        "vs_violations":
            vs_violations,

        "max_drift":
            max_drift,

        "final_drift":
            final_drift,

        "path":
            path,

        "final_alt":
            altitude,

        "final_vs":
            vertical_speed,
    }


# ============================================================
# RIDGE SWEEP
# ============================================================

print("\n")
print("=" * 120)
print("COLLECTIVE HEAD RIDGE DISTILLATION")
print("=" * 120)


XtX = (
    X.T
    @
    X
)


Xty = (
    X.T
    @
    y
)


identity = np.eye(
    X.shape[1],
    dtype=np.float64
)


results = []


for ridge in RIDGE_VALUES:

    # ========================================================
    # Ridge around existing PPO weights
    # ========================================================

    A = (
        XtX
        +
        ridge
        *
        identity
    )


    b = (
        Xty
        +
        ridge
        *
        original_params
    )


    fitted_params = np.linalg.solve(
        A,
        b
    )


    fitted_weight = (
        fitted_params[
            :-1
        ]
    )


    fitted_bias = float(
        fitted_params[
            -1
        ]
    )


    # ========================================================
    # FRESH MODEL EVERY TEST
    # ========================================================

    candidate = PPO.load(
        SOURCE_MODEL
    )


    # ========================================================
    # ONLY ROW 0 CHANGES
    #
    # Rows:
    #
    # 1 elevator
    # 2 aileron
    # 3 rudder
    #
    # remain byte-for-byte from source.
    # ========================================================

    with torch.no_grad():

        candidate
        .policy
        .action_net
        .weight[
            0
        ].copy_(
            torch.as_tensor(
                fitted_weight,
                dtype=(
                    candidate
                    .policy
                    .action_net
                    .weight
                    .dtype
                ),
                device=(
                    candidate
                    .policy
                    .action_net
                    .weight
                    .device
                )
            )
        )


        candidate
        .policy
        .action_net
        .bias[
            0
        ].copy_(
            torch.tensor(
                fitted_bias,
                dtype=(
                    candidate
                    .policy
                    .action_net
                    .bias
                    .dtype
                ),
                device=(
                    candidate
                    .policy
                    .action_net
                    .bias
                    .device
                )
            )
        )


    # ========================================================
    # TEACHER-OFF 120 SECOND VALIDATION
    # ========================================================

    result = validate(
        candidate,
        detailed=False
    )


    result[
        "ridge"
    ] = ridge


    results.append(
        (
            result,
            candidate
        )
    )


    if result[
        "success"
    ]:

        icon = "🏆"

    else:

        icon = "•"


    print(
        f"{icon} "
        f"RIDGE={ridge:8.2f} | "
        f"SUCCESS={str(result['success']):5s} | "
        f"MEAN={result['mean_alt']:7.3f} | "
        f"RANGE=["
        f"{result['min_alt']:.3f},"
        f"{result['max_alt']:.3f}] | "
        f"MAXERR={result['max_error']:5.3f} | "
        f"MAXVS={result['max_abs_vs']:5.3f} | "
        f"VIOL={result['alt_violations']:4d} | "
        f"MAXDRIFT={result['max_drift']:5.3f} | "
        f"PATH={result['path']:5.3f}"
    )


# ============================================================
# SORT
# ============================================================

results.sort(
    key=lambda pair:
        pair[0][
            "score"
        ]
)


best_result = (
    results[0][0]
)


best_model = (
    results[0][1]
)


print("\n")
print("=" * 120)
print("BEST DISTILLED CANDIDATE")
print("=" * 120)


print(
    "RIDGE:",
    best_result[
        "ridge"
    ]
)


print(
    "SUCCESS:",
    best_result[
        "success"
    ]
)


print(
    "MEAN ALT:",
    round(
        best_result[
            "mean_alt"
        ],
        3
    )
)


print(
    "RANGE:",
    round(
        best_result[
            "min_alt"
        ],
        3
    ),
    "->",
    round(
        best_result[
            "max_alt"
        ],
        3
    )
)


print(
    "MAX ERROR:",
    round(
        best_result[
            "max_error"
        ],
        3
    )
)


print(
    "MAX |VS|:",
    round(
        best_result[
            "max_abs_vs"
        ],
        3
    )
)


print(
    "MAX DRIFT:",
    round(
        best_result[
            "max_drift"
        ],
        3
    )
)


print(
    "FINAL DRIFT:",
    round(
        best_result[
            "final_drift"
        ],
        3
    )
)


print(
    "PATH:",
    round(
        best_result[
            "path"
        ],
        3
    )
)


# ============================================================
# DETAILED FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 120)
print("FINAL 120 SECOND TEACHER-OFF VALIDATION")
print("=" * 120)

print(
    "Teacher              : OFF"
)

print(
    "PD controller        : OFF"
)

print(
    "Classical XY control : OFF"
)

print(
    "Runtime bias         : OFF"
)

print(
    "Policy               : SINGLE 4-ACTION PPO"
)

print()


final_result = validate(
    best_model,
    detailed=True
)


print("\n")
print("=" * 120)
print("FINAL RESULT")
print("=" * 120)


print(
    "SUSTAINED SUCCESS :",
    final_result[
        "success"
    ]
)


print(
    "MEAN ALTITUDE     :",
    round(
        final_result[
            "mean_alt"
        ],
        3
    ),
    "ft"
)


print(
    "STD ALTITUDE      :",
    round(
        final_result[
            "std_alt"
        ],
        3
    ),
    "ft"
)


print(
    "MIN ALTITUDE      :",
    round(
        final_result[
            "min_alt"
        ],
        3
    ),
    "ft"
)


print(
    "MAX ALTITUDE      :",
    round(
        final_result[
            "max_alt"
        ],
        3
    ),
    "ft"
)


print(
    "MAX ALT ERROR     :",
    round(
        final_result[
            "max_error"
        ],
        3
    ),
    "ft"
)


print(
    "MAX |VS|          :",
    round(
        final_result[
            "max_abs_vs"
        ],
        3
    ),
    "ft/s"
)


print(
    "ALT VIOLATIONS    :",
    final_result[
        "alt_violations"
    ]
)


print(
    "VS VIOLATIONS     :",
    final_result[
        "vs_violations"
    ]
)


print(
    "MAX DRIFT         :",
    round(
        final_result[
            "max_drift"
        ],
        3
    ),
    "ft"
)


print(
    "FINAL DRIFT       :",
    round(
        final_result[
            "final_drift"
        ],
        3
    ),
    "ft"
)


print(
    "PATH              :",
    round(
        final_result[
            "path"
        ],
        3
    ),
    "ft"
)


print(
    "FINAL ALTITUDE    :",
    round(
        final_result[
            "final_alt"
        ],
        3
    ),
    "ft"
)


print(
    "FINAL VS          :",
    round(
        final_result[
            "final_vs"
        ],
        3
    ),
    "ft/s"
)


# ============================================================
# SAVE ONLY ON REAL FINAL SUCCESS
# ============================================================

if final_result[
    "success"
]:

    FINAL_PATH = (
        f"{OUTPUT_DIR}/"
        "AH1S_STAGE1_FINAL_DISTILLED"
    )


    best_model.save(
        FINAL_PATH
    )


    print("\n")
    print(
        "🏆🏆🏆 STAGE 1 COMPLETE"
    )

    print()
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
        "PD controller OFF"
    )

    print(
        "Classical XY controller OFF"
    )

    print(
        "Runtime bias OFF"
    )

    print(
        "120 second sustained validation PASSED"
    )

    print("\nFINAL MODEL:")

    print(
        FINAL_PATH
        +
        ".zip"
    )

else:

    print("\n")
    print(
        "⚠️ PD teacher is successful, "
        "but collective-head distillation "
        "did not yet reproduce it."
    )

    print(
        "Source model remains untouched."
    )


print("=" * 120)
