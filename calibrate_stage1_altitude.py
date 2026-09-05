
import numpy as np

from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill
)


# ============================================================
# SUCCESSFUL SINGLE 4-ACTION PPO
# ============================================================

MODEL_PATH = (
    "models_stage1_distill/"
    "AH1S_STAGE1_DISTILL_SUCCESS.zip"
)


model = PPO.load(
    MODEL_PATH
)


# ============================================================
# TARGET
# ============================================================

TARGET_ALTITUDE = 300.0


# ============================================================
# ALTITUDE CORRECTION REGION
#
# Below 275 ft:
# no correction.
#
# 275 -> 295:
# smoothly introduce correction.
#
# Above 295:
# full correction.
#
# This protects the already-good climb.
# ============================================================

GATE_START_ALT = 275.0
GATE_FULL_ALT = 295.0


# ============================================================
# COLLECTIVE BIAS SWEEP
#
# These are PHYSICAL collective corrections.
#
# Example:
#
# -0.0010 means:
#
# original collective 0.6130
# becomes              0.6120
#
# Very small changes on purpose.
# ============================================================

COLLECTIVE_BIASES = [

     0.0000,

    -0.0003,
    -0.0006,
    -0.0009,

    -0.0012,
    -0.0015,
    -0.0018,

    -0.0021,
    -0.0024,
    -0.0027,
]


# ============================================================
# GATE
# ============================================================

def altitude_gate(
    altitude
):

    if altitude <= GATE_START_ALT:

        return 0.0


    if altitude >= GATE_FULL_ALT:

        return 1.0


    return float(
        (
            altitude
            -
            GATE_START_ALT
        )
        /
        (
            GATE_FULL_ALT
            -
            GATE_START_ALT
        )
    )


# ============================================================
# RUN ONE BIAS
# ============================================================

def run_bias(
    collective_bias,
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


    strict_stable_steps = 0


    required_strict_steps = int(
        10.0
        /
        dt
    )


    strict_success = False

    failed = False


    next_print = 0.0


    max_time = 105.0

    max_steps = int(
        max_time
        /
        dt
    )


    # ========================================================
    # TRAJECTORY METRICS
    # ========================================================

    min_alt_after_55 = float(
        "inf"
    )

    max_alt_after_55 = -float(
        "inf"
    )


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

        action, _ = model.predict(
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

        current_state = (
            env._state()
        )


        current_altitude = float(
            current_state[
                "altitude"
            ]
        )


        # ====================================================
        # SMOOTH ALTITUDE GATE
        # ====================================================

        gate = altitude_gate(
            current_altitude
        )


        # ====================================================
        # CONVERT PHYSICAL COLLECTIVE BIAS
        # TO PPO NORMALIZED ACTION BIAS
        #
        # collective =
        # 0.620 + 0.030 * action[0]
        #
        # therefore:
        #
        # delta_action =
        # delta_collective / 0.030
        # ====================================================

        normalized_bias = (
            collective_bias
            /
            0.030
        )


        action[0] += (
            gate
            *
            normalized_bias
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


        drift = float(
            info[
                "drift"
            ]
        )


        max_drift = float(
            info[
                "max_drift"
            ]
        )


        path = float(
            info[
                "path"
            ]
        )


        vn = float(
            info[
                "vn"
            ]
        )


        ve = float(
            info[
                "ve"
            ]
        )


        horizontal_speed = float(
            np.hypot(
                vn,
                ve
            )
        )


        if t >= 55.0:

            min_alt_after_55 = min(
                min_alt_after_55,
                altitude
            )


            max_alt_after_55 = max(
                max_alt_after_55,
                altitude
            )


        # ====================================================
        # STRICT FINAL STAGE-1 CRITERIA
        # ====================================================

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

            drift
            <=
            5.0

            and

            horizontal_speed
            <=
            1.5

            and

            max_drift
            <=
            8.0

            and

            path
            <=
            25.0
        )


        if strict_stable:

            strict_stable_steps += 1

        else:

            strict_stable_steps = 0


        if (
            strict_stable_steps
            >=
            required_strict_steps
        ):

            strict_success = True


        # ====================================================
        # DETAIL
        # ====================================================

        if (
            detailed
            and
            t >= next_print
        ):

            # Actual physical bias currently applied
            actual_bias = (
                collective_bias
                *
                gate
            )


            print(
                f"t={t:6.1f}s | "
                f"ALT={altitude:7.2f} | "
                f"ERR={altitude_error:5.2f} | "
                f"VS={vertical_speed:6.2f} | "
                f"DRIFT={drift:5.2f} | "
                f"MAX={max_drift:5.2f} | "
                f"PATH={path:5.2f} | "
                f"COL={info['collective']:.5f} | "
                f"BIAS={actual_bias:+.5f}"
            )


            next_print += 5.0


        # ====================================================
        # FAILURE
        #
        # Ignore parent "success" termination.
        #
        # Parent criterion is still ±10 ft.
        # We want ±5 ft.
        # ====================================================

        if (
            terminated
            and
            not info[
                "success"
            ]
        ):

            failed = True
            break


        # Parent can return terminated=True because its old
        # success condition was reached.
        #
        # We intentionally continue simulation.


        if altitude > 360.0:

            failed = True
            break


        if altitude < 250.0 and t > 70.0:

            failed = True
            break


        if drift > 15.0:

            failed = True
            break


        # Once strict success is maintained 10 seconds,
        # this bias is good enough.
        if strict_success:

            break


    env.close()


    # ========================================================
    # RESULT
    # ========================================================

    if (
        min_alt_after_55
        ==
        float("inf")
    ):

        min_alt_after_55 = altitude


    if (
        max_alt_after_55
        ==
        -float("inf")
    ):

        max_alt_after_55 = altitude


    return {

        "bias":
            collective_bias,

        "success":
            strict_success,

        "failed":
            failed,

        "time":
            t,

        "altitude":
            altitude,

        "altitude_error":
            altitude_error,

        "vertical_speed":
            vertical_speed,

        "max_drift":
            max_drift,

        "final_drift":
            drift,

        "path":
            path,

        "north":
            float(
                info["north"]
            ),

        "east":
            float(
                info["east"]
            ),

        "vn":
            vn,

        "ve":
            ve,

        "min_alt_after_55":
            min_alt_after_55,

        "max_alt_after_55":
            max_alt_after_55,
    }


# ============================================================
# SWEEP
# ============================================================

print("=" * 120)

print(
    "STAGE 1 ALTITUDE BIAS CALIBRATION"
)

print("=" * 120)


print(
    "\nModel:"
)

print(
    MODEL_PATH
)


print(
    "\nTarget altitude:",
    TARGET_ALTITUDE,
    "ft"
)


print(
    "\nXY controller: NONE"
)

print(
    "Teacher: NONE"
)

print(
    "Flight model: SINGLE 4-ACTION PPO"
)


print("\n")
print("=" * 120)

print(
    "BIAS SWEEP"
)

print("=" * 120)


results = []


for bias in COLLECTIVE_BIASES:

    result = run_bias(
        bias,
        detailed=False
    )


    results.append(
        result
    )


    if result["success"]:

        icon = "🏆"

    elif result["failed"]:

        icon = "❌"

    else:

        icon = "✅"


    print(
        f"{icon} "
        f"BIAS={bias:+.4f} | "
        f"ALT={result['altitude']:7.2f} | "
        f"ERR={result['altitude_error']:5.2f} | "
        f"VS={result['vertical_speed']:6.2f} | "
        f"MAX={result['max_drift']:5.2f} | "
        f"FINAL={result['final_drift']:5.2f} | "
        f"PATH={result['path']:5.2f} | "
        f"ALT55+=["
        f"{result['min_alt_after_55']:.2f}, "
        f"{result['max_alt_after_55']:.2f}]"
    )


# ============================================================
# RANK
# ============================================================

def score(
    result
):

    # Primary:
    # altitude accuracy

    # Then:
    # low vertical speed

    # Then:
    # preserve XY geometry

    return (

        20.0
        *
        result[
            "altitude_error"
        ]

        +

        5.0
        *
        abs(
            result[
                "vertical_speed"
            ]
        )

        +

        3.0
        *
        result[
            "max_drift"
        ]

        +

        result[
            "path"
        ]
    )


results.sort(
    key=score
)


print("\n")
print("=" * 120)

print(
    "RANKED RESULTS"
)

print("=" * 120)


for i, result in enumerate(
    results,
    start=1
):

    print(
        f"{i:2d}. "
        f"BIAS={result['bias']:+.4f} | "
        f"SUCCESS={str(result['success']):5s} | "
        f"ALT={result['altitude']:7.2f} | "
        f"ERR={result['altitude_error']:5.2f} | "
        f"VS={result['vertical_speed']:6.2f} | "
        f"MAX={result['max_drift']:5.2f} | "
        f"FINAL={result['final_drift']:5.2f} | "
        f"PATH={result['path']:5.2f}"
    )


# ============================================================
# DETAILED BEST
# ============================================================

best = results[0]


print("\n")
print("=" * 120)

print(
    "DETAILED BEST ALTITUDE CALIBRATION"
)

print("=" * 120)


print(
    "BIAS =",
    best["bias"]
)


print()


best_detailed = run_bias(
    best["bias"],
    detailed=True
)


print("\n")
print("=" * 120)

print(
    "FINAL BEST RESULT"
)

print("=" * 120)


print(
    "STRICT SUCCESS :",
    best_detailed[
        "success"
    ]
)


print(
    "BIAS           :",
    best_detailed[
        "bias"
    ]
)


print(
    "TIME           :",
    round(
        best_detailed[
            "time"
        ],
        2
    ),
    "s"
)


print(
    "ALTITUDE       :",
    round(
        best_detailed[
            "altitude"
        ],
        2
    ),
    "ft"
)


print(
    "ALT ERROR      :",
    round(
        best_detailed[
            "altitude_error"
        ],
        2
    ),
    "ft"
)


print(
    "VERTICAL SPEED :",
    round(
        best_detailed[
            "vertical_speed"
        ],
        3
    ),
    "ft/s"
)


print(
    "MAX DRIFT      :",
    round(
        best_detailed[
            "max_drift"
        ],
        2
    ),
    "ft"
)


print(
    "FINAL DRIFT    :",
    round(
        best_detailed[
            "final_drift"
        ],
        2
    ),
    "ft"
)


print(
    "PATH           :",
    round(
        best_detailed[
            "path"
        ],
        2
    ),
    "ft"
)


print(
    "NORTH          :",
    round(
        best_detailed[
            "north"
        ],
        3
    ),
    "ft"
)


print(
    "EAST           :",
    round(
        best_detailed[
            "east"
        ],
        3
    ),
    "ft"
)


print(
    "VN             :",
    round(
        best_detailed[
            "vn"
        ],
        4
    ),
    "ft/s"
)


print(
    "VE             :",
    round(
        best_detailed[
            "ve"
        ],
        4
    ),
    "ft/s"
)


print(
    "ALT 55s+ RANGE :",
    round(
        best_detailed[
            "min_alt_after_55"
        ],
        2
    ),
    "->",
    round(
        best_detailed[
            "max_alt_after_55"
        ],
        2
    ),
    "ft"
)


print("=" * 120)
