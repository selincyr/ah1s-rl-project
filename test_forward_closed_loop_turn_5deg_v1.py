from pathlib import Path
import csv
import json
import math

import numpy as np


# =====================================================================
# AH-1S / JSBSim
# TRUE MISSION — FORWARD-FLIGHT CLOSED-LOOP +5 DEG TURN V1
# =====================================================================
#
# Data-grounded seed:
#   selected forward-flight coupling = coord_ap30_rm060
#       lateral residual  : +0.30
#       yaw residual      : -0.60
#
# This script does NOT use a fixed-duration pulse.
# It uses heading-error feedback and automatically tapers the residuals
# as the helicopter approaches entry_heading + 5 degrees.
#
# Stage-2 neural policy remains the base controller for:
#   collective
#   elevator / forward-flight behavior
#
# The new temporary TURN TEACHER controls only residual:
#   action[2] lateral
#   action[3] yaw
#
# NO descent.
# NO landing.
# NO PPO training.
#
# =====================================================================


AUTH_SOURCE = Path("diagnose_forward_turn_authority_v1.py")

RESULT_DIR = Path(
    "results_forward_closed_loop_turn_5deg_v1"
)
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# LOAD DEFINITIONS FROM THE ALREADY-RUN FORWARD AUTHORITY SCRIPT
# WITHOUT RE-RUNNING ITS 7-CASE EXPERIMENT
# =====================================================================

if not AUTH_SOURCE.exists():
    raise FileNotFoundError(
        f"Missing required source: {AUTH_SOURCE}"
    )

source_text = AUTH_SOURCE.read_text(
    encoding="utf-8"
)

RUN_MARKER = (
    'print("=" * 120)\n'
    'print("FORWARD-FLIGHT TURN AUTHORITY IDENTIFICATION V1")'
)

if RUN_MARKER not in source_text:
    raise RuntimeError(
        "Could not locate run marker in "
        "diagnose_forward_turn_authority_v1.py"
    )

prefix = source_text.split(
    RUN_MARKER,
    1,
)[0]

ns = {
    "__name__": "forward_turn_5deg_base",
    "__file__": str(AUTH_SOURCE),
}

exec(
    compile(
        prefix,
        str(AUTH_SOURCE),
        "exec",
    ),
    ns,
)

build_forward_entry = ns[
    "build_forward_entry"
]
close_case = ns[
    "close_case"
]
snapshot = ns[
    "snapshot"
]
raw_policy_cycle = ns[
    "raw_policy_cycle"
]
stage2_model = ns[
    "stage2_model"
]


# =====================================================================
# TURN TARGET / CONTROLLER
# =====================================================================

TARGET_DELTA_HEADING_DEG = 5.0

# Identified safe coordinated seed:
MAX_POSITIVE_AILERON_DELTA = 0.30
MAX_NEGATIVE_RUDDER_DELTA = 0.60

# Allow modest reverse authority for overshoot correction.
MAX_REVERSE_AILERON_DELTA = 0.20
MAX_REVERSE_RUDDER_DELTA = 0.35

# Heading feedback.
#
# At +5 deg error:
#   -0.12 * 5 = -0.60
# which exactly reproduces the selected safe rudder seed.
HEADING_KP_RUDDER = 0.12

# Positive yaw rate means we are already rotating toward +heading.
# This term reduces the negative rudder command before target crossing.
YAW_RATE_KD_RUDDER = 0.12

# Lateral residual follows turn demand.
# At +5 deg error:
#   +0.06 * 5 = +0.30
# which exactly reproduces selected coord_ap30_rm060.
HEADING_KP_AILERON = 0.06

# Small roll-leveling contribution.
# Positive delta_a2 was measured to move roll in the positive direction.
ROLL_LEVEL_KP_AILERON = 0.02


# =====================================================================
# SUCCESS / SAFETY
# =====================================================================

MAX_TURN_TIME_S = 80.0

TARGET_HEADING_TOL_DEG = 0.50
TARGET_YAW_RATE_TOL_DEG_S = 0.25

TARGET_ALT_MIN_FT = 295.0
TARGET_ALT_MAX_FT = 305.0
TARGET_VS_TOL_FPS = 0.75

TARGET_MAX_ABS_ROLL_DEG = 4.0
TARGET_MIN_FORWARD_SPEED_FPS = 5.0

TARGET_HOLD_SECONDS = 3.0


# Broad diagnostic safety envelope, not final Stage-4 acceptance.
SAFE_ALT_MIN_FT = 290.0
SAFE_ALT_MAX_FT = 310.0
SAFE_MAX_ABS_ROLL_DEG = 12.0
SAFE_MAX_ABS_PITCH_DEG = 10.0
SAFE_MAX_ABS_YAW_RATE_DEG_S = 20.0
SAFE_MIN_FORWARD_SPEED_FPS = 2.0


def wrap_deg(x):
    return float(
        (
            float(x)
            +
            180.0
        )
        %
        360.0
        -
        180.0
    )


def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def safety_reason(state):
    roll_deg = math.degrees(
        state["roll_rad"]
    )

    pitch_deg = math.degrees(
        state["pitch_rad"]
    )

    yaw_rate_deg_s = math.degrees(
        state["yaw_rate_rad_s"]
    )

    if state["altitude_ft"] < SAFE_ALT_MIN_FT:
        return "altitude_low"

    if state["altitude_ft"] > SAFE_ALT_MAX_FT:
        return "altitude_high"

    if abs(roll_deg) > SAFE_MAX_ABS_ROLL_DEG:
        return "roll_limit"

    if abs(pitch_deg) > SAFE_MAX_ABS_PITCH_DEG:
        return "pitch_limit"

    if abs(yaw_rate_deg_s) > SAFE_MAX_ABS_YAW_RATE_DEG_S:
        return "yaw_rate_limit"

    if (
        state["forward_speed_fps"]
        <
        SAFE_MIN_FORWARD_SPEED_FPS
    ):
        return "forward_speed_low"

    return ""


def controller(
    state,
    target_heading_error_deg,
):
    current_heading_error_deg = float(
        state["heading_error_deg"]
    )

    heading_error_deg = wrap_deg(
        target_heading_error_deg
        -
        current_heading_error_deg
    )

    yaw_rate_deg_s = math.degrees(
        state["yaw_rate_rad_s"]
    )

    roll_deg = math.degrees(
        state["roll_rad"]
    )

    # -------------------------------------------------------------
    # RUDDER / PEDAL
    #
    # Positive heading error needs negative rudder residual.
    # Positive yaw rate damps that command as target approaches.
    # -------------------------------------------------------------

    delta_a3 = (
        -
        HEADING_KP_RUDDER
        *
        heading_error_deg
        +
        YAW_RATE_KD_RUDDER
        *
        yaw_rate_deg_s
    )

    delta_a3 = float(
        np.clip(
            delta_a3,
            -
            MAX_NEGATIVE_RUDDER_DELTA,
            +
            MAX_REVERSE_RUDDER_DELTA,
        )
    )

    # -------------------------------------------------------------
    # AILERON / LATERAL
    #
    # Positive heading demand uses positive lateral residual,
    # exactly as identified by coord_ap30_rm060.
    #
    # A small roll-level term remains active near the target.
    # -------------------------------------------------------------

    delta_a2 = (
        HEADING_KP_AILERON
        *
        heading_error_deg
        -
        ROLL_LEVEL_KP_AILERON
        *
        roll_deg
    )

    delta_a2 = float(
        np.clip(
            delta_a2,
            -
            MAX_REVERSE_AILERON_DELTA,
            +
            MAX_POSITIVE_AILERON_DELTA,
        )
    )

    return {
        "heading_error_deg": float(
            heading_error_deg
        ),
        "yaw_rate_deg_s": float(
            yaw_rate_deg_s
        ),
        "roll_deg": float(
            roll_deg
        ),
        "delta_a2": float(
            delta_a2
        ),
        "delta_a3": float(
            delta_a3
        ),
    }


def target_state_ok(
    state,
    ctrl,
):
    roll_deg = math.degrees(
        state["roll_rad"]
    )

    return bool(
        abs(
            ctrl[
                "heading_error_deg"
            ]
        )
        <=
        TARGET_HEADING_TOL_DEG

        and

        abs(
            ctrl[
                "yaw_rate_deg_s"
            ]
        )
        <=
        TARGET_YAW_RATE_TOL_DEG_S

        and

        TARGET_ALT_MIN_FT
        <=
        state["altitude_ft"]
        <=
        TARGET_ALT_MAX_FT

        and

        abs(
            state[
                "vertical_speed_fps"
            ]
        )
        <=
        TARGET_VS_TOL_FPS

        and

        abs(
            roll_deg
        )
        <=
        TARGET_MAX_ABS_ROLL_DEG

        and

        state[
            "forward_speed_fps"
        ]
        >=
        TARGET_MIN_FORWARD_SPEED_FPS
    )


# =====================================================================
# BUILD TRUE SAME-FDM FORWARD-FLIGHT ENTRY
# =====================================================================

print("=" * 120)
print("TRUE MISSION — CLOSED-LOOP +5 DEG FORWARD-FLIGHT TURN V1")
print("=" * 120)
print("No descent. No landing. No PPO training.")
print("Turn teacher is temporary and controls only lateral/yaw residuals.")
print()

start = build_forward_entry()

try:
    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]

    initial = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    target_heading_error_deg = wrap_deg(
        initial["heading_error_deg"]
        +
        TARGET_DELTA_HEADING_DEG
    )

    print(
        f"ENTRY | "
        f"FWD={initial['forward_ft']:.3f} ft | "
        f"V={initial['forward_speed_fps']:.3f} ft/s | "
        f"ALT={initial['altitude_ft']:.3f} ft | "
        f"VS={initial['vertical_speed_fps']:+.3f} ft/s | "
        f"HDG={initial['heading_error_deg']:+.3f} deg | "
        f"ROLL={math.degrees(initial['roll_rad']):+.3f} deg"
    )

    print(
        f"TARGET HEADING ERROR = "
        f"{target_heading_error_deg:+.3f} deg "
        f"(entry + {TARGET_DELTA_HEADING_DEG:.1f} deg)"
    )

    print()

    trace = []

    hold_s = 0.0
    success = False
    termination = "time_limit"

    max_heading_change_deg = 0.0
    max_abs_roll_deg = abs(
        math.degrees(
            initial["roll_rad"]
        )
    )

    max_abs_pitch_deg = abs(
        math.degrees(
            initial["pitch_rad"]
        )
    )

    min_altitude_ft = float(
        initial["altitude_ft"]
    )

    max_altitude_ft = float(
        initial["altitude_ft"]
    )

    min_forward_speed_fps = float(
        initial["forward_speed_fps"]
    )

    max_abs_vs_fps = abs(
        initial["vertical_speed_fps"]
    )

    max_abs_yaw_rate_deg_s = abs(
        math.degrees(
            initial["yaw_rate_rad_s"]
        )
    )

    last_print_second = -1

    max_steps = int(
        MAX_TURN_TIME_S
        /
        dt
    )

    state = initial.copy()

    for step in range(
        max_steps
    ):
        before = snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        obs2 = np.asarray(
            env2._get_obs(),
            dtype=np.float32,
        )

        base_action, _ = (
            stage2_model.predict(
                obs2,
                deterministic=True,
            )
        )

        base_action = np.asarray(
            base_action,
            dtype=np.float32,
        ).reshape(-1)

        ctrl = controller(
            before,
            target_heading_error_deg,
        )

        action = base_action.copy()

        # Stage-2 neural collective/elevator stay untouched.
        action[2] = float(
            np.clip(
                action[2]
                +
                ctrl["delta_a2"],
                -1.0,
                +1.0,
            )
        )

        action[3] = float(
            np.clip(
                action[3]
                +
                ctrl["delta_a3"],
                -1.0,
                +1.0,
            )
        )

        state, used = raw_policy_cycle(
            env2,
            fdm,
            action,
            lat0,
            lon0,
            mission_heading,
        )

        elapsed = (
            step
            +
            1
        ) * dt

        after_ctrl = controller(
            state,
            target_heading_error_deg,
        )

        heading_change_deg = wrap_deg(
            state["heading_error_deg"]
            -
            initial["heading_error_deg"]
        )

        roll_deg = math.degrees(
            state["roll_rad"]
        )

        pitch_deg = math.degrees(
            state["pitch_rad"]
        )

        yaw_rate_deg_s = math.degrees(
            state["yaw_rate_rad_s"]
        )

        max_heading_change_deg = max(
            max_heading_change_deg,
            heading_change_deg,
        )

        max_abs_roll_deg = max(
            max_abs_roll_deg,
            abs(
                roll_deg
            ),
        )

        max_abs_pitch_deg = max(
            max_abs_pitch_deg,
            abs(
                pitch_deg
            ),
        )

        min_altitude_ft = min(
            min_altitude_ft,
            state["altitude_ft"],
        )

        max_altitude_ft = max(
            max_altitude_ft,
            state["altitude_ft"],
        )

        min_forward_speed_fps = min(
            min_forward_speed_fps,
            state["forward_speed_fps"],
        )

        max_abs_vs_fps = max(
            max_abs_vs_fps,
            abs(
                state[
                    "vertical_speed_fps"
                ]
            ),
        )

        max_abs_yaw_rate_deg_s = max(
            max_abs_yaw_rate_deg_s,
            abs(
                yaw_rate_deg_s
            ),
        )

        in_target = target_state_ok(
            state,
            after_ctrl,
        )

        if in_target:
            hold_s += dt
        else:
            hold_s = 0.0

        trace.append({
            "time_s": float(
                elapsed
            ),

            "forward_ft": float(
                state["forward_ft"]
            ),

            "forward_delta_from_entry_ft": float(
                state["forward_ft"]
                -
                initial["forward_ft"]
            ),

            "forward_speed_fps": float(
                state["forward_speed_fps"]
            ),

            "cross_track_ft": float(
                state["cross_track_ft"]
            ),

            "cross_track_delta_from_entry_ft": float(
                state["cross_track_ft"]
                -
                initial["cross_track_ft"]
            ),

            "lateral_speed_fps": float(
                state["lateral_speed_fps"]
            ),

            "altitude_ft": float(
                state["altitude_ft"]
            ),

            "vertical_speed_fps": float(
                state[
                    "vertical_speed_fps"
                ]
            ),

            "heading_error_deg": float(
                state["heading_error_deg"]
            ),

            "target_heading_error_deg": float(
                target_heading_error_deg
            ),

            "target_error_deg": float(
                after_ctrl[
                    "heading_error_deg"
                ]
            ),

            "heading_change_from_entry_deg": float(
                heading_change_deg
            ),

            "roll_deg": float(
                roll_deg
            ),

            "pitch_deg": float(
                pitch_deg
            ),

            "yaw_rate_deg_s": float(
                yaw_rate_deg_s
            ),

            "delta_a2": float(
                after_ctrl[
                    "delta_a2"
                ]
            ),

            "delta_a3": float(
                after_ctrl[
                    "delta_a3"
                ]
            ),

            "base_a0": float(
                base_action[0]
            ),

            "base_a1": float(
                base_action[1]
            ),

            "base_a2": float(
                base_action[2]
            ),

            "base_a3": float(
                base_action[3]
            ),

            "used_a0": float(
                used[0]
            ),

            "used_a1": float(
                used[1]
            ),

            "used_a2": float(
                used[2]
            ),

            "used_a3": float(
                used[3]
            ),

            "hold_s": float(
                hold_s
            ),

            "target_state_ok": bool(
                in_target
            ),
        })

        whole_second = int(
            elapsed
        )

        if (
            whole_second
            !=
            last_print_second
        ):
            last_print_second = (
                whole_second
            )

            print(
                f"t={elapsed:6.1f}s | "
                f"dHDG={heading_change_deg:+6.3f}° | "
                f"ERR={after_ctrl['heading_error_deg']:+6.3f}° | "
                f"YR={yaw_rate_deg_s:+6.3f}°/s | "
                f"ROLL={roll_deg:+6.3f}° | "
                f"ALT={state['altitude_ft']:7.3f} | "
                f"VS={state['vertical_speed_fps']:+6.3f} | "
                f"V={state['forward_speed_fps']:6.3f} | "
                f"dA2={after_ctrl['delta_a2']:+5.3f} | "
                f"dA3={after_ctrl['delta_a3']:+5.3f} | "
                f"HOLD={hold_s:4.1f}"
            )

        reason = safety_reason(
            state
        )

        if reason:
            termination = (
                "safety:"
                +
                reason
            )
            break

        if (
            hold_s
            >=
            TARGET_HOLD_SECONDS
        ):
            success = True
            termination = "success"
            break


    final = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    final_ctrl = controller(
        final,
        target_heading_error_deg,
    )

    forward_delta_ft = (
        final["forward_ft"]
        -
        initial["forward_ft"]
    )

    cross_delta_ft = (
        final["cross_track_ft"]
        -
        initial["cross_track_ft"]
    )

    if (
        abs(
            forward_delta_ft
        )
        >
        1e-9
        or
        abs(
            cross_delta_ft
        )
        >
        1e-9
    ):
        cumulative_path_deflection_deg = math.degrees(
            math.atan2(
                cross_delta_ft,
                forward_delta_ft,
            )
        )
    else:
        cumulative_path_deflection_deg = 0.0


    print()
    print("=" * 120)
    print("5-DEG CLOSED-LOOP TURN RESULT")
    print("=" * 120)

    print(
        "SUCCESS:",
        success,
    )

    print(
        "TERMINATION:",
        termination,
    )

    print(
        f"final heading change = "
        f"{wrap_deg(final['heading_error_deg'] - initial['heading_error_deg']):+.3f} deg"
    )

    print(
        f"final target error   = "
        f"{final_ctrl['heading_error_deg']:+.3f} deg"
    )

    print(
        f"final yaw rate       = "
        f"{math.degrees(final['yaw_rate_rad_s']):+.3f} deg/s"
    )

    print(
        f"final roll           = "
        f"{math.degrees(final['roll_rad']):+.3f} deg"
    )

    print(
        f"final pitch          = "
        f"{math.degrees(final['pitch_rad']):+.3f} deg"
    )

    print(
        f"final altitude       = "
        f"{final['altitude_ft']:.3f} ft"
    )

    print(
        f"final VS             = "
        f"{final['vertical_speed_fps']:+.3f} ft/s"
    )

    print(
        f"final forward speed  = "
        f"{final['forward_speed_fps']:.3f} ft/s"
    )

    print(
        f"forward distance during test = "
        f"{forward_delta_ft:.3f} ft"
    )

    print(
        f"cross-track change during test = "
        f"{cross_delta_ft:+.3f} ft"
    )

    print(
        f"cumulative path deflection = "
        f"{cumulative_path_deflection_deg:+.3f} deg"
    )

    print(
        f"qualified hold       = "
        f"{hold_s:.3f} s"
    )

    print()
    print(
        f"max heading change   = "
        f"{max_heading_change_deg:+.3f} deg"
    )

    print(
        f"max |roll|           = "
        f"{max_abs_roll_deg:.3f} deg"
    )

    print(
        f"min/max altitude     = "
        f"{min_altitude_ft:.3f} / "
        f"{max_altitude_ft:.3f} ft"
    )

    print(
        f"min forward speed    = "
        f"{min_forward_speed_fps:.3f} ft/s"
    )

    print(
        f"max |VS|             = "
        f"{max_abs_vs_fps:.3f} ft/s"
    )

    print(
        f"max |yaw rate|       = "
        f"{max_abs_yaw_rate_deg_s:.3f} deg/s"
    )


    summary = {
        "training_type": "NONE",
        "temporary_teacher_used": True,
        "descent_used": False,
        "landing_used": False,

        "selected_seed": (
            "coord_ap30_rm060"
        ),

        "same_fdm": bool(
            start["same_fdm"]
        ),

        "target_delta_heading_deg": float(
            TARGET_DELTA_HEADING_DEG
        ),

        "entry": {
            "forward_ft": float(
                initial["forward_ft"]
            ),
            "forward_speed_fps": float(
                initial["forward_speed_fps"]
            ),
            "altitude_ft": float(
                initial["altitude_ft"]
            ),
            "vertical_speed_fps": float(
                initial[
                    "vertical_speed_fps"
                ]
            ),
            "heading_error_deg": float(
                initial[
                    "heading_error_deg"
                ]
            ),
            "roll_deg": float(
                math.degrees(
                    initial["roll_rad"]
                )
            ),
            "pitch_deg": float(
                math.degrees(
                    initial["pitch_rad"]
                )
            ),
        },

        "success": bool(
            success
        ),

        "termination": str(
            termination
        ),

        "final_heading_change_deg": float(
            wrap_deg(
                final["heading_error_deg"]
                -
                initial["heading_error_deg"]
            )
        ),

        "final_target_error_deg": float(
            final_ctrl[
                "heading_error_deg"
            ]
        ),

        "final_yaw_rate_deg_s": float(
            math.degrees(
                final[
                    "yaw_rate_rad_s"
                ]
            )
        ),

        "final_roll_deg": float(
            math.degrees(
                final["roll_rad"]
            )
        ),

        "final_pitch_deg": float(
            math.degrees(
                final["pitch_rad"]
            )
        ),

        "final_altitude_ft": float(
            final["altitude_ft"]
        ),

        "final_vertical_speed_fps": float(
            final[
                "vertical_speed_fps"
            ]
        ),

        "final_forward_speed_fps": float(
            final[
                "forward_speed_fps"
            ]
        ),

        "forward_distance_during_test_ft": float(
            forward_delta_ft
        ),

        "cross_track_change_during_test_ft": float(
            cross_delta_ft
        ),

        "cumulative_path_deflection_deg": float(
            cumulative_path_deflection_deg
        ),

        "qualified_hold_s": float(
            hold_s
        ),

        "max_heading_change_deg": float(
            max_heading_change_deg
        ),

        "max_abs_roll_deg": float(
            max_abs_roll_deg
        ),

        "min_altitude_ft": float(
            min_altitude_ft
        ),

        "max_altitude_ft": float(
            max_altitude_ft
        ),

        "min_forward_speed_fps": float(
            min_forward_speed_fps
        ),

        "max_abs_vs_fps": float(
            max_abs_vs_fps
        ),

        "max_abs_yaw_rate_deg_s": float(
            max_abs_yaw_rate_deg_s
        ),

        "controller": {
            "heading_kp_rudder": float(
                HEADING_KP_RUDDER
            ),
            "yaw_rate_kd_rudder": float(
                YAW_RATE_KD_RUDDER
            ),
            "heading_kp_aileron": float(
                HEADING_KP_AILERON
            ),
            "roll_level_kp_aileron": float(
                ROLL_LEVEL_KP_AILERON
            ),
            "max_positive_aileron_delta": float(
                MAX_POSITIVE_AILERON_DELTA
            ),
            "max_negative_rudder_delta": float(
                MAX_NEGATIVE_RUDDER_DELTA
            ),
        },
    }


    save_csv(
        RESULT_DIR
        /
        "turn_trace.csv",
        trace,
    )

    with (
        RESULT_DIR
        /
        "final_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


    print()
    print("Saved:")
    print(
        " ",
        RESULT_DIR
        /
        "turn_trace.csv"
    )
    print(
        " ",
        RESULT_DIR
        /
        "final_summary.json"
    )

    print()
    print(
        "READY FOR 10-DEG TURN:",
        bool(
            success
            and
            abs(
                cumulative_path_deflection_deg
            )
            >
            0.05
        ),
    )

finally:
    close_case(
        start
    )
