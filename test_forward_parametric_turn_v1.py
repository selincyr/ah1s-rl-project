%%writefile /content/ah1s-rl-project/test_forward_parametric_turn_v1.py

from pathlib import Path
import csv
import json
import math

import numpy as np


# ================================================================
# AH-1S / JSBSim
# PARAMETRIC FORWARD-FLIGHT TURN TEACHER V1
#
# USER INPUT:
#   +50   -> positive direction 50 deg
#   +200  -> positive direction 200 deg
#   -50   -> negative direction 50 deg
#   +360  -> full positive revolution
#
# IMPORTANT:
# We do NOT reduce +200 to -160.
# We track cumulative/unwrapped heading change.
#
# Stage-2 policy:
#   action[0] collective
#   action[1] elevator
#
# Turn teacher residual:
#   action[2] aileron
#   action[3] rudder
# ================================================================


AUTH_SOURCE = Path(
    "diagnose_forward_turn_authority_v1.py"
)

RESULT_DIR = Path(
    "results_forward_parametric_turn_v1"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# USER COMMAND
# ================================================================

TARGET_TURN_DEG = float(
    input(
        "Kaç derece dönsün? "
        "(örn: 50, 200, 360, -50): "
    )
)

if abs(TARGET_TURN_DEG) < 1e-6:
    raise ValueError(
        "Dönüş açısı 0 olamaz."
    )


# ================================================================
# LOAD EXISTING LOCKED FORWARD-FLIGHT DEFINITIONS
# ================================================================

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
    1
)[0]

ns = {
    "__name__": "parametric_turn_base",
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

# Turn entry during established Stage-2 forward flight.
ns["TURN_ENTRY_FORWARD_FT"] = 160.0

build_forward_entry = ns["build_forward_entry"]
close_case = ns["close_case"]
snapshot = ns["snapshot"]
raw_policy_cycle = ns["raw_policy_cycle"]
stage2_model = ns["stage2_model"]


# ================================================================
# ANGLE HELPERS
# ================================================================

def wrap_deg(x):
    """
    Map angle to [-180, +180).
    Used ONLY for incremental heading differences
    and final wrapped heading display.

    NOT used to shorten the user's requested turn.
    """
    return float(
        (float(x) + 180.0) % 360.0 - 180.0
    )


def heading_0_360(x):
    return float(
        float(x) % 360.0
    )


# ================================================================
# TURN TEACHER PARAMETERS
# ================================================================

# Existing identified positive-direction authority.
MAX_AILERON_TURN = 0.30
MAX_RUDDER_TURN = 0.60

# Smaller reverse commands near/after target.
MAX_AILERON_BRAKE = 0.20
MAX_RUDDER_BRAKE = 0.35

# Heading controller.
HEADING_KP_RUDDER = 0.12
YAW_RATE_KD_RUDDER = 0.12

HEADING_KP_AILERON = 0.06
ROLL_LEVEL_KP_AILERON = 0.02


# ================================================================
# TARGET / HOLD
# ================================================================

TARGET_HEADING_TOL_DEG = 1.0
TARGET_YAW_RATE_TOL_DEG_S = 0.50
TARGET_MAX_ABS_ROLL_DEG = 5.0

TARGET_HOLD_SECONDS = 3.0

# Continue flying after completing turn so we can show that the
# helicopter establishes the new course.
POST_TURN_FORWARD_SECONDS = 10.0

MAX_TEST_TIME_S = 180.0


# ================================================================
# SAFETY / DIAGNOSTIC LIMITS
# ================================================================

SAFE_ALT_MIN_FT = 285.0
SAFE_ALT_MAX_FT = 315.0

SAFE_MAX_ABS_ROLL_DEG = 15.0
SAFE_MAX_ABS_PITCH_DEG = 12.0
SAFE_MAX_ABS_YAW_RATE_DEG_S = 25.0

SAFE_MIN_FORWARD_SPEED_FPS = 2.0


# ================================================================
# TURN TEACHER
# ================================================================

def parametric_turn_teacher(
    state,
    remaining_turn_deg,
):
    """
    remaining_turn_deg is UNWRAPPED.

    Example:
        requested = +200 deg
        completed = +130 deg
        remaining = +70 deg

    Therefore the controller keeps rotating in the requested
    direction instead of converting +200 into -160.
    """

    yaw_rate_deg_s = math.degrees(
        state["yaw_rate_rad_s"]
    )

    roll_deg = math.degrees(
        state["roll_rad"]
    )

    # ------------------------------------------------------------
    # RUDDER
    #
    # Existing project convention:
    # positive heading demand -> negative rudder residual.
    #
    # Therefore:
    #   remaining +  => rudder -
    #   remaining -  => rudder +
    # ------------------------------------------------------------

    delta_a3 = (
        -HEADING_KP_RUDDER
        * remaining_turn_deg
        +
        YAW_RATE_KD_RUDDER
        * yaw_rate_deg_s
    )

    # ------------------------------------------------------------
    # AILERON
    #
    # Existing positive turn:
    #   positive heading demand -> positive aileron residual
    # ------------------------------------------------------------

    delta_a2 = (
        HEADING_KP_AILERON
        * remaining_turn_deg
        -
        ROLL_LEVEL_KP_AILERON
        * roll_deg
    )

    # ------------------------------------------------------------
    # Direction-aware saturation
    # ------------------------------------------------------------

    if remaining_turn_deg >= 0.0:

        # Main positive turn authority.
        delta_a2 = float(
            np.clip(
                delta_a2,
                -MAX_AILERON_BRAKE,
                +MAX_AILERON_TURN,
            )
        )

        delta_a3 = float(
            np.clip(
                delta_a3,
                -MAX_RUDDER_TURN,
                +MAX_RUDDER_BRAKE,
            )
        )

    else:

        # Mirrored negative-direction controller.
        delta_a2 = float(
            np.clip(
                delta_a2,
                -MAX_AILERON_TURN,
                +MAX_AILERON_BRAKE,
            )
        )

        delta_a3 = float(
            np.clip(
                delta_a3,
                -MAX_RUDDER_BRAKE,
                +MAX_RUDDER_TURN,
            )
        )

    return {
        "remaining_turn_deg": float(
            remaining_turn_deg
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


# ================================================================
# SAFETY
# ================================================================

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

    if state["forward_speed_fps"] < SAFE_MIN_FORWARD_SPEED_FPS:
        return "forward_speed_low"

    return ""


# ================================================================
# RUN TRUE SAME-FDM MISSION
# ================================================================

print()
print("=" * 120)
print("PARAMETRIC FORWARD-FLIGHT TURN")
print("=" * 120)

print(
    f"REQUESTED TURN = "
    f"{TARGET_TURN_DEG:+.2f} deg"
)

start = build_forward_entry()

try:

    env2 = start["env2"]
    fdm = start["fdm"]

    lat0 = start["lat0"]
    lon0 = start["lon0"]

    mission_heading = start[
        "mission_heading"
    ]

    dt = start["dt"]

    initial = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    # This project's snapshot heading_error_deg is relative to
    # mission_heading, so it is convenient for tracking the turn.
    initial_heading_deg = float(
        initial["heading_error_deg"]
    )

    target_unwrapped_deg = (
        initial_heading_deg
        +
        TARGET_TURN_DEG
    )

    target_wrapped_deg = wrap_deg(
        target_unwrapped_deg
    )

    print(
        f"ENTRY HEADING ERROR = "
        f"{initial_heading_deg:+.3f} deg"
    )

    print(
        f"TARGET UNWRAPPED    = "
        f"{target_unwrapped_deg:+.3f} deg"
    )

    print(
        f"TARGET WRAPPED      = "
        f"{target_wrapped_deg:+.3f} deg"
    )

    # Tell Stage-2 environment that the NEW desired heading is
    # the final heading, so its observation/reference is consistent
    # after the turn.
    env2.target_heading = (
        mission_heading
        +
        math.radians(
            target_wrapped_deg
        )
    )

    # ------------------------------------------------------------
    # UNWRAPPED HEADING TRACKER
    # ------------------------------------------------------------

    previous_wrapped_heading = float(
        initial["heading_error_deg"]
    )

    cumulative_turn_deg = 0.0

    trace = []

    target_hold_s = 0.0
    post_turn_s = 0.0

    turn_complete = False
    success = False

    termination = "time_limit"

    max_steps = int(
        MAX_TEST_TIME_S / dt
    )

    for step in range(max_steps):

        before = snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        # --------------------------------------------------------
        # UPDATE CUMULATIVE / UNWRAPPED HEADING
        # --------------------------------------------------------

        current_wrapped_heading = float(
            before["heading_error_deg"]
        )

        incremental_heading_change = wrap_deg(
            current_wrapped_heading
            -
            previous_wrapped_heading
        )

        cumulative_turn_deg += (
            incremental_heading_change
        )

        previous_wrapped_heading = (
            current_wrapped_heading
        )

        remaining_turn_deg = (
            TARGET_TURN_DEG
            -
            cumulative_turn_deg
        )

        # --------------------------------------------------------
        # BASE STAGE-2 POLICY
        # --------------------------------------------------------

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

        action = base_action.copy()

        # --------------------------------------------------------
        # TURN PHASE
        # --------------------------------------------------------

        if not turn_complete:

            ctrl = parametric_turn_teacher(
                before,
                remaining_turn_deg,
            )

            # Add teacher residual only to lateral/yaw channels.
            action[2] = np.clip(
                base_action[2]
                +
                ctrl["delta_a2"],
                -1.0,
                +1.0,
            )

            action[3] = np.clip(
                base_action[3]
                +
                ctrl["delta_a3"],
                -1.0,
                +1.0,
            )

        else:

            # ----------------------------------------------------
            # TARGET REACHED:
            # teacher OFF.
            #
            # Stage-2 neural policy alone continues forward flight
            # using the new env2.target_heading reference.
            # ----------------------------------------------------

            ctrl = {
                "remaining_turn_deg":
                    float(remaining_turn_deg),

                "yaw_rate_deg_s":
                    float(
                        math.degrees(
                            before[
                                "yaw_rate_rad_s"
                            ]
                        )
                    ),

                "roll_deg":
                    float(
                        math.degrees(
                            before[
                                "roll_rad"
                            ]
                        )
                    ),

                "delta_a2": 0.0,
                "delta_a3": 0.0,
            }

        # --------------------------------------------------------
        # APPLY ACTION TO LIVE JSBSIM
        # --------------------------------------------------------

        state, used_action = raw_policy_cycle(
            env2,
            fdm,
            action,
            lat0,
            lon0,
            mission_heading,
        )

        yaw_rate_deg_s = math.degrees(
            state["yaw_rate_rad_s"]
        )

        roll_deg = math.degrees(
            state["roll_rad"]
        )

        # --------------------------------------------------------
        # COMPLETION CONDITION
        # --------------------------------------------------------

        angle_ok = (
            abs(remaining_turn_deg)
            <= TARGET_HEADING_TOL_DEG
        )

        yaw_ok = (
            abs(yaw_rate_deg_s)
            <= TARGET_YAW_RATE_TOL_DEG_S
        )

        roll_ok = (
            abs(roll_deg)
            <= TARGET_MAX_ABS_ROLL_DEG
        )

        if (
            angle_ok
            and yaw_ok
            and roll_ok
        ):

            target_hold_s += dt

        else:

            if not turn_complete:
                target_hold_s = 0.0

        if (
            not turn_complete
            and
            target_hold_s
            >= TARGET_HOLD_SECONDS
        ):

            turn_complete = True

            print()
            print(
                "TURN COMPLETE | "
                f"requested="
                f"{TARGET_TURN_DEG:+.2f} | "
                f"completed="
                f"{cumulative_turn_deg:+.2f}"
            )

            print(
                "Teacher OFF -> "
                "Stage-2 policy continues "
                "on new heading."
            )

            print()

        # --------------------------------------------------------
        # POST-TURN STRAIGHT FLIGHT
        # --------------------------------------------------------

        if turn_complete:

            post_turn_s += dt

            if (
                post_turn_s
                >= POST_TURN_FORWARD_SECONDS
            ):

                success = True
                termination = (
                    "turn_and_new_heading_forward_pass"
                )
                break

        # --------------------------------------------------------
        # LOG
        # --------------------------------------------------------

        trace.append(
            {
                "time_s":
                    float(
                        (step + 1) * dt
                    ),

                "requested_turn_deg":
                    float(
                        TARGET_TURN_DEG
                    ),

                "cumulative_turn_deg":
                    float(
                        cumulative_turn_deg
                    ),

                "remaining_turn_deg":
                    float(
                        remaining_turn_deg
                    ),

                "wrapped_heading_error_deg":
                    float(
                        state[
                            "heading_error_deg"
                        ]
                    ),

                "target_wrapped_heading_deg":
                    float(
                        target_wrapped_deg
                    ),

                "altitude_ft":
                    float(
                        state["altitude_ft"]
                    ),

                "vertical_speed_fps":
                    float(
                        state[
                            "vertical_speed_fps"
                        ]
                    ),

                "forward_ft":
                    float(
                        state["forward_ft"]
                    ),

                "forward_speed_fps":
                    float(
                        state[
                            "forward_speed_fps"
                        ]
                    ),

                "roll_deg":
                    float(
                        roll_deg
                    ),

                "pitch_deg":
                    float(
                        math.degrees(
                            state["pitch_rad"]
                        )
                    ),

                "yaw_rate_deg_s":
                    float(
                        yaw_rate_deg_s
                    ),

                "teacher_delta_a2":
                    float(
                        ctrl["delta_a2"]
                    ),

                "teacher_delta_a3":
                    float(
                        ctrl["delta_a3"]
                    ),

                "used_collective":
                    float(
                        used_action[0]
                    ),

                "used_elevator":
                    float(
                        used_action[1]
                    ),

                "used_aileron":
                    float(
                        used_action[2]
                    ),

                "used_rudder":
                    float(
                        used_action[3]
                    ),

                "turn_complete":
                    bool(
                        turn_complete
                    ),
            }
        )

        # --------------------------------------------------------
        # SAFETY
        # --------------------------------------------------------

        reason = safety_reason(
            state
        )

        if reason:

            termination = (
                "safety:"
                +
                reason
            )

            print(
                "SAFETY STOP:",
                reason
            )

            break

        # --------------------------------------------------------
        # TERMINAL DISPLAY
        # --------------------------------------------------------

        if step % max(
            1,
            int(1.0 / dt)
        ) == 0:

            print(
                f"t="
                f"{(step + 1)*dt:6.1f}s | "

                f"TURN="
                f"{cumulative_turn_deg:+7.2f}/"
                f"{TARGET_TURN_DEG:+7.2f} | "

                f"REM="
                f"{remaining_turn_deg:+7.2f} | "

                f"HDG="
                f"{state['heading_error_deg']:+7.2f} | "

                f"ROLL="
                f"{roll_deg:+6.2f} | "

                f"YR="
                f"{yaw_rate_deg_s:+6.2f} | "

                f"ALT="
                f"{state['altitude_ft']:7.2f} | "

                f"V="
                f"{state['forward_speed_fps']:6.2f}"
            )


    # ============================================================
    # SAVE RESULTS
    # ============================================================

    if trace:

        trace_path = (
            RESULT_DIR
            /
            "parametric_turn_trace.csv"
        )

        with trace_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    trace[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                trace
            )

    final_state = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    summary = {
        "success":
            bool(success),

        "termination":
            str(termination),

        "requested_turn_deg":
            float(
                TARGET_TURN_DEG
            ),

        "cumulative_turn_deg":
            float(
                cumulative_turn_deg
            ),

        "turn_error_deg":
            float(
                TARGET_TURN_DEG
                -
                cumulative_turn_deg
            ),

        "target_wrapped_heading_deg":
            float(
                target_wrapped_deg
            ),

        "final_heading_error_deg":
            float(
                final_state[
                    "heading_error_deg"
                ]
            ),

        "final_forward_speed_fps":
            float(
                final_state[
                    "forward_speed_fps"
                ]
            ),

        "final_altitude_ft":
            float(
                final_state[
                    "altitude_ft"
                ]
            ),

        "teacher_off_after_turn":
            True,

        "post_turn_forward_seconds":
            float(
                post_turn_s
            ),
    }

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
    print("=" * 120)

    print(
        "PARAMETRIC TURN PASS:",
        success
    )

    print(
        "REQUESTED:",
        f"{TARGET_TURN_DEG:+.3f} deg"
    )

    print(
        "COMPLETED:",
        f"{cumulative_turn_deg:+.3f} deg"
    )

    print(
        "ERROR:",
        f"{TARGET_TURN_DEG - cumulative_turn_deg:+.3f} deg"
    )

    print(
        "TERMINATION:",
        termination
    )

    print("=" * 120)


finally:

    close_case(
        start
    )
