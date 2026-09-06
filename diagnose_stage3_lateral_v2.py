from __future__ import annotations
#
"""
AH-1S / JSBSim
STAGE 3 — ENDPOINT LATERAL CONTROL DIAGNOSTIC V2
=================================================

Purpose
-------
This script does NOT train a model and does NOT claim Stage-3 success.

It produces two pieces of evidence before any lateral gain search:

A) BASELINE QUALIFICATION
   - Locked Stage-1 neural policy
   - Locked Stage-2 HYBRID FINAL neural policy
   - Same continuous JSBSim FDM object
   - Existing Stage-3 longitudinal + altitude teacher only
   - Lateral action remains at the Stage-2 cruise value (-0.230)
   - Records where/why cross-track grows and whether the longitudinal
     endpoint stop still remains valid with the rebuilt Stage-2 policy.

B) LOW-SPEED AILERON AUTHORITY IDENTIFICATION
   - Every pulse case starts from a fresh, deterministic,
     continuous Stage1 -> Stage2 -> Stage3 trajectory.
   - Longitudinal and altitude logic are unchanged.
   - Only action[2] is pulsed around the cruise aileron action.
   - Measures cross-track, lateral speed, roll, altitude and the
     actual JSBSim aileron command.
   - This determines sign/authority before designing the endpoint
     lateral feedback controller.

Why this exists
---------------
The prior endpoint-lateral V1 run ended at the 15-ft cross-track safety
limit before its lateral gate became active. Therefore a gain sweep with
that gate did not test the gains at all.

This diagnostic fixes the experiment, not the controller:
it gathers evidence first. The next script should use these results to
select activation timing and then calibrate lateral feedback.

No Stage-1/Stage-2 model is modified.
No PPO training is performed.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill
from helicopter_env_stage2_refine_mapped import HelicopterEnvStage2RefineMapped


# =====================================================================
# LOCKED MODELS
# =====================================================================

STAGE1_MODEL_PATH = Path(
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL_PATH = Path(
    "models_stage2_hybrid_final/"
    "AH1S_STAGE2_HYBRID_FINAL.zip"
)


# =====================================================================
# OUTPUT
# =====================================================================

OUT_DIR = Path(
    "results_stage3_lateral_diagnostic_v2"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASELINE_TRACE_CSV = (
    OUT_DIR / "baseline_trace.csv"
)

AUTHORITY_SUMMARY_CSV = (
    OUT_DIR / "aileron_authority_sweep.csv"
)

AUTHORITY_BEST_TRACE_CSV = (
    OUT_DIR / "aileron_authority_best_trace.csv"
)

SUMMARY_JSON = (
    OUT_DIR / "diagnostic_summary.json"
)


# =====================================================================
# MISSION / EXISTING STAGE-3 LONGITUDINAL + ALTITUDE CONTROLLER
# =====================================================================

TARGET_FORWARD_FT = 300.0
TARGET_ALT_FT = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
STAGE3_MAX_TIME = 75.0

HANDOFF_STABLE_TIME = 5.0
DEFAULT_CONTROL_DT = 0.075

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040

# Stage 3 starts taking over the longitudinal/altitude channels
# after Stage 2 has flown 80 ft from the hover point.
TEACHER_ENABLE_FORWARD_FT = 80.0

# Existing longitudinal controller. We do NOT modify these values here.
K_POS = 0.055
V_FWD_MAX = 9.0
BRAKE_LEAD_FT = 8.0
V_REV_MAX = 2.0
KV = 1.00
ELEVATOR_TRIM_ACTION = 0.013725
Q_DAMP = 2.0
MAX_ELEVATOR_RESIDUAL = 1.0

# Existing altitude controller. We do NOT modify these values here.
COLLECTIVE_BIAS = 0.22
ALT_KP = 0.030
VS_KD = 0.120
MAX_ALT_CORR = 0.45

# Stage-2 lateral/yaw values retained during baseline and authority ID.
CRUISE_AILERON_ACTION = -0.230
RUDDER_ACTION = 0.0

# Endpoint acceptance values.
STOP_POS_TOL_FT = 5.0
STOP_SPEED_TOL_FPS = 0.60
STOP_HOLD_SECONDS = 5.0

CROSS_TOL_FT = 5.0
LAT_SPEED_TOL_FPS = 0.60
ALT_MIN = 295.0
ALT_MAX = 305.0
VS_TOL_FPS = 0.75
HEADING_TOL_DEG = 1.0

# Safety. This is intentionally unchanged from V1.
ALT_SAFE_MIN = 288.0
ALT_SAFE_MAX = 312.0
MAX_ABS_PITCH_DEG = 8.0
MAX_ABS_ROLL_DEG = 10.0
MAX_CROSS_SAFE_FT = 15.0

EARTH_RADIUS_FT = 20_902_231.0


# =====================================================================
# AUTHORITY IDENTIFICATION DESIGN
# =====================================================================

# Use the baseline itself to find a safe low-speed state.
# We do not activate endpoint feedback here.
AUTH_MIN_FORWARD_FT = 235.0
AUTH_MAX_ABS_CROSS_FT = 4.0
AUTH_MAX_FORWARD_SPEED_FPS = 4.0

# Pulse action[2] around the Stage-2 cruise value.
# Physical mapped aileron delta is approximately:
#     AILERON_SCALE * action2_delta
AILERON_ACTION_DELTAS = [
    -0.40,
    -0.20,
    0.00,
    +0.20,
    +0.40,
]

PULSE_SECONDS = 2.0
RECOVERY_SECONDS = 2.0


# =====================================================================
# HELPERS
# =====================================================================

def rule(text: str):
    print()
    print("=" * 150)
    print(text)
    print("=" * 150)


def fdm_float(
    fdm,
    key,
    default=float("nan"),
):
    try:
        return float(
            fdm[key]
        )
    except Exception:
        return float(
            default
        )


def first_finite(
    fdm,
    keys,
    default=float("nan"),
):
    for key in keys:
        value = fdm_float(
            fdm,
            key,
        )

        if np.isfinite(
            value
        ):
            return value

    return float(
        default
    )


def info_float(
    info,
    key,
    default=float("nan"),
):
    try:
        return float(
            info.get(
                key,
                default,
            )
        )
    except Exception:
        return float(
            default
        )


def get_fdm(env):
    direct = getattr(
        env,
        "fdm",
        None,
    )

    if direct is not None:
        return direct

    base = getattr(
        env,
        "base_env",
        None,
    )

    if (
        base is not None
        and getattr(
            base,
            "fdm",
            None,
        )
        is not None
    ):
        return base.fdm

    raise RuntimeError(
        "Active JSBSim FDM not found."
    )


def latitude_deg(fdm):
    return first_finite(
        fdm,
        [
            "position/lat-gc-deg",
            "position/lat-geod-deg",
        ],
    )


def longitude_deg(fdm):
    return first_finite(
        fdm,
        [
            "position/long-gc-deg",
            "position/long-geod-deg",
        ],
    )


def heading_rad(fdm):
    return first_finite(
        fdm,
        [
            "attitude/heading-true-rad",
            "attitude/psi-rad",
        ],
    )


def wrap_angle(value):
    return math.atan2(
        math.sin(value),
        math.cos(value),
    )


def local_ne_ft(
    lat,
    lon,
    lat0,
    lon0,
):
    dlat = math.radians(
        lat - lat0
    )

    dlon = math.radians(
        lon - lon0
    )

    north = (
        EARTH_RADIUS_FT
        *
        dlat
    )

    east = (
        EARTH_RADIUS_FT
        *
        math.cos(
            math.radians(
                lat0
            )
        )
        *
        dlon
    )

    return (
        float(north),
        float(east),
    )


def mission_axes(
    north,
    east,
    heading,
):
    c = math.cos(
        heading
    )

    s = math.sin(
        heading
    )

    forward = (
        north * c
        +
        east * s
    )

    cross = (
        -north * s
        +
        east * c
    )

    return (
        float(forward),
        float(cross),
    )


def geometry(
    fdm,
    lat0,
    lon0,
    mission_heading,
):
    north, east = (
        local_ne_ft(
            latitude_deg(fdm),
            longitude_deg(fdm),
            lat0,
            lon0,
        )
    )

    return mission_axes(
        north,
        east,
        mission_heading,
    )


def mission_ground_velocity(
    fdm,
    mission_heading,
):
    vn = first_finite(
        fdm,
        [
            "velocities/v-north-fps",
            "velocities/v-north-fps",
        ],
    )

    ve = first_finite(
        fdm,
        [
            "velocities/v-east-fps",
            "velocities/v-east-fps",
        ],
    )

    if (
        not np.isfinite(vn)
        or
        not np.isfinite(ve)
    ):
        return (
            float("nan"),
            float("nan"),
        )

    return mission_axes(
        vn,
        ve,
        mission_heading,
    )


def physical_commands(
    fdm,
):
    return {
        "physical_collective_cmd":
            fdm_float(
                fdm,
                "fcs/collective-cmd-norm",
            ),

        "physical_elevator_cmd":
            fdm_float(
                fdm,
                "fcs/elevator-cmd-norm",
            ),

        "physical_aileron_cmd":
            fdm_float(
                fdm,
                "fcs/aileron-cmd-norm",
            ),

        "physical_rudder_cmd":
            fdm_float(
                fdm,
                "fcs/rudder-cmd-norm",
            ),
    }


def snapshot(
    fdm,
    lat0,
    lon0,
    mission_heading,
):
    forward, cross = geometry(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    heading_now = heading_rad(
        fdm
    )

    heading_error = wrap_angle(
        heading_now
        -
        mission_heading
    )

    ground_fwd, ground_cross = (
        mission_ground_velocity(
            fdm,
            mission_heading,
        )
    )

    state = {
        "forward_ft":
            float(
                forward
            ),

        "position_error_ft":
            float(
                TARGET_FORWARD_FT
                -
                forward
            ),

        "cross_track_ft":
            float(
                cross
            ),

        "altitude_ft":
            fdm_float(
                fdm,
                "position/h-agl-ft",
            ),

        # Existing controller velocity definitions.
        "forward_speed_fps":
            fdm_float(
                fdm,
                "velocities/u-aero-fps",
                0.0,
            ),

        "lateral_speed_fps":
            fdm_float(
                fdm,
                "velocities/v-aero-fps",
                0.0,
            ),

        # Extra evidence only. Not used by the controller.
        "mission_ground_forward_speed_fps":
            float(
                ground_fwd
            ),

        "mission_ground_cross_speed_fps":
            float(
                ground_cross
            ),

        "vertical_speed_fps":
            fdm_float(
                fdm,
                "velocities/h-dot-fps",
                0.0,
            ),

        "pitch_deg":
            math.degrees(
                fdm_float(
                    fdm,
                    "attitude/pitch-rad",
                    0.0,
                )
            ),

        "roll_deg":
            math.degrees(
                fdm_float(
                    fdm,
                    "attitude/roll-rad",
                    0.0,
                )
            ),

        "pitch_rate_rad_s":
            fdm_float(
                fdm,
                "velocities/q-rad_sec",
                0.0,
            ),

        "heading_error_deg":
            math.degrees(
                heading_error
            ),
    }

    state.update(
        physical_commands(
            fdm
        )
    )

    return state


def safety_reason(
    state,
):
    if (
        state["altitude_ft"]
        <
        ALT_SAFE_MIN
    ):
        return "altitude_below_safe"

    if (
        state["altitude_ft"]
        >
        ALT_SAFE_MAX
    ):
        return "altitude_above_safe"

    if (
        abs(
            state[
                "pitch_deg"
            ]
        )
        >
        MAX_ABS_PITCH_DEG
    ):
        return "pitch_limit"

    if (
        abs(
            state[
                "roll_deg"
            ]
        )
        >
        MAX_ABS_ROLL_DEG
    ):
        return "roll_limit"

    if (
        abs(
            state[
                "cross_track_ft"
            ]
        )
        >
        MAX_CROSS_SAFE_FT
    ):
        return "cross_track_safety_limit"

    return ""


def endpoint_stop_now(
    state,
):
    return bool(
        abs(
            state[
                "position_error_ft"
            ]
        )
        <=
        STOP_POS_TOL_FT
        and
        abs(
            state[
                "forward_speed_fps"
            ]
        )
        <=
        STOP_SPEED_TOL_FPS
    )


def endpoint_lateral_now(
    state,
):
    return bool(
        abs(
            state[
                "cross_track_ft"
            ]
        )
        <=
        CROSS_TOL_FT
        and
        abs(
            state[
                "lateral_speed_fps"
            ]
        )
        <=
        LAT_SPEED_TOL_FPS
    )


def endpoint_full_hover_now(
    state,
):
    return bool(
        endpoint_stop_now(
            state
        )
        and
        endpoint_lateral_now(
            state
        )
        and
        ALT_MIN
        <=
        state[
            "altitude_ft"
        ]
        <=
        ALT_MAX
        and
        abs(
            state[
                "vertical_speed_fps"
            ]
        )
        <=
        VS_TOL_FPS
        and
        abs(
            state[
                "heading_error_deg"
            ]
        )
        <=
        HEADING_TOL_DEG
    )


def env_control_dt(
    env,
):
    dt = float(
        getattr(
            env,
            "dt",
            DEFAULT_CONTROL_DT,
        )
        or
        DEFAULT_CONTROL_DT
    )

    if (
        not np.isfinite(dt)
        or
        dt <= 0.0
    ):
        dt = DEFAULT_CONTROL_DT

    return dt


def physics_steps(
    env,
):
    value = getattr(
        env,
        "PHYSICS_STEPS",
        10,
    )

    try:
        value = int(
            value
        )
    except Exception:
        value = 10

    return max(
        1,
        value,
    )


def raw_cycle(
    env2,
    action,
):
    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(
        -1
    )

    action = np.clip(
        action,
        -1.0,
        +1.0,
    ).astype(
        np.float32
    )

    # Because env2 is HelicopterEnvStage2RefineMapped, this calls
    # the repaired mapped action path.
    env2._apply_action(
        action
    )

    for _ in range(
        physics_steps(
            env2
        )
    ):
        if not env2.fdm.run():
            raise RuntimeError(
                "JSBSim stopped during Stage-3 diagnostic."
            )

    # Keep environment bookkeeping coherent even though we bypass
    # reward/termination logic intentionally for teacher diagnostics.
    if hasattr(
        env2,
        "previous_action",
    ):
        try:
            env2.previous_action = (
                action.copy()
            )
        except Exception:
            pass

    return action


# =====================================================================
# MODELS
# =====================================================================

for required_path in [
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required model missing: {required_path}"
        )

stage1_model = PPO.load(
    str(
        STAGE1_MODEL_PATH
    )
)

stage2_model = PPO.load(
    str(
        STAGE2_MODEL_PATH
    )
)


# =====================================================================
# TRUE CONTINUOUS STAGE1 -> STAGE2 -> STAGE3 START
# =====================================================================

def build_start(
    detailed=False,
):
    env1 = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False,
        )
    )

    obs1, info1 = (
        env1.reset()
    )

    fdm = get_fdm(
        env1
    )

    active_id = id(
        fdm
    )

    mission_heading = heading_rad(
        fdm
    )

    dt1 = env_control_dt(
        env1
    )

    stable_time = 0.0
    stage1_time = 0.0

    for step in range(
        int(
            STAGE1_MAX_TIME
            /
            dt1
        )
    ):
        action1, _ = (
            stage1_model.predict(
                obs1,
                deterministic=True,
            )
        )

        (
            obs1,
            _,
            terminated,
            truncated,
            info1,
        ) = env1.step(
            action1
        )

        stage1_time = (
            step + 1
        ) * dt1

        altitude = info_float(
            info1,
            "altitude",
        )

        vertical_speed = info_float(
            info1,
            "vertical_speed",
        )

        vn = info_float(
            info1,
            "vn",
            0.0,
        )

        ve = info_float(
            info1,
            "ve",
            0.0,
        )

        horizontal_speed = float(
            np.hypot(
                vn,
                ve,
            )
        )

        drift = info_float(
            info1,
            "drift",
            999.0,
        )

        stable = bool(
            295.0
            <=
            altitude
            <=
            305.0
            and
            abs(
                vertical_speed
            )
            <=
            0.50
            and
            horizontal_speed
            <=
            1.0
            and
            drift
            <=
            3.0
        )

        stable_time = (
            stable_time
            +
            dt1
            if stable
            else 0.0
        )

        if (
            stable_time
            >=
            HANDOFF_STABLE_TIME
        ):
            break

        if (
            terminated
            and
            not bool(
                info1.get(
                    "success",
                    False,
                )
            )
        ):
            env1.close()
            raise RuntimeError(
                "Stage 1 failed before handoff."
            )

        if truncated:
            env1.close()
            raise RuntimeError(
                "Stage 1 truncated before handoff."
            )

    if (
        stable_time
        <
        HANDOFF_STABLE_TIME
    ):
        env1.close()
        raise RuntimeError(
            "Stable Stage-1 handoff not reached."
        )

    stage1_handoff = {
        "stage1_time_s":
            float(
                stage1_time
            ),

        "altitude_ft":
            info_float(
                info1,
                "altitude",
            ),

        "vertical_speed_fps":
            info_float(
                info1,
                "vertical_speed",
            ),

        "horizontal_speed_fps":
            float(
                np.hypot(
                    info_float(
                        info1,
                        "vn",
                        0.0,
                    ),
                    info_float(
                        info1,
                        "ve",
                        0.0,
                    ),
                )
            ),

        "drift_ft":
            info_float(
                info1,
                "drift",
            ),
    }

    lat0 = latitude_deg(
        fdm
    )

    lon0 = longitude_deg(
        fdm
    )

    sim_time_before_attach = (
        first_finite(
            fdm,
            [
                "simulation/sim-time-sec",
            ],
        )
    )

    env2 = (
        HelicopterEnvStage2RefineMapped(
            aileron_scale=
                AILERON_SCALE,

            rudder_scale=
                RUDDER_SCALE,
        )
    )

    # Disposable reset initializes the Python-side Stage-2 object.
    env2.reset()

    env2.fdm = fdm

    if hasattr(
        env2,
        "forward_distance",
    ):
        env2.forward_distance = (
            0.0
        )

    if hasattr(
        env2,
        "target_heading",
    ):
        env2.target_heading = float(
            mission_heading
        )

    for attr in [
        "steps",
        "target_hold_steps",
        "hold_steps",
        "success_hold_steps",
    ]:
        if hasattr(
            env2,
            attr,
        ):
            setattr(
                env2,
                attr,
                0,
            )

    if (
        id(
            get_fdm(
                env2
            )
        )
        !=
        active_id
    ):
        env2.fdm = None
        env1.close()

        raise RuntimeError(
            "FDM continuity failed."
        )

    sim_time_after_attach = (
        first_finite(
            fdm,
            [
                "simulation/sim-time-sec",
            ],
        )
    )

    if (
        np.isfinite(
            sim_time_before_attach
        )
        and
        np.isfinite(
            sim_time_after_attach
        )
        and
        abs(
            sim_time_after_attach
            -
            sim_time_before_attach
        )
        >
        1e-9
    ):
        env2.fdm = None
        env1.close()

        raise RuntimeError(
            "Simulation clock changed during Stage-2 attach."
        )

    obs2 = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    dt2 = env_control_dt(
        env2
    )

    stage2_time = 0.0

    for step in range(
        int(
            STAGE2_MAX_TIME
            /
            dt2
        )
    ):
        action2, _ = (
            stage2_model.predict(
                obs2,
                deterministic=True,
            )
        )

        action2 = np.asarray(
            action2,
            dtype=np.float32,
        ).reshape(
            -1
        )

        (
            obs2,
            _,
            terminated,
            truncated,
            info2,
        ) = env2.step(
            action2
        )

        obs2 = np.asarray(
            obs2,
            dtype=np.float32,
        )

        stage2_time = (
            step + 1
        ) * dt2

        forward, cross = geometry(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        if (
            forward
            >=
            TEACHER_ENABLE_FORWARD_FT
        ):
            if hasattr(
                env2,
                "forward_distance",
            ):
                env2.forward_distance = float(
                    forward
                )

            break

        if (
            terminated
            and
            not bool(
                info2.get(
                    "success",
                    False,
                )
            )
        ):
            env2.fdm = None
            env1.close()

            raise RuntimeError(
                "Stage 2 failed before Stage-3 takeover."
            )

        if truncated:
            env2.fdm = None
            env1.close()

            raise RuntimeError(
                "Stage 2 truncated before Stage-3 takeover."
            )

    start_state = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    continuity = {
        "same_fdm_object":
            bool(
                id(
                    get_fdm(
                        env2
                    )
                )
                ==
                active_id
            ),

        "sim_clock_reset":
            bool(
                np.isfinite(
                    sim_time_before_attach
                )
                and
                np.isfinite(
                    sim_time_after_attach
                )
                and
                abs(
                    sim_time_after_attach
                    -
                    sim_time_before_attach
                )
                >
                1e-9
            ),

        "stage1_handoff":
            stage1_handoff,

        "stage2_to_stage3_time_s":
            float(
                stage2_time
            ),

        "stage3_start_state":
            start_state,
    }

    if detailed:
        print(
            "Same FDM object       :",
            continuity[
                "same_fdm_object"
            ],
        )

        print(
            "Simulation clock reset:",
            continuity[
                "sim_clock_reset"
            ],
        )

        print(
            "Stage-1 handoff       :",
            stage1_handoff,
        )

        print(
            "Stage-3 start state   :",
            start_state,
        )

    return (
        env1,
        env2,
        fdm,
        obs2,
        lat0,
        lon0,
        mission_heading,
        continuity,
    )


# =====================================================================
# EXISTING STAGE-3 LONGITUDINAL + ALTITUDE ACTION
# LATERAL CHANNEL IS EXPLICITLY PROVIDED BY THE CALLER.
# =====================================================================

def stage3_action(
    obs,
    state,
    aileron_action,
):
    base_action, _ = (
        stage2_model.predict(
            obs,
            deterministic=True,
        )
    )

    base_action = np.asarray(
        base_action,
        dtype=np.float32,
    ).reshape(
        -1
    )

    action = (
        base_action.copy()
    )

    altitude_error = (
        TARGET_ALT_FT
        -
        state[
            "altitude_ft"
        ]
    )

    collective_correction = (
        COLLECTIVE_BIAS
        +
        ALT_KP
        *
        altitude_error
        -
        VS_KD
        *
        state[
            "vertical_speed_fps"
        ]
    )

    collective_correction = float(
        np.clip(
            collective_correction,
            -MAX_ALT_CORR,
            +MAX_ALT_CORR,
        )
    )

    action[0] = float(
        np.clip(
            base_action[0]
            +
            collective_correction,
            -1.0,
            +1.0,
        )
    )

    v_des = float(
        np.clip(
            K_POS
            *
            (
                state[
                    "position_error_ft"
                ]
                -
                BRAKE_LEAD_FT
            ),
            -V_REV_MAX,
            +V_FWD_MAX,
        )
    )

    speed_error = (
        state[
            "forward_speed_fps"
        ]
        -
        v_des
    )

    elevator_residual = (
        -KV
        *
        speed_error
        +
        Q_DAMP
        *
        state[
            "pitch_rate_rad_s"
        ]
    )

    elevator_residual = float(
        np.clip(
            elevator_residual,
            -MAX_ELEVATOR_RESIDUAL,
            +MAX_ELEVATOR_RESIDUAL,
        )
    )

    action[1] = float(
        np.clip(
            ELEVATOR_TRIM_ACTION
            +
            elevator_residual,
            -1.0,
            +1.0,
        )
    )

    action[2] = float(
        np.clip(
            aileron_action,
            -1.0,
            +1.0,
        )
    )

    action[3] = (
        RUDDER_ACTION
    )

    return (
        action.astype(
            np.float32
        ),
        float(
            v_des
        ),
        float(
            collective_correction
        ),
        float(
            elevator_residual
        ),
    )


# =====================================================================
# ONE RAW STAGE-3 STEP
# =====================================================================

def stage3_step(
    env2,
    obs2,
    fdm,
    lat0,
    lon0,
    mission_heading,
    aileron_action,
):
    state_before = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    (
        action,
        v_des,
        collective_correction,
        elevator_residual,
    ) = stage3_action(
        obs2,
        state_before,
        aileron_action,
    )

    raw_cycle(
        env2,
        action,
    )

    state_after = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    if hasattr(
        env2,
        "forward_distance",
    ):
        env2.forward_distance = float(
            state_after[
                "forward_ft"
            ]
        )

    if hasattr(
        env2,
        "steps",
    ):
        try:
            env2.steps += 1
        except Exception:
            pass

    obs2 = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    row = dict(
        state_after
    )

    row.update(
        {
            "v_des_fps":
                float(
                    v_des
                ),

            "collective_correction":
                float(
                    collective_correction
                ),

            "elevator_residual":
                float(
                    elevator_residual
                ),

            "action0":
                float(
                    action[0]
                ),

            "action1":
                float(
                    action[1]
                ),

            "action2":
                float(
                    action[2]
                ),

            "action3":
                float(
                    action[3]
                ),
        }
    )

    return (
        obs2,
        state_after,
        row,
    )


# =====================================================================
# CSV WRITER
# =====================================================================

def write_rows(
    path,
    rows,
):
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
        writer.writerows(
            rows
        )


# =====================================================================
# PHASE A — BASELINE QUALIFICATION
# =====================================================================

def run_baseline(
    detailed=True,
):
    (
        env1,
        env2,
        fdm,
        obs2,
        lat0,
        lon0,
        mission_heading,
        continuity,
    ) = build_start(
        detailed=detailed,
    )

    dt = env_control_dt(
        env2
    )

    trace = []

    stop_hold = 0.0
    max_stop_hold = 0.0

    full_hover_hold = 0.0
    max_full_hover_hold = 0.0

    max_cross = 0.0

    min_alt = +999.0
    max_alt = -999.0

    termination_reason = (
        "stage3_time_limit"
    )

    next_print = 0.0

    threshold_events = {}

    for step in range(
        int(
            STAGE3_MAX_TIME
            /
            dt
        )
    ):
        (
            obs2,
            state,
            row,
        ) = stage3_step(
            env2,
            obs2,
            fdm,
            lat0,
            lon0,
            mission_heading,
            CRUISE_AILERON_ACTION,
        )

        t = (
            step + 1
        ) * dt

        row[
            "time_s"
        ] = float(
            t
        )

        row[
            "phase"
        ] = "baseline"

        trace.append(
            row
        )

        max_cross = max(
            max_cross,
            abs(
                state[
                    "cross_track_ft"
                ]
            ),
        )

        min_alt = min(
            min_alt,
            state[
                "altitude_ft"
            ],
        )

        max_alt = max(
            max_alt,
            state[
                "altitude_ft"
            ],
        )

        stop_now = endpoint_stop_now(
            state
        )

        full_hover_now = (
            endpoint_full_hover_now(
                state
            )
        )

        stop_hold = (
            stop_hold
            +
            dt
            if stop_now
            else 0.0
        )

        full_hover_hold = (
            full_hover_hold
            +
            dt
            if full_hover_now
            else 0.0
        )

        max_stop_hold = max(
            max_stop_hold,
            stop_hold,
        )

        max_full_hover_hold = max(
            max_full_hover_hold,
            full_hover_hold,
        )

        # Evidence: when important cross-track thresholds are first crossed.
        for threshold in [
            3.0,
            4.0,
            5.0,
            10.0,
            15.0,
        ]:
            key = (
                f"cross_{threshold:.0f}ft"
            )

            if (
                key
                not in
                threshold_events
                and
                abs(
                    state[
                        "cross_track_ft"
                    ]
                )
                >=
                threshold
            ):
                threshold_events[
                    key
                ] = {
                    "time_s":
                        float(
                            t
                        ),

                    "forward_ft":
                        float(
                            state[
                                "forward_ft"
                            ]
                        ),

                    "forward_speed_fps":
                        float(
                            state[
                                "forward_speed_fps"
                            ]
                        ),

                    "cross_track_ft":
                        float(
                            state[
                                "cross_track_ft"
                            ]
                        ),

                    "lateral_speed_fps":
                        float(
                            state[
                                "lateral_speed_fps"
                            ]
                        ),

                    "altitude_ft":
                        float(
                            state[
                                "altitude_ft"
                            ]
                        ),
                }

        if (
            detailed
            and
            t
            >=
            next_print
        ):
            print(
                f"BASE | "
                f"t={t:6.2f}s | "
                f"FWD={state['forward_ft']:7.2f} | "
                f"V={state['forward_speed_fps']:+5.2f} | "
                f"X={state['cross_track_ft']:+7.2f} | "
                f"LAT={state['lateral_speed_fps']:+5.2f} | "
                f"ALT={state['altitude_ft']:7.2f} | "
                f"A2={row['action2']:+6.3f} | "
                f"AILcmd={row['physical_aileron_cmd']:+8.5f}"
            )

            next_print += 1.5

        reason = safety_reason(
            state
        )

        if reason:
            termination_reason = (
                reason
            )
            break

    final_state = (
        trace[-1]
        if trace
        else {}
    )

    # Important semantics:
    # "persistent at final" means the condition is still being held at
    # termination. "ever held" is reported separately.
    persistent_stop_at_final = bool(
        stop_hold
        >=
        STOP_HOLD_SECONDS
    )

    ever_stop_5s = bool(
        max_stop_hold
        >=
        STOP_HOLD_SECONDS
    )

    persistent_full_hover_at_final = bool(
        full_hover_hold
        >=
        STOP_HOLD_SECONDS
    )

    ever_full_hover_5s = bool(
        max_full_hover_hold
        >=
        STOP_HOLD_SECONDS
    )

    # Select a deterministic authority-test trigger from the actual baseline.
    trigger_row = None

    for row in trace:
        if (
            row[
                "forward_ft"
            ]
            >=
            AUTH_MIN_FORWARD_FT
            and
            abs(
                row[
                    "cross_track_ft"
                ]
            )
            <=
            AUTH_MAX_ABS_CROSS_FT
            and
            abs(
                row[
                    "forward_speed_fps"
                ]
            )
            <=
            AUTH_MAX_FORWARD_SPEED_FPS
            and
            ALT_MIN
            <=
            row[
                "altitude_ft"
            ]
            <=
            ALT_MAX
        ):
            trigger_row = (
                row.copy()
            )
            break

    summary = {
        "same_fdm_object":
            continuity[
                "same_fdm_object"
            ],

        "sim_clock_reset":
            continuity[
                "sim_clock_reset"
            ],

        "termination_reason":
            termination_reason,

        "trace_samples":
            len(
                trace
            ),

        "max_cross_ft":
            float(
                max_cross
            ),

        "min_alt_ft":
            float(
                min_alt
            ),

        "max_alt_ft":
            float(
                max_alt
            ),

        "final_forward_ft":
            float(
                final_state.get(
                    "forward_ft",
                    float("nan"),
                )
            ),

        "final_position_error_ft":
            float(
                final_state.get(
                    "position_error_ft",
                    float("nan"),
                )
            ),

        "final_forward_speed_fps":
            float(
                final_state.get(
                    "forward_speed_fps",
                    float("nan"),
                )
            ),

        "final_cross_ft":
            float(
                final_state.get(
                    "cross_track_ft",
                    float("nan"),
                )
            ),

        "final_lateral_speed_fps":
            float(
                final_state.get(
                    "lateral_speed_fps",
                    float("nan"),
                )
            ),

        "final_alt_ft":
            float(
                final_state.get(
                    "altitude_ft",
                    float("nan"),
                )
            ),

        "final_vs_fps":
            float(
                final_state.get(
                    "vertical_speed_fps",
                    float("nan"),
                )
            ),

        "current_stop_hold_s":
            float(
                stop_hold
            ),

        "max_stop_hold_s":
            float(
                max_stop_hold
            ),

        "persistent_stop_at_final":
            persistent_stop_at_final,

        "ever_stop_5s":
            ever_stop_5s,

        "current_full_hover_hold_s":
            float(
                full_hover_hold
            ),

        "max_full_hover_hold_s":
            float(
                max_full_hover_hold
            ),

        "persistent_full_hover_at_final":
            persistent_full_hover_at_final,

        "ever_full_hover_5s":
            ever_full_hover_5s,

        "threshold_events":
            threshold_events,

        "authority_trigger":
            trigger_row,
    }

    env2.fdm = None
    env1.close()

    return (
        summary,
        trace,
    )


# =====================================================================
# PHASE B — LOW-SPEED AILERON AUTHORITY IDENTIFICATION
# =====================================================================

def run_authority_case(
    action_delta,
    trigger,
    detailed=False,
):
    (
        env1,
        env2,
        fdm,
        obs2,
        lat0,
        lon0,
        mission_heading,
        continuity,
    ) = build_start(
        detailed=False,
    )

    dt = env_control_dt(
        env2
    )

    pulse_steps = max(
        1,
        int(
            round(
                PULSE_SECONDS
                /
                dt
            )
        ),
    )

    recovery_steps = max(
        1,
        int(
            round(
                RECOVERY_SECONDS
                /
                dt
            )
        ),
    )

    trace = []

    reached_trigger = False
    safe = True
    termination_reason = ""

    state_at_trigger = None
    state_after_pulse = None
    state_after_recovery = None

    trigger_forward = float(
        trigger[
            "forward_ft"
        ]
    )

    trigger_speed = abs(
        float(
            trigger[
                "forward_speed_fps"
            ]
        )
    )

    # A small tolerance makes the deterministic trigger robust to
    # sub-step differences while preserving the same physical region.
    trigger_speed_limit = (
        trigger_speed
        +
        0.20
    )

    next_print = 0.0

    for step in range(
        int(
            STAGE3_MAX_TIME
            /
            dt
        )
    ):
        state_before = snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        if (
            state_before[
                "forward_ft"
            ]
            >=
            trigger_forward
            and
            abs(
                state_before[
                    "forward_speed_fps"
                ]
            )
            <=
            trigger_speed_limit
        ):
            reached_trigger = True
            state_at_trigger = (
                state_before.copy()
            )
            break

        (
            obs2,
            state,
            row,
        ) = stage3_step(
            env2,
            obs2,
            fdm,
            lat0,
            lon0,
            mission_heading,
            CRUISE_AILERON_ACTION,
        )

        t = (
            step + 1
        ) * dt

        row[
            "time_s"
        ] = float(
            t
        )

        row[
            "phase"
        ] = "approach"

        trace.append(
            row
        )

        reason = safety_reason(
            state
        )

        if reason:
            safe = False
            termination_reason = (
                reason
            )
            break

    if (
        reached_trigger
        and
        safe
    ):
        pulse_aileron = float(
            np.clip(
                CRUISE_AILERON_ACTION
                +
                action_delta,
                -1.0,
                +1.0,
            )
        )

        for pulse_index in range(
            pulse_steps
        ):
            (
                obs2,
                state,
                row,
            ) = stage3_step(
                env2,
                obs2,
                fdm,
                lat0,
                lon0,
                mission_heading,
                pulse_aileron,
            )

            row[
                "time_s"
            ] = float(
                len(
                    trace
                )
                *
                dt
            )

            row[
                "phase"
            ] = "pulse"

            trace.append(
                row
            )

            reason = safety_reason(
                state
            )

            if reason:
                safe = False
                termination_reason = (
                    reason
                )
                break

        if (
            safe
            and
            trace
        ):
            state_after_pulse = (
                trace[-1].copy()
            )

        if safe:
            for _ in range(
                recovery_steps
            ):
                (
                    obs2,
                    state,
                    row,
                ) = stage3_step(
                    env2,
                    obs2,
                    fdm,
                    lat0,
                    lon0,
                    mission_heading,
                    CRUISE_AILERON_ACTION,
                )

                row[
                    "time_s"
                ] = float(
                    len(
                        trace
                    )
                    *
                    dt
                )

                row[
                    "phase"
                ] = "recovery"

                trace.append(
                    row
                )

                reason = safety_reason(
                    state
                )

                if reason:
                    safe = False
                    termination_reason = (
                        reason
                    )
                    break

        if (
            trace
            and
            reached_trigger
        ):
            state_after_recovery = (
                trace[-1].copy()
            )

    if (
        not reached_trigger
        and
        not termination_reason
    ):
        termination_reason = (
            "authority_trigger_not_reached"
        )

    def val(
        state,
        key,
    ):
        if state is None:
            return float(
                "nan"
            )

        return float(
            state[
                key
            ]
        )

    result = {
        "action2_delta":
            float(
                action_delta
            ),

        "action2_pulse":
            float(
                np.clip(
                    CRUISE_AILERON_ACTION
                    +
                    action_delta,
                    -1.0,
                    +1.0,
                )
            ),

        "mapped_physical_delta_nominal":
            float(
                AILERON_SCALE
                *
                action_delta
            ),

        "reached_trigger":
            bool(
                reached_trigger
            ),

        "safe":
            bool(
                safe
            ),

        "termination_reason":
            termination_reason,

        "start_forward_ft":
            val(
                state_at_trigger,
                "forward_ft",
            ),

        "start_cross_ft":
            val(
                state_at_trigger,
                "cross_track_ft",
            ),

        "start_lateral_speed_fps":
            val(
                state_at_trigger,
                "lateral_speed_fps",
            ),

        "start_roll_deg":
            val(
                state_at_trigger,
                "roll_deg",
            ),

        "start_alt_ft":
            val(
                state_at_trigger,
                "altitude_ft",
            ),

        "pulse_end_cross_ft":
            val(
                state_after_pulse,
                "cross_track_ft",
            ),

        "pulse_end_lateral_speed_fps":
            val(
                state_after_pulse,
                "lateral_speed_fps",
            ),

        "pulse_end_roll_deg":
            val(
                state_after_pulse,
                "roll_deg",
            ),

        "pulse_end_alt_ft":
            val(
                state_after_pulse,
                "altitude_ft",
            ),

        "recovery_cross_ft":
            val(
                state_after_recovery,
                "cross_track_ft",
            ),

        "recovery_lateral_speed_fps":
            val(
                state_after_recovery,
                "lateral_speed_fps",
            ),

        "recovery_roll_deg":
            val(
                state_after_recovery,
                "roll_deg",
            ),

        "recovery_alt_ft":
            val(
                state_after_recovery,
                "altitude_ft",
            ),

        "delta_lat_speed_pulse":
            (
                val(
                    state_after_pulse,
                    "lateral_speed_fps",
                )
                -
                val(
                    state_at_trigger,
                    "lateral_speed_fps",
                )
            ),

        "delta_cross_pulse":
            (
                val(
                    state_after_pulse,
                    "cross_track_ft",
                )
                -
                val(
                    state_at_trigger,
                    "cross_track_ft",
                )
            ),

        "actual_aileron_cmd_start":
            val(
                state_at_trigger,
                "physical_aileron_cmd",
            ),

        "actual_aileron_cmd_pulse_end":
            val(
                state_after_pulse,
                "physical_aileron_cmd",
            ),

        "actual_aileron_cmd_delta":
            (
                val(
                    state_after_pulse,
                    "physical_aileron_cmd",
                )
                -
                val(
                    state_at_trigger,
                    "physical_aileron_cmd",
                )
            ),

        "max_abs_cross_during_case":
            float(
                max(
                    (
                        abs(
                            row[
                                "cross_track_ft"
                            ]
                        )
                        for row in trace
                    ),
                    default=float(
                        "nan"
                    ),
                )
            ),
    }

    if detailed:
        print(
            "Trigger:",
            state_at_trigger,
        )

        print(
            "Pulse end:",
            state_after_pulse,
        )

        print(
            "Recovery:",
            state_after_recovery,
        )

    env2.fdm = None
    env1.close()

    return (
        result,
        trace,
    )


# =====================================================================
# MAIN
# =====================================================================

rule(
    "STAGE 3 — ENDPOINT LATERAL DIAGNOSTIC V2"
)

print(
    "Stage 1 model:",
    STAGE1_MODEL_PATH,
)

print(
    "Stage 2 model:",
    STAGE2_MODEL_PATH,
)

print()
print(
    "Training                         : NONE"
)
print(
    "Stage-1 runtime teacher          : OFF"
)
print(
    "Stage-2 runtime teacher          : OFF"
)
print(
    "Stage-1 -> Stage-2 FDM reset     : FORBIDDEN / VERIFIED"
)
print(
    "Stage-2 -> Stage-3 FDM reset     : NONE"
)
print(
    "Stage-3 lateral feedback         : OFF in baseline"
)
print(
    "Only aileron pulse varies in Phase B."
)

rule(
    "PHASE A — BASELINE QUALIFICATION"
)

baseline_summary, baseline_trace = (
    run_baseline(
        detailed=True,
    )
)

write_rows(
    BASELINE_TRACE_CSV,
    baseline_trace,
)

print()
print(
    "BASELINE RESULT"
)

for key in [
    "same_fdm_object",
    "sim_clock_reset",
    "termination_reason",
    "max_cross_ft",
    "min_alt_ft",
    "max_alt_ft",
    "final_forward_ft",
    "final_position_error_ft",
    "final_forward_speed_fps",
    "final_cross_ft",
    "final_lateral_speed_fps",
    "final_alt_ft",
    "final_vs_fps",
    "current_stop_hold_s",
    "max_stop_hold_s",
    "persistent_stop_at_final",
    "ever_stop_5s",
    "current_full_hover_hold_s",
    "max_full_hover_hold_s",
    "persistent_full_hover_at_final",
]:
    print(
        f"{key:34s}: "
        f"{baseline_summary[key]}"
    )

print()
print(
    "Cross-track threshold events:"
)

for key, value in (
    baseline_summary[
        "threshold_events"
    ].items()
):
    print(
        f"  {key:12s}: "
        f"{value}"
    )

trigger = (
    baseline_summary[
        "authority_trigger"
    ]
)

print()
print(
    "Selected authority-test trigger:"
)
print(
    trigger
)

if trigger is None:
    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "baseline":
                    baseline_summary,

                "authority":
                    [],
            },
            f,
            indent=2,
        )

    raise RuntimeError(
        "No safe low-speed authority-test trigger was found. "
        "Do NOT guess a lateral gain. Inspect baseline_trace.csv."
    )


rule(
    "PHASE B — LOW-SPEED AILERON AUTHORITY IDENTIFICATION"
)

authority_results = []
authority_traces = {}

for case_no, action_delta in enumerate(
    AILERON_ACTION_DELTAS,
    start=1,
):
    result, trace = (
        run_authority_case(
            action_delta,
            trigger,
            detailed=False,
        )
    )

    authority_results.append(
        result
    )

    authority_traces[
        float(
            action_delta
        )
    ] = trace

    print(
        f"A{case_no:02d} | "
        f"dA2={action_delta:+.2f} | "
        f"A2={result['action2_pulse']:+.3f} | "
        f"REACH={result['reached_trigger']} | "
        f"SAFE={result['safe']} | "
        f"dLAT={result['delta_lat_speed_pulse']:+.4f} | "
        f"dX={result['delta_cross_pulse']:+.4f} | "
        f"ROLL={result['pulse_end_roll_deg']:+.3f}deg | "
        f"AILcmdΔ={result['actual_aileron_cmd_delta']:+.6f} | "
        f"ALT={result['pulse_end_alt_ft']:.3f}"
    )

write_rows(
    AUTHORITY_SUMMARY_CSV,
    authority_results,
)


# =====================================================================
# AUTHORITY QUALITY CHECK
# =====================================================================

valid_nonzero = [
    r
    for r in authority_results
    if (
        r[
            "reached_trigger"
        ]
        and
        r[
            "safe"
        ]
        and
        abs(
            r[
                "action2_delta"
            ]
        )
        >
        1e-9
        and
        np.isfinite(
            r[
                "delta_lat_speed_pulse"
            ]
        )
        and
        np.isfinite(
            r[
                "actual_aileron_cmd_delta"
            ]
        )
    )
]

mapping_alive = bool(
    valid_nonzero
    and
    max(
        abs(
            r[
                "actual_aileron_cmd_delta"
            ]
        )
        for r in valid_nonzero
    )
    >
    1e-5
)

lateral_authority_alive = bool(
    valid_nonzero
    and
    max(
        abs(
            r[
                "delta_lat_speed_pulse"
            ]
        )
        for r in valid_nonzero
    )
    >
    0.02
)

# A useful sign test: compare the most positive and most negative
# action-delta cases.
negative_cases = sorted(
    [
        r
        for r in valid_nonzero
        if r[
            "action2_delta"
        ]
        <
        0.0
    ],
    key=lambda r:
        r[
            "action2_delta"
        ],
)

positive_cases = sorted(
    [
        r
        for r in valid_nonzero
        if r[
            "action2_delta"
        ]
        >
        0.0
    ],
    key=lambda r:
        r[
            "action2_delta"
        ],
)

sign_separation = float(
    "nan"
)

if (
    negative_cases
    and
    positive_cases
):
    most_negative = (
        negative_cases[0]
    )

    most_positive = (
        positive_cases[-1]
    )

    sign_separation = float(
        most_positive[
            "delta_lat_speed_pulse"
        ]
        -
        most_negative[
            "delta_lat_speed_pulse"
        ]
    )


# Choose a trace only for evidence viewing; this is NOT a controller
# selection. Prefer the largest safe absolute response.
if valid_nonzero:
    evidence_case = max(
        valid_nonzero,
        key=lambda r:
            abs(
                r[
                    "delta_lat_speed_pulse"
                ]
            ),
    )

    evidence_trace = (
        authority_traces[
            float(
                evidence_case[
                    "action2_delta"
                ]
            )
        ]
    )

    write_rows(
        AUTHORITY_BEST_TRACE_CSV,
        evidence_trace,
    )

else:
    evidence_case = None


diagnostic_summary = {
    "models": {
        "stage1":
            str(
                STAGE1_MODEL_PATH
            ),

        "stage2":
            str(
                STAGE2_MODEL_PATH
            ),
    },

    "controller_constants": {
        "K_POS":
            K_POS,

        "V_FWD_MAX":
            V_FWD_MAX,

        "BRAKE_LEAD_FT":
            BRAKE_LEAD_FT,

        "KV":
            KV,

        "COLLECTIVE_BIAS":
            COLLECTIVE_BIAS,

        "ALT_KP":
            ALT_KP,

        "VS_KD":
            VS_KD,

        "CRUISE_AILERON_ACTION":
            CRUISE_AILERON_ACTION,
    },

    "baseline":
        baseline_summary,

    "authority_test": {
        "trigger":
            trigger,

        "pulse_seconds":
            PULSE_SECONDS,

        "recovery_seconds":
            RECOVERY_SECONDS,

        "mapping_alive":
            mapping_alive,

        "lateral_authority_alive":
            lateral_authority_alive,

        "sign_separation":
            sign_separation,

        "cases":
            authority_results,

        "evidence_case":
            evidence_case,
    },
}

with SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        diagnostic_summary,
        f,
        indent=2,
    )


rule(
    "STAGE 3 LATERAL DIAGNOSTIC V2 — FINAL"
)

print(
    "Mapped action[2] -> physical aileron path alive :",
    mapping_alive,
)

print(
    "Low-speed lateral authority measurable          :",
    lateral_authority_alive,
)

print(
    "Positive-vs-negative lateral response separation:",
    sign_separation,
)

print()
print(
    "Baseline persistent stop AT FINAL:",
    baseline_summary[
        "persistent_stop_at_final"
    ],
)

print(
    "Baseline ever held stop for 5 s    :",
    baseline_summary[
        "ever_stop_5s"
    ],
)

print(
    "Baseline termination reason         :",
    baseline_summary[
        "termination_reason"
    ],
)

print()
print(
    "Saved baseline trace :",
    BASELINE_TRACE_CSV,
)

print(
    "Saved authority sweep:",
    AUTHORITY_SUMMARY_CSV,
)

print(
    "Saved evidence trace :",
    AUTHORITY_BEST_TRACE_CSV,
)

print(
    "Saved summary        :",
    SUMMARY_JSON,
)

print()

if (
    mapping_alive
    and
    lateral_authority_alive
):
    print(
        "DIAGNOSTIC PASS: actuator mapping and low-speed lateral authority "
        "are both demonstrated."
    )

    print(
        "NEXT: use this measured sign/authority and baseline threshold "
        "events to design the Stage-3 lateral feedback calibration."
    )

else:
    print(
        "DIAGNOSTIC NOT PASSED."
    )

    print(
        "Do NOT run a lateral gain sweep until the failed diagnostic item "
        "is resolved."
    )
