from __future__ import annotations

"""
AH-1S / JSBSim
STAGE 3 — DATA-GROUNDED ENDPOINT LATERAL TEACHER CALIBRATION V3
===============================================================

This script is intentionally a controller-calibration experiment, NOT PPO
training.

Evidence chain
--------------
1) Stage 1 neural policy is fixed.
2) Stage 2 HYBRID FINAL neural policy is fixed.
3) The same JSBSim FDM object is carried continuously into Stage 3.
4) Existing Stage-3 longitudinal and altitude controllers are kept unchanged.
5) The lateral activation point and local neutral aileron action are derived
   from the saved V2 diagnostic outputs, not guessed.
6) Only the lateral feedback parameters (trim offset, Kp, Kd) are swept.
7) Every candidate starts from a fresh deterministic continuous
   Stage1 -> Stage2 -> Stage3 trajectory.
8) A final detailed repeat is required before the teacher can be called locked.

No model is modified.
No PPO training is performed.
No runtime "success" flag is trusted by itself; geometry and hold time are
validated explicitly.
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
# REQUIRED DIAGNOSTIC EVIDENCE
# =====================================================================

DIAGNOSTIC_DIR = Path(
    "results_stage3_lateral_diagnostic_v2"
)

DIAGNOSTIC_SUMMARY_JSON = (
    DIAGNOSTIC_DIR
    /
    "diagnostic_summary.json"
)

AUTHORITY_SWEEP_CSV = (
    DIAGNOSTIC_DIR
    /
    "aileron_authority_sweep.csv"
)


# =====================================================================
# OUTPUT
# =====================================================================

OUT_DIR = Path(
    "results_stage3_lateral_teacher_v3"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SWEEP_CSV = (
    OUT_DIR
    /
    "lateral_teacher_sweep.csv"
)

BEST_TRACE_CSV = (
    OUT_DIR
    /
    "best_teacher_trace.csv"
)

FINAL_SUMMARY_JSON = (
    OUT_DIR
    /
    "final_summary.json"
)


# =====================================================================
# MISSION / UNCHANGED STAGE-3 LONGITUDINAL + ALTITUDE CONTROLLER
# =====================================================================

TARGET_FORWARD_FT = 300.0
TARGET_ALT_FT = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
STAGE3_MAX_TIME = 90.0

HANDOFF_STABLE_TIME = 5.0
DEFAULT_CONTROL_DT = 0.075

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040

# Stage-3 longitudinal / altitude logic begins after 80 ft forward.
TEACHER_ENABLE_FORWARD_FT = 80.0

# Existing longitudinal controller — intentionally unchanged.
K_POS = 0.055
V_FWD_MAX = 9.0
BRAKE_LEAD_FT = 8.0
V_REV_MAX = 2.0
KV = 1.00
ELEVATOR_TRIM_ACTION = 0.013725
Q_DAMP = 2.0
MAX_ELEVATOR_RESIDUAL = 1.0

# Existing altitude controller — intentionally unchanged.
COLLECTIVE_BIAS = 0.22
ALT_KP = 0.030
VS_KD = 0.120
MAX_ALT_CORR = 0.45

# Stage-2 lateral/yaw command before the data-derived endpoint phase.
CRUISE_AILERON_ACTION = -0.230
RUDDER_ACTION = 0.0


# =====================================================================
# ACCEPTANCE / PRESENTATION / SAFETY
# =====================================================================

STOP_POS_TOL_FT = 5.0
STOP_SPEED_TOL_FPS = 0.60
STOP_HOLD_SECONDS = 5.0

CROSS_TOL_FT = 5.0
LAT_SPEED_TOL_FPS = 0.60
LATERAL_HOLD_SECONDS = 5.0

ALT_MIN = 295.0
ALT_MAX = 305.0
VS_TOL_FPS = 0.75
HEADING_TOL_DEG = 1.0

# Stronger Stage-3 presentation corridor:
# the whole Stage-3 trajectory should stay inside it.
PRESENTATION_MAX_CROSS_FT = 5.0
PRESENTATION_ALT_MIN = 295.0
PRESENTATION_ALT_MAX = 305.0

# Safety cutoffs remain wider than presentation criteria.
ALT_SAFE_MIN = 288.0
ALT_SAFE_MAX = 312.0
MAX_ABS_PITCH_DEG = 8.0
MAX_ABS_ROLL_DEG = 10.0
MAX_CROSS_SAFE_FT = 15.0

EARTH_RADIUS_FT = 20_902_231.0


# =====================================================================
# LATERAL SEARCH DESIGN
#
# Activation and trim centre are loaded from the diagnostic results.
# Only these small offsets and gain grids are searched.
# =====================================================================

TRIM_OFFSETS = [
    -0.03,
    0.00,
    +0.03,
]

LATERAL_KP_VALUES = [
    0.015,
    0.025,
    0.035,
    0.045,
]

LATERAL_KD_VALUES = [
    0.10,
    0.18,
    0.26,
]


# =====================================================================
# HELPERS
# =====================================================================

def rule(text):
    print()
    print("=" * 154)
    print(text)
    print("=" * 154)


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
    if getattr(
        env,
        "fdm",
        None,
    ) is not None:
        return env.fdm

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
        ) is not None
    ):
        return base.fdm

    raise RuntimeError(
        "Active FDM not found."
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
    north, east = local_ne_ft(
        latitude_deg(fdm),
        longitude_deg(fdm),
        lat0,
        lon0,
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
        ],
    )

    ve = first_finite(
        fdm,
        [
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


def physical_commands(fdm):
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

    heading_error = wrap_angle(
        heading_rad(fdm)
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

        # Existing controller coordinates.
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

        # Logged only as independent geometry evidence.
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


def safety_reason(state):
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
            state["pitch_deg"]
        )
        >
        MAX_ABS_PITCH_DEG
    ):
        return "pitch_limit"

    if (
        abs(
            state["roll_deg"]
        )
        >
        MAX_ABS_ROLL_DEG
    ):
        return "roll_limit"

    if (
        abs(
            state["cross_track_ft"]
        )
        >
        MAX_CROSS_SAFE_FT
    ):
        return "cross_track_safety_limit"

    return ""


def endpoint_stop_now(state):
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


def endpoint_lateral_now(state):
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


def endpoint_hover_now(state):
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


def env_control_dt(env):
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


def physics_steps(env):
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

    # env2 is the mapped Stage-2 environment; therefore this applies the
    # repaired action[2]/action[3] actuator path before JSBSim integration.
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
                "JSBSim stopped during Stage-3 lateral calibration."
            )

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
# LOAD AND VERIFY V2 DIAGNOSTIC EVIDENCE
# =====================================================================

for path in [
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
    DIAGNOSTIC_SUMMARY_JSON,
    AUTHORITY_SWEEP_CSV,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input missing: {path}"
        )


with DIAGNOSTIC_SUMMARY_JSON.open(
    "r",
    encoding="utf-8",
) as f:
    diagnostic = json.load(
        f
    )


authority_block = diagnostic.get(
    "authority_test",
    {},
)

if not bool(
    authority_block.get(
        "mapping_alive",
        False,
    )
):
    raise RuntimeError(
        "V2 diagnostic did not prove the mapped aileron path."
    )

if not bool(
    authority_block.get(
        "lateral_authority_alive",
        False,
    )
):
    raise RuntimeError(
        "V2 diagnostic did not prove low-speed lateral authority."
    )


trigger = authority_block.get(
    "trigger",
    None,
)

if not trigger:
    raise RuntimeError(
        "V2 diagnostic does not contain a valid authority trigger."
    )


authority_rows = []

with AUTHORITY_SWEEP_CSV.open(
    "r",
    newline="",
    encoding="utf-8",
) as f:
    reader = csv.DictReader(
        f
    )

    for row in reader:
        try:
            reached = (
                str(
                    row[
                        "reached_trigger"
                    ]
                ).strip().lower()
                ==
                "true"
            )

            safe = (
                str(
                    row[
                        "safe"
                    ]
                ).strip().lower()
                ==
                "true"
            )

            action2 = float(
                row[
                    "action2_pulse"
                ]
            )

            delta_lat = float(
                row[
                    "delta_lat_speed_pulse"
                ]
            )
        except Exception:
            continue

        if (
            reached
            and
            safe
            and
            np.isfinite(
                action2
            )
            and
            np.isfinite(
                delta_lat
            )
        ):
            authority_rows.append(
                (
                    action2,
                    delta_lat,
                )
            )


if len(
    authority_rows
) < 3:
    raise RuntimeError(
        "Not enough valid V2 authority cases for a neutral-action fit."
    )


authority_actions = np.asarray(
    [
        item[0]
        for item in authority_rows
    ],
    dtype=np.float64,
)

authority_delta_lat = np.asarray(
    [
        item[1]
        for item in authority_rows
    ],
    dtype=np.float64,
)


# Local linear fit at the diagnosed low-speed operating point:
#
#   Δv_lat(2 s) ≈ slope * action2 + intercept
#
# This is NOT claimed to be a global helicopter model. It is only used
# to center the small endpoint trim search.
slope, intercept = np.polyfit(
    authority_actions,
    authority_delta_lat,
    1,
)

if (
    not np.isfinite(
        slope
    )
    or
    abs(
        slope
    )
    <
    1e-6
):
    raise RuntimeError(
        "Authority fit slope is invalid."
    )


IDENTIFIED_NEUTRAL_A2 = float(
    -intercept
    /
    slope
)

IDENTIFIED_A2_MIN = float(
    np.min(
        authority_actions
    )
)

IDENTIFIED_A2_MAX = float(
    np.max(
        authority_actions
    )
)

# Lateral endpoint phase begins at the actual safe/low-speed operating
# point selected by the V2 diagnostic. This is data-derived, not guessed.
LATERAL_ENABLE_FORWARD_FT = float(
    trigger[
        "forward_ft"
    ]
)

DIAGNOSTIC_ENABLE_SPEED_FPS = float(
    abs(
        trigger[
            "forward_speed_fps"
        ]
    )
)

# Since the trajectory is deterministic, forward position is the actual
# phase trigger. The measured speed is retained as a qualification check.
LATERAL_ACTIVATION_MAX_SPEED_FPS = (
    DIAGNOSTIC_ENABLE_SPEED_FPS
    +
    0.25
)


HOVER_TRIMS = [
    float(
        np.clip(
            IDENTIFIED_NEUTRAL_A2
            +
            offset,
            IDENTIFIED_A2_MIN,
            IDENTIFIED_A2_MAX,
        )
    )
    for offset in TRIM_OFFSETS
]


# =====================================================================
# MODELS
# =====================================================================

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

def build_start():
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

    for _ in range(
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

    # Disposable reset initializes only the Python-side Stage-2 object.
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

    same_fdm = bool(
        id(
            get_fdm(
                env2
            )
        )
        ==
        active_id
    )

    if not same_fdm:
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

    sim_clock_reset = bool(
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
    )

    if sim_clock_reset:
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

    for _ in range(
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

        forward, _cross = geometry(
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

    obs2 = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    return (
        env1,
        env2,
        fdm,
        obs2,
        lat0,
        lon0,
        mission_heading,
        same_fdm,
        sim_clock_reset,
    )


# =====================================================================
# STAGE-3 ACTION
# =====================================================================

def build_stage3_action(
    obs,
    state,
    lateral_mode_active,
    hover_trim,
    lateral_kp,
    lateral_kd,
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

    # ---------------------------------------------------------
    # UNCHANGED altitude controller
    # ---------------------------------------------------------

    altitude_error = (
        TARGET_ALT_FT
        -
        state[
            "altitude_ft"
        ]
    )

    col_corr = (
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

    col_corr = float(
        np.clip(
            col_corr,
            -MAX_ALT_CORR,
            +MAX_ALT_CORR,
        )
    )

    action[0] = float(
        np.clip(
            base_action[0]
            +
            col_corr,
            -1.0,
            +1.0,
        )
    )

    # ---------------------------------------------------------
    # UNCHANGED longitudinal controller
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # ONLY calibrated channel: lateral action[2]
    # ---------------------------------------------------------

    if lateral_mode_active:
        lateral_corr = (
            -lateral_kp
            *
            state[
                "cross_track_ft"
            ]
            -
            lateral_kd
            *
            state[
                "lateral_speed_fps"
            ]
        )

        raw_aileron = (
            hover_trim
            +
            lateral_corr
        )

        # Stay inside the authority envelope that was physically tested
        # in V2. If more authority is needed later, that requires a new
        # authority experiment rather than silently expanding commands.
        action[2] = float(
            np.clip(
                raw_aileron,
                IDENTIFIED_A2_MIN,
                IDENTIFIED_A2_MAX,
            )
        )

    else:
        lateral_corr = 0.0
        raw_aileron = (
            CRUISE_AILERON_ACTION
        )
        action[2] = (
            CRUISE_AILERON_ACTION
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
            col_corr
        ),
        float(
            elevator_residual
        ),
        float(
            lateral_corr
        ),
        float(
            raw_aileron
        ),
    )


# =====================================================================
# ONE CANDIDATE
# =====================================================================

def run_case(
    hover_trim,
    lateral_kp,
    lateral_kd,
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
        same_fdm,
        sim_clock_reset,
    ) = build_start()

    dt = env_control_dt(
        env2
    )

    trace = []

    safe = True
    termination_reason = (
        "stage3_time_limit"
    )

    lateral_mode_active = False
    activation_state = None

    stop_hold = 0.0
    lateral_hold = 0.0
    full_hover_hold = 0.0

    max_stop_hold = 0.0
    max_lateral_hold = 0.0
    max_full_hover_hold = 0.0

    max_cross = 0.0
    min_alt = +999.0
    max_alt = -999.0

    max_abs_pitch = 0.0
    max_abs_roll = 0.0
    max_abs_heading = 0.0

    min_action2 = +999.0
    max_action2 = -999.0

    next_print = 0.0

    endpoint_hover_5s = False

    final_state = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

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

        # One-way mission phase transition. Once endpoint lateral
        # stabilization begins, it stays active; no gate chatter.
        if (
            not lateral_mode_active
            and
            state_before[
                "forward_ft"
            ]
            >=
            LATERAL_ENABLE_FORWARD_FT
        ):
            if (
                abs(
                    state_before[
                        "forward_speed_fps"
                    ]
                )
                >
                LATERAL_ACTIVATION_MAX_SPEED_FPS
            ):
                safe = False
                termination_reason = (
                    "activation_speed_not_qualified"
                )
                final_state = (
                    state_before.copy()
                )
                break

            lateral_mode_active = True
            activation_state = (
                state_before.copy()
            )

        (
            action,
            v_des,
            col_corr,
            elevator_residual,
            lateral_corr,
            raw_aileron,
        ) = build_stage3_action(
            obs2,
            state_before,
            lateral_mode_active,
            hover_trim,
            lateral_kp,
            lateral_kd,
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

        t = (
            step + 1
        ) * dt

        stop_now = endpoint_stop_now(
            state_after
        )

        lateral_now = endpoint_lateral_now(
            state_after
        )

        hover_now = endpoint_hover_now(
            state_after
        )

        stop_hold = (
            stop_hold
            +
            dt
            if stop_now
            else 0.0
        )

        lateral_hold = (
            lateral_hold
            +
            dt
            if lateral_now
            else 0.0
        )

        full_hover_hold = (
            full_hover_hold
            +
            dt
            if hover_now
            else 0.0
        )

        max_stop_hold = max(
            max_stop_hold,
            stop_hold,
        )

        max_lateral_hold = max(
            max_lateral_hold,
            lateral_hold,
        )

        max_full_hover_hold = max(
            max_full_hover_hold,
            full_hover_hold,
        )

        max_cross = max(
            max_cross,
            abs(
                state_after[
                    "cross_track_ft"
                ]
            ),
        )

        min_alt = min(
            min_alt,
            state_after[
                "altitude_ft"
            ],
        )

        max_alt = max(
            max_alt,
            state_after[
                "altitude_ft"
            ],
        )

        max_abs_pitch = max(
            max_abs_pitch,
            abs(
                state_after[
                    "pitch_deg"
                ]
            ),
        )

        max_abs_roll = max(
            max_abs_roll,
            abs(
                state_after[
                    "roll_deg"
                ]
            ),
        )

        max_abs_heading = max(
            max_abs_heading,
            abs(
                state_after[
                    "heading_error_deg"
                ]
            ),
        )

        min_action2 = min(
            min_action2,
            float(
                action[2]
            ),
        )

        max_action2 = max(
            max_action2,
            float(
                action[2]
            ),
        )

        row = dict(
            state_after
        )

        row.update(
            {
                "time_s":
                    float(
                        t
                    ),

                "lateral_mode_active":
                    bool(
                        lateral_mode_active
                    ),

                "stop_now":
                    bool(
                        stop_now
                    ),

                "lateral_now":
                    bool(
                        lateral_now
                    ),

                "hover_now":
                    bool(
                        hover_now
                    ),

                "stop_hold_s":
                    float(
                        stop_hold
                    ),

                "lateral_hold_s":
                    float(
                        lateral_hold
                    ),

                "full_hover_hold_s":
                    float(
                        full_hover_hold
                    ),

                "v_des_fps":
                    float(
                        v_des
                    ),

                "collective_correction":
                    float(
                        col_corr
                    ),

                "elevator_residual":
                    float(
                        elevator_residual
                    ),

                "lateral_correction":
                    float(
                        lateral_corr
                    ),

                "raw_aileron_action":
                    float(
                        raw_aileron
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

        trace.append(
            row
        )

        final_state = (
            state_after.copy()
        )

        reason = safety_reason(
            state_after
        )

        if reason:
            safe = False
            termination_reason = (
                reason
            )
            break

        if (
            full_hover_hold
            >=
            STOP_HOLD_SECONDS
        ):
            endpoint_hover_5s = True
            termination_reason = (
                "endpoint_hover_5s"
            )
            break

        if (
            detailed
            and
            t
            >=
            next_print
        ):
            print(
                f"t={t:6.2f}s | "
                f"FWD={state_after['forward_ft']:7.2f} | "
                f"V={state_after['forward_speed_fps']:+5.2f} | "
                f"X={state_after['cross_track_ft']:+6.2f} | "
                f"LAT={state_after['lateral_speed_fps']:+5.2f} | "
                f"ALT={state_after['altitude_ft']:7.2f} | "
                f"VS={state_after['vertical_speed_fps']:+5.2f} | "
                f"A2={action[2]:+6.3f} | "
                f"AILcmd={state_after['physical_aileron_cmd']:+8.5f} | "
                f"LAT_ON={lateral_mode_active} | "
                f"HOLD={full_hover_hold:4.2f}s"
            )

            next_print += 1.5

    presentation_cross_pass = bool(
        max_cross
        <=
        PRESENTATION_MAX_CROSS_FT
    )

    presentation_altitude_pass = bool(
        min_alt
        >=
        PRESENTATION_ALT_MIN
        and
        max_alt
        <=
        PRESENTATION_ALT_MAX
    )

    persistent_stop_at_final = bool(
        stop_hold
        >=
        STOP_HOLD_SECONDS
    )

    persistent_lateral_at_final = bool(
        lateral_hold
        >=
        LATERAL_HOLD_SECONDS
    )

    persistent_hover_at_final = bool(
        full_hover_hold
        >=
        STOP_HOLD_SECONDS
    )

    teacher_pass = bool(
        same_fdm
        and
        not sim_clock_reset
        and
        safe
        and
        endpoint_hover_5s
        and
        persistent_stop_at_final
        and
        persistent_lateral_at_final
        and
        persistent_hover_at_final
        and
        presentation_cross_pass
        and
        presentation_altitude_pass
    )

    result = {
        "hover_trim":
            float(
                hover_trim
            ),

        "lateral_kp":
            float(
                lateral_kp
            ),

        "lateral_kd":
            float(
                lateral_kd
            ),

        "same_fdm_object":
            bool(
                same_fdm
            ),

        "sim_clock_reset":
            bool(
                sim_clock_reset
            ),

        "safe":
            bool(
                safe
            ),

        "termination_reason":
            termination_reason,

        "lateral_activated":
            bool(
                activation_state
                is not None
            ),

        "activation_forward_ft":
            float(
                activation_state[
                    "forward_ft"
                ]
            )
            if activation_state
            is not None
            else float(
                "nan"
            ),

        "activation_forward_speed_fps":
            float(
                activation_state[
                    "forward_speed_fps"
                ]
            )
            if activation_state
            is not None
            else float(
                "nan"
            ),

        "activation_cross_ft":
            float(
                activation_state[
                    "cross_track_ft"
                ]
            )
            if activation_state
            is not None
            else float(
                "nan"
            ),

        "endpoint_hover_5s":
            bool(
                endpoint_hover_5s
            ),

        "persistent_stop_at_final":
            persistent_stop_at_final,

        "persistent_lateral_at_final":
            persistent_lateral_at_final,

        "persistent_hover_at_final":
            persistent_hover_at_final,

        "current_stop_hold_s":
            float(
                stop_hold
            ),

        "current_lateral_hold_s":
            float(
                lateral_hold
            ),

        "current_full_hover_hold_s":
            float(
                full_hover_hold
            ),

        "max_stop_hold_s":
            float(
                max_stop_hold
            ),

        "max_lateral_hold_s":
            float(
                max_lateral_hold
            ),

        "max_full_hover_hold_s":
            float(
                max_full_hover_hold
            ),

        "presentation_cross_pass":
            presentation_cross_pass,

        "presentation_altitude_pass":
            presentation_altitude_pass,

        "teacher_pass":
            teacher_pass,

        "final_forward_ft":
            float(
                final_state[
                    "forward_ft"
                ]
            ),

        "final_position_error_ft":
            float(
                final_state[
                    "position_error_ft"
                ]
            ),

        "final_forward_speed_fps":
            float(
                final_state[
                    "forward_speed_fps"
                ]
            ),

        "final_cross_ft":
            float(
                final_state[
                    "cross_track_ft"
                ]
            ),

        "final_lateral_speed_fps":
            float(
                final_state[
                    "lateral_speed_fps"
                ]
            ),

        "final_alt_ft":
            float(
                final_state[
                    "altitude_ft"
                ]
            ),

        "final_vs_fps":
            float(
                final_state[
                    "vertical_speed_fps"
                ]
            ),

        "final_heading_error_deg":
            float(
                final_state[
                    "heading_error_deg"
                ]
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

        "max_abs_pitch_deg":
            float(
                max_abs_pitch
            ),

        "max_abs_roll_deg":
            float(
                max_abs_roll
            ),

        "max_abs_heading_error_deg":
            float(
                max_abs_heading
            ),

        "action2_min":
            float(
                min_action2
            ),

        "action2_max":
            float(
                max_action2
            ),

        "trace":
            trace,
    }

    env2.fdm = None
    env1.close()

    return result


# =====================================================================
# PRINT IDENTIFICATION EVIDENCE
# =====================================================================

rule(
    "STAGE 3 — DATA-GROUNDED LATERAL TEACHER V3"
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
    "Training                    : NONE"
)
print(
    "Stage-1 runtime teacher     : OFF"
)
print(
    "Stage-2 runtime teacher     : OFF"
)
print(
    "Stage-3 longitudinal params : UNCHANGED"
)
print(
    "Stage-3 altitude params     : UNCHANGED"
)
print(
    "Only lateral trim/Kp/Kd are swept."
)

rule(
    "A — V2 DIAGNOSTIC EVIDENCE USED BY THIS CALIBRATION"
)

print(
    f"Authority fit: ΔLAT(2s) ≈ "
    f"{slope:+.6f} * action2 "
    f"{intercept:+.6f}"
)

print(
    f"Local zero-ΔLAT action2 estimate : "
    f"{IDENTIFIED_NEUTRAL_A2:+.6f}"
)

print(
    f"Physically tested action2 range  : "
    f"[{IDENTIFIED_A2_MIN:+.3f}, "
    f"{IDENTIFIED_A2_MAX:+.3f}]"
)

print(
    f"Data-derived activation forward  : "
    f"{LATERAL_ENABLE_FORWARD_FT:.3f} ft"
)

print(
    f"Diagnostic speed at activation   : "
    f"{DIAGNOSTIC_ENABLE_SPEED_FPS:.3f} ft/s"
)

print(
    f"Activation qualification limit   : "
    f"{LATERAL_ACTIVATION_MAX_SPEED_FPS:.3f} ft/s"
)

print(
    "Trim candidates                  :",
    [
        round(
            value,
            6,
        )
        for value in HOVER_TRIMS
    ],
)

print(
    "Kp candidates                    :",
    LATERAL_KP_VALUES,
)

print(
    "Kd candidates                    :",
    LATERAL_KD_VALUES,
)


# =====================================================================
# SWEEP
# =====================================================================

rule(
    "B — LATERAL-ONLY TEACHER SWEEP"
)

results = []
case_no = 0

for hover_trim in HOVER_TRIMS:
    for lateral_kp in LATERAL_KP_VALUES:
        for lateral_kd in LATERAL_KD_VALUES:
            case_no += 1

            result = run_case(
                hover_trim,
                lateral_kp,
                lateral_kd,
                detailed=False,
            )

            results.append(
                result
            )

            print(
                f"C{case_no:02d} | "
                f"TRIM={hover_trim:+.4f} | "
                f"KP={lateral_kp:.3f} | "
                f"KD={lateral_kd:.2f} | "
                f"PASS={str(result['teacher_pass']):5s} | "
                f"SAFE={str(result['safe']):5s} | "
                f"HOVER5={str(result['endpoint_hover_5s']):5s} | "
                f"XMAX={result['max_cross_ft']:5.2f} | "
                f"FWD={result['final_forward_ft']:7.2f} | "
                f"V={result['final_forward_speed_fps']:+5.2f} | "
                f"X={result['final_cross_ft']:+6.2f} | "
                f"LAT={result['final_lateral_speed_fps']:+5.2f} | "
                f"ALT={result['final_alt_ft']:7.2f} | "
                f"HOLD={result['current_full_hover_hold_s']:4.2f}s"
            )


rows = []

for result in results:
    rows.append(
        {
            key: value
            for key, value
            in result.items()
            if key
            !=
            "trace"
        }
    )

write_rows(
    SWEEP_CSV,
    rows,
)


# =====================================================================
# SELECTION
#
# Pass criteria dominate. Geometry is then preferred over merely
# minimizing final error.
# =====================================================================

def selection_key(result):
    return (
        0
        if result[
            "teacher_pass"
        ]
        else 1,

        0
        if result[
            "endpoint_hover_5s"
        ]
        else 1,

        0
        if result[
            "presentation_cross_pass"
        ]
        else 1,

        0
        if result[
            "presentation_altitude_pass"
        ]
        else 1,

        0
        if result[
            "safe"
        ]
        else 1,

        result[
            "max_cross_ft"
        ],

        abs(
            result[
                "final_position_error_ft"
            ]
        ),

        abs(
            result[
                "final_forward_speed_fps"
            ]
        ),

        abs(
            result[
                "final_cross_ft"
            ]
        ),

        abs(
            result[
                "final_lateral_speed_fps"
            ]
        ),

        abs(
            result[
                "final_alt_ft"
            ]
            -
            TARGET_ALT_FT
        ),
    )


results.sort(
    key=selection_key
)


rule(
    "C — TOP LATERAL TEACHER CANDIDATES"
)

for rank, result in enumerate(
    results[:10],
    start=1,
):
    print(
        f"{rank:2d}. "
        f"TRIM={result['hover_trim']:+.4f} | "
        f"KP={result['lateral_kp']:.3f} | "
        f"KD={result['lateral_kd']:.2f} | "
        f"PASS={result['teacher_pass']} | "
        f"HOVER5={result['endpoint_hover_5s']} | "
        f"XMAX={result['max_cross_ft']:.3f} | "
        f"FWD={result['final_forward_ft']:.3f} | "
        f"V={result['final_forward_speed_fps']:+.3f} | "
        f"X={result['final_cross_ft']:+.3f} | "
        f"LAT={result['final_lateral_speed_fps']:+.3f} | "
        f"ALT={result['final_alt_ft']:.3f} | "
        f"HOLD={result['current_full_hover_hold_s']:.2f}s"
    )


best = results[0]


# =====================================================================
# FINAL DETAILED REPEAT
# =====================================================================

rule(
    "D — BEST CANDIDATE: FRESH TRUE CONTINUOUS DETAILED REPEAT"
)

print(
    "Selected trim:",
    best[
        "hover_trim"
    ],
)

print(
    "Selected Kp  :",
    best[
        "lateral_kp"
    ],
)

print(
    "Selected Kd  :",
    best[
        "lateral_kd"
    ],
)

best_detailed = run_case(
    best[
        "hover_trim"
    ],
    best[
        "lateral_kp"
    ],
    best[
        "lateral_kd"
    ],
    detailed=True,
)

write_rows(
    BEST_TRACE_CSV,
    best_detailed[
        "trace"
    ],
)


# =====================================================================
# FINAL REPORT
# =====================================================================

rule(
    "STAGE 3 LATERAL TEACHER V3 — FINAL RESULT"
)

for key in [
    "same_fdm_object",
    "sim_clock_reset",
    "safe",
    "termination_reason",
    "lateral_activated",
    "activation_forward_ft",
    "activation_forward_speed_fps",
    "activation_cross_ft",
    "hover_trim",
    "lateral_kp",
    "lateral_kd",
    "presentation_cross_pass",
    "presentation_altitude_pass",
    "endpoint_hover_5s",
    "persistent_stop_at_final",
    "persistent_lateral_at_final",
    "persistent_hover_at_final",
    "current_stop_hold_s",
    "current_lateral_hold_s",
    "current_full_hover_hold_s",
    "max_stop_hold_s",
    "max_lateral_hold_s",
    "max_full_hover_hold_s",
    "final_forward_ft",
    "final_position_error_ft",
    "final_forward_speed_fps",
    "final_cross_ft",
    "final_lateral_speed_fps",
    "final_alt_ft",
    "final_vs_fps",
    "final_heading_error_deg",
    "max_cross_ft",
    "min_alt_ft",
    "max_alt_ft",
    "max_abs_pitch_deg",
    "max_abs_roll_deg",
    "max_abs_heading_error_deg",
    "action2_min",
    "action2_max",
    "teacher_pass",
]:
    print(
        f"{key:36s}: "
        f"{best_detailed[key]}"
    )


final_summary = {
    "method":
        "system identification -> data-grounded lateral teacher calibration",

    "training":
        "none",

    "stage1_model":
        str(
            STAGE1_MODEL_PATH
        ),

    "stage2_model":
        str(
            STAGE2_MODEL_PATH
        ),

    "diagnostic_evidence": {
        "authority_fit_slope":
            float(
                slope
            ),

        "authority_fit_intercept":
            float(
                intercept
            ),

        "identified_local_neutral_action2":
            float(
                IDENTIFIED_NEUTRAL_A2
            ),

        "identified_action2_min":
            float(
                IDENTIFIED_A2_MIN
            ),

        "identified_action2_max":
            float(
                IDENTIFIED_A2_MAX
            ),

        "activation_forward_ft":
            float(
                LATERAL_ENABLE_FORWARD_FT
            ),

        "diagnostic_activation_speed_fps":
            float(
                DIAGNOSTIC_ENABLE_SPEED_FPS
            ),
    },

    "unchanged_longitudinal": {
        "K_POS":
            K_POS,

        "V_FWD_MAX":
            V_FWD_MAX,

        "BRAKE_LEAD_FT":
            BRAKE_LEAD_FT,

        "KV":
            KV,
    },

    "unchanged_altitude": {
        "COLLECTIVE_BIAS":
            COLLECTIVE_BIAS,

        "ALT_KP":
            ALT_KP,

        "VS_KD":
            VS_KD,
    },

    "final_teacher":
        {
            key: value
            for key, value
            in best_detailed.items()
            if key
            !=
            "trace"
        },
}


with FINAL_SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final_summary,
        f,
        indent=2,
    )


print()
print(
    "Saved sweep :",
    SWEEP_CSV,
)

print(
    "Saved trace :",
    BEST_TRACE_CSV,
)

print(
    "Saved summary:",
    FINAL_SUMMARY_JSON,
)

print()

if best_detailed[
    "teacher_pass"
]:
    rule(
        "STAGE 3 TEACHER — LOCKED"
    )

    print(
        "Continuous geometry              : PASS"
    )

    print(
        "Longitudinal stop revalidated    : PASS"
    )

    print(
        "Endpoint lateral hold            : PASS"
    )

    print(
        "5-second full endpoint hover     : PASS"
    )

    print(
        "Stage-3 altitude corridor        : PASS"
    )

    print(
        "Stage-3 max cross-track <= 5 ft  : PASS"
    )

    print(
        "Runtime PPO training in this file: NONE"
    )

    print()
    print(
        "NEXT: collect this teacher trajectory, distill it into a "
        "Stage-3 neural PPO actor, then perform reward-based PPO "
        "fine-tuning and a teacher-OFF continuous validation."
    )

else:
    rule(
        "STAGE 3 TEACHER — NOT LOCKED"
    )

    print(
        "No success claim is made."
    )

    print(
        "Do NOT distill yet. Use the final trace and sweep table to "
        "identify the single failed criterion before changing the "
        "controller design."
    )
