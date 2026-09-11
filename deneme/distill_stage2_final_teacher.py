from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill,
)

from helicopter_env_stage2_refine_mapped import (
    HelicopterEnvStage2RefineMapped,
)


# ============================================================
# MODELS / OUTPUT
# ============================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

SOURCE_STAGE2_MODEL = (
    "models_stage2_refine/"
    "AH1S_STAGE2_REFINE_SUCCESS.zip"
)

OUTPUT_DIR = Path(
    "models_stage2_distilled"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MODEL = (
    OUTPUT_DIR
    /
    "AH1S_STAGE2_DISTILLED_SUCCESS"
)


# ============================================================
# LOCKED TEACHER — FINAL V4 RESULT
# ============================================================

TARGET_ALT = 300.0
TARGET_DISTANCE = 300.0

# Altitude teacher locked earlier.
COL_BIAS = 0.24
ALT_KP = 0.030
VS_KD = 0.120

COL_BIAS_FADE_DISTANCE = 240.0
MAX_COL_ACTION_CORR = 0.40

# Final V4 straight-flight teacher.
TEACHER_AILERON_ACTION = -0.230
TEACHER_RUDDER_ACTION = 0.000

# Repaired Stage-2 physical mapping.
AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040


# ============================================================
# VALIDATION CRITERIA
# ============================================================

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
HANDOFF_STABLE_TIME = 5.0

PRESENT_MIN_ALT = 290.0
PRESENT_MAX_ALT = 310.0
PRESENT_MAX_CROSS = 5.0

CROSS_ALT_MIN = 295.0
CROSS_ALT_MAX = 305.0
CROSS_MAX_ABS_VS = 2.0

# Teacher gave 3.793 ft. We would like the autonomous PPO to
# remain below the original 5-ft presentation limit.
# <=4 ft is reported separately as margin quality.
MARGIN_MAX_CROSS = 4.0

EARTH_RADIUS_FT = 20_902_231.0


# ============================================================
# COLLECTIVE-HEAD RIDGE SEARCH
#
# Rows:
#   0 collective -> fit teacher target with ridge
#   1 elevator   -> EXACTLY PRESERVED
#   2 aileron    -> constant teacher encoded into PPO head
#   3 rudder     -> constant teacher encoded into PPO head
#
# Why constant rows 2/3?
# The old Stage-2 PPO was trained while those physical action paths
# were disconnected. Their old learned values are therefore not a
# valid lateral/yaw policy. The calibrated teacher is explicitly:
#
#   action[2] = -0.230
#   action[3] =  0.000
#
# Encoding those constants in the PPO action head removes all runtime
# teacher/controller logic while reproducing the validated teacher.
# ============================================================

RIDGE_VALUES = [
    0.0001,
    0.0003,
    0.0010,
    0.0030,
    0.0100,
    0.0300,
    0.1000,
    0.3000,
    1.0000,
    3.0000,
    10.0000,
    30.0000,
    100.0000,
    300.0000,
]


# ============================================================
# HELPERS
# ============================================================

def rule(
    text,
):
    print()
    print(
        "="
        *
        144
    )
    print(
        text
    )
    print(
        "="
        *
        144
    )


def fdm_float(
    fdm,
    key,
    default=float("nan"),
):
    try:
        return float(
            fdm[
                key
            ]
        )
    except Exception:
        return float(
            default
        )


def get_fdm(
    env,
):
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

    if base is not None:
        nested = getattr(
            base,
            "fdm",
            None,
        )

        if nested is not None:
            return nested

    raise RuntimeError(
        "Active JSBSim FDM not found."
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


def latitude_deg(
    fdm,
):
    for key in [
        "position/lat-gc-deg",
        "position/lat-geod-deg",
    ]:
        value = fdm_float(
            fdm,
            key,
        )

        if np.isfinite(
            value
        ):
            return value

    return float(
        "nan"
    )


def longitude_deg(
    fdm,
):
    return fdm_float(
        fdm,
        "position/long-gc-deg",
    )


def heading_rad(
    fdm,
):
    for key in [
        "attitude/heading-true-rad",
        "attitude/psi-rad",
    ]:
        value = fdm_float(
            fdm,
            key,
        )

        if np.isfinite(
            value
        ):
            return value

    return float(
        "nan"
    )


def wrap_angle(
    value,
):
    return math.atan2(
        math.sin(
            value
        ),
        math.cos(
            value
        ),
    )


def local_ne_ft(
    lat,
    lon,
    lat0,
    lon0,
):
    dlat = math.radians(
        lat
        -
        lat0
    )

    dlon = math.radians(
        lon
        -
        lon0
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
        float(
            north
        ),
        float(
            east
        ),
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
        north
        *
        c
        +
        east
        *
        s
    )

    cross = (
        -north
        *
        s
        +
        east
        *
        c
    )

    return (
        float(
            forward
        ),
        float(
            cross
        ),
    )


# ============================================================
# STAGE-1 TRUE HANDOFF
# ============================================================

stage1_model = PPO.load(
    STAGE1_MODEL_PATH
)


def build_stage1_handoff():
    env1 = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False,
        )
    )

    obs, info = env1.reset()

    fdm = get_fdm(
        env1
    )

    mission_heading = (
        heading_rad(
            fdm
        )
    )

    dt = float(
        getattr(
            env1,
            "dt",
            0.075,
        )
    )

    if (
        not np.isfinite(
            dt
        )
        or
        dt <= 0.0
    ):
        dt = 0.075

    stable_time = 0.0
    handoff = None

    for step in range(
        int(
            STAGE1_MAX_TIME
            /
            dt
        )
    ):
        action, _ = (
            stage1_model.predict(
                obs,
                deterministic=True,
            )
        )

        (
            obs,
            _,
            terminated,
            truncated,
            info,
        ) = env1.step(
            action
        )

        altitude = info_float(
            info,
            "altitude",
        )

        vertical_speed = info_float(
            info,
            "vertical_speed",
        )

        vn = info_float(
            info,
            "vn",
            0.0,
        )

        ve = info_float(
            info,
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
            info,
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
            dt
            if stable
            else 0.0
        )

        if (
            stable_time
            >=
            HANDOFF_STABLE_TIME
        ):
            handoff = {
                "altitude":
                    altitude,

                "vertical_speed":
                    vertical_speed,

                "horizontal_speed":
                    horizontal_speed,

                "drift":
                    drift,
            }

            break

        if (
            terminated
            and
            not bool(
                info.get(
                    "success",
                    False,
                )
            )
        ):
            break

        if truncated:
            break

    if handoff is None:
        env1.close()

        raise RuntimeError(
            "Stable Stage-1 handoff was not reached."
        )

    return (
        env1,
        fdm,
        mission_heading,
        handoff,
    )


# ============================================================
# ATTACH MAPPED STAGE 2
# ============================================================

def attach_stage2(
    active_fdm,
    mission_heading,
):
    env2 = (
        HelicopterEnvStage2RefineMapped(
            aileron_scale=
                AILERON_SCALE,

            rudder_scale=
                RUDDER_SCALE,
        )
    )

    # Disposable reset only initializes Stage-2 Python bookkeeping.
    env2.reset()

    env2.fdm = (
        active_fdm
    )

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

    obs = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    return (
        env2,
        obs,
    )


# ============================================================
# LOCKED ALTITUDE TEACHER
# ============================================================

def collective_correction(
    altitude,
    vertical_speed,
    forward_distance,
):
    distance_gate = float(
        np.clip(
            1.0
            -
            forward_distance
            /
            COL_BIAS_FADE_DISTANCE,
            0.0,
            1.0,
        )
    )

    altitude_error = (
        TARGET_ALT
        -
        altitude
    )

    correction = (
        COL_BIAS
        *
        distance_gate

        +
        ALT_KP
        *
        altitude_error

        -
        VS_KD
        *
        vertical_speed
    )

    return float(
        np.clip(
            correction,
            -MAX_COL_ACTION_CORR,
            +MAX_COL_ACTION_CORR,
        )
    )


def build_teacher_action(
    source_model,
    obs,
    fdm,
    env2,
):
    base_action, _ = (
        source_model.predict(
            obs,
            deterministic=True,
        )
    )

    base_action = np.asarray(
        base_action,
        dtype=np.float32,
    ).reshape(-1)

    teacher_action = (
        base_action.copy()
    )

    altitude = fdm_float(
        fdm,
        "position/h-agl-ft",
    )

    vertical_speed = fdm_float(
        fdm,
        "velocities/h-dot-fps",
    )

    forward_distance = float(
        getattr(
            env2,
            "forward_distance",
            0.0,
        )
    )

    teacher_action[0] = float(
        np.clip(
            base_action[0]
            +
            collective_correction(
                altitude,
                vertical_speed,
                forward_distance,
            ),
            -1.0,
            +1.0,
        )
    )

    teacher_action[1] = float(
        base_action[1]
    )

    teacher_action[2] = (
        TEACHER_AILERON_ACTION
    )

    teacher_action[3] = (
        TEACHER_RUDDER_ACTION
    )

    return (
        teacher_action.astype(
            np.float32
        ),
        base_action,
    )


# ============================================================
# TRUE CONTINUOUS FLIGHT EVALUATION
# ============================================================

def evaluate_model(
    model,
    use_runtime_teacher,
    detailed=False,
):
    (
        env1,
        fdm,
        mission_heading,
        handoff,
    ) = build_stage1_handoff()

    active_fdm_id = id(
        fdm
    )

    handoff_lat = (
        latitude_deg(
            fdm
        )
    )

    handoff_lon = (
        longitude_deg(
            fdm
        )
    )

    env2, obs = attach_stage2(
        fdm,
        mission_heading,
    )

    if (
        id(
            get_fdm(
                env2
            )
        )
        !=
        active_fdm_id
    ):
        raise RuntimeError(
            "FDM continuity failed."
        )

    dt = float(
        getattr(
            env2,
            "dt",
            0.075,
        )
    )

    if (
        not np.isfinite(
            dt
        )
        or
        dt <= 0.0
    ):
        dt = 0.075

    min_alt = float(
        handoff[
            "altitude"
        ]
    )

    max_alt = float(
        handoff[
            "altitude"
        ]
    )

    max_cross = 0.0
    max_abs_lat = 0.0
    max_abs_heading = 0.0
    max_abs_roll = 0.0

    crossing = None
    failure = False
    termination_reason = ""

    next_print = 0.0

    action0_min = +999.0
    action0_max = -999.0

    action2_min = +999.0
    action2_max = -999.0

    action3_min = +999.0
    action3_max = -999.0

    for step in range(
        int(
            STAGE2_MAX_TIME
            /
            dt
        )
    ):
        if use_runtime_teacher:
            action, _ = (
                build_teacher_action(
                    model,
                    obs,
                    fdm,
                    env2,
                )
            )
        else:
            action, _ = (
                model.predict(
                    obs,
                    deterministic=True,
                )
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)

        action0_min = min(
            action0_min,
            float(
                action[0]
            ),
        )

        action0_max = max(
            action0_max,
            float(
                action[0]
            ),
        )

        action2_min = min(
            action2_min,
            float(
                action[2]
            ),
        )

        action2_max = max(
            action2_max,
            float(
                action[2]
            ),
        )

        action3_min = min(
            action3_min,
            float(
                action[3]
            ),
        )

        action3_max = max(
            action3_max,
            float(
                action[3]
            ),
        )

        (
            obs,
            _,
            terminated,
            truncated,
            info,
        ) = env2.step(
            action
        )

        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        t = (
            step + 1
        ) * dt

        altitude = info_float(
            info,
            "altitude",
            fdm_float(
                fdm,
                "position/h-agl-ft",
            ),
        )

        vertical_speed = info_float(
            info,
            "vertical_speed",
            fdm_float(
                fdm,
                "velocities/h-dot-fps",
                0.0,
            ),
        )

        lateral_velocity = info_float(
            info,
            "lateral_velocity",
            fdm_float(
                fdm,
                "velocities/v-aero-fps",
                0.0,
            ),
        )

        roll = info_float(
            info,
            "roll",
            fdm_float(
                fdm,
                "attitude/roll-rad",
                0.0,
            ),
        )

        heading_now = heading_rad(
            fdm
        )

        heading_error = wrap_angle(
            heading_now
            -
            mission_heading
        )

        distance = info_float(
            info,
            "forward_distance",
            getattr(
                env2,
                "forward_distance",
                0.0,
            ),
        )

        lat = latitude_deg(
            fdm
        )

        lon = longitude_deg(
            fdm
        )

        north, east = (
            local_ne_ft(
                lat,
                lon,
                handoff_lat,
                handoff_lon,
            )
        )

        ground_forward, cross = (
            mission_axes(
                north,
                east,
                mission_heading,
            )
        )

        min_alt = min(
            min_alt,
            altitude,
        )

        max_alt = max(
            max_alt,
            altitude,
        )

        max_cross = max(
            max_cross,
            abs(
                cross
            ),
        )

        max_abs_lat = max(
            max_abs_lat,
            abs(
                lateral_velocity
            ),
        )

        max_abs_heading = max(
            max_abs_heading,
            abs(
                heading_error
            ),
        )

        max_abs_roll = max(
            max_abs_roll,
            abs(
                roll
            ),
        )

        if (
            detailed
            and
            t >= next_print
        ):
            print(
                f"t={t:6.2f}s | "
                f"D={distance:7.2f} | "
                f"GND={ground_forward:7.2f} | "
                f"X={cross:+7.3f} | "
                f"ALT={altitude:7.3f} | "
                f"VS={vertical_speed:+6.3f} | "
                f"LAT={lateral_velocity:+6.3f} | "
                f"HEAD={math.degrees(heading_error):+6.3f}deg | "
                f"ROLL={math.degrees(roll):+6.3f}deg | "
                f"A={np.array2string(action, precision=4, floatmode='fixed')}"
            )

            next_print += 2.5

        if (
            distance
            >=
            TARGET_DISTANCE
        ):
            crossing = {
                "altitude":
                    float(
                        altitude
                    ),

                "vertical_speed":
                    float(
                        vertical_speed
                    ),

                "cross":
                    float(
                        cross
                    ),

                "ground_forward":
                    float(
                        ground_forward
                    ),

                "heading_error_deg":
                    float(
                        math.degrees(
                            heading_error
                        )
                    ),

                "lateral_velocity":
                    float(
                        lateral_velocity
                    ),
            }

            break

        if terminated:
            if not bool(
                info.get(
                    "success",
                    False,
                )
            ):
                failure = True

                termination_reason = str(
                    info.get(
                        "termination_reason",
                        "terminated",
                    )
                )

            break

        if truncated:
            termination_reason = (
                "truncated"
            )
            break

    env2.fdm = None
    env1.close()

    reached = bool(
        crossing is not None
    )

    if reached:
        crossing_alt = (
            crossing[
                "altitude"
            ]
        )

        crossing_vs = (
            crossing[
                "vertical_speed"
            ]
        )

        crossing_cross = (
            crossing[
                "cross"
            ]
        )

        crossing_ground = (
            crossing[
                "ground_forward"
            ]
        )

        crossing_heading = (
            crossing[
                "heading_error_deg"
            ]
        )

        crossing_lat = (
            crossing[
                "lateral_velocity"
            ]
        )
    else:
        crossing_alt = float(
            "nan"
        )

        crossing_vs = float(
            "nan"
        )

        crossing_cross = float(
            "nan"
        )

        crossing_ground = float(
            "nan"
        )

        crossing_heading = float(
            "nan"
        )

        crossing_lat = float(
            "nan"
        )

    presentation_pass = bool(
        reached

        and
        not failure

        and
        min_alt
        >=
        PRESENT_MIN_ALT

        and
        max_alt
        <=
        PRESENT_MAX_ALT

        and
        max_cross
        <=
        PRESENT_MAX_CROSS

        and
        CROSS_ALT_MIN
        <=
        crossing_alt
        <=
        CROSS_ALT_MAX

        and
        abs(
            crossing_vs
        )
        <=
        CROSS_MAX_ABS_VS
    )

    margin_pass = bool(
        presentation_pass

        and
        max_cross
        <=
        MARGIN_MAX_CROSS
    )

    return {
        "reached_300":
            reached,

        "failure":
            failure,

        "termination_reason":
            termination_reason,

        "handoff_alt":
            float(
                handoff[
                    "altitude"
                ]
            ),

        "handoff_vs":
            float(
                handoff[
                    "vertical_speed"
                ]
            ),

        "min_alt":
            float(
                min_alt
            ),

        "max_alt":
            float(
                max_alt
            ),

        "max_drop":
            float(
                handoff[
                    "altitude"
                ]
                -
                min_alt
            ),

        "max_cross":
            float(
                max_cross
            ),

        "max_abs_lat":
            float(
                max_abs_lat
            ),

        "max_abs_heading_deg":
            float(
                math.degrees(
                    max_abs_heading
                )
            ),

        "max_abs_roll_deg":
            float(
                math.degrees(
                    max_abs_roll
                )
            ),

        "crossing_alt":
            float(
                crossing_alt
            ),

        "crossing_vs":
            float(
                crossing_vs
            ),

        "crossing_cross":
            float(
                crossing_cross
            ),

        "crossing_ground":
            float(
                crossing_ground
            ),

        "crossing_heading_deg":
            float(
                crossing_heading
            ),

        "crossing_lat":
            float(
                crossing_lat
            ),

        "action0_min":
            float(
                action0_min
            ),

        "action0_max":
            float(
                action0_max
            ),

        "action2_min":
            float(
                action2_min
            ),

        "action2_max":
            float(
                action2_max
            ),

        "action3_min":
            float(
                action3_min
            ),

        "action3_max":
            float(
                action3_max
            ),

        "presentation_pass":
            presentation_pass,

        "margin_pass":
            margin_pass,
    }


def print_result(
    label,
    r,
):
    print(
        f"{label:18s} | "
        f"PASS={str(r['presentation_pass']):5s} | "
        f"MARGIN={str(r['margin_pass']):5s} | "
        f"MINALT={r['min_alt']:7.3f} | "
        f"MAXALT={r['max_alt']:7.3f} | "
        f"MAX_X={r['max_cross']:6.3f} | "
        f"CROSS_X={r['crossing_cross']:+7.3f} | "
        f"ALT@300={r['crossing_alt']:7.3f} | "
        f"VS@300={r['crossing_vs']:+6.3f} | "
        f"HEAD={r['max_abs_heading_deg']:5.3f}deg"
    )


# ============================================================
# SOURCE MODEL + TEACHER REFERENCE
# ============================================================

rule(
    "STAGE 2 FINAL DISTILLATION — LOCKED TEACHER -> SINGLE PPO"
)

print(
    "Source Stage-2 model:"
)

print(
    SOURCE_STAGE2_MODEL
)

print()
print(
    "Runtime teacher in FINAL model: OFF"
)

print(
    "Runtime altitude controller in FINAL model: OFF"
)

print(
    "Runtime lateral controller in FINAL model: OFF"
)

print(
    "Mapped Stage-2 environment remains required because it repairs "
    "the original action[2]/action[3] actuator wiring."
)

source_model = PPO.load(
    SOURCE_STAGE2_MODEL
)

rule(
    "A — LOCKED TEACHER REFERENCE"
)

teacher_reference = evaluate_model(
    source_model,
    use_runtime_teacher=True,
    detailed=False,
)

print_result(
    "TEACHER",
    teacher_reference,
)

if not teacher_reference[
    "presentation_pass"
]:
    raise RuntimeError(
        "Locked V4 teacher did not reproduce. "
        "Do not distill a non-reproducible teacher."
    )


# ============================================================
# COLLECT TEACHER TRAJECTORY
# ============================================================

rule(
    "B — COLLECT TEACHER TRAJECTORY"
)

(
    env1,
    fdm,
    mission_heading,
    handoff,
) = build_stage1_handoff()

env2, obs = attach_stage2(
    fdm,
    mission_heading,
)

dt = float(
    getattr(
        env2,
        "dt",
        0.075,
    )
)

if (
    not np.isfinite(
        dt
    )
    or
    dt <= 0.0
):
    dt = 0.075


observations = []
target_collective = []
sample_weights = []

next_print = 0.0

for step in range(
    int(
        STAGE2_MAX_TIME
        /
        dt
    )
):
    (
        teacher_action,
        base_action,
    ) = build_teacher_action(
        source_model,
        obs,
        fdm,
        env2,
    )

    distance_before = float(
        getattr(
            env2,
            "forward_distance",
            0.0,
        )
    )

    # Handoff / acceleration region matters most for the collective
    # correction because that was where the old Stage-2 dive occurred.
    if distance_before < 80.0:
        weight = 8.0

    elif distance_before < 180.0:
        weight = 4.0

    else:
        weight = 2.0

    observations.append(
        np.asarray(
            obs,
            dtype=np.float32,
        ).copy()
    )

    target_collective.append(
        float(
            teacher_action[0]
        )
    )

    sample_weights.append(
        float(
            weight
        )
    )

    (
        obs,
        _,
        terminated,
        truncated,
        info,
    ) = env2.step(
        teacher_action
    )

    obs = np.asarray(
        obs,
        dtype=np.float32,
    )

    t = (
        step + 1
    ) * dt

    distance = info_float(
        info,
        "forward_distance",
        getattr(
            env2,
            "forward_distance",
            0.0,
        ),
    )

    if t >= next_print:
        print(
            f"t={t:6.2f}s | "
            f"D={distance:7.2f} | "
            f"base a0={base_action[0]:+7.4f} | "
            f"teacher a0={teacher_action[0]:+7.4f} | "
            f"a2={teacher_action[2]:+6.3f} | "
            f"a3={teacher_action[3]:+6.3f}"
        )

        next_print += 5.0

    if (
        distance
        >=
        TARGET_DISTANCE
    ):
        break

    if terminated or truncated:
        break


env2.fdm = None
env1.close()

observations = np.asarray(
    observations,
    dtype=np.float32,
)

target_collective = np.asarray(
    target_collective,
    dtype=np.float64,
)

sample_weights = np.asarray(
    sample_weights,
    dtype=np.float64,
)

print()
print(
    "Teacher samples:",
    observations.shape
)

print(
    "Collective target range:",
    float(
        np.min(
            target_collective
        )
    ),
    "->",
    float(
        np.max(
            target_collective
        )
    ),
)


# ============================================================
# FROZEN ACTOR FEATURES
# ============================================================

rule(
    "C — EXTRACT FROZEN STAGE-2 ACTOR FEATURES"
)

device = (
    source_model.device
)

obs_tensor = torch.as_tensor(
    observations,
    dtype=torch.float32,
    device=device,
)

latent_batches = []

BATCH_SIZE = 512

with torch.no_grad():
    for start in range(
        0,
        len(
            observations
        ),
        BATCH_SIZE,
    ):
        batch = obs_tensor[
            start:
            start + BATCH_SIZE
        ]

        features = (
            source_model
            .policy
            .extract_features(
                batch
            )
        )

        if isinstance(
            features,
            tuple,
        ):
            actor_features = (
                features[0]
            )
        else:
            actor_features = (
                features
            )

        latent_pi = (
            source_model
            .policy
            .mlp_extractor
            .forward_actor(
                actor_features
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
    axis=0,
).astype(
    np.float64
)

print(
    "Latent shape:",
    latent.shape
)


# ============================================================
# WEIGHTED RIDGE FOR COLLECTIVE ROW ONLY
# ============================================================

ones = np.ones(
    (
        latent.shape[0],
        1,
    ),
    dtype=np.float64,
)

X = np.concatenate(
    [
        latent,
        ones,
    ],
    axis=1,
)

sqrt_w = np.sqrt(
    sample_weights
).reshape(
    -1,
    1,
)

Xw = (
    X
    *
    sqrt_w
)

yw = (
    target_collective
    *
    sqrt_w[:, 0]
)

XtX = (
    Xw.T
    @
    Xw
)

Xty = (
    Xw.T
    @
    yw
)

identity = np.eye(
    X.shape[1],
    dtype=np.float64,
)

source_weight = (
    source_model
    .policy
    .action_net
    .weight
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float64
    )
)

source_bias = (
    source_model
    .policy
    .action_net
    .bias
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float64
    )
)

original_collective = (
    np.concatenate(
        [
            source_weight[0],
            np.array(
                [
                    source_bias[0]
                ],
                dtype=np.float64,
            ),
        ]
    )
)

# Exact elevator row preservation reference.
source_row1_w = (
    source_model
    .policy
    .action_net
    .weight[1]
    .detach()
    .cpu()
    .clone()
)

source_row1_b = (
    source_model
    .policy
    .action_net
    .bias[1]
    .detach()
    .cpu()
    .clone()
)


# ============================================================
# CANDIDATE BUILDER
# ============================================================

def build_candidate(
    ridge,
):
    A = (
        XtX
        +
        float(
            ridge
        )
        *
        identity
    )

    b = (
        Xty
        +
        float(
            ridge
        )
        *
        original_collective
    )

    fitted_collective = (
        np.linalg.solve(
            A,
            b,
        )
    )

    candidate = PPO.load(
        SOURCE_STAGE2_MODEL
    )

    policy = (
        candidate.policy
    )

    weight = (
        policy
        .action_net
        .weight
    )

    bias = (
        policy
        .action_net
        .bias
    )

    with torch.no_grad():
        # --------------------------------------------------------
        # ROW 0 — collective teacher distillation
        # --------------------------------------------------------
        weight[0].copy_(
            torch.as_tensor(
                fitted_collective[
                    :-1
                ],
                dtype=weight.dtype,
                device=weight.device,
            )
        )

        bias[0].copy_(
            torch.tensor(
                float(
                    fitted_collective[
                        -1
                    ]
                ),
                dtype=bias.dtype,
                device=bias.device,
            )
        )

        # --------------------------------------------------------
        # ROW 2 — final calibrated constant aileron action
        # --------------------------------------------------------
        weight[2].zero_()

        bias[2].fill_(
            TEACHER_AILERON_ACTION
        )

        # --------------------------------------------------------
        # ROW 3 — final calibrated constant rudder action
        # --------------------------------------------------------
        weight[3].zero_()

        bias[3].fill_(
            TEACHER_RUDDER_ACTION
        )

    # Verify row 1 is truly untouched.
    row1_same = bool(
        torch.equal(
            policy
            .action_net
            .weight[1]
            .detach()
            .cpu(),
            source_row1_w,
        )
        and
        torch.equal(
            policy
            .action_net
            .bias[1]
            .detach()
            .cpu(),
            source_row1_b,
        )
    )

    if not row1_same:
        raise RuntimeError(
            "Elevator row changed unexpectedly."
        )

    return (
        candidate,
        fitted_collective,
    )


# ============================================================
# RIDGE VALIDATION
# ============================================================

rule(
    "D — TEACHER-OFF RIDGE VALIDATION"
)

results = []

for ridge in RIDGE_VALUES:
    candidate, fitted = (
        build_candidate(
            ridge
        )
    )

    result = evaluate_model(
        candidate,
        use_runtime_teacher=False,
        detailed=False,
    )

    result[
        "ridge"
    ] = float(
        ridge
    )

    results.append(
        (
            result,
            candidate,
        )
    )

    print(
        f"ridge={ridge:8.4f} | "
        f"PASS={str(result['presentation_pass']):5s} | "
        f"MARGIN={str(result['margin_pass']):5s} | "
        f"MINALT={result['min_alt']:7.3f} | "
        f"MAXALT={result['max_alt']:7.3f} | "
        f"MAX_X={result['max_cross']:6.3f} | "
        f"CROSS_X={result['crossing_cross']:+7.3f} | "
        f"ALT@300={result['crossing_alt']:7.3f} | "
        f"VS@300={result['crossing_vs']:+6.3f} | "
        f"A0=[{result['action0_min']:+.3f},{result['action0_max']:+.3f}] | "
        f"A2=[{result['action2_min']:+.3f},{result['action2_max']:+.3f}] | "
        f"A3=[{result['action3_min']:+.3f},{result['action3_max']:+.3f}]"
    )


# ============================================================
# SELECT BEST
#
# PASS first. Then margin. Then geometry. Then altitude.
# ============================================================

def selection_key(
    pair,
):
    r = pair[0]

    return (
        0
        if r[
            "presentation_pass"
        ]
        else 1,

        0
        if r[
            "margin_pass"
        ]
        else 1,

        r[
            "max_cross"
        ],

        abs(
            r[
                "crossing_cross"
            ]
        )
        if np.isfinite(
            r[
                "crossing_cross"
            ]
        )
        else 999.0,

        abs(
            r[
                "crossing_alt"
            ]
            -
            TARGET_ALT
        )
        if np.isfinite(
            r[
                "crossing_alt"
            ]
        )
        else 999.0,

        abs(
            r[
                "crossing_vs"
            ]
        )
        if np.isfinite(
            r[
                "crossing_vs"
            ]
        )
        else 999.0,
    )


results.sort(
    key=selection_key
)

best_result, best_model = (
    results[0]
)


# ============================================================
# FINAL FULL VALIDATION
# ============================================================

rule(
    "E — BEST DISTILLED STAGE-2 PPO"
)

print(
    "Selected ridge:",
    best_result[
        "ridge"
    ],
)

print_result(
    "BEST",
    best_result,
)

rule(
    "F — FINAL TRUE CONTINUOUS TEACHER-OFF FLIGHT"
)

print(
    "Stage 1 teacher            : OFF"
)

print(
    "Stage 1 runtime controllers: OFF"
)

print(
    "Stage 2 altitude teacher   : OFF"
)

print(
    "Stage 2 lateral teacher    : OFF"
)

print(
    "Stage 2 rudder teacher     : OFF"
)

print(
    "Runtime action bias code   : OFF"
)

print(
    "Stage 2 policy             : SINGLE 4-ACTION PPO"
)

print(
    "Stage 2 mapped env         : ON (actuator wiring repair only)"
)

print()

final_result = evaluate_model(
    best_model,
    use_runtime_teacher=False,
    detailed=True,
)


# ============================================================
# FINAL REPORT
# ============================================================

rule(
    "STAGE 2 DISTILLED FINAL RESULT"
)

for key in [
    "reached_300",
    "failure",
    "termination_reason",
    "handoff_alt",
    "handoff_vs",
    "min_alt",
    "max_alt",
    "max_drop",
    "max_cross",
    "max_abs_lat",
    "max_abs_heading_deg",
    "max_abs_roll_deg",
    "crossing_alt",
    "crossing_vs",
    "crossing_cross",
    "crossing_ground",
    "crossing_heading_deg",
    "crossing_lat",
    "action0_min",
    "action0_max",
    "action2_min",
    "action2_max",
    "action3_min",
    "action3_max",
    "presentation_pass",
    "margin_pass",
]:
    print(
        f"{key:28s}: "
        f"{final_result[key]}"
    )


# ============================================================
# SAVE ONLY IF TRUE CONTINUOUS TEACHER-OFF PASS
# ============================================================

if final_result[
    "presentation_pass"
]:
    best_model.save(
        OUTPUT_MODEL
    )

    rule(
        "STAGE 1 + STAGE 2 LOCKED"
    )

    print(
        "TRUE continuous mission:"
    )

    print(
        "takeoff -> 300 ft hover -> 300 ft straight forward"
    )

    print()

    print(
        "Stage-2 teacher runtime: OFF"
    )

    print(
        "Stage-2 PPO actions:"
    )

    print(
        "  action[0] collective : DISTILLED"
    )

    print(
        "  action[1] elevator   : ORIGINAL PPO / UNCHANGED"
    )

    print(
        "  action[2] aileron    : DISTILLED CONSTANT -0.230"
    )

    print(
        "  action[3] rudder     : DISTILLED CONSTANT  0.000"
    )

    print()

    print(
        "Saved:"
    )

    print(
        str(
            OUTPUT_MODEL
        )
        +
        ".zip"
    )

    print()

    if final_result[
        "margin_pass"
    ]:
        print(
            "Cross-track margin <=4 ft: PASS"
        )
    else:
        print(
            "Cross-track <=5 ft: PASS"
        )

        print(
            "Cross-track <=4 ft margin: NOT retained after distillation"
        )

    print()

    print(
        "NEXT STEP: Stage-2 endpoint -> stop / transition validation."
    )

else:
    rule(
        "DISTILLATION NOT ACCEPTED"
    )

    print(
        "No new Stage-2 model was saved."
    )

    print(
        "The calibrated teacher remains valid, but the PPO collective "
        "head did not imitate it accurately enough."
    )

    print(
        "Do NOT change Stage 1 and do NOT restart lateral calibration."
    )

    print(
        "Only the Stage-2 collective-head distillation would need a "
        "different fit strategy."
    )
