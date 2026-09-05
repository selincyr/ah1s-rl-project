from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill,
)

from helicopter_env_stage2_refine_mapped import (
    HelicopterEnvStage2RefineMapped,
)


# ============================================================
# MODELS
# ============================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL_PATH = (
    "models_stage2_refine/"
    "AH1S_STAGE2_REFINE_SUCCESS.zip"
)


# ============================================================
# LOCKED ALTITUDE TEACHER
# ============================================================

COL_BIAS = 0.24
ALT_KP = 0.030
VS_KD = 0.120

TARGET_ALT = 300.0
TARGET_DISTANCE = 300.0

COL_BIAS_FADE_DISTANCE = 240.0
MAX_COL_ACTION_CORR = 0.40


# ============================================================
# REPAIRED PHYSICAL ACTION SCALES
# ============================================================

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040


# ============================================================
# SEARCH
#
# IMPORTANT:
# The original Stage-2 PPO's action[2]/action[3] heads were never
# physically connected during training. Therefore their old output
# values are NOT trusted as meaningful controls.
#
# During calibration:
#   action[0] -> original PPO + locked altitude teacher
#   action[1] -> original PPO
#   action[2] -> explicit calibration command
#   action[3] -> explicit calibration command
#
# Once a good lateral teacher is found, we can train/distill the
# now-live PPO lateral/yaw heads.
# ============================================================

AXIS_VALUES = [
    -1.00,
    -0.75,
    -0.50,
    -0.25,
     0.00,
    +0.25,
    +0.50,
    +0.75,
    +1.00,
]

LOCAL_OFFSETS = [
    -0.20,
    -0.10,
     0.00,
    +0.10,
    +0.20,
]


# ============================================================
# QUALITY
# ============================================================

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0

HANDOFF_STABLE_TIME = 5.0

PRESENT_MIN_ALT = 290.0
PRESENT_MAX_ALT = 310.0
PRESENT_MAX_CROSS = 5.0

CROSS_ALT_MIN = 295.0
CROSS_ALT_MAX = 305.0
CROSS_MAX_VS = 2.0

EARTH_RADIUS_FT = 20_902_231.0


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path(
    "results_stage2_lateral_v3_mapped"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AIL_CSV = (
    OUT_DIR
    /
    "phase_a_aileron_only.csv"
)

RUD_CSV = (
    OUT_DIR
    /
    "phase_b_rudder_only.csv"
)

COMBO_CSV = (
    OUT_DIR
    /
    "phase_c_combined.csv"
)

FINE_CSV = (
    OUT_DIR
    /
    "phase_d_fine.csv"
)

TRACE_CSV = (
    OUT_DIR
    /
    "best_mapped_trace.csv"
)


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
        140
    )
    print(
        text
    )
    print(
        "="
        *
        140
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


def save_rows(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = (
            csv.DictWriter(
                file,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# LOAD MODELS
# ============================================================

stage1_model = PPO.load(
    STAGE1_MODEL_PATH
)

stage2_model = PPO.load(
    STAGE2_MODEL_PATH
)


# ============================================================
# STAGE 1 TRUE HANDOFF
# ============================================================

def build_handoff():
    env = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False,
        )
    )

    obs, info = (
        env.reset()
    )

    fdm = get_fdm(
        env
    )

    mission_heading = (
        heading_rad(
            fdm
        )
    )

    dt = float(
        getattr(
            env,
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
        ) = env.step(
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

        if stable:
            stable_time += (
                dt
            )
        else:
            stable_time = 0.0

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
        env.close()

        raise RuntimeError(
            "Stable Stage-1 handoff not reached."
        )

    return (
        env,
        fdm,
        mission_heading,
        handoff,
    )


# ============================================================
# ATTACH REPAIRED STAGE 2
# ============================================================

def attach_stage2(
    active_fdm,
    mission_heading,
):
    env = (
        HelicopterEnvStage2RefineMapped(
            aileron_scale=
                AILERON_SCALE,

            rudder_scale=
                RUDDER_SCALE,
        )
    )

    # Disposable reset only initializes Stage-2 Python state.
    env.reset()

    env.fdm = (
        active_fdm
    )

    if hasattr(
        env,
        "forward_distance",
    ):
        env.forward_distance = (
            0.0
        )

    if hasattr(
        env,
        "target_heading",
    ):
        env.target_heading = float(
            mission_heading
        )

    for attr in [
        "steps",
        "target_hold_steps",
        "hold_steps",
        "success_hold_steps",
    ]:
        if hasattr(
            env,
            attr,
        ):
            setattr(
                env,
                attr,
                0,
            )

    obs = np.asarray(
        env._get_obs(),
        dtype=np.float32,
    )

    return (
        env,
        obs,
    )


# ============================================================
# ALTITUDE TEACHER
# ============================================================

def collective_correction(
    altitude,
    vertical_speed,
    forward_distance,
):
    gate = float(
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
        gate

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


# ============================================================
# MAPPING SELF TEST
# ============================================================

def mapping_probe(
    a2,
    a3,
):
    (
        env1,
        fdm,
        mission_heading,
        handoff,
    ) = build_handoff()

    active_id = id(
        fdm
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
        active_id
    ):
        raise RuntimeError(
            "FDM continuity failed."
        )

    base_action, _ = (
        stage2_model.predict(
            obs,
            deterministic=True,
        )
    )

    action = np.asarray(
        base_action,
        dtype=np.float32,
    ).reshape(-1)

    action[2] = float(
        a2
    )

    action[3] = float(
        a3
    )

    before_ail = fdm_float(
        fdm,
        "fcs/aileron-cmd-norm",
    )

    before_rud = fdm_float(
        fdm,
        "fcs/rudder-cmd-norm",
    )

    (
        _,
        _,
        _,
        _,
        info,
    ) = env2.step(
        action
    )

    after_ail = fdm_float(
        fdm,
        "fcs/aileron-cmd-norm",
    )

    after_rud = fdm_float(
        fdm,
        "fcs/rudder-cmd-norm",
    )

    result = {
        "a2":
            float(
                a2
            ),

        "a3":
            float(
                a3
            ),

        "before_ail":
            before_ail,

        "after_ail":
            after_ail,

        "before_rud":
            before_rud,

        "after_rud":
            after_rud,

        "info_ail":
            info_float(
                info,
                "aileron",
            ),

        "info_rud":
            info_float(
                info,
                "rudder",
            ),
    }

    env2.fdm = None
    env1.close()

    return result


# ============================================================
# FULL TRUE CONTINUOUS FLIGHT
# ============================================================

def run_case(
    aileron_command,
    rudder_command,
    detailed=False,
):
    (
        env1,
        fdm,
        mission_heading,
        handoff,
    ) = build_handoff()

    active_id = id(
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
        active_id
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

    trace = []

    next_print = 0.0

    for step in range(
        int(
            STAGE2_MAX_TIME
            /
            dt
        )
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
        ).reshape(-1)

        action = (
            base_action.copy()
        )

        altitude_before = fdm_float(
            fdm,
            "position/h-agl-ft",
        )

        vs_before = fdm_float(
            fdm,
            "velocities/h-dot-fps",
        )

        distance_before = float(
            getattr(
                env2,
                "forward_distance",
                0.0,
            )
        )

        action[0] = float(
            np.clip(
                action[0]
                +
                collective_correction(
                    altitude_before,
                    vs_before,
                    distance_before,
                ),
                -1.0,
                +1.0,
            )
        )

        # IMPORTANT:
        # Ignore the old unconnected PPO lateral/yaw outputs during
        # calibration. These are explicit teacher commands.
        action[2] = float(
            np.clip(
                aileron_command,
                -1.0,
                +1.0,
            )
        )

        action[3] = float(
            np.clip(
                rudder_command,
                -1.0,
                +1.0,
            )
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
            step
            +
            1
        ) * dt

        altitude = info_float(
            info,
            "altitude",
            fdm_float(
                fdm,
                "position/h-agl-ft",
            ),
        )

        vs = info_float(
            info,
            "vertical_speed",
            fdm_float(
                fdm,
                "velocities/h-dot-fps",
                0.0,
            ),
        )

        fwd = info_float(
            info,
            "forward_velocity",
            fdm_float(
                fdm,
                "velocities/u-aero-fps",
                0.0,
            ),
        )

        lat_vel = info_float(
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

        heading_now = (
            heading_rad(
                fdm
            )
        )

        heading_error = (
            wrap_angle(
                heading_now
                -
                mission_heading
            )
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

        physical_ail = fdm_float(
            fdm,
            "fcs/aileron-cmd-norm",
        )

        physical_rud = fdm_float(
            fdm,
            "fcs/rudder-cmd-norm",
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
                lat_vel
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

        row = {
            "time_s":
                float(
                    t
                ),

            "distance_ft":
                float(
                    distance
                ),

            "ground_forward_ft":
                float(
                    ground_forward
                ),

            "cross_track_ft":
                float(
                    cross
                ),

            "altitude_ft":
                float(
                    altitude
                ),

            "vertical_speed_fps":
                float(
                    vs
                ),

            "forward_speed_fps":
                float(
                    fwd
                ),

            "lateral_speed_fps":
                float(
                    lat_vel
                ),

            "heading_error_deg":
                float(
                    math.degrees(
                        heading_error
                    )
                ),

            "roll_deg":
                float(
                    math.degrees(
                        roll
                    )
                ),

            "aileron_command":
                float(
                    action[2]
                ),

            "rudder_command":
                float(
                    action[3]
                ),

            "physical_aileron":
                float(
                    physical_ail
                ),

            "physical_rudder":
                float(
                    physical_rud
                ),
        }

        trace.append(
            row
        )

        if (
            detailed
            and
            t
            >=
            next_print
        ):
            print(
                f"t={t:6.2f}s | "
                f"D={distance:7.2f} | "
                f"GND={ground_forward:7.2f} | "
                f"X={cross:+7.2f} | "
                f"ALT={altitude:7.2f} | "
                f"VS={vs:+6.2f} | "
                f"LAT={lat_vel:+6.2f} | "
                f"HEAD={math.degrees(heading_error):+6.2f}deg | "
                f"ROLL={math.degrees(roll):+6.2f}deg | "
                f"AIL={physical_ail:+.5f} | "
                f"RUD={physical_rud:+.5f}"
            )

            next_print += (
                2.5
            )

        if (
            distance
            >=
            TARGET_DISTANCE
        ):
            crossing = (
                row.copy()
            )
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
        c_alt = float(
            crossing[
                "altitude_ft"
            ]
        )

        c_vs = float(
            crossing[
                "vertical_speed_fps"
            ]
        )

        c_cross = float(
            crossing[
                "cross_track_ft"
            ]
        )

        c_heading = float(
            crossing[
                "heading_error_deg"
            ]
        )

        c_lat = float(
            crossing[
                "lateral_speed_fps"
            ]
        )

        c_ground = float(
            crossing[
                "ground_forward_ft"
            ]
        )
    else:
        c_alt = float(
            "nan"
        )
        c_vs = float(
            "nan"
        )
        c_cross = float(
            "nan"
        )
        c_heading = float(
            "nan"
        )
        c_lat = float(
            "nan"
        )
        c_ground = float(
            "nan"
        )

    result = {
        "aileron_command":
            float(
                aileron_command
            ),

        "rudder_command":
            float(
                rudder_command
            ),

        "reached_300":
            reached,

        "failure":
            bool(
                failure
            ),

        "termination_reason":
            termination_reason,

        "handoff_alt":
            float(
                handoff[
                    "altitude"
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
            c_alt,

        "crossing_vs":
            c_vs,

        "crossing_cross":
            c_cross,

        "crossing_heading_deg":
            c_heading,

        "crossing_lat":
            c_lat,

        "crossing_ground":
            c_ground,

        "presentation_pass":
            bool(
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
                c_alt
                <=
                CROSS_ALT_MAX

                and
                abs(
                    c_vs
                )
                <=
                CROSS_MAX_VS
            ),

        "trace":
            trace,
    }

    return result


# ============================================================
# SCORE
# ============================================================

def score_case(
    r,
):
    if not r[
        "reached_300"
    ]:
        return (
            1_000_000.0
            +
            1000.0
            *
            r[
                "max_cross"
            ]
        )

    low_alt = max(
        0.0,
        PRESENT_MIN_ALT
        -
        r[
            "min_alt"
        ],
    )

    high_alt = max(
        0.0,
        r[
            "max_alt"
        ]
        -
        PRESENT_MAX_ALT,
    )

    return float(
        1500.0
        *
        r[
            "max_cross"
        ]

        +
        350.0
        *
        abs(
            r[
                "crossing_cross"
            ]
        )

        +
        150.0
        *
        r[
            "max_abs_lat"
        ]

        +
        70.0
        *
        r[
            "max_abs_heading_deg"
        ]

        +
        35.0
        *
        r[
            "max_abs_roll_deg"
        ]

        +
        3000.0
        *
        low_alt

        +
        2000.0
        *
        high_alt

        +
        200.0
        *
        abs(
            r[
                "crossing_alt"
            ]
            -
            TARGET_ALT
        )

        +
        100.0
        *
        abs(
            r[
                "crossing_vs"
            ]
        )
    )


def row_without_trace(
    r,
):
    return {
        key:
            value
        for key, value
        in r.items()
        if key
        !=
        "trace"
    }


def print_case(
    label,
    r,
):
    print(
        f"{label:12s} | "
        f"AIL={r['aileron_command']:+.3f} | "
        f"RUD={r['rudder_command']:+.3f} | "
        f"MAX_X={r['max_cross']:6.2f} | "
        f"CROSS_X={r['crossing_cross']:+7.2f} | "
        f"LAT={r['max_abs_lat']:5.2f} | "
        f"HEAD={r['max_abs_heading_deg']:5.2f}deg | "
        f"ROLL={r['max_abs_roll_deg']:5.2f}deg | "
        f"MIN_ALT={r['min_alt']:7.2f} | "
        f"REACH={r['reached_300']} | "
        f"PASS={r['presentation_pass']}"
    )


# ============================================================
# HEADER
# ============================================================

rule(
    "STAGE 2 LATERAL V3 — REPAIRED ACTION MAPPING"
)

print(
    "Stage 1: LOCKED"
)

print(
    "Original Stage-2 model: unchanged"
)

print(
    f"Mapped aileron physical scale: ±{AILERON_SCALE:.3f}"
)

print(
    f"Mapped rudder physical scale : ±{RUDDER_SCALE:.3f}"
)

print(
    "Old PPO action[2]/[3] values are ignored during this calibration."
)


# ============================================================
# A — MAPPING PROOF
# ============================================================

rule(
    "A — REPAIRED ENV.STEP AUTHORITY PROOF"
)

probes = []

for a2, a3 in [
    (-1.0, 0.0),
    (+1.0, 0.0),
    (0.0, -1.0),
    (0.0, +1.0),
]:
    p = mapping_probe(
        a2,
        a3,
    )

    probes.append(
        p
    )

    print(
        f"a2={a2:+.1f}, a3={a3:+.1f} | "
        f"AIL {p['before_ail']:+.6f} -> {p['after_ail']:+.6f} | "
        f"RUD {p['before_rud']:+.6f} -> {p['after_rud']:+.6f} | "
        f"info=({p['info_ail']:+.6f},{p['info_rud']:+.6f})"
    )

ail_authority = (
    abs(
        probes[0][
            "after_ail"
        ]
        -
        probes[1][
            "after_ail"
        ]
    )
    >
    0.02
)

rud_authority = (
    abs(
        probes[2][
            "after_rud"
        ]
        -
        probes[3][
            "after_rud"
        ]
    )
    >
    0.03
)

print()
print(
    "AILERON ACTION PATH LIVE:",
    ail_authority,
)

print(
    "RUDDER ACTION PATH LIVE :",
    rud_authority,
)

if not (
    ail_authority
    and
    rud_authority
):
    raise RuntimeError(
        "Mapped action path verification failed. "
        "Do not continue calibration."
    )


# ============================================================
# REFERENCE
# ============================================================

rule(
    "B — REFERENCE WITH LIVE MAPPING, ZERO LATERAL/YAW COMMAND"
)

reference = run_case(
    0.0,
    0.0,
)

print_case(
    "REFERENCE",
    reference,
)


# ============================================================
# PHASE A — AILERON ONLY
# ============================================================

rule(
    "C — AILERON-ONLY SWEEP"
)

ail_rows = []

for index, ail in enumerate(
    AXIS_VALUES,
    start=1,
):
    r = run_case(
        ail,
        0.0,
    )

    row = (
        row_without_trace(
            r
        )
    )

    row[
        "score"
    ] = score_case(
        r
    )

    ail_rows.append(
        row
    )

    print_case(
        f"A{index:02d}",
        r,
    )

save_rows(
    AIL_CSV,
    ail_rows,
)

ail_sorted = sorted(
    ail_rows,
    key=lambda x:
        x[
            "score"
        ],
)


# ============================================================
# PHASE B — RUDDER ONLY
# ============================================================

rule(
    "D — RUDDER-ONLY SWEEP"
)

rud_rows = []

for index, rud in enumerate(
    AXIS_VALUES,
    start=1,
):
    r = run_case(
        0.0,
        rud,
    )

    row = (
        row_without_trace(
            r
        )
    )

    row[
        "score"
    ] = score_case(
        r
    )

    rud_rows.append(
        row
    )

    print_case(
        f"R{index:02d}",
        r,
    )

save_rows(
    RUD_CSV,
    rud_rows,
)

rud_sorted = sorted(
    rud_rows,
    key=lambda x:
        x[
            "score"
        ],
)


# ============================================================
# PHASE C — COMBINE TOP 3 OF EACH AXIS
# ============================================================

rule(
    "E — COMBINE TOP 3 AILERON + TOP 3 RUDDER COMMANDS"
)

top_ails = [
    row[
        "aileron_command"
    ]
    for row in ail_sorted[
        :3
    ]
]

top_ruds = [
    row[
        "rudder_command"
    ]
    for row in rud_sorted[
        :3
    ]
]

print(
    "Top aileron commands:",
    top_ails,
)

print(
    "Top rudder commands :",
    top_ruds,
)

combo_rows = []

combo_index = 0

for ail in top_ails:
    for rud in top_ruds:
        combo_index += 1

        r = run_case(
            ail,
            rud,
        )

        row = (
            row_without_trace(
                r
            )
        )

        row[
            "score"
        ] = score_case(
            r
        )

        combo_rows.append(
            row
        )

        print_case(
            f"C{combo_index:02d}",
            r,
        )

save_rows(
    COMBO_CSV,
    combo_rows,
)

combo_sorted = sorted(
    combo_rows,
    key=lambda x:
        x[
            "score"
        ],
)

best_combo = (
    combo_sorted[0]
)


# ============================================================
# PHASE D — LOCAL FINE SEARCH
# ============================================================

rule(
    "F — LOCAL FINE SEARCH"
)

fine_ails = sorted(
    {
        float(
            np.clip(
                best_combo[
                    "aileron_command"
                ]
                +
                offset,
                -1.0,
                +1.0,
            )
        )
        for offset
        in LOCAL_OFFSETS
    }
)

fine_ruds = sorted(
    {
        float(
            np.clip(
                best_combo[
                    "rudder_command"
                ]
                +
                offset,
                -1.0,
                +1.0,
            )
        )
        for offset
        in LOCAL_OFFSETS
    }
)

print(
    "Fine aileron:",
    fine_ails,
)

print(
    "Fine rudder :",
    fine_ruds,
)

fine_rows = []

fine_index = 0

for ail in fine_ails:
    for rud in fine_ruds:
        fine_index += 1

        r = run_case(
            ail,
            rud,
        )

        row = (
            row_without_trace(
                r
            )
        )

        row[
            "score"
        ] = score_case(
            r
        )

        fine_rows.append(
            row
        )

        print_case(
            f"F{fine_index:02d}",
            r,
        )

save_rows(
    FINE_CSV,
    fine_rows,
)

fine_sorted = sorted(
    fine_rows,
    key=lambda x:
        x[
            "score"
        ],
)

best = (
    fine_sorted[0]
)


# ============================================================
# TOP RESULTS
# ============================================================

rule(
    "TOP 10 MAPPED LATERAL CANDIDATES"
)

for rank, row in enumerate(
    fine_sorted[
        :10
    ],
    start=1,
):
    print(
        f"{rank:2d}. "
        f"AIL={row['aileron_command']:+.3f} | "
        f"RUD={row['rudder_command']:+.3f} | "
        f"MAX_X={row['max_cross']:6.3f} | "
        f"CROSS_X={row['crossing_cross']:+7.3f} | "
        f"LAT={row['max_abs_lat']:5.3f} | "
        f"HEAD={row['max_abs_heading_deg']:5.3f}deg | "
        f"MIN_ALT={row['min_alt']:7.3f} | "
        f"CROSS_ALT={row['crossing_alt']:7.3f} | "
        f"VS={row['crossing_vs']:+6.3f} | "
        f"PASS={row['presentation_pass']} | "
        f"SCORE={row['score']:.1f}"
    )


# ============================================================
# FULL BEST FLIGHT
# ============================================================

rule(
    "BEST MAPPED CONFIG — FULL DETAILED FLIGHT"
)

print(
    f"AILERON COMMAND: "
    f"{best['aileron_command']:+.3f}"
)

print(
    f"RUDDER COMMAND : "
    f"{best['rudder_command']:+.3f}"
)

best_result = run_case(
    best[
        "aileron_command"
    ],
    best[
        "rudder_command"
    ],
    detailed=True,
)

save_rows(
    TRACE_CSV,
    best_result[
        "trace"
    ],
)


# ============================================================
# FINAL
# ============================================================

rule(
    "STAGE 2 LATERAL V3 MAPPED FINAL RESULT"
)

for key in [
    "aileron_command",
    "rudder_command",
    "reached_300",
    "failure",
    "termination_reason",
    "handoff_alt",
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
    "crossing_heading_deg",
    "crossing_lat",
    "crossing_ground",
    "presentation_pass",
]:
    print(
        f"{key:27s}: "
        f"{best_result[key]}"
    )

print()

improvement = (
    reference[
        "max_cross"
    ]
    -
    best_result[
        "max_cross"
    ]
)

print(
    f"CROSS-TRACK IMPROVEMENT: "
    f"{improvement:+.3f} ft"
)

if best_result[
    "presentation_pass"
]:
    print(
        "PRESENTATION QUALITY: TRUE"
    )

    print(
        "Next: train/distill the repaired Stage-2 PPO lateral/yaw heads "
        "and then remove the runtime teacher."
    )

elif improvement > 10.0:
    print(
        "PRESENTATION QUALITY: FALSE"
    )

    print(
        "But the repaired control path has strong authority."
    )

    print(
        "Next: add small lateral/roll feedback around this best mapped "
        "feed-forward command."
    )

else:
    print(
        "PRESENTATION QUALITY: FALSE"
    )

    print(
        "The action path is repaired, but constant residual commands "
        "are not enough."
    )

    print(
        "Next: run explicit pulse identification on the now-live mapped axes."
    )

print()
print(
    "Saved:",
    AIL_CSV,
)
print(
    "Saved:",
    RUD_CSV,
)
print(
    "Saved:",
    COMBO_CSV,
)
print(
    "Saved:",
    FINE_CSV,
)
print(
    "Saved:",
    TRACE_CSV,
)
