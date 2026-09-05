from __future__ import annotations

import csv
import json
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

TARGET_ALT = 300.0
TARGET_DISTANCE = 300.0

COL_BIAS = 0.24
ALT_KP = 0.030
VS_KD = 0.120

COL_BIAS_FADE_DISTANCE = 240.0
MAX_COL_ACTION_CORR = 0.40


# ============================================================
# REPAIRED MAPPING
# ============================================================

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040


# ============================================================
# V4 MICRO SEARCH
#
# V3 already proved:
#
#   AIL=-0.25, RUD=0.00 -> PRESENTATION PASS
#   MAX cross-track      -> 4.76 ft
#
# The previous score accidentally preferred AIL=-0.25,RUD=+0.30
# even though that candidate FAILED the 5-ft geometry threshold.
#
# V4 therefore does NOT add another feedback controller.
# It searches tightly around the actual passing point and selects:
#
#   1) PASS candidates first
#   2) smallest MAX |cross-track|
#   3) smallest endpoint |cross-track|
#   4) smallest heading error
#
# Goal: create margin before PPO distillation.
# ============================================================

AILERON_FINE_VALUES = [
    -0.300,
    -0.285,
    -0.270,
    -0.260,
    -0.250,
    -0.240,
    -0.230,
    -0.215,
    -0.200,
]

RUDDER_FINE_VALUES = [
    -0.150,
    -0.100,
    -0.050,
     0.000,
    +0.050,
    +0.100,
    +0.150,
]

# First: aileron-only fine scan.
# Second: top 3 aileron values x all small rudder values.

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
HANDOFF_STABLE_TIME = 5.0

PRESENT_MIN_ALT = 290.0
PRESENT_MAX_ALT = 310.0
PRESENT_MAX_CROSS = 5.0
PRESENT_CROSS_ALT_MIN = 295.0
PRESENT_CROSS_ALT_MAX = 305.0
PRESENT_CROSS_MAX_VS = 2.0

# Stronger "lock margin" target.
MARGIN_MAX_CROSS = 4.0

EARTH_RADIUS_FT = 20_902_231.0


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path(
    "results_stage2_lateral_v4_micro"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AIL_ONLY_CSV = (
    OUT_DIR
    / "aileron_only_micro.csv"
)

COMBO_CSV = (
    OUT_DIR
    / "aileron_rudder_micro.csv"
)

BEST_TRACE_CSV = (
    OUT_DIR
    / "best_v4_trace.csv"
)

BEST_CONFIG_JSON = (
    OUT_DIR
    / "best_v4_teacher_config.json"
)


# ============================================================
# HELPERS
# ============================================================

def rule(text):
    print()
    print("=" * 140)
    print(text)
    print("=" * 140)


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

    if base is not None:
        nested = getattr(
            base,
            "fdm",
            None,
        )

        if nested is not None:
            return nested

    raise RuntimeError(
        "Active FDM not found."
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


def latitude_deg(fdm):
    for key in [
        "position/lat-gc-deg",
        "position/lat-geod-deg",
    ]:
        x = fdm_float(
            fdm,
            key,
        )

        if np.isfinite(x):
            return x

    return float("nan")


def longitude_deg(fdm):
    return fdm_float(
        fdm,
        "position/long-gc-deg",
    )


def heading_rad(fdm):
    for key in [
        "attitude/heading-true-rad",
        "attitude/psi-rad",
    ]:
        x = fdm_float(
            fdm,
            key,
        )

        if np.isfinite(x):
            return x

    return float("nan")


def wrap_angle(x):
    return math.atan2(
        math.sin(x),
        math.cos(x),
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
        * dlat
    )

    east = (
        EARTH_RADIUS_FT
        * math.cos(
            math.radians(lat0)
        )
        * dlon
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


# ============================================================
# MODELS
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

    mission_heading = heading_rad(
        fdm
    )

    dt = float(
        getattr(
            env1,
            "dt",
            0.075,
        )
    )

    if (
        not np.isfinite(dt)
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

        alt = info_float(
            info,
            "altitude",
        )

        vs = info_float(
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

        hs = float(
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
            <= alt
            <= 305.0

            and
            abs(vs)
            <= 0.50

            and
            hs
            <= 1.0

            and
            drift
            <= 3.0
        )

        stable_time = (
            stable_time + dt
            if stable
            else 0.0
        )

        if (
            stable_time
            >= HANDOFF_STABLE_TIME
        ):
            handoff = {
                "altitude":
                    alt,

                "vs":
                    vs,

                "hs":
                    hs,

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
            "Stable Stage-1 handoff not reached."
        )

    return (
        env1,
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
    env2 = (
        HelicopterEnvStage2RefineMapped(
            aileron_scale=
                AILERON_SCALE,

            rudder_scale=
                RUDDER_SCALE,
        )
    )

    # Disposable reset initializes Stage-2 Python bookkeeping only.
    env2.reset()

    env2.fdm = (
        active_fdm
    )

    if hasattr(
        env2,
        "forward_distance",
    ):
        env2.forward_distance = 0.0

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


# ============================================================
# ONE CONTINUOUS FLIGHT
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

    lat0 = latitude_deg(
        fdm
    )

    lon0 = longitude_deg(
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

    dt = float(
        getattr(
            env2,
            "dt",
            0.075,
        )
    )

    if (
        not np.isfinite(dt)
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

        alt_before = fdm_float(
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
                    alt_before,
                    vs_before,
                    distance_before,
                ),
                -1.0,
                +1.0,
            )
        )

        # Explicit lateral/yaw teacher.
        action[2] = float(
            aileron_command
        )

        action[3] = float(
            rudder_command
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

        alt = info_float(
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
                lat0,
                lon0,
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
            alt,
        )

        max_alt = max(
            max_alt,
            alt,
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
                    alt
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

            "aileron_teacher":
                float(
                    action[2]
                ),

            "rudder_teacher":
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
                f"X={cross:+7.3f} | "
                f"ALT={alt:7.3f} | "
                f"VS={vs:+6.3f} | "
                f"LAT={lat_vel:+6.3f} | "
                f"HEAD={math.degrees(heading_error):+6.3f}deg | "
                f"ROLL={math.degrees(roll):+6.3f}deg | "
                f"AIL={physical_ail:+.6f} | "
                f"RUD={physical_rud:+.6f}"
            )

            next_print += 2.5

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

    reached = (
        crossing is not None
    )

    if reached:
        crossing_alt = float(
            crossing[
                "altitude_ft"
            ]
        )

        crossing_vs = float(
            crossing[
                "vertical_speed_fps"
            ]
        )

        crossing_cross = float(
            crossing[
                "cross_track_ft"
            ]
        )

        crossing_heading = float(
            crossing[
                "heading_error_deg"
            ]
        )

        crossing_lat = float(
            crossing[
                "lateral_speed_fps"
            ]
        )

        crossing_ground = float(
            crossing[
                "ground_forward_ft"
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

        crossing_heading = float(
            "nan"
        )

        crossing_lat = float(
            "nan"
        )

        crossing_ground = float(
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
        PRESENT_CROSS_ALT_MIN
        <=
        crossing_alt
        <=
        PRESENT_CROSS_ALT_MAX

        and
        abs(
            crossing_vs
        )
        <=
        PRESENT_CROSS_MAX_VS
    )

    margin_pass = bool(
        presentation_pass
        and
        max_cross
        <=
        MARGIN_MAX_CROSS
    )

    return {
        "aileron_command":
            float(
                aileron_command
            ),

        "rudder_command":
            float(
                rudder_command
            ),

        "reached_300":
            bool(
                reached
            ),

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
            crossing_alt,

        "crossing_vs":
            crossing_vs,

        "crossing_cross":
            crossing_cross,

        "crossing_heading_deg":
            crossing_heading,

        "crossing_lat":
            crossing_lat,

        "crossing_ground":
            crossing_ground,

        "presentation_pass":
            presentation_pass,

        "margin_pass":
            margin_pass,

        "trace":
            trace,
    }


# ============================================================
# SELECTION
# ============================================================

def selection_key(
    r,
):
    """
    Correct ordering:
      presentation pass first,
      then max geometry,
      then endpoint,
      then heading/lateral quality.
    """

    return (
        0
        if r[
            "presentation_pass"
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

        r[
            "max_abs_heading_deg"
        ],

        r[
            "max_abs_lat"
        ],
    )


def without_trace(
    r,
):
    return {
        k: v
        for k, v
        in r.items()
        if k != "trace"
    }


def print_case(
    label,
    r,
):
    print(
        f"{label:10s} | "
        f"AIL={r['aileron_command']:+.3f} | "
        f"RUD={r['rudder_command']:+.3f} | "
        f"MAX_X={r['max_cross']:6.3f} | "
        f"CROSS_X={r['crossing_cross']:+7.3f} | "
        f"LAT={r['max_abs_lat']:5.3f} | "
        f"HEAD={r['max_abs_heading_deg']:5.3f}deg | "
        f"MIN_ALT={r['min_alt']:7.3f} | "
        f"ALT@300={r['crossing_alt']:7.3f} | "
        f"VS@300={r['crossing_vs']:+6.3f} | "
        f"PASS={r['presentation_pass']} | "
        f"MARGIN={r['margin_pass']}"
    )


# ============================================================
# START
# ============================================================

rule(
    "STAGE 2 LATERAL V4 — MICRO SEARCH AROUND THE ACTUAL PASSING POINT"
)

print(
    "Stage 1: LOCKED"
)

print(
    "Mapped Stage-2 action path: FIXED"
)

print(
    "Altitude teacher: LOCKED"
)

print(
    "No new feedback controller."
)

print(
    "Known V3 passing point: AIL=-0.250, RUD=0.000, MAX_X≈4.76 ft"
)


# ============================================================
# REFERENCE EXACT V3 PASS
# ============================================================

rule(
    "A — EXACT V3 PASS RECHECK"
)

known = run_case(
    -0.250,
    0.000,
)

print_case(
    "KNOWN",
    known,
)

if not known[
    "presentation_pass"
]:
    raise RuntimeError(
        "The previously passing AIL=-0.25,RUD=0 configuration "
        "did not reproduce. Stop before distillation."
    )


# ============================================================
# PHASE B — AILERON MICRO
# ============================================================

rule(
    "B — AILERON-ONLY MICRO SEARCH"
)

ail_rows = []

for index, ail in enumerate(
    AILERON_FINE_VALUES,
    start=1,
):
    r = run_case(
        ail,
        0.0,
    )

    row = without_trace(
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
    AIL_ONLY_CSV,
    ail_rows,
)

ail_sorted = sorted(
    ail_rows,
    key=selection_key,
)

top_ail_values = [
    row[
        "aileron_command"
    ]
    for row in ail_sorted[
        :3
    ]
]

print()
print(
    "Top 3 aileron values:",
    top_ail_values,
)


# ============================================================
# PHASE C — SMALL RUDDER SEARCH AROUND TOP AILERON
# ============================================================

rule(
    "C — TOP-3 AILERON × SMALL RUDDER SEARCH"
)

combo_rows = []

case_no = 0

for ail in top_ail_values:
    for rud in RUDDER_FINE_VALUES:
        case_no += 1

        r = run_case(
            ail,
            rud,
        )

        row = without_trace(
            r
        )

        combo_rows.append(
            row
        )

        print_case(
            f"C{case_no:02d}",
            r,
        )

save_rows(
    COMBO_CSV,
    combo_rows,
)

all_rows = (
    ail_rows
    +
    combo_rows
)

all_sorted = sorted(
    all_rows,
    key=selection_key,
)

best = all_sorted[0]


# ============================================================
# TOP RESULTS
# ============================================================

rule(
    "TOP 12 V4 MICRO CANDIDATES"
)

for rank, row in enumerate(
    all_sorted[
        :12
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
        f"ALT@300={row['crossing_alt']:7.3f} | "
        f"VS@300={row['crossing_vs']:+6.3f} | "
        f"PASS={row['presentation_pass']} | "
        f"MARGIN={row['margin_pass']}"
    )


# ============================================================
# FULL BEST
# ============================================================

rule(
    "BEST V4 TEACHER — FULL DETAILED FLIGHT"
)

print(
    f"AILERON COMMAND = "
    f"{best['aileron_command']:+.3f}"
)

print(
    f"RUDDER COMMAND  = "
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
    BEST_TRACE_CSV,
    best_result[
        "trace"
    ],
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "stage1_model":
        STAGE1_MODEL_PATH,

    "stage2_source_model":
        STAGE2_MODEL_PATH,

    "mapped_environment":
        "HelicopterEnvStage2RefineMapped",

    "mapping": {
        "aileron_scale":
            AILERON_SCALE,

        "rudder_scale":
            RUDDER_SCALE,
    },

    "altitude_teacher": {
        "col_bias":
            COL_BIAS,

        "alt_kp":
            ALT_KP,

        "vs_kd":
            VS_KD,

        "bias_fade_distance_ft":
            COL_BIAS_FADE_DISTANCE,

        "max_action_correction":
            MAX_COL_ACTION_CORR,
    },

    "lateral_teacher": {
        "aileron_command":
            best_result[
                "aileron_command"
            ],

        "rudder_command":
            best_result[
                "rudder_command"
            ],
    },

    "metrics": {
        key:
            value
        for key, value
        in without_trace(
            best_result
        ).items()
    },
}

with BEST_CONFIG_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

rule(
    "STAGE 2 LATERAL V4 MICRO FINAL RESULT"
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
    "margin_pass",
]:
    print(
        f"{key:27s}: "
        f"{best_result[key]}"
    )

print()

if best_result[
    "presentation_pass"
]:
    print(
        "TEACHER PRESENTATION QUALITY: TRUE"
    )

    if best_result[
        "margin_pass"
    ]:
        print(
            "CROSS-TRACK MARGIN TARGET (<=4 ft): TRUE"
        )
    else:
        print(
            "CROSS-TRACK MARGIN TARGET (<=4 ft): FALSE"
        )

    print(
        "Next step: distill the locked altitude teacher and the "
        "selected lateral/yaw commands into a single Stage-2 PPO."
    )

else:
    print(
        "TEACHER PRESENTATION QUALITY: FALSE"
    )

    print(
        "Do not distill. Only then consider a small feedback term."
    )

print()
print(
    "Saved:",
    AIL_ONLY_CSV,
)

print(
    "Saved:",
    COMBO_CSV,
)

print(
    "Saved:",
    BEST_TRACE_CSV,
)

print(
    "Saved:",
    BEST_CONFIG_JSON,
)
