from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill,
)
from helicopter_env_stage2_refine import (
    HelicopterEnvStage2Refine,
)


# ============================================================
# MODELS
# ============================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

# IMPORTANT:
# Use the ORIGINAL successful Stage-2 refine model.
# Do NOT use the failed quality-finetune model from the previous run.
STAGE2_MODEL_PATH = (
    "models_stage2_refine/"
    "AH1S_STAGE2_REFINE_SUCCESS.zip"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path(
    "results_stage2_handoff_teacher_v1"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALT_CSV = OUT_DIR / "phase_a_altitude_search.csv"
LAT_CSV = OUT_DIR / "phase_b_lateral_search.csv"
BEST_CSV = OUT_DIR / "best_teacher_trace.csv"


# ============================================================
# MISSION / QUALITY TARGETS
# ============================================================

TARGET_ALT = 300.0
TARGET_DISTANCE = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0

HANDOFF_ALT_MIN = 295.0
HANDOFF_ALT_MAX = 305.0
HANDOFF_MAX_VS = 0.50
HANDOFF_MAX_HS = 1.0
HANDOFF_MAX_DRIFT = 3.0
HANDOFF_STABLE_TIME = 5.0

PRESENT_MIN_ALT = 290.0
PRESENT_MAX_ALT = 310.0
PRESENT_MAX_CROSS = 5.0
PRESENT_CROSS_ALT_MIN = 295.0
PRESENT_CROSS_ALT_MAX = 305.0
PRESENT_CROSS_MAX_VS = 2.0

EARTH_RADIUS_FT = 20_902_231.0


# ============================================================
# TEACHER SEARCH
#
# Teacher is ONLY a calibration tool.
# If it works, we later distill its behavior into PPO.
#
# Collective correction:
#
#   + start/feed-forward bias that fades with forward distance
#   + altitude error feedback
#   + vertical-speed damping
#
# All corrections here are PPO-normalized action residuals.
# ============================================================

COL_BIAS_VALUES = [
    0.00,
    0.08,
    0.16,
    0.24,
]

ALT_KP_VALUES = [
    0.010,
    0.020,
    0.030,
]

VS_KD_VALUES = [
    0.040,
    0.080,
    0.120,
]

COL_BIAS_FADE_DISTANCE = 240.0
MAX_COL_ACTION_CORR = 0.40


# Lateral correction is calibrated AFTER the best altitude teacher.
#
# We do not use cross-track position as a controller input because
# Stage-2 PPO does not observe cross-track position directly.
# We only use roll and lateral velocity, both observable by Stage 2.
AIL_SIGN_VALUES = [
    -1.0,
    +1.0,
]

ROLL_K_VALUES = [
    0.5,
    1.5,
    3.0,
]

LAT_K_VALUES = [
    0.020,
    0.050,
    0.080,
    0.120,
]

MAX_AIL_ACTION_CORR = 0.40


# ============================================================
# HELPERS
# ============================================================

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
        "Active JSBSim FDM could not be found."
    )


def latitude_deg(fdm):
    for key in [
        "position/lat-gc-deg",
        "position/lat-geod-deg",
    ]:
        value = fdm_float(
            fdm,
            key,
        )
        if np.isfinite(value):
            return value

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
        value = fdm_float(
            fdm,
            key,
        )
        if np.isfinite(value):
            return value

    return float("nan")


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
    c = math.cos(heading)
    s = math.sin(heading)

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


def horizontal_speed_stage1(info):
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

    return float(
        np.hypot(vn, ve)
    )


def print_rule(title):
    print()
    print("=" * 132)
    print(title)
    print("=" * 132)


# ============================================================
# LOAD ONCE
# ============================================================

stage1_model = PPO.load(
    STAGE1_MODEL_PATH
)

stage2_model = PPO.load(
    STAGE2_MODEL_PATH
)


# ============================================================
# STAGE-1 HANDOFF BUILDER
# ============================================================

def build_stage1_handoff():
    env1 = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False,
        )
    )

    obs1, info1 = env1.reset()

    fdm = get_fdm(env1)

    initial_heading = heading_rad(
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
                obs1,
                deterministic=True,
            )
        )

        (
            obs1,
            reward,
            terminated,
            truncated,
            info1,
        ) = env1.step(
            action
        )

        altitude = info_float(
            info1,
            "altitude",
        )

        vs = info_float(
            info1,
            "vertical_speed",
        )

        drift = info_float(
            info1,
            "drift",
            999.0,
        )

        hs = horizontal_speed_stage1(
            info1
        )

        stable = (
            HANDOFF_ALT_MIN
            <= altitude
            <= HANDOFF_ALT_MAX
            and
            abs(vs)
            <= HANDOFF_MAX_VS
            and
            hs
            <= HANDOFF_MAX_HS
            and
            drift
            <= HANDOFF_MAX_DRIFT
        )

        if stable:
            stable_time += dt
        else:
            stable_time = 0.0

        if stable_time >= HANDOFF_STABLE_TIME:
            handoff = {
                "altitude": altitude,
                "vs": vs,
                "hs": hs,
                "drift": drift,
                "time": (step + 1) * dt,
            }
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
            break

        if truncated:
            break

    if handoff is None:
        env1.close()
        raise RuntimeError(
            "Stage-1 stable handoff was not reached."
        )

    return (
        env1,
        fdm,
        initial_heading,
        handoff,
    )


# ============================================================
# PREPARE STAGE 2 ON SAME FDM
# ============================================================

def attach_stage2(
    active_fdm,
    mission_heading,
):
    env2 = (
        HelicopterEnvStage2Refine()
    )

    # Disposable reset only initializes Stage-2 Python bookkeeping.
    env2.reset()

    # Attach the real aircraft.
    env2.fdm = active_fdm

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

    obs2 = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    return (
        env2,
        obs2,
    )


# ============================================================
# TEACHER
# ============================================================

def collective_teacher_correction(
    altitude,
    vertical_speed,
    forward_distance,
    col_bias,
    alt_kp,
    vs_kd,
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

    corr = (
        col_bias
        *
        distance_gate
        +
        alt_kp
        *
        altitude_error
        -
        vs_kd
        *
        vertical_speed
    )

    return float(
        np.clip(
            corr,
            -MAX_COL_ACTION_CORR,
            +MAX_COL_ACTION_CORR,
        )
    )


def aileron_teacher_correction(
    roll,
    lateral_velocity,
    sign,
    roll_k,
    lat_k,
):
    raw = (
        roll_k
        *
        roll
        +
        lat_k
        *
        lateral_velocity
    )

    corr = (
        sign
        *
        raw
    )

    return float(
        np.clip(
            corr,
            -MAX_AIL_ACTION_CORR,
            +MAX_AIL_ACTION_CORR,
        )
    )


# ============================================================
# ONE TRUE CONTINUOUS FLIGHT
# ============================================================

def run_case(
    col_bias=0.0,
    alt_kp=0.0,
    vs_kd=0.0,
    use_lateral=False,
    ail_sign=1.0,
    roll_k=0.0,
    lat_k=0.0,
    detailed=False,
):
    (
        env1,
        fdm,
        mission_heading,
        handoff,
    ) = build_stage1_handoff()

    handoff_lat = latitude_deg(
        fdm
    )
    handoff_lon = longitude_deg(
        fdm
    )

    stage1_fdm_id = id(fdm)

    env2, obs2 = attach_stage2(
        fdm,
        mission_heading,
    )

    if id(
        get_fdm(env2)
    ) != stage1_fdm_id:
        raise RuntimeError(
            "FDM identity changed at handoff."
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

    min_alt = handoff[
        "altitude"
    ]
    max_alt = handoff[
        "altitude"
    ]

    max_cross = 0.0
    max_abs_lat = 0.0
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
                obs2,
                deterministic=True,
            )
        )

        action = np.asarray(
            base_action,
            dtype=np.float32,
        ).reshape(-1).copy()

        altitude_before = fdm_float(
            fdm,
            "position/h-agl-ft",
        )

        vs_before = fdm_float(
            fdm,
            "velocities/h-dot-fps",
        )

        # The environment owns mission distance accounting.
        distance_before = float(
            getattr(
                env2,
                "forward_distance",
                0.0,
            )
        )

        roll_before = fdm_float(
            fdm,
            "attitude/roll-rad",
            0.0,
        )

        lat_before = fdm_float(
            fdm,
            "velocities/v-aero-fps",
            0.0,
        )

        col_corr = (
            collective_teacher_correction(
                altitude_before,
                vs_before,
                distance_before,
                col_bias,
                alt_kp,
                vs_kd,
            )
        )

        action[0] = float(
            np.clip(
                action[0]
                +
                col_corr,
                -1.0,
                +1.0,
            )
        )

        ail_corr = 0.0

        if use_lateral:
            ail_corr = (
                aileron_teacher_correction(
                    roll_before,
                    lat_before,
                    ail_sign,
                    roll_k,
                    lat_k,
                )
            )

            action[2] = float(
                np.clip(
                    action[2]
                    +
                    ail_corr,
                    -1.0,
                    +1.0,
                )
            )

        (
            obs2,
            reward,
            terminated,
            truncated,
            info2,
        ) = env2.step(
            action
        )

        obs2 = np.asarray(
            obs2,
            dtype=np.float32,
        )

        t = (
            step + 1
        ) * dt

        altitude = info_float(
            info2,
            "altitude",
            fdm_float(
                fdm,
                "position/h-agl-ft",
            ),
        )

        vs = info_float(
            info2,
            "vertical_speed",
            fdm_float(
                fdm,
                "velocities/h-dot-fps",
            ),
        )

        fwd = info_float(
            info2,
            "forward_velocity",
            fdm_float(
                fdm,
                "velocities/u-aero-fps",
                0.0,
            ),
        )

        lat_vel = info_float(
            info2,
            "lateral_velocity",
            fdm_float(
                fdm,
                "velocities/v-aero-fps",
                0.0,
            ),
        )

        roll = info_float(
            info2,
            "roll",
            fdm_float(
                fdm,
                "attitude/roll-rad",
                0.0,
            ),
        )

        pitch = info_float(
            info2,
            "pitch",
            fdm_float(
                fdm,
                "attitude/pitch-rad",
                0.0,
            ),
        )

        distance = info_float(
            info2,
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

        north, east = local_ne_ft(
            lat,
            lon,
            handoff_lat,
            handoff_lon,
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
            abs(cross),
        )

        max_abs_lat = max(
            max_abs_lat,
            abs(lat_vel),
        )

        max_abs_roll = max(
            max_abs_roll,
            abs(roll),
        )

        row = {
            "time_s": t,
            "distance_ft": distance,
            "ground_forward_ft": ground_forward,
            "cross_track_ft": cross,
            "altitude_ft": altitude,
            "vertical_speed_fps": vs,
            "forward_speed_fps": fwd,
            "lateral_speed_fps": lat_vel,
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "base_a0": float(base_action[0]),
            "base_a1": float(base_action[1]),
            "base_a2": float(base_action[2]),
            "base_a3": float(base_action[3]),
            "col_corr": col_corr,
            "ail_corr": ail_corr,
            "used_a0": float(action[0]),
            "used_a2": float(action[2]),
            "physical_collective": info_float(
                info2,
                "collective",
                fdm_float(
                    fdm,
                    "fcs/collective-cmd-norm",
                ),
            ),
            "physical_aileron": info_float(
                info2,
                "aileron",
                fdm_float(
                    fdm,
                    "fcs/aileron-cmd-norm",
                ),
            ),
        }

        trace.append(row)

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"D={distance:7.2f} | "
                f"GND={ground_forward:7.2f} | "
                f"X={cross:+7.2f} | "
                f"ALT={altitude:7.2f} | "
                f"VS={vs:+6.2f} | "
                f"FWD={fwd:+6.2f} | "
                f"LAT={lat_vel:+6.2f} | "
                f"ROLL={math.degrees(roll):+6.2f} | "
                f"COLcorr={col_corr:+6.3f} | "
                f"AILcorr={ail_corr:+6.3f}"
            )
            next_print += 2.5

        if distance >= TARGET_DISTANCE:
            crossing = row.copy()
            break

        if terminated:
            if not bool(
                info2.get(
                    "success",
                    False,
                )
            ):
                failure = True
                termination_reason = str(
                    info2.get(
                        "termination_reason",
                        "terminated",
                    )
                )

            # If environment success occurs before our 300-ft crossing,
            # keep the result as non-crossing rather than pretending pass.
            break

        if truncated:
            termination_reason = "truncated"
            break

    # Avoid env2.close() touching the shared active FDM object.
    env2.fdm = None
    env1.close()

    reached = (
        crossing is not None
    )

    max_drop = (
        handoff["altitude"]
        -
        min_alt
    )

    if reached:
        crossing_alt = crossing[
            "altitude_ft"
        ]
        crossing_vs = crossing[
            "vertical_speed_fps"
        ]
        crossing_cross = crossing[
            "cross_track_ft"
        ]
        crossing_ground = crossing[
            "ground_forward_ft"
        ]
        crossing_time = crossing[
            "time_s"
        ]
    else:
        crossing_alt = float("nan")
        crossing_vs = float("nan")
        crossing_cross = float("nan")
        crossing_ground = float("nan")
        crossing_time = float("nan")

    present_pass = bool(
        reached
        and
        not failure
        and
        min_alt >= PRESENT_MIN_ALT
        and
        max_alt <= PRESENT_MAX_ALT
        and
        max_cross <= PRESENT_MAX_CROSS
        and
        PRESENT_CROSS_ALT_MIN
        <= crossing_alt
        <= PRESENT_CROSS_ALT_MAX
        and
        abs(crossing_vs)
        <= PRESENT_CROSS_MAX_VS
    )

    result = {
        "reached_300": reached,
        "failure": failure,
        "termination_reason": termination_reason,
        "handoff_alt": handoff["altitude"],
        "handoff_vs": handoff["vs"],
        "min_alt": min_alt,
        "max_alt": max_alt,
        "max_drop": max_drop,
        "max_cross": max_cross,
        "max_abs_lat": max_abs_lat,
        "max_abs_roll_deg": math.degrees(
            max_abs_roll
        ),
        "crossing_alt": crossing_alt,
        "crossing_vs": crossing_vs,
        "crossing_cross": crossing_cross,
        "crossing_ground": crossing_ground,
        "crossing_time": crossing_time,
        "presentation_pass": present_pass,
        "trace": trace,
    }

    return result


# ============================================================
# SCORES
# ============================================================

def altitude_score(r):
    if not r["reached_300"]:
        return (
            1_000_000.0
            +
            10_000.0
            *
            max(
                0.0,
                PRESENT_MIN_ALT
                -
                r["min_alt"],
            )
        )

    corridor_low = max(
        0.0,
        PRESENT_MIN_ALT
        -
        r["min_alt"],
    )

    corridor_high = max(
        0.0,
        r["max_alt"]
        -
        PRESENT_MAX_ALT,
    )

    crossing_alt_err = abs(
        r["crossing_alt"]
        -
        TARGET_ALT
    )

    return (
        1200.0 * corridor_low
        +
        900.0 * corridor_high
        +
        140.0 * r["max_drop"]
        +
        120.0 * crossing_alt_err
        +
        40.0 * abs(
            r["crossing_vs"]
        )
        +
        5.0 * r["max_cross"]
    )


def lateral_score(r):
    if not r["reached_300"]:
        return 1_000_000.0

    low_alt = max(
        0.0,
        PRESENT_MIN_ALT
        -
        r["min_alt"],
    )

    high_alt = max(
        0.0,
        r["max_alt"]
        -
        PRESENT_MAX_ALT,
    )

    return (
        900.0 * r["max_cross"]
        +
        250.0 * abs(
            r["crossing_cross"]
        )
        +
        120.0 * r["max_abs_lat"]
        +
        40.0 * r["max_abs_roll_deg"]
        +
        1200.0 * low_alt
        +
        900.0 * high_alt
        +
        50.0 * abs(
            r["crossing_alt"]
            -
            TARGET_ALT
        )
    )


def save_rows(
    path,
    rows,
):
    if not rows:
        return

    fields = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# BASELINE
# ============================================================

print_rule(
    "STAGE 2 CONTINUOUS-HANDOFF TEACHER CALIBRATION V1"
)

print(
    "NO TRAINING."
)

print(
    "Stage 1 remains locked."
)

print(
    "Source Stage 2 remains untouched."
)

print(
    "\nStage 1:",
    STAGE1_MODEL_PATH,
)

print(
    "Stage 2:",
    STAGE2_MODEL_PATH,
)

print_rule(
    "BASELINE — ORIGINAL STAGE 2, FULL 300-FT FLIGHT"
)

baseline = run_case()

for key in [
    "reached_300",
    "failure",
    "handoff_alt",
    "min_alt",
    "max_alt",
    "max_drop",
    "max_cross",
    "max_abs_lat",
    "max_abs_roll_deg",
    "crossing_alt",
    "crossing_vs",
    "crossing_cross",
    "crossing_ground",
    "presentation_pass",
]:
    print(
        f"{key:24s}: "
        f"{baseline[key]}"
    )


# ============================================================
# PHASE A — ALTITUDE
# ============================================================

print_rule(
    "PHASE A — COLLECTIVE TEACHER SEARCH"
)

print(
    "Goal: remove the 30+ ft transition dive while preserving forward flight."
)

alt_results = []

case_no = 0

for col_bias in COL_BIAS_VALUES:
    for alt_kp in ALT_KP_VALUES:
        for vs_kd in VS_KD_VALUES:
            case_no += 1

            r = run_case(
                col_bias=col_bias,
                alt_kp=alt_kp,
                vs_kd=vs_kd,
                use_lateral=False,
            )

            score = altitude_score(r)

            row = {
                "case": case_no,
                "score": score,
                "col_bias": col_bias,
                "alt_kp": alt_kp,
                "vs_kd": vs_kd,
                "reached_300": r["reached_300"],
                "failure": r["failure"],
                "min_alt": r["min_alt"],
                "max_alt": r["max_alt"],
                "max_drop": r["max_drop"],
                "max_cross": r["max_cross"],
                "crossing_alt": r["crossing_alt"],
                "crossing_vs": r["crossing_vs"],
                "crossing_cross": r["crossing_cross"],
                "presentation_pass": r[
                    "presentation_pass"
                ],
            }

            alt_results.append(row)

            print(
                f"A{case_no:02d} | "
                f"B={col_bias:.2f} "
                f"KP={alt_kp:.3f} "
                f"KD={vs_kd:.3f} | "
                f"MIN={r['min_alt']:.2f} "
                f"DROP={r['max_drop']:.2f} "
                f"X={r['max_cross']:.2f} "
                f"CROSS_ALT={r['crossing_alt']:.2f} "
                f"VS={r['crossing_vs']:+.2f} "
                f"REACH={r['reached_300']} "
                f"SCORE={score:.1f}"
            )

save_rows(
    ALT_CSV,
    alt_results,
)

alt_sorted = sorted(
    alt_results,
    key=lambda x: x["score"],
)

print_rule(
    "TOP 10 PHASE-A ALTITUDE CANDIDATES"
)

for rank, row in enumerate(
    alt_sorted[:10],
    start=1,
):
    print(
        f"{rank:2d}. "
        f"B={row['col_bias']:.2f} "
        f"KP={row['alt_kp']:.3f} "
        f"KD={row['vs_kd']:.3f} | "
        f"MIN={row['min_alt']:.2f} "
        f"DROP={row['max_drop']:.2f} "
        f"CROSS_ALT={row['crossing_alt']:.2f} "
        f"VS={row['crossing_vs']:+.2f} "
        f"X={row['max_cross']:.2f} "
        f"SCORE={row['score']:.1f}"
    )

best_alt = alt_sorted[0]

print(
    "\nBEST ALTITUDE TEACHER:"
)

print(best_alt)


# ============================================================
# PHASE B — LATERAL
# ============================================================

print_rule(
    "PHASE B — AILERON / LATERAL TEACHER SEARCH"
)

print(
    "Best Phase-A collective teacher is fixed."
)

print(
    "Now search roll + lateral-velocity damping only."
)

lat_results = []

case_no = 0

for sign in AIL_SIGN_VALUES:
    for roll_k in ROLL_K_VALUES:
        for lat_k in LAT_K_VALUES:
            case_no += 1

            r = run_case(
                col_bias=best_alt[
                    "col_bias"
                ],
                alt_kp=best_alt[
                    "alt_kp"
                ],
                vs_kd=best_alt[
                    "vs_kd"
                ],
                use_lateral=True,
                ail_sign=sign,
                roll_k=roll_k,
                lat_k=lat_k,
            )

            score = lateral_score(r)

            row = {
                "case": case_no,
                "score": score,
                "ail_sign": sign,
                "roll_k": roll_k,
                "lat_k": lat_k,
                "col_bias": best_alt[
                    "col_bias"
                ],
                "alt_kp": best_alt[
                    "alt_kp"
                ],
                "vs_kd": best_alt[
                    "vs_kd"
                ],
                "reached_300": r["reached_300"],
                "failure": r["failure"],
                "min_alt": r["min_alt"],
                "max_alt": r["max_alt"],
                "max_drop": r["max_drop"],
                "max_cross": r["max_cross"],
                "max_abs_lat": r[
                    "max_abs_lat"
                ],
                "max_abs_roll_deg": r[
                    "max_abs_roll_deg"
                ],
                "crossing_alt": r[
                    "crossing_alt"
                ],
                "crossing_vs": r[
                    "crossing_vs"
                ],
                "crossing_cross": r[
                    "crossing_cross"
                ],
                "presentation_pass": r[
                    "presentation_pass"
                ],
            }

            lat_results.append(row)

            print(
                f"B{case_no:02d} | "
                f"SIGN={sign:+.0f} "
                f"RK={roll_k:.2f} "
                f"LK={lat_k:.3f} | "
                f"MIN={r['min_alt']:.2f} "
                f"DROP={r['max_drop']:.2f} "
                f"MAX_X={r['max_cross']:.2f} "
                f"LAT={r['max_abs_lat']:.2f} "
                f"ROLL={r['max_abs_roll_deg']:.2f}deg "
                f"CROSS_X={r['crossing_cross']:+.2f} "
                f"REACH={r['reached_300']} "
                f"SCORE={score:.1f}"
            )

save_rows(
    LAT_CSV,
    lat_results,
)

lat_sorted = sorted(
    lat_results,
    key=lambda x: x["score"],
)

print_rule(
    "TOP 10 PHASE-B LATERAL CANDIDATES"
)

for rank, row in enumerate(
    lat_sorted[:10],
    start=1,
):
    print(
        f"{rank:2d}. "
        f"SIGN={row['ail_sign']:+.0f} "
        f"RK={row['roll_k']:.2f} "
        f"LK={row['lat_k']:.3f} | "
        f"MIN={row['min_alt']:.2f} "
        f"DROP={row['max_drop']:.2f} "
        f"MAX_X={row['max_cross']:.2f} "
        f"CROSS_X={row['crossing_cross']:+.2f} "
        f"ALT={row['crossing_alt']:.2f} "
        f"VS={row['crossing_vs']:+.2f} "
        f"SCORE={row['score']:.1f}"
    )

best = lat_sorted[0]

print_rule(
    "BEST COMBINED TEACHER — FULL DETAILED FLIGHT"
)

print(
    "Collective:",
    f"B={best['col_bias']:.2f}, "
    f"KP={best['alt_kp']:.3f}, "
    f"KD={best['vs_kd']:.3f}",
)

print(
    "Lateral:",
    f"SIGN={best['ail_sign']:+.0f}, "
    f"ROLL_K={best['roll_k']:.2f}, "
    f"LAT_K={best['lat_k']:.3f}",
)

best_result = run_case(
    col_bias=best[
        "col_bias"
    ],
    alt_kp=best[
        "alt_kp"
    ],
    vs_kd=best[
        "vs_kd"
    ],
    use_lateral=True,
    ail_sign=best[
        "ail_sign"
    ],
    roll_k=best[
        "roll_k"
    ],
    lat_k=best[
        "lat_k"
    ],
    detailed=True,
)

save_rows(
    BEST_CSV,
    best_result[
        "trace"
    ],
)

print_rule(
    "STAGE 2 TEACHER V1 FINAL RESULT"
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
    "max_abs_roll_deg",
    "crossing_alt",
    "crossing_vs",
    "crossing_cross",
    "crossing_ground",
    "crossing_time",
    "presentation_pass",
]:
    print(
        f"{key:24s}: "
        f"{best_result[key]}"
    )

print()

if best_result[
    "presentation_pass"
]:
    print(
        "PRESENTATION QUALITY: TRUE"
    )
    print(
        "Next step: distill collective + aileron teacher behavior "
        "into the Stage-2 PPO action heads."
    )
else:
    print(
        "PRESENTATION QUALITY: FALSE"
    )
    print(
        "Do NOT train yet."
    )
    print(
        "Use the Phase-A / Phase-B top tables to refine only the "
        "remaining control axis."
    )

print()
print(
    "Saved:",
    ALT_CSV,
)
print(
    "Saved:",
    LAT_CSV,
)
print(
    "Saved:",
    BEST_CSV,
)
