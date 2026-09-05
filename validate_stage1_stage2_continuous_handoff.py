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
from helicopter_env_stage2_refine import (
    HelicopterEnvStage2Refine,
)


# =====================================================================
# PATHS
# =====================================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL_PATH = (
    "models_stage2_refine/"
    "AH1S_STAGE2_REFINE_SUCCESS.zip"
)

RESULT_DIR = Path(
    "results_stage1_stage2_handoff"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRACE_CSV = RESULT_DIR / "continuous_handoff_trace.csv"
SUMMARY_JSON = RESULT_DIR / "continuous_handoff_summary.json"


# =====================================================================
# VALIDATION SETTINGS
# =====================================================================

# Stage 1 must first establish a genuinely stable hover.
MAX_STAGE1_TIME = 120.0
REQUIRED_STABLE_TIME = 5.0

HANDOFF_ALT_MIN = 295.0
HANDOFF_ALT_MAX = 305.0
HANDOFF_MAX_ABS_VS = 0.50
HANDOFF_MAX_HSPEED = 1.00
HANDOFF_MAX_DRIFT = 3.00

# Stage 2 is only a validation here. No training.
MAX_STAGE2_TIME = 70.0

TARGET_FORWARD_DISTANCE = 300.0

# Presentation-quality criteria.
ALTITUDE_CORRIDOR_MIN = 290.0
ALTITUDE_CORRIDOR_MAX = 310.0
TARGET_CROSSING_ALT_MIN = 295.0
TARGET_CROSSING_ALT_MAX = 305.0
MAX_CROSS_TRACK = 5.0
MAX_TARGET_CROSSING_ABS_VS = 2.0

# Geometry.
EARTH_RADIUS_FT = 20_902_231.0

PRINT_EVERY_STAGE1 = 5.0
PRINT_EVERY_STAGE2 = 2.5


# =====================================================================
# HELPERS
# =====================================================================

def as_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return float(default)


def info_float(
    info,
    *keys,
    default=float("nan"),
):
    for key in keys:
        if key in info:
            try:
                return float(info[key])
            except Exception:
                pass

    return float(default)


def get_active_fdm(env):
    """
    Return the exact JSBSim FGFDMExec used by an environment.

    Stage-1 variants have existed both as direct envs and wrappers,
    so this deliberately supports both layouts.
    """

    direct = getattr(
        env,
        "fdm",
        None,
    )

    if direct is not None:
        return direct

    base_env = getattr(
        env,
        "base_env",
        None,
    )

    if base_env is not None:
        nested = getattr(
            base_env,
            "fdm",
            None,
        )

        if nested is not None:
            return nested

    raise RuntimeError(
        "Could not locate the active JSBSim FDM object."
    )


def fdm_float(
    fdm,
    key,
    default=float("nan"),
):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def sim_time_seconds(fdm):
    candidates = [
        "simulation/sim-time-sec",
        "simulation/sim-time-sec",
    ]

    for key in candidates:
        value = fdm_float(
            fdm,
            key,
            default=float("nan"),
        )

        if np.isfinite(value):
            return value

    return float("nan")


def heading_rad(fdm):
    return fdm_float(
        fdm,
        "attitude/psi-rad",
        default=float("nan"),
    )


def latitude_deg(fdm):
    for key in [
        "position/lat-gc-deg",
        "position/lat-geod-deg",
    ]:
        value = fdm_float(
            fdm,
            key,
            default=float("nan"),
        )

        if np.isfinite(value):
            return value

    return float("nan")


def longitude_deg(fdm):
    return fdm_float(
        fdm,
        "position/long-gc-deg",
        default=float("nan"),
    )


def local_ne_ft(
    lat_deg,
    lon_deg,
    ref_lat_deg,
    ref_lon_deg,
):
    """
    Small-area local tangent-plane approximation.

    Returns:
        north_ft, east_ft
    """

    if not all(
        np.isfinite(
            [
                lat_deg,
                lon_deg,
                ref_lat_deg,
                ref_lon_deg,
            ]
        )
    ):
        return (
            float("nan"),
            float("nan"),
        )

    dlat = math.radians(
        lat_deg - ref_lat_deg
    )

    dlon = math.radians(
        lon_deg - ref_lon_deg
    )

    ref_lat_rad = math.radians(
        ref_lat_deg
    )

    north = (
        EARTH_RADIUS_FT
        * dlat
    )

    east = (
        EARTH_RADIUS_FT
        * math.cos(ref_lat_rad)
        * dlon
    )

    return (
        float(north),
        float(east),
    )


def project_to_mission_axes(
    north_ft,
    east_ft,
    mission_heading_rad,
):
    """
    JSBSim heading is clockwise from North.

    forward = projection onto heading vector
    cross   = right-positive cross-track
    """

    if not all(
        np.isfinite(
            [
                north_ft,
                east_ft,
                mission_heading_rad,
            ]
        )
    ):
        return (
            float("nan"),
            float("nan"),
        )

    c = math.cos(
        mission_heading_rad
    )

    s = math.sin(
        mission_heading_rad
    )

    forward = (
        north_ft * c
        +
        east_ft * s
    )

    cross_track = (
        -north_ft * s
        +
        east_ft * c
    )

    return (
        float(forward),
        float(cross_track),
    )


def horizontal_speed_stage1(info):
    vn = info_float(
        info,
        "vn",
        default=float("nan"),
    )

    ve = info_float(
        info,
        "ve",
        default=float("nan"),
    )

    if np.isfinite(vn) and np.isfinite(ve):
        return float(
            np.hypot(
                vn,
                ve,
            )
        )

    fwd = info_float(
        info,
        "forward_velocity",
        default=0.0,
    )

    lat = info_float(
        info,
        "lateral_velocity",
        default=0.0,
    )

    return float(
        np.hypot(
            fwd,
            lat,
        )
    )


def stage1_state_line(
    t,
    info,
):
    altitude = info_float(
        info,
        "altitude",
    )

    vs = info_float(
        info,
        "vertical_speed",
    )

    drift = info_float(
        info,
        "drift",
    )

    north = info_float(
        info,
        "north",
    )

    east = info_float(
        info,
        "east",
    )

    hs = horizontal_speed_stage1(
        info
    )

    return (
        f"S1 | t={t:6.2f}s | "
        f"ALT={altitude:7.2f} | "
        f"VS={vs:+6.3f} | "
        f"HS={hs:6.3f} | "
        f"N={north:+7.2f} | "
        f"E={east:+7.2f} | "
        f"DRIFT={drift:6.2f}"
    )


def print_header(text):
    print(
        "\n"
        +
        "=" * 146
    )

    print(text)

    print(
        "=" * 146
    )


# =====================================================================
# LOAD POLICIES
# =====================================================================

print_header(
    "STAGE 1 -> STAGE 2 TRUE CONTINUOUS JSBSIM HANDOFF VALIDATION"
)

print(
    "\nNO TRAINING."
)

print(
    "NO Stage-1 runtime teacher / PD / classical XY helper."
)

print(
    "NO reset of the ACTIVE JSBSim FDM at handoff."
)

print(
    "\nStage 1:"
)

print(
    STAGE1_MODEL_PATH
)

print(
    "\nStage 2:"
)

print(
    STAGE2_MODEL_PATH
)

stage1_model = PPO.load(
    STAGE1_MODEL_PATH
)

stage2_model = PPO.load(
    STAGE2_MODEL_PATH
)

print(
    "\nStage-1 observation shape:",
    stage1_model.observation_space.shape,
)

print(
    "Stage-2 observation shape:",
    stage2_model.observation_space.shape,
)

print(
    "Stage-1 action shape     :",
    stage1_model.action_space.shape,
)

print(
    "Stage-2 action shape     :",
    stage2_model.action_space.shape,
)

if stage1_model.action_space.shape != (4,):
    raise RuntimeError(
        "Stage-1 model is not a 4-action policy."
    )

if stage2_model.action_space.shape != (4,):
    raise RuntimeError(
        "Stage-2 model is not a 4-action policy."
    )


# =====================================================================
# STAGE 1 — REAL TAKEOFF AND STABLE HOVER
# =====================================================================

print_header(
    "PHASE A — STAGE 1 LOCKED PPO: TAKEOFF -> 300 FT -> STABLE HOVER"
)

stage1_env = (
    HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )
)

obs1, info1 = stage1_env.reset()

stage1_fdm = get_active_fdm(
    stage1_env
)

stage1_fdm_id = id(
    stage1_fdm
)

mission_heading = heading_rad(
    stage1_fdm
)

takeoff_lat = latitude_deg(
    stage1_fdm
)

takeoff_lon = longitude_deg(
    stage1_fdm
)

stage1_dt = as_float(
    getattr(
        stage1_env,
        "dt",
        0.075,
    ),
    0.075,
)

if not np.isfinite(stage1_dt) or stage1_dt <= 0.0:
    stage1_dt = 0.075

max_stage1_steps = int(
    math.ceil(
        MAX_STAGE1_TIME
        /
        stage1_dt
    )
)

stable_seconds = 0.0
handoff_reached = False

next_stage1_print = 0.0

stage1_last_info = info1
handoff_stage1_action = None
handoff_time_stage1 = None

for step in range(
    max_stage1_steps
):
    action1, _ = (
        stage1_model.predict(
            obs1,
            deterministic=True,
        )
    )

    action1 = np.asarray(
        action1,
        dtype=np.float32,
    ).reshape(-1)

    (
        obs1,
        reward1,
        terminated1,
        truncated1,
        info1,
    ) = stage1_env.step(
        action1
    )

    t1 = (
        (step + 1)
        *
        stage1_dt
    )

    stage1_last_info = info1
    handoff_stage1_action = (
        action1.copy()
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
        default=float("inf"),
    )

    hs = horizontal_speed_stage1(
        info1
    )

    stable_now = (
        HANDOFF_ALT_MIN
        <= altitude
        <= HANDOFF_ALT_MAX
        and
        abs(vs)
        <= HANDOFF_MAX_ABS_VS
        and
        hs
        <= HANDOFF_MAX_HSPEED
        and
        drift
        <= HANDOFF_MAX_DRIFT
    )

    if stable_now:
        stable_seconds += (
            stage1_dt
        )
    else:
        stable_seconds = 0.0

    if t1 >= next_stage1_print:
        print(
            stage1_state_line(
                t1,
                info1,
            )
            +
            f" | STABLE={stable_seconds:4.1f}s"
        )

        next_stage1_print += (
            PRINT_EVERY_STAGE1
        )

    if (
        terminated1
        and
        not bool(
            info1.get(
                "success",
                False,
            )
        )
    ):
        raise RuntimeError(
            "Stage 1 suffered a physical failure before handoff. "
            f"Info: {info1}"
        )

    if stable_seconds >= REQUIRED_STABLE_TIME:
        handoff_reached = True
        handoff_time_stage1 = t1
        break

    if truncated1:
        raise RuntimeError(
            "Stage 1 truncated before a stable handoff state was found."
        )


if not handoff_reached:
    raise RuntimeError(
        "No presentation-quality Stage-1 handoff state was found "
        f"within {MAX_STAGE1_TIME:.1f} seconds."
    )


# =====================================================================
# CAPTURE EXACT HANDOFF STATE
# =====================================================================

active_fdm = get_active_fdm(
    stage1_env
)

if id(active_fdm) != stage1_fdm_id:
    raise RuntimeError(
        "Stage-1 FDM object unexpectedly changed before handoff."
    )

handoff_sim_time = sim_time_seconds(
    active_fdm
)

handoff_lat = latitude_deg(
    active_fdm
)

handoff_lon = longitude_deg(
    active_fdm
)

handoff_heading = heading_rad(
    active_fdm
)

handoff_altitude = info_float(
    stage1_last_info,
    "altitude",
)

handoff_vs = info_float(
    stage1_last_info,
    "vertical_speed",
)

handoff_hspeed = (
    horizontal_speed_stage1(
        stage1_last_info
    )
)

handoff_drift = info_float(
    stage1_last_info,
    "drift",
)

handoff_north = info_float(
    stage1_last_info,
    "north",
)

handoff_east = info_float(
    stage1_last_info,
    "east",
)

print_header(
    "HANDOFF SNAPSHOT — LAST STAGE-1 STATE"
)

print(
    f"Stage-1 flight time : {handoff_time_stage1:.3f} s"
)

print(
    f"JSBSim sim time     : {handoff_sim_time:.6f} s"
)

print(
    f"FDM object id       : {stage1_fdm_id}"
)

print(
    f"Altitude            : {handoff_altitude:.3f} ft"
)

print(
    f"Vertical speed      : {handoff_vs:+.3f} ft/s"
)

print(
    f"Horizontal speed    : {handoff_hspeed:.3f} ft/s"
)

print(
    f"Takeoff drift       : {handoff_drift:.3f} ft"
)

print(
    f"North / East        : {handoff_north:+.3f} / {handoff_east:+.3f} ft"
)

print(
    f"Mission heading     : {math.degrees(mission_heading):.3f} deg"
)

print(
    f"Current heading     : {math.degrees(handoff_heading):.3f} deg"
)

print(
    "Last Stage-1 action:",
    np.array2string(
        handoff_stage1_action,
        precision=5,
        floatmode="fixed",
    ),
)


# =====================================================================
# PREPARE STAGE 2 INTERNAL STATE WITHOUT TOUCHING ACTIVE FDM
# =====================================================================

print_header(
    "PHASE B — ATTACH STAGE 2 REFINE POLICY TO THE SAME ACTIVE JSBSIM FDM"
)

# We intentionally allow Stage-2 reset() to initialize its Python-side
# bookkeeping on its OWN disposable FDM. That disposable FDM is NOT the
# aircraft currently flying under Stage 1.
#
# Immediately after that, Stage 2 is attached to Stage 1's still-running
# FDM object. Therefore the real aircraft state is never reset.

stage2_env = (
    HelicopterEnvStage2Refine()
)

dummy_obs2, dummy_info2 = (
    stage2_env.reset()
)

dummy_fdm = getattr(
    stage2_env,
    "fdm",
    None,
)

dummy_fdm_id = (
    id(dummy_fdm)
    if dummy_fdm is not None
    else None
)

print(
    "Disposable Stage-2 reset FDM id:",
    dummy_fdm_id,
)

print(
    "Active Stage-1 FDM id before attach:",
    stage1_fdm_id,
)

# Critical line:
stage2_env.fdm = active_fdm

# Start Stage-2 mission distance at zero AT THE HANDOFF POINT.
if hasattr(
    stage2_env,
    "forward_distance",
):
    stage2_env.forward_distance = 0.0

# Keep the original Stage-1 mission heading as the forward axis.
if (
    hasattr(
        stage2_env,
        "target_heading",
    )
    and
    np.isfinite(
        mission_heading
    )
):
    stage2_env.target_heading = (
        float(
            mission_heading
        )
    )

# Some Stage-2 variants expose an initial heading separately.
for attr in [
    "initial_heading",
    "initial_heading_rad",
]:
    if (
        hasattr(
            stage2_env,
            attr,
        )
        and
        np.isfinite(
            mission_heading
        )
    ):
        setattr(
            stage2_env,
            attr,
            float(
                mission_heading
            ),
        )

# Ensure mission counters begin at the handoff.
for attr in [
    "steps",
    "target_hold_steps",
    "hold_steps",
    "success_hold_steps",
]:
    if hasattr(
        stage2_env,
        attr,
    ):
        setattr(
            stage2_env,
            attr,
            0,
        )

attached_fdm = get_active_fdm(
    stage2_env
)

attached_fdm_id = id(
    attached_fdm
)

sim_time_after_attach = (
    sim_time_seconds(
        attached_fdm
    )
)

print(
    "\nACTIVE FDM SAME OBJECT:",
    attached_fdm_id
    ==
    stage1_fdm_id,
)

print(
    "Stage-2 attached FDM id:",
    attached_fdm_id,
)

print(
    f"Sim time before attach : {handoff_sim_time:.6f} s"
)

print(
    f"Sim time after attach  : {sim_time_after_attach:.6f} s"
)

sim_clock_continuous = True

if (
    np.isfinite(
        handoff_sim_time
    )
    and
    np.isfinite(
        sim_time_after_attach
    )
):
    sim_clock_continuous = (
        abs(
            sim_time_after_attach
            -
            handoff_sim_time
        )
        <
        1e-6
    )

print(
    "SIM CLOCK RESET:",
    not sim_clock_continuous,
)

if attached_fdm_id != stage1_fdm_id:
    raise RuntimeError(
        "FAIL: Stage 2 is not attached to the exact Stage-1 FDM object."
    )

if not sim_clock_continuous:
    raise RuntimeError(
        "FAIL: active JSBSim simulation clock changed during handoff."
    )


# =====================================================================
# BUILD THE FIRST 12-D STAGE-2 OBSERVATION FROM THE SAME FDM STATE
# =====================================================================

obs2 = np.asarray(
    stage2_env._get_obs(),
    dtype=np.float32,
)

print(
    "\nFirst Stage-2 observation shape:",
    obs2.shape,
)

print(
    "Expected by Stage-2 PPO          :",
    stage2_model.observation_space.shape,
)

if (
    obs2.shape
    !=
    stage2_model.observation_space.shape
):
    raise RuntimeError(
        "Stage-2 observation shape mismatch after continuous attachment."
    )

print(
    "First Stage-2 observation:",
    np.array2string(
        obs2,
        precision=5,
        floatmode="fixed",
    ),
)


# =====================================================================
# STAGE 2 — FORWARD FLIGHT ON SAME FDM
# =====================================================================

print_header(
    "PHASE C — STAGE 2 FORWARD FLIGHT: SAME FDM, NO RESET, NO TRAINING"
)

trace = []

stage2_start_sim_time = (
    sim_time_seconds(
        active_fdm
    )
)

min_altitude = handoff_altitude
max_altitude = handoff_altitude
max_abs_cross_track = 0.0
max_abs_heading_error = 0.0
max_abs_roll = 0.0
max_abs_pitch = 0.0

target_crossing = None
physical_failure = False
termination_reason = ""
stage2_success_flag = False

last_info2 = {}

next_stage2_print = 0.0

# Fallback control time if a sim-time property is unavailable.
fallback_stage2_dt = (
    stage1_dt
)

max_stage2_steps = int(
    math.ceil(
        MAX_STAGE2_TIME
        /
        fallback_stage2_dt
    )
)

for step2 in range(
    max_stage2_steps
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
    ).reshape(-1)

    (
        obs2,
        reward2,
        terminated2,
        truncated2,
        info2,
    ) = stage2_env.step(
        action2
    )

    obs2 = np.asarray(
        obs2,
        dtype=np.float32,
    )

    last_info2 = info2

    current_sim_time = (
        sim_time_seconds(
            active_fdm
        )
    )

    if (
        np.isfinite(
            current_sim_time
        )
        and
        np.isfinite(
            stage2_start_sim_time
        )
    ):
        t2 = (
            current_sim_time
            -
            stage2_start_sim_time
        )
    else:
        t2 = (
            (step2 + 1)
            *
            fallback_stage2_dt
        )

    altitude2 = info_float(
        info2,
        "altitude",
        default=fdm_float(
            active_fdm,
            "position/h-agl-ft",
        ),
    )

    vs2 = info_float(
        info2,
        "vertical_speed",
        default=fdm_float(
            active_fdm,
            "velocities/h-dot-fps",
        ),
    )

    fwd_speed2 = info_float(
        info2,
        "forward_velocity",
        default=fdm_float(
            active_fdm,
            "velocities/u-aero-fps",
        ),
    )

    lateral_speed2 = info_float(
        info2,
        "lateral_velocity",
        default=fdm_float(
            active_fdm,
            "velocities/v-aero-fps",
        ),
    )

    heading_error2 = info_float(
        info2,
        "heading_error",
        default=float("nan"),
    )

    roll2 = info_float(
        info2,
        "roll",
        default=fdm_float(
            active_fdm,
            "attitude/roll-rad",
        ),
    )

    pitch2 = info_float(
        info2,
        "pitch",
        default=fdm_float(
            active_fdm,
            "attitude/pitch-rad",
        ),
    )

    model_distance = info_float(
        info2,
        "forward_distance",
        default=as_float(
            getattr(
                stage2_env,
                "forward_distance",
                float("nan"),
            )
        ),
    )

    lat2 = latitude_deg(
        active_fdm
    )

    lon2 = longitude_deg(
        active_fdm
    )

    north_from_handoff, east_from_handoff = (
        local_ne_ft(
            lat2,
            lon2,
            handoff_lat,
            handoff_lon,
        )
    )

    ground_forward, cross_track = (
        project_to_mission_axes(
            north_from_handoff,
            east_from_handoff,
            mission_heading,
        )
    )

    min_altitude = min(
        min_altitude,
        altitude2,
    )

    max_altitude = max(
        max_altitude,
        altitude2,
    )

    if np.isfinite(
        cross_track
    ):
        max_abs_cross_track = max(
            max_abs_cross_track,
            abs(
                cross_track
            ),
        )

    if np.isfinite(
        heading_error2
    ):
        max_abs_heading_error = max(
            max_abs_heading_error,
            abs(
                heading_error2
            ),
        )

    if np.isfinite(
        roll2
    ):
        max_abs_roll = max(
            max_abs_roll,
            abs(
                roll2
            ),
        )

    if np.isfinite(
        pitch2
    ):
        max_abs_pitch = max(
            max_abs_pitch,
            abs(
                pitch2
            ),
        )

    row = {
        "phase": "stage2",
        "t_stage2_s": float(
            t2
        ),
        "sim_time_s": float(
            current_sim_time
        ),
        "altitude_ft": float(
            altitude2
        ),
        "vertical_speed_fps": float(
            vs2
        ),
        "forward_speed_fps": float(
            fwd_speed2
        ),
        "lateral_speed_fps": float(
            lateral_speed2
        ),
        "stage2_forward_distance_ft": float(
            model_distance
        ),
        "ground_forward_from_handoff_ft": float(
            ground_forward
        ),
        "cross_track_ft": float(
            cross_track
        ),
        "heading_error_rad": float(
            heading_error2
        ),
        "roll_rad": float(
            roll2
        ),
        "pitch_rad": float(
            pitch2
        ),
        "action0": float(
            action2[0]
        ),
        "action1": float(
            action2[1]
        ),
        "action2": float(
            action2[2]
        ),
        "action3": float(
            action2[3]
        ),
        "reward": float(
            reward2
        ),
        "terminated": bool(
            terminated2
        ),
        "truncated": bool(
            truncated2
        ),
        "success": bool(
            info2.get(
                "success",
                False,
            )
        ),
        "termination_reason": str(
            info2.get(
                "termination_reason",
                "",
            )
        ),
    }

    trace.append(
        row
    )

    if (
        target_crossing is None
        and
        np.isfinite(
            model_distance
        )
        and
        model_distance
        >=
        TARGET_FORWARD_DISTANCE
    ):
        target_crossing = (
            row.copy()
        )

    if t2 >= next_stage2_print:
        print(
            f"S2 | t={t2:6.2f}s | "
            f"DIST={model_distance:7.2f} | "
            f"GND_FWD={ground_forward:7.2f} | "
            f"XTRK={cross_track:+7.2f} | "
            f"ALT={altitude2:7.2f} | "
            f"VS={vs2:+6.2f} | "
            f"FWD={fwd_speed2:+6.2f} | "
            f"LAT={lateral_speed2:+6.2f} | "
            f"HEAD={math.degrees(heading_error2):+6.2f}deg | "
            f"R={math.degrees(roll2):+6.2f}deg | "
            f"P={math.degrees(pitch2):+6.2f}deg"
        )

        next_stage2_print += (
            PRINT_EVERY_STAGE2
        )

    stage2_success_flag = bool(
        info2.get(
            "success",
            False,
        )
    )

    if terminated2:
        termination_reason = str(
            info2.get(
                "termination_reason",
                "",
            )
        )

        # A termination is considered physical failure unless the
        # environment explicitly declares success or we have already
        # crossed the requested 300-ft mission distance.
        crossed = (
            target_crossing
            is not None
        )

        if not (
            stage2_success_flag
            or
            crossed
        ):
            physical_failure = True

        break

    if truncated2:
        termination_reason = "truncated"
        break

    # Once the real Stage-2 mission distance is crossed, we have the
    # handoff answer we need. Do not continue into a different task.
    if (
        target_crossing
        is not None
    ):
        break


# =====================================================================
# SAVE TRACE
# =====================================================================

if trace:
    fieldnames = list(
        trace[0].keys()
    )

    with TRACE_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            trace
        )


# =====================================================================
# FINAL METRICS
# =====================================================================

if trace:
    final_row = trace[-1]
else:
    raise RuntimeError(
        "Stage 2 produced no trajectory samples."
    )

final_model_distance = float(
    final_row[
        "stage2_forward_distance_ft"
    ]
)

final_ground_forward = float(
    final_row[
        "ground_forward_from_handoff_ft"
    ]
)

final_cross_track = float(
    final_row[
        "cross_track_ft"
    ]
)

final_altitude = float(
    final_row[
        "altitude_ft"
    ]
)

final_vs = float(
    final_row[
        "vertical_speed_fps"
    ]
)

altitude_drop = (
    handoff_altitude
    -
    min_altitude
)

altitude_rise = (
    max_altitude
    -
    handoff_altitude
)

handoff_ok = (
    HANDOFF_ALT_MIN
    <=
    handoff_altitude
    <=
    HANDOFF_ALT_MAX
    and
    abs(
        handoff_vs
    )
    <=
    HANDOFF_MAX_ABS_VS
    and
    handoff_hspeed
    <=
    HANDOFF_MAX_HSPEED
    and
    handoff_drift
    <=
    HANDOFF_MAX_DRIFT
)

reached_target = (
    target_crossing
    is not None
)

altitude_corridor_ok = (
    min_altitude
    >=
    ALTITUDE_CORRIDOR_MIN
    and
    max_altitude
    <=
    ALTITUDE_CORRIDOR_MAX
)

cross_track_ok = (
    max_abs_cross_track
    <=
    MAX_CROSS_TRACK
)

if target_crossing is not None:
    crossing_altitude = float(
        target_crossing[
            "altitude_ft"
        ]
    )

    crossing_vs = float(
        target_crossing[
            "vertical_speed_fps"
        ]
    )

    crossing_cross_track = float(
        target_crossing[
            "cross_track_ft"
        ]
    )

    crossing_ground_forward = float(
        target_crossing[
            "ground_forward_from_handoff_ft"
        ]
    )

    crossing_model_distance = float(
        target_crossing[
            "stage2_forward_distance_ft"
        ]
    )

    crossing_time = float(
        target_crossing[
            "t_stage2_s"
        ]
    )

    crossing_altitude_ok = (
        TARGET_CROSSING_ALT_MIN
        <=
        crossing_altitude
        <=
        TARGET_CROSSING_ALT_MAX
    )

    crossing_vs_ok = (
        abs(
            crossing_vs
        )
        <=
        MAX_TARGET_CROSSING_ABS_VS
    )
else:
    crossing_altitude = float("nan")
    crossing_vs = float("nan")
    crossing_cross_track = float("nan")
    crossing_ground_forward = float("nan")
    crossing_model_distance = float("nan")
    crossing_time = float("nan")

    crossing_altitude_ok = False
    crossing_vs_ok = False


continuous_fdm_ok = (
    attached_fdm_id
    ==
    stage1_fdm_id
    and
    sim_clock_continuous
)

overall_pass = (
    continuous_fdm_ok
    and
    handoff_ok
    and
    reached_target
    and
    not physical_failure
    and
    altitude_corridor_ok
    and
    cross_track_ok
    and
    crossing_altitude_ok
    and
    crossing_vs_ok
)


# =====================================================================
# REPORT
# =====================================================================

print_header(
    "STAGE 1 -> STAGE 2 CONTINUOUS HANDOFF RESULT"
)

print(
    f"ACTIVE FDM CONTINUOUS          : {continuous_fdm_ok}"
)

print(
    f"STAGE-1 HANDOFF QUALITY        : {handoff_ok}"
)

print(
    f"REACHED 300 FT FORWARD         : {reached_target}"
)

print(
    f"PHYSICAL FAILURE               : {physical_failure}"
)

print(
    f"TERMINATION REASON             : {termination_reason or 'none'}"
)

print()

print(
    f"HANDOFF ALTITUDE               : {handoff_altitude:.3f} ft"
)

print(
    f"HANDOFF VS                     : {handoff_vs:+.3f} ft/s"
)

print(
    f"HANDOFF HORIZONTAL SPEED       : {handoff_hspeed:.3f} ft/s"
)

print(
    f"HANDOFF TAKEOFF DRIFT          : {handoff_drift:.3f} ft"
)

print()

print(
    f"FORWARD-FLIGHT MIN ALT         : {min_altitude:.3f} ft"
)

print(
    f"FORWARD-FLIGHT MAX ALT         : {max_altitude:.3f} ft"
)

print(
    f"MAX DROP FROM HANDOFF          : {altitude_drop:.3f} ft"
)

print(
    f"MAX RISE FROM HANDOFF          : {altitude_rise:.3f} ft"
)

print(
    f"ALTITUDE 290-310 THROUGHOUT    : {altitude_corridor_ok}"
)

print()

print(
    f"MAX |CROSS TRACK|              : {max_abs_cross_track:.3f} ft"
)

print(
    f"CROSS TRACK <= 5 FT            : {cross_track_ok}"
)

print(
    f"MAX |HEADING ERROR|            : {math.degrees(max_abs_heading_error):.3f} deg"
)

print(
    f"MAX |ROLL|                     : {math.degrees(max_abs_roll):.3f} deg"
)

print(
    f"MAX |PITCH|                    : {math.degrees(max_abs_pitch):.3f} deg"
)

print()

if target_crossing is not None:
    print(
        f"300-FT CROSSING TIME           : {crossing_time:.3f} s"
    )

    print(
        f"STAGE2 DIST @ CROSSING         : {crossing_model_distance:.3f} ft"
    )

    print(
        f"GROUND FWD @ CROSSING          : {crossing_ground_forward:.3f} ft"
    )

    print(
        f"ALTITUDE @ CROSSING            : {crossing_altitude:.3f} ft"
    )

    print(
        f"VS @ CROSSING                  : {crossing_vs:+.3f} ft/s"
    )

    print(
        f"CROSS TRACK @ CROSSING         : {crossing_cross_track:+.3f} ft"
    )

    print(
        f"CROSSING ALT 295-305           : {crossing_altitude_ok}"
    )

    print(
        f"CROSSING |VS| <= 2             : {crossing_vs_ok}"
    )
else:
    print(
        "300-FT CROSSING                : NOT REACHED"
    )

print()

print(
    f"FINAL STAGE2 DISTANCE          : {final_model_distance:.3f} ft"
)

print(
    f"FINAL GROUND FORWARD           : {final_ground_forward:.3f} ft"
)

print(
    f"FINAL CROSS TRACK              : {final_cross_track:+.3f} ft"
)

print(
    f"FINAL ALTITUDE                 : {final_altitude:.3f} ft"
)

print(
    f"FINAL VS                       : {final_vs:+.3f} ft/s"
)

print()

print(
    f"CONTINUOUS HANDOFF PASS        : {overall_pass}"
)


summary = {
    "models": {
        "stage1": STAGE1_MODEL_PATH,
        "stage2": STAGE2_MODEL_PATH,
    },
    "continuity": {
        "stage1_fdm_id": int(
            stage1_fdm_id
        ),
        "stage2_attached_fdm_id": int(
            attached_fdm_id
        ),
        "same_fdm_object": bool(
            attached_fdm_id
            ==
            stage1_fdm_id
        ),
        "sim_clock_continuous": bool(
            sim_clock_continuous
        ),
        "handoff_sim_time_s": float(
            handoff_sim_time
        ),
    },
    "handoff": {
        "stage1_time_s": float(
            handoff_time_stage1
        ),
        "altitude_ft": float(
            handoff_altitude
        ),
        "vertical_speed_fps": float(
            handoff_vs
        ),
        "horizontal_speed_fps": float(
            handoff_hspeed
        ),
        "takeoff_drift_ft": float(
            handoff_drift
        ),
        "north_ft": float(
            handoff_north
        ),
        "east_ft": float(
            handoff_east
        ),
        "quality_pass": bool(
            handoff_ok
        ),
    },
    "stage2": {
        "reached_300_ft": bool(
            reached_target
        ),
        "physical_failure": bool(
            physical_failure
        ),
        "termination_reason": str(
            termination_reason
        ),
        "min_altitude_ft": float(
            min_altitude
        ),
        "max_altitude_ft": float(
            max_altitude
        ),
        "max_drop_from_handoff_ft": float(
            altitude_drop
        ),
        "max_rise_from_handoff_ft": float(
            altitude_rise
        ),
        "altitude_290_310_pass": bool(
            altitude_corridor_ok
        ),
        "max_abs_cross_track_ft": float(
            max_abs_cross_track
        ),
        "cross_track_5ft_pass": bool(
            cross_track_ok
        ),
        "max_abs_heading_error_deg": float(
            math.degrees(
                max_abs_heading_error
            )
        ),
        "max_abs_roll_deg": float(
            math.degrees(
                max_abs_roll
            )
        ),
        "max_abs_pitch_deg": float(
            math.degrees(
                max_abs_pitch
            )
        ),
        "final_distance_ft": float(
            final_model_distance
        ),
        "final_ground_forward_ft": float(
            final_ground_forward
        ),
        "final_cross_track_ft": float(
            final_cross_track
        ),
        "final_altitude_ft": float(
            final_altitude
        ),
        "final_vs_fps": float(
            final_vs
        ),
        "environment_success_flag": bool(
            stage2_success_flag
        ),
    },
    "crossing": {
        "time_s": float(
            crossing_time
        ),
        "stage2_distance_ft": float(
            crossing_model_distance
        ),
        "ground_forward_ft": float(
            crossing_ground_forward
        ),
        "altitude_ft": float(
            crossing_altitude
        ),
        "vertical_speed_fps": float(
            crossing_vs
        ),
        "cross_track_ft": float(
            crossing_cross_track
        ),
        "altitude_295_305_pass": bool(
            crossing_altitude_ok
        ),
        "abs_vs_le_2_pass": bool(
            crossing_vs_ok
        ),
    },
    "overall_pass": bool(
        overall_pass
    ),
}

with SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(
    "\nSaved trace  :",
    TRACE_CSV,
)

print(
    "Saved summary:",
    SUMMARY_JSON,
)

print_header(
    "WHAT THIS TEST MEANS"
)

if overall_pass:
    print(
        "PASS: Stage 1 and Stage 2 work as a true continuous mission "
        "on the same JSBSim aircraft state."
    )

    print(
        "Next step: lock the Stage1->Stage2 handoff and move to the "
        "stop/transition + vertical landing stage."
    )
else:
    print(
        "NOT LOCKED YET: the script has isolated exactly what fails "
        "during the real Stage1->Stage2 transition."
    )

    print(
        "Do NOT retrain blindly. Use the printed altitude-drop, "
        "cross-track and crossing metrics to decide the next fix."
    )
