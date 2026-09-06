from __future__ import annotations

"""
AH-1S STAGE 2 HYBRID RECOVERY / FINAL BUILD
===========================================

Purpose
-------
Rebuild a strong Stage-2 learned policy without the lost Stage-2 checkpoint.

Pipeline:
1) Reproduce the locked Stage-1 -> Stage-2 continuous handoff.
2) Reconstruct the already-validated Stage-2 reference flight from the
   previously recorded successful teacher-off action trace.
3) Add only small altitude/vertical-speed feedback around that reference
   and automatically choose the best teacher that still passes the
   presentation geometry.
4) Collect teacher state-action data around the true handoff.
5) Behavior-clone the complete Stage-2 teacher into a fresh 4-action PPO actor.
6) Validate the neural policy with teacher OFF.
7) Fine-tune the cloned policy with PPO reinforcement learning on the
   mapped JSBSim Stage-2 environment.
8) Validate every RL candidate on the TRUE continuous Stage1 -> Stage2
   JSBSim flight, teacher OFF.
9) Save only a passing final neural policy.

Final runtime:
    Stage-1 teacher            OFF
    Stage-2 teacher            OFF
    Classical altitude helper OFF
    Classical lateral helper  OFF
    PPO learned policy         ON
    Mapped Stage-2 env         ON only for repaired actuator wiring

This is intentionally a hybrid training pipeline:
    recorded/teacher guidance -> distillation -> PPO RL fine-tuning
"""

import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill
from helicopter_env_stage2_refine_mapped import HelicopterEnvStage2RefineMapped


# =====================================================================
# PATHS
# =====================================================================

STAGE1_MODEL_PATH = Path(
    "models_stage1_early_distilled/AH1S_STAGE1_EARLY_DISTILLED.zip"
)

OUT_DIR = Path("models_stage2_hybrid_final")
RESULT_DIR = Path("results_stage2_hybrid_final")

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BC_MODEL_PATH = OUT_DIR / "AH1S_STAGE2_BC_WARMSTART"
FINAL_MODEL_PATH = OUT_DIR / "AH1S_STAGE2_HYBRID_FINAL"

if not STAGE1_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Locked Stage-1 model missing: {STAGE1_MODEL_PATH}"
    )


# =====================================================================
# LOCKED MISSION CONSTANTS
# =====================================================================

SEED = 42

TARGET_ALT = 300.0
TARGET_DISTANCE = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
HANDOFF_STABLE_TIME = 5.0

PRESENT_MIN_ALT = 290.0
PRESENT_MAX_ALT = 310.0
PRESENT_MAX_CROSS = 5.0

CROSS_ALT_MIN = 295.0
CROSS_ALT_MAX = 305.0
CROSS_MAX_ABS_VS = 2.0

MARGIN_MAX_CROSS = 4.0

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040

TEACHER_AILERON_ACTION = -0.230
TEACHER_RUDDER_ACTION = 0.000

EARTH_RADIUS_FT = 20_902_231.0


# =====================================================================
# SUCCESSFUL STAGE-2 REFERENCE TRACE
#
# These points are from the previously validated teacher-OFF Stage-2
# final flight.  We use them only as a training reference to recover
# the lost Stage-2 checkpoint.  Final flight never reads this table.
# =====================================================================

REF_DISTANCE = np.array(
    [
        0.00,
        0.89,
        5.62,
        15.69,
        31.62,
        51.64,
        75.10,
        101.74,
        129.09,
        157.24,
        186.63,
        215.27,
        243.92,
        273.47,
        300.46,
    ],
    dtype=np.float64,
)

REF_COLLECTIVE_ACTION = np.array(
    [
        -0.3487,
        -0.3621,
        -0.3904,
        -0.4252,
        -0.4562,
        -0.4774,
        -0.4936,
        -0.5119,
        -0.5369,
        -0.5715,
        -0.6163,
        -0.6657,
        -0.7170,
        -0.7678,
        -0.809686,
    ],
    dtype=np.float64,
)

REF_ELEVATOR_ACTION = np.array(
    [
        -0.0211,
        -0.0139,
        -0.0069,
        -0.0016,
        -0.0022,
        -0.0097,
        -0.0210,
        -0.0320,
        -0.0385,
        -0.0389,
        -0.0331,
        -0.0229,
        -0.0103,
        +0.0030,
        +0.013725,
    ],
    dtype=np.float64,
)


# =====================================================================
# TRAINING SETTINGS
# =====================================================================

# Small teacher feedback search around the recorded successful neural
# trajectory.  The reference schedule is already strong; these gains
# only absorb small reconstruction deviations.
TEACHER_GAIN_CANDIDATES = [
    # kp_alt, kd_vs, max collective correction
    (0.000, 0.000, 0.000),
    (0.004, 0.020, 0.050),
    (0.006, 0.030, 0.060),
    (0.008, 0.040, 0.070),
    (0.010, 0.050, 0.080),
    (0.012, 0.060, 0.090),
]

DATA_ROLLOUTS = 7
BC_EPOCHS = 320
BC_BATCH_SIZE = 256
BC_LR = 8e-4

# Real PPO RL updates after distillation.
# Each candidate starts from the exact same BC warm-start.
RL_CANDIDATES = [
    # learning_rate, timesteps
    (1e-6, 1024),
    (2e-6, 2048),
    (5e-6, 2048),
    (1e-5, 2048),
]

RL_EXPLORATION_STD = 0.05


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass


# =====================================================================
# HELPERS
# =====================================================================

def rule(text: str):
    print()
    print("=" * 144)
    print(text)
    print("=" * 144)


def fdm_float(fdm, key, default=float("nan")):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def info_float(info, key, default=float("nan")):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def get_fdm(env):
    direct = getattr(env, "fdm", None)
    if direct is not None:
        return direct

    base = getattr(env, "base_env", None)
    if base is not None:
        nested = getattr(base, "fdm", None)
        if nested is not None:
            return nested

    raise RuntimeError("Active JSBSim FDM not found.")


def latitude_deg(fdm):
    for key in [
        "position/lat-gc-deg",
        "position/lat-geod-deg",
    ]:
        value = fdm_float(fdm, key)
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
        value = fdm_float(fdm, key)
        if np.isfinite(value):
            return value
    return float("nan")


def wrap_angle(value):
    return math.atan2(
        math.sin(value),
        math.cos(value),
    )


def local_ne_ft(lat, lon, lat0, lon0):
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)

    north = EARTH_RADIUS_FT * dlat
    east = (
        EARTH_RADIUS_FT
        * math.cos(math.radians(lat0))
        * dlon
    )

    return float(north), float(east)


def mission_axes(north, east, heading):
    c = math.cos(heading)
    s = math.sin(heading)

    forward = north * c + east * s
    cross = -north * s + east * c

    return float(forward), float(cross)


# =====================================================================
# STAGE 1 TRUE HANDOFF
# =====================================================================

stage1_model = PPO.load(
    str(STAGE1_MODEL_PATH)
)


def build_stage1_handoff():
    env1 = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env1.reset()

    fdm = get_fdm(env1)
    mission_heading = heading_rad(fdm)

    dt = float(
        getattr(
            env1,
            "dt",
            0.075,
        )
    )

    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.075

    stable_time = 0.0
    handoff = None

    for _ in range(
        int(STAGE1_MAX_TIME / dt)
    ):
        action, _ = stage1_model.predict(
            obs,
            deterministic=True,
        )

        (
            obs,
            _,
            terminated,
            truncated,
            info,
        ) = env1.step(action)

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
            np.hypot(vn, ve)
        )

        drift = info_float(
            info,
            "drift",
            999.0,
        )

        stable = bool(
            295.0 <= altitude <= 305.0
            and abs(vertical_speed) <= 0.50
            and horizontal_speed <= 1.0
            and drift <= 3.0
        )

        stable_time = (
            stable_time + dt
            if stable
            else 0.0
        )

        if stable_time >= HANDOFF_STABLE_TIME:
            handoff = {
                "altitude": altitude,
                "vertical_speed": vertical_speed,
                "horizontal_speed": horizontal_speed,
                "drift": drift,
            }
            break

        if (
            terminated
            and not bool(
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


# =====================================================================
# ATTACH MAPPED STAGE 2 TO SAME ACTIVE FDM
# =====================================================================

def attach_stage2(
    active_fdm,
    mission_heading,
):
    env2 = HelicopterEnvStage2RefineMapped(
        aileron_scale=AILERON_SCALE,
        rudder_scale=RUDDER_SCALE,
    )

    # Disposable reset initializes Python bookkeeping only.
    env2.reset()

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
        if hasattr(env2, attr):
            setattr(
                env2,
                attr,
                0,
            )

    if hasattr(
        env2,
        "previous_action",
    ):
        try:
            env2.previous_action = np.zeros(
                4,
                dtype=np.float32,
            )
        except Exception:
            pass

    obs = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    return env2, obs


# =====================================================================
# RECONSTRUCTED TEACHER
# =====================================================================

def reference_schedule(
    forward_distance,
):
    d = float(
        np.clip(
            forward_distance,
            REF_DISTANCE[0],
            REF_DISTANCE[-1],
        )
    )

    a0 = float(
        np.interp(
            d,
            REF_DISTANCE,
            REF_COLLECTIVE_ACTION,
        )
    )

    a1 = float(
        np.interp(
            d,
            REF_DISTANCE,
            REF_ELEVATOR_ACTION,
        )
    )

    return a0, a1


def teacher_action(
    env2,
    fdm,
    kp_alt,
    kd_vs,
    max_collective_correction,
):
    d = float(
        getattr(
            env2,
            "forward_distance",
            0.0,
        )
    )

    base_a0, a1 = reference_schedule(d)

    altitude = fdm_float(
        fdm,
        "position/h-agl-ft",
        TARGET_ALT,
    )

    vertical_speed = fdm_float(
        fdm,
        "velocities/h-dot-fps",
        0.0,
    )

    correction = (
        float(kp_alt)
        * (TARGET_ALT - altitude)
        -
        float(kd_vs)
        * vertical_speed
    )

    if max_collective_correction > 0.0:
        correction = float(
            np.clip(
                correction,
                -max_collective_correction,
                +max_collective_correction,
            )
        )
    else:
        correction = 0.0

    action = np.array(
        [
            np.clip(
                base_a0 + correction,
                -1.0,
                +1.0,
            ),
            np.clip(
                a1,
                -1.0,
                +1.0,
            ),
            TEACHER_AILERON_ACTION,
            TEACHER_RUDDER_ACTION,
        ],
        dtype=np.float32,
    )

    return action


# =====================================================================
# CONTINUOUS FLIGHT EVALUATION
# =====================================================================

def evaluate_controller(
    policy_model=None,
    teacher_cfg=None,
    detailed=False,
):
    (
        env1,
        fdm,
        mission_heading,
        handoff,
    ) = build_stage1_handoff()

    active_fdm_id = id(fdm)

    handoff_lat = latitude_deg(fdm)
    handoff_lon = longitude_deg(fdm)

    env2, obs = attach_stage2(
        fdm,
        mission_heading,
    )

    if id(get_fdm(env2)) != active_fdm_id:
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

    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.075

    min_alt = float(
        handoff["altitude"]
    )

    max_alt = float(
        handoff["altitude"]
    )

    max_cross = 0.0
    max_abs_lat = 0.0
    max_abs_heading = 0.0
    max_abs_roll = 0.0

    crossing = None
    failure = False
    termination_reason = ""

    next_print = 0.0

    action_min = np.full(
        4,
        +999.0,
        dtype=np.float64,
    )

    action_max = np.full(
        4,
        -999.0,
        dtype=np.float64,
    )

    trace = []

    for step in range(
        int(STAGE2_MAX_TIME / dt)
    ):
        if teacher_cfg is not None:
            action = teacher_action(
                env2,
                fdm,
                *teacher_cfg,
            )

        elif policy_model is not None:
            action, _ = (
                policy_model.predict(
                    obs,
                    deterministic=True,
                )
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)

        else:
            raise ValueError(
                "Need policy_model or teacher_cfg."
            )

        action = np.clip(
            action,
            -1.0,
            +1.0,
        ).astype(np.float32)

        action_min = np.minimum(
            action_min,
            action,
        )

        action_max = np.maximum(
            action_max,
            action,
        )

        (
            obs,
            _,
            terminated,
            truncated,
            info,
        ) = env2.step(action)

        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        t = (step + 1) * dt

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

        lat = latitude_deg(fdm)
        lon = longitude_deg(fdm)

        north, east = local_ne_ft(
            lat,
            lon,
            handoff_lat,
            handoff_lon,
        )

        ground_forward, cross = mission_axes(
            north,
            east,
            mission_heading,
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
            abs(lateral_velocity),
        )

        max_abs_heading = max(
            max_abs_heading,
            abs(heading_error),
        )

        max_abs_roll = max(
            max_abs_roll,
            abs(roll),
        )

        trace.append(
            {
                "time": t,
                "distance": distance,
                "ground_forward": ground_forward,
                "cross": cross,
                "altitude": altitude,
                "vertical_speed": vertical_speed,
                "lateral_velocity": lateral_velocity,
                "heading_error_deg": math.degrees(
                    heading_error
                ),
                "roll_deg": math.degrees(roll),
                "a0": float(action[0]),
                "a1": float(action[1]),
                "a2": float(action[2]),
                "a3": float(action[3]),
            }
        )

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"D={distance:7.2f} | "
                f"GND={ground_forward:7.2f} | "
                f"X={cross:+7.3f} | "
                f"ALT={altitude:7.3f} | "
                f"VS={vertical_speed:+6.3f} | "
                f"LAT={lateral_velocity:+6.3f} | "
                f"HEAD={math.degrees(heading_error):+6.3f}deg | "
                f"A={np.array2string(action, precision=4, floatmode='fixed')}"
            )

            next_print += 2.5

        if distance >= TARGET_DISTANCE:
            crossing = {
                "altitude": float(altitude),
                "vertical_speed": float(
                    vertical_speed
                ),
                "cross": float(cross),
                "ground_forward": float(
                    ground_forward
                ),
                "heading_error_deg": float(
                    math.degrees(
                        heading_error
                    )
                ),
                "lateral_velocity": float(
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
            termination_reason = "truncated"
            break

    env2.fdm = None
    env1.close()

    reached = bool(
        crossing is not None
    )

    if reached:
        crossing_alt = crossing[
            "altitude"
        ]
        crossing_vs = crossing[
            "vertical_speed"
        ]
        crossing_cross = crossing[
            "cross"
        ]
        crossing_ground = crossing[
            "ground_forward"
        ]
        crossing_heading = crossing[
            "heading_error_deg"
        ]
        crossing_lat = crossing[
            "lateral_velocity"
        ]
    else:
        crossing_alt = float("nan")
        crossing_vs = float("nan")
        crossing_cross = float("nan")
        crossing_ground = float("nan")
        crossing_heading = float("nan")
        crossing_lat = float("nan")

    presentation_pass = bool(
        reached
        and not failure
        and min_alt >= PRESENT_MIN_ALT
        and max_alt <= PRESENT_MAX_ALT
        and max_cross <= PRESENT_MAX_CROSS
        and CROSS_ALT_MIN
        <= crossing_alt
        <= CROSS_ALT_MAX
        and abs(
            crossing_vs
        )
        <= CROSS_MAX_ABS_VS
    )

    margin_pass = bool(
        presentation_pass
        and max_cross <= MARGIN_MAX_CROSS
    )

    result = {
        "reached_300": reached,
        "failure": failure,
        "termination_reason":
            termination_reason,
        "handoff_alt": float(
            handoff["altitude"]
        ),
        "handoff_vs": float(
            handoff["vertical_speed"]
        ),
        "handoff_drift": float(
            handoff["drift"]
        ),
        "min_alt": float(min_alt),
        "max_alt": float(max_alt),
        "max_drop": float(
            handoff["altitude"]
            -
            min_alt
        ),
        "max_cross": float(max_cross),
        "max_abs_lat": float(
            max_abs_lat
        ),
        "max_abs_heading_deg": float(
            math.degrees(
                max_abs_heading
            )
        ),
        "max_abs_roll_deg": float(
            math.degrees(
                max_abs_roll
            )
        ),
        "crossing_alt": float(
            crossing_alt
        ),
        "crossing_vs": float(
            crossing_vs
        ),
        "crossing_cross": float(
            crossing_cross
        ),
        "crossing_ground": float(
            crossing_ground
        ),
        "crossing_heading_deg": float(
            crossing_heading
        ),
        "crossing_lat": float(
            crossing_lat
        ),
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "presentation_pass":
            presentation_pass,
        "margin_pass": margin_pass,
        "trace": trace,
    }

    return result


def print_result(
    label,
    r,
):
    print(
        f"{label:24s} | "
        f"PASS={str(r['presentation_pass']):5s} | "
        f"MARGIN={str(r['margin_pass']):5s} | "
        f"MINALT={r['min_alt']:7.3f} | "
        f"MAXALT={r['max_alt']:7.3f} | "
        f"MAX_X={r['max_cross']:6.3f} | "
        f"CROSS_X={r['crossing_cross']:+7.3f} | "
        f"ALT@300={r['crossing_alt']:7.3f} | "
        f"VS@300={r['crossing_vs']:+6.3f}"
    )


def selection_key(r):
    return (
        0
        if r["presentation_pass"]
        else 1,
        0
        if r["margin_pass"]
        else 1,
        r["max_cross"],
        abs(
            r["crossing_cross"]
        )
        if np.isfinite(
            r["crossing_cross"]
        )
        else 999.0,
        abs(
            r["crossing_alt"]
            -
            TARGET_ALT
        )
        if np.isfinite(
            r["crossing_alt"]
        )
        else 999.0,
        abs(
            r["crossing_vs"]
        )
        if np.isfinite(
            r["crossing_vs"]
        )
        else 999.0,
    )


# =====================================================================
# A — RECONSTRUCT / VERIFY THE TEACHER
# =====================================================================

rule(
    "A — RECONSTRUCT STAGE-2 REFERENCE TEACHER"
)

teacher_candidates = []

for cfg in TEACHER_GAIN_CANDIDATES:
    kp_alt, kd_vs, max_corr = cfg

    r = evaluate_controller(
        teacher_cfg=cfg,
        detailed=False,
    )

    r["teacher_kp_alt"] = kp_alt
    r["teacher_kd_vs"] = kd_vs
    r["teacher_max_corr"] = max_corr

    teacher_candidates.append(
        r
    )

    print_result(
        (
            f"KP={kp_alt:.3f} "
            f"KD={kd_vs:.3f} "
            f"C={max_corr:.3f}"
        ),
        r,
    )


teacher_candidates.sort(
    key=selection_key
)

best_teacher_result = (
    teacher_candidates[0]
)

BEST_TEACHER_CFG = (
    best_teacher_result[
        "teacher_kp_alt"
    ],
    best_teacher_result[
        "teacher_kd_vs"
    ],
    best_teacher_result[
        "teacher_max_corr"
    ],
)

rule(
    "BEST RECONSTRUCTED TEACHER"
)

print(
    "Teacher cfg:",
    BEST_TEACHER_CFG
)

print_result(
    "BEST TEACHER",
    best_teacher_result,
)

if not best_teacher_result[
    "presentation_pass"
]:
    with open(
        RESULT_DIR
        /
        "teacher_recovery_failed.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                k: v
                for k, v
                in best_teacher_result.items()
                if k != "trace"
            },
            f,
            indent=2,
        )

    raise RuntimeError(
        "Reconstructed Stage-2 teacher did not pass. "
        "STOP here and inspect printed geometry before training."
    )


# =====================================================================
# B — COLLECT LOCAL TEACHER DATA
# =====================================================================

rule(
    "B — COLLECT TEACHER DATA AROUND TRUE STAGE1->STAGE2 HANDOFF"
)

dataset_obs = []
dataset_actions = []
dataset_weights = []

noise_scales = [
    (0.000, 0.000, 0.000, 0.000),
    (0.010, 0.004, 0.004, 0.002),
    (0.015, 0.006, 0.006, 0.003),
    (0.020, 0.008, 0.008, 0.004),
    (0.012, 0.005, 0.005, 0.002),
    (0.018, 0.007, 0.007, 0.003),
    (0.008, 0.003, 0.003, 0.001),
]

rng = np.random.default_rng(
    SEED
)

for episode in range(
    DATA_ROLLOUTS
):
    (
        env1,
        fdm,
        mission_heading,
        _,
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

    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.075

    stored = 0

    noise_scale = np.array(
        noise_scales[
            episode
            %
            len(
                noise_scales
            )
        ],
        dtype=np.float32,
    )

    for _ in range(
        int(STAGE2_MAX_TIME / dt)
    ):
        target = teacher_action(
            env2,
            fdm,
            *BEST_TEACHER_CFG,
        )

        d = float(
            getattr(
                env2,
                "forward_distance",
                0.0,
            )
        )

        if d < 80.0:
            w = 5.0
        elif d < 180.0:
            w = 3.0
        else:
            w = 2.0

        dataset_obs.append(
            np.asarray(
                obs,
                dtype=np.float32,
            ).copy()
        )

        dataset_actions.append(
            target.copy()
        )

        dataset_weights.append(
            w
        )

        if np.any(
            noise_scale > 0.0
        ):
            noise = rng.normal(
                0.0,
                noise_scale,
            ).astype(
                np.float32
            )

            executed = np.clip(
                target + noise,
                -1.0,
                +1.0,
            ).astype(
                np.float32
            )
        else:
            executed = target

        (
            obs,
            _,
            terminated,
            truncated,
            info,
        ) = env2.step(
            executed
        )

        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        stored += 1

        distance = info_float(
            info,
            "forward_distance",
            getattr(
                env2,
                "forward_distance",
                0.0,
            ),
        )

        if distance >= TARGET_DISTANCE:
            break

        if terminated or truncated:
            break

    env2.fdm = None
    env1.close()

    print(
        f"Episode {episode + 1:2d} | "
        f"stored={stored:4d} | "
        f"noise={noise_scale.tolist()}"
    )


dataset_obs = np.asarray(
    dataset_obs,
    dtype=np.float32,
)

dataset_actions = np.asarray(
    dataset_actions,
    dtype=np.float32,
)

dataset_weights = np.asarray(
    dataset_weights,
    dtype=np.float32,
)

print()
print(
    "Dataset observations:",
    dataset_obs.shape
)
print(
    "Dataset actions     :",
    dataset_actions.shape
)


# =====================================================================
# C — FRESH PPO + BEHAVIOR CLONING / DISTILLATION
# =====================================================================

rule(
    "C — DISTILL COMPLETE TEACHER INTO FRESH PPO ACTOR"
)

bc_env = Monitor(
    HelicopterEnvStage2RefineMapped(
        aileron_scale=AILERON_SCALE,
        rudder_scale=RUDDER_SCALE,
    )
)

policy_kwargs = dict(
    activation_fn=nn.Tanh,
    net_arch=dict(
        pi=[128, 128],
        vf=[128, 128],
    ),
)

model = PPO(
    "MlpPolicy",
    bc_env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=5,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=0,
    seed=SEED,
    device="auto",
)

device = model.device

obs_tensor = torch.as_tensor(
    dataset_obs,
    dtype=torch.float32,
    device=device,
)

target_tensor = torch.as_tensor(
    dataset_actions,
    dtype=torch.float32,
    device=device,
)

sample_weight_tensor = torch.as_tensor(
    dataset_weights,
    dtype=torch.float32,
    device=device,
)

# Elevator is numerically small but dynamically important.
action_loss_weight = torch.as_tensor(
    [1.0, 8.0, 2.0, 1.0],
    dtype=torch.float32,
    device=device,
).reshape(
    1,
    4,
)

actor_params = []

for name, param in (
    model.policy.named_parameters()
):
    # Actor branch + action head.
    # Do not BC-train value branch or log_std.
    if (
        "mlp_extractor.policy_net"
        in name
        or
        "action_net"
        in name
    ):
        actor_params.append(
            param
        )

if not actor_params:
    raise RuntimeError(
        "Could not locate PPO actor parameters."
    )

bc_optimizer = torch.optim.Adam(
    actor_params,
    lr=BC_LR,
)

indices = np.arange(
    len(
        dataset_obs
    )
)

best_bc_loss = float("inf")

for epoch in range(
    1,
    BC_EPOCHS + 1,
):
    rng.shuffle(indices)

    epoch_loss = 0.0
    batches = 0

    for start in range(
        0,
        len(indices),
        BC_BATCH_SIZE,
    ):
        idx_np = indices[
            start:
            start + BC_BATCH_SIZE
        ]

        idx = torch.as_tensor(
            idx_np,
            dtype=torch.long,
            device=device,
        )

        batch_obs = obs_tensor[
            idx
        ]

        batch_target = target_tensor[
            idx
        ]

        batch_weight = sample_weight_tensor[
            idx
        ].reshape(
            -1,
            1,
        )

        features = (
            model.policy
            .extract_features(
                batch_obs
            )
        )

        if isinstance(
            features,
            tuple,
        ):
            actor_features = features[
                0
            ]
        else:
            actor_features = features

        latent_pi = (
            model.policy
            .mlp_extractor
            .forward_actor(
                actor_features
            )
        )

        pred = (
            model.policy
            .action_net(
                latent_pi
            )
        )

        squared = (
            pred
            -
            batch_target
        ) ** 2

        loss = (
            squared
            *
            action_loss_weight
            *
            batch_weight
        ).mean()

        bc_optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            actor_params,
            1.0,
        )

        bc_optimizer.step()

        epoch_loss += float(
            loss.detach().cpu()
        )
        batches += 1

    epoch_loss /= max(
        1,
        batches,
    )

    best_bc_loss = min(
        best_bc_loss,
        epoch_loss,
    )

    if (
        epoch == 1
        or epoch % 40 == 0
        or epoch == BC_EPOCHS
    ):
        print(
            f"BC epoch {epoch:4d}/{BC_EPOCHS} | "
            f"loss={epoch_loss:.8f}"
        )


# Encode the locked lateral/yaw teacher exactly into the PPO action head.
with torch.no_grad():
    model.policy.action_net.weight[
        2
    ].zero_()

    model.policy.action_net.bias[
        2
    ].fill_(
        TEACHER_AILERON_ACTION
    )

    model.policy.action_net.weight[
        3
    ].zero_()

    model.policy.action_net.bias[
        3
    ].fill_(
        TEACHER_RUDDER_ACTION
    )

    # Low exploration for later safe RL refinement.
    if hasattr(
        model.policy,
        "log_std",
    ):
        model.policy.log_std.fill_(
            math.log(
                RL_EXPLORATION_STD
            )
        )


model.save(
    str(
        BC_MODEL_PATH
    )
)

bc_env.close()

print()
print(
    "Saved BC warm-start:",
    str(
        BC_MODEL_PATH
    )
    +
    ".zip"
)


# =====================================================================
# D — TEACHER-OFF BC VALIDATION
# =====================================================================

rule(
    "D — TEACHER-OFF BC VALIDATION"
)

bc_result = evaluate_controller(
    policy_model=model,
    detailed=True,
)

print_result(
    "BC PPO",
    bc_result,
)

with open(
    RESULT_DIR
    /
    "bc_validation.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            k: v
            for k, v
            in bc_result.items()
            if k != "trace"
        },
        f,
        indent=2,
    )

if not bc_result[
    "presentation_pass"
]:
    rule(
        "BC DISTILLATION DID NOT PASS"
    )

    print(
        "The reconstructed teacher is valid, "
        "but the fresh PPO actor did not imitate it accurately enough."
    )

    print(
        "STOP before PPO fine-tuning and inspect the printed geometry."
    )

    raise RuntimeError(
        "BC teacher-off validation failed."
    )


# =====================================================================
# E — PPO RL FINE-TUNING
#
# Every candidate starts from identical passing BC warm-start.
# We keep teacher OFF in the training environment.
# Rows 2/3 stay fixed at the distilled lateral/yaw constants.
# =====================================================================

rule(
    "E — PPO REINFORCEMENT-LEARNING FINE-TUNING"
)

rl_results = []

for idx, (
    lr,
    timesteps,
) in enumerate(
    RL_CANDIDATES,
    start=1,
):
    print()
    print(
        f"RL candidate {idx}/{len(RL_CANDIDATES)} | "
        f"lr={lr:.1e} | steps={timesteps}"
    )

    train_env = Monitor(
        HelicopterEnvStage2RefineMapped(
            aileron_scale=AILERON_SCALE,
            rudder_scale=RUDDER_SCALE,
        )
    )

    candidate = PPO.load(
        str(
            BC_MODEL_PATH
        )
        +
        ".zip",
        env=train_env,
        device="auto",
    )

    candidate.learning_rate = float(
        lr
    )

    candidate.lr_schedule = (
        lambda progress_remaining,
        lr=float(lr):
        lr
    )

    candidate.ent_coef = 0.0
    candidate.target_kl = 0.005

    # Keep exploration small but nonzero.
    with torch.no_grad():
        if hasattr(
            candidate.policy,
            "log_std",
        ):
            candidate.policy.log_std.fill_(
                math.log(
                    RL_EXPLORATION_STD
                )
            )

        candidate.policy.action_net.weight[
            2
        ].zero_()

        candidate.policy.action_net.bias[
            2
        ].fill_(
            TEACHER_AILERON_ACTION
        )

        candidate.policy.action_net.weight[
            3
        ].zero_()

        candidate.policy.action_net.bias[
            3
        ].fill_(
            TEACHER_RUDDER_ACTION
        )

    # Freeze only action-head rows 2/3 during PPO.
    def freeze_lateral_weight_rows(
        grad
    ):
        g = grad.clone()
        g[2].zero_()
        g[3].zero_()
        return g

    def freeze_lateral_bias_rows(
        grad
    ):
        g = grad.clone()
        g[2] = 0.0
        g[3] = 0.0
        return g

    hook_w = (
        candidate.policy
        .action_net
        .weight
        .register_hook(
            freeze_lateral_weight_rows
        )
    )

    hook_b = (
        candidate.policy
        .action_net
        .bias
        .register_hook(
            freeze_lateral_bias_rows
        )
    )

    candidate.learn(
        total_timesteps=int(
            timesteps
        ),
        reset_num_timesteps=True,
        progress_bar=True,
    )

    hook_w.remove()
    hook_b.remove()

    # Reassert exact lateral/yaw distilled rows after optimizer updates.
    with torch.no_grad():
        candidate.policy.action_net.weight[
            2
        ].zero_()

        candidate.policy.action_net.bias[
            2
        ].fill_(
            TEACHER_AILERON_ACTION
        )

        candidate.policy.action_net.weight[
            3
        ].zero_()

        candidate.policy.action_net.bias[
            3
        ].fill_(
            TEACHER_RUDDER_ACTION
        )

    candidate_path = (
        OUT_DIR
        /
        (
            "AH1S_STAGE2_RL_"
            f"{idx:02d}_"
            f"{timesteps}STEPS"
        )
    )

    candidate.save(
        str(
            candidate_path
        )
    )

    train_env.close()

    r = evaluate_controller(
        policy_model=candidate,
        detailed=False,
    )

    r["rl_lr"] = float(lr)
    r["rl_timesteps"] = int(
        timesteps
    )
    r["rl_candidate_path"] = (
        str(
            candidate_path
        )
        +
        ".zip"
    )

    rl_results.append(
        r
    )

    print_result(
        f"RL {idx}",
        r,
    )


# =====================================================================
# F — SELECT A PASSING RL-FINE-TUNED MODEL
# =====================================================================

passing_rl = [
    r
    for r in rl_results
    if r[
        "presentation_pass"
    ]
]

if not passing_rl:
    rule(
        "NO PPO RL CANDIDATE RETAINED PRESENTATION QUALITY"
    )

    print(
        "BC policy passes, but none of the short PPO RL refinements "
        "retained the strict continuous-flight corridor."
    )

    print(
        "No final hybrid model will be falsely labeled successful."
    )

    raise RuntimeError(
        "All PPO RL fine-tune candidates failed continuous validation."
    )


passing_rl.sort(
    key=selection_key
)

best_rl_result = passing_rl[
    0
]

best_rl_model = PPO.load(
    best_rl_result[
        "rl_candidate_path"
    ]
)

best_rl_model.save(
    str(
        FINAL_MODEL_PATH
    )
)


# =====================================================================
# G — FINAL TRUE CONTINUOUS TEACHER-OFF VALIDATION
# =====================================================================

rule(
    "G — FINAL TRUE CONTINUOUS STAGE1->STAGE2 TEACHER-OFF FLIGHT"
)

print(
    "Stage-1 teacher            : OFF"
)
print(
    "Stage-1 runtime controllers: OFF"
)
print(
    "Stage-2 teacher            : OFF"
)
print(
    "Stage-2 altitude controller: OFF"
)
print(
    "Stage-2 lateral controller : OFF"
)
print(
    "Stage-2 training method    : DISTILLATION + PPO RL FINE-TUNE"
)
print(
    "Stage-2 policy             : SINGLE 4-ACTION PPO"
)
print(
    "Mapped Stage-2 env         : ON (actuator wiring repair only)"
)

final_result = evaluate_controller(
    policy_model=best_rl_model,
    detailed=True,
)

print_result(
    "FINAL HYBRID PPO",
    final_result,
)

if not final_result[
    "presentation_pass"
]:
    raise RuntimeError(
        "Unexpected final re-validation failure. "
        "Final model was NOT accepted."
    )


# =====================================================================
# SAVE REPORT / TRACE
# =====================================================================

summary = {
    k: v
    for k, v in final_result.items()
    if k != "trace"
}

summary[
    "training_method"
] = (
    "teacher-assisted behavior cloning/distillation "
    "+ PPO reinforcement-learning fine-tuning"
)

summary[
    "teacher_runtime"
] = False

summary[
    "final_model"
] = (
    str(
        FINAL_MODEL_PATH
    )
    +
    ".zip"
)

summary[
    "selected_rl_lr"
] = best_rl_result[
    "rl_lr"
]

summary[
    "selected_rl_timesteps"
] = best_rl_result[
    "rl_timesteps"
]

summary[
    "selected_teacher_cfg"
] = {
    "kp_alt":
        BEST_TEACHER_CFG[0],
    "kd_vs":
        BEST_TEACHER_CFG[1],
    "max_collective_correction":
        BEST_TEACHER_CFG[2],
}

with open(
    RESULT_DIR
    /
    "final_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

trace_path = (
    RESULT_DIR
    /
    "final_trace.csv"
)

with open(
    trace_path,
    "w",
    encoding="utf-8",
) as f:
    header = [
        "time",
        "distance",
        "ground_forward",
        "cross",
        "altitude",
        "vertical_speed",
        "lateral_velocity",
        "heading_error_deg",
        "roll_deg",
        "a0",
        "a1",
        "a2",
        "a3",
    ]

    f.write(
        ",".join(header)
        +
        "\n"
    )

    for row in final_result[
        "trace"
    ]:
        f.write(
            ",".join(
                str(
                    row[key]
                )
                for key in header
            )
            +
            "\n"
        )


# =====================================================================
# FINAL REPORT
# =====================================================================

rule(
    "STAGE 2 HYBRID FINAL — LOCKED"
)

print(
    "TRUE continuous mission:"
)
print(
    "takeoff -> 300 ft hover -> 300 ft straight forward"
)
print()

print(
    "Teacher at FINAL runtime : OFF"
)
print(
    "PPO RL fine-tuning       : YES"
)
print(
    "Distillation             : YES"
)
print(
    "Mapped actuator repair   : YES"
)
print()

print(
    "Selected RL learning rate:",
    best_rl_result[
        "rl_lr"
    ]
)

print(
    "Selected RL timesteps    :",
    best_rl_result[
        "rl_timesteps"
    ]
)

print()
print(
    "FINAL MODEL:"
)
print(
    str(
        FINAL_MODEL_PATH
    )
    +
    ".zip"
)

print()
print(
    "RESULT SUMMARY:"
)
print(
    RESULT_DIR
    /
    "final_summary.json"
)

print(
    "RESULT TRACE:"
)
print(
    trace_path
)

print()
print(
    "NEXT: back up the final Stage-2 model immediately, "
    "then resume Stage-3 endpoint lateral hover."
)
