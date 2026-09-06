from __future__ import annotations

"""
AH-1S / JSBSim
STAGE 4 COLLECTIVE / DESCENT AUTHORITY DIAGNOSTIC V1
====================================================

Purpose
-------
This script does NOT train Stage 4 and does NOT change any locked model.
It performs a controlled system-identification experiment around the
already locked Stage-3 endpoint hover.

Method
------
1) Reproduce the true continuous locked Stage1 -> Stage2 -> Stage3 flight.
2) Require the Stage-3 PPO policy, with all runtime teachers/controllers OFF,
   to achieve the 5 s endpoint hover handoff.
3) For each collective residual test, rebuild that same deterministic handoff
   from scratch so every pulse starts from the same physical mission state.
4) Keep the locked Stage-3 PPO policy active as the stabilizing baseline on all
   four channels, but add a known residual ONLY to action[0] (collective) for
   a short pulse. Other PPO actions are left unchanged.
5) Measure vertical-speed response, altitude response, physical collective
   command, and horizontal/attitude coupling.
6) Save CSV + JSON evidence. No gain tuning and no Stage-4 policy training is
   performed here.

Interpretation
--------------
This is a local CLOSED-LOOP collective-authority identification around the
Stage-3 endpoint hover. It is intentionally safer than an open-loop collective
step and is sufficient to establish sign, usable authority, and a first
quantitative descent-response model before designing the Stage-4 teacher.
"""

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill
from helicopter_env_stage2_refine_mapped import HelicopterEnvStage2RefineMapped


# =====================================================================
# LOCKED INPUTS — DO NOT MODIFY
# =====================================================================

STAGE1_MODEL_PATH = Path(
    "models_stage1_early_distilled/AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL_PATH = Path(
    "models_stage2_hybrid_final/AH1S_STAGE2_HYBRID_FINAL.zip"
)

STAGE3_MODEL_PATH = Path(
    "models_stage3_hybrid_final/AH1S_STAGE3_HYBRID_FINAL.zip"
)

STAGE3_SUMMARY_PATH = Path(
    "results_stage3_hybrid_final/final_summary.json"
)

RESULT_DIR = Path(
    "results_stage4_collective_diagnostic_v1"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

for required in [
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
    STAGE3_MODEL_PATH,
    STAGE3_SUMMARY_PATH,
]:
    if not required.exists():
        raise FileNotFoundError(f"Required locked input missing: {required}")


# =====================================================================
# REPRODUCIBILITY / MISSION CONSTANTS
# =====================================================================

SEED = 42
np.random.seed(SEED)

TARGET_FORWARD_FT = 300.0
TARGET_ALT_FT = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
STAGE3_MAX_TIME = 90.0

HANDOFF_STABLE_TIME = 5.0
STAGE3_TAKEOVER_FORWARD_FT = 80.0
DEFAULT_CONTROL_DT = 0.075

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040
EARTH_RADIUS_FT = 20_902_231.0

# Stage-3 acceptance criteria used by the locked final build.
STOP_POS_TOL_FT = 5.0
STOP_SPEED_TOL_FPS = 0.60
STOP_HOLD_SECONDS = 5.0
CROSS_TOL_FT = 5.0
LAT_SPEED_TOL_FPS = 0.60
ALT_MIN = 295.0
ALT_MAX = 305.0
VS_TOL_FPS = 0.75
HEADING_TOL_DEG = 1.0

# Broad safety envelope while reproducing Stage 3.
STAGE3_ALT_SAFE_MIN = 288.0
STAGE3_ALT_SAFE_MAX = 312.0
STAGE3_MAX_ABS_PITCH_DEG = 8.0
STAGE3_MAX_ABS_ROLL_DEG = 10.0
STAGE3_MAX_CROSS_SAFE_FT = 15.0

# Stage-4 authority experiment.
# Conservative residuals first: no training, no permanent controller change.
COLLECTIVE_DELTAS = [
    -0.15,
    -0.10,
    -0.05,
    0.00,
    +0.05,
    +0.10,
    +0.15,
]

PULSE_SECONDS = 2.0
RECOVERY_SECONDS = 3.0

# Safety only for the short authority test. These are intentionally wider
# than mission acceptance but still stop a clearly bad perturbation early.
ID_ALT_SAFE_MIN = 270.0
ID_ALT_SAFE_MAX = 315.0
ID_MAX_ABS_PITCH_DEG = 10.0
ID_MAX_ABS_ROLL_DEG = 12.0
ID_MAX_ABS_CROSS_FT = 10.0
ID_MAX_ABS_POSITION_ERROR_FT = 10.0


# =====================================================================
# MODELS
# =====================================================================

stage1_model = PPO.load(str(STAGE1_MODEL_PATH))
stage2_model = PPO.load(str(STAGE2_MODEL_PATH))
stage3_model = PPO.load(str(STAGE3_MODEL_PATH))


# =====================================================================
# SMALL HELPERS
# =====================================================================

def rule(text: str) -> None:
    print()
    print("=" * 154)
    print(text)
    print("=" * 154)


def fdm_float(fdm, key, default=float("nan")) -> float:
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def first_finite(fdm, keys, default=float("nan")) -> float:
    for key in keys:
        value = fdm_float(fdm, key)
        if np.isfinite(value):
            return value
    return float(default)


def info_float(info, key, default=float("nan")) -> float:
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def get_fdm(env):
    if getattr(env, "fdm", None) is not None:
        return env.fdm
    base = getattr(env, "base_env", None)
    if base is not None and getattr(base, "fdm", None) is not None:
        return base.fdm
    raise RuntimeError("Active FDM not found.")


def latitude_deg(fdm) -> float:
    return first_finite(fdm, ["position/lat-gc-deg", "position/lat-geod-deg"])


def longitude_deg(fdm) -> float:
    return first_finite(fdm, ["position/long-gc-deg", "position/long-geod-deg"])


def heading_rad(fdm) -> float:
    return first_finite(fdm, ["attitude/heading-true-rad", "attitude/psi-rad"])


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def local_ne_ft(lat, lon, lat0, lon0):
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    north = EARTH_RADIUS_FT * dlat
    east = EARTH_RADIUS_FT * math.cos(math.radians(lat0)) * dlon
    return float(north), float(east)


def mission_axes(north, east, heading):
    c = math.cos(heading)
    s = math.sin(heading)
    forward = north * c + east * s
    cross = -north * s + east * c
    return float(forward), float(cross)


def geometry(fdm, lat0, lon0, mission_heading):
    north, east = local_ne_ft(
        latitude_deg(fdm),
        longitude_deg(fdm),
        lat0,
        lon0,
    )
    return mission_axes(north, east, mission_heading)


def mission_ground_velocity(fdm, mission_heading):
    vn = first_finite(fdm, ["velocities/v-north-fps"])
    ve = first_finite(fdm, ["velocities/v-east-fps"])
    if not np.isfinite(vn) or not np.isfinite(ve):
        return float("nan"), float("nan")
    return mission_axes(vn, ve, mission_heading)


def physical_commands(fdm):
    return {
        "physical_collective_cmd": fdm_float(fdm, "fcs/collective-cmd-norm"),
        "physical_elevator_cmd": fdm_float(fdm, "fcs/elevator-cmd-norm"),
        "physical_aileron_cmd": fdm_float(fdm, "fcs/aileron-cmd-norm"),
        "physical_rudder_cmd": fdm_float(fdm, "fcs/rudder-cmd-norm"),
    }


def snapshot(fdm, lat0, lon0, mission_heading):
    forward, cross = geometry(fdm, lat0, lon0, mission_heading)
    heading_error = wrap_angle(heading_rad(fdm) - mission_heading)
    ground_fwd, ground_cross = mission_ground_velocity(fdm, mission_heading)

    state = {
        "forward_ft": float(forward),
        "position_error_ft": float(TARGET_FORWARD_FT - forward),
        "cross_track_ft": float(cross),
        "altitude_ft": fdm_float(fdm, "position/h-agl-ft"),
        "forward_speed_fps": fdm_float(fdm, "velocities/u-aero-fps", 0.0),
        "lateral_speed_fps": fdm_float(fdm, "velocities/v-aero-fps", 0.0),
        "mission_ground_forward_speed_fps": float(ground_fwd),
        "mission_ground_cross_speed_fps": float(ground_cross),
        "vertical_speed_fps": fdm_float(fdm, "velocities/h-dot-fps", 0.0),
        "heading_error_rad": float(heading_error),
        "heading_error_deg": float(math.degrees(heading_error)),
        "pitch_rad": fdm_float(fdm, "attitude/pitch-rad", 0.0),
        "roll_rad": fdm_float(fdm, "attitude/roll-rad", 0.0),
        "roll_rate_rad_s": fdm_float(fdm, "velocities/p-rad_sec", 0.0),
        "pitch_rate_rad_s": fdm_float(fdm, "velocities/q-rad_sec", 0.0),
        "yaw_rate_rad_s": fdm_float(fdm, "velocities/r-rad_sec", 0.0),
        "rotor_rpm": fdm_float(fdm, "propulsion/engine/rotor-rpm", 323.0),
    }
    state.update(physical_commands(fdm))
    return state


def stage3_observation(state):
    obs = np.array(
        [
            (state["altitude_ft"] - TARGET_ALT_FT) / 10.0,
            state["vertical_speed_fps"] / 5.0,
            state["position_error_ft"] / 250.0,
            state["forward_speed_fps"] / 12.0,
            state["cross_track_ft"] / 10.0,
            state["lateral_speed_fps"] / 5.0,
            math.sin(state["heading_error_rad"]),
            math.cos(state["heading_error_rad"]),
            state["pitch_rad"] / 0.20,
            state["roll_rad"] / 0.20,
            state["roll_rate_rad_s"] / 1.0,
            state["pitch_rate_rad_s"] / 1.0,
            state["yaw_rate_rad_s"] / 1.0,
            state["rotor_rpm"] / 400.0,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -10.0, +10.0).astype(np.float32)


def endpoint_stop_now(state) -> bool:
    return bool(
        abs(state["position_error_ft"]) <= STOP_POS_TOL_FT
        and abs(state["forward_speed_fps"]) <= STOP_SPEED_TOL_FPS
    )


def endpoint_lateral_now(state) -> bool:
    return bool(
        abs(state["cross_track_ft"]) <= CROSS_TOL_FT
        and abs(state["lateral_speed_fps"]) <= LAT_SPEED_TOL_FPS
    )


def endpoint_hover_now(state) -> bool:
    return bool(
        endpoint_stop_now(state)
        and endpoint_lateral_now(state)
        and ALT_MIN <= state["altitude_ft"] <= ALT_MAX
        and abs(state["vertical_speed_fps"]) <= VS_TOL_FPS
        and abs(state["heading_error_deg"]) <= HEADING_TOL_DEG
    )


def stage3_safety_reason(state) -> str:
    if state["altitude_ft"] < STAGE3_ALT_SAFE_MIN:
        return "altitude_below_stage3_safe"
    if state["altitude_ft"] > STAGE3_ALT_SAFE_MAX:
        return "altitude_above_stage3_safe"
    if abs(math.degrees(state["pitch_rad"])) > STAGE3_MAX_ABS_PITCH_DEG:
        return "pitch_stage3_limit"
    if abs(math.degrees(state["roll_rad"])) > STAGE3_MAX_ABS_ROLL_DEG:
        return "roll_stage3_limit"
    if abs(state["cross_track_ft"]) > STAGE3_MAX_CROSS_SAFE_FT:
        return "cross_stage3_limit"
    return ""


def id_safety_reason(state) -> str:
    if state["altitude_ft"] < ID_ALT_SAFE_MIN:
        return "altitude_below_id_safe"
    if state["altitude_ft"] > ID_ALT_SAFE_MAX:
        return "altitude_above_id_safe"
    if abs(math.degrees(state["pitch_rad"])) > ID_MAX_ABS_PITCH_DEG:
        return "pitch_id_limit"
    if abs(math.degrees(state["roll_rad"])) > ID_MAX_ABS_ROLL_DEG:
        return "roll_id_limit"
    if abs(state["cross_track_ft"]) > ID_MAX_ABS_CROSS_FT:
        return "cross_id_limit"
    if abs(state["position_error_ft"]) > ID_MAX_ABS_POSITION_ERROR_FT:
        return "forward_position_id_limit"
    return ""


def env_control_dt(env) -> float:
    dt = float(getattr(env, "dt", DEFAULT_CONTROL_DT) or DEFAULT_CONTROL_DT)
    if not np.isfinite(dt) or dt <= 0.0:
        dt = DEFAULT_CONTROL_DT
    return dt


def physics_steps(env) -> int:
    try:
        value = int(getattr(env, "PHYSICS_STEPS", 10))
    except Exception:
        value = 10
    return max(1, value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def close_handoff(start) -> None:
    env2 = start.get("env2")
    env1 = start.get("env1")
    if env2 is not None:
        try:
            env2.fdm = None
        except Exception:
            pass
    if env1 is not None:
        try:
            env1.close()
        except Exception:
            pass


# =====================================================================
# RAW STAGE-3/4 PHYSICS CYCLE
# =====================================================================

def raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -1.0, +1.0).astype(np.float32)

    # Mapped Stage-2 actuator path: this is the already validated wiring.
    env2._apply_action(action)

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError("JSBSim stopped during Stage-4 diagnostic.")

    if hasattr(env2, "previous_action"):
        try:
            env2.previous_action = action.copy()
        except Exception:
            pass

    # Match the validated Stage-3 raw-cycle bookkeeping order.
    state = snapshot(fdm, lat0, lon0, mission_heading)

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = float(state["forward_ft"])

    if hasattr(env2, "steps"):
        try:
            env2.steps += 1
        except Exception:
            pass

    # Refresh Stage-2 bookkeeping even though Stage 3/4 uses its own obs.
    try:
        env2._get_obs()
    except Exception:
        pass

    return state, action


# =====================================================================
# TRUE CONTINUOUS STAGE1 -> STAGE2 -> LOCKED STAGE3 HANDOFF
# =====================================================================

def build_stage4_handoff(detailed=False):
    env1 = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )
    obs1, info1 = env1.reset()

    fdm = get_fdm(env1)
    active_fdm_id = id(fdm)
    mission_heading = heading_rad(fdm)
    dt1 = env_control_dt(env1)

    sim_time_at_stage1_start = first_finite(
        fdm,
        ["simulation/sim-time-sec"],
    )

    stable_time = 0.0
    stage1_elapsed = 0.0

    for _ in range(int(STAGE1_MAX_TIME / dt1)):
        action1, _ = stage1_model.predict(obs1, deterministic=True)
        obs1, _, terminated, truncated, info1 = env1.step(action1)
        stage1_elapsed += dt1

        altitude = info_float(info1, "altitude")
        vertical_speed = info_float(info1, "vertical_speed")
        vn = info_float(info1, "vn", 0.0)
        ve = info_float(info1, "ve", 0.0)
        horizontal_speed = float(np.hypot(vn, ve))
        drift = info_float(info1, "drift", 999.0)

        stable = bool(
            295.0 <= altitude <= 305.0
            and abs(vertical_speed) <= 0.50
            and horizontal_speed <= 1.0
            and drift <= 3.0
        )
        stable_time = stable_time + dt1 if stable else 0.0

        if stable_time >= HANDOFF_STABLE_TIME:
            break

        if terminated and not bool(info1.get("success", False)):
            env1.close()
            raise RuntimeError("Stage 1 failed before stable handoff.")
        if truncated:
            env1.close()
            raise RuntimeError("Stage 1 truncated before stable handoff.")

    if stable_time < HANDOFF_STABLE_TIME:
        env1.close()
        raise RuntimeError("Stable Stage-1 handoff was not reached.")

    # Mission geometry origin = true Stage-1 hover handoff.
    lat0 = latitude_deg(fdm)
    lon0 = longitude_deg(fdm)

    sim_time_before_attach = first_finite(fdm, ["simulation/sim-time-sec"])

    env2 = HelicopterEnvStage2RefineMapped(
        aileron_scale=AILERON_SCALE,
        rudder_scale=RUDDER_SCALE,
    )

    # Disposable reset initializes Python-side bookkeeping only.
    env2.reset()
    env2.fdm = fdm

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = 0.0
    if hasattr(env2, "target_heading"):
        env2.target_heading = float(mission_heading)

    for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
        if hasattr(env2, attr):
            setattr(env2, attr, 0)

    if id(get_fdm(env2)) != active_fdm_id:
        env2.fdm = None
        env1.close()
        raise RuntimeError("FDM continuity failed during Stage-2 attach.")

    sim_time_after_attach = first_finite(fdm, ["simulation/sim-time-sec"])
    clock_reset_on_attach = bool(
        np.isfinite(sim_time_before_attach)
        and np.isfinite(sim_time_after_attach)
        and abs(sim_time_after_attach - sim_time_before_attach) > 1e-9
    )
    if clock_reset_on_attach:
        env2.fdm = None
        env1.close()
        raise RuntimeError("Simulation clock changed during Stage-2 attach.")

    obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
    dt2 = env_control_dt(env2)
    stage2_elapsed = 0.0

    reached_stage3_takeover = False

    for _ in range(int(STAGE2_MAX_TIME / dt2)):
        action2, _ = stage2_model.predict(obs2, deterministic=True)
        action2 = np.asarray(action2, dtype=np.float32).reshape(-1)

        obs2, _, terminated, truncated, info2 = env2.step(action2)
        obs2 = np.asarray(obs2, dtype=np.float32)
        stage2_elapsed += dt2

        forward, _cross = geometry(fdm, lat0, lon0, mission_heading)
        if forward >= STAGE3_TAKEOVER_FORWARD_FT:
            if hasattr(env2, "forward_distance"):
                env2.forward_distance = float(forward)
            reached_stage3_takeover = True
            break

        if terminated and not bool(info2.get("success", False)):
            env2.fdm = None
            env1.close()
            raise RuntimeError("Stage 2 failed before Stage-3 takeover.")
        if truncated:
            env2.fdm = None
            env1.close()
            raise RuntimeError("Stage 2 truncated before Stage-3 takeover.")

    if not reached_stage3_takeover:
        env2.fdm = None
        env1.close()
        raise RuntimeError("Stage-3 takeover distance was not reached.")

    # Refresh bookkeeping after the true forward-distance correction.
    try:
        env2._get_obs()
    except Exception:
        pass

    dt3 = dt2
    hover_hold = 0.0
    stage3_elapsed = 0.0

    min_alt = +999.0
    max_alt = -999.0
    max_cross = 0.0
    max_abs_pitch = 0.0
    max_abs_roll = 0.0
    max_abs_heading = 0.0

    next_print = 0.0
    state = snapshot(fdm, lat0, lon0, mission_heading)

    for step in range(int(STAGE3_MAX_TIME / dt3)):
        obs3 = stage3_observation(state)
        action3, _ = stage3_model.predict(obs3, deterministic=True)
        action3 = np.asarray(action3, dtype=np.float32).reshape(-1)

        state, _used = raw_policy_cycle(
            env2,
            fdm,
            action3,
            lat0,
            lon0,
            mission_heading,
        )

        stage3_elapsed = (step + 1) * dt3

        min_alt = min(min_alt, state["altitude_ft"])
        max_alt = max(max_alt, state["altitude_ft"])
        max_cross = max(max_cross, abs(state["cross_track_ft"]))
        max_abs_pitch = max(max_abs_pitch, abs(math.degrees(state["pitch_rad"])))
        max_abs_roll = max(max_abs_roll, abs(math.degrees(state["roll_rad"])))
        max_abs_heading = max(max_abs_heading, abs(state["heading_error_deg"]))

        reason = stage3_safety_reason(state)
        if reason:
            env2.fdm = None
            env1.close()
            raise RuntimeError(f"Locked Stage 3 violated safety before handoff: {reason}")

        hover_hold = hover_hold + dt3 if endpoint_hover_now(state) else 0.0

        if detailed and stage3_elapsed + 1e-9 >= next_print:
            print(
                f"Stage3 t={stage3_elapsed:6.2f}s | "
                f"FWD={state['forward_ft']:7.2f} | "
                f"V={state['forward_speed_fps']:+6.2f} | "
                f"X={state['cross_track_ft']:+6.2f} | "
                f"LAT={state['lateral_speed_fps']:+6.2f} | "
                f"ALT={state['altitude_ft']:7.2f} | "
                f"VS={state['vertical_speed_fps']:+6.2f} | "
                f"HOLD={hover_hold:4.2f}s"
            )
            next_print += 10.0

        if hover_hold >= STOP_HOLD_SECONDS:
            break

    handoff_pass = bool(
        hover_hold >= STOP_HOLD_SECONDS
        and endpoint_hover_now(state)
    )

    if not handoff_pass:
        env2.fdm = None
        env1.close()
        raise RuntimeError(
            "Locked Stage-3 final PPO did not reproduce the 5 s endpoint hover. "
            "Do not run Stage-4 identification."
        )

    sim_time_stage4_start = first_finite(fdm, ["simulation/sim-time-sec"])

    return {
        "env1": env1,
        "env2": env2,
        "fdm": fdm,
        "lat0": float(lat0),
        "lon0": float(lon0),
        "mission_heading": float(mission_heading),
        "active_fdm_id": int(active_fdm_id),
        "same_fdm": bool(id(fdm) == active_fdm_id and id(get_fdm(env2)) == active_fdm_id),
        "clock_reset_on_attach": bool(clock_reset_on_attach),
        "sim_time_at_stage1_start": float(sim_time_at_stage1_start),
        "sim_time_stage4_start": float(sim_time_stage4_start),
        "stage1_elapsed_s": float(stage1_elapsed),
        "stage2_elapsed_s": float(stage2_elapsed),
        "stage3_elapsed_s": float(stage3_elapsed),
        "stage3_hover_hold_s": float(hover_hold),
        "stage3_min_alt_ft": float(min_alt),
        "stage3_max_alt_ft": float(max_alt),
        "stage3_max_cross_ft": float(max_cross),
        "stage3_max_abs_pitch_deg": float(max_abs_pitch),
        "stage3_max_abs_roll_deg": float(max_abs_roll),
        "stage3_max_abs_heading_deg": float(max_abs_heading),
        "state": state.copy(),
        "handoff_pass": bool(handoff_pass),
    }


# =====================================================================
# STAGE-4 COLLECTIVE PULSE CASE
# =====================================================================

def run_collective_case(delta_action0: float, case_index: int, detailed=False):
    start = build_stage4_handoff(detailed=False)

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = env_control_dt(env2)

    start_state = snapshot(fdm, lat0, lon0, mission_heading)
    start_obs3 = stage3_observation(start_state)
    start_policy_action, _ = stage3_model.predict(start_obs3, deterministic=True)
    start_policy_action = np.asarray(start_policy_action, dtype=np.float32).reshape(-1)

    pulse_steps = max(1, int(round(PULSE_SECONDS / dt)))
    recovery_steps = max(1, int(round(RECOVERY_SECONDS / dt)))
    total_steps = pulse_steps + recovery_steps

    trace = []
    safe = True
    termination_reason = "completed"
    pulse_end_state = start_state.copy()

    min_alt = start_state["altitude_ft"]
    max_alt = start_state["altitude_ft"]
    max_abs_cross = abs(start_state["cross_track_ft"])
    max_abs_pos_err = abs(start_state["position_error_ft"])
    max_abs_pitch = abs(math.degrees(start_state["pitch_rad"]))
    max_abs_roll = abs(math.degrees(start_state["roll_rad"]))
    max_abs_heading = abs(start_state["heading_error_deg"])
    min_vs = start_state["vertical_speed_fps"]
    max_vs = start_state["vertical_speed_fps"]

    applied_a0_pulse = []
    policy_a0_pulse = []
    physical_collective_pulse = []

    for step in range(total_steps):
        t_before = step * dt
        phase = "pulse" if step < pulse_steps else "recovery"

        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        obs3 = stage3_observation(state_before)
        policy_action, _ = stage3_model.predict(obs3, deterministic=True)
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)

        applied_action = policy_action.copy()
        residual = float(delta_action0 if phase == "pulse" else 0.0)
        applied_action[0] = float(
            np.clip(applied_action[0] + residual, -1.0, +1.0)
        )

        state_after, used_action = raw_policy_cycle(
            env2,
            fdm,
            applied_action,
            lat0,
            lon0,
            mission_heading,
        )

        t_after = (step + 1) * dt

        min_alt = min(min_alt, state_after["altitude_ft"])
        max_alt = max(max_alt, state_after["altitude_ft"])
        max_abs_cross = max(max_abs_cross, abs(state_after["cross_track_ft"]))
        max_abs_pos_err = max(max_abs_pos_err, abs(state_after["position_error_ft"]))
        max_abs_pitch = max(max_abs_pitch, abs(math.degrees(state_after["pitch_rad"])))
        max_abs_roll = max(max_abs_roll, abs(math.degrees(state_after["roll_rad"])))
        max_abs_heading = max(max_abs_heading, abs(state_after["heading_error_deg"]))
        min_vs = min(min_vs, state_after["vertical_speed_fps"])
        max_vs = max(max_vs, state_after["vertical_speed_fps"])

        if phase == "pulse":
            applied_a0_pulse.append(float(used_action[0]))
            policy_a0_pulse.append(float(policy_action[0]))
            physical_collective_pulse.append(float(state_after["physical_collective_cmd"]))

        if step == pulse_steps - 1:
            pulse_end_state = state_after.copy()

        trace.append(
            {
                "case_index": int(case_index),
                "delta_action0": float(delta_action0),
                "time_s": float(t_after),
                "phase": phase,
                "policy_action0": float(policy_action[0]),
                "applied_action0": float(used_action[0]),
                "applied_collective_residual": float(residual),
                "action1": float(used_action[1]),
                "action2": float(used_action[2]),
                "action3": float(used_action[3]),
                **{k: float(v) for k, v in state_after.items()},
            }
        )

        reason = id_safety_reason(state_after)
        if reason:
            safe = False
            termination_reason = reason
            break

        if detailed and (
            step == 0
            or step == pulse_steps - 1
            or step == total_steps - 1
        ):
            print(
                f"    {phase:8s} t={t_after:5.2f}s | "
                f"A0policy={policy_action[0]:+7.4f} | "
                f"A0used={used_action[0]:+7.4f} | "
                f"COLL={state_after['physical_collective_cmd']:+7.4f} | "
                f"ALT={state_after['altitude_ft']:7.2f} | "
                f"VS={state_after['vertical_speed_fps']:+6.3f} | "
                f"FWD={state_after['forward_ft']:7.2f} | "
                f"X={state_after['cross_track_ft']:+6.2f}"
            )

    final_state = snapshot(fdm, lat0, lon0, mission_heading)

    if not trace:
        close_handoff(start)
        raise RuntimeError("Collective diagnostic case produced no samples.")

    physical_start = float(start_state["physical_collective_cmd"])
    mean_physical_pulse = float(np.mean(physical_collective_pulse)) if physical_collective_pulse else float("nan")
    mean_policy_a0_pulse = float(np.mean(policy_a0_pulse)) if policy_a0_pulse else float("nan")
    mean_applied_a0_pulse = float(np.mean(applied_a0_pulse)) if applied_a0_pulse else float("nan")

    actual_pulse_duration = min(PULSE_SECONDS, len(applied_a0_pulse) * dt)
    delta_vs_end = float(
        pulse_end_state["vertical_speed_fps"] - start_state["vertical_speed_fps"]
    )
    delta_alt_end = float(
        pulse_end_state["altitude_ft"] - start_state["altitude_ft"]
    )

    avg_vertical_accel = float(
        delta_vs_end / actual_pulse_duration
        if actual_pulse_duration > 1e-9
        else float("nan")
    )

    result = {
        "case_index": int(case_index),
        "delta_action0": float(delta_action0),
        "safe": bool(safe),
        "termination_reason": str(termination_reason),
        "handoff_pass": bool(start["handoff_pass"]),
        "same_fdm": bool(start["same_fdm"]),
        "clock_reset_on_attach": bool(start["clock_reset_on_attach"]),
        "pulse_seconds_requested": float(PULSE_SECONDS),
        "pulse_seconds_actual": float(actual_pulse_duration),
        "recovery_seconds_requested": float(RECOVERY_SECONDS),
        "start_forward_ft": float(start_state["forward_ft"]),
        "start_position_error_ft": float(start_state["position_error_ft"]),
        "start_cross_track_ft": float(start_state["cross_track_ft"]),
        "start_altitude_ft": float(start_state["altitude_ft"]),
        "start_vertical_speed_fps": float(start_state["vertical_speed_fps"]),
        "start_forward_speed_fps": float(start_state["forward_speed_fps"]),
        "start_lateral_speed_fps": float(start_state["lateral_speed_fps"]),
        "start_policy_action0": float(start_policy_action[0]),
        "start_physical_collective_cmd": physical_start,
        "mean_policy_action0_pulse": mean_policy_a0_pulse,
        "mean_applied_action0_pulse": mean_applied_a0_pulse,
        "mean_physical_collective_cmd_pulse": mean_physical_pulse,
        "mean_physical_collective_delta": float(mean_physical_pulse - physical_start),
        "pulse_end_altitude_ft": float(pulse_end_state["altitude_ft"]),
        "pulse_end_vertical_speed_fps": float(pulse_end_state["vertical_speed_fps"]),
        "delta_vertical_speed_end_fps": delta_vs_end,
        "delta_altitude_end_ft": delta_alt_end,
        "average_vertical_accel_fps2": avg_vertical_accel,
        "final_forward_ft": float(final_state["forward_ft"]),
        "final_position_error_ft": float(final_state["position_error_ft"]),
        "final_cross_track_ft": float(final_state["cross_track_ft"]),
        "final_altitude_ft": float(final_state["altitude_ft"]),
        "final_vertical_speed_fps": float(final_state["vertical_speed_fps"]),
        "final_forward_speed_fps": float(final_state["forward_speed_fps"]),
        "final_lateral_speed_fps": float(final_state["lateral_speed_fps"]),
        "min_altitude_ft": float(min_alt),
        "max_altitude_ft": float(max_alt),
        "min_vertical_speed_fps": float(min_vs),
        "max_vertical_speed_fps": float(max_vs),
        "max_abs_cross_track_ft": float(max_abs_cross),
        "max_abs_position_error_ft": float(max_abs_pos_err),
        "max_abs_pitch_deg": float(max_abs_pitch),
        "max_abs_roll_deg": float(max_abs_roll),
        "max_abs_heading_error_deg": float(max_abs_heading),
        "trace": trace,
    }

    close_handoff(start)
    return result


# =====================================================================
# CSV HELPERS
# =====================================================================

def write_dict_rows_csv(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =====================================================================
# A — LOCKED HANDOFF QUALIFICATION
# =====================================================================

rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 HANDOFF QUALIFICATION")

print("Stage-1 model:", STAGE1_MODEL_PATH)
print("Stage-2 model:", STAGE2_MODEL_PATH)
print("Stage-3 model:", STAGE3_MODEL_PATH)
print("Stage-1 runtime teacher: OFF")
print("Stage-2 runtime teacher: OFF")
print("Stage-3 runtime teacher/controller: OFF")
print("Stage-3 runtime: locked PPO policy only")
print("Stage-4 training: NONE")

qualification = build_stage4_handoff(detailed=True)
q = qualification["state"]

print()
print(
    "HANDOFF | "
    f"PASS={qualification['handoff_pass']} | "
    f"same_fdm={qualification['same_fdm']} | "
    f"clock_reset={qualification['clock_reset_on_attach']} | "
    f"FWD={q['forward_ft']:.3f} | "
    f"V={q['forward_speed_fps']:+.3f} | "
    f"X={q['cross_track_ft']:+.3f} | "
    f"LAT={q['lateral_speed_fps']:+.3f} | "
    f"ALT={q['altitude_ft']:.3f} | "
    f"VS={q['vertical_speed_fps']:+.3f} | "
    f"HDG={q['heading_error_deg']:+.3f}deg | "
    f"HOLD={qualification['stage3_hover_hold_s']:.2f}s"
)

handoff_summary = {
    k: v
    for k, v in qualification.items()
    if k not in {"env1", "env2", "fdm", "state"}
}
handoff_summary["state"] = {k: float(v) for k, v in q.items()}
handoff_summary["stage1_model_sha256"] = sha256_file(STAGE1_MODEL_PATH)
handoff_summary["stage2_model_sha256"] = sha256_file(STAGE2_MODEL_PATH)
handoff_summary["stage3_model_sha256"] = sha256_file(STAGE3_MODEL_PATH)
handoff_summary["git_head"] = git_head()

with (RESULT_DIR / "handoff_summary.json").open("w", encoding="utf-8") as f:
    json.dump(handoff_summary, f, indent=2)

close_handoff(qualification)


# =====================================================================
# B — COLLECTIVE AUTHORITY SWEEP
# =====================================================================

rule("B — STAGE-4 COLLECTIVE AUTHORITY SWEEP")

print(
    "Each case rebuilds the full locked Stage1 -> Stage2 -> Stage3 flight.\n"
    "During the 2 s pulse, only a residual is added to PPO action[0].\n"
    "The locked Stage-3 PPO remains the stabilizing baseline; no Stage-4 teacher exists yet."
)
print()

case_results = []
all_trace_rows = []

for idx, delta in enumerate(COLLECTIVE_DELTAS, start=1):
    print(f"Case {idx}/{len(COLLECTIVE_DELTAS)} | collective residual {delta:+.3f}")

    try:
        result = run_collective_case(delta, idx, detailed=True)
    except Exception as exc:
        result = {
            "case_index": int(idx),
            "delta_action0": float(delta),
            "safe": False,
            "termination_reason": f"exception: {exc}",
            "handoff_pass": False,
            "same_fdm": False,
            "clock_reset_on_attach": None,
            "trace": [],
        }

    trace = result.pop("trace", [])
    all_trace_rows.extend(trace)
    case_results.append(result)

    if "delta_vertical_speed_end_fps" in result:
        print(
            "  RESULT | "
            f"SAFE={result['safe']} | "
            f"dVS@pulse={result['delta_vertical_speed_end_fps']:+.3f} fps | "
            f"dALT@pulse={result['delta_altitude_end_ft']:+.3f} ft | "
            f"aZ~={result['average_vertical_accel_fps2']:+.3f} ft/s^2 | "
            f"COLL dMean={result['mean_physical_collective_delta']:+.5f} | "
            f"ALT=[{result['min_altitude_ft']:.2f},{result['max_altitude_ft']:.2f}] | "
            f"Xmax={result['max_abs_cross_track_ft']:.2f} | "
            f"PosErrMax={result['max_abs_position_error_ft']:.2f}"
        )
    else:
        print(
            "  RESULT | SAFE=False | ",
            result["termination_reason"],
        )
    print()

# Save sweep without nested trace.
summary_rows = []
for r in case_results:
    summary_rows.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))})

write_dict_rows_csv(
    RESULT_DIR / "collective_authority_sweep.csv",
    summary_rows,
)

if all_trace_rows:
    write_dict_rows_csv(
        RESULT_DIR / "collective_authority_trace.csv",
        all_trace_rows,
    )


# =====================================================================
# C — LOCAL RESPONSE FIT / DECISION EVIDENCE
# =====================================================================

rule("C — LOCAL COLLECTIVE RESPONSE FIT")

fit_rows = [
    r
    for r in case_results
    if bool(r.get("safe", False))
    and np.isfinite(float(r.get("delta_vertical_speed_end_fps", float("nan"))))
]

fit_summary = {
    "fit_available": False,
    "authority_measurable": False,
    "direction_consistent": False,
    "ready_for_stage4_teacher_design": False,
}

if len(fit_rows) >= 3:
    x = np.array([float(r["delta_action0"]) for r in fit_rows], dtype=float)
    y_vs = np.array([float(r["delta_vertical_speed_end_fps"]) for r in fit_rows], dtype=float)
    y_phys = np.array([float(r["mean_physical_collective_delta"]) for r in fit_rows], dtype=float)

    slope_vs, intercept_vs = np.polyfit(x, y_vs, 1)
    slope_phys, intercept_phys = np.polyfit(x, y_phys, 1)

    yhat = slope_vs * x + intercept_vs
    ss_res = float(np.sum((y_vs - yhat) ** 2))
    ss_tot = float(np.sum((y_vs - np.mean(y_vs)) ** 2))
    r2_vs = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    low_idx = int(np.argmin(x))
    high_idx = int(np.argmax(x))
    response_span = float(y_vs[high_idx] - y_vs[low_idx])

    authority_measurable = bool(abs(response_span) >= 0.25)
    direction_consistent = bool(response_span > 0.0 and slope_vs > 0.0)

    # Mildest safe negative residual that creates a meaningful downward
    # vertical-speed change by the end of the pulse. This is evidence only,
    # not a locked Stage-4 teacher setting.
    descent_candidates = sorted(
        [
            r
            for r in fit_rows
            if float(r["delta_action0"]) < 0.0
            and float(r["delta_vertical_speed_end_fps"]) <= -0.50
        ],
        key=lambda r: abs(float(r["delta_action0"])),
    )

    first_useful_descent = None
    if descent_candidates:
        r = descent_candidates[0]
        first_useful_descent = {
            "delta_action0": float(r["delta_action0"]),
            "delta_vertical_speed_end_fps": float(r["delta_vertical_speed_end_fps"]),
            "delta_altitude_end_ft": float(r["delta_altitude_end_ft"]),
            "average_vertical_accel_fps2": float(r["average_vertical_accel_fps2"]),
            "max_abs_cross_track_ft": float(r["max_abs_cross_track_ft"]),
            "max_abs_position_error_ft": float(r["max_abs_position_error_ft"]),
        }

    fit_summary = {
        "fit_available": True,
        "safe_case_count": int(len(fit_rows)),
        "delta_vs_fit": {
            "slope_fps_per_action": float(slope_vs),
            "intercept_fps": float(intercept_vs),
            "r2": float(r2_vs),
        },
        "physical_collective_fit": {
            "slope_cmd_per_action": float(slope_phys),
            "intercept_cmd": float(intercept_phys),
        },
        "response_span_fps": float(response_span),
        "authority_measurable": bool(authority_measurable),
        "direction_consistent": bool(direction_consistent),
        "inferred_sign": (
            "increasing action[0] increases vertical speed; negative residual drives descent"
            if direction_consistent
            else "not yet established reliably"
        ),
        "first_useful_descent_evidence": first_useful_descent,
        "ready_for_stage4_teacher_design": bool(
            authority_measurable
            and direction_consistent
            and len(fit_rows) >= 5
        ),
    }

    print(
        "dVS(2s) ~= "
        f"{slope_vs:+.6f} * collective_residual "
        f"{intercept_vs:+.6f}   (R^2={r2_vs:.4f})"
    )
    print(
        "physical collective delta ~= "
        f"{slope_phys:+.6f} * collective_residual "
        f"{intercept_phys:+.6f}"
    )
    print(f"response span (-max to +max residual): {response_span:+.6f} fps")
    print("authority measurable:", authority_measurable)
    print("direction consistent:", direction_consistent)

    if first_useful_descent is not None:
        print(
            "first useful safe descent evidence: "
            f"delta={first_useful_descent['delta_action0']:+.3f}, "
            f"dVS={first_useful_descent['delta_vertical_speed_end_fps']:+.3f} fps, "
            f"dALT={first_useful_descent['delta_altitude_end_ft']:+.3f} ft"
        )
    else:
        print("No tested negative residual produced >=0.50 fps downward dVS; widen only after review.")
else:
    print("Too few safe cases for a local fit. Do not design the Stage-4 teacher yet.")


# =====================================================================
# D — SAVE AUDIT RECORD
# =====================================================================

rule("D — SAVE STAGE-4 DIAGNOSTIC EVIDENCE")

full_summary = {
    "method": {
        "type": "closed_loop_collective_authority_identification",
        "training_performed": False,
        "stage4_teacher_used": False,
        "baseline_runtime": "locked Stage-3 PPO policy",
        "perturbed_channel": "action[0] collective only",
        "pulse_seconds": float(PULSE_SECONDS),
        "recovery_seconds": float(RECOVERY_SECONDS),
        "collective_deltas": [float(x) for x in COLLECTIVE_DELTAS],
        "seed": int(SEED),
    },
    "locked_inputs": {
        "stage1_model": str(STAGE1_MODEL_PATH),
        "stage2_model": str(STAGE2_MODEL_PATH),
        "stage3_model": str(STAGE3_MODEL_PATH),
        "stage1_sha256": sha256_file(STAGE1_MODEL_PATH),
        "stage2_sha256": sha256_file(STAGE2_MODEL_PATH),
        "stage3_sha256": sha256_file(STAGE3_MODEL_PATH),
        "git_head": git_head(),
    },
    "handoff_qualification": handoff_summary,
    "cases": case_results,
    "fit": fit_summary,
}

with (RESULT_DIR / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
    json.dump(full_summary, f, indent=2)

# Save a standalone trace for the mildest useful descent evidence if available.
useful = fit_summary.get("first_useful_descent_evidence")
if useful is not None:
    chosen_delta = float(useful["delta_action0"])
    chosen_rows = [
        row
        for row in all_trace_rows
        if abs(float(row["delta_action0"]) - chosen_delta) < 1e-12
    ]
    if chosen_rows:
        write_dict_rows_csv(
            RESULT_DIR / "first_useful_descent_trace.csv",
            chosen_rows,
        )

print("Saved:")
print(" ", RESULT_DIR / "handoff_summary.json")
print(" ", RESULT_DIR / "collective_authority_sweep.csv")
print(" ", RESULT_DIR / "collective_authority_trace.csv")
print(" ", RESULT_DIR / "diagnostic_summary.json")
if useful is not None:
    print(" ", RESULT_DIR / "first_useful_descent_trace.csv")

print()
print("READY FOR STAGE-4 TEACHER DESIGN:", fit_summary.get("ready_for_stage4_teacher_design", False))
print()
print(
    "NEXT: send this console output (or diagnostic_summary.json) for review. "
    "Do NOT train Stage 4 yet; first use the measured collective sign/authority "
    "to design the vertical-descent teacher and endpoint XY hold."
)
