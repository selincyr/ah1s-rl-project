from __future__ import annotations

"""
AH-1S / JSBSim
STAGE 4 LONGITUDINAL DESCENT MAPPING CALIBRATION V1
================================================

Purpose
-------
Recalibrate only the Stage-4 lateral station-keeping loop during the already
identified 300 ft -> 30 ft descent regime.  This script is TEACHER CALIBRATION
only: no Stage-4 PPO training, no distillation, and no modification of the
locked Stage-1/2/3 models.

Evidence carried forward
------------------------
1) Collective identification:
   - negative action[0] residual produces descent;
   - dVS(2 s) ~= 2.226895 * residual + 0.002683 (R^2 ~= 1.0);
   - residuals down to -0.50 were physically exercised without saturation;
   - -0.45 sustained for 6 s produced ~-1.065 ft/s mean descent.
2) Longitudinal V3 qualification:
   - transfer at forward >= 294 ft with |Vfwd| <= 1.0 ft/s;
   - hold action[1] = -1.0 while other channels stay on the locked Stage-3 PPO;
   - do NOT descend until the original endpoint envelope has been held >=5 s;
   - that transition held the endpoint corridor for 300 s with max post-entry
     forward error 2.31 ft and max post-entry cross-track 1.37 ft.

V2 teacher design
-----------------
A) Reproduce the locked Stage-3 5 s endpoint hover once as an audit guard.
B) For every descent candidate, rebuild the true continuous Stage1->2->3
   flight and transfer Stage-4 at the qualified 294 ft / <=1.0 ft/s state.
C) Before descent, keep the locked Stage-3 policy on the non-longitudinal
   channels and force action[1] = -1.0 until the full endpoint envelope has
   been held continuously for >=5 s.
D) Freeze the Stage-3 collective action at that qualified pre-descent state as
   the hover reference.  The vertical teacher applies only a bounded collective
   residual in the physically tested [-0.50,+0.15] envelope.
E) During descent, longitudinal action[1] remains the qualified -1.0 maximum
   braking command.  The vertical regime is fixed at the best V2 diagnostic
   working point (Vmax=1.20 ft/s, VS_Kp=0.35).
F) Sweep ONLY lateral Kp/Kd while keeping the Stage-3 lateral trim unchanged.
   Positive A2 authority is capped at +0.4283, the largest descending-regime
   value directly exercised by the preceding authority diagnostic.  A
   candidate passes only if it reaches a stable 30 ft hover for >=5 s while
   the COMPLETE descent stays inside the 5 ft forward and cross-track
   presentation corridors.

Touchdown below 30 ft remains intentionally deferred.  If this script passes,
the next experiment is near-ground / ground-effect / touchdown identification,
not PPO training yet.
"""

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
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
    "results_stage4_teacher_off_corrective_v2"
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

# Stage-4 teacher target: validate the descent regime first, stopping at
# 30 ft AGL. Touchdown/ground-effect logic is intentionally a later stage.
DESCENT_TARGET_ALT_FT = 30.0
DESCENT_MAX_TIME = 520.0
DESCENT_SETTLE_SECONDS = 5.0
DESCENT_ALT_TOL_FT = 2.0
DESCENT_VS_TOL_FPS = 0.25

# Horizontal endpoint criteria throughout Stage 4.
STAGE4_POS_TOL_FT = 5.0
STAGE4_FWD_SPEED_TOL_FPS = 0.60
STAGE4_CROSS_TOL_FT = 5.0
STAGE4_LAT_SPEED_TOL_FPS = 0.60
STAGE4_HEADING_TOL_DEG = 1.0

# Teacher presentation corridor: no visible migration away from the endpoint.
PRESENTATION_MAX_POSITION_ERROR_FT = 5.0
PRESENTATION_MAX_CROSS_FT = 5.0

# Broader fail-fast safety limits.
ID_ALT_SAFE_MIN = 20.0
ID_ALT_SAFE_MAX = 307.0
ID_MAX_ABS_PITCH_DEG = 10.0
ID_MAX_ABS_ROLL_DEG = 12.0
ID_MAX_ABS_CROSS_FT = 10.0
ID_MAX_ABS_POSITION_ERROR_FT = 10.0
STAGE4_MAX_ABS_HEADING_DEG = 5.0
STAGE4_MIN_VERTICAL_SPEED_FPS = -2.5
STAGE4_MAX_VERTICAL_SPEED_FPS = +2.0

# Measured V2 identification evidence (for metadata / audit only).
ID_DVS_2S_SLOPE = 2.226895
ID_DVS_2S_INTERCEPT = 0.002683
ID_PHYSICAL_COLLECTIVE_SLOPE = 0.024397

# Vertical teacher.  The action residual bounds stay inside the physically
# tested Stage-4 authority envelope: -0.50 was tested in V2; +0.15 was tested
# in V1.  We do not silently ask for unqualified authority.
COLLECTIVE_RESIDUAL_MIN = -0.50
COLLECTIVE_RESIDUAL_MAX = +0.15
ALT_TO_VS_GAIN = 0.060
MAX_UPWARD_RECOVERY_VS = 0.30

# Vertical regime is LOCKED here from the best Stage-4 descent V2 diagnostic
# working point.  This experiment changes only lateral feedback gains.
LOCKED_DESCENT_VMAX = 1.20
LOCKED_VS_KP = 0.35

# Descending-regime lateral authority was directly exercised up to A2=+0.4283
# for 2 s and remained safe with a linear response. Lateral gains are LOCKED
# here at the cleanest V2 operating point (C12): Kp=0.100, Kd=0.35.
OLD_IDENTIFIED_A2_MAX = +0.170
DESCENT_TESTED_A2_MAX = +0.4283
LOCKED_LATERAL_KP = 0.100
LOCKED_LATERAL_KD = 0.35

# Qualified Stage-4 mission-manager / pre-descent transition from V3.
STAGE4_ENTRY_FORWARD_FT = 294.0
STAGE4_ENTRY_MAX_SPEED_FPS = 1.00
PRE_DESCENT_HOLD_SECONDS = 5.0
FIXED_BRAKE_A1 = -1.0

# Stage-4 descending-regime elevator identification.
# Existing normalized action1=-1 maps to physical elevator ~= -0.180.
# A sustained -0.003 physical residual (physical ~= -0.183) stayed SAFE and
# inside the 5-ft corridor for 12 s; stronger sustained residuals exited it.
# The Stage-4-specific mapping therefore uses ONLY [-0.183,-0.180].
ELEVATOR_PHYSICAL_MIN = -0.183
ELEVATOR_PHYSICAL_MAX = -0.180
ELEVATOR_PHYSICAL_SPAN = ELEVATOR_PHYSICAL_MAX - ELEVATOR_PHYSICAL_MIN
ID_ELEV_DV_2S_SLOPE = 17.280924
ID_ELEV_DV_2S_INTERCEPT = -0.001712

# Sweep only the physical-feedback gains. Positive braking demand moves the
# mapped elevator toward -0.183; release is capped at the qualified -0.180.
LOCKED_LONG_KPOS_PHYS = 0.0004
LOCKED_LONG_KV_PHYS = 0.002
LOCKED_VS_KI = 0.0040
LOW_ALT_BIAS_FLOOR_CANDIDATES = [-0.04, -0.08, -0.12, -0.16, -0.20, -0.24]
VERTICAL_BIAS_MIN = -0.25
VERTICAL_BIAS_MAX = +0.10
CAPTURE_BLEND_START_ALT_FT = 50.0

LATERAL_TRIM_ACTION = 0.00018325989908678578
IDENTIFIED_A2_MIN = -0.630
RUDDER_ACTION = 0.0


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
            raise RuntimeError("JSBSim stopped during Stage-4 descent teacher calibration.")

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


def raw_stage4_mapped_cycle(env2, fdm, action, lat0, lon0, mission_heading):
    """Run one Stage-4 cycle with the calibrated action1 physical mapping."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -1.0, +1.0).astype(np.float32)

    env2._apply_action(action)
    base_elevator = fdm_float(fdm, "fcs/elevator-cmd-norm")
    mapped_elevator = stage4_action1_to_physical(float(action[1]))
    fdm["fcs/elevator-cmd-norm"] = mapped_elevator

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError("JSBSim stopped during Stage-4 longitudinal mapping calibration.")

    if hasattr(env2, "previous_action"):
        try:
            env2.previous_action = action.copy()
        except Exception:
            pass

    state = snapshot(fdm, lat0, lon0, mission_heading)
    if hasattr(env2, "forward_distance"):
        env2.forward_distance = float(state["forward_ft"])
    if hasattr(env2, "steps"):
        try:
            env2.steps += 1
        except Exception:
            pass
    try:
        env2._get_obs()
    except Exception:
        pass

    return state, action, float(base_elevator), float(mapped_elevator)


# =====================================================================
# TRUE CONTINUOUS STAGE1 -> STAGE2 -> LOCKED STAGE3 HANDOFF
# =====================================================================

def build_stage4_handoff(
    detailed=False,
    require_full_hold=True,
    custom_entry_forward_ft=None,
    custom_entry_max_speed_fps=1.0,
):
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

        if require_full_hold:
            if hover_hold >= STOP_HOLD_SECONDS:
                break
        else:
            # Integrated Stage-4 transition diagnostic.  If no custom trigger is
            # supplied, use the first state inside the locked endpoint envelope
            # (the V2 behavior).  For V3 we may intentionally transfer slightly
            # earlier during the braking approach, but descent is NOT allowed
            # until Stage-4 itself subsequently establishes the full endpoint hold.
            if custom_entry_forward_ft is None:
                entry_ok = endpoint_hover_now(state)
            else:
                entry_ok = bool(
                    state["forward_ft"] >= float(custom_entry_forward_ft)
                    and state["forward_ft"] <= TARGET_FORWARD_FT + STOP_POS_TOL_FT
                    and abs(state["forward_speed_fps"]) <= float(custom_entry_max_speed_fps)
                    and abs(state["cross_track_ft"]) <= CROSS_TOL_FT
                    and abs(state["lateral_speed_fps"]) <= LAT_SPEED_TOL_FPS
                    and ALT_MIN <= state["altitude_ft"] <= ALT_MAX
                    and abs(state["vertical_speed_fps"]) <= VS_TOL_FPS
                    and abs(state["heading_error_deg"]) <= HEADING_TOL_DEG
                )
            if entry_ok:
                break

    if require_full_hold:
        handoff_pass = bool(hover_hold >= STOP_HOLD_SECONDS and endpoint_hover_now(state))
    elif custom_entry_forward_ft is None:
        handoff_pass = bool(endpoint_hover_now(state))
    else:
        handoff_pass = bool(
            state["forward_ft"] >= float(custom_entry_forward_ft)
            and state["forward_ft"] <= TARGET_FORWARD_FT + STOP_POS_TOL_FT
            and abs(state["forward_speed_fps"]) <= float(custom_entry_max_speed_fps)
            and abs(state["cross_track_ft"]) <= CROSS_TOL_FT
            and abs(state["lateral_speed_fps"]) <= LAT_SPEED_TOL_FPS
            and ALT_MIN <= state["altitude_ft"] <= ALT_MAX
            and abs(state["vertical_speed_fps"]) <= VS_TOL_FPS
            and abs(state["heading_error_deg"]) <= HEADING_TOL_DEG
        )

    if not handoff_pass:
        env2.fdm = None
        env1.close()
        if require_full_hold:
            raise RuntimeError(
                "Locked Stage-3 final PPO did not reproduce the 5 s endpoint hover. "
                "Do not run Stage-4 identification."
            )
        raise RuntimeError(
            "Locked Stage-3 PPO never reached the requested Stage-4 transition state. "
            "Do not run Stage-4 entry-margin identification."
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
# STAGE-4 DESCENT TEACHER
# =====================================================================

def desired_vertical_speed(altitude_ft: float, descent_vmax: float) -> float:
    """Smooth altitude-to-descent-rate schedule, tapered to zero at 30 ft."""
    raw = ALT_TO_VS_GAIN * (DESCENT_TARGET_ALT_FT - float(altitude_ft))
    return float(np.clip(raw, -float(descent_vmax), +MAX_UPWARD_RECOVERY_VS))


def physical_elevator_to_stage4_action1(physical_cmd: float) -> float:
    """Map measured physical Stage-4 elevator band to normalized [-1,+1]."""
    p = float(np.clip(physical_cmd, ELEVATOR_PHYSICAL_MIN, ELEVATOR_PHYSICAL_MAX))
    a = 2.0 * (p - ELEVATOR_PHYSICAL_MIN) / ELEVATOR_PHYSICAL_SPAN - 1.0
    return float(np.clip(a, -1.0, +1.0))


def stage4_action1_to_physical(action1: float) -> float:
    a = float(np.clip(action1, -1.0, +1.0))
    return float(ELEVATOR_PHYSICAL_MIN + 0.5 * (a + 1.0) * ELEVATOR_PHYSICAL_SPAN)


def build_stage4_teacher_action(
    state, hover_action0: float, descent_vmax: float, vs_kp: float, vs_ki: float,
    vertical_bias: float, low_alt_bias_floor: float, dt: float,
    long_kpos_phys: float, long_kv_phys: float,
):
    vs_des = desired_vertical_speed(state["altitude_ft"], descent_vmax)
    vs_error = vs_des - state["vertical_speed_fps"]
    candidate_bias = float(np.clip(vertical_bias + vs_ki * vs_error * dt, VERTICAL_BIAS_MIN, VERTICAL_BIAS_MAX))
    capture_blend = float(np.clip((state["altitude_ft"] - DESCENT_TARGET_ALT_FT) / max(1e-6, CAPTURE_BLEND_START_ALT_FT - DESCENT_TARGET_ALT_FT), 0.0, 1.0))
    # V1 faded the integral contribution all the way to zero at 30 ft.
    # That structurally forced a low-altitude steady-state equilibrium near 35-40 ft.
    # Here the high-altitude adaptive bias is blended toward a tested candidate
    # low-altitude feedforward floor instead of toward zero.
    effective_bias = float(low_alt_bias_floor + (candidate_bias - low_alt_bias_floor) * capture_blend)
    p_term = float(vs_kp) * vs_error
    residual_unclipped = p_term + effective_bias
    collective_residual = float(np.clip(residual_unclipped, COLLECTIVE_RESIDUAL_MIN, COLLECTIVE_RESIDUAL_MAX))
    pushing_low = collective_residual <= COLLECTIVE_RESIDUAL_MIN + 1e-9 and vs_error < 0.0
    pushing_high = collective_residual >= COLLECTIVE_RESIDUAL_MAX - 1e-9 and vs_error > 0.0
    if pushing_low or pushing_high:
        candidate_bias = float(vertical_bias)
        effective_bias = float(low_alt_bias_floor + (candidate_bias - low_alt_bias_floor) * capture_blend)
        residual_unclipped = p_term + effective_bias
        collective_residual = float(np.clip(residual_unclipped, COLLECTIVE_RESIDUAL_MIN, COLLECTIVE_RESIDUAL_MAX))
    action0 = float(np.clip(hover_action0 + collective_residual, -1.0, +1.0))
    ahead_ft = float(state["forward_ft"] - TARGET_FORWARD_FT)
    brake_demand = long_kpos_phys * ahead_ft + long_kv_phys * float(state["forward_speed_fps"])
    elevator_delta_phys = float(np.clip(-brake_demand, -0.003, 0.0))
    desired_physical_elevator = float(np.clip(ELEVATOR_PHYSICAL_MAX + elevator_delta_phys, ELEVATOR_PHYSICAL_MIN, ELEVATOR_PHYSICAL_MAX))
    action1 = physical_elevator_to_stage4_action1(desired_physical_elevator)
    lateral_corr = -LOCKED_LATERAL_KP * state["cross_track_ft"] - LOCKED_LATERAL_KD * state["lateral_speed_fps"]
    raw_a2 = LATERAL_TRIM_ACTION + lateral_corr
    action2 = float(np.clip(raw_a2, IDENTIFIED_A2_MIN, DESCENT_TESTED_A2_MAX))
    action3 = float(RUDDER_ACTION)
    action = np.asarray([action0, action1, action2, action3], dtype=np.float32)
    return action, {
        "vs_des_fps": float(vs_des), "vs_error_fps": float(vs_error), "vs_p_term": float(p_term),
        "vertical_bias_state": float(candidate_bias), "vertical_bias_effective": float(effective_bias),
        "low_alt_bias_floor": float(low_alt_bias_floor), "capture_blend": float(capture_blend), "collective_residual_unclipped": float(residual_unclipped),
        "collective_residual": float(collective_residual), "ahead_ft": float(ahead_ft),
        "brake_demand_phys": float(brake_demand), "elevator_delta_phys": float(elevator_delta_phys),
        "desired_physical_elevator": float(desired_physical_elevator), "normalized_action1": float(action1),
        "lateral_corr": float(lateral_corr), "raw_action2": float(raw_a2),
    }, float(candidate_bias)

def stage4_settle_now(state) -> bool:
    return bool(
        abs(state["altitude_ft"] - DESCENT_TARGET_ALT_FT) <= DESCENT_ALT_TOL_FT
        and abs(state["vertical_speed_fps"]) <= DESCENT_VS_TOL_FPS
        and abs(state["position_error_ft"]) <= STAGE4_POS_TOL_FT
        and abs(state["forward_speed_fps"]) <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(state["cross_track_ft"]) <= STAGE4_CROSS_TOL_FT
        and abs(state["lateral_speed_fps"]) <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(state["heading_error_deg"]) <= STAGE4_HEADING_TOL_DEG
    )


def stage4_teacher_safety_reason(state) -> str:
    if state["altitude_ft"] < ID_ALT_SAFE_MIN:
        return "altitude_below_stage4_safe"
    if state["altitude_ft"] > ID_ALT_SAFE_MAX:
        return "altitude_above_stage4_safe"
    if state["vertical_speed_fps"] < STAGE4_MIN_VERTICAL_SPEED_FPS:
        return "descent_rate_below_stage4_safe"
    if state["vertical_speed_fps"] > STAGE4_MAX_VERTICAL_SPEED_FPS:
        return "climb_rate_above_stage4_safe"
    if abs(math.degrees(state["pitch_rad"])) > ID_MAX_ABS_PITCH_DEG:
        return "pitch_stage4_limit"
    if abs(math.degrees(state["roll_rad"])) > ID_MAX_ABS_ROLL_DEG:
        return "roll_stage4_limit"
    if abs(state["cross_track_ft"]) > ID_MAX_ABS_CROSS_FT:
        return "cross_stage4_limit"
    if abs(state["position_error_ft"]) > ID_MAX_ABS_POSITION_ERROR_FT:
        return "forward_position_stage4_limit"
    if abs(state["heading_error_deg"]) > STAGE4_MAX_ABS_HEADING_DEG:
        return "heading_stage4_limit"
    return ""


def run_teacher_case(
    low_alt_bias_floor: float,
    candidate_index: int,
    detailed: bool = False,
):
    vs_ki = float(LOCKED_VS_KI)
    descent_vmax = float(LOCKED_DESCENT_VMAX)
    vs_kp = float(LOCKED_VS_KP)
    long_kpos_phys = float(LOCKED_LONG_KPOS_PHYS)
    long_kv_phys = float(LOCKED_LONG_KV_PHYS)
    # Integrated Stage-4 entry uses the V3-qualified earlier braking transfer.
    start = build_stage4_handoff(
        detailed=False,
        require_full_hold=False,
        custom_entry_forward_ft=STAGE4_ENTRY_FORWARD_FT,
        custom_entry_max_speed_fps=STAGE4_ENTRY_MAX_SPEED_FPS,
    )
    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = env_control_dt(env2)

    # -----------------------------------------------------------------
    # PRE-DESCENT HOLD / TRANSITION
    # -----------------------------------------------------------------
    # Keep the locked Stage-3 PPO on the remaining channels and force only the
    # V3-qualified longitudinal actuator-bound braking action.  Descent is
    # forbidden until the ORIGINAL full endpoint envelope has been held for
    # >=5 s continuously.
    pre_hold = 0.0
    pre_hold_max = 0.0
    endpoint_entered = False
    endpoint_entry_time = float("nan")
    transition_trace = []
    transition_reason = "pre_descent_hold_timeout"
    transition_safe = True
    max_abs_pos_after_entry = 0.0
    max_abs_cross_after_entry = 0.0

    PRE_HOLD_MAX_TIME = 40.0
    for step in range(int(PRE_HOLD_MAX_TIME / dt)):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        obs3 = stage3_observation(state_before)
        base, _ = stage3_model.predict(obs3, deterministic=True)
        action = np.asarray(base, dtype=np.float32).reshape(-1).copy()
        action[1] = FIXED_BRAKE_A1

        state, used = raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading)
        t = (step + 1) * dt

        in_endpoint = endpoint_hover_now(state)
        if in_endpoint and not endpoint_entered:
            endpoint_entered = True
            endpoint_entry_time = float(t)
        if endpoint_entered:
            max_abs_pos_after_entry = max(
                max_abs_pos_after_entry, abs(state["position_error_ft"])
            )
            max_abs_cross_after_entry = max(
                max_abs_cross_after_entry, abs(state["cross_track_ft"])
            )

        pre_hold = pre_hold + dt if in_endpoint else 0.0
        pre_hold_max = max(pre_hold_max, pre_hold)

        transition_trace.append({
            "candidate_index": int(candidate_index),
            "phase": "pre_descent_hold",
            "time_s": float(t),
            "descent_vmax_fps": float(descent_vmax),
            "vs_kp": float(vs_kp),
            "vs_ki": float(vs_ki),
        "low_alt_bias_floor": float(low_alt_bias_floor),
            "low_alt_bias_floor": float(low_alt_bias_floor),
            "long_kpos_phys": float(long_kpos_phys),
            "long_kv_phys": float(long_kv_phys),
            "action0": float(used[0]),
            "action1": float(used[1]),
            "action2": float(used[2]),
            "action3": float(used[3]),
            **{k: float(v) for k, v in state.items()},
            "pre_descent_hold_s": float(pre_hold),
            "settle_hold_s": 0.0,
        })

        reason = stage3_safety_reason(state)
        if reason:
            transition_safe = False
            transition_reason = f"pre_descent_{reason}"
            break

        if endpoint_entered:
            if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
                transition_safe = False
                transition_reason = "pre_descent_left_endpoint_position_corridor"
                break
            if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
                transition_safe = False
                transition_reason = "pre_descent_left_endpoint_cross_corridor"
                break

        if pre_hold >= PRE_DESCENT_HOLD_SECONDS:
            transition_reason = "pre_descent_hold_qualified"
            break

    if (not transition_safe) or pre_hold < PRE_DESCENT_HOLD_SECONDS:
        final_state = snapshot(fdm, lat0, lon0, mission_heading)
        result = {
            "candidate_index": int(candidate_index),
            "descent_vmax_fps": float(descent_vmax),
            "vs_kp": float(vs_kp),
            "long_kpos_phys": float(long_kpos_phys),
            "long_kv_phys": float(long_kv_phys),
            "safe": False,
            "termination_reason": str(transition_reason),
            "presentation_pass": False,
            "final_settle_pass": False,
            "teacher_pass": False,
            "same_fdm": bool(start["same_fdm"]),
            "clock_reset_on_attach": bool(start["clock_reset_on_attach"]),
            "stage4_entry_forward_ft": float(start["state"]["forward_ft"]),
            "stage4_entry_forward_speed_fps": float(start["state"]["forward_speed_fps"]),
            "endpoint_entry_time_s": float(endpoint_entry_time),
            "pre_descent_hold_s": float(pre_hold),
            "pre_descent_max_hold_s": float(pre_hold_max),
            "max_abs_position_error_prehold_after_entry_ft": float(max_abs_pos_after_entry),
            "max_abs_cross_prehold_after_entry_ft": float(max_abs_cross_after_entry),
            "duration_s": float(len(transition_trace) * dt),
            "final_altitude_ft": float(final_state["altitude_ft"]),
            "final_vertical_speed_fps": float(final_state["vertical_speed_fps"]),
            "final_forward_ft": float(final_state["forward_ft"]),
            "final_position_error_ft": float(final_state["position_error_ft"]),
            "final_forward_speed_fps": float(final_state["forward_speed_fps"]),
            "final_cross_track_ft": float(final_state["cross_track_ft"]),
            "final_lateral_speed_fps": float(final_state["lateral_speed_fps"]),
            "final_heading_error_deg": float(final_state["heading_error_deg"]),
            "settle_hold_s": 0.0,
            "max_settle_hold_s": 0.0,
            "max_abs_cross_track_ft": float(max_abs_cross_after_entry),
            "max_abs_position_error_ft": float(max_abs_pos_after_entry),
            "min_vertical_speed_fps": float(final_state["vertical_speed_fps"]),
            "collective_residual_min_seen": 0.0,
            "collective_residual_max_seen": 0.0,
            "collective_residual_bound_fraction": 0.0,
        }
        close_handoff(start)
        return result, transition_trace

    # Freeze collective hover reference only AFTER the qualified Stage-4
    # pre-descent endpoint hold.  This avoids using an OOD Stage-3 altitude
    # policy throughout the 270 ft descent while retaining a physically
    # observed hover reference from the same continuous FDM.
    pre_descent_state = snapshot(fdm, lat0, lon0, mission_heading)
    obs3 = stage3_observation(pre_descent_state)
    hover_action, _ = stage3_model.predict(obs3, deterministic=True)
    hover_action = np.asarray(hover_action, dtype=np.float32).reshape(-1)
    hover_action0 = float(hover_action[0])

    initial_state = pre_descent_state.copy()
    trace = list(transition_trace)
    settle_hold = 0.0
    max_settle_hold = 0.0
    safe = True
    termination_reason = "stage4_time_limit"
    vertical_bias = 0.0
    vertical_bias_min_seen = 0.0
    vertical_bias_max_seen = 0.0

    min_alt = initial_state["altitude_ft"]
    max_alt = initial_state["altitude_ft"]
    min_vs = initial_state["vertical_speed_fps"]
    max_vs = initial_state["vertical_speed_fps"]
    max_abs_cross = abs(initial_state["cross_track_ft"])
    max_abs_pos_err = abs(initial_state["position_error_ft"])
    max_abs_pitch = abs(math.degrees(initial_state["pitch_rad"]))
    max_abs_roll = abs(math.degrees(initial_state["roll_rad"]))
    max_abs_heading = abs(initial_state["heading_error_deg"])
    min_a0 = +999.0
    max_a0 = -999.0
    min_a1 = +999.0
    max_a1 = -999.0
    min_a2 = +999.0
    max_a2 = -999.0
    min_a3 = +999.0
    max_a3 = -999.0
    residual_min_seen = +999.0
    residual_max_seen = -999.0
    residual_bound_hits = 0
    a2_upper_bound_hits = 0
    elevator_min_seen = +999.0
    elevator_max_seen = -999.0
    elevator_delta_min_seen = +999.0
    elevator_delta_max_seen = -999.0
    elevator_min_bound_hits = 0

    crossed = {200.0: False, 100.0: False, 50.0: False, 40.0: False, 35.0: False, 32.0: False}
    next_print = 0.0

    total_steps = int(DESCENT_MAX_TIME / dt)
    for step in range(total_steps):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        action, ctrl, vertical_bias = build_stage4_teacher_action(
            state_before, hover_action0=hover_action0, descent_vmax=descent_vmax,
            vs_kp=vs_kp, vs_ki=vs_ki, vertical_bias=vertical_bias,
            low_alt_bias_floor=low_alt_bias_floor, dt=dt,
            long_kpos_phys=long_kpos_phys, long_kv_phys=long_kv_phys,
        )
        vertical_bias_min_seen = min(vertical_bias_min_seen, vertical_bias)
        vertical_bias_max_seen = max(vertical_bias_max_seen, vertical_bias)

        if (
            abs(ctrl["collective_residual"] - COLLECTIVE_RESIDUAL_MIN) < 1e-8
            or abs(ctrl["collective_residual"] - COLLECTIVE_RESIDUAL_MAX) < 1e-8
        ):
            residual_bound_hits += 1
        if float(action[2]) >= DESCENT_TESTED_A2_MAX - 1e-6:
            a2_upper_bound_hits += 1

        state, used_action, base_elevator_cmd, mapped_elevator_cmd = raw_stage4_mapped_cycle(
            env2, fdm, action, lat0, lon0, mission_heading
        )
        t = (step + 1) * dt

        min_alt = min(min_alt, state["altitude_ft"])
        max_alt = max(max_alt, state["altitude_ft"])
        min_vs = min(min_vs, state["vertical_speed_fps"])
        max_vs = max(max_vs, state["vertical_speed_fps"])
        max_abs_cross = max(max_abs_cross, abs(state["cross_track_ft"]))
        max_abs_pos_err = max(max_abs_pos_err, abs(state["position_error_ft"]))
        max_abs_pitch = max(max_abs_pitch, abs(math.degrees(state["pitch_rad"])))
        max_abs_roll = max(max_abs_roll, abs(math.degrees(state["roll_rad"])))
        max_abs_heading = max(max_abs_heading, abs(state["heading_error_deg"]))

        min_a0, max_a0 = min(min_a0, float(used_action[0])), max(max_a0, float(used_action[0]))
        min_a1, max_a1 = min(min_a1, float(used_action[1])), max(max_a1, float(used_action[1]))
        min_a2, max_a2 = min(min_a2, float(used_action[2])), max(max_a2, float(used_action[2]))
        min_a3, max_a3 = min(min_a3, float(used_action[3])), max(max_a3, float(used_action[3]))
        residual_min_seen = min(residual_min_seen, ctrl["collective_residual"])
        residual_max_seen = max(residual_max_seen, ctrl["collective_residual"])
        elevator_min_seen = min(elevator_min_seen, mapped_elevator_cmd)
        elevator_max_seen = max(elevator_max_seen, mapped_elevator_cmd)
        elevator_delta_min_seen = min(elevator_delta_min_seen, ctrl["elevator_delta_phys"])
        elevator_delta_max_seen = max(elevator_delta_max_seen, ctrl["elevator_delta_phys"])
        if mapped_elevator_cmd <= ELEVATOR_PHYSICAL_MIN + 1e-8:
            elevator_min_bound_hits += 1

        settle_hold = settle_hold + dt if stage4_settle_now(state) else 0.0
        max_settle_hold = max(max_settle_hold, settle_hold)

        trace.append({
            "candidate_index": int(candidate_index),
            "phase": "descent",
            "time_s": float(t),
            "descent_vmax_fps": float(descent_vmax),
            "vs_kp": float(vs_kp),
            "long_kpos_phys": float(long_kpos_phys),
            "long_kv_phys": float(long_kv_phys),
            "hover_action0": float(hover_action0),
            "action0": float(used_action[0]),
            "action1": float(used_action[1]),
            "action2": float(used_action[2]),
            "action3": float(used_action[3]),
            "base_elevator_cmd_before_stage4_mapping": float(base_elevator_cmd),
            "mapped_physical_elevator_cmd": float(mapped_elevator_cmd),
            **{k: float(v) for k, v in ctrl.items()},
            **{k: float(v) for k, v in state.items()},
            "pre_descent_hold_s": float(pre_hold),
            "settle_hold_s": float(settle_hold),
        })

        reason = stage4_teacher_safety_reason(state)
        if reason:
            safe = False
            termination_reason = reason
            break

        # Presentation corridor is a hard requirement for the complete descent.
        if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
            safe = False
            termination_reason = "forward_position_stage4_limit"
            break
        if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
            safe = False
            termination_reason = "cross_track_stage4_limit"
            break

        for level in list(crossed.keys()):
            if (not crossed[level]) and state["altitude_ft"] <= level:
                crossed[level] = True
                if detailed:
                    print(
                        f"  ALT<{level:5.0f} | t={t:7.2f}s | "
                        f"ALT={state['altitude_ft']:7.2f} VS={state['vertical_speed_fps']:+6.3f} | "
                        f"FWD={state['forward_ft']:7.2f} V={state['forward_speed_fps']:+6.3f} | "
                        f"X={state['cross_track_ft']:+6.2f} LAT={state['lateral_speed_fps']:+6.3f} | "
                        f"A0={used_action[0]:+7.4f} dA0={ctrl['collective_residual']:+6.3f} VB={ctrl['vertical_bias_effective']:+6.3f} | "
                        f"A1={used_action[1]:+6.3f} ELEV={mapped_elevator_cmd:+.6f}"
                    )

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  t={t:7.2f}s | ALT={state['altitude_ft']:7.2f} | "
                f"VS={state['vertical_speed_fps']:+6.3f}/{ctrl['vs_des_fps']:+5.2f} | "
                f"FWD={state['forward_ft']:7.2f} err={state['position_error_ft']:+6.2f} "
                f"V={state['forward_speed_fps']:+6.3f} | "
                f"X={state['cross_track_ft']:+6.2f} LAT={state['lateral_speed_fps']:+6.3f} | "
                f"HDG={state['heading_error_deg']:+5.2f} | "
                f"VB={ctrl['vertical_bias_effective']:+6.3f} dA0={ctrl['collective_residual']:+6.3f} | A1={used_action[1]:+6.3f} ELEV={mapped_elevator_cmd:+.6f} | HOLD={settle_hold:4.2f}s"
            )
            next_print += 20.0

        if settle_hold >= DESCENT_SETTLE_SECONDS:
            termination_reason = "stable_30ft_hover_5s"
            break

    final_state = snapshot(fdm, lat0, lon0, mission_heading)
    duration_s = float(len(trace) * dt)
    presentation_pass = bool(
        max_abs_cross <= PRESENTATION_MAX_CROSS_FT
        and max_abs_pos_err <= PRESENTATION_MAX_POSITION_ERROR_FT
    )
    final_settle_pass = bool(
        settle_hold >= DESCENT_SETTLE_SECONDS and stage4_settle_now(final_state)
    )
    teacher_pass = bool(safe and presentation_pass and final_settle_pass)

    result = {
        "candidate_index": int(candidate_index),
        "descent_vmax_fps": float(descent_vmax),
        "vs_kp": float(vs_kp),
        "vs_ki": float(vs_ki),
        "low_alt_bias_floor": float(low_alt_bias_floor),
        "long_kpos_phys": float(long_kpos_phys),
        "long_kv_phys": float(long_kv_phys),
        "safe": bool(safe),
        "termination_reason": str(termination_reason),
        "presentation_pass": bool(presentation_pass),
        "final_settle_pass": bool(final_settle_pass),
        "teacher_pass": bool(teacher_pass),
        "same_fdm": bool(start["same_fdm"]),
        "clock_reset_on_attach": bool(start["clock_reset_on_attach"]),
        "stage4_entry_trigger_forward_ft": float(STAGE4_ENTRY_FORWARD_FT),
        "stage4_entry_forward_ft": float(start["state"]["forward_ft"]),
        "stage4_entry_forward_speed_fps": float(start["state"]["forward_speed_fps"]),
        "endpoint_entry_time_s": float(endpoint_entry_time),
        "pre_descent_hold_s": float(pre_hold),
        "pre_descent_max_hold_s": float(pre_hold_max),
        "max_abs_position_error_prehold_after_entry_ft": float(max_abs_pos_after_entry),
        "max_abs_cross_prehold_after_entry_ft": float(max_abs_cross_after_entry),
        "hover_action0": float(hover_action0),
        "duration_s": float(duration_s),
        "settle_hold_s": float(settle_hold),
        "max_settle_hold_s": float(max_settle_hold),
        "initial_altitude_ft": float(initial_state["altitude_ft"]),
        "initial_forward_ft": float(initial_state["forward_ft"]),
        "initial_cross_track_ft": float(initial_state["cross_track_ft"]),
        "final_altitude_ft": float(final_state["altitude_ft"]),
        "final_vertical_speed_fps": float(final_state["vertical_speed_fps"]),
        "final_forward_ft": float(final_state["forward_ft"]),
        "final_position_error_ft": float(final_state["position_error_ft"]),
        "final_forward_speed_fps": float(final_state["forward_speed_fps"]),
        "final_cross_track_ft": float(final_state["cross_track_ft"]),
        "final_lateral_speed_fps": float(final_state["lateral_speed_fps"]),
        "final_heading_error_deg": float(final_state["heading_error_deg"]),
        "min_altitude_ft": float(min_alt),
        "max_altitude_ft": float(max_alt),
        "min_vertical_speed_fps": float(min_vs),
        "max_vertical_speed_fps": float(max_vs),
        "max_abs_cross_track_ft": float(max_abs_cross),
        "max_abs_position_error_ft": float(max_abs_pos_err),
        "max_abs_pitch_deg": float(max_abs_pitch),
        "max_abs_roll_deg": float(max_abs_roll),
        "max_abs_heading_error_deg": float(max_abs_heading),
        "vertical_bias_min_seen": float(vertical_bias_min_seen),
        "vertical_bias_max_seen": float(vertical_bias_max_seen),
        "action0_min": float(min_a0),
        "action0_max": float(max_a0),
        "action1_min": float(min_a1),
        "action1_max": float(max_a1),
        "action2_min": float(min_a2),
        "action2_max": float(max_a2),
        "action3_min": float(min_a3),
        "action3_max": float(max_a3),
        "collective_residual_min_seen": float(residual_min_seen),
        "collective_residual_max_seen": float(residual_max_seen),
        "collective_residual_bound_fraction": float(residual_bound_hits / max(1, len(trace))),
        "action2_upper_bound_fraction": float(a2_upper_bound_hits / max(1, len(trace) - len(transition_trace))),
        "physical_elevator_min": float(elevator_min_seen),
        "physical_elevator_max": float(elevator_max_seen),
        "physical_elevator_delta_min": float(elevator_delta_min_seen),
        "physical_elevator_delta_max": float(elevator_delta_max_seen),
        "physical_elevator_min_bound_fraction": float(elevator_min_bound_hits / max(1, len(trace) - len(transition_trace))),
    }

    close_handoff(start)
    return result, trace


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def candidate_score(row):
    return (
        0 if row["teacher_pass"] else 1,
        0 if row["safe"] else 1,
        row["max_abs_position_error_ft"],
        row["max_abs_cross_track_ft"],
        row.get("collective_residual_bound_fraction", 1.0),
        abs(row.get("vertical_bias_min_seen", 0.0)),
        row.get("physical_elevator_min_bound_fraction", 1.0),
        abs(row["final_altitude_ft"] - DESCENT_TARGET_ALT_FT),
        abs(row["final_vertical_speed_fps"]),
        row["duration_s"],
    )


# =====================================================================
# MAIN
# =====================================================================


# =====================================================================
# STAGE-4 NEAR-GROUND DIAGNOSTIC V1
# =====================================================================

QUALIFIED_LOW_ALT_BIAS_FLOOR = -0.240
NEAR_GROUND_TARGETS_FT = [25.0, 20.0, 15.0, 12.0, 10.0, 9.0]
NEAR_GROUND_VMAX_FPS = 0.45
NEAR_GROUND_ALT_TO_VS_GAIN = 0.08
NEAR_GROUND_SETTLE_ALT_TOL_FT = 1.0
NEAR_GROUND_SETTLE_VS_TOL_FPS = 0.20
NEAR_GROUND_SETTLE_SECONDS = 3.0
NEAR_GROUND_LEVEL_MAX_TIME = 100.0
NEAR_GROUND_MIN_SAFE_AGL_FT = 7.2
NEAR_GROUND_MAX_DESCENT_RATE_FPS = -1.0

CONTACT_KEYS = [
    "gear/unit[0]/WOW",
    "gear/unit[1]/WOW",
    "gear/unit[2]/WOW",
    "gear/unit[0]/wow",
    "gear/unit[1]/wow",
    "gear/unit[2]/wow",
    "gear/unit[0]/compression-ft",
    "gear/unit[1]/compression-ft",
    "gear/unit[2]/compression-ft",
]


def contact_snapshot(fdm):
    out = {}
    for key in CONTACT_KEYS:
        value = fdm_float(fdm, key)
        if np.isfinite(value):
            out[key] = float(value)
    return out


def any_wow(contact):
    vals = []
    for key, value in contact.items():
        if key.lower().endswith("/wow"):
            vals.append(float(value))
    return bool(vals and max(vals) > 0.5)


def build_qualified_30ft_open(detailed=False):
    start = build_stage4_handoff(
        detailed=False,
        require_full_hold=False,
        custom_entry_forward_ft=STAGE4_ENTRY_FORWARD_FT,
        custom_entry_max_speed_fps=STAGE4_ENTRY_MAX_SPEED_FPS,
    )
    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = env_control_dt(env2)

    pre_hold = 0.0
    endpoint_entered = False
    for _ in range(int(40.0 / dt)):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        obs3 = stage3_observation(state_before)
        base, _ = stage3_model.predict(obs3, deterministic=True)
        action = np.asarray(base, dtype=np.float32).reshape(-1).copy()
        action[1] = FIXED_BRAKE_A1
        state, _ = raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading)

        in_endpoint = endpoint_hover_now(state)
        endpoint_entered = endpoint_entered or in_endpoint
        pre_hold = pre_hold + dt if in_endpoint else 0.0

        reason = stage3_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(f"Pre-descent Stage-4 hold safety failure: {reason}")
        if endpoint_entered:
            if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
                close_handoff(start)
                raise RuntimeError("Pre-descent hold left forward corridor.")
            if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
                close_handoff(start)
                raise RuntimeError("Pre-descent hold left cross corridor.")

        if pre_hold >= PRE_DESCENT_HOLD_SECONDS:
            break

    if pre_hold < PRE_DESCENT_HOLD_SECONDS:
        close_handoff(start)
        raise RuntimeError("Qualified Stage-4 pre-descent hold was not reproduced.")

    pre_state = snapshot(fdm, lat0, lon0, mission_heading)
    obs3 = stage3_observation(pre_state)
    hover_action, _ = stage3_model.predict(obs3, deterministic=True)
    hover_action = np.asarray(hover_action, dtype=np.float32).reshape(-1)
    hover_action0 = float(hover_action[0])

    vertical_bias = 0.0
    settle_hold = 0.0
    descent_trace = []
    for step in range(int(DESCENT_MAX_TIME / dt)):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        action, ctrl, vertical_bias = build_stage4_teacher_action(
            state_before,
            hover_action0=hover_action0,
            descent_vmax=LOCKED_DESCENT_VMAX,
            vs_kp=LOCKED_VS_KP,
            vs_ki=LOCKED_VS_KI,
            vertical_bias=vertical_bias,
            low_alt_bias_floor=QUALIFIED_LOW_ALT_BIAS_FLOOR,
            dt=dt,
            long_kpos_phys=LOCKED_LONG_KPOS_PHYS,
            long_kv_phys=LOCKED_LONG_KV_PHYS,
        )
        state, used, base_elev, mapped_elev = raw_stage4_mapped_cycle(
            env2, fdm, action, lat0, lon0, mission_heading
        )
        t = (step + 1) * dt

        settle_hold = settle_hold + dt if stage4_settle_now(state) else 0.0

        descent_trace.append({
            "phase": "qualified_300_to_30",
            "time_s": float(t),
            "action0": float(used[0]),
            "action1": float(used[1]),
            "action2": float(used[2]),
            "action3": float(used[3]),
            "base_elevator_cmd_before_stage4_mapping": float(base_elev),
            "mapped_physical_elevator_cmd": float(mapped_elev),
            **{k: float(v) for k, v in ctrl.items()},
            **{k: float(v) for k, v in state.items()},
        })

        reason = stage4_teacher_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(f"Qualified 300->30 teacher failed before near-ground ID: {reason}")
        if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
            close_handoff(start)
            raise RuntimeError("Qualified 300->30 teacher left forward corridor.")
        if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
            close_handoff(start)
            raise RuntimeError("Qualified 300->30 teacher left cross corridor.")

        if detailed and (step == 0 or abs((t % 30.0)) < dt):
            print(
                f"  300->30 t={t:7.2f}s ALT={state['altitude_ft']:6.2f} "
                f"VS={state['vertical_speed_fps']:+6.3f} FWD={state['forward_ft']:7.2f} "
                f"X={state['cross_track_ft']:+6.2f} dA0={ctrl['collective_residual']:+6.3f}"
            )

        if settle_hold >= DESCENT_SETTLE_SECONDS:
            break

    final30 = snapshot(fdm, lat0, lon0, mission_heading)
    if not (settle_hold >= DESCENT_SETTLE_SECONDS and stage4_settle_now(final30)):
        close_handoff(start)
        raise RuntimeError("Selected 300->30 teacher did not reproduce stable 30-ft qualification.")

    return {
        **start,
        "dt": float(dt),
        "hover_action0": float(hover_action0),
        "vertical_bias": float(vertical_bias),
        "state30": final30.copy(),
        "settle_hold_30_s": float(settle_hold),
        "descent_trace": descent_trace,
    }


def build_near_ground_action(state, target_alt_ft, hover_action0, vertical_bias, dt):
    vs_des = NEAR_GROUND_ALT_TO_VS_GAIN * (float(target_alt_ft) - state["altitude_ft"])
    vs_des = float(np.clip(vs_des, -NEAR_GROUND_VMAX_FPS, +0.20))
    vs_error = vs_des - state["vertical_speed_fps"]

    candidate_bias = float(np.clip(
        vertical_bias + LOCKED_VS_KI * vs_error * dt,
        VERTICAL_BIAS_MIN,
        VERTICAL_BIAS_MAX,
    ))
    effective_bias = float(candidate_bias)
    p_term = float(LOCKED_VS_KP * vs_error)
    residual_unclipped = p_term + effective_bias
    collective_residual = float(np.clip(
        residual_unclipped,
        COLLECTIVE_RESIDUAL_MIN,
        COLLECTIVE_RESIDUAL_MAX,
    ))

    if (
        collective_residual <= COLLECTIVE_RESIDUAL_MIN + 1e-9 and vs_error < 0.0
    ) or (
        collective_residual >= COLLECTIVE_RESIDUAL_MAX - 1e-9 and vs_error > 0.0
    ):
        candidate_bias = float(vertical_bias)
        effective_bias = float(candidate_bias)
        residual_unclipped = p_term + effective_bias
        collective_residual = float(np.clip(
            residual_unclipped,
            COLLECTIVE_RESIDUAL_MIN,
            COLLECTIVE_RESIDUAL_MAX,
        ))

    action0 = float(np.clip(hover_action0 + collective_residual, -1.0, +1.0))

    ahead_ft = float(state["forward_ft"] - TARGET_FORWARD_FT)
    brake_demand = (
        LOCKED_LONG_KPOS_PHYS * ahead_ft
        + LOCKED_LONG_KV_PHYS * float(state["forward_speed_fps"])
    )
    elevator_delta_phys = float(np.clip(-brake_demand, -0.003, 0.0))
    desired_physical_elevator = float(np.clip(
        ELEVATOR_PHYSICAL_MAX + elevator_delta_phys,
        ELEVATOR_PHYSICAL_MIN,
        ELEVATOR_PHYSICAL_MAX,
    ))
    action1 = physical_elevator_to_stage4_action1(desired_physical_elevator)

    lateral_corr = (
        -LOCKED_LATERAL_KP * state["cross_track_ft"]
        -LOCKED_LATERAL_KD * state["lateral_speed_fps"]
    )
    action2 = float(np.clip(
        LATERAL_TRIM_ACTION + lateral_corr,
        IDENTIFIED_A2_MIN,
        DESCENT_TESTED_A2_MAX,
    ))
    action3 = float(RUDDER_ACTION)

    return (
        np.asarray([action0, action1, action2, action3], dtype=np.float32),
        {
            "target_alt_ft": float(target_alt_ft),
            "vs_des_fps": float(vs_des),
            "vs_error_fps": float(vs_error),
            "vs_p_term": float(p_term),
            "vertical_bias_state": float(candidate_bias),
            "vertical_bias_effective": float(effective_bias),
            "collective_residual_unclipped": float(residual_unclipped),
            "collective_residual": float(collective_residual),
            "desired_physical_elevator": float(desired_physical_elevator),
            "normalized_action1": float(action1),
            "raw_action2": float(LATERAL_TRIM_ACTION + lateral_corr),
        },
        float(candidate_bias),
    )


def near_ground_safety_reason(state):
    if state["altitude_ft"] < NEAR_GROUND_MIN_SAFE_AGL_FT:
        return "agl_below_identification_floor"
    if state["vertical_speed_fps"] < NEAR_GROUND_MAX_DESCENT_RATE_FPS:
        return "near_ground_descent_rate_too_high"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "near_ground_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "near_ground_cross_corridor_exit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "near_ground_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "near_ground_roll_limit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "near_ground_heading_limit"
    return ""


def level_settle_now(state, target_alt_ft):
    return bool(
        abs(state["altitude_ft"] - float(target_alt_ft)) <= NEAR_GROUND_SETTLE_ALT_TOL_FT
        and abs(state["vertical_speed_fps"]) <= NEAR_GROUND_SETTLE_VS_TOL_FPS
        and abs(state["position_error_ft"]) <= STAGE4_POS_TOL_FT
        and abs(state["forward_speed_fps"]) <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(state["cross_track_ft"]) <= STAGE4_CROSS_TOL_FT
        and abs(state["lateral_speed_fps"]) <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(state["heading_error_deg"]) <= STAGE4_HEADING_TOL_DEG
    )



# =====================================================================
# STAGE-4 NEAR-GROUND LONGITUDINAL / ELEVATOR DIAGNOSTIC V1
# =====================================================================

TRIGGER_FORWARD_FT = 298.0
PULSE_SECONDS = 2.0
PULSE_DELTAS_PHYS = [0.000, 0.002, 0.004, 0.006, 0.008, 0.010]

SUSTAINED_SECONDS = 12.0
SUSTAINED_DELTAS_PHYS = [0.002, 0.004, 0.006, 0.008]

TARGET_25_FT = 25.0
TARGET_20_FT = 20.0


def raw_stage4_elevator_offset_cycle(
    env2, fdm, action, elevator_delta_phys, lat0, lon0, mission_heading
):
    """
    Identification-only cycle.
    The normalized Stage-4 action is unchanged; after the verified Stage-4
    action1 mapping, apply a small additional physical elevator offset before
    physics. This is NOT a final runtime mapping/controller.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -1.0, +1.0).astype(np.float32)

    env2._apply_action(action)
    base_elevator = fdm_float(fdm, "fcs/elevator-cmd-norm")

    mapped_elevator = stage4_action1_to_physical(float(action[1]))
    diagnostic_elevator = float(np.clip(
        mapped_elevator + float(elevator_delta_phys),
        -1.0,
        +1.0,
    ))
    fdm["fcs/elevator-cmd-norm"] = diagnostic_elevator

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError("JSBSim stopped during near-ground elevator diagnostic.")

    if hasattr(env2, "previous_action"):
        try:
            env2.previous_action = action.copy()
        except Exception:
            pass

    state = snapshot(fdm, lat0, lon0, mission_heading)
    if hasattr(env2, "forward_distance"):
        env2.forward_distance = float(state["forward_ft"])
    if hasattr(env2, "steps"):
        try:
            env2.steps += 1
        except Exception:
            pass
    try:
        env2._get_obs()
    except Exception:
        pass

    return (
        state,
        action,
        float(base_elevator),
        float(mapped_elevator),
        float(diagnostic_elevator),
    )


def build_25ft_then_trigger20():
    """
    Reproduce the qualified 300->30 ft teacher, then the already-tested
    25-ft level. Continue toward 20 ft until the backward drift reaches
    FWD <= 298 ft, while still inside the 5-ft presentation corridor.
    """
    start = build_qualified_30ft_open(detailed=False)
    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    # First reproduce the passing 25-ft level.
    settle_hold = 0.0
    for _ in range(int(NEAR_GROUND_LEVEL_MAX_TIME / dt)):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        action, ctrl, vertical_bias = build_near_ground_action(
            state_before,
            target_alt_ft=TARGET_25_FT,
            hover_action0=hover_action0,
            vertical_bias=vertical_bias,
            dt=dt,
        )
        state, _, _, _ = raw_stage4_mapped_cycle(
            env2, fdm, action, lat0, lon0, mission_heading
        )

        reason = near_ground_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(f"25-ft qualification failed: {reason}")

        contact = contact_snapshot(fdm)
        if any_wow(contact):
            close_handoff(start)
            raise RuntimeError("Unexpected WOW during 25-ft qualification.")

        settle_hold = settle_hold + dt if level_settle_now(state, TARGET_25_FT) else 0.0
        if settle_hold >= NEAR_GROUND_SETTLE_SECONDS:
            break

    if settle_hold < NEAR_GROUND_SETTLE_SECONDS:
        close_handoff(start)
        raise RuntimeError("25-ft level did not reproduce.")

    state25 = snapshot(fdm, lat0, lon0, mission_heading)

    # Continue toward 20 ft only until the diagnosed backward drift becomes
    # visible, but before the 295-ft safety boundary.
    trigger_elapsed = 0.0
    trigger_state = None
    for _ in range(int(40.0 / dt)):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        action, ctrl, vertical_bias = build_near_ground_action(
            state_before,
            target_alt_ft=TARGET_20_FT,
            hover_action0=hover_action0,
            vertical_bias=vertical_bias,
            dt=dt,
        )
        state, _, _, _ = raw_stage4_mapped_cycle(
            env2, fdm, action, lat0, lon0, mission_heading
        )
        trigger_elapsed += dt

        reason = near_ground_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(f"20-ft trigger approach failed before trigger: {reason}")

        if state["forward_ft"] <= TRIGGER_FORWARD_FT:
            trigger_state = state.copy()
            break

    if trigger_state is None:
        close_handoff(start)
        raise RuntimeError("Near-ground backward-drift trigger was not reached.")

    return {
        **start,
        "vertical_bias": float(vertical_bias),
        "state25": state25,
        "trigger_state": trigger_state,
        "trigger_elapsed_s": float(trigger_elapsed),
    }


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)



# =====================================================================
# STAGE-4 NEAR-GROUND LONGITUDINAL MAPPING CALIBRATION V1
# =====================================================================

# High-altitude Stage-4 mapping remains untouched and qualified:
#   action1=-1 -> -0.183
#   action1=+1 -> -0.180
#
# Near-ground authority diagnostic established safe/useful positive release
# authority through -0.178 physical elevator.  Only below 30 ft we therefore
# test a dedicated normalization band:
NEAR_ELEVATOR_PHYSICAL_MIN = -0.183
NEAR_ELEVATOR_PHYSICAL_MAX = -0.178
NEAR_ELEVATOR_PHYSICAL_SPAN = (
    NEAR_ELEVATOR_PHYSICAL_MAX - NEAR_ELEVATOR_PHYSICAL_MIN
)

# Feedback is expressed directly in physical-elevator units.
# delta = -Kpos*(FWD-300) - Kv*Vfwd, then constrained to measured authority.
NEAR_LONG_KPOS_GRID = [0.0005, 0.0007, 0.0009]
NEAR_LONG_KV_GRID = [0.002, 0.004, 0.006]

NEAR_BRAKE_DELTA_MIN = -0.003   # qualified earlier (-0.183)
NEAR_RELEASE_DELTA_MAX = +0.002 # qualified near ground (-0.178)

NEAR_MAPPING_SAT_FRACTION_WARN = 0.50


def near_physical_elevator_to_action1(physical_cmd: float) -> float:
    p = float(np.clip(
        physical_cmd,
        NEAR_ELEVATOR_PHYSICAL_MIN,
        NEAR_ELEVATOR_PHYSICAL_MAX,
    ))
    a = (
        2.0 * (p - NEAR_ELEVATOR_PHYSICAL_MIN)
        / NEAR_ELEVATOR_PHYSICAL_SPAN
        - 1.0
    )
    return float(np.clip(a, -1.0, +1.0))


def near_action1_to_physical(action1: float) -> float:
    a = float(np.clip(action1, -1.0, +1.0))
    return float(
        NEAR_ELEVATOR_PHYSICAL_MIN
        + 0.5 * (a + 1.0) * NEAR_ELEVATOR_PHYSICAL_SPAN
    )


def raw_stage4_near_mapped_cycle(
    env2, fdm, action, lat0, lon0, mission_heading
):
    """
    Near-ground-only action1 normalization.
    Other actuator channels retain the already qualified Stage-4 wiring.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -1.0, +1.0).astype(np.float32)

    env2._apply_action(action)
    base_elevator = fdm_float(fdm, "fcs/elevator-cmd-norm")

    mapped_elevator = near_action1_to_physical(float(action[1]))
    fdm["fcs/elevator-cmd-norm"] = mapped_elevator

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError(
                "JSBSim stopped during near-ground longitudinal mapping calibration."
            )

    if hasattr(env2, "previous_action"):
        try:
            env2.previous_action = action.copy()
        except Exception:
            pass

    state = snapshot(fdm, lat0, lon0, mission_heading)

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = float(state["forward_ft"])
    if hasattr(env2, "steps"):
        try:
            env2.steps += 1
        except Exception:
            pass
    try:
        env2._get_obs()
    except Exception:
        pass

    return state, action, float(base_elevator), float(mapped_elevator)


def build_near_ground_action_mapped(
    state,
    target_alt_ft,
    hover_action0,
    vertical_bias,
    dt,
    long_kpos_phys,
    long_kv_phys,
):
    # Vertical controller is unchanged from the near-ground diagnostic.
    vs_des = NEAR_GROUND_ALT_TO_VS_GAIN * (
        float(target_alt_ft) - state["altitude_ft"]
    )
    vs_des = float(np.clip(vs_des, -NEAR_GROUND_VMAX_FPS, +0.20))
    vs_error = vs_des - state["vertical_speed_fps"]

    candidate_bias = float(np.clip(
        vertical_bias + LOCKED_VS_KI * vs_error * dt,
        VERTICAL_BIAS_MIN,
        VERTICAL_BIAS_MAX,
    ))
    effective_bias = float(candidate_bias)
    p_term = float(LOCKED_VS_KP * vs_error)

    residual_unclipped = p_term + effective_bias
    collective_residual = float(np.clip(
        residual_unclipped,
        COLLECTIVE_RESIDUAL_MIN,
        COLLECTIVE_RESIDUAL_MAX,
    ))

    # Anti-windup at the already qualified collective bounds.
    pushing_low = (
        collective_residual <= COLLECTIVE_RESIDUAL_MIN + 1e-9
        and vs_error < 0.0
    )
    pushing_high = (
        collective_residual >= COLLECTIVE_RESIDUAL_MAX - 1e-9
        and vs_error > 0.0
    )
    if pushing_low or pushing_high:
        candidate_bias = float(vertical_bias)
        effective_bias = float(candidate_bias)
        residual_unclipped = p_term + effective_bias
        collective_residual = float(np.clip(
            residual_unclipped,
            COLLECTIVE_RESIDUAL_MIN,
            COLLECTIVE_RESIDUAL_MAX,
        ))

    action0 = float(np.clip(
        hover_action0 + collective_residual,
        -1.0,
        +1.0,
    ))

    # New two-sided near-ground physical longitudinal feedback.
    ahead_ft = float(state["forward_ft"] - TARGET_FORWARD_FT)
    vfwd = float(state["forward_speed_fps"])
    elevator_delta_phys_raw = (
        -float(long_kpos_phys) * ahead_ft
        -float(long_kv_phys) * vfwd
    )
    elevator_delta_phys = float(np.clip(
        elevator_delta_phys_raw,
        NEAR_BRAKE_DELTA_MIN,
        NEAR_RELEASE_DELTA_MAX,
    ))
    desired_physical_elevator = float(np.clip(
        ELEVATOR_PHYSICAL_MAX + elevator_delta_phys,
        NEAR_ELEVATOR_PHYSICAL_MIN,
        NEAR_ELEVATOR_PHYSICAL_MAX,
    ))
    action1 = near_physical_elevator_to_action1(
        desired_physical_elevator
    )

    # Lateral/yaw remain exactly at the prior operating point.
    lateral_corr = (
        -LOCKED_LATERAL_KP * state["cross_track_ft"]
        -LOCKED_LATERAL_KD * state["lateral_speed_fps"]
    )
    raw_action2 = float(LATERAL_TRIM_ACTION + lateral_corr)
    action2 = float(np.clip(
        raw_action2,
        IDENTIFIED_A2_MIN,
        DESCENT_TESTED_A2_MAX,
    ))
    action3 = float(RUDDER_ACTION)

    return (
        np.asarray(
            [action0, action1, action2, action3],
            dtype=np.float32,
        ),
        {
            "target_alt_ft": float(target_alt_ft),
            "vs_des_fps": float(vs_des),
            "vs_error_fps": float(vs_error),
            "vs_p_term": float(p_term),
            "vertical_bias_state": float(candidate_bias),
            "vertical_bias_effective": float(effective_bias),
            "collective_residual_unclipped": float(residual_unclipped),
            "collective_residual": float(collective_residual),
            "ahead_ft": float(ahead_ft),
            "vfwd_fps": float(vfwd),
            "elevator_delta_phys_raw": float(elevator_delta_phys_raw),
            "elevator_delta_phys": float(elevator_delta_phys),
            "desired_physical_elevator": float(
                desired_physical_elevator
            ),
            "normalized_action1": float(action1),
            "raw_action2": float(raw_action2),
        },
        float(candidate_bias),
    )


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_near_ground_candidate(long_kpos_phys, long_kv_phys, detailed=False):
    start = build_qualified_30ft_open(detailed=False)

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    trace = []
    level_results = []

    overall_safe = True
    termination_reason = "completed_near_ground_ladder"
    max_abs_position = 0.0
    max_abs_cross = 0.0
    min_altitude = +999.0
    min_vertical_speed = +999.0
    min_physical_elevator = +999.0
    max_physical_elevator = -999.0
    brake_sat_steps = 0
    release_sat_steps = 0
    total_steps = 0

    for level_index, target_alt in enumerate(
        NEAR_GROUND_TARGETS_FT, start=1
    ):
        settle_hold = 0.0
        level_reason = "level_time_limit"
        level_elapsed = 0.0
        next_print = 0.0

        for step in range(int(NEAR_GROUND_LEVEL_MAX_TIME / dt)):
            state_before = snapshot(
                fdm, lat0, lon0, mission_heading
            )

            action, ctrl, vertical_bias = (
                build_near_ground_action_mapped(
                    state_before,
                    target_alt_ft=target_alt,
                    hover_action0=hover_action0,
                    vertical_bias=vertical_bias,
                    dt=dt,
                    long_kpos_phys=long_kpos_phys,
                    long_kv_phys=long_kv_phys,
                )
            )

            state, used, base_elev, mapped_elev = (
                raw_stage4_near_mapped_cycle(
                    env2,
                    fdm,
                    action,
                    lat0,
                    lon0,
                    mission_heading,
                )
            )

            level_elapsed = (step + 1) * dt
            total_steps += 1

            if (
                ctrl["elevator_delta_phys"]
                <= NEAR_BRAKE_DELTA_MIN + 1e-8
            ):
                brake_sat_steps += 1
            if (
                ctrl["elevator_delta_phys"]
                >= NEAR_RELEASE_DELTA_MAX - 1e-8
            ):
                release_sat_steps += 1

            max_abs_position = max(
                max_abs_position,
                abs(state["position_error_ft"]),
            )
            max_abs_cross = max(
                max_abs_cross,
                abs(state["cross_track_ft"]),
            )
            min_altitude = min(
                min_altitude,
                state["altitude_ft"],
            )
            min_vertical_speed = min(
                min_vertical_speed,
                state["vertical_speed_fps"],
            )
            min_physical_elevator = min(
                min_physical_elevator,
                mapped_elev,
            )
            max_physical_elevator = max(
                max_physical_elevator,
                mapped_elev,
            )

            settle_hold = (
                settle_hold + dt
                if level_settle_now(state, target_alt)
                else 0.0
            )

            contact = contact_snapshot(fdm)
            wow = any_wow(contact)

            trace.append({
                "long_kpos_phys": float(long_kpos_phys),
                "long_kv_phys": float(long_kv_phys),
                "level_index": int(level_index),
                "target_alt_ft": float(target_alt),
                "time_in_level_s": float(level_elapsed),
                "action0": float(used[0]),
                "action1": float(used[1]),
                "action2": float(used[2]),
                "action3": float(used[3]),
                "base_elevator_before_near_mapping": float(base_elev),
                "mapped_physical_elevator": float(mapped_elev),
                "wow_detected": int(wow),
                **{k: float(v) for k, v in ctrl.items()},
                **{k: float(v) for k, v in state.items()},
            })

            reason = near_ground_safety_reason(state)
            if reason:
                overall_safe = False
                level_reason = reason
                termination_reason = reason
                break

            if wow:
                overall_safe = False
                level_reason = (
                    "unexpected_weight_on_wheels_before_touchdown"
                )
                termination_reason = level_reason
                break

            if detailed and level_elapsed + 1e-9 >= next_print:
                print(
                    f"  target={target_alt:4.1f} "
                    f"t={level_elapsed:6.2f}s | "
                    f"ALT={state['altitude_ft']:6.2f} "
                    f"VS={state['vertical_speed_fps']:+6.3f} | "
                    f"FWD={state['forward_ft']:7.2f} "
                    f"err={state['position_error_ft']:+5.2f} "
                    f"V={state['forward_speed_fps']:+6.3f} | "
                    f"X={state['cross_track_ft']:+5.2f} "
                    f"LAT={state['lateral_speed_fps']:+6.3f} | "
                    f"ELEV={mapped_elev:+.6f} "
                    f"dE={ctrl['elevator_delta_phys']:+.5f} | "
                    f"HOLD={settle_hold:4.2f}s"
                )
                next_print += 10.0

            if settle_hold >= NEAR_GROUND_SETTLE_SECONDS:
                level_reason = "level_settled_3s"
                break

        final_level_state = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        level_pass = bool(
            overall_safe
            and level_reason == "level_settled_3s"
            and settle_hold >= NEAR_GROUND_SETTLE_SECONDS
        )

        level_results.append({
            "level_index": int(level_index),
            "target_alt_ft": float(target_alt),
            "pass": bool(level_pass),
            "reason": str(level_reason),
            "duration_s": float(level_elapsed),
            "settle_hold_s": float(settle_hold),
            "final_altitude_ft": float(
                final_level_state["altitude_ft"]
            ),
            "final_vertical_speed_fps": float(
                final_level_state["vertical_speed_fps"]
            ),
            "final_forward_ft": float(
                final_level_state["forward_ft"]
            ),
            "final_position_error_ft": float(
                final_level_state["position_error_ft"]
            ),
            "final_forward_speed_fps": float(
                final_level_state["forward_speed_fps"]
            ),
            "final_cross_track_ft": float(
                final_level_state["cross_track_ft"]
            ),
            "final_lateral_speed_fps": float(
                final_level_state["lateral_speed_fps"]
            ),
            "final_heading_error_deg": float(
                final_level_state["heading_error_deg"]
            ),
            "final_physical_elevator": float(
                final_level_state["physical_elevator_cmd"]
            ),
            "vertical_bias_state": float(vertical_bias),
        })

        if not level_pass:
            break

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    teacher_pass = bool(
        overall_safe
        and len(level_results) == len(NEAR_GROUND_TARGETS_FT)
        and all(r["pass"] for r in level_results)
        and final_state["altitude_ft"] >= NEAR_GROUND_MIN_SAFE_AGL_FT
        and abs(final_state["position_error_ft"])
            <= PRESENTATION_MAX_POSITION_ERROR_FT
        and abs(final_state["cross_track_ft"])
            <= PRESENTATION_MAX_CROSS_FT
        and abs(final_state["forward_speed_fps"])
            <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(final_state["lateral_speed_fps"])
            <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(final_state["heading_error_deg"])
            <= STAGE4_HEADING_TOL_DEG
    )

    result = {
        "teacher_pass": bool(teacher_pass),
        "safe": bool(overall_safe),
        "termination_reason": str(termination_reason),
        "long_kpos_phys": float(long_kpos_phys),
        "long_kv_phys": float(long_kv_phys),
        "levels_passed": int(
            sum(1 for r in level_results if r["pass"])
        ),
        "levels_total": int(len(NEAR_GROUND_TARGETS_FT)),
        "final_altitude_ft": float(final_state["altitude_ft"]),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_forward_ft": float(final_state["forward_ft"]),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "final_heading_error_deg": float(
            final_state["heading_error_deg"]
        ),
        "max_abs_position_error_ft": float(max_abs_position),
        "max_abs_cross_track_ft": float(max_abs_cross),
        "min_altitude_ft": float(min_altitude),
        "min_vertical_speed_fps": float(min_vertical_speed),
        "physical_elevator_min": float(min_physical_elevator),
        "physical_elevator_max": float(max_physical_elevator),
        "brake_saturation_fraction": float(
            brake_sat_steps / max(1, total_steps)
        ),
        "release_saturation_fraction": float(
            release_sat_steps / max(1, total_steps)
        ),
        "vertical_bias_final": float(vertical_bias),
        "level_results": level_results,
    }

    close_handoff(start)
    return result, trace


def candidate_sort_key(row):
    return (
        0 if row["teacher_pass"] else 1,
        0 if row["safe"] else 1,
        -row["levels_passed"],
        row["max_abs_position_error_ft"],
        row["max_abs_cross_track_ft"],
        row["release_saturation_fraction"],
        row["brake_saturation_fraction"],
        abs(row["final_position_error_ft"]),
        abs(row["final_forward_speed_fps"]),
    )



# =====================================================================
# STAGE-4 CONTINUOUS NEAR-GROUND DESCENT V2
# =====================================================================

CONTINUOUS_TARGET_ALT_FT = 9.0
CONTINUOUS_MAX_TIME_S = 180.0
CONTINUOUS_SETTLE_ALT_TOL_FT = 1.0
CONTINUOUS_SETTLE_VS_TOL_FPS = 0.20
CONTINUOUS_SETTLE_SECONDS = 3.0

CROSSING_LEVELS_FT = [25.0, 20.0, 15.0, 12.0, 10.0]


def continuous_settle_now(state):
    return bool(
        abs(state["altitude_ft"] - CONTINUOUS_TARGET_ALT_FT)
            <= CONTINUOUS_SETTLE_ALT_TOL_FT
        and abs(state["vertical_speed_fps"])
            <= CONTINUOUS_SETTLE_VS_TOL_FPS
        and abs(state["position_error_ft"])
            <= STAGE4_POS_TOL_FT
        and abs(state["forward_speed_fps"])
            <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(state["cross_track_ft"])
            <= STAGE4_CROSS_TOL_FT
        and abs(state["lateral_speed_fps"])
            <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(state["heading_error_deg"])
            <= STAGE4_HEADING_TOL_DEG
    )


def run_continuous_near_ground_candidate(
    long_kpos_phys,
    long_kv_phys,
    detailed=False,
):
    """
    Mission-relevant test:
      qualified 300->30 ft teacher
      -> continuous descent toward 9 ft
      -> no artificial holds at 25/20/15/12/10 ft.
    Intermediate altitudes are observation checkpoints only.
    """
    start = build_qualified_30ft_open(detailed=False)

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    initial_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    trace = []
    crossings = {}
    settle_hold = 0.0
    safe = True
    wow_seen = False
    termination = "continuous_time_limit"

    max_abs_position = abs(initial_state["position_error_ft"])
    max_abs_cross = abs(initial_state["cross_track_ft"])
    min_altitude = initial_state["altitude_ft"]
    min_vertical_speed = initial_state["vertical_speed_fps"]

    min_elevator = +999.0
    max_elevator = -999.0
    min_collective = +999.0
    max_collective = -999.0
    min_residual = +999.0
    max_residual = -999.0

    brake_sat_steps = 0
    release_sat_steps = 0
    collective_low_sat_steps = 0
    total_steps = 0

    next_print = 0.0

    for step in range(int(CONTINUOUS_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_mapped(
                state_before,
                target_alt_ft=CONTINUOUS_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                long_kpos_phys=long_kpos_phys,
                long_kv_phys=long_kv_phys,
            )
        )

        state, used, base_elev, mapped_elev = (
            raw_stage4_near_mapped_cycle(
                env2,
                fdm,
                action,
                lat0,
                lon0,
                mission_heading,
            )
        )

        t = (step + 1) * dt
        total_steps += 1

        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        wow_seen = wow_seen or wow

        if (
            ctrl["elevator_delta_phys"]
            <= NEAR_BRAKE_DELTA_MIN + 1e-8
        ):
            brake_sat_steps += 1
        if (
            ctrl["elevator_delta_phys"]
            >= NEAR_RELEASE_DELTA_MAX - 1e-8
        ):
            release_sat_steps += 1
        if (
            ctrl["collective_residual"]
            <= COLLECTIVE_RESIDUAL_MIN + 1e-8
        ):
            collective_low_sat_steps += 1

        max_abs_position = max(
            max_abs_position,
            abs(state["position_error_ft"]),
        )
        max_abs_cross = max(
            max_abs_cross,
            abs(state["cross_track_ft"]),
        )
        min_altitude = min(
            min_altitude,
            state["altitude_ft"],
        )
        min_vertical_speed = min(
            min_vertical_speed,
            state["vertical_speed_fps"],
        )
        min_elevator = min(min_elevator, mapped_elev)
        max_elevator = max(max_elevator, mapped_elev)
        min_collective = min(
            min_collective,
            state["physical_collective_cmd"],
        )
        max_collective = max(
            max_collective,
            state["physical_collective_cmd"],
        )
        min_residual = min(
            min_residual,
            ctrl["collective_residual"],
        )
        max_residual = max(
            max_residual,
            ctrl["collective_residual"],
        )

        settle_hold = (
            settle_hold + dt
            if continuous_settle_now(state)
            else 0.0
        )

        for level in CROSSING_LEVELS_FT:
            key = f"{level:.1f}"
            if key not in crossings and state["altitude_ft"] <= level:
                crossings[key] = {
                    "time_s": float(t),
                    "altitude_ft": float(state["altitude_ft"]),
                    "vertical_speed_fps": float(
                        state["vertical_speed_fps"]
                    ),
                    "forward_ft": float(state["forward_ft"]),
                    "position_error_ft": float(
                        state["position_error_ft"]
                    ),
                    "forward_speed_fps": float(
                        state["forward_speed_fps"]
                    ),
                    "cross_track_ft": float(
                        state["cross_track_ft"]
                    ),
                    "lateral_speed_fps": float(
                        state["lateral_speed_fps"]
                    ),
                    "heading_error_deg": float(
                        state["heading_error_deg"]
                    ),
                    "collective_residual": float(
                        ctrl["collective_residual"]
                    ),
                    "vertical_bias_state": float(
                        vertical_bias
                    ),
                    "physical_collective_cmd": float(
                        state["physical_collective_cmd"]
                    ),
                    "physical_elevator_cmd": float(
                        mapped_elev
                    ),
                }

        row = {
            "time_s": float(t),
            "target_alt_ft": float(CONTINUOUS_TARGET_ALT_FT),
            "long_kpos_phys": float(long_kpos_phys),
            "long_kv_phys": float(long_kv_phys),
            "action0": float(used[0]),
            "action1": float(used[1]),
            "action2": float(used[2]),
            "action3": float(used[3]),
            "mapped_physical_elevator": float(mapped_elev),
            "base_elevator_before_near_mapping": float(base_elev),
            "wow_detected": int(wow),
            "settle_hold_s": float(settle_hold),
            **{k: float(v) for k, v in ctrl.items()},
            **{k: float(v) for k, v in state.items()},
        }
        trace.append(row)

        reason = near_ground_safety_reason(state)
        if reason:
            safe = False
            termination = reason
            break

        # 9 ft is intentionally above the expected ground-contact AGL.
        # Any WOW before the dedicated touchdown test is therefore unexpected.
        if wow:
            safe = False
            termination = (
                "unexpected_weight_on_wheels_before_touchdown_test"
            )
            break

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  t={t:6.2f}s | "
                f"ALT={state['altitude_ft']:6.2f} "
                f"VS={state['vertical_speed_fps']:+6.3f}/"
                f"{ctrl['vs_des_fps']:+5.2f} | "
                f"FWD={state['forward_ft']:7.2f} "
                f"err={state['position_error_ft']:+5.2f} "
                f"V={state['forward_speed_fps']:+6.3f} | "
                f"X={state['cross_track_ft']:+5.2f} "
                f"LAT={state['lateral_speed_fps']:+6.3f} | "
                f"dA0={ctrl['collective_residual']:+6.3f} "
                f"VB={vertical_bias:+6.3f} | "
                f"ELEV={mapped_elev:+.6f} | "
                f"HOLD={settle_hold:4.2f}s"
            )
            next_print += 10.0

        if settle_hold >= CONTINUOUS_SETTLE_SECONDS:
            termination = "stable_9ft_hover_3s"
            break

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    passed = bool(
        safe
        and termination == "stable_9ft_hover_3s"
        and settle_hold >= CONTINUOUS_SETTLE_SECONDS
        and not wow_seen
        and abs(final_state["position_error_ft"])
            <= PRESENTATION_MAX_POSITION_ERROR_FT
        and abs(final_state["cross_track_ft"])
            <= PRESENTATION_MAX_CROSS_FT
        and abs(final_state["forward_speed_fps"])
            <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(final_state["lateral_speed_fps"])
            <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(final_state["heading_error_deg"])
            <= STAGE4_HEADING_TOL_DEG
    )

    result = {
        "pass": bool(passed),
        "safe": bool(safe),
        "termination": str(termination),
        "long_kpos_phys": float(long_kpos_phys),
        "long_kv_phys": float(long_kv_phys),
        "final_altitude_ft": float(final_state["altitude_ft"]),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_forward_ft": float(final_state["forward_ft"]),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "final_heading_error_deg": float(
            final_state["heading_error_deg"]
        ),
        "settle_hold_s": float(settle_hold),
        "wow_seen": bool(wow_seen),
        "max_abs_position_error_ft": float(max_abs_position),
        "max_abs_cross_track_ft": float(max_abs_cross),
        "min_altitude_ft": float(min_altitude),
        "min_vertical_speed_fps": float(min_vertical_speed),
        "vertical_bias_final": float(vertical_bias),
        "physical_elevator_min": float(min_elevator),
        "physical_elevator_max": float(max_elevator),
        "physical_collective_min": float(min_collective),
        "physical_collective_max": float(max_collective),
        "collective_residual_min": float(min_residual),
        "collective_residual_max": float(max_residual),
        "brake_saturation_fraction": float(
            brake_sat_steps / max(1, total_steps)
        ),
        "release_saturation_fraction": float(
            release_sat_steps / max(1, total_steps)
        ),
        "collective_low_saturation_fraction": float(
            collective_low_sat_steps / max(1, total_steps)
        ),
        "crossings": crossings,
    }

    close_handoff(start)
    return result, trace


def continuous_sort_key(row):
    return (
        0 if row["pass"] else 1,
        0 if row["safe"] else 1,
        abs(row["final_altitude_ft"] - CONTINUOUS_TARGET_ALT_FT),
        row["max_abs_position_error_ft"],
        row["max_abs_cross_track_ft"],
        row["release_saturation_fraction"],
        row["brake_saturation_fraction"],
        abs(row["final_position_error_ft"]),
        abs(row["final_forward_speed_fps"]),
    )



# =====================================================================
# STAGE-4 LOW-ALTITUDE VERTICAL-BIAS CALIBRATION V2
# =====================================================================

# Keep the clean near-ground longitudinal operating point from continuous V2.
FIXED_NEAR_LONG_KPOS = 0.0005
FIXED_NEAR_LONG_KV = 0.006

# The diagnostic proved useful/safe extra negative collective authority while
# remaining inside the already-qualified total residual envelope [-0.50,+0.15].
# We therefore calibrate only the LOW-ALTITUDE adaptive-bias lower bound.
LOW_ALT_BIAS_ONSET_FT = 25.0
LOW_ALT_BIAS_MIN_GRID = [-0.30, -0.32, -0.34, -0.36, -0.38]

LOW_ALT_TARGET_FT = 9.0
MAX_TIME_S = 260.0
FINAL_ALT_TOL_FT = 1.0
FINAL_VS_TOL_FPS = 0.20
FINAL_HOLD_SECONDS = 3.0
CROSSING_LEVELS_FT = [25.0, 20.0, 15.0, 12.0, 10.0]

# The total collective residual remains hard-clipped to the previously
# qualified Stage-4 authority envelope. No new actuator authority is created.


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def final_settle_now(state):
    return bool(
        abs(state["altitude_ft"] - LOW_ALT_TARGET_FT) <= FINAL_ALT_TOL_FT
        and abs(state["vertical_speed_fps"]) <= FINAL_VS_TOL_FPS
        and abs(state["position_error_ft"]) <= STAGE4_POS_TOL_FT
        and abs(state["forward_speed_fps"]) <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(state["cross_track_ft"]) <= STAGE4_CROSS_TOL_FT
        and abs(state["lateral_speed_fps"]) <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(state["heading_error_deg"]) <= STAGE4_HEADING_TOL_DEG
    )


def build_near_ground_action_low_alt_bias(
    state,
    target_alt_ft,
    hover_action0,
    vertical_bias,
    dt,
    low_alt_bias_min,
):
    # Same vertical-speed schedule as continuous V2.
    vs_des = NEAR_GROUND_ALT_TO_VS_GAIN * (
        float(target_alt_ft) - state["altitude_ft"]
    )
    vs_des = float(np.clip(vs_des, -NEAR_GROUND_VMAX_FPS, +0.20))
    vs_error = vs_des - state["vertical_speed_fps"]

    # Above 25 ft, preserve the prior bias lower bound exactly.
    # Below 25 ft, allow the integrator to use more of the already-qualified
    # collective residual authority.
    active_bias_min = (
        VERTICAL_BIAS_MIN
        if state["altitude_ft"] >= LOW_ALT_BIAS_ONSET_FT
        else float(low_alt_bias_min)
    )

    candidate_bias = float(np.clip(
        vertical_bias + LOCKED_VS_KI * vs_error * dt,
        active_bias_min,
        VERTICAL_BIAS_MAX,
    ))

    p_term = float(LOCKED_VS_KP * vs_error)
    residual_unclipped = p_term + candidate_bias
    collective_residual = float(np.clip(
        residual_unclipped,
        COLLECTIVE_RESIDUAL_MIN,
        COLLECTIVE_RESIDUAL_MAX,
    ))

    # Anti-windup at the already-qualified TOTAL collective-residual bounds.
    pushing_low = (
        collective_residual <= COLLECTIVE_RESIDUAL_MIN + 1e-9
        and vs_error < 0.0
    )
    pushing_high = (
        collective_residual >= COLLECTIVE_RESIDUAL_MAX - 1e-9
        and vs_error > 0.0
    )
    if pushing_low or pushing_high:
        candidate_bias = float(vertical_bias)
        p_term = float(LOCKED_VS_KP * vs_error)
        residual_unclipped = p_term + candidate_bias
        collective_residual = float(np.clip(
            residual_unclipped,
            COLLECTIVE_RESIDUAL_MIN,
            COLLECTIVE_RESIDUAL_MAX,
        ))

    action0 = float(np.clip(
        hover_action0 + collective_residual,
        -1.0,
        +1.0,
    ))

    # Near-ground longitudinal mapping and feedback remain unchanged.
    ahead_ft = float(state["forward_ft"] - TARGET_FORWARD_FT)
    vfwd = float(state["forward_speed_fps"])
    elevator_delta_phys_raw = (
        -FIXED_NEAR_LONG_KPOS * ahead_ft
        -FIXED_NEAR_LONG_KV * vfwd
    )
    elevator_delta_phys = float(np.clip(
        elevator_delta_phys_raw,
        NEAR_BRAKE_DELTA_MIN,
        NEAR_RELEASE_DELTA_MAX,
    ))
    desired_physical_elevator = float(np.clip(
        ELEVATOR_PHYSICAL_MAX + elevator_delta_phys,
        NEAR_ELEVATOR_PHYSICAL_MIN,
        NEAR_ELEVATOR_PHYSICAL_MAX,
    ))
    action1 = near_physical_elevator_to_action1(
        desired_physical_elevator
    )

    lateral_corr = (
        -LOCKED_LATERAL_KP * state["cross_track_ft"]
        -LOCKED_LATERAL_KD * state["lateral_speed_fps"]
    )
    raw_action2 = float(LATERAL_TRIM_ACTION + lateral_corr)
    action2 = float(np.clip(
        raw_action2,
        IDENTIFIED_A2_MIN,
        DESCENT_TESTED_A2_MAX,
    ))
    action3 = float(RUDDER_ACTION)

    return (
        np.asarray(
            [action0, action1, action2, action3],
            dtype=np.float32,
        ),
        {
            "target_alt_ft": float(target_alt_ft),
            "vs_des_fps": float(vs_des),
            "vs_error_fps": float(vs_error),
            "vs_p_term": float(p_term),
            "active_bias_min": float(active_bias_min),
            "vertical_bias_state": float(candidate_bias),
            "collective_residual_unclipped": float(
                residual_unclipped
            ),
            "collective_residual": float(collective_residual),
            "ahead_ft": float(ahead_ft),
            "vfwd_fps": float(vfwd),
            "elevator_delta_phys_raw": float(
                elevator_delta_phys_raw
            ),
            "elevator_delta_phys": float(elevator_delta_phys),
            "desired_physical_elevator": float(
                desired_physical_elevator
            ),
            "normalized_action1": float(action1),
            "raw_action2": float(raw_action2),
        },
        float(candidate_bias),
    )


def run_bias_candidate(low_alt_bias_min, detailed=False):
    start = build_qualified_30ft_open(detailed=False)

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    trace = []
    crossings = {}
    settle_hold = 0.0
    safe = True
    wow_seen = False
    termination = "time_limit"

    max_abs_position = 0.0
    max_abs_cross = 0.0
    min_altitude = +999.0
    min_vs = +999.0
    min_bias = +999.0
    max_bias = -999.0
    min_residual = +999.0
    max_residual = -999.0
    min_collective = +999.0
    max_collective = -999.0
    min_elevator = +999.0
    max_elevator = -999.0

    total_steps = 0
    total_resid_low_sat = 0
    bias_min_sat = 0
    elev_brake_sat = 0
    elev_release_sat = 0

    next_print = 0.0

    for step in range(int(MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=LOW_ALT_TARGET_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=low_alt_bias_min,
            )
        )

        state, used, base_elev, mapped_elev = (
            raw_stage4_near_mapped_cycle(
                env2,
                fdm,
                action,
                lat0,
                lon0,
                mission_heading,
            )
        )

        t = (step + 1) * dt
        total_steps += 1

        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        wow_seen = wow_seen or wow

        settle_hold = (
            settle_hold + dt
            if final_settle_now(state)
            else 0.0
        )

        max_abs_position = max(
            max_abs_position,
            abs(state["position_error_ft"]),
        )
        max_abs_cross = max(
            max_abs_cross,
            abs(state["cross_track_ft"]),
        )
        min_altitude = min(min_altitude, state["altitude_ft"])
        min_vs = min(min_vs, state["vertical_speed_fps"])
        min_bias = min(min_bias, vertical_bias)
        max_bias = max(max_bias, vertical_bias)
        min_residual = min(
            min_residual,
            ctrl["collective_residual"],
        )
        max_residual = max(
            max_residual,
            ctrl["collective_residual"],
        )
        min_collective = min(
            min_collective,
            state["physical_collective_cmd"],
        )
        max_collective = max(
            max_collective,
            state["physical_collective_cmd"],
        )
        min_elevator = min(min_elevator, mapped_elev)
        max_elevator = max(max_elevator, mapped_elev)

        if (
            ctrl["collective_residual"]
            <= COLLECTIVE_RESIDUAL_MIN + 1e-8
        ):
            total_resid_low_sat += 1

        if (
            state_before["altitude_ft"] < LOW_ALT_BIAS_ONSET_FT
            and vertical_bias <= float(low_alt_bias_min) + 1e-8
        ):
            bias_min_sat += 1

        if (
            ctrl["elevator_delta_phys"]
            <= NEAR_BRAKE_DELTA_MIN + 1e-8
        ):
            elev_brake_sat += 1

        if (
            ctrl["elevator_delta_phys"]
            >= NEAR_RELEASE_DELTA_MAX - 1e-8
        ):
            elev_release_sat += 1

        for level in CROSSING_LEVELS_FT:
            key = f"{level:.1f}"
            if key not in crossings and state["altitude_ft"] <= level:
                crossings[key] = {
                    "time_s": float(t),
                    "altitude_ft": float(state["altitude_ft"]),
                    "vertical_speed_fps": float(
                        state["vertical_speed_fps"]
                    ),
                    "forward_ft": float(state["forward_ft"]),
                    "position_error_ft": float(
                        state["position_error_ft"]
                    ),
                    "forward_speed_fps": float(
                        state["forward_speed_fps"]
                    ),
                    "cross_track_ft": float(
                        state["cross_track_ft"]
                    ),
                    "lateral_speed_fps": float(
                        state["lateral_speed_fps"]
                    ),
                    "heading_error_deg": float(
                        state["heading_error_deg"]
                    ),
                    "vertical_bias_state": float(
                        vertical_bias
                    ),
                    "collective_residual": float(
                        ctrl["collective_residual"]
                    ),
                    "physical_collective_cmd": float(
                        state["physical_collective_cmd"]
                    ),
                    "physical_elevator_cmd": float(mapped_elev),
                }

        trace.append({
            "time_s": float(t),
            "low_alt_bias_min": float(low_alt_bias_min),
            "action0": float(used[0]),
            "action1": float(used[1]),
            "action2": float(used[2]),
            "action3": float(used[3]),
            "mapped_physical_elevator": float(mapped_elev),
            "base_elevator_before_near_mapping": float(base_elev),
            "wow_detected": int(wow),
            "settle_hold_s": float(settle_hold),
            **{k: float(v) for k, v in ctrl.items()},
            **{k: float(v) for k, v in state.items()},
        })

        reason = near_ground_safety_reason(state)
        if reason:
            safe = False
            termination = reason
            break

        # This is still a pre-touchdown calibration. WOW must remain zero.
        if wow:
            safe = False
            termination = (
                "unexpected_weight_on_wheels_before_touchdown"
            )
            break

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  t={t:6.2f}s | "
                f"ALT={state['altitude_ft']:6.2f} "
                f"VS={state['vertical_speed_fps']:+6.3f}/"
                f"{ctrl['vs_des_fps']:+5.2f} | "
                f"FWD={state['forward_ft']:7.2f} "
                f"err={state['position_error_ft']:+5.2f} "
                f"V={state['forward_speed_fps']:+6.3f} | "
                f"X={state['cross_track_ft']:+5.2f} "
                f"LAT={state['lateral_speed_fps']:+6.3f} | "
                f"VB={vertical_bias:+6.3f} "
                f"VBmin={ctrl['active_bias_min']:+6.3f} | "
                f"dA0={ctrl['collective_residual']:+6.3f} "
                f"COLL={state['physical_collective_cmd']:+.6f} | "
                f"ELEV={mapped_elev:+.6f} "
                f"HOLD={settle_hold:4.2f}s"
            )
            next_print += 10.0

        if settle_hold >= FINAL_HOLD_SECONDS:
            termination = "stable_9ft_hover_3s"
            break

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    passed = bool(
        safe
        and termination == "stable_9ft_hover_3s"
        and settle_hold >= FINAL_HOLD_SECONDS
        and not wow_seen
        and abs(final_state["position_error_ft"])
            <= PRESENTATION_MAX_POSITION_ERROR_FT
        and abs(final_state["cross_track_ft"])
            <= PRESENTATION_MAX_CROSS_FT
        and abs(final_state["forward_speed_fps"])
            <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(final_state["lateral_speed_fps"])
            <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(final_state["heading_error_deg"])
            <= STAGE4_HEADING_TOL_DEG
    )

    result = {
        "pass": bool(passed),
        "safe": bool(safe),
        "termination": str(termination),
        "low_alt_bias_min": float(low_alt_bias_min),
        "final_altitude_ft": float(final_state["altitude_ft"]),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_forward_ft": float(final_state["forward_ft"]),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "final_heading_error_deg": float(
            final_state["heading_error_deg"]
        ),
        "settle_hold_s": float(settle_hold),
        "wow_seen": bool(wow_seen),
        "max_abs_position_error_ft": float(max_abs_position),
        "max_abs_cross_track_ft": float(max_abs_cross),
        "min_altitude_ft": float(min_altitude),
        "min_vertical_speed_fps": float(min_vs),
        "vertical_bias_min_observed": float(min_bias),
        "vertical_bias_max_observed": float(max_bias),
        "collective_residual_min": float(min_residual),
        "collective_residual_max": float(max_residual),
        "physical_collective_min": float(min_collective),
        "physical_collective_max": float(max_collective),
        "physical_elevator_min": float(min_elevator),
        "physical_elevator_max": float(max_elevator),
        "total_residual_low_saturation_fraction": float(
            total_resid_low_sat / max(1, total_steps)
        ),
        "bias_min_saturation_fraction": float(
            bias_min_sat / max(1, total_steps)
        ),
        "elevator_brake_saturation_fraction": float(
            elev_brake_sat / max(1, total_steps)
        ),
        "elevator_release_saturation_fraction": float(
            elev_release_sat / max(1, total_steps)
        ),
        "crossings": crossings,
    }

    close_handoff(start)
    return result, trace


def rank_key(row):
    return (
        0 if row["pass"] else 1,
        0 if row["safe"] else 1,
        abs(row["final_altitude_ft"] - LOW_ALT_TARGET_FT),
        row["max_abs_position_error_ft"],
        row["max_abs_cross_track_ft"],
        row["total_residual_low_saturation_fraction"],
        row["bias_min_saturation_fraction"],
        abs(row["final_position_error_ft"]),
        abs(row["final_forward_speed_fps"]),
    )


# Guard against accidental collision with the locked Stage-3 observation target.
if abs(float(TARGET_ALT_FT) - 300.0) > 1e-9:
    raise RuntimeError(
        f"Configuration error: locked Stage-3 TARGET_ALT_FT must remain 300.0, got {TARGET_ALT_FT}"
    )


# =====================================================================
# STAGE-4 NATIVE ACTION0 HEADROOM DIAGNOSTIC V1
# =====================================================================

# Use the best low-alt operating point that actually reached the native
# residual floor. This is diagnostic only, not a final teacher lock.
DIAG_LOW_ALT_BIAS_MIN = -0.360
DIAG_NEAR_LONG_KPOS = 0.0005
DIAG_NEAR_LONG_KV = 0.006

TRIGGER_ALT_MAX_FT = 19.0
TRIGGER_VS_ABS_MAX_FPS = 0.10
TRIGGER_MAX_TIME_S = 280.0

PULSE_SECONDS = 2.0
HEADROOM_FRACTIONS = [0.0, 0.33, 0.66, 1.0]

SUSTAINED_SECONDS = 30.0
SUSTAINED_HEADROOM_FRACTIONS = [0.50, 1.0]

DIAG_MIN_AGL_FT = 12.5
DIAG_MAX_DOWN_VS_FPS = -0.80


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_native_headroom_trigger():
    """
    Rebuild locked chain -> qualified 300->30 teacher -> low-alt continuous
    descent using biasMin=-0.36 until the previous V2 total-residual floor
    (-0.50) is active near the ~18-19 ft equilibrium.
    """
    start = build_qualified_30ft_open(detailed=False)

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    trigger_state = None
    trigger_ctrl = None
    trigger_action = None
    elapsed = 0.0

    for step in range(int(TRIGGER_MAX_TIME_S / dt)):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)

        action, ctrl, vertical_bias = build_near_ground_action_low_alt_bias(
            state_before,
            target_alt_ft=LOW_ALT_TARGET_FT,
            hover_action0=hover_action0,
            vertical_bias=vertical_bias,
            dt=dt,
            low_alt_bias_min=DIAG_LOW_ALT_BIAS_MIN,
        )

        state, used, _, mapped_elev = raw_stage4_near_mapped_cycle(
            env2,
            fdm,
            action,
            lat0,
            lon0,
            mission_heading,
        )

        elapsed = (step + 1) * dt

        reason = near_ground_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(
                f"Native-action0 trigger approach failed before trigger: {reason}"
            )

        if any_wow(contact_snapshot(fdm)):
            close_handoff(start)
            raise RuntimeError("Unexpected WOW before native-action0 trigger.")

        at_residual_floor = ctrl["collective_residual"] <= COLLECTIVE_RESIDUAL_MIN + 1e-8
        near_equilibrium = (
            state["altitude_ft"] <= TRIGGER_ALT_MAX_FT
            and abs(state["vertical_speed_fps"]) <= TRIGGER_VS_ABS_MAX_FPS
        )

        if at_residual_floor and near_equilibrium:
            trigger_state = state.copy()
            trigger_ctrl = dict(ctrl)
            trigger_action = np.asarray(used, dtype=np.float32).copy()
            break

    if trigger_state is None:
        close_handoff(start)
        raise RuntimeError(
            "Did not reproduce the ~18-19 ft native residual-floor equilibrium."
        )

    native_action0 = float(trigger_action[0])
    remaining_action0_headroom = float(max(0.0, native_action0 - (-1.0)))

    return {
        **start,
        "vertical_bias": float(vertical_bias),
        "trigger_state": trigger_state,
        "trigger_ctrl": trigger_ctrl,
        "trigger_action": trigger_action,
        "trigger_elapsed_s": float(elapsed),
        "native_action0": float(native_action0),
        "remaining_action0_headroom": float(remaining_action0_headroom),
    }


def raw_native_action0_cycle(
    env2,
    fdm,
    action,
    forced_action0,
    lat0,
    lon0,
    mission_heading,
):
    """
    Identification-only cycle using ONLY the existing normalized action range.
    No direct physical collective override is applied.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    action[0] = float(np.clip(forced_action0, -1.0, +1.0))
    action = np.clip(action, -1.0, +1.0).astype(np.float32)

    env2._apply_action(action)

    # Preserve the already-tested near-ground elevator mapping.
    mapped_elevator = near_action1_to_physical(float(action[1]))
    fdm["fcs/elevator-cmd-norm"] = mapped_elevator

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError(
                "JSBSim stopped during native action0 headroom diagnostic."
            )

    if hasattr(env2, "previous_action"):
        try:
            env2.previous_action = action.copy()
        except Exception:
            pass

    state = snapshot(fdm, lat0, lon0, mission_heading)

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = float(state["forward_ft"])
    if hasattr(env2, "steps"):
        try:
            env2.steps += 1
        except Exception:
            pass
    try:
        env2._get_obs()
    except Exception:
        pass

    return state, action, float(mapped_elevator)


def diag_safety_reason(state):
    if state["altitude_ft"] < DIAG_MIN_AGL_FT:
        return "diagnostic_agl_floor"
    if state["vertical_speed_fps"] < DIAG_MAX_DOWN_VS_FPS:
        return "diagnostic_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "diagnostic_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "diagnostic_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "diagnostic_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "diagnostic_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "diagnostic_roll_limit"
    return ""



# =====================================================================
# STAGE-4 LOW-ALTITUDE PHYSICAL COLLECTIVE IDENTIFICATION V2
# =====================================================================

# Diagnostic operating point only.
PHYS_ID_LOW_ALT_BIAS_MIN = -0.360
PHYS_ID_LONG_KPOS = 0.0005
PHYS_ID_LONG_KV = 0.006

# First reproduce the native action0=-1 low-altitude equilibrium.
PHYS_ID_TRIGGER_ALT_MAX_FT = 18.0
PHYS_ID_TRIGGER_ABS_VS_MAX_FPS = 0.04
PHYS_ID_TRIGGER_MAX_TIME_S = 340.0

# Baseline physical collective at native action0=-1 was measured ~0.590000.
# Probe ONLY small additional reductions in physical collective.
PHYSICAL_COLLECTIVE_DELTAS = [
    0.00000,
    -0.00050,
    -0.00100,
    -0.00150,
    -0.00200,
    -0.00250,
    -0.00300,
]
PHYS_PULSE_SECONDS = 2.0

SUSTAINED_PHYSICAL_COLLECTIVE_DELTAS = [
    -0.00125,
    -0.00150,
    -0.00175,
    -0.00200,
]
PHYS_SUSTAINED_SECONDS = 12.0

PHYS_DIAG_MIN_AGL_FT = 11.5
PHYS_DIAG_MAX_DOWN_VS_FPS = -0.70


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_native_min_equilibrium():
    """
    Locked chain -> qualified 300->30 -> low-alt controller until residual floor
    -> then hold native action0=-1 until a repeatable low-altitude equilibrium.
    Returns SAME open FDM.
    """
    start = build_qualified_30ft_open(detailed=False)

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    reached_floor = False
    elapsed = 0.0

    for step in range(int(PHYS_ID_TRIGGER_MAX_TIME_S / dt)):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)

        action, ctrl, vertical_bias = build_near_ground_action_low_alt_bias(
            state_before,
            target_alt_ft=LOW_ALT_TARGET_FT,
            hover_action0=hover_action0,
            vertical_bias=vertical_bias,
            dt=dt,
            low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
        )

        if ctrl["collective_residual"] <= COLLECTIVE_RESIDUAL_MIN + 1e-8:
            reached_floor = True

        # Once the qualified residual floor is reached, explicitly exhaust the
        # remaining native normalized action0 headroom by using action0=-1.
        if reached_floor:
            forced_a0 = -1.0
            state, used, mapped_elev = raw_native_action0_cycle(
                env2,
                fdm,
                action,
                forced_action0=forced_a0,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        else:
            state, used, _, mapped_elev = raw_stage4_near_mapped_cycle(
                env2,
                fdm,
                action,
                lat0,
                lon0,
                mission_heading,
            )

        elapsed = (step + 1) * dt

        reason = near_ground_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(
                f"Physical-collective trigger approach failed: {reason}"
            )

        if any_wow(contact_snapshot(fdm)):
            close_handoff(start)
            raise RuntimeError("Unexpected WOW before physical-collective ID.")

        if (
            reached_floor
            and state["altitude_ft"] <= PHYS_ID_TRIGGER_ALT_MAX_FT
            and abs(state["vertical_speed_fps"]) <= PHYS_ID_TRIGGER_ABS_VS_MAX_FPS
        ):
            trigger_state = state.copy()
            trigger_action = np.asarray(used, dtype=np.float32).copy()
            baseline_physical_collective = float(
                state["physical_collective_cmd"]
            )
            return {
                **start,
                "vertical_bias": float(vertical_bias),
                "trigger_state": trigger_state,
                "trigger_action": trigger_action,
                "baseline_physical_collective": baseline_physical_collective,
                "trigger_elapsed_s": float(elapsed),
            }

    close_handoff(start)
    raise RuntimeError(
        "Did not reproduce native action0=-1 low-altitude equilibrium."
    )


def raw_physical_collective_cycle(
    env2,
    fdm,
    action,
    physical_collective_cmd,
    lat0,
    lon0,
    mission_heading,
):
    """
    Identification-only physical collective override.

    action0 remains native -1. The normal env action is applied first, then only
    fcs/collective-cmd-norm is overwritten to the requested diagnostic value.
    Near-ground elevator mapping is preserved.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    action[0] = -1.0
    action = np.clip(action, -1.0, +1.0).astype(np.float32)

    env2._apply_action(action)

    mapped_elevator = near_action1_to_physical(float(action[1]))
    fdm["fcs/elevator-cmd-norm"] = mapped_elevator

    requested_collective = float(np.clip(
        physical_collective_cmd,
        0.0,
        1.0,
    ))
    fdm["fcs/collective-cmd-norm"] = requested_collective

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError(
                "JSBSim stopped during physical collective identification."
            )

    if hasattr(env2, "previous_action"):
        try:
            env2.previous_action = action.copy()
        except Exception:
            pass

    state = snapshot(fdm, lat0, lon0, mission_heading)

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = float(state["forward_ft"])
    if hasattr(env2, "steps"):
        try:
            env2.steps += 1
        except Exception:
            pass
    try:
        env2._get_obs()
    except Exception:
        pass

    return state, action, float(mapped_elevator), requested_collective


def physical_diag_safety_reason(state):
    if state["altitude_ft"] < PHYS_DIAG_MIN_AGL_FT:
        return "physical_diag_agl_floor"
    if state["vertical_speed_fps"] < PHYS_DIAG_MAX_DOWN_VS_FPS:
        return "physical_diag_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "physical_diag_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "physical_diag_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "physical_diag_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "physical_diag_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "physical_diag_roll_limit"
    return ""



# =====================================================================
# STAGE-4 LOW-ALT PHYSICAL COLLECTIVE — FOCUSED SUSTAINED V3
# =====================================================================

# V2 established that useful short-pulse authority begins around -0.0020
# physical collective and becomes clear at -0.0025 ... -0.0030.
# V2 sustained tests stopped at -0.0020, so this V3 tests only the missing
# sustained band. No new PPO training and no final actuator mapping yet.

V3_SUSTAINED_DELTAS = [
    -0.00225,
    -0.00250,
    -0.00275,
    -0.00300,
    -0.00325,
]
V3_SUSTAINED_SECONDS = 12.0

# Still pre-touchdown identification.
V3_MIN_AGL_FT = 11.5
V3_MAX_DOWN_VS_FPS = -0.70


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def v3_safety_reason(state):
    if state["altitude_ft"] < V3_MIN_AGL_FT:
        return "v3_agl_floor"
    if state["vertical_speed_fps"] < V3_MAX_DOWN_VS_FPS:
        return "v3_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "v3_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "v3_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "v3_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "v3_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "v3_roll_limit"
    return ""



# =====================================================================
# STAGE-4 GROUND-EFFECT COLLECTIVE EQUILIBRIUM CURVE V1
# =====================================================================

# V3 proved that a fixed collective reduction simply moves the helicopter to
# a lower near-ground equilibrium and that the response becomes strongly
# altitude-dependent.  This diagnostic therefore maps the equilibrium curve
# directly instead of fitting one global linear model.

CURVE_COLLECTIVE_LEVELS = [
    0.58650,
    0.58600,
    0.58550,
    0.58500,
    0.58450,
    0.58400,
    0.58350,
    0.58300,
    0.58250,
]

CURVE_LEVEL_MAX_TIME_S = 60.0
CURVE_SETTLE_VS_ABS_FPS = 0.030
CURVE_SETTLE_SECONDS = 5.0

# Pre-touchdown diagnostic guardrails.
CURVE_MIN_AGL_FT = 11.5
CURVE_MAX_DOWN_VS_FPS = -0.55


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def curve_safety_reason(state):
    if state["altitude_ft"] < CURVE_MIN_AGL_FT:
        return "curve_agl_floor"
    if state["vertical_speed_fps"] < CURVE_MAX_DOWN_VS_FPS:
        return "curve_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "curve_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "curve_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "curve_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "curve_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "curve_roll_limit"
    return ""



# =====================================================================
# STAGE-4 GROUND-EFFECT COLLECTIVE EQUILIBRIUM CURVE V2
# =====================================================================

# V1 exposed a methodological issue: a 5-s low-VS condition could be met too
# early after a tiny collective step, before the slow near-ground transient had
# fully developed. V2 therefore uses:
#   1) larger 0.001 physical-collective continuation steps,
#   2) a mandatory dwell before equilibrium can be accepted,
#   3) a rolling altitude-range + vertical-speed stability check,
#   4) immediate stop if a level does not genuinely settle.

CURVE_COLLECTIVE_LEVELS = [
    0.58750,
    0.58650,
    0.58550,
    0.58450,
    0.58350,
    0.58250,
    0.58150,
    0.58050,
    0.57950,
    0.57850,
    0.57750,
    0.57650,
    0.57550,
]

CURVE_MIN_DWELL_S = 15.0
CURVE_STABILITY_WINDOW_S = 5.0
CURVE_LEVEL_MAX_TIME_S = 50.0

CURVE_VS_MEAN_ABS_MAX_FPS = 0.025
CURVE_VS_PEAK_ABS_MAX_FPS = 0.050
CURVE_ALT_RANGE_MAX_FT = 0.15

# Still pre-touchdown. We allow the curve to approach the 9-ft regime but stop
# well before the expected skid-contact AGL (~6.3 ft from reset00 evidence).
CURVE_MIN_AGL_FT = 9.5
CURVE_MAX_DOWN_VS_FPS = -0.50


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def curve_safety_reason(state):
    if state["altitude_ft"] < CURVE_MIN_AGL_FT:
        return "curve_agl_floor"
    if state["vertical_speed_fps"] < CURVE_MAX_DOWN_VS_FPS:
        return "curve_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "curve_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "curve_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "curve_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "curve_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "curve_roll_limit"
    return ""


def rolling_equilibrium(window_rows):
    if not window_rows:
        return False, {}
    vs = np.asarray(
        [r["vertical_speed_fps"] for r in window_rows],
        dtype=float,
    )
    alt = np.asarray(
        [r["altitude_ft"] for r in window_rows],
        dtype=float,
    )
    stats = {
        "vs_mean_fps": float(np.mean(vs)),
        "vs_abs_peak_fps": float(np.max(np.abs(vs))),
        "altitude_mean_ft": float(np.mean(alt)),
        "altitude_range_ft": float(np.max(alt) - np.min(alt)),
    }
    ok = bool(
        abs(stats["vs_mean_fps"]) <= CURVE_VS_MEAN_ABS_MAX_FPS
        and stats["vs_abs_peak_fps"] <= CURVE_VS_PEAK_ABS_MAX_FPS
        and stats["altitude_range_ft"] <= CURVE_ALT_RANGE_MAX_FT
    )
    return ok, stats



# =====================================================================
# STAGE-4 GROUND-EFFECT CURVE EXTENSION V3
# =====================================================================

# Reproduce the already-confirmed V2 curve down to 0.5755, then extend only
# the lower end. No PPO training, no touchdown, no controller retuning.

ANCHOR_LEVELS = [
    0.58750,
    0.58650,
    0.58550,
    0.58450,
    0.58350,
    0.58250,
    0.58150,
    0.58050,
    0.57950,
    0.57850,
    0.57750,
    0.57650,
    0.57550,
]

EXTENSION_LEVELS = [
    0.57450,
    0.57350,
    0.57250,
    0.57150,
    0.57050,
    0.56950,
]

# Same robust equilibrium definition as V2.
EXT_MIN_DWELL_S = 15.0
EXT_STABILITY_WINDOW_S = 5.0
EXT_LEVEL_MAX_TIME_S = 50.0

EXT_VS_MEAN_ABS_MAX_FPS = 0.025
EXT_VS_PEAK_ABS_MAX_FPS = 0.050
EXT_ALT_RANGE_MAX_FT = 0.15

# Still pre-touchdown identification.
EXT_MIN_AGL_FT = 9.5
EXT_MAX_DOWN_VS_FPS = -0.50


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ext_safety_reason(state):
    if state["altitude_ft"] < EXT_MIN_AGL_FT:
        return "extension_agl_floor"
    if state["vertical_speed_fps"] < EXT_MAX_DOWN_VS_FPS:
        return "extension_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "extension_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "extension_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "extension_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "extension_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "extension_roll_limit"
    return ""


def robust_equilibrium(window_rows):
    if not window_rows:
        return False, {}
    vs = np.asarray(
        [r["vertical_speed_fps"] for r in window_rows],
        dtype=float,
    )
    alt = np.asarray(
        [r["altitude_ft"] for r in window_rows],
        dtype=float,
    )
    stats = {
        "vs_mean_fps": float(np.mean(vs)),
        "vs_abs_peak_fps": float(np.max(np.abs(vs))),
        "altitude_mean_ft": float(np.mean(alt)),
        "altitude_range_ft": float(np.max(alt) - np.min(alt)),
    }
    ok = bool(
        abs(stats["vs_mean_fps"]) <= EXT_VS_MEAN_ABS_MAX_FPS
        and stats["vs_abs_peak_fps"] <= EXT_VS_PEAK_ABS_MAX_FPS
        and stats["altitude_range_ft"] <= EXT_ALT_RANGE_MAX_FT
    )
    return ok, stats


def hold_collective_to_equilibrium(
    env2,
    fdm,
    lat0,
    lon0,
    mission_heading,
    dt,
    hover_action0,
    vertical_bias,
    physical_collective,
    phase,
    level_index,
    trace_rows,
):
    window_steps = max(
        2,
        int(round(EXT_STABILITY_WINDOW_S / dt)),
    )
    window = []
    eq_stats = {}
    term = "level_time_limit"

    min_alt = +999.0
    min_vs = +999.0
    max_abs_pos = 0.0
    max_abs_cross = 0.0

    for step in range(int(EXT_LEVEL_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=LOW_ALT_TARGET_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        state, used, mapped_elev, requested_collective = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=physical_collective,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )

        t = (step + 1) * dt

        min_alt = min(min_alt, state["altitude_ft"])
        min_vs = min(min_vs, state["vertical_speed_fps"])
        max_abs_pos = max(
            max_abs_pos,
            abs(state["position_error_ft"]),
        )
        max_abs_cross = max(
            max_abs_cross,
            abs(state["cross_track_ft"]),
        )

        window.append({
            "vertical_speed_fps": float(
                state["vertical_speed_fps"]
            ),
            "altitude_ft": float(state["altitude_ft"]),
        })
        if len(window) > window_steps:
            window.pop(0)

        stable_now = False
        stats = {}
        if (
            t >= EXT_MIN_DWELL_S
            and len(window) >= window_steps
        ):
            stable_now, stats = robust_equilibrium(window)

        trace_rows.append({
            "phase": str(phase),
            "level_index": int(level_index),
            "time_in_level_s": float(t),
            "requested_physical_collective": float(
                requested_collective
            ),
            "actual_physical_collective_cmd": float(
                state["physical_collective_cmd"]
            ),
            "mapped_physical_elevator": float(mapped_elev),
            "eligible_for_equilibrium": int(
                t >= EXT_MIN_DWELL_S
            ),
            "stable_window": int(stable_now),
            "window_vs_mean_fps": float(
                stats.get("vs_mean_fps", float("nan"))
            ),
            "window_vs_abs_peak_fps": float(
                stats.get("vs_abs_peak_fps", float("nan"))
            ),
            "window_altitude_range_ft": float(
                stats.get("altitude_range_ft", float("nan"))
            ),
            **{k: float(v) for k, v in state.items()},
        })

        reason = ext_safety_reason(state)
        if reason:
            term = reason
            return {
                "safe": False,
                "settled": False,
                "termination": term,
                "vertical_bias": float(vertical_bias),
                "final_state": state,
                "eq_stats": {},
                "min_altitude_ft": float(min_alt),
                "min_vertical_speed_fps": float(min_vs),
                "max_abs_position_error_ft": float(max_abs_pos),
                "max_abs_cross_track_ft": float(max_abs_cross),
            }

        if any_wow(contact_snapshot(fdm)):
            term = "unexpected_wow"
            return {
                "safe": False,
                "settled": False,
                "termination": term,
                "vertical_bias": float(vertical_bias),
                "final_state": state,
                "eq_stats": {},
                "min_altitude_ft": float(min_alt),
                "min_vertical_speed_fps": float(min_vs),
                "max_abs_position_error_ft": float(max_abs_pos),
                "max_abs_cross_track_ft": float(max_abs_cross),
            }

        if stable_now:
            eq_stats = stats
            term = "equilibrium_confirmed"
            return {
                "safe": True,
                "settled": True,
                "termination": term,
                "vertical_bias": float(vertical_bias),
                "final_state": state,
                "eq_stats": eq_stats,
                "min_altitude_ft": float(min_alt),
                "min_vertical_speed_fps": float(min_vs),
                "max_abs_position_error_ft": float(max_abs_pos),
                "max_abs_cross_track_ft": float(max_abs_cross),
            }

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    return {
        "safe": True,
        "settled": False,
        "termination": "unsettled_level",
        "vertical_bias": float(vertical_bias),
        "final_state": final_state,
        "eq_stats": {},
        "min_altitude_ft": float(min_alt),
        "min_vertical_speed_fps": float(min_vs),
        "max_abs_position_error_ft": float(max_abs_pos),
        "max_abs_cross_track_ft": float(max_abs_cross),
    }



# =====================================================================
# STAGE-4 GROUND-EFFECT CURVE EXTENSION V4
# =====================================================================

# V3 confirmed the lower curve safely to:
#   COLL=0.569500 -> EQALT=10.932 ft
# This is only 0.182 ft above the conservative 10.75-ft capture-design gate.
# V4 therefore extends ONLY the final lower end, using the exact same robust
# equilibrium criteria and pre-touchdown safety guards.

ANCHOR_LEVELS = [
    0.58750,
    0.58650,
    0.58550,
    0.58450,
    0.58350,
    0.58250,
    0.58150,
    0.58050,
    0.57950,
    0.57850,
    0.57750,
    0.57650,
    0.57550,
    0.57450,
    0.57350,
    0.57250,
    0.57150,
    0.57050,
    0.56950,
]

EXTENSION_LEVELS = [
    0.56850,
    0.56750,
    0.56650,
    0.56550,
]

EXT_MIN_DWELL_S = 15.0
EXT_STABILITY_WINDOW_S = 5.0
EXT_LEVEL_MAX_TIME_S = 50.0

EXT_VS_MEAN_ABS_MAX_FPS = 0.025
EXT_VS_PEAK_ABS_MAX_FPS = 0.050
EXT_ALT_RANGE_MAX_FT = 0.15

# Still explicitly PRE-touchdown.
EXT_MIN_AGL_FT = 9.5
EXT_MAX_DOWN_VS_FPS = -0.50


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ext_safety_reason(state):
    if state["altitude_ft"] < EXT_MIN_AGL_FT:
        return "extension_agl_floor"
    if state["vertical_speed_fps"] < EXT_MAX_DOWN_VS_FPS:
        return "extension_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "extension_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "extension_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "extension_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "extension_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "extension_roll_limit"
    return ""


def robust_equilibrium(window_rows):
    if not window_rows:
        return False, {}

    vs = np.asarray(
        [r["vertical_speed_fps"] for r in window_rows],
        dtype=float,
    )
    alt = np.asarray(
        [r["altitude_ft"] for r in window_rows],
        dtype=float,
    )

    stats = {
        "vs_mean_fps": float(np.mean(vs)),
        "vs_abs_peak_fps": float(np.max(np.abs(vs))),
        "altitude_mean_ft": float(np.mean(alt)),
        "altitude_range_ft": float(np.max(alt) - np.min(alt)),
    }

    ok = bool(
        abs(stats["vs_mean_fps"]) <= EXT_VS_MEAN_ABS_MAX_FPS
        and stats["vs_abs_peak_fps"] <= EXT_VS_PEAK_ABS_MAX_FPS
        and stats["altitude_range_ft"] <= EXT_ALT_RANGE_MAX_FT
    )
    return ok, stats


def hold_collective_to_equilibrium(
    env2,
    fdm,
    lat0,
    lon0,
    mission_heading,
    dt,
    hover_action0,
    vertical_bias,
    physical_collective,
    phase,
    level_index,
    trace_rows,
):
    window_steps = max(
        2,
        int(round(EXT_STABILITY_WINDOW_S / dt)),
    )
    window = []

    min_alt = +999.0
    min_vs = +999.0
    max_abs_pos = 0.0
    max_abs_cross = 0.0

    for step in range(int(EXT_LEVEL_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=LOW_ALT_TARGET_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        state, used, mapped_elev, requested_collective = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=physical_collective,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )

        t = (step + 1) * dt

        min_alt = min(min_alt, state["altitude_ft"])
        min_vs = min(min_vs, state["vertical_speed_fps"])
        max_abs_pos = max(
            max_abs_pos,
            abs(state["position_error_ft"]),
        )
        max_abs_cross = max(
            max_abs_cross,
            abs(state["cross_track_ft"]),
        )

        window.append({
            "vertical_speed_fps": float(
                state["vertical_speed_fps"]
            ),
            "altitude_ft": float(state["altitude_ft"]),
        })
        if len(window) > window_steps:
            window.pop(0)

        stable_now = False
        stats = {}
        if (
            t >= EXT_MIN_DWELL_S
            and len(window) >= window_steps
        ):
            stable_now, stats = robust_equilibrium(window)

        trace_rows.append({
            "phase": str(phase),
            "level_index": int(level_index),
            "time_in_level_s": float(t),
            "requested_physical_collective": float(
                requested_collective
            ),
            "actual_physical_collective_cmd": float(
                state["physical_collective_cmd"]
            ),
            "mapped_physical_elevator": float(mapped_elev),
            "stable_window": int(stable_now),
            "window_vs_mean_fps": float(
                stats.get("vs_mean_fps", float("nan"))
            ),
            "window_vs_abs_peak_fps": float(
                stats.get("vs_abs_peak_fps", float("nan"))
            ),
            "window_altitude_range_ft": float(
                stats.get("altitude_range_ft", float("nan"))
            ),
            **{k: float(v) for k, v in state.items()},
        })

        reason = ext_safety_reason(state)
        if reason:
            return {
                "safe": False,
                "settled": False,
                "termination": reason,
                "vertical_bias": float(vertical_bias),
                "final_state": state,
                "eq_stats": {},
                "min_altitude_ft": float(min_alt),
                "min_vertical_speed_fps": float(min_vs),
                "max_abs_position_error_ft": float(max_abs_pos),
                "max_abs_cross_track_ft": float(max_abs_cross),
            }

        if any_wow(contact_snapshot(fdm)):
            return {
                "safe": False,
                "settled": False,
                "termination": "unexpected_wow",
                "vertical_bias": float(vertical_bias),
                "final_state": state,
                "eq_stats": {},
                "min_altitude_ft": float(min_alt),
                "min_vertical_speed_fps": float(min_vs),
                "max_abs_position_error_ft": float(max_abs_pos),
                "max_abs_cross_track_ft": float(max_abs_cross),
            }

        if stable_now:
            return {
                "safe": True,
                "settled": True,
                "termination": "equilibrium_confirmed",
                "vertical_bias": float(vertical_bias),
                "final_state": state,
                "eq_stats": stats,
                "min_altitude_ft": float(min_alt),
                "min_vertical_speed_fps": float(min_vs),
                "max_abs_position_error_ft": float(max_abs_pos),
                "max_abs_cross_track_ft": float(max_abs_cross),
            }

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    return {
        "safe": True,
        "settled": False,
        "termination": "unsettled_level",
        "vertical_bias": float(vertical_bias),
        "final_state": final_state,
        "eq_stats": {},
        "min_altitude_ft": float(min_alt),
        "min_vertical_speed_fps": float(min_vs),
        "max_abs_position_error_ft": float(max_abs_pos),
        "max_abs_cross_track_ft": float(max_abs_cross),
    }



# =====================================================================
# STAGE-4 9-FT CAPTURE CALIBRATION V1
# =====================================================================

# IMPORTANT:
# V4 directly confirmed the ground-effect curve only down to ~10.17 ft.
# 9.0 ft is therefore NOT inside the measured interpolation domain.
# This controller uses:
#   - exact measured interpolation inside the confirmed domain,
#   - bounded LOCAL extrapolation below the last confirmed point,
#   - vertical-speed feedback to drive/capture 9 ft,
#   - unchanged near-ground longitudinal/lateral control.
#
# No Stage-4 PPO training is performed here.

CURVE_ANCHOR_CSV = Path(
    "results_stage4_ground_effect_curve_extension_v4/anchor_reproduction.csv"
)
CURVE_EXTENSION_CSV = Path(
    "results_stage4_ground_effect_curve_extension_v4/final_curve_extension.csv"
)

CAPTURE_TARGET_ALT_FT = 9.0
CAPTURE_ALT_TOL_FT = 0.50
CAPTURE_VS_TOL_FPS = 0.15
CAPTURE_HOLD_SECONDS = 5.0
CAPTURE_MAX_TIME_S = 120.0

CAPTURE_VS_GAIN = 0.10
CAPTURE_MAX_DESCENT_FPS = 0.22
CAPTURE_MAX_CLIMB_FPS = 0.12

# Candidate local-extrapolation slope multipliers and physical VS feedback.
EXTRAP_SLOPE_MULTIPLIERS = [0.85, 1.00, 1.15]
CAPTURE_KVS_GRID = [0.006, 0.010, 0.014]

# Hard physical guardrails for calibration only.
CAPTURE_PHYSICAL_COLL_MIN = 0.5560
CAPTURE_PHYSICAL_COLL_MAX = 0.5910

# Still pre-touchdown.
CAPTURE_MIN_AGL_FT = 8.20
CAPTURE_MAX_DOWN_VS_FPS = -0.45


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_confirmed_curve_points():
    if not CURVE_ANCHOR_CSV.exists():
        raise FileNotFoundError(
            f"Missing confirmed V4 curve file: {CURVE_ANCHOR_CSV}"
        )
    if not CURVE_EXTENSION_CSV.exists():
        raise FileNotFoundError(
            f"Missing confirmed V4 curve file: {CURVE_EXTENSION_CSV}"
        )

    points = []

    for path in [CURVE_ANCHOR_CSV, CURVE_EXTENSION_CSV]:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                safe = str(row.get("safe", "")).lower() == "true"
                settled = str(row.get("settled", "")).lower() == "true"
                if not (safe and settled):
                    continue

                alt = float(row["equilibrium_altitude_mean_ft"])
                coll = float(row["physical_collective"])

                if np.isfinite(alt) and np.isfinite(coll):
                    points.append((alt, coll))

    # Deduplicate by collective; keep exact saved altitude.
    by_coll = {}
    for alt, coll in points:
        by_coll[round(coll, 8)] = (alt, coll)
    points = list(by_coll.values())

    # Sort by altitude ascending for np.interp.
    points.sort(key=lambda x: x[0])

    if len(points) < 8:
        raise RuntimeError(
            f"Too few confirmed curve points loaded: {len(points)}"
        )

    # Confirm expected monotonic relation: higher altitude => higher collective.
    for i in range(len(points) - 1):
        if points[i + 1][0] <= points[i][0]:
            raise RuntimeError("Curve altitude ordering failure.")
        if points[i + 1][1] < points[i][1] - 1e-6:
            raise RuntimeError(
                "Confirmed curve is not monotonic in the expected direction."
            )

    return points


def fit_local_lower_curve_slope(points, n=5):
    """
    Fit physical_collective = slope * altitude + intercept using only the
    lowest n confirmed V4 points. This slope is used ONLY below the measured
    domain and is multiplied by a small calibration factor.
    """
    local = points[: min(n, len(points))]
    x = np.asarray([p[0] for p in local], dtype=float)
    y = np.asarray([p[1] for p in local], dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-12
        else 1.0
    )

    if slope <= 0.0:
        raise RuntimeError(
            f"Unexpected local curve slope: {slope}"
        )

    return float(slope), float(intercept), float(r2)


def feedforward_collective(
    altitude_ft,
    points,
    local_slope,
    slope_multiplier,
):
    """
    Piecewise interpolation within measured V4 domain.
    Bounded local linear extrapolation below the lowest measured altitude.
    """
    alts = np.asarray([p[0] for p in points], dtype=float)
    colls = np.asarray([p[1] for p in points], dtype=float)

    alt = float(altitude_ft)
    lowest_alt = float(alts[0])
    lowest_coll = float(colls[0])
    highest_alt = float(alts[-1])
    highest_coll = float(colls[-1])

    if alt < lowest_alt:
        ff = lowest_coll + (
            float(local_slope)
            * float(slope_multiplier)
            * (alt - lowest_alt)
        )
        mode = "local_extrapolation"
    elif alt > highest_alt:
        ff = highest_coll
        mode = "upper_clamp"
    else:
        ff = float(np.interp(alt, alts, colls))
        mode = "measured_interpolation"

    ff = float(np.clip(
        ff,
        CAPTURE_PHYSICAL_COLL_MIN,
        CAPTURE_PHYSICAL_COLL_MAX,
    ))

    return ff, mode


def capture_safety_reason(state):
    if state["altitude_ft"] < CAPTURE_MIN_AGL_FT:
        return "capture_agl_floor"
    if state["vertical_speed_fps"] < CAPTURE_MAX_DOWN_VS_FPS:
        return "capture_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "capture_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "capture_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "capture_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "capture_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "capture_roll_limit"
    return ""


def capture_hold_now(state):
    return bool(
        abs(state["altitude_ft"] - CAPTURE_TARGET_ALT_FT)
            <= CAPTURE_ALT_TOL_FT
        and abs(state["vertical_speed_fps"])
            <= CAPTURE_VS_TOL_FPS
        and abs(state["position_error_ft"])
            <= PRESENTATION_MAX_POSITION_ERROR_FT
        and abs(state["forward_speed_fps"])
            <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(state["cross_track_ft"])
            <= PRESENTATION_MAX_CROSS_FT
        and abs(state["lateral_speed_fps"])
            <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(state["heading_error_deg"])
            <= STAGE4_HEADING_TOL_DEG
    )


def run_capture_candidate(
    slope_multiplier,
    capture_kvs,
    points,
    local_slope,
    detailed=False,
):
    # Reproduce full locked chain and the stable native A0=-1 low-alt state.
    run = build_native_min_equilibrium()

    env2 = run["env2"]
    fdm = run["fdm"]
    lat0 = run["lat0"]
    lon0 = run["lon0"]
    mission_heading = run["mission_heading"]
    dt = run["dt"]
    hover_action0 = run["hover_action0"]
    vertical_bias = run["vertical_bias"]

    trace = []
    hold_s = 0.0
    safe = True
    termination = "capture_time_limit"
    wow_seen = False

    max_abs_pos = 0.0
    max_abs_cross = 0.0
    min_alt = +999.0
    min_vs = +999.0
    min_coll = +999.0
    max_coll = -999.0
    extrap_steps = 0
    total_steps = 0

    next_print = 0.0

    for step in range(int(CAPTURE_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        # Preserve existing horizontal/lateral controller exactly.
        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        ff_coll, ff_mode = feedforward_collective(
            state_before["altitude_ft"],
            points,
            local_slope,
            slope_multiplier,
        )

        vs_des = CAPTURE_VS_GAIN * (
            CAPTURE_TARGET_ALT_FT
            - state_before["altitude_ft"]
        )
        vs_des = float(np.clip(
            vs_des,
            -CAPTURE_MAX_DESCENT_FPS,
            +CAPTURE_MAX_CLIMB_FPS,
        ))

        vs_error = (
            vs_des - state_before["vertical_speed_fps"]
        )

        physical_collective_cmd = float(np.clip(
            ff_coll + float(capture_kvs) * vs_error,
            CAPTURE_PHYSICAL_COLL_MIN,
            CAPTURE_PHYSICAL_COLL_MAX,
        ))

        state, used, mapped_elev, requested_collective = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=physical_collective_cmd,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )

        t = (step + 1) * dt
        total_steps += 1

        if ff_mode == "local_extrapolation":
            extrap_steps += 1

        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        wow_seen = wow_seen or wow

        hold_s = (
            hold_s + dt
            if capture_hold_now(state)
            else 0.0
        )

        max_abs_pos = max(
            max_abs_pos,
            abs(state["position_error_ft"]),
        )
        max_abs_cross = max(
            max_abs_cross,
            abs(state["cross_track_ft"]),
        )
        min_alt = min(min_alt, state["altitude_ft"])
        min_vs = min(min_vs, state["vertical_speed_fps"])
        min_coll = min(
            min_coll,
            state["physical_collective_cmd"],
        )
        max_coll = max(
            max_coll,
            state["physical_collective_cmd"],
        )

        trace.append({
            "time_s": float(t),
            "slope_multiplier": float(slope_multiplier),
            "capture_kvs": float(capture_kvs),
            "feedforward_mode": str(ff_mode),
            "feedforward_collective": float(ff_coll),
            "vs_des_fps": float(vs_des),
            "vs_error_fps": float(vs_error),
            "requested_physical_collective": float(
                requested_collective
            ),
            "actual_physical_collective_cmd": float(
                state["physical_collective_cmd"]
            ),
            "mapped_physical_elevator": float(mapped_elev),
            "hold_s": float(hold_s),
            "wow_detected": int(wow),
            **{k: float(v) for k, v in state.items()},
        })

        reason = capture_safety_reason(state)
        if reason:
            safe = False
            termination = reason
            break

        if wow:
            safe = False
            termination = "unexpected_wow_before_touchdown_test"
            break

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  t={t:6.2f}s | "
                f"ALT={state['altitude_ft']:6.3f} "
                f"VS={state['vertical_speed_fps']:+6.3f}/"
                f"{vs_des:+5.2f} | "
                f"FWD={state['forward_ft']:7.2f} "
                f"err={state['position_error_ft']:+5.2f} "
                f"V={state['forward_speed_fps']:+6.3f} | "
                f"X={state['cross_track_ft']:+5.2f} "
                f"LAT={state['lateral_speed_fps']:+6.3f} | "
                f"FF={ff_coll:.6f} "
                f"COLL={state['physical_collective_cmd']:.6f} "
                f"mode={ff_mode} | "
                f"HOLD={hold_s:4.2f}s"
            )
            next_print += 5.0

        if hold_s >= CAPTURE_HOLD_SECONDS:
            termination = "stable_9ft_hover_5s"
            break

    final = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    passed = bool(
        safe
        and termination == "stable_9ft_hover_5s"
        and hold_s >= CAPTURE_HOLD_SECONDS
        and not wow_seen
        and abs(final["altitude_ft"] - CAPTURE_TARGET_ALT_FT)
            <= CAPTURE_ALT_TOL_FT
        and abs(final["vertical_speed_fps"])
            <= CAPTURE_VS_TOL_FPS
        and abs(final["position_error_ft"])
            <= PRESENTATION_MAX_POSITION_ERROR_FT
        and abs(final["forward_speed_fps"])
            <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(final["cross_track_ft"])
            <= PRESENTATION_MAX_CROSS_FT
        and abs(final["lateral_speed_fps"])
            <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(final["heading_error_deg"])
            <= STAGE4_HEADING_TOL_DEG
    )

    result = {
        "pass": bool(passed),
        "safe": bool(safe),
        "termination": str(termination),
        "slope_multiplier": float(slope_multiplier),
        "capture_kvs": float(capture_kvs),
        "final_altitude_ft": float(final["altitude_ft"]),
        "final_vertical_speed_fps": float(
            final["vertical_speed_fps"]
        ),
        "final_forward_ft": float(final["forward_ft"]),
        "final_position_error_ft": float(
            final["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final["lateral_speed_fps"]
        ),
        "final_heading_error_deg": float(
            final["heading_error_deg"]
        ),
        "hold_s": float(hold_s),
        "wow_seen": bool(wow_seen),
        "max_abs_position_error_ft": float(max_abs_pos),
        "max_abs_cross_track_ft": float(max_abs_cross),
        "min_altitude_ft": float(min_alt),
        "min_vertical_speed_fps": float(min_vs),
        "physical_collective_min": float(min_coll),
        "physical_collective_max": float(max_coll),
        "extrapolation_fraction": float(
            extrap_steps / max(1, total_steps)
        ),
    }

    close_handoff(run)
    return result, trace


def rank_key(row):
    return (
        0 if row["pass"] else 1,
        0 if row["safe"] else 1,
        abs(row["final_altitude_ft"] - CAPTURE_TARGET_ALT_FT),
        abs(row["final_vertical_speed_fps"]),
        row["max_abs_position_error_ft"],
        row["max_abs_cross_track_ft"],
        abs(row["slope_multiplier"] - 1.0),
        abs(row["capture_kvs"] - 0.010),
    )



# =====================================================================
# STAGE-4 TOUCHDOWN / WOW IDENTIFICATION V1
# =====================================================================

# Locked from the successful 9-ft capture calibration.
TD_SLOPE_MULTIPLIER = 1.15
TD_CAPTURE_KVS = 0.014

# Touchdown-identification descent-rate candidates.  These are deliberately
# below the desired <=0.25 ft/s touchdown rate.
TD_VS_DES_CANDIDATES = [-0.12, -0.18, -0.24]

TD_PRECONTACT_MAX_TIME_S = 50.0
TD_POSTCONTACT_SECONDS = 5.0

# Conservative guards before first contact.
TD_MIN_AGL_NO_CONTACT_FT = 5.50
TD_MAX_DOWN_VS_FPS = -0.35

# Physical collective is allowed to continue the already-calibrated local
# extrapolation, but only through a rate-limited command.
TD_PHYSICAL_COLL_MIN = 0.5350
TD_PHYSICAL_COLL_MAX = 0.5910
TD_COLL_RATE_LIMIT_PER_S = 0.0025

# After first WOW, hold near the measured contact collective with a small
# vertical-speed damping correction; this is identification, not the final
# landing controller.
TD_CONTACT_KVS = 0.010
TD_CONTACT_COLL_BAND = 0.0030

# Stable landed evidence.
TD_LANDED_HOLD_SECONDS = 5.0
TD_LANDED_VS_TOL_FPS = 0.15
TD_LANDED_FWD_SPEED_TOL_FPS = 0.60
TD_LANDED_LAT_SPEED_TOL_FPS = 0.60
TD_LANDED_HEADING_TOL_DEG = 1.0

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def compression_values(contact):
    vals = []
    for key, value in contact.items():
        if key.lower().endswith("/compression-ft"):
            vals.append(float(value))
    return vals


def max_compression(contact):
    vals = compression_values(contact)
    return float(max(vals)) if vals else 0.0


def wow_values(contact):
    vals = []
    for key, value in contact.items():
        if key.lower().endswith("/wow"):
            vals.append(float(value))
    return vals


def build_selected_9ft_capture_open(
    points,
    local_slope,
    detailed=False,
):
    """
    Rebuild full locked chain and reproduce the selected C09 9-ft capture.
    Returns the SAME open FDM after >=5 s stable 9-ft hover.
    """
    run = build_native_min_equilibrium()

    env2 = run["env2"]
    fdm = run["fdm"]
    lat0 = run["lat0"]
    lon0 = run["lon0"]
    mission_heading = run["mission_heading"]
    dt = run["dt"]
    hover_action0 = run["hover_action0"]
    vertical_bias = run["vertical_bias"]

    hold_s = 0.0
    next_print = 0.0
    last_requested_coll = float(
        snapshot(fdm, lat0, lon0, mission_heading)[
            "physical_collective_cmd"
        ]
    )

    for step in range(int(CAPTURE_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        ff_coll, ff_mode = feedforward_collective(
            state_before["altitude_ft"],
            points,
            local_slope,
            TD_SLOPE_MULTIPLIER,
        )

        vs_des = CAPTURE_VS_GAIN * (
            CAPTURE_TARGET_ALT_FT
            - state_before["altitude_ft"]
        )
        vs_des = float(np.clip(
            vs_des,
            -CAPTURE_MAX_DESCENT_FPS,
            +CAPTURE_MAX_CLIMB_FPS,
        ))

        vs_error = (
            vs_des - state_before["vertical_speed_fps"]
        )

        requested = float(np.clip(
            ff_coll + TD_CAPTURE_KVS * vs_error,
            CAPTURE_PHYSICAL_COLL_MIN,
            CAPTURE_PHYSICAL_COLL_MAX,
        ))
        last_requested_coll = requested

        state, used, mapped_elev, requested_collective = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )

        t = (step + 1) * dt

        if any_wow(contact_snapshot(fdm)):
            close_handoff(run)
            raise RuntimeError(
                "Unexpected WOW while reproducing selected 9-ft capture."
            )

        reason = capture_safety_reason(state)
        if reason:
            close_handoff(run)
            raise RuntimeError(
                f"Selected 9-ft capture reproduction failed: {reason}"
            )

        hold_s = (
            hold_s + dt
            if capture_hold_now(state)
            else 0.0
        )

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  capture t={t:6.2f}s | "
                f"ALT={state['altitude_ft']:6.3f} "
                f"VS={state['vertical_speed_fps']:+6.3f} | "
                f"FWDerr={state['position_error_ft']:+5.2f} "
                f"X={state['cross_track_ft']:+5.2f} | "
                f"COLL={state['physical_collective_cmd']:.6f} "
                f"HOLD={hold_s:4.2f}s"
            )
            next_print += 10.0

        if hold_s >= CAPTURE_HOLD_SECONDS:
            final_state = snapshot(
                fdm, lat0, lon0, mission_heading
            )
            return {
                **run,
                "vertical_bias": float(vertical_bias),
                "state": final_state,
                "capture_hold_s": float(hold_s),
                "last_requested_collective": float(
                    last_requested_coll
                ),
            }

    close_handoff(run)
    raise RuntimeError(
        "Could not reproduce the selected 9-ft capture within time limit."
    )


def touchdown_precontact_safety_reason(state):
    if state["altitude_ft"] < TD_MIN_AGL_NO_CONTACT_FT:
        return "no_wow_below_expected_contact_floor"
    if state["vertical_speed_fps"] < TD_MAX_DOWN_VS_FPS:
        return "touchdown_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "touchdown_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "touchdown_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "touchdown_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "touchdown_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "touchdown_roll_limit"
    return ""


def touchdown_landed_now(state, contact):
    return bool(
        any_wow(contact)
        and max_compression(contact) > 0.0
        and abs(state["vertical_speed_fps"])
            <= TD_LANDED_VS_TOL_FPS
        and abs(state["position_error_ft"])
            <= PRESENTATION_MAX_POSITION_ERROR_FT
        and abs(state["forward_speed_fps"])
            <= TD_LANDED_FWD_SPEED_TOL_FPS
        and abs(state["cross_track_ft"])
            <= PRESENTATION_MAX_CROSS_FT
        and abs(state["lateral_speed_fps"])
            <= TD_LANDED_LAT_SPEED_TOL_FPS
        and abs(state["heading_error_deg"])
            <= TD_LANDED_HEADING_TOL_DEG
    )


def rate_limit(value, previous, max_delta):
    lo = float(previous) - float(max_delta)
    hi = float(previous) + float(max_delta)
    return float(np.clip(value, lo, hi))


def run_touchdown_candidate(
    vs_des_touchdown,
    points,
    local_slope,
    detailed=False,
):
    start = build_selected_9ft_capture_open(
        points,
        local_slope,
        detailed=False,
    )

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    trace = []

    initial_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    previous_collective = float(
        initial_state["physical_collective_cmd"]
    )

    first_contact = None
    contact_collective = None
    post_contact_elapsed = 0.0
    landed_hold_s = 0.0
    wow_loss_after_contact_s = 0.0
    max_compression_seen = 0.0
    max_abs_position = abs(
        initial_state["position_error_ft"]
    )
    max_abs_cross = abs(
        initial_state["cross_track_ft"]
    )
    min_altitude = initial_state["altitude_ft"]
    min_vertical_speed = initial_state["vertical_speed_fps"]
    max_abs_pitch_deg = abs(
        math.degrees(initial_state["pitch_rad"])
    )
    max_abs_roll_deg = abs(
        math.degrees(initial_state["roll_rad"])
    )

    safe = True
    termination = "touchdown_time_limit"
    next_print = 0.0

    total_time_limit = (
        TD_PRECONTACT_MAX_TIME_S
        + TD_POSTCONTACT_SECONDS
        + 2.0
    )

    for step in range(int(total_time_limit / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        if first_contact is None:
            ff_coll, ff_mode = feedforward_collective(
                state_before["altitude_ft"],
                points,
                local_slope,
                TD_SLOPE_MULTIPLIER,
            )

            vs_error = (
                float(vs_des_touchdown)
                - state_before["vertical_speed_fps"]
            )

            desired_collective = float(np.clip(
                ff_coll + TD_CAPTURE_KVS * vs_error,
                TD_PHYSICAL_COLL_MIN,
                TD_PHYSICAL_COLL_MAX,
            ))
        else:
            ff_mode = "post_contact_hold"
            vs_error = (
                0.0 - state_before["vertical_speed_fps"]
            )
            desired_collective = float(np.clip(
                float(contact_collective)
                + TD_CONTACT_KVS * vs_error,
                float(contact_collective)
                    - TD_CONTACT_COLL_BAND,
                float(contact_collective)
                    + TD_CONTACT_COLL_BAND,
            ))
            desired_collective = float(np.clip(
                desired_collective,
                TD_PHYSICAL_COLL_MIN,
                TD_PHYSICAL_COLL_MAX,
            ))

        max_step_delta = TD_COLL_RATE_LIMIT_PER_S * dt
        requested_collective = rate_limit(
            desired_collective,
            previous_collective,
            max_step_delta,
        )

        state, used, mapped_elev, applied_collective = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested_collective,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )

        previous_collective = float(applied_collective)
        t = (step + 1) * dt

        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        comp_max = max_compression(contact)
        max_compression_seen = max(
            max_compression_seen,
            comp_max,
        )

        max_abs_position = max(
            max_abs_position,
            abs(state["position_error_ft"]),
        )
        max_abs_cross = max(
            max_abs_cross,
            abs(state["cross_track_ft"]),
        )
        min_altitude = min(
            min_altitude,
            state["altitude_ft"],
        )
        min_vertical_speed = min(
            min_vertical_speed,
            state["vertical_speed_fps"],
        )
        max_abs_pitch_deg = max(
            max_abs_pitch_deg,
            abs(math.degrees(state["pitch_rad"])),
        )
        max_abs_roll_deg = max(
            max_abs_roll_deg,
            abs(math.degrees(state["roll_rad"])),
        )

        if first_contact is None and wow:
            first_contact = {
                "time_s": float(t),
                "altitude_ft": float(state["altitude_ft"]),
                "vertical_speed_fps": float(
                    state["vertical_speed_fps"]
                ),
                "forward_ft": float(state["forward_ft"]),
                "position_error_ft": float(
                    state["position_error_ft"]
                ),
                "forward_speed_fps": float(
                    state["forward_speed_fps"]
                ),
                "cross_track_ft": float(
                    state["cross_track_ft"]
                ),
                "lateral_speed_fps": float(
                    state["lateral_speed_fps"]
                ),
                "heading_error_deg": float(
                    state["heading_error_deg"]
                ),
                "pitch_deg": float(
                    math.degrees(state["pitch_rad"])
                ),
                "roll_deg": float(
                    math.degrees(state["roll_rad"])
                ),
                "physical_collective_cmd": float(
                    state["physical_collective_cmd"]
                ),
                "physical_elevator_cmd": float(
                    state["physical_elevator_cmd"]
                ),
                "max_compression_ft": float(comp_max),
                "contact": {
                    k: float(v)
                    for k, v in contact.items()
                },
            }
            contact_collective = float(
                state["physical_collective_cmd"]
            )
            post_contact_elapsed = 0.0
            landed_hold_s = 0.0

        if first_contact is None:
            reason = touchdown_precontact_safety_reason(
                state
            )
            if reason:
                safe = False
                termination = reason
                break

            if t >= TD_PRECONTACT_MAX_TIME_S:
                safe = False
                termination = "no_wow_within_precontact_time_limit"
                break
        else:
            post_contact_elapsed += dt

            if wow:
                wow_loss_after_contact_s = 0.0
            else:
                wow_loss_after_contact_s += dt

            if touchdown_landed_now(state, contact):
                landed_hold_s += dt
            else:
                landed_hold_s = 0.0

            # A meaningful WOW loss after first contact is treated as bounce.
            if wow_loss_after_contact_s >= 0.30:
                termination = "bounce_wow_loss"
                break

            if (
                post_contact_elapsed
                >= TD_POSTCONTACT_SECONDS
            ):
                termination = (
                    "stable_landed_5s"
                    if landed_hold_s >= TD_LANDED_HOLD_SECONDS
                    else "post_contact_observation_complete"
                )
                break

        trace.append({
            "time_s": float(t),
            "commanded_touchdown_vs_fps": float(
                vs_des_touchdown
            ),
            "phase": (
                "pre_contact"
                if first_contact is None
                else "post_contact"
            ),
            "feedforward_mode": str(ff_mode),
            "desired_physical_collective": float(
                desired_collective
            ),
            "applied_physical_collective": float(
                applied_collective
            ),
            "mapped_physical_elevator": float(mapped_elev),
            "wow_detected": int(wow),
            "max_compression_ft": float(comp_max),
            "landed_hold_s": float(landed_hold_s),
            **{
                f"contact_{k}": float(v)
                for k, v in contact.items()
            },
            **{k: float(v) for k, v in state.items()},
        })

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  t={t:6.2f}s | "
                f"phase={'PRE' if first_contact is None else 'POST'} | "
                f"ALT={state['altitude_ft']:6.3f} "
                f"VS={state['vertical_speed_fps']:+6.3f} | "
                f"FWDerr={state['position_error_ft']:+5.2f} "
                f"X={state['cross_track_ft']:+5.2f} | "
                f"COLL={state['physical_collective_cmd']:.6f} | "
                f"WOW={int(wow)} "
                f"COMP={comp_max:.4f} "
                f"LANDHOLD={landed_hold_s:.2f}s"
            )
            next_print += 2.0

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    final_contact = contact_snapshot(fdm)

    stable_landed = bool(
        first_contact is not None
        and safe
        and termination == "stable_landed_5s"
        and landed_hold_s >= TD_LANDED_HOLD_SECONDS
        and any_wow(final_contact)
        and max_compression(final_contact) > 0.0
    )

    first_contact_vs_ok = bool(
        first_contact is not None
        and abs(first_contact["vertical_speed_fps"]) <= 0.25
    )

    result = {
        "commanded_touchdown_vs_fps": float(
            vs_des_touchdown
        ),
        "safe": bool(safe),
        "termination": str(termination),
        "contact_detected": bool(
            first_contact is not None
        ),
        "stable_landed": bool(stable_landed),
        "first_contact_vs_ok": bool(
            first_contact_vs_ok
        ),
        "first_contact": first_contact,
        "final_altitude_ft": float(
            final_state["altitude_ft"]
        ),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_forward_ft": float(
            final_state["forward_ft"]
        ),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "final_heading_error_deg": float(
            final_state["heading_error_deg"]
        ),
        "final_wow": bool(any_wow(final_contact)),
        "final_max_compression_ft": float(
            max_compression(final_contact)
        ),
        "landed_hold_s": float(landed_hold_s),
        "max_compression_seen_ft": float(
            max_compression_seen
        ),
        "max_abs_position_error_ft": float(
            max_abs_position
        ),
        "max_abs_cross_track_ft": float(
            max_abs_cross
        ),
        "min_altitude_ft": float(min_altitude),
        "min_vertical_speed_fps": float(
            min_vertical_speed
        ),
        "max_abs_pitch_deg": float(
            max_abs_pitch_deg
        ),
        "max_abs_roll_deg": float(
            max_abs_roll_deg
        ),
    }

    close_handoff(start)
    return result, trace


def touchdown_rank_key(row):
    contact = row.get("first_contact") or {}
    contact_vs = abs(
        float(contact.get("vertical_speed_fps", 999.0))
    )
    return (
        0 if row["stable_landed"] else 1,
        0 if row["safe"] else 1,
        0 if row["contact_detected"] else 1,
        0 if row["first_contact_vs_ok"] else 1,
        abs(contact_vs - 0.15),
        row["max_abs_position_error_ft"],
        row["max_abs_cross_track_ft"],
    )



# =====================================================================
# STAGE-4 LEVEL-8 LONG-HOLD TOUCHDOWN DIAGNOSTIC V1
# =====================================================================

# V2 reached reduction=0.008 safely but the level did not satisfy the
# existing equilibrium criterion within 30 s. This diagnostic changes
# only the observation time: it reproduces reductions 0.001..0.008 and
# holds the 0.008 level for up to 120 s. No stronger collective command
# is introduced until we know whether that level settles or reaches WOW.

STAIR_REDUCTIONS = [
    0.001, 0.002, 0.003, 0.004,
    0.005, 0.006, 0.007, 0.008,
]

STAIR_LEVEL_MAX_TIME_S = 120.0
STAIR_MIN_DWELL_S = 8.0
STAIR_STABILITY_WINDOW_S = 3.0

STAIR_EQ_VS_MEAN_ABS_MAX = 0.035
STAIR_EQ_VS_PEAK_ABS_MAX = 0.070
STAIR_EQ_ALT_RANGE_MAX_FT = 0.18

# Pre-contact safety only.  reset00 evidence suggests ground contact is near
# ~6.3 ft AGL, but this run measures it rather than assuming it.
STAIR_NO_WOW_MIN_AGL_FT = 5.20
STAIR_MAX_DOWN_VS_FPS = -0.30

# Smooth physical collective transitions between steps.
STAIR_COLL_RATE_LIMIT_PER_S = 0.0020

# Once first WOW occurs, freeze the contact collective for a short observation
# window.  Final post-contact hold control is calibrated later.
POST_CONTACT_OBSERVE_S = 3.0
BOUNCE_WOW_LOSS_S = 0.30


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def staircase_safety_reason(state):
    if state["altitude_ft"] < STAIR_NO_WOW_MIN_AGL_FT:
        return "no_wow_below_contact_search_floor"
    if state["vertical_speed_fps"] < STAIR_MAX_DOWN_VS_FPS:
        return "staircase_descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "staircase_forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "staircase_cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "staircase_heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "staircase_pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "staircase_roll_limit"
    return ""


def staircase_equilibrium(window):
    if not window:
        return False, {}

    vs = np.asarray(
        [r["vertical_speed_fps"] for r in window],
        dtype=float,
    )
    alt = np.asarray(
        [r["altitude_ft"] for r in window],
        dtype=float,
    )

    stats = {
        "vs_mean_fps": float(np.mean(vs)),
        "vs_abs_peak_fps": float(np.max(np.abs(vs))),
        "altitude_mean_ft": float(np.mean(alt)),
        "altitude_range_ft": float(np.max(alt) - np.min(alt)),
    }

    ok = bool(
        abs(stats["vs_mean_fps"]) <= STAIR_EQ_VS_MEAN_ABS_MAX
        and stats["vs_abs_peak_fps"] <= STAIR_EQ_VS_PEAK_ABS_MAX
        and stats["altitude_range_ft"] <= STAIR_EQ_ALT_RANGE_MAX_FT
    )

    return ok, stats


def run_contact_staircase(points, local_slope, detailed=True):
    start = build_selected_9ft_capture_open(
        points,
        local_slope,
        detailed=False,
    )

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    start_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    start_coll = float(start_state["physical_collective_cmd"])
    previous_coll = float(start_coll)

    trace_rows = []
    level_rows = []

    first_contact = None
    first_contact_level = None
    contact_collective = None

    safe = True
    termination = "staircase_exhausted_without_wow"

    max_abs_pos = abs(start_state["position_error_ft"])
    max_abs_cross = abs(start_state["cross_track_ft"])
    min_alt = start_state["altitude_ft"]
    min_vs = start_state["vertical_speed_fps"]
    max_comp_seen = 0.0

    print(
        "CONTACT STAIRCASE START | "
        f"ALT={start_state['altitude_ft']:.3f} "
        f"VS={start_state['vertical_speed_fps']:+.3f} | "
        f"FWDerr={start_state['position_error_ft']:+.3f} "
        f"X={start_state['cross_track_ft']:+.3f} | "
        f"COLLstart={start_coll:.6f}"
    )

    window_steps = max(
        2,
        int(round(STAIR_STABILITY_WINDOW_S / dt)),
    )

    for level_idx, reduction in enumerate(
        STAIR_REDUCTIONS,
        start=1,
    ):
        target_coll = float(
            np.clip(
                start_coll - float(reduction),
                0.0,
                1.0,
            )
        )

        window = []
        level_term = "level_time_limit"
        eq_stats = {}

        level_min_alt = +999.0
        level_min_vs = +999.0
        level_max_abs_pos = 0.0
        level_max_abs_cross = 0.0

        if detailed:
            print(
                f"Level {level_idx:02d}/{len(STAIR_REDUCTIONS)} | "
                f"targetCOLL={target_coll:.6f} "
                f"(reduction={reduction:.3f})"
            )

        for step in range(int(STAIR_LEVEL_MAX_TIME_S / dt)):
            state_before = snapshot(
                fdm, lat0, lon0, mission_heading
            )

            # Preserve verified horizontal/lateral controller.
            action, ctrl, vertical_bias = (
                build_near_ground_action_low_alt_bias(
                    state_before,
                    target_alt_ft=CAPTURE_TARGET_ALT_FT,
                    hover_action0=hover_action0,
                    vertical_bias=vertical_bias,
                    dt=dt,
                    low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
                )
            )

            max_step_delta = (
                STAIR_COLL_RATE_LIMIT_PER_S * dt
            )
            requested_coll = rate_limit(
                target_coll,
                previous_coll,
                max_step_delta,
            )

            state, used, mapped_elev, applied_coll = (
                raw_physical_collective_cycle(
                    env2,
                    fdm,
                    action,
                    physical_collective_cmd=requested_coll,
                    lat0=lat0,
                    lon0=lon0,
                    mission_heading=mission_heading,
                )
            )
            previous_coll = float(applied_coll)

            t = (step + 1) * dt
            contact = contact_snapshot(fdm)
            wow = any_wow(contact)
            comp = max_compression(contact)

            max_comp_seen = max(max_comp_seen, comp)
            max_abs_pos = max(
                max_abs_pos,
                abs(state["position_error_ft"]),
            )
            max_abs_cross = max(
                max_abs_cross,
                abs(state["cross_track_ft"]),
            )
            min_alt = min(min_alt, state["altitude_ft"])
            min_vs = min(min_vs, state["vertical_speed_fps"])

            level_min_alt = min(
                level_min_alt,
                state["altitude_ft"],
            )
            level_min_vs = min(
                level_min_vs,
                state["vertical_speed_fps"],
            )
            level_max_abs_pos = max(
                level_max_abs_pos,
                abs(state["position_error_ft"]),
            )
            level_max_abs_cross = max(
                level_max_abs_cross,
                abs(state["cross_track_ft"]),
            )

            window.append({
                "vertical_speed_fps": float(
                    state["vertical_speed_fps"]
                ),
                "altitude_ft": float(
                    state["altitude_ft"]
                ),
            })
            if len(window) > window_steps:
                window.pop(0)

            stable_now = False
            stats = {}
            if (
                t >= STAIR_MIN_DWELL_S
                and len(window) >= window_steps
            ):
                stable_now, stats = staircase_equilibrium(
                    window
                )

            trace_rows.append({
                "level_index": int(level_idx),
                "time_in_level_s": float(t),
                "collective_reduction": float(reduction),
                "target_physical_collective": float(
                    target_coll
                ),
                "applied_physical_collective": float(
                    applied_coll
                ),
                "mapped_physical_elevator": float(
                    mapped_elev
                ),
                "wow_detected": int(wow),
                "max_compression_ft": float(comp),
                "stable_window": int(stable_now),
                "window_vs_mean_fps": float(
                    stats.get(
                        "vs_mean_fps",
                        float("nan"),
                    )
                ),
                "window_vs_abs_peak_fps": float(
                    stats.get(
                        "vs_abs_peak_fps",
                        float("nan"),
                    )
                ),
                "window_altitude_range_ft": float(
                    stats.get(
                        "altitude_range_ft",
                        float("nan"),
                    )
                ),
                **{
                    f"contact_{k}": float(v)
                    for k, v in contact.items()
                },
                **{k: float(v) for k, v in state.items()},
            })

            if wow:
                first_contact = {
                    "time_in_level_s": float(t),
                    "level_index": int(level_idx),
                    "collective_reduction": float(
                        reduction
                    ),
                    "altitude_ft": float(
                        state["altitude_ft"]
                    ),
                    "vertical_speed_fps": float(
                        state["vertical_speed_fps"]
                    ),
                    "forward_ft": float(
                        state["forward_ft"]
                    ),
                    "position_error_ft": float(
                        state["position_error_ft"]
                    ),
                    "forward_speed_fps": float(
                        state["forward_speed_fps"]
                    ),
                    "cross_track_ft": float(
                        state["cross_track_ft"]
                    ),
                    "lateral_speed_fps": float(
                        state["lateral_speed_fps"]
                    ),
                    "heading_error_deg": float(
                        state["heading_error_deg"]
                    ),
                    "pitch_deg": float(
                        math.degrees(state["pitch_rad"])
                    ),
                    "roll_deg": float(
                        math.degrees(state["roll_rad"])
                    ),
                    "physical_collective_cmd": float(
                        state["physical_collective_cmd"]
                    ),
                    "physical_elevator_cmd": float(
                        state["physical_elevator_cmd"]
                    ),
                    "max_compression_ft": float(comp),
                    "contact": {
                        k: float(v)
                        for k, v in contact.items()
                    },
                }
                first_contact_level = int(level_idx)
                contact_collective = float(
                    state["physical_collective_cmd"]
                )
                level_term = "first_wow_detected"
                termination = "first_wow_detected"
                break

            reason = staircase_safety_reason(state)
            if reason:
                safe = False
                level_term = reason
                termination = reason
                break

            if stable_now:
                eq_stats = stats
                level_term = "equilibrium_confirmed"
                break

        final_level_state = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        level_rows.append({
            "level_index": int(level_idx),
            "collective_reduction": float(
                reduction
            ),
            "target_physical_collective": float(
                target_coll
            ),
            "termination": str(level_term),
            "safe": bool(safe),
            "first_wow": bool(
                first_contact is not None
                and first_contact_level == level_idx
            ),
            "equilibrium_altitude_mean_ft": float(
                eq_stats.get(
                    "altitude_mean_ft",
                    float("nan"),
                )
            ),
            "equilibrium_vs_mean_fps": float(
                eq_stats.get(
                    "vs_mean_fps",
                    float("nan"),
                )
            ),
            "final_altitude_ft": float(
                final_level_state["altitude_ft"]
            ),
            "final_vertical_speed_fps": float(
                final_level_state["vertical_speed_fps"]
            ),
            "final_position_error_ft": float(
                final_level_state["position_error_ft"]
            ),
            "final_cross_track_ft": float(
                final_level_state["cross_track_ft"]
            ),
            "min_altitude_ft": float(
                level_min_alt
            ),
            "min_vertical_speed_fps": float(
                level_min_vs
            ),
            "max_abs_position_error_ft": float(
                level_max_abs_pos
            ),
            "max_abs_cross_track_ft": float(
                level_max_abs_cross
            ),
        })

        if detailed:
            print(
                f"  TERM={level_term} | "
                f"ALT={final_level_state['altitude_ft']:.3f} "
                f"VS={final_level_state['vertical_speed_fps']:+.3f} | "
                f"FWDerr={final_level_state['position_error_ft']:+.2f} "
                f"X={final_level_state['cross_track_ft']:+.2f} | "
                f"minALT={level_min_alt:.3f} "
                f"minVS={level_min_vs:+.3f}"
            )

        if not safe or first_contact is not None:
            break

        # If a level did not settle and still produced ongoing descent, do not
        # stack a stronger collective reduction on top of it.
        if level_term != "equilibrium_confirmed":
            safe = False
            termination = "unsettled_level_before_next_step"
            break

    # Short fixed-collective contact observation, if contact was found.
    post_rows = []
    wow_loss_s = 0.0
    wow_persistent_s = 0.0
    bounce = False

    if first_contact is not None:
        for step in range(
            int(POST_CONTACT_OBSERVE_S / dt)
        ):
            state_before = snapshot(
                fdm, lat0, lon0, mission_heading
            )

            action, ctrl, vertical_bias = (
                build_near_ground_action_low_alt_bias(
                    state_before,
                    target_alt_ft=CAPTURE_TARGET_ALT_FT,
                    hover_action0=hover_action0,
                    vertical_bias=vertical_bias,
                    dt=dt,
                    low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
                )
            )

            state, used, mapped_elev, applied_coll = (
                raw_physical_collective_cycle(
                    env2,
                    fdm,
                    action,
                    physical_collective_cmd=contact_collective,
                    lat0=lat0,
                    lon0=lon0,
                    mission_heading=mission_heading,
                )
            )

            t = (step + 1) * dt
            contact = contact_snapshot(fdm)
            wow = any_wow(contact)
            comp = max_compression(contact)

            if wow:
                wow_persistent_s += dt
                wow_loss_s = 0.0
            else:
                wow_loss_s += dt
                if wow_loss_s >= BOUNCE_WOW_LOSS_S:
                    bounce = True

            post_rows.append({
                "time_after_contact_s": float(t),
                "wow_detected": int(wow),
                "max_compression_ft": float(comp),
                "applied_physical_collective": float(
                    applied_coll
                ),
                **{
                    f"contact_{k}": float(v)
                    for k, v in contact.items()
                },
                **{k: float(v) for k, v in state.items()},
            })

            if bounce:
                termination = "bounce_after_first_wow"
                break

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    final_contact = contact_snapshot(fdm)

    contact_vs_ok = bool(
        first_contact is not None
        and abs(
            first_contact["vertical_speed_fps"]
        ) <= 0.25
    )

    ready_touchdown_hold = bool(
        safe
        and first_contact is not None
        and contact_vs_ok
    )

    result = {
        "safe": bool(safe),
        "termination": str(termination),
        "contact_detected": bool(
            first_contact is not None
        ),
        "first_contact": first_contact,
        "contact_vs_ok": bool(contact_vs_ok),
        "bounce_detected": bool(bounce),
        "wow_persistent_s": float(
            wow_persistent_s
        ),
        "final_wow": bool(
            any_wow(final_contact)
        ),
        "final_max_compression_ft": float(
            max_compression(final_contact)
        ),
        "final_altitude_ft": float(
            final_state["altitude_ft"]
        ),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "max_abs_position_error_ft": float(
            max_abs_pos
        ),
        "max_abs_cross_track_ft": float(
            max_abs_cross
        ),
        "min_altitude_ft": float(min_alt),
        "min_vertical_speed_fps": float(
            min_vs
        ),
        "max_compression_seen_ft": float(
            max_comp_seen
        ),
        "ready_for_touchdown_hold_calibration": bool(
            ready_touchdown_hold
        ),
    }

    close_handoff(start)
    return result, level_rows, trace_rows, post_rows



# =====================================================================
# STAGE-4 TOUCHDOWN HALF-STEP CONTINUATION V3
# =====================================================================

# Locked evidence entering this diagnostic:
#   9-ft hover controller: slopeMult=1.15, Kvs=0.014
#   reduction=0.008 -> COLL ~= 0.552444
#   confirmed equilibrium near ALT=8.104 ft
#   min observed AGL=7.857 ft
#   WOW=0
#
# This V3 reproduces the already-confirmed 0.001..0.008 staircase, then
# continues only in 0.0005 physical-collective increments until actual
# JSBSim WOW/compression is detected or a safety/equilibrium gate fails.

ANCHOR_REDUCTIONS = [
    0.001, 0.002, 0.003, 0.004,
    0.005, 0.006, 0.007, 0.008,
]

EXTENSION_REDUCTIONS = [
    0.0085, 0.0090, 0.0095, 0.0100,
    0.0105, 0.0110, 0.0115, 0.0120,
    0.0125, 0.0130, 0.0135, 0.0140,
    0.0145, 0.0150, 0.0155, 0.0160,
]

LEVEL_MAX_TIME_S = 120.0
LEVEL_MIN_DWELL_S = 8.0
LEVEL_STABILITY_WINDOW_S = 3.0

EQ_VS_MEAN_ABS_MAX = 0.035
EQ_VS_PEAK_ABS_MAX = 0.070
EQ_ALT_RANGE_MAX_FT = 0.18

# Pre-contact safety. Actual contact is measured from WOW/compression.
NO_WOW_MIN_AGL_FT = 5.20
MAX_DOWN_VS_FPS = -0.30

COLL_RATE_LIMIT_PER_S = 0.0020

POST_CONTACT_OBSERVE_S = 3.0
BOUNCE_WOW_LOSS_S = 0.30


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safety_reason(state):
    if state["altitude_ft"] < NO_WOW_MIN_AGL_FT:
        return "no_wow_below_contact_search_floor"
    if state["vertical_speed_fps"] < MAX_DOWN_VS_FPS:
        return "descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "roll_limit"
    return ""


def equilibrium_ok(window):
    if not window:
        return False, {}

    vs = np.asarray(
        [r["vertical_speed_fps"] for r in window],
        dtype=float,
    )
    alt = np.asarray(
        [r["altitude_ft"] for r in window],
        dtype=float,
    )

    stats = {
        "vs_mean_fps": float(np.mean(vs)),
        "vs_abs_peak_fps": float(np.max(np.abs(vs))),
        "altitude_mean_ft": float(np.mean(alt)),
        "altitude_range_ft": float(np.max(alt) - np.min(alt)),
    }

    ok = bool(
        abs(stats["vs_mean_fps"]) <= EQ_VS_MEAN_ABS_MAX
        and stats["vs_abs_peak_fps"] <= EQ_VS_PEAK_ABS_MAX
        and stats["altitude_range_ft"] <= EQ_ALT_RANGE_MAX_FT
    )
    return ok, stats


def hold_level(
    env2,
    fdm,
    lat0,
    lon0,
    mission_heading,
    dt,
    hover_action0,
    vertical_bias,
    target_coll,
    reduction,
    phase,
    level_index,
    previous_coll,
    trace_rows,
):
    window_steps = max(
        2,
        int(round(LEVEL_STABILITY_WINDOW_S / dt)),
    )
    window = []

    level_min_alt = +999.0
    level_min_vs = +999.0
    level_max_abs_pos = 0.0
    level_max_abs_cross = 0.0

    eq_stats = {}
    termination = "level_time_limit"

    for step in range(int(LEVEL_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        # Preserve the verified horizontal/lateral controller.
        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        max_step_delta = COLL_RATE_LIMIT_PER_S * dt
        requested_coll = rate_limit(
            target_coll,
            previous_coll,
            max_step_delta,
        )

        state, used, mapped_elev, applied_coll = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested_coll,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )
        previous_coll = float(applied_coll)

        t = (step + 1) * dt
        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        comp = max_compression(contact)

        level_min_alt = min(
            level_min_alt,
            state["altitude_ft"],
        )
        level_min_vs = min(
            level_min_vs,
            state["vertical_speed_fps"],
        )
        level_max_abs_pos = max(
            level_max_abs_pos,
            abs(state["position_error_ft"]),
        )
        level_max_abs_cross = max(
            level_max_abs_cross,
            abs(state["cross_track_ft"]),
        )

        window.append({
            "vertical_speed_fps": float(
                state["vertical_speed_fps"]
            ),
            "altitude_ft": float(
                state["altitude_ft"]
            ),
        })
        if len(window) > window_steps:
            window.pop(0)

        stable_now = False
        stats = {}
        if (
            t >= LEVEL_MIN_DWELL_S
            and len(window) >= window_steps
        ):
            stable_now, stats = equilibrium_ok(window)

        trace_rows.append({
            "phase": str(phase),
            "level_index": int(level_index),
            "time_in_level_s": float(t),
            "collective_reduction": float(reduction),
            "target_physical_collective": float(target_coll),
            "applied_physical_collective": float(applied_coll),
            "mapped_physical_elevator": float(mapped_elev),
            "wow_detected": int(wow),
            "max_compression_ft": float(comp),
            "stable_window": int(stable_now),
            "window_vs_mean_fps": float(
                stats.get("vs_mean_fps", float("nan"))
            ),
            "window_vs_abs_peak_fps": float(
                stats.get("vs_abs_peak_fps", float("nan"))
            ),
            "window_altitude_range_ft": float(
                stats.get("altitude_range_ft", float("nan"))
            ),
            **{
                f"contact_{k}": float(v)
                for k, v in contact.items()
            },
            **{k: float(v) for k, v in state.items()},
        })

        if wow:
            first_contact = {
                "phase": str(phase),
                "level_index": int(level_index),
                "time_in_level_s": float(t),
                "collective_reduction": float(reduction),
                "altitude_ft": float(state["altitude_ft"]),
                "vertical_speed_fps": float(
                    state["vertical_speed_fps"]
                ),
                "forward_ft": float(state["forward_ft"]),
                "position_error_ft": float(
                    state["position_error_ft"]
                ),
                "forward_speed_fps": float(
                    state["forward_speed_fps"]
                ),
                "cross_track_ft": float(
                    state["cross_track_ft"]
                ),
                "lateral_speed_fps": float(
                    state["lateral_speed_fps"]
                ),
                "heading_error_deg": float(
                    state["heading_error_deg"]
                ),
                "pitch_deg": float(
                    math.degrees(state["pitch_rad"])
                ),
                "roll_deg": float(
                    math.degrees(state["roll_rad"])
                ),
                "physical_collective_cmd": float(
                    state["physical_collective_cmd"]
                ),
                "physical_elevator_cmd": float(
                    state["physical_elevator_cmd"]
                ),
                "max_compression_ft": float(comp),
                "contact": {
                    k: float(v)
                    for k, v in contact.items()
                },
            }
            return {
                "safe": True,
                "settled": False,
                "termination": "first_wow_detected",
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "final_state": state,
                "eq_stats": {},
                "first_contact": first_contact,
                "min_altitude_ft": float(level_min_alt),
                "min_vertical_speed_fps": float(level_min_vs),
                "max_abs_position_error_ft": float(
                    level_max_abs_pos
                ),
                "max_abs_cross_track_ft": float(
                    level_max_abs_cross
                ),
            }

        reason = safety_reason(state)
        if reason:
            return {
                "safe": False,
                "settled": False,
                "termination": reason,
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "final_state": state,
                "eq_stats": {},
                "first_contact": None,
                "min_altitude_ft": float(level_min_alt),
                "min_vertical_speed_fps": float(level_min_vs),
                "max_abs_position_error_ft": float(
                    level_max_abs_pos
                ),
                "max_abs_cross_track_ft": float(
                    level_max_abs_cross
                ),
            }

        if stable_now:
            eq_stats = stats
            return {
                "safe": True,
                "settled": True,
                "termination": "equilibrium_confirmed",
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "final_state": state,
                "eq_stats": eq_stats,
                "first_contact": None,
                "min_altitude_ft": float(level_min_alt),
                "min_vertical_speed_fps": float(level_min_vs),
                "max_abs_position_error_ft": float(
                    level_max_abs_pos
                ),
                "max_abs_cross_track_ft": float(
                    level_max_abs_cross
                ),
            }

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    return {
        "safe": True,
        "settled": False,
        "termination": "unsettled_level",
        "vertical_bias": float(vertical_bias),
        "previous_coll": float(previous_coll),
        "final_state": final_state,
        "eq_stats": {},
        "first_contact": None,
        "min_altitude_ft": float(level_min_alt),
        "min_vertical_speed_fps": float(level_min_vs),
        "max_abs_position_error_ft": float(
            level_max_abs_pos
        ),
        "max_abs_cross_track_ft": float(
            level_max_abs_cross
        ),
    }


def run_halfstep_search(points, local_slope):
    start = build_selected_9ft_capture_open(
        points,
        local_slope,
        detailed=False,
    )

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    state0 = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    start_coll = float(state0["physical_collective_cmd"])
    previous_coll = float(start_coll)

    anchor_rows = []
    extension_rows = []
    trace_rows = []
    post_rows = []

    first_contact = None
    contact_collective = None
    overall_safe = True
    termination = "extension_exhausted_without_wow"

    print(
        "START | "
        f"ALT={state0['altitude_ft']:.3f} "
        f"VS={state0['vertical_speed_fps']:+.3f} | "
        f"FWDerr={state0['position_error_ft']:+.3f} "
        f"X={state0['cross_track_ft']:+.3f} | "
        f"COLLstart={start_coll:.6f}"
    )

    # Reproduce locked 0.001..0.008 staircase.
    for idx, reduction in enumerate(
        ANCHOR_REDUCTIONS,
        start=1,
    ):
        target_coll = start_coll - reduction
        result = hold_level(
            env2, fdm, lat0, lon0, mission_heading,
            dt, hover_action0, vertical_bias,
            target_coll, reduction, "anchor", idx,
            previous_coll, trace_rows,
        )

        vertical_bias = result["vertical_bias"]
        previous_coll = result["previous_coll"]
        s = result["final_state"]
        eq = result["eq_stats"]

        anchor_rows.append({
            "phase": "anchor",
            "level_index": int(idx),
            "collective_reduction": float(reduction),
            "target_physical_collective": float(target_coll),
            "safe": bool(result["safe"]),
            "settled": bool(result["settled"]),
            "termination": str(result["termination"]),
            "equilibrium_altitude_mean_ft": float(
                eq.get("altitude_mean_ft", float("nan"))
            ),
            "equilibrium_vs_mean_fps": float(
                eq.get("vs_mean_fps", float("nan"))
            ),
            "final_altitude_ft": float(s["altitude_ft"]),
            "final_vertical_speed_fps": float(
                s["vertical_speed_fps"]
            ),
            "final_position_error_ft": float(
                s["position_error_ft"]
            ),
            "final_cross_track_ft": float(
                s["cross_track_ft"]
            ),
            "min_altitude_ft": float(
                result["min_altitude_ft"]
            ),
            "min_vertical_speed_fps": float(
                result["min_vertical_speed_fps"]
            ),
        })

        print(
            f"Anchor {idx:02d}/08 red={reduction:.4f} | "
            f"SETTLED={result['settled']} SAFE={result['safe']} "
            f"TERM={result['termination']} | "
            f"EQALT={anchor_rows[-1]['equilibrium_altitude_mean_ft']:.3f} "
            f"EQVS={anchor_rows[-1]['equilibrium_vs_mean_fps']:+.3f} | "
            f"minALT={result['min_altitude_ft']:.3f} "
            f"minVS={result['min_vertical_speed_fps']:+.3f}"
        )

        if (
            not result["safe"]
            or not result["settled"]
            or result["first_contact"] is not None
        ):
            overall_safe = bool(result["safe"])
            termination = result["termination"]
            first_contact = result["first_contact"]
            contact_collective = (
                None
                if first_contact is None
                else first_contact["physical_collective_cmd"]
            )
            break

    if first_contact is None and overall_safe and len(anchor_rows) == len(ANCHOR_REDUCTIONS):
        print()
        print("HALF-STEP EXTENSION:")

        for idx, reduction in enumerate(
            EXTENSION_REDUCTIONS,
            start=1,
        ):
            target_coll = start_coll - reduction
            result = hold_level(
                env2, fdm, lat0, lon0, mission_heading,
                dt, hover_action0, vertical_bias,
                target_coll, reduction, "extension", idx,
                previous_coll, trace_rows,
            )

            vertical_bias = result["vertical_bias"]
            previous_coll = result["previous_coll"]
            s = result["final_state"]
            eq = result["eq_stats"]

            row = {
                "phase": "extension",
                "level_index": int(idx),
                "collective_reduction": float(reduction),
                "target_physical_collective": float(target_coll),
                "safe": bool(result["safe"]),
                "settled": bool(result["settled"]),
                "termination": str(result["termination"]),
                "equilibrium_altitude_mean_ft": float(
                    eq.get("altitude_mean_ft", float("nan"))
                ),
                "equilibrium_vs_mean_fps": float(
                    eq.get("vs_mean_fps", float("nan"))
                ),
                "final_altitude_ft": float(s["altitude_ft"]),
                "final_vertical_speed_fps": float(
                    s["vertical_speed_fps"]
                ),
                "final_position_error_ft": float(
                    s["position_error_ft"]
                ),
                "final_cross_track_ft": float(
                    s["cross_track_ft"]
                ),
                "min_altitude_ft": float(
                    result["min_altitude_ft"]
                ),
                "min_vertical_speed_fps": float(
                    result["min_vertical_speed_fps"]
                ),
            }
            extension_rows.append(row)

            print(
                f"Level {idx:02d}/{len(EXTENSION_REDUCTIONS)} "
                f"red={reduction:.4f} | "
                f"SETTLED={result['settled']} SAFE={result['safe']} "
                f"TERM={result['termination']} | "
                f"EQALT={row['equilibrium_altitude_mean_ft']:.3f} "
                f"EQVS={row['equilibrium_vs_mean_fps']:+.3f} | "
                f"finalALT={row['final_altitude_ft']:.3f} "
                f"finalVS={row['final_vertical_speed_fps']:+.3f} | "
                f"minALT={row['min_altitude_ft']:.3f} "
                f"minVS={row['min_vertical_speed_fps']:+.3f}"
            )

            if result["first_contact"] is not None:
                first_contact = result["first_contact"]
                contact_collective = float(
                    first_contact["physical_collective_cmd"]
                )
                termination = "first_wow_detected"
                break

            if not result["safe"]:
                overall_safe = False
                termination = result["termination"]
                break

            if not result["settled"]:
                termination = "unsettled_extension_level"
                break

    # Short observation after first WOW, no final landing controller yet.
    bounce = False
    wow_loss_s = 0.0
    wow_persistent_s = 0.0

    if first_contact is not None:
        for step in range(int(POST_CONTACT_OBSERVE_S / dt)):
            state_before = snapshot(
                fdm, lat0, lon0, mission_heading
            )

            action, ctrl, vertical_bias = (
                build_near_ground_action_low_alt_bias(
                    state_before,
                    target_alt_ft=CAPTURE_TARGET_ALT_FT,
                    hover_action0=hover_action0,
                    vertical_bias=vertical_bias,
                    dt=dt,
                    low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
                )
            )

            state, used, mapped_elev, applied_coll = (
                raw_physical_collective_cycle(
                    env2,
                    fdm,
                    action,
                    physical_collective_cmd=contact_collective,
                    lat0=lat0,
                    lon0=lon0,
                    mission_heading=mission_heading,
                )
            )

            t = (step + 1) * dt
            contact = contact_snapshot(fdm)
            wow = any_wow(contact)
            comp = max_compression(contact)

            if wow:
                wow_persistent_s += dt
                wow_loss_s = 0.0
            else:
                wow_loss_s += dt
                if wow_loss_s >= BOUNCE_WOW_LOSS_S:
                    bounce = True

            post_rows.append({
                "time_after_contact_s": float(t),
                "wow_detected": int(wow),
                "max_compression_ft": float(comp),
                "applied_physical_collective": float(
                    applied_coll
                ),
                **{
                    f"contact_{k}": float(v)
                    for k, v in contact.items()
                },
                **{k: float(v) for k, v in state.items()},
            })

            if bounce:
                termination = "bounce_after_first_wow"
                break

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    final_contact = contact_snapshot(fdm)

    ready_hold = bool(
        overall_safe
        and first_contact is not None
        and abs(first_contact["vertical_speed_fps"]) <= 0.25
    )

    result = {
        "safe": bool(overall_safe),
        "termination": str(termination),
        "contact_detected": bool(first_contact is not None),
        "first_contact": first_contact,
        "bounce_detected": bool(bounce),
        "wow_persistent_s": float(wow_persistent_s),
        "final_wow": bool(any_wow(final_contact)),
        "final_max_compression_ft": float(
            max_compression(final_contact)
        ),
        "final_altitude_ft": float(
            final_state["altitude_ft"]
        ),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "ready_for_touchdown_hold_calibration": bool(
            ready_hold
        ),
    }

    close_handoff(start)
    return (
        result,
        anchor_rows,
        extension_rows,
        trace_rows,
        post_rows,
    )



# =====================================================================
# STAGE-4 TOUCHDOWN QUARTER-STEP CONTINUATION V4
# =====================================================================

# Locked evidence entering this diagnostic:
#   9-ft hover controller: slopeMult=1.15, Kvs=0.014
#   reduction=0.0160 -> physical collective ~= 0.544444
#   confirmed equilibrium near ALT=7.071 ft
#   min observed AGL=7.062 ft
#   WOW=0
#
# V4 reproduces every previously confirmed level, then continues in 0.00025
# physical-collective reduction increments. This is only contact identification.
# No Stage-4 PPO training and no final landed-hold controller here.

ANCHOR_REDUCTIONS = [
    0.0010, 0.0020, 0.0030, 0.0040,
    0.0050, 0.0060, 0.0070, 0.0080,
    0.0085, 0.0090, 0.0095, 0.0100,
    0.0105, 0.0110, 0.0115, 0.0120,
    0.0125, 0.0130, 0.0135, 0.0140,
    0.0145, 0.0150, 0.0155, 0.0160,
]

EXTENSION_REDUCTIONS = [
    0.01625, 0.01650, 0.01675, 0.01700,
    0.01725, 0.01750, 0.01775, 0.01800,
    0.01825, 0.01850, 0.01875, 0.01900,
    0.01925, 0.01950, 0.01975, 0.02000,
]

LEVEL_MAX_TIME_S = 120.0
LEVEL_MIN_DWELL_S = 8.0
LEVEL_STABILITY_WINDOW_S = 3.0

EQ_VS_MEAN_ABS_MAX = 0.035
EQ_VS_PEAK_ABS_MAX = 0.070
EQ_ALT_RANGE_MAX_FT = 0.18

NO_WOW_MIN_AGL_FT = 5.20
MAX_DOWN_VS_FPS = -0.30

COLL_RATE_LIMIT_PER_S = 0.0020

POST_CONTACT_OBSERVE_S = 3.0
BOUNCE_WOW_LOSS_S = 0.30


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safety_reason(state):
    if state["altitude_ft"] < NO_WOW_MIN_AGL_FT:
        return "no_wow_below_contact_search_floor"
    if state["vertical_speed_fps"] < MAX_DOWN_VS_FPS:
        return "descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "roll_limit"
    return ""


def equilibrium_ok(window):
    if not window:
        return False, {}

    vs = np.asarray(
        [r["vertical_speed_fps"] for r in window],
        dtype=float,
    )
    alt = np.asarray(
        [r["altitude_ft"] for r in window],
        dtype=float,
    )

    stats = {
        "vs_mean_fps": float(np.mean(vs)),
        "vs_abs_peak_fps": float(np.max(np.abs(vs))),
        "altitude_mean_ft": float(np.mean(alt)),
        "altitude_range_ft": float(np.max(alt) - np.min(alt)),
    }

    ok = bool(
        abs(stats["vs_mean_fps"]) <= EQ_VS_MEAN_ABS_MAX
        and stats["vs_abs_peak_fps"] <= EQ_VS_PEAK_ABS_MAX
        and stats["altitude_range_ft"] <= EQ_ALT_RANGE_MAX_FT
    )
    return ok, stats


def hold_level(
    env2,
    fdm,
    lat0,
    lon0,
    mission_heading,
    dt,
    hover_action0,
    vertical_bias,
    target_coll,
    reduction,
    phase,
    level_index,
    previous_coll,
    trace_rows,
):
    window_steps = max(
        2,
        int(round(LEVEL_STABILITY_WINDOW_S / dt)),
    )
    window = []

    level_min_alt = +999.0
    level_min_vs = +999.0
    level_max_abs_pos = 0.0
    level_max_abs_cross = 0.0

    for step in range(int(LEVEL_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        max_step_delta = COLL_RATE_LIMIT_PER_S * dt
        requested_coll = rate_limit(
            target_coll,
            previous_coll,
            max_step_delta,
        )

        state, used, mapped_elev, applied_coll = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested_coll,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )
        previous_coll = float(applied_coll)

        t = (step + 1) * dt
        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        comp = max_compression(contact)

        level_min_alt = min(level_min_alt, state["altitude_ft"])
        level_min_vs = min(level_min_vs, state["vertical_speed_fps"])
        level_max_abs_pos = max(
            level_max_abs_pos, abs(state["position_error_ft"])
        )
        level_max_abs_cross = max(
            level_max_abs_cross, abs(state["cross_track_ft"])
        )

        window.append({
            "vertical_speed_fps": float(state["vertical_speed_fps"]),
            "altitude_ft": float(state["altitude_ft"]),
        })
        if len(window) > window_steps:
            window.pop(0)

        stable_now = False
        stats = {}
        if t >= LEVEL_MIN_DWELL_S and len(window) >= window_steps:
            stable_now, stats = equilibrium_ok(window)

        trace_rows.append({
            "phase": str(phase),
            "level_index": int(level_index),
            "time_in_level_s": float(t),
            "collective_reduction": float(reduction),
            "target_physical_collective": float(target_coll),
            "applied_physical_collective": float(applied_coll),
            "mapped_physical_elevator": float(mapped_elev),
            "wow_detected": int(wow),
            "max_compression_ft": float(comp),
            "stable_window": int(stable_now),
            "window_vs_mean_fps": float(
                stats.get("vs_mean_fps", float("nan"))
            ),
            "window_vs_abs_peak_fps": float(
                stats.get("vs_abs_peak_fps", float("nan"))
            ),
            "window_altitude_range_ft": float(
                stats.get("altitude_range_ft", float("nan"))
            ),
            **{
                f"contact_{k}": float(v)
                for k, v in contact.items()
            },
            **{k: float(v) for k, v in state.items()},
        })

        if wow:
            return {
                "safe": True,
                "settled": False,
                "termination": "first_wow_detected",
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "final_state": state,
                "eq_stats": {},
                "first_contact": {
                    "phase": str(phase),
                    "level_index": int(level_index),
                    "time_in_level_s": float(t),
                    "collective_reduction": float(reduction),
                    "altitude_ft": float(state["altitude_ft"]),
                    "vertical_speed_fps": float(
                        state["vertical_speed_fps"]
                    ),
                    "forward_ft": float(state["forward_ft"]),
                    "position_error_ft": float(
                        state["position_error_ft"]
                    ),
                    "forward_speed_fps": float(
                        state["forward_speed_fps"]
                    ),
                    "cross_track_ft": float(
                        state["cross_track_ft"]
                    ),
                    "lateral_speed_fps": float(
                        state["lateral_speed_fps"]
                    ),
                    "heading_error_deg": float(
                        state["heading_error_deg"]
                    ),
                    "pitch_deg": float(
                        math.degrees(state["pitch_rad"])
                    ),
                    "roll_deg": float(
                        math.degrees(state["roll_rad"])
                    ),
                    "physical_collective_cmd": float(
                        state["physical_collective_cmd"]
                    ),
                    "physical_elevator_cmd": float(
                        state["physical_elevator_cmd"]
                    ),
                    "max_compression_ft": float(comp),
                    "contact": {
                        k: float(v)
                        for k, v in contact.items()
                    },
                },
                "min_altitude_ft": float(level_min_alt),
                "min_vertical_speed_fps": float(level_min_vs),
                "max_abs_position_error_ft": float(
                    level_max_abs_pos
                ),
                "max_abs_cross_track_ft": float(
                    level_max_abs_cross
                ),
            }

        reason = safety_reason(state)
        if reason:
            return {
                "safe": False,
                "settled": False,
                "termination": reason,
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "final_state": state,
                "eq_stats": {},
                "first_contact": None,
                "min_altitude_ft": float(level_min_alt),
                "min_vertical_speed_fps": float(level_min_vs),
                "max_abs_position_error_ft": float(
                    level_max_abs_pos
                ),
                "max_abs_cross_track_ft": float(
                    level_max_abs_cross
                ),
            }

        if stable_now:
            return {
                "safe": True,
                "settled": True,
                "termination": "equilibrium_confirmed",
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "final_state": state,
                "eq_stats": stats,
                "first_contact": None,
                "min_altitude_ft": float(level_min_alt),
                "min_vertical_speed_fps": float(level_min_vs),
                "max_abs_position_error_ft": float(
                    level_max_abs_pos
                ),
                "max_abs_cross_track_ft": float(
                    level_max_abs_cross
                ),
            }

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    return {
        "safe": True,
        "settled": False,
        "termination": "unsettled_level",
        "vertical_bias": float(vertical_bias),
        "previous_coll": float(previous_coll),
        "final_state": final_state,
        "eq_stats": {},
        "first_contact": None,
        "min_altitude_ft": float(level_min_alt),
        "min_vertical_speed_fps": float(level_min_vs),
        "max_abs_position_error_ft": float(
            level_max_abs_pos
        ),
        "max_abs_cross_track_ft": float(
            level_max_abs_cross
        ),
    }


def run_quarterstep_search(points, local_slope):
    start = build_selected_9ft_capture_open(
        points,
        local_slope,
        detailed=False,
    )

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    state0 = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    start_coll = float(state0["physical_collective_cmd"])
    previous_coll = float(start_coll)

    anchor_rows = []
    extension_rows = []
    trace_rows = []
    post_rows = []

    first_contact = None
    contact_collective = None
    safe = True
    termination = "extension_exhausted_without_wow"

    print(
        "START | "
        f"ALT={state0['altitude_ft']:.3f} "
        f"VS={state0['vertical_speed_fps']:+.3f} | "
        f"FWDerr={state0['position_error_ft']:+.3f} "
        f"X={state0['cross_track_ft']:+.3f} | "
        f"COLLstart={start_coll:.6f}"
    )

    # Reproduce all previously confirmed points.
    for idx, reduction in enumerate(ANCHOR_REDUCTIONS, start=1):
        target_coll = start_coll - reduction

        result = hold_level(
            env2, fdm, lat0, lon0, mission_heading,
            dt, hover_action0, vertical_bias,
            target_coll, reduction, "anchor", idx,
            previous_coll, trace_rows,
        )

        vertical_bias = result["vertical_bias"]
        previous_coll = result["previous_coll"]
        s = result["final_state"]
        eq = result["eq_stats"]

        row = {
            "phase": "anchor",
            "level_index": int(idx),
            "collective_reduction": float(reduction),
            "target_physical_collective": float(target_coll),
            "safe": bool(result["safe"]),
            "settled": bool(result["settled"]),
            "termination": str(result["termination"]),
            "equilibrium_altitude_mean_ft": float(
                eq.get("altitude_mean_ft", float("nan"))
            ),
            "equilibrium_vs_mean_fps": float(
                eq.get("vs_mean_fps", float("nan"))
            ),
            "final_altitude_ft": float(s["altitude_ft"]),
            "final_vertical_speed_fps": float(
                s["vertical_speed_fps"]
            ),
            "final_position_error_ft": float(
                s["position_error_ft"]
            ),
            "final_cross_track_ft": float(
                s["cross_track_ft"]
            ),
            "min_altitude_ft": float(
                result["min_altitude_ft"]
            ),
            "min_vertical_speed_fps": float(
                result["min_vertical_speed_fps"]
            ),
        }
        anchor_rows.append(row)

        print(
            f"Anchor {idx:02d}/{len(ANCHOR_REDUCTIONS)} "
            f"red={reduction:.5f} | "
            f"SETTLED={result['settled']} SAFE={result['safe']} "
            f"TERM={result['termination']} | "
            f"EQALT={row['equilibrium_altitude_mean_ft']:.3f} "
            f"EQVS={row['equilibrium_vs_mean_fps']:+.3f} | "
            f"minALT={row['min_altitude_ft']:.3f} "
            f"minVS={row['min_vertical_speed_fps']:+.3f}"
        )

        if result["first_contact"] is not None:
            first_contact = result["first_contact"]
            contact_collective = float(
                first_contact["physical_collective_cmd"]
            )
            termination = "first_wow_detected_during_anchor_reproduction"
            break

        if not result["safe"]:
            safe = False
            termination = result["termination"]
            break

        if not result["settled"]:
            safe = False
            termination = "anchor_reproduction_unsettled"
            break

    if (
        safe
        and first_contact is None
        and len(anchor_rows) == len(ANCHOR_REDUCTIONS)
    ):
        print()
        print("QUARTER-STEP EXTENSION:")

        for idx, reduction in enumerate(
            EXTENSION_REDUCTIONS,
            start=1,
        ):
            target_coll = start_coll - reduction

            result = hold_level(
                env2, fdm, lat0, lon0, mission_heading,
                dt, hover_action0, vertical_bias,
                target_coll, reduction, "extension", idx,
                previous_coll, trace_rows,
            )

            vertical_bias = result["vertical_bias"]
            previous_coll = result["previous_coll"]
            s = result["final_state"]
            eq = result["eq_stats"]

            row = {
                "phase": "extension",
                "level_index": int(idx),
                "collective_reduction": float(reduction),
                "target_physical_collective": float(target_coll),
                "safe": bool(result["safe"]),
                "settled": bool(result["settled"]),
                "termination": str(result["termination"]),
                "equilibrium_altitude_mean_ft": float(
                    eq.get("altitude_mean_ft", float("nan"))
                ),
                "equilibrium_vs_mean_fps": float(
                    eq.get("vs_mean_fps", float("nan"))
                ),
                "final_altitude_ft": float(s["altitude_ft"]),
                "final_vertical_speed_fps": float(
                    s["vertical_speed_fps"]
                ),
                "final_position_error_ft": float(
                    s["position_error_ft"]
                ),
                "final_cross_track_ft": float(
                    s["cross_track_ft"]
                ),
                "min_altitude_ft": float(
                    result["min_altitude_ft"]
                ),
                "min_vertical_speed_fps": float(
                    result["min_vertical_speed_fps"]
                ),
            }
            extension_rows.append(row)

            print(
                f"Level {idx:02d}/{len(EXTENSION_REDUCTIONS)} "
                f"red={reduction:.5f} | "
                f"SETTLED={result['settled']} SAFE={result['safe']} "
                f"TERM={result['termination']} | "
                f"EQALT={row['equilibrium_altitude_mean_ft']:.3f} "
                f"EQVS={row['equilibrium_vs_mean_fps']:+.3f} | "
                f"finalALT={row['final_altitude_ft']:.3f} "
                f"finalVS={row['final_vertical_speed_fps']:+.3f} | "
                f"minALT={row['min_altitude_ft']:.3f} "
                f"minVS={row['min_vertical_speed_fps']:+.3f}"
            )

            if result["first_contact"] is not None:
                first_contact = result["first_contact"]
                contact_collective = float(
                    first_contact["physical_collective_cmd"]
                )
                termination = "first_wow_detected"
                break

            if not result["safe"]:
                safe = False
                termination = result["termination"]
                break

            if not result["settled"]:
                termination = "unsettled_extension_level"
                break

    bounce = False
    wow_loss_s = 0.0
    wow_persistent_s = 0.0

    if first_contact is not None:
        for step in range(int(POST_CONTACT_OBSERVE_S / dt)):
            state_before = snapshot(
                fdm, lat0, lon0, mission_heading
            )

            action, ctrl, vertical_bias = (
                build_near_ground_action_low_alt_bias(
                    state_before,
                    target_alt_ft=CAPTURE_TARGET_ALT_FT,
                    hover_action0=hover_action0,
                    vertical_bias=vertical_bias,
                    dt=dt,
                    low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
                )
            )

            state, used, mapped_elev, applied_coll = (
                raw_physical_collective_cycle(
                    env2,
                    fdm,
                    action,
                    physical_collective_cmd=contact_collective,
                    lat0=lat0,
                    lon0=lon0,
                    mission_heading=mission_heading,
                )
            )

            t = (step + 1) * dt
            contact = contact_snapshot(fdm)
            wow = any_wow(contact)
            comp = max_compression(contact)

            if wow:
                wow_persistent_s += dt
                wow_loss_s = 0.0
            else:
                wow_loss_s += dt
                if wow_loss_s >= BOUNCE_WOW_LOSS_S:
                    bounce = True

            post_rows.append({
                "time_after_contact_s": float(t),
                "wow_detected": int(wow),
                "max_compression_ft": float(comp),
                "applied_physical_collective": float(applied_coll),
                **{
                    f"contact_{k}": float(v)
                    for k, v in contact.items()
                },
                **{k: float(v) for k, v in state.items()},
            })

            if bounce:
                termination = "bounce_after_first_wow"
                break

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    final_contact = contact_snapshot(fdm)

    ready_hold = bool(
        safe
        and first_contact is not None
        and abs(first_contact["vertical_speed_fps"]) <= 0.25
    )

    result = {
        "safe": bool(safe),
        "termination": str(termination),
        "contact_detected": bool(first_contact is not None),
        "first_contact": first_contact,
        "bounce_detected": bool(bounce),
        "wow_persistent_s": float(wow_persistent_s),
        "final_wow": bool(any_wow(final_contact)),
        "final_max_compression_ft": float(
            max_compression(final_contact)
        ),
        "final_altitude_ft": float(
            final_state["altitude_ft"]
        ),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "ready_for_touchdown_hold_calibration": bool(
            ready_hold
        ),
    }

    close_handoff(start)

    return (
        result,
        anchor_rows,
        extension_rows,
        trace_rows,
        post_rows,
    )



# =====================================================================
# STAGE-4 LANDED HOLD CALIBRATION V1
# =====================================================================

# Locked first-contact evidence from V4:
#   reduction=0.01925
#   AGL=6.762 ft
#   VS=-0.015 ft/s
#   COLL≈0.541194
#   WOW persisted for 3.0 s, no bounce
#
# V1 reproduces that exact first-contact path, then tests only a tiny
# post-contact collective offset around the measured contact collective.
# No PPO training. No retuning of Stage1/2/3 or near-ground horizontal control.

ANCHOR_REDUCTIONS = [
    0.0010, 0.0020, 0.0030, 0.0040,
    0.0050, 0.0060, 0.0070, 0.0080,
    0.0085, 0.0090, 0.0095, 0.0100,
    0.0105, 0.0110, 0.0115, 0.0120,
    0.0125, 0.0130, 0.0135, 0.0140,
    0.0145, 0.0150, 0.0155, 0.0160,
]

CONTACT_REDUCTIONS = [
    0.01625, 0.01650, 0.01675, 0.01700,
    0.01725, 0.01750, 0.01775, 0.01800,
    0.01825, 0.01850, 0.01875, 0.01900,
    0.01925,
]

# Tiny post-contact authority sweep around measured contact collective.
LANDED_COLL_OFFSETS = [
    0.00000,
    -0.00025,
    -0.00050,
]

LANDED_OBSERVE_S = 8.0
LANDED_REQUIRED_HOLD_S = 5.0
LANDED_WOW_LOSS_FAIL_S = 0.30

# Exact post-contact collective values already proven in standalone landed-hold
# qualification. V4 continuous touchdown reaches first WOW at ~0.541294, so
# these targets test whether a tiny additional unloading is required for
# persistent WOW in the continuous mission state.
LANDED_TARGET_COLLECTIVES = [
    0.541194,
    0.540944,
    0.540694,
]

LANDED_VS_TOL_FPS = 0.15
LANDED_FWD_SPEED_TOL_FPS = 0.60
LANDED_LAT_SPEED_TOL_FPS = 0.60
LANDED_HEADING_TOL_DEG = 1.0

LANDED_COLL_RATE_LIMIT_PER_S = 0.0020

LEVEL_MAX_TIME_S = 120.0
LEVEL_MIN_DWELL_S = 8.0
LEVEL_STABILITY_WINDOW_S = 3.0

EQ_VS_MEAN_ABS_MAX = 0.035
EQ_VS_PEAK_ABS_MAX = 0.070
EQ_ALT_RANGE_MAX_FT = 0.18

NO_WOW_MIN_AGL_FT = 5.20
MAX_DOWN_VS_FPS = -0.30


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def equilibrium_ok(window):
    if not window:
        return False, {}

    vs = np.asarray(
        [r["vertical_speed_fps"] for r in window],
        dtype=float,
    )
    alt = np.asarray(
        [r["altitude_ft"] for r in window],
        dtype=float,
    )

    stats = {
        "vs_mean_fps": float(np.mean(vs)),
        "vs_abs_peak_fps": float(np.max(np.abs(vs))),
        "altitude_mean_ft": float(np.mean(alt)),
        "altitude_range_ft": float(np.max(alt) - np.min(alt)),
    }

    ok = bool(
        abs(stats["vs_mean_fps"]) <= EQ_VS_MEAN_ABS_MAX
        and stats["vs_abs_peak_fps"] <= EQ_VS_PEAK_ABS_MAX
        and stats["altitude_range_ft"] <= EQ_ALT_RANGE_MAX_FT
    )
    return ok, stats


def precontact_safety_reason(state):
    if state["altitude_ft"] < NO_WOW_MIN_AGL_FT:
        return "no_wow_below_contact_search_floor"
    if state["vertical_speed_fps"] < MAX_DOWN_VS_FPS:
        return "descent_rate_limit"
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "forward_corridor_exit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "cross_corridor_exit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "roll_limit"
    return ""


def landed_accept_now(state, contact):
    return bool(
        any_wow(contact)
        and max_compression(contact) > 0.0
        and abs(state["vertical_speed_fps"]) <= LANDED_VS_TOL_FPS
        and abs(state["position_error_ft"]) <= PRESENTATION_MAX_POSITION_ERROR_FT
        and abs(state["forward_speed_fps"]) <= LANDED_FWD_SPEED_TOL_FPS
        and abs(state["cross_track_ft"]) <= PRESENTATION_MAX_CROSS_FT
        and abs(state["lateral_speed_fps"]) <= LANDED_LAT_SPEED_TOL_FPS
        and abs(state["heading_error_deg"]) <= LANDED_HEADING_TOL_DEG
    )


def hold_precontact_level(
    env2,
    fdm,
    lat0,
    lon0,
    mission_heading,
    dt,
    hover_action0,
    vertical_bias,
    target_coll,
    reduction,
    previous_coll,
):
    window_steps = max(
        2,
        int(round(LEVEL_STABILITY_WINDOW_S / dt)),
    )
    window = []

    for step in range(int(LEVEL_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        max_step_delta = LANDED_COLL_RATE_LIMIT_PER_S * dt
        requested_coll = rate_limit(
            target_coll,
            previous_coll,
            max_step_delta,
        )

        state, used, mapped_elev, applied_coll = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested_coll,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )
        previous_coll = float(applied_coll)

        t = (step + 1) * dt
        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        comp = max_compression(contact)

        window.append({
            "vertical_speed_fps": float(state["vertical_speed_fps"]),
            "altitude_ft": float(state["altitude_ft"]),
        })
        if len(window) > window_steps:
            window.pop(0)

        stable_now = False
        stats = {}
        if t >= LEVEL_MIN_DWELL_S and len(window) >= window_steps:
            stable_now, stats = equilibrium_ok(window)

        if wow:
            return {
                "safe": True,
                "settled": False,
                "contact": True,
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "state": state,
                "contact_snapshot": contact,
                "contact_collective": float(applied_coll),
                "contact_compression_ft": float(comp),
                "reduction": float(reduction),
            }

        reason = precontact_safety_reason(state)
        if reason:
            return {
                "safe": False,
                "settled": False,
                "contact": False,
                "termination": reason,
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "state": state,
            }

        if stable_now:
            return {
                "safe": True,
                "settled": True,
                "contact": False,
                "vertical_bias": float(vertical_bias),
                "previous_coll": float(previous_coll),
                "state": state,
                "eq_stats": stats,
            }

    return {
        "safe": True,
        "settled": False,
        "contact": False,
        "termination": "unsettled_level",
        "vertical_bias": float(vertical_bias),
        "previous_coll": float(previous_coll),
        "state": snapshot(fdm, lat0, lon0, mission_heading),
    }


def build_first_contact_open(points, local_slope):
    start = build_selected_9ft_capture_open(
        points,
        local_slope,
        detailed=False,
    )

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    vertical_bias = start["vertical_bias"]

    state0 = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    start_coll = float(state0["physical_collective_cmd"])
    previous_coll = float(start_coll)

    all_reductions = ANCHOR_REDUCTIONS + CONTACT_REDUCTIONS

    for idx, reduction in enumerate(all_reductions, start=1):
        target_coll = start_coll - reduction

        result = hold_precontact_level(
            env2,
            fdm,
            lat0,
            lon0,
            mission_heading,
            dt,
            hover_action0,
            vertical_bias,
            target_coll,
            reduction,
            previous_coll,
        )

        vertical_bias = result["vertical_bias"]
        previous_coll = result["previous_coll"]

        if not result["safe"]:
            close_handoff(start)
            raise RuntimeError(
                f"Pre-contact reproduction safety failure at reduction "
                f"{reduction:.5f}: {result.get('termination')}"
            )

        if result["contact"]:
            fc_state = result["state"]
            return {
                **start,
                "vertical_bias": float(vertical_bias),
                "first_contact_state": fc_state,
                "first_contact_snapshot": result["contact_snapshot"],
                "contact_collective": float(result["contact_collective"]),
                "contact_compression_ft": float(
                    result["contact_compression_ft"]
                ),
                "contact_reduction": float(reduction),
            }

        if not result["settled"]:
            close_handoff(start)
            raise RuntimeError(
                f"Pre-contact reproduction did not settle at reduction "
                f"{reduction:.5f}."
            )

    close_handoff(start)
    raise RuntimeError(
        "Expected V4 first WOW contact was not reproduced."
    )


def run_landed_hold_candidate(
    collective_offset,
    points,
    local_slope,
    detailed=False,
):
    run = build_first_contact_open(
        points,
        local_slope,
    )

    env2 = run["env2"]
    fdm = run["fdm"]
    lat0 = run["lat0"]
    lon0 = run["lon0"]
    mission_heading = run["mission_heading"]
    dt = run["dt"]
    hover_action0 = run["hover_action0"]
    vertical_bias = run["vertical_bias"]

    contact_coll = float(run["contact_collective"])
    target_coll = float(contact_coll + collective_offset)
    previous_coll = float(contact_coll)

    trace_rows = []

    hold_s = 0.0
    wow_loss_s = 0.0
    bounce = False
    safe = True
    termination = "landed_observation_complete"

    max_comp = float(run["contact_compression_ft"])
    min_comp = float(run["contact_compression_ft"])

    max_abs_pos = abs(
        run["first_contact_state"]["position_error_ft"]
    )
    max_abs_cross = abs(
        run["first_contact_state"]["cross_track_ft"]
    )

    next_print = 0.0

    for step in range(int(LANDED_OBSERVE_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        max_step_delta = LANDED_COLL_RATE_LIMIT_PER_S * dt
        requested_coll = rate_limit(
            target_coll,
            previous_coll,
            max_step_delta,
        )

        state, used, mapped_elev, applied_coll = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested_coll,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )
        previous_coll = float(applied_coll)

        t = (step + 1) * dt
        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        comp = max_compression(contact)

        max_comp = max(max_comp, comp)
        min_comp = min(min_comp, comp)
        max_abs_pos = max(
            max_abs_pos,
            abs(state["position_error_ft"]),
        )
        max_abs_cross = max(
            max_abs_cross,
            abs(state["cross_track_ft"]),
        )

        if wow:
            wow_loss_s = 0.0
        else:
            wow_loss_s += dt
            if wow_loss_s >= LANDED_WOW_LOSS_FAIL_S:
                bounce = True
                safe = False
                termination = "wow_loss_or_bounce"
                break

        if landed_accept_now(state, contact):
            hold_s += dt
        else:
            hold_s = 0.0

        trace_rows.append({
            "time_after_contact_s": float(t),
            "collective_offset": float(collective_offset),
            "target_physical_collective": float(target_coll),
            "applied_physical_collective": float(applied_coll),
            "mapped_physical_elevator": float(mapped_elev),
            "wow_detected": int(wow),
            "max_compression_ft": float(comp),
            "landed_hold_s": float(hold_s),
            **{
                f"contact_{k}": float(v)
                for k, v in contact.items()
            },
            **{k: float(v) for k, v in state.items()},
        })

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  t={t:5.2f}s | "
                f"ALT={state['altitude_ft']:.3f} "
                f"VS={state['vertical_speed_fps']:+.3f} | "
                f"FWDerr={state['position_error_ft']:+.2f} "
                f"V={state['forward_speed_fps']:+.3f} | "
                f"X={state['cross_track_ft']:+.2f} "
                f"LAT={state['lateral_speed_fps']:+.3f} | "
                f"WOW={int(wow)} "
                f"COMP={comp:.4f} "
                f"COLL={applied_coll:.6f} "
                f"HOLD={hold_s:.2f}s"
            )
            next_print += 1.0

        if hold_s >= LANDED_REQUIRED_HOLD_S:
            termination = "stable_landed_hold_5s"
            break

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    final_contact = contact_snapshot(fdm)

    passed = bool(
        safe
        and not bounce
        and termination == "stable_landed_hold_5s"
        and hold_s >= LANDED_REQUIRED_HOLD_S
        and landed_accept_now(final_state, final_contact)
    )

    result = {
        "pass": bool(passed),
        "safe": bool(safe),
        "termination": str(termination),
        "collective_offset": float(collective_offset),
        "contact_collective": float(contact_coll),
        "target_landed_collective": float(target_coll),
        "contact_reduction": float(run["contact_reduction"]),
        "contact_altitude_ft": float(
            run["first_contact_state"]["altitude_ft"]
        ),
        "contact_vertical_speed_fps": float(
            run["first_contact_state"]["vertical_speed_fps"]
        ),
        "contact_position_error_ft": float(
            run["first_contact_state"]["position_error_ft"]
        ),
        "contact_cross_track_ft": float(
            run["first_contact_state"]["cross_track_ft"]
        ),
        "contact_lateral_speed_fps": float(
            run["first_contact_state"]["lateral_speed_fps"]
        ),
        "contact_compression_ft": float(
            run["contact_compression_ft"]
        ),
        "landed_hold_s": float(hold_s),
        "bounce": bool(bounce),
        "final_wow": bool(any_wow(final_contact)),
        "final_compression_ft": float(
            max_compression(final_contact)
        ),
        "max_compression_ft": float(max_comp),
        "min_compression_ft": float(min_comp),
        "final_altitude_ft": float(
            final_state["altitude_ft"]
        ),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_forward_ft": float(
            final_state["forward_ft"]
        ),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "final_heading_error_deg": float(
            final_state["heading_error_deg"]
        ),
        "max_abs_position_error_ft": float(max_abs_pos),
        "max_abs_cross_track_ft": float(max_abs_cross),
    }

    close_handoff(run)
    return result, trace_rows


def landed_rank_key(row):
    return (
        0 if row["pass"] else 1,
        0 if row["safe"] else 1,
        0 if row["final_wow"] else 1,
        0 if not row["bounce"] else 1,
        -row["landed_hold_s"],
        abs(row["collective_offset"]),
        row["max_abs_position_error_ft"],
        row["max_abs_cross_track_ft"],
    )



# =====================================================================
# FULL STAGE-4 TEACHER BUILD / VALIDATION V1
# =====================================================================

# This is still teacher/control validation only:
#   - no Stage-4 PPO training
#   - no distillation
#   - no modification of locked Stage1/2/3 models
#
# The run is continuous on ONE JSBSim FDM from the qualified Stage-4 entry
# through:
#   pre-descent endpoint hold
#   300 -> 30 ft
#   near-ground transition to native action0=-1 equilibrium
#   continuous capture to 9 ft + 5 s hover
#   continuous touchdown approach using a rate-limited collective ramp
#     through the measured lower ground-effect/contact authority range
#     with a slower final ramp below 7.1 ft to reduce rebound energy
#   actual WOW contact
#   5 s landed hold at the measured qualified contact collective

LOWER_ANCHOR_CSV = Path(
    "results_stage4_touchdown_quarterstep_v4/anchor_levels.csv"
)
LOWER_EXTENSION_CSV = Path(
    "results_stage4_touchdown_quarterstep_v4/quarterstep_extension_levels.csv"
)
LOWER_SUMMARY_JSON = Path(
    "results_stage4_touchdown_quarterstep_v4/final_summary.json"
)

FULL_RESULT_DIR = Path("results_stage4_teacher_off_corrective_v2")
FULL_RESULT_DIR.mkdir(parents=True, exist_ok=True)

TOUCHDOWN_VS_DES_HIGH = -0.10
TOUCHDOWN_VS_DES_LOW = -0.02
TOUCHDOWN_TAPER_ALT_FT = 7.10
TOUCHDOWN_KVS = 0.014

# V2 used the local equilibrium collective as feedforward. That can settle at
# successive near-ground equilibria instead of forcing continuous progress.
# V3 continuously ramps the physical collective through the already-measured
# safe authority range, with VS feedback providing damping.
TOUCHDOWN_BASELINE_RAMP_PER_S = 0.00040
TOUCHDOWN_NEAR_CONTACT_RAMP_PER_S = 0.00010
TOUCHDOWN_COLL_RATE_LIMIT_PER_S = 0.0015
TOUCHDOWN_COLL_BELOW_CONTACT_MARGIN = 0.0020
TOUCHDOWN_COLL_ABOVE_START_MARGIN = 0.0005

TOUCHDOWN_NO_WOW_MIN_AGL_FT = 5.20
TOUCHDOWN_MAX_DOWN_VS_FPS = -0.30
TOUCHDOWN_MAX_TIME_S = 80.0

LANDED_TARGET_OFFSET = 0.0
LANDED_REQUIRED_HOLD_S = 5.0
LANDED_MAX_TIME_S = 10.0
LANDED_WOW_LOSS_FAIL_S = 0.30


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pava_nondecreasing(y):
    """Simple equal-weight PAVA for a nondecreasing sequence."""
    blocks = []
    for i, val in enumerate([float(v) for v in y]):
        blocks.append({
            "start": i,
            "end": i,
            "weight": 1.0,
            "mean": val,
        })
        while (
            len(blocks) >= 2
            and blocks[-2]["mean"] > blocks[-1]["mean"]
        ):
            b = blocks.pop()
            a = blocks.pop()
            w = a["weight"] + b["weight"]
            m = (
                a["mean"] * a["weight"]
                + b["mean"] * b["weight"]
            ) / w
            blocks.append({
                "start": a["start"],
                "end": b["end"],
                "weight": w,
                "mean": m,
            })
    out = np.empty(len(y), dtype=float)
    for b in blocks:
        out[b["start"]:b["end"] + 1] = b["mean"]
    return out


def load_lower_touchdown_curve():
    missing = [
        p for p in [
            LOWER_ANCHOR_CSV,
            LOWER_EXTENSION_CSV,
            LOWER_SUMMARY_JSON,
        ]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing touchdown-identification result files: "
            + ", ".join(str(p) for p in missing)
        )

    pts = []

    for path in [LOWER_ANCHOR_CSV, LOWER_EXTENSION_CSV]:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                safe = str(row.get("safe", "")).lower() == "true"
                settled = str(row.get("settled", "")).lower() == "true"
                if not (safe and settled):
                    continue
                alt = float(row["equilibrium_altitude_mean_ft"])
                coll = float(row["target_physical_collective"])
                if np.isfinite(alt) and np.isfinite(coll):
                    pts.append((alt, coll))

    with open(LOWER_SUMMARY_JSON) as f:
        summary = json.load(f)

    result = summary.get("result", {})
    fc = result.get("first_contact")
    if not fc:
        raise RuntimeError(
            "V4 touchdown summary does not contain first_contact evidence."
        )

    contact_alt = float(fc["altitude_ft"])
    contact_coll = float(fc["physical_collective_cmd"])
    contact_vs = float(fc["vertical_speed_fps"])

    pts.append((contact_alt, contact_coll))

    # Merge nearly duplicate altitudes before isotonic projection.
    pts.sort(key=lambda x: x[0])

    merged = []
    for alt, coll in pts:
        if merged and abs(alt - merged[-1][0]) < 0.01:
            a0, c0, n0 = merged[-1]
            n1 = n0 + 1
            merged[-1] = (
                (a0 * n0 + alt) / n1,
                (c0 * n0 + coll) / n1,
                n1,
            )
        else:
            merged.append((alt, coll, 1))

    alts = np.asarray([m[0] for m in merged], dtype=float)
    colls_raw = np.asarray([m[1] for m in merged], dtype=float)

    # Physical expectation: higher equilibrium altitude requires >= collective.
    colls_iso = pava_nondecreasing(colls_raw)

    points = [
        (float(a), float(c))
        for a, c in zip(alts, colls_iso)
    ]

    return {
        "points": points,
        "contact_altitude_ft": contact_alt,
        "contact_collective": contact_coll,
        "contact_vertical_speed_fps": contact_vs,
        "raw_point_count": len(pts),
        "projected_point_count": len(points),
    }


def lower_feedforward_collective(
    altitude_ft,
    points,
    start_altitude_ft,
    start_collective,
):
    # Include the actual fresh 9-ft hover point for this run.
    run_points = list(points) + [
        (float(start_altitude_ft), float(start_collective))
    ]
    run_points.sort(key=lambda x: x[0])

    x = np.asarray([p[0] for p in run_points], dtype=float)
    y = np.asarray([p[1] for p in run_points], dtype=float)

    # Re-project after adding the fresh start point.
    y = pava_nondecreasing(y)

    alt = float(altitude_ft)

    if alt <= x[0]:
        ff = float(y[0])
    elif alt >= x[-1]:
        ff = float(y[-1])
    else:
        ff = float(np.interp(alt, x, y))

    return float(np.clip(
        ff,
        TOUCHDOWN_COLL_MIN,
        TOUCHDOWN_COLL_MAX,
    ))


def full_safety_reason(state, allow_contact=False):
    if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
        return "forward_position_limit"
    if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
        return "cross_track_limit"
    if abs(state["heading_error_deg"]) > 2.0:
        return "heading_limit"
    if abs(math.degrees(state["pitch_rad"])) > 8.0:
        return "pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 10.0:
        return "roll_limit"
    if state["vertical_speed_fps"] < -2.5 and not allow_contact:
        return "vertical_speed_limit"
    return ""


def touchdown_precontact_reason(state):
    if state["altitude_ft"] < TOUCHDOWN_NO_WOW_MIN_AGL_FT:
        return "no_wow_below_contact_floor"
    if state["vertical_speed_fps"] < TOUCHDOWN_MAX_DOWN_VS_FPS:
        return "touchdown_descent_rate_limit"
    return full_safety_reason(state, allow_contact=False)


def append_trace(
    trace,
    phase,
    time_s,
    state,
    action=None,
    extra=None,
):
    row = {
        "phase": str(phase),
        "time_s": float(time_s),
    }

    if action is not None:
        a = np.asarray(action, dtype=float).reshape(-1)
        for i in range(min(4, len(a))):
            row[f"teacher_action{i}"] = float(a[i])

    if extra:
        for k, v in extra.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
            else:
                row[k] = v

    for k, v in state.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            row[k] = float(v)

    trace.append(row)


def landed_accept_now(state, contact):
    return bool(
        any_wow(contact)
        and max_compression(contact) > 0.0
        and abs(state["vertical_speed_fps"]) <= 0.15
        and abs(state["position_error_ft"]) <= STAGE4_POS_TOL_FT
        and abs(state["forward_speed_fps"]) <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(state["cross_track_ft"]) <= STAGE4_CROSS_TOL_FT
        and abs(state["lateral_speed_fps"]) <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(state["heading_error_deg"]) <= STAGE4_HEADING_TOL_DEG
    )


def run_full_stage4_teacher(landed_target_collective, detailed=True):
    upper_curve_points = load_confirmed_curve_points()
    upper_local_slope, _, upper_local_r2 = (
        fit_local_lower_curve_slope(upper_curve_points, n=5)
    )
    lower_curve = load_lower_touchdown_curve()

    start = build_stage4_handoff(
        detailed=False,
        require_full_hold=False,
        custom_entry_forward_ft=STAGE4_ENTRY_FORWARD_FT,
        custom_entry_max_speed_fps=STAGE4_ENTRY_MAX_SPEED_FPS,
    )

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = env_control_dt(env2)

    trace = []
    phase_summary = {}
    mission_time = 0.0

    max_abs_pos = 0.0
    max_abs_cross = 0.0
    max_abs_heading = 0.0
    min_alt = +999.0
    min_vs = +999.0

    def update_metrics(state):
        nonlocal max_abs_pos, max_abs_cross, max_abs_heading, min_alt, min_vs
        max_abs_pos = max(max_abs_pos, abs(state["position_error_ft"]))
        max_abs_cross = max(max_abs_cross, abs(state["cross_track_ft"]))
        max_abs_heading = max(
            max_abs_heading,
            abs(state["heading_error_deg"]),
        )
        min_alt = min(min_alt, state["altitude_ft"])
        min_vs = min(min_vs, state["vertical_speed_fps"])

    # ---------------------------------------------------------------
    # 1) PRE-DESCENT ENDPOINT HOLD
    # ---------------------------------------------------------------
    pre_hold = 0.0
    endpoint_entered = False
    phase_start_t = mission_time

    for step in range(int(40.0 / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )
        obs3 = stage3_observation(state_before)
        base, _ = stage3_model.predict(
            obs3,
            deterministic=True,
        )
        action = np.asarray(
            base,
            dtype=np.float32,
        ).reshape(-1).copy()
        action[1] = FIXED_BRAKE_A1

        state, used = raw_policy_cycle(
            env2,
            fdm,
            action,
            lat0,
            lon0,
            mission_heading,
        )
        mission_time += dt

        in_endpoint = endpoint_hover_now(state)
        endpoint_entered = endpoint_entered or in_endpoint
        pre_hold = pre_hold + dt if in_endpoint else 0.0

        update_metrics(state)
        append_trace(
            trace,
            "pre_descent_hold",
            mission_time,
            state,
            used,
            {"hold_s": pre_hold},
        )

        reason = stage3_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(
                f"Full teacher pre-descent safety failure: {reason}"
            )

        if endpoint_entered:
            reason = full_safety_reason(state)
            if reason:
                close_handoff(start)
                raise RuntimeError(
                    f"Full teacher pre-descent corridor failure: {reason}"
                )

        if pre_hold >= PRE_DESCENT_HOLD_SECONDS:
            break

    if pre_hold < PRE_DESCENT_HOLD_SECONDS:
        close_handoff(start)
        raise RuntimeError(
            "Full teacher failed to reproduce 5-s endpoint pre-descent hold."
        )

    phase_summary["pre_descent_hold"] = {
        "duration_s": float(mission_time - phase_start_t),
        "hold_s": float(pre_hold),
    }

    # Freeze qualified Stage3 collective reference.
    pre_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    obs3 = stage3_observation(pre_state)
    hover_action, _ = stage3_model.predict(
        obs3,
        deterministic=True,
    )
    hover_action = np.asarray(
        hover_action,
        dtype=np.float32,
    ).reshape(-1)
    hover_action0 = float(hover_action[0])

    # Presentation corridor accounting starts after the qualified endpoint
    # hold. The mission manager is intentionally allowed to assume control at
    # the earlier 294-ft braking gate, which is outside the ±5-ft endpoint
    # corridor by construction.
    max_abs_pos = abs(pre_state["position_error_ft"])
    max_abs_cross = abs(pre_state["cross_track_ft"])
    max_abs_heading = abs(pre_state["heading_error_deg"])
    min_alt = pre_state["altitude_ft"]
    min_vs = pre_state["vertical_speed_fps"]

    # ---------------------------------------------------------------
    # 2) QUALIFIED 300 -> 30 FT
    # ---------------------------------------------------------------
    vertical_bias = 0.0
    settle30 = 0.0
    phase_start_t = mission_time

    for step in range(int(DESCENT_MAX_TIME / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = build_stage4_teacher_action(
            state_before,
            hover_action0=hover_action0,
            descent_vmax=LOCKED_DESCENT_VMAX,
            vs_kp=LOCKED_VS_KP,
            vs_ki=LOCKED_VS_KI,
            vertical_bias=vertical_bias,
            low_alt_bias_floor=QUALIFIED_LOW_ALT_BIAS_FLOOR,
            dt=dt,
            long_kpos_phys=LOCKED_LONG_KPOS_PHYS,
            long_kv_phys=LOCKED_LONG_KV_PHYS,
        )

        state, used, base_elev, mapped_elev = raw_stage4_mapped_cycle(
            env2,
            fdm,
            action,
            lat0,
            lon0,
            mission_heading,
        )
        mission_time += dt

        settle30 = (
            settle30 + dt
            if stage4_settle_now(state)
            else 0.0
        )

        update_metrics(state)
        append_trace(
            trace,
            "descent_300_to_30",
            mission_time,
            state,
            used,
            {
                "vs_des_fps": ctrl["vs_des_fps"],
                "vertical_bias_effective": ctrl["vertical_bias_effective"],
                "collective_residual": ctrl["collective_residual"],
                "mapped_physical_elevator_cmd": mapped_elev,
                "settle30_s": settle30,
            },
        )

        reason = stage4_teacher_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(
                f"Full teacher 300->30 safety failure: {reason}"
            )

        reason = full_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(
                f"Full teacher 300->30 corridor failure: {reason}"
            )

        if settle30 >= DESCENT_SETTLE_SECONDS:
            break

    state30 = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    if not (
        settle30 >= DESCENT_SETTLE_SECONDS
        and stage4_settle_now(state30)
    ):
        close_handoff(start)
        raise RuntimeError(
            "Full teacher did not reproduce qualified 30-ft hover."
        )

    phase_summary["descent_300_to_30"] = {
        "duration_s": float(mission_time - phase_start_t),
        "final_altitude_ft": float(state30["altitude_ft"]),
        "settle_hold_s": float(settle30),
    }

    # ---------------------------------------------------------------
    # 3) 30 FT -> NATIVE A0=-1 LOW-ALT EQUILIBRIUM
    # ---------------------------------------------------------------
    phase_start_t = mission_time
    reached_floor = False
    native_trigger = None

    for step in range(int(PHYS_ID_TRIGGER_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=LOW_ALT_TARGET_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        if (
            ctrl["collective_residual"]
            <= COLLECTIVE_RESIDUAL_MIN + 1e-8
        ):
            reached_floor = True

        if reached_floor:
            state, used, mapped_elev = raw_native_action0_cycle(
                env2,
                fdm,
                action,
                forced_action0=-1.0,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
            mode = "native_action0_min"
        else:
            state, used, _, mapped_elev = raw_stage4_near_mapped_cycle(
                env2,
                fdm,
                action,
                lat0,
                lon0,
                mission_heading,
            )
            mode = "residual_controller"

        mission_time += dt
        update_metrics(state)
        append_trace(
            trace,
            "near_ground_30_to_native_eq",
            mission_time,
            state,
            used,
            {
                "mode": mode,
                "vs_des_fps": ctrl["vs_des_fps"],
                "collective_residual": ctrl["collective_residual"],
                "vertical_bias_state": ctrl["vertical_bias_state"],
                "mapped_physical_elevator_cmd": mapped_elev,
            },
        )

        reason = near_ground_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(
                f"Full teacher 30->native-equilibrium safety failure: {reason}"
            )

        if any_wow(contact_snapshot(fdm)):
            close_handoff(start)
            raise RuntimeError(
                "Unexpected WOW before 9-ft capture phase."
            )

        if (
            reached_floor
            and state["altitude_ft"]
                <= PHYS_ID_TRIGGER_ALT_MAX_FT
            and abs(state["vertical_speed_fps"])
                <= PHYS_ID_TRIGGER_ABS_VS_MAX_FPS
        ):
            native_trigger = state.copy()
            break

    if native_trigger is None:
        close_handoff(start)
        raise RuntimeError(
            "Full teacher did not reproduce native low-altitude equilibrium."
        )

    phase_summary["near_ground_30_to_native_eq"] = {
        "duration_s": float(mission_time - phase_start_t),
        "final_altitude_ft": float(native_trigger["altitude_ft"]),
        "final_vertical_speed_fps": float(
            native_trigger["vertical_speed_fps"]
        ),
    }

    # ---------------------------------------------------------------
    # 4) CONTINUOUS NATIVE EQ -> 9 FT + 5 S HOVER
    # ---------------------------------------------------------------
    phase_start_t = mission_time
    hover9 = 0.0
    start9_state = None

    for step in range(int(CAPTURE_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        ff_coll, ff_mode = feedforward_collective(
            state_before["altitude_ft"],
            upper_curve_points,
            upper_local_slope,
            TD_SLOPE_MULTIPLIER,
        )

        vs_des = CAPTURE_VS_GAIN * (
            CAPTURE_TARGET_ALT_FT
            - state_before["altitude_ft"]
        )
        vs_des = float(np.clip(
            vs_des,
            -CAPTURE_MAX_DESCENT_FPS,
            +CAPTURE_MAX_CLIMB_FPS,
        ))

        vs_error = (
            vs_des - state_before["vertical_speed_fps"]
        )

        requested_coll = float(np.clip(
            ff_coll + TD_CAPTURE_KVS * vs_error,
            CAPTURE_PHYSICAL_COLL_MIN,
            CAPTURE_PHYSICAL_COLL_MAX,
        ))

        state, used, mapped_elev, applied_coll = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested_coll,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )
        mission_time += dt

        if capture_hold_now(state):
            hover9 += dt
            if start9_state is None:
                start9_state = state.copy()
        else:
            hover9 = 0.0

        update_metrics(state)
        append_trace(
            trace,
            "capture_to_9ft",
            mission_time,
            state,
            used,
            {
                "feedforward_mode": ff_mode,
                "feedforward_collective": ff_coll,
                "vs_des_fps": vs_des,
                "requested_physical_collective": requested_coll,
                "applied_physical_collective": applied_coll,
                "mapped_physical_elevator_cmd": mapped_elev,
                "hover9_s": hover9,
            },
        )

        if any_wow(contact_snapshot(fdm)):
            close_handoff(start)
            raise RuntimeError(
                "Unexpected WOW during 9-ft capture."
            )

        reason = capture_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(
                f"Full teacher 9-ft capture safety failure: {reason}"
            )

        if hover9 >= CAPTURE_HOLD_SECONDS:
            break

    state9 = snapshot(
        fdm, lat0, lon0, mission_heading
    )

    if hover9 < CAPTURE_HOLD_SECONDS:
        close_handoff(start)
        raise RuntimeError(
            "Full teacher did not reproduce 5-s 9-ft hover."
        )

    phase_summary["capture_to_9ft"] = {
        "duration_s": float(mission_time - phase_start_t),
        "final_altitude_ft": float(state9["altitude_ft"]),
        "hover_hold_s": float(hover9),
        "upper_curve_local_r2": float(upper_local_r2),
    }

    # ---------------------------------------------------------------
    # 5) CONTINUOUS 9 FT -> FIRST WOW
    # ---------------------------------------------------------------
    phase_start_t = mission_time

    touchdown_start_alt = float(state9["altitude_ft"])
    touchdown_start_coll = float(
        state9["physical_collective_cmd"]
    )
    previous_coll = float(touchdown_start_coll)

    contact_coll_ref = float(
        lower_curve["contact_collective"]
    )
    touchdown_coll_min = float(
        contact_coll_ref
        - TOUCHDOWN_COLL_BELOW_CONTACT_MARGIN
    )
    touchdown_coll_max = float(
        touchdown_start_coll
        + TOUCHDOWN_COLL_ABOVE_START_MARGIN
    )

    first_contact = None
    next_touchdown_print = 0.0
    ramp_baseline_coll = float(touchdown_start_coll)

    for step in range(int(TOUCHDOWN_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        touchdown_elapsed = float(step * dt)

        # Continuous baseline progression through the measured safe collective
        # range. Above 7.1 ft keep the proven V3 rate. Near contact, slow the
        # baseline ramp 4x and request only -0.02 ft/s so first WOW is reached
        # quasi-statically rather than with unnecessary rebound energy.
        if state_before["altitude_ft"] > TOUCHDOWN_TAPER_ALT_FT:
            ramp_rate = TOUCHDOWN_BASELINE_RAMP_PER_S
            vs_des = TOUCHDOWN_VS_DES_HIGH
        else:
            ramp_rate = TOUCHDOWN_NEAR_CONTACT_RAMP_PER_S
            vs_des = TOUCHDOWN_VS_DES_LOW

        ramp_baseline_coll = float(max(
            contact_coll_ref,
            ramp_baseline_coll - ramp_rate * dt,
        ))

        vs_error = (
            vs_des - state_before["vertical_speed_fps"]
        )

        desired_coll = float(np.clip(
            ramp_baseline_coll
            + TOUCHDOWN_KVS * vs_error,
            touchdown_coll_min,
            touchdown_coll_max,
        ))

        max_delta = TOUCHDOWN_COLL_RATE_LIMIT_PER_S * dt
        requested_coll = rate_limit(
            desired_coll,
            previous_coll,
            max_delta,
        )

        state, used, mapped_elev, applied_coll = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested_coll,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )
        previous_coll = float(applied_coll)
        mission_time += dt

        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        comp = max_compression(contact)

        update_metrics(state)
        append_trace(
            trace,
            "continuous_touchdown",
            mission_time,
            state,
            used,
            {
                "touchdown_elapsed_s": touchdown_elapsed,
                "ramp_baseline_collective": ramp_baseline_coll,
                "ramp_rate_per_s": ramp_rate,
                "contact_collective_reference": contact_coll_ref,
                "vs_des_fps": vs_des,
                "vs_error_fps": vs_error,
                "desired_physical_collective": desired_coll,
                "applied_physical_collective": applied_coll,
                "mapped_physical_elevator_cmd": mapped_elev,
                "wow_detected": int(wow),
                "max_compression_ft": comp,
            },
        )

        if detailed and touchdown_elapsed + 1e-9 >= next_touchdown_print:
            print(
                f"Touchdown t={touchdown_elapsed:6.2f}s | "
                f"ALT={state['altitude_ft']:6.3f} "
                f"VS={state['vertical_speed_fps']:+6.3f}/"
                f"{vs_des:+5.2f} | "
                f"FWDerr={state['position_error_ft']:+5.2f} "
                f"X={state['cross_track_ft']:+5.2f} | "
                f"RAMP={ramp_baseline_coll:.6f} "
                f"COLL={state['physical_collective_cmd']:.6f} | "
                f"WOW={int(wow)} COMP={comp:.4f}"
            )
            next_touchdown_print += 5.0

        if wow:
            first_contact = {
                "mission_time_s": float(mission_time),
                "touchdown_elapsed_s": float(
                    touchdown_elapsed
                ),
                "altitude_ft": float(state["altitude_ft"]),
                "vertical_speed_fps": float(
                    state["vertical_speed_fps"]
                ),
                "forward_ft": float(state["forward_ft"]),
                "position_error_ft": float(
                    state["position_error_ft"]
                ),
                "forward_speed_fps": float(
                    state["forward_speed_fps"]
                ),
                "cross_track_ft": float(
                    state["cross_track_ft"]
                ),
                "lateral_speed_fps": float(
                    state["lateral_speed_fps"]
                ),
                "heading_error_deg": float(
                    state["heading_error_deg"]
                ),
                "physical_collective_cmd": float(
                    state["physical_collective_cmd"]
                ),
                "ramp_baseline_collective": float(
                    ramp_baseline_coll
                ),
                "max_compression_ft": float(comp),
            }
            break

        reason = touchdown_precontact_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(
                f"Full teacher continuous touchdown safety failure: {reason}"
            )

    if first_contact is None:
        final_td_state = snapshot(
            fdm, lat0, lon0, mission_heading
        )
        close_handoff(start)
        raise RuntimeError(
            "Full teacher continuous touchdown did not reach WOW. "
            f"Final ALT={final_td_state['altitude_ft']:.3f}, "
            f"VS={final_td_state['vertical_speed_fps']:+.3f}, "
            f"COLL={final_td_state['physical_collective_cmd']:.6f}."
        )

    if abs(first_contact["vertical_speed_fps"]) > 0.25:
        close_handoff(start)
        raise RuntimeError(
            "Full teacher touchdown VS exceeded 0.25 ft/s."
        )

    phase_summary["continuous_touchdown"] = {
        "duration_s": float(mission_time - phase_start_t),
        "contact": first_contact,
        "measured_reference_contact_altitude_ft": float(
            lower_curve["contact_altitude_ft"]
        ),
        "measured_reference_contact_collective": float(
            lower_curve["contact_collective"]
        ),
    }

    # ---------------------------------------------------------------
    # 6) WOW -> 5 S STABLE LANDED HOLD
    # ---------------------------------------------------------------
    phase_start_t = mission_time

    target_landed_coll = float(
        landed_target_collective
    )
    previous_coll = float(
        first_contact["physical_collective_cmd"]
    )

    landed_hold = 0.0
    wow_loss_s = 0.0
    bounce = False

    for step in range(int(LANDED_MAX_TIME_S / dt)):
        state_before = snapshot(
            fdm, lat0, lon0, mission_heading
        )

        action, ctrl, vertical_bias = (
            build_near_ground_action_low_alt_bias(
                state_before,
                target_alt_ft=CAPTURE_TARGET_ALT_FT,
                hover_action0=hover_action0,
                vertical_bias=vertical_bias,
                dt=dt,
                low_alt_bias_min=PHYS_ID_LOW_ALT_BIAS_MIN,
            )
        )

        max_delta = LANDED_COLL_RATE_LIMIT_PER_S * dt
        requested_coll = rate_limit(
            target_landed_coll,
            previous_coll,
            max_delta,
        )

        state, used, mapped_elev, applied_coll = (
            raw_physical_collective_cycle(
                env2,
                fdm,
                action,
                physical_collective_cmd=requested_coll,
                lat0=lat0,
                lon0=lon0,
                mission_heading=mission_heading,
            )
        )
        previous_coll = float(applied_coll)
        mission_time += dt

        contact = contact_snapshot(fdm)
        wow = any_wow(contact)
        comp = max_compression(contact)

        if wow:
            wow_loss_s = 0.0
        else:
            wow_loss_s += dt

        if wow_loss_s >= LANDED_WOW_LOSS_FAIL_S:
            bounce = True
            break

        if landed_accept_now(state, contact):
            landed_hold += dt
        else:
            landed_hold = 0.0

        update_metrics(state)
        append_trace(
            trace,
            "landed_hold",
            mission_time,
            state,
            used,
            {
                "target_landed_collective": target_landed_coll,
                "applied_physical_collective": applied_coll,
                "mapped_physical_elevator_cmd": mapped_elev,
                "wow_detected": int(wow),
                "max_compression_ft": comp,
                "landed_hold_s": landed_hold,
            },
        )

        if landed_hold >= LANDED_REQUIRED_HOLD_S:
            break

    final_state = snapshot(
        fdm, lat0, lon0, mission_heading
    )
    final_contact = contact_snapshot(fdm)

    landed_pass = bool(
        not bounce
        and landed_hold >= LANDED_REQUIRED_HOLD_S
        and landed_accept_now(
            final_state,
            final_contact,
        )
    )

    phase_summary["landed_hold"] = {
        "duration_s": float(mission_time - phase_start_t),
        "hold_s": float(landed_hold),
        "bounce": bool(bounce),
        "final_wow": bool(any_wow(final_contact)),
        "final_compression_ft": float(
            max_compression(final_contact)
        ),
    }

    presentation_pass = bool(
        max_abs_pos <= PRESENTATION_MAX_POSITION_ERROR_FT
        and max_abs_cross <= PRESENTATION_MAX_CROSS_FT
    )

    full_pass = bool(
        landed_pass
        and presentation_pass
        and abs(first_contact["vertical_speed_fps"]) <= 0.25
        and abs(final_state["forward_speed_fps"])
            <= STAGE4_FWD_SPEED_TOL_FPS
        and abs(final_state["lateral_speed_fps"])
            <= STAGE4_LAT_SPEED_TOL_FPS
        and abs(final_state["heading_error_deg"])
            <= STAGE4_HEADING_TOL_DEG
    )

    result = {
        "pass": bool(full_pass),
        "same_fdm": bool(start["same_fdm"]),
        "clock_reset_on_attach": bool(
            start["clock_reset_on_attach"]
        ),
        "mission_time_s": float(mission_time),
        "presentation_pass": bool(presentation_pass),
        "max_abs_position_error_ft": float(max_abs_pos),
        "max_abs_cross_track_ft": float(max_abs_cross),
        "max_abs_heading_error_deg": float(max_abs_heading),
        "min_altitude_ft": float(min_alt),
        "min_vertical_speed_fps": float(min_vs),
        "pre_descent_hold_s": float(pre_hold),
        "hover9_s": float(hover9),
        "touchdown_first_contact": first_contact,
        "touchdown_vs_ok": bool(
            abs(first_contact["vertical_speed_fps"]) <= 0.25
        ),
        "landed_hold_s": float(landed_hold),
        "bounce": bool(bounce),
        "final_wow": bool(any_wow(final_contact)),
        "final_compression_ft": float(
            max_compression(final_contact)
        ),
        "final_altitude_ft": float(
            final_state["altitude_ft"]
        ),
        "final_vertical_speed_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "final_forward_ft": float(
            final_state["forward_ft"]
        ),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "final_cross_track_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "final_heading_error_deg": float(
            final_state["heading_error_deg"]
        ),
        "phase_summary": phase_summary,
        "landed_target_collective": float(target_landed_coll),
        "lower_curve_raw_point_count": int(
            lower_curve["raw_point_count"]
        ),
        "lower_curve_projected_point_count": int(
            lower_curve["projected_point_count"]
        ),
    }

    if detailed:
        print(
            "FULL TEACHER RESULT | "
            f"PASS={result['pass']} "
            f"same_fdm={result['same_fdm']} "
            f"clock_reset={result['clock_reset_on_attach']} | "
            f"CONTACT_ALT={first_contact['altitude_ft']:.3f} "
            f"CONTACT_VS={first_contact['vertical_speed_fps']:+.3f} | "
            f"LANDED_HOLD={landed_hold:.2f}s "
            f"WOW={result['final_wow']} "
            f"BOUNCE={result['bounce']} | "
            f"FWDerr={result['final_position_error_ft']:+.3f} "
            f"V={result['final_forward_speed_fps']:+.3f} | "
            f"X={result['final_cross_track_ft']:+.3f} "
            f"LAT={result['final_lateral_speed_fps']:+.3f}"
        )

    close_handoff(start)

    return result, trace




# =====================================================================
# LOCKED FULL STAGE-4 TEACHER FINAL V1
# =====================================================================

# Selected and independently revalidated in
# calibrate_stage4_continuous_landed_hold_v2.py
LOCKED_LANDED_TARGET_COLLECTIVE = 0.540944



# =====================================================================
# STAGE-4 CORRECTIVE DISTILLED V2 — TEACHER-OFF JSBSIM VALIDATION
# =====================================================================
#
# From the Stage-4 handoff onward:
#   - Stage-4 teacher controller OFF
#   - Stage-4 classical vertical/longitudinal/lateral controllers OFF
#   - touchdown teacher OFF
#   - landed-hold teacher OFF
#   - distilled neural policy commands ALL FOUR physical actuators
#
# The finite-state mission manager remains active only to select the phase
# one-hot feature and acceptance/transition criteria. It does NOT generate
# actuator commands.
#
# Locked Stage1/2/3 are used only to reproduce the qualified Stage-4 handoff.

BC_MODEL = Path(
    "models_stage4_corrective_distilled_v2/AH1S_STAGE4_CORRECTIVE_DISTILLED_V2.zip"
)
OBS_NORM_PATH = Path(
    "models_stage4_corrective_distilled_v2/stage4_obs_normalization.npz"
)
ACTION_MAP_PATH = Path(
    "models_stage4_corrective_distilled_v2/stage4_action_mapping.npz"
)

TEACHER_OFF_RESULT_DIR = Path(
    "results_stage4_teacher_off_corrective_v2"
)
TEACHER_OFF_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PHASES = [
    "pre_descent_hold",
    "descent_300_to_30",
    "near_ground_30_to_native_eq",
    "capture_to_9ft",
    "continuous_touchdown",
    "landed_hold",
]
PHASE_TO_ID = {
    name: i for i, name in enumerate(PHASES)
}

PRE_HOLD_FORWARD_ERROR_LIMIT_FT = 7.0

PHASE_TIMEOUT_S = {
    "pre_descent_hold": 60.0,
    "descent_300_to_30": 430.0,
    "near_ground_30_to_native_eq": 180.0,
    "capture_to_9ft": 180.0,
    "continuous_touchdown": 110.0,
    "landed_hold": 15.0,
}

# Physical student action bounds are loaded from the saved mapping file.


def save_rows_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )
        writer.writeheader()
        writer.writerows(rows)


def student_actor_mean(policy, obs_tensor):
    features = policy.extract_features(
        obs_tensor
    )
    latent_pi = policy.mlp_extractor.forward_actor(
        features
    )
    return policy.action_net(latent_pi)


def build_student_obs(
    state,
    contact,
    phase_name,
    obs_mean,
    obs_std,
):
    heading = state["heading_error_rad"]

    raw = np.asarray(
        [
            state["altitude_ft"],
            state["vertical_speed_fps"],
            state["position_error_ft"],
            state["forward_speed_fps"],
            state["cross_track_ft"],
            state["lateral_speed_fps"],
            math.sin(heading),
            math.cos(heading),
            state["pitch_rad"],
            state["roll_rad"],
            state["roll_rate_rad_s"],
            state["pitch_rate_rad_s"],
            state["yaw_rate_rad_s"],
            state["rotor_rpm"],
            state["physical_collective_cmd"],
            state["physical_elevator_cmd"],
            state["physical_aileron_cmd"],
            state["physical_rudder_cmd"],
            1.0 if any_wow(contact) else 0.0,
            max_compression(contact),
        ]
        + [
            1.0 if i == PHASE_TO_ID[phase_name]
            else 0.0
            for i in range(len(PHASES))
        ],
        dtype=np.float32,
    )

    if raw.shape != (26,):
        raise RuntimeError(
            f"Student observation shape mismatch: "
            f"{raw.shape}"
        )

    norm = (
        (raw - obs_mean) / obs_std
    ).astype(np.float32)

    return raw, norm


def normalized_to_physical(
    z,
    physical_low,
    physical_high,
):
    z = np.asarray(
        z,
        dtype=np.float32,
    )
    z = np.clip(
        z,
        -1.0,
        +1.0,
    )

    return (
        physical_low
        + 0.5
        * (z + 1.0)
        * (physical_high - physical_low)
    ).astype(np.float32)


def student_predict_physical(
    model,
    obs_norm,
    device,
    physical_low,
    physical_high,
):
    with torch.no_grad():
        x = torch.as_tensor(
            obs_norm.reshape(1, -1),
            dtype=torch.float32,
            device=device,
        )
        z = student_actor_mean(
            model.policy,
            x,
        )
        z = torch.clamp(
            z,
            -1.0,
            +1.0,
        )
        z_np = (
            z.cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

    physical = normalized_to_physical(
        z_np,
        physical_low,
        physical_high,
    )

    return z_np, physical


def raw_student_physical_cycle(
    env2,
    fdm,
    physical_action,
    lat0,
    lon0,
    mission_heading,
):
    a = np.asarray(
        physical_action,
        dtype=np.float32,
    ).reshape(-1)

    if a.shape != (4,):
        raise RuntimeError(
            f"Expected 4-D physical action, got {a.shape}"
        )

    fdm["fcs/collective-cmd-norm"] = float(a[0])
    fdm["fcs/elevator-cmd-norm"] = float(a[1])
    fdm["fcs/aileron-cmd-norm"] = float(a[2])
    fdm["fcs/rudder-cmd-norm"] = float(a[3])

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError(
                "JSBSim stopped during Stage-4 "
                "teacher-OFF student cycle."
            )

    state = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = float(
            state["forward_ft"]
        )

    if hasattr(env2, "steps"):
        try:
            env2.steps += 1
        except Exception:
            pass

    try:
        env2._get_obs()
    except Exception:
        pass

    return state


def common_student_safety_reason(
    state,
    phase_name,
    contact,
    endpoint_envelope_seen=False,
):
    # Corridor/safety checks only. They do not command the aircraft.
    #
    # Stage-4 is intentionally allowed to take control at ~294 ft, which is
    # about 6 ft before the 300-ft endpoint. Therefore the official ±5-ft
    # endpoint corridor cannot be applied before the endpoint hover envelope
    # has first been reached. Before that first entry, use only the qualified
    # pre-hold guard. Once the envelope has been reached, enforce ±5 ft.
    if phase_name == "pre_descent_hold" and not endpoint_envelope_seen:
        if abs(state["position_error_ft"]) > PRE_HOLD_FORWARD_ERROR_LIMIT_FT:
            return "pre_hold_forward_guard_exit"
    else:
        if abs(state["position_error_ft"]) > 5.0:
            return "forward_corridor_exit"

    if abs(state["cross_track_ft"]) > 5.0:
        return "cross_corridor_exit"

    if abs(state["heading_error_deg"]) > 2.0:
        return "heading_limit"

    if abs(
        math.degrees(
            state["pitch_rad"]
        )
    ) > 8.0:
        return "pitch_limit"

    if abs(
        math.degrees(
            state["roll_rad"]
        )
    ) > 10.0:
        return "roll_limit"

    if (
        phase_name != "landed_hold"
        and state["vertical_speed_fps"] < -2.5
    ):
        return "descent_rate_limit"

    if (
        phase_name == "continuous_touchdown"
        and not any_wow(contact)
        and state["altitude_ft"] < 5.20
    ):
        return "no_wow_below_contact_floor"

    return ""


def run_stage4_student_teacher_off(
    entry_forward_ft=294.0,
    entry_max_speed_fps=1.0,
    detailed=True,
):
    if not BC_MODEL.exists():
        raise FileNotFoundError(
            f"Missing BC model: {BC_MODEL}"
        )

    if not OBS_NORM_PATH.exists():
        raise FileNotFoundError(
            f"Missing observation normalization: "
            f"{OBS_NORM_PATH}"
        )

    if not ACTION_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing action mapping: "
            f"{ACTION_MAP_PATH}"
        )

    obs_pack = np.load(
        OBS_NORM_PATH
    )
    obs_mean = (
        obs_pack["mean"]
        .astype(np.float32)
    )
    obs_std = (
        obs_pack["std"]
        .astype(np.float32)
    )

    action_pack = np.load(
        ACTION_MAP_PATH
    )
    physical_low = (
        action_pack["physical_low"]
        .astype(np.float32)
    )
    physical_high = (
        action_pack["physical_high"]
        .astype(np.float32)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = PPO.load(
        str(BC_MODEL),
        device=device,
    )
    model.policy.eval()

    # Only locked Stage1/2/3 operate before this handoff.
    start = build_stage4_handoff(
        detailed=False,
        require_full_hold=False,
        custom_entry_forward_ft=float(
            entry_forward_ft
        ),
        custom_entry_max_speed_fps=float(
            entry_max_speed_fps
        ),
    )

    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start[
        "mission_heading"
    ]
    dt = env_control_dt(env2)

    state = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    trace = []

    phase_name = "pre_descent_hold"
    phase_elapsed = 0.0
    mission_elapsed = 0.0

    endpoint_hold_s = 0.0
    endpoint_envelope_seen = False
    settle30_s = 0.0
    hover9_s = 0.0
    landed_hold_s = 0.0

    first_contact = None
    bounce = False
    wow_loss_s = 0.0

    max_abs_pos = 0.0
    max_abs_cross = 0.0
    max_abs_heading = 0.0
    min_alt = +999.0
    min_vs = +999.0

    transition_log = []

    success = False
    termination = "unknown"

    max_total_steps = int(
        sum(PHASE_TIMEOUT_S.values())
        / dt
    ) + 100

    for step in range(max_total_steps):
        contact_before = contact_snapshot(
            fdm
        )

        obs_raw, obs_norm = (
            build_student_obs(
                state,
                contact_before,
                phase_name,
                obs_mean,
                obs_std,
            )
        )

        z_action, physical_action = (
            student_predict_physical(
                model,
                obs_norm,
                device,
                physical_low,
                physical_high,
            )
        )

        state = raw_student_physical_cycle(
            env2,
            fdm,
            physical_action,
            lat0,
            lon0,
            mission_heading,
        )

        mission_elapsed += dt
        phase_elapsed += dt

        contact = contact_snapshot(
            fdm
        )
        wow = any_wow(contact)
        compression = max_compression(
            contact
        )

        endpoint_now = bool(
            phase_name == "pre_descent_hold"
            and endpoint_hover_now(state)
        )
        endpoint_envelope_seen = bool(
            endpoint_envelope_seen or endpoint_now
        )

        # Presentation metrics start once the official endpoint hover
        # envelope has first been reached; the 294-ft braking entry is
        # intentionally outside the ±5-ft endpoint corridor.
        if (
            phase_name != "pre_descent_hold"
            or endpoint_envelope_seen
        ):
            max_abs_pos = max(
                max_abs_pos,
                abs(
                    state[
                        "position_error_ft"
                    ]
                ),
            )
            max_abs_cross = max(
                max_abs_cross,
                abs(
                    state[
                        "cross_track_ft"
                    ]
                ),
            )
            max_abs_heading = max(
                max_abs_heading,
                abs(
                    state[
                        "heading_error_deg"
                    ]
                ),
            )

        min_alt = min(
            min_alt,
            state["altitude_ft"],
        )
        min_vs = min(
            min_vs,
            state[
                "vertical_speed_fps"
            ],
        )

        row = {
            "step": int(step),
            "mission_time_s": float(
                mission_elapsed
            ),
            "phase": phase_name,
            "phase_elapsed_s": float(
                phase_elapsed
            ),
            "endpoint_hold_s": float(
                endpoint_hold_s
            ),
            "settle30_s": float(
                settle30_s
            ),
            "hover9_s": float(
                hover9_s
            ),
            "landed_hold_s": float(
                landed_hold_s
            ),
            "wow": int(wow),
            "compression_ft": float(
                compression
            ),
        }

        for i in range(26):
            row[f"obs_raw_{i}"] = float(
                obs_raw[i]
            )
            row[f"obs_norm_{i}"] = float(
                obs_norm[i]
            )

        for i in range(4):
            row[
                f"student_action_norm_{i}"
            ] = float(z_action[i])
            row[
                f"student_action_phys_{i}"
            ] = float(
                physical_action[i]
            )

        for k, v in state.items():
            if isinstance(
                v,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                row[k] = float(v)

        trace.append(row)

        # Safety checks after the student has acted.
        reason = common_student_safety_reason(
            state,
            phase_name,
            contact,
            endpoint_envelope_seen=
                endpoint_envelope_seen,
        )
        if reason:
            termination = (
                f"{phase_name}:{reason}"
            )
            break

        # -------------------------------------------------------------
        # Mission-manager transitions only. No control law here.
        # -------------------------------------------------------------
        if phase_name == "pre_descent_hold":
            if endpoint_now:
                endpoint_hold_s += dt
            else:
                endpoint_hold_s = 0.0

            if (
                endpoint_hold_s
                >= PRE_DESCENT_HOLD_SECONDS
            ):
                transition_log.append({
                    "from": phase_name,
                    "to": "descent_300_to_30",
                    "time_s": float(
                        mission_elapsed
                    ),
                    "state": state.copy(),
                })
                phase_name = (
                    "descent_300_to_30"
                )
                phase_elapsed = 0.0
                settle30_s = 0.0

        elif phase_name == "descent_300_to_30":
            if stage4_settle_now(state):
                settle30_s += dt
            else:
                settle30_s = 0.0

            if (
                settle30_s
                >= DESCENT_SETTLE_SECONDS
            ):
                transition_log.append({
                    "from": phase_name,
                    "to": (
                        "near_ground_30_to_native_eq"
                    ),
                    "time_s": float(
                        mission_elapsed
                    ),
                    "state": state.copy(),
                })
                phase_name = (
                    "near_ground_30_to_native_eq"
                )
                phase_elapsed = 0.0

        elif (
            phase_name
            == "near_ground_30_to_native_eq"
        ):
            if (
                state["altitude_ft"]
                <= PHYS_ID_TRIGGER_ALT_MAX_FT
                and abs(
                    state[
                        "vertical_speed_fps"
                    ]
                )
                <= PHYS_ID_TRIGGER_ABS_VS_MAX_FPS
            ):
                transition_log.append({
                    "from": phase_name,
                    "to": "capture_to_9ft",
                    "time_s": float(
                        mission_elapsed
                    ),
                    "state": state.copy(),
                })
                phase_name = (
                    "capture_to_9ft"
                )
                phase_elapsed = 0.0
                hover9_s = 0.0

        elif phase_name == "capture_to_9ft":
            if capture_hold_now(state):
                hover9_s += dt
            else:
                hover9_s = 0.0

            if (
                hover9_s
                >= CAPTURE_HOLD_SECONDS
            ):
                transition_log.append({
                    "from": phase_name,
                    "to": "continuous_touchdown",
                    "time_s": float(
                        mission_elapsed
                    ),
                    "state": state.copy(),
                })
                phase_name = (
                    "continuous_touchdown"
                )
                phase_elapsed = 0.0

        elif (
            phase_name
            == "continuous_touchdown"
        ):
            if wow:
                first_contact = {
                    "mission_time_s": float(
                        mission_elapsed
                    ),
                    "altitude_ft": float(
                        state["altitude_ft"]
                    ),
                    "vertical_speed_fps": float(
                        state[
                            "vertical_speed_fps"
                        ]
                    ),
                    "forward_ft": float(
                        state["forward_ft"]
                    ),
                    "position_error_ft": float(
                        state[
                            "position_error_ft"
                        ]
                    ),
                    "forward_speed_fps": float(
                        state[
                            "forward_speed_fps"
                        ]
                    ),
                    "cross_track_ft": float(
                        state[
                            "cross_track_ft"
                        ]
                    ),
                    "lateral_speed_fps": float(
                        state[
                            "lateral_speed_fps"
                        ]
                    ),
                    "heading_error_deg": float(
                        state[
                            "heading_error_deg"
                        ]
                    ),
                    "physical_collective_cmd": float(
                        state[
                            "physical_collective_cmd"
                        ]
                    ),
                    "compression_ft": float(
                        compression
                    ),
                }

                transition_log.append({
                    "from": phase_name,
                    "to": "landed_hold",
                    "time_s": float(
                        mission_elapsed
                    ),
                    "state": state.copy(),
                })
                phase_name = (
                    "landed_hold"
                )
                phase_elapsed = 0.0
                landed_hold_s = 0.0
                wow_loss_s = 0.0

        elif phase_name == "landed_hold":
            if wow:
                wow_loss_s = 0.0
            else:
                wow_loss_s += dt

            if wow_loss_s >= 0.30:
                bounce = True
                termination = (
                    "landed_hold:wow_loss_or_bounce"
                )
                break

            if landed_accept_now(
                state,
                contact,
            ):
                landed_hold_s += dt
            else:
                landed_hold_s = 0.0

            if landed_hold_s >= 5.0:
                success = True
                termination = (
                    "stable_landed_hold_5s"
                )
                break

        if (
            phase_elapsed
            > PHASE_TIMEOUT_S[phase_name]
        ):
            termination = (
                f"{phase_name}:timeout"
            )
            break

        if detailed and (
            step == 0
            or step % max(
                1,
                int(round(10.0 / dt)),
            )
            == 0
        ):
            print(
                f"t={mission_elapsed:7.2f}s | "
                f"phase={phase_name:28s} | "
                f"ALT={state['altitude_ft']:7.2f} "
                f"VS={state['vertical_speed_fps']:+6.3f} | "
                f"FWDerr={state['position_error_ft']:+6.2f} "
                f"V={state['forward_speed_fps']:+6.3f} | "
                f"X={state['cross_track_ft']:+6.2f} "
                f"LAT={state['lateral_speed_fps']:+6.3f} | "
                f"WOW={int(wow)}"
            )

    final_state = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )
    final_contact = contact_snapshot(
        fdm
    )

    touchdown_vs_ok = bool(
        first_contact is not None
        and abs(
            first_contact[
                "vertical_speed_fps"
            ]
        )
        <= 0.25
    )

    presentation_pass = bool(
        max_abs_pos <= 5.0
        and max_abs_cross <= 5.0
    )

    final_pass = bool(
        success
        and presentation_pass
        and touchdown_vs_ok
        and first_contact is not None
        and not bounce
        and any_wow(final_contact)
        and landed_hold_s >= 5.0
        and abs(
            final_state[
                "forward_speed_fps"
            ]
        )
        <= 0.60
        and abs(
            final_state[
                "lateral_speed_fps"
            ]
        )
        <= 0.60
        and abs(
            final_state[
                "heading_error_deg"
            ]
        )
        <= 1.0
    )

    result = {
        "pass": bool(final_pass),
        "stage4_teacher_runtime": False,
        "stage4_classical_controllers_runtime": False,
        "student_policy_only_after_handoff": True,
        "student_model": str(BC_MODEL),
        "corrective_distillation_version": "V2 collective-head-only",
        "reward_based_ppo_training_used": False,
        "same_fdm": bool(
            start["same_fdm"]
        ),
        "clock_reset_on_attach": bool(
            start[
                "clock_reset_on_attach"
            ]
        ),
        "entry_forward_ft": float(
            entry_forward_ft
        ),
        "entry_max_speed_fps": float(
            entry_max_speed_fps
        ),
        "actual_handoff_forward_ft": float(
            start["state"]["forward_ft"]
        ),
        "actual_handoff_forward_speed_fps": float(
            start["state"][
                "forward_speed_fps"
            ]
        ),
        "termination": str(
            termination
        ),
        "mission_elapsed_s": float(
            mission_elapsed
        ),
        "presentation_pass": bool(
            presentation_pass
        ),
        "max_abs_position_error_ft": float(
            max_abs_pos
        ),
        "max_abs_cross_track_ft": float(
            max_abs_cross
        ),
        "max_abs_heading_error_deg": float(
            max_abs_heading
        ),
        "min_altitude_ft": float(
            min_alt
        ),
        "min_vertical_speed_fps": float(
            min_vs
        ),
        "endpoint_hold_s": float(
            endpoint_hold_s
        ),
        "endpoint_envelope_seen": bool(
            endpoint_envelope_seen
        ),
        "pre_hold_forward_error_limit_ft": float(
            PRE_HOLD_FORWARD_ERROR_LIMIT_FT
        ),
        "settle30_s": float(
            settle30_s
        ),
        "hover9_s": float(
            hover9_s
        ),
        "touchdown_first_contact": (
            first_contact
        ),
        "touchdown_vs_ok": bool(
            touchdown_vs_ok
        ),
        "landed_hold_s": float(
            landed_hold_s
        ),
        "bounce": bool(
            bounce
        ),
        "final_wow": bool(
            any_wow(final_contact)
        ),
        "final_compression_ft": float(
            max_compression(
                final_contact
            )
        ),
        "final_altitude_ft": float(
            final_state[
                "altitude_ft"
            ]
        ),
        "final_vertical_speed_fps": float(
            final_state[
                "vertical_speed_fps"
            ]
        ),
        "final_forward_ft": float(
            final_state["forward_ft"]
        ),
        "final_position_error_ft": float(
            final_state[
                "position_error_ft"
            ]
        ),
        "final_forward_speed_fps": float(
            final_state[
                "forward_speed_fps"
            ]
        ),
        "final_cross_track_ft": float(
            final_state[
                "cross_track_ft"
            ]
        ),
        "final_lateral_speed_fps": float(
            final_state[
                "lateral_speed_fps"
            ]
        ),
        "final_heading_error_deg": float(
            final_state[
                "heading_error_deg"
            ]
        ),
        "transitions": transition_log,
    }

    close_handoff(start)

    return result, trace


rule(
    "A — STAGE-4 DISTILLED POLICY "
    "TEACHER-OFF VALIDATION"
)

result, trace = (
    run_stage4_student_teacher_off(
        entry_forward_ft=294.0,
        entry_max_speed_fps=1.0,
        detailed=True,
    )
)


rule(
    "B — STAGE-4 TEACHER-OFF "
    "CORRECTIVE V2 CONCLUSION"
)

print(
    "TEACHER-OFF RESULT | "
    f"PASS={result['pass']} | "
    f"same_fdm={result['same_fdm']} | "
    f"clock_reset="
    f"{result['clock_reset_on_attach']} | "
    f"termination={result['termination']}"
)

if (
    result[
        "touchdown_first_contact"
    ]
    is not None
):
    fc = result[
        "touchdown_first_contact"
    ]
    print(
        "FIRST CONTACT | "
        f"AGL={fc['altitude_ft']:.3f} "
        f"VS={fc['vertical_speed_fps']:+.3f} | "
        f"FWDerr="
        f"{fc['position_error_ft']:+.3f} "
        f"V={fc['forward_speed_fps']:+.3f} | "
        f"X={fc['cross_track_ft']:+.3f} "
        f"LAT={fc['lateral_speed_fps']:+.3f} | "
        f"COLL="
        f"{fc['physical_collective_cmd']:.6f}"
    )
else:
    print(
        "FIRST CONTACT | not reached"
    )

print(
    "FINAL | "
    f"ALT={result['final_altitude_ft']:.3f} "
    f"VS="
    f"{result['final_vertical_speed_fps']:+.3f} | "
    f"FWD={result['final_forward_ft']:.3f} "
    f"err="
    f"{result['final_position_error_ft']:+.3f} "
    f"V="
    f"{result['final_forward_speed_fps']:+.3f} | "
    f"X={result['final_cross_track_ft']:+.3f} "
    f"LAT="
    f"{result['final_lateral_speed_fps']:+.3f} | "
    f"WOW={result['final_wow']} "
    f"COMP="
    f"{result['final_compression_ft']:.4f} | "
    f"LANDED_HOLD="
    f"{result['landed_hold_s']:.2f}s"
)

print(
    "STAGE-4 TEACHER-OFF BC PASS: "
    f"{result['pass']}"
)

print(
    "READY FOR REWARD-BASED PPO "
    "FINE-TUNING: "
    f"{result['pass']}"
)

save_rows_csv(
    TEACHER_OFF_RESULT_DIR
    / "teacher_off_trace.csv",
    trace,
)

with open(
    TEACHER_OFF_RESULT_DIR
    / "final_summary.json",
    "w",
) as f:
    json.dump(
        result,
        f,
        indent=2,
    )

print("Saved:")
print(
    f"  {TEACHER_OFF_RESULT_DIR / 'teacher_off_trace.csv'}"
)
print(
    f"  {TEACHER_OFF_RESULT_DIR / 'final_summary.json'}"
)
print()
print(
    "No Stage-4 teacher or classical controller "
    "generated actuator commands after handoff in this test."
)
