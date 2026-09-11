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
    "results_stage4_vertical_pi_v1"
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
VS_KI_CANDIDATES = [0.0005, 0.0010, 0.0020, 0.0040, 0.0060]
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
    vertical_bias: float, dt: float, long_kpos_phys: float, long_kv_phys: float,
):
    vs_des = desired_vertical_speed(state["altitude_ft"], descent_vmax)
    vs_error = vs_des - state["vertical_speed_fps"]
    candidate_bias = float(np.clip(vertical_bias + vs_ki * vs_error * dt, VERTICAL_BIAS_MIN, VERTICAL_BIAS_MAX))
    capture_blend = float(np.clip((state["altitude_ft"] - DESCENT_TARGET_ALT_FT) / max(1e-6, CAPTURE_BLEND_START_ALT_FT - DESCENT_TARGET_ALT_FT), 0.0, 1.0))
    effective_bias = candidate_bias * capture_blend
    p_term = float(vs_kp) * vs_error
    residual_unclipped = p_term + effective_bias
    collective_residual = float(np.clip(residual_unclipped, COLLECTIVE_RESIDUAL_MIN, COLLECTIVE_RESIDUAL_MAX))
    pushing_low = collective_residual <= COLLECTIVE_RESIDUAL_MIN + 1e-9 and vs_error < 0.0
    pushing_high = collective_residual >= COLLECTIVE_RESIDUAL_MAX - 1e-9 and vs_error > 0.0
    if pushing_low or pushing_high:
        candidate_bias = float(vertical_bias)
        effective_bias = candidate_bias * capture_blend
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
        "capture_blend": float(capture_blend), "collective_residual_unclipped": float(residual_unclipped),
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
    vs_ki: float,
    candidate_index: int,
    detailed: bool = False,
):
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
            vs_kp=vs_kp, vs_ki=vs_ki, vertical_bias=vertical_bias, dt=dt,
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

rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")
print("Stage-1 model:", STAGE1_MODEL_PATH)
print("Stage-2 model:", STAGE2_MODEL_PATH)
print("Stage-3 model:", STAGE3_MODEL_PATH)
print("Stage-1 runtime teacher: OFF")
print("Stage-2 runtime teacher: OFF")
print("Stage-3 runtime teacher/controller: OFF")
print("Stage-3 runtime: locked PPO policy only")
print("Stage-4 PPO training: NONE")
print("Stage-4 vertical PI V1 target: eliminate the low-altitude vertical steady-state error while preserving the qualified horizontal geometry")

qualification = build_stage4_handoff(detailed=True, require_full_hold=True)
qstate = qualification["state"]
print(
    "\nHANDOFF | "
    f"PASS={qualification['handoff_pass']} | "
    f"same_fdm={qualification['same_fdm']} | "
    f"clock_reset={qualification['clock_reset_on_attach']} | "
    f"FWD={qstate['forward_ft']:.3f} | V={qstate['forward_speed_fps']:+.3f} | "
    f"X={qstate['cross_track_ft']:+.3f} | LAT={qstate['lateral_speed_fps']:+.3f} | "
    f"ALT={qstate['altitude_ft']:.3f} | VS={qstate['vertical_speed_fps']:+.3f} | "
    f"HDG={qstate['heading_error_deg']:+.3f}deg | "
    f"HOLD={qualification['stage3_hover_hold_s']:.2f}s"
)

handoff_summary = {
    k: v for k, v in qualification.items()
    if k not in {"env1", "env2", "fdm", "state"}
}
handoff_summary["state"] = qstate
close_handoff(qualification)

rule("B — STAGE-4 VERTICAL PI BIAS SWEEP DURING QUALIFIED DESCENT")
print("No Stage-4 PPO training. No Stage-4 runtime neural controller during descent.")
print("Mission-manager entry and pre-descent A1=-1.0 braking are fixed from V3 qualification.")
print("Descent begins only after Stage-4 itself holds the original endpoint envelope for >=5 s.")
print("Vertical P baseline fixed: Vmax=1.20 ft/s, VS_Kp=0.35; only VS_Ki is swept.")
print(f"Horizontal fixed at prior best geometry: LONG_Kpos={LOCKED_LONG_KPOS_PHYS:.4f}, LONG_Kv={LOCKED_LONG_KV_PHYS:.3f}; lateral Kp=0.10, Kd=0.35.")
print(f"Adaptive vertical bias bounds [{VERTICAL_BIAS_MIN:+.2f},{VERTICAL_BIAS_MAX:+.2f}], faded below {CAPTURE_BLEND_START_ALT_FT:.0f} ft.")
print(f"Action1 mapping: -1 -> {ELEVATOR_PHYSICAL_MIN:+.6f}, +1 -> {ELEVATOR_PHYSICAL_MAX:+.6f}")
print(
    f"Qualified collective residual envelope: "
    f"[{COLLECTIVE_RESIDUAL_MIN:+.2f}, {COLLECTIVE_RESIDUAL_MAX:+.2f}]"
)
print(
    f"Qualified entry: FWD>={STAGE4_ENTRY_FORWARD_FT:.1f} ft, |V|<={STAGE4_ENTRY_MAX_SPEED_FPS:.2f} ft/s; "
    f"fixed A1={FIXED_BRAKE_A1:+.2f}; tested descent A2 upper={DESCENT_TESTED_A2_MAX:+.4f}"
)

candidate_rows = []
all_candidate_traces = []
for idx, vs_ki in enumerate(VS_KI_CANDIDATES, start=1):
    print(f"Candidate {idx:2d}/{len(VS_KI_CANDIDATES)} | VS_Ki={vs_ki:.4f}")
    result, trace = run_teacher_case(vs_ki=vs_ki, candidate_index=idx, detailed=False)
    candidate_rows.append(result)
    all_candidate_traces.extend(trace)
    print(
        f"  PASS={result['teacher_pass']} SAFE={result['safe']} TERM={result['termination_reason']} | "
        f"t={result['duration_s']:.1f}s | ALT={result['final_altitude_ft']:.2f} VS={result['final_vertical_speed_fps']:+.3f} | "
        f"FWD={result['final_forward_ft']:.2f} err={result['final_position_error_ft']:+.2f} V={result['final_forward_speed_fps']:+.3f} | "
        f"X={result['final_cross_track_ft']:+.2f} LAT={result['final_lateral_speed_fps']:+.3f} | "
        f"MAXerr={result['max_abs_position_error_ft']:.2f} XMAX={result['max_abs_cross_track_ft']:.2f} | "
        f"VBias=[{result['vertical_bias_min_seen']:+.3f},{result['vertical_bias_max_seen']:+.3f}] | "
        f"dA0=[{result['collective_residual_min_seen']:+.3f},{result['collective_residual_max_seen']:+.3f}]"
    )

write_csv(RESULT_DIR / "vertical_pi_sweep.csv", candidate_rows)
write_csv(RESULT_DIR / "vertical_pi_candidate_traces.csv", all_candidate_traces)

passing = [r for r in candidate_rows if r["teacher_pass"]]
ranked = sorted(candidate_rows, key=candidate_score)

rule("C — STAGE-4 VERTICAL PI SELECTION")
for rank, row in enumerate(ranked[: min(6, len(ranked))], start=1):
    print(
        f"#{rank} C{row['candidate_index']:02d} | PASS={row['teacher_pass']} SAFE={row['safe']} | "
        f"VS_Ki={row['vs_ki']:.4f} | "
        f"t={row['duration_s']:.1f}s | ALT={row['final_altitude_ft']:.2f} "
        f"VS={row['final_vertical_speed_fps']:+.3f} | "
        f"MAXerr={row['max_abs_position_error_ft']:.2f} XMAX={row['max_abs_cross_track_ft']:.2f}"
    )

if not passing:
    best = ranked[0]
    diagnostic = {
        "ready": False,
        "reason": "No Stage-4 vertical PI candidate passed the full 300->30 ft descent. Do not distill or proceed to landing.",
        "best_nonpassing_candidate": best,
        "candidate_count": len(candidate_rows),
        "git_head": git_head(),
        "stage1_sha256": sha256_file(STAGE1_MODEL_PATH),
        "stage2_sha256": sha256_file(STAGE2_MODEL_PATH),
        "stage3_sha256": sha256_file(STAGE3_MODEL_PATH),
    }
    with (RESULT_DIR / "diagnostic_summary.json").open("w") as f:
        json.dump(diagnostic, f, indent=2)
    print("\nNO PASSING VERTICAL-PI CANDIDATE.")
    print("Saved diagnostic evidence. Do NOT train Stage 4 and do NOT tune landing yet.")
    raise SystemExit(2)

best = sorted(passing, key=candidate_score)[0]
print(
    "\nSELECTED: "
    f"C{best['candidate_index']:02d} | VS_Ki={best['vs_ki']:.4f}"
)

rule("D — FRESH DETAILED REPEAT OF SELECTED VERTICAL PI")
final_result, final_trace = run_teacher_case(
    vs_ki=best["vs_ki"], candidate_index=best["candidate_index"], detailed=True,
)

write_csv(RESULT_DIR / "best_teacher_trace.csv", final_trace)

final_summary = {
    "stage": "Stage 4 vertical PI V1: adaptive collective bias during qualified 300->30 ft descent",
    "teacher_pass": bool(final_result["teacher_pass"]),
    "training_performed": False,
    "reinforcement_learning_performed": False,
    "policy_distillation_performed": False,
    "full_landing_attempted": False,
    "touchdown_attempted": False,
    "same_continuous_fdm": bool(final_result["same_fdm"]),
    "clock_reset_on_attach": bool(final_result["clock_reset_on_attach"]),
    "locked_models": {
        "stage1": str(STAGE1_MODEL_PATH),
        "stage2": str(STAGE2_MODEL_PATH),
        "stage3": str(STAGE3_MODEL_PATH),
    },
    "system_identification_basis": {
        "dvs_2s_slope": ID_DVS_2S_SLOPE,
        "dvs_2s_intercept": ID_DVS_2S_INTERCEPT,
        "physical_collective_slope": ID_PHYSICAL_COLLECTIVE_SLOPE,
        "qualified_collective_residual_min": COLLECTIVE_RESIDUAL_MIN,
        "qualified_collective_residual_max": COLLECTIVE_RESIDUAL_MAX,
        "elevator_dvfwd_2s_slope": ID_ELEV_DV_2S_SLOPE,
        "elevator_dvfwd_2s_intercept": ID_ELEV_DV_2S_INTERCEPT,
        "sustained_safe_physical_elevator_min": ELEVATOR_PHYSICAL_MIN,
        "qualified_release_physical_elevator_max": ELEVATOR_PHYSICAL_MAX,
    },
    "selected_parameters": {
        "descent_vmax_fps": float(LOCKED_DESCENT_VMAX),
        "vs_kp": float(LOCKED_VS_KP),
        "alt_to_vs_gain": ALT_TO_VS_GAIN,
        "target_altitude_ft": DESCENT_TARGET_ALT_FT,
        "stage4_entry_forward_ft": STAGE4_ENTRY_FORWARD_FT,
        "stage4_entry_max_speed_fps": STAGE4_ENTRY_MAX_SPEED_FPS,
        "pre_descent_hold_seconds": PRE_DESCENT_HOLD_SECONDS,
        "prehold_fixed_brake_action1": FIXED_BRAKE_A1,
        "vs_ki": float(best["vs_ki"]),
        "vertical_bias_min": VERTICAL_BIAS_MIN,
        "vertical_bias_max": VERTICAL_BIAS_MAX,
        "capture_blend_start_alt_ft": CAPTURE_BLEND_START_ALT_FT,
        "long_kpos_phys": float(LOCKED_LONG_KPOS_PHYS),
        "long_kv_phys": float(LOCKED_LONG_KV_PHYS),
        "elevator_physical_min": ELEVATOR_PHYSICAL_MIN,
        "elevator_physical_max": ELEVATOR_PHYSICAL_MAX,
        "locked_lateral_kp": LOCKED_LATERAL_KP,
        "locked_lateral_kd": LOCKED_LATERAL_KD,
        "lateral_trim_action": LATERAL_TRIM_ACTION,
        "old_action2_upper": OLD_IDENTIFIED_A2_MAX,
        "tested_descent_action2_upper": DESCENT_TESTED_A2_MAX,
    },
    "acceptance": {
        "target_altitude_tolerance_ft": DESCENT_ALT_TOL_FT,
        "vertical_speed_tolerance_fps": DESCENT_VS_TOL_FPS,
        "settle_hold_seconds": DESCENT_SETTLE_SECONDS,
        "max_abs_forward_position_error_ft": PRESENTATION_MAX_POSITION_ERROR_FT,
        "max_abs_cross_track_ft": PRESENTATION_MAX_CROSS_FT,
    },
    "fresh_repeat": final_result,
    "git_head": git_head(),
    "sha256": {
        "stage1": sha256_file(STAGE1_MODEL_PATH),
        "stage2": sha256_file(STAGE2_MODEL_PATH),
        "stage3": sha256_file(STAGE3_MODEL_PATH),
    },
}

with (RESULT_DIR / "final_summary.json").open("w") as f:
    json.dump(final_summary, f, indent=2)

rule("STAGE 4 VERTICAL PI V1 — RESULT")
print(
    f"PASS={final_result['teacher_pass']} | SAFE={final_result['safe']} | "
    f"same_fdm={final_result['same_fdm']} | clock_reset={final_result['clock_reset_on_attach']}"
)
print(
    f"Final: ALT={final_result['final_altitude_ft']:.3f} ft | "
    f"VS={final_result['final_vertical_speed_fps']:+.3f} ft/s | "
    f"FWD={final_result['final_forward_ft']:.3f} ft | "
    f"err={final_result['final_position_error_ft']:+.3f} ft | "
    f"V={final_result['final_forward_speed_fps']:+.3f} ft/s | "
    f"X={final_result['final_cross_track_ft']:+.3f} ft | "
    f"LAT={final_result['final_lateral_speed_fps']:+.3f} ft/s | "
    f"HDG={final_result['final_heading_error_deg']:+.3f} deg | "
    f"HOLD={final_result['settle_hold_s']:.2f}s"
)
print(
    f"Whole descent: max|pos err|={final_result['max_abs_position_error_ft']:.3f} ft | "
    f"max|cross|={final_result['max_abs_cross_track_ft']:.3f} ft | "
    f"VSmin={final_result['min_vertical_speed_fps']:+.3f} ft/s | "
    f"pitchmax={final_result['max_abs_pitch_deg']:.3f} deg | "
    f"rollmax={final_result['max_abs_roll_deg']:.3f} deg | "
    f"hdgmax={final_result['max_abs_heading_error_deg']:.3f} deg"
)
print("Saved:")
print(" ", RESULT_DIR / "vertical_pi_sweep.csv")
print(" ", RESULT_DIR / "vertical_pi_candidate_traces.csv")
print(" ", RESULT_DIR / "best_teacher_trace.csv")
print(" ", RESULT_DIR / "final_summary.json")

if not final_result["teacher_pass"]:
    raise RuntimeError(
        "Selected candidate did not reproduce on the fresh repeat. Do not lock or distill Stage 4."
    )

print("\nSTAGE-4 300->30 FT DESCENT TEACHER QUALIFIED WITH CALIBRATED LONGITUDINAL MAPPING.")
print("NEXT: lock this 300->30 ft teacher regime, then calibrate the near-ground 30 ft -> touchdown regime before any Stage-4 PPO training.")
