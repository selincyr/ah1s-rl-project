from __future__ import annotations

"""
AH-1S / JSBSim
STAGE 4 DESCENDING-REGIME ELEVATOR AUTHORITY DIAGNOSTIC V1
==========================================================

Purpose
-------
The 300->30 ft Stage-4 descent now keeps lateral error small with the selected
lateral diagnostic working point (Kp=0.10, Kd=0.35), but the run terminates on
forward-position error while normalized action[1] is already fixed at -1.0.
Therefore ordinary longitudinal gain tuning cannot add braking authority.

This script does NOT train PPO and does NOT modify the locked Stage-1/2/3
models.  It performs system identification only.  It first reproduces the
locked Stage-3 handoff, then recreates the qualified Stage-4 early transfer,
pre-descent endpoint hold, and current descent regime.  At a repeatable point
inside the descent it applies small direct physical elevator-command residuals
around the existing action[1]=-1.0 mapping and measures forward-speed response,
position response, pitch/altitude coupling, and safety.

If additional negative physical elevator command is both effective and safe,
a later Stage-4-specific actuator mapping may be calibrated so a dedicated
Stage-4 PPO policy can command that verified physical range.  Such a mapping
would be actuator normalization/calibration, not a runtime teacher.
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
    "results_stage4_lateral_teacher_v2"
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
DESCENT_MAX_TIME = 360.0
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
# for 2 s and remained safe with a linear response.  Do not command beyond
# that tested positive authority in this calibration.
OLD_IDENTIFIED_A2_MAX = +0.170
DESCENT_TESTED_A2_MAX = +0.4283

LATERAL_KP_CANDIDATES = [0.040, 0.060, 0.080, 0.100]
LATERAL_KD_CANDIDATES = [0.15, 0.25, 0.35]

# Qualified Stage-4 mission-manager / longitudinal transition from V3.
STAGE4_ENTRY_FORWARD_FT = 294.0
STAGE4_ENTRY_MAX_SPEED_FPS = 1.00
PRE_DESCENT_HOLD_SECONDS = 5.0
FIXED_BRAKE_A1 = -1.0

# Descent horizontal teacher.  Longitudinal control is NOT re-tuned here:
# V3 qualified the actuator-bound command action[1] = -1.0 for 300 s after
# an early 294 ft transfer.  Only the lateral feedback gains are recalibrated here.
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


def build_stage4_teacher_action(
    state,
    hover_action0: float,
    descent_vmax: float,
    vs_kp: float,
    lateral_kp: float,
    lateral_kd: float,
):
    # -------------------------- vertical --------------------------
    vs_des = desired_vertical_speed(state["altitude_ft"], descent_vmax)
    vs_error = vs_des - state["vertical_speed_fps"]
    collective_residual = float(
        np.clip(
            float(vs_kp) * vs_error,
            COLLECTIVE_RESIDUAL_MIN,
            COLLECTIVE_RESIDUAL_MAX,
        )
    )
    action0 = float(np.clip(hover_action0 + collective_residual, -1.0, +1.0))

    # ------------------------ longitudinal ------------------------
    # This is not a fitted feedback law.  It is the exact actuator-bound
    # braking command that V3 qualified for a 300 s endpoint hold after the
    # early Stage-4 transfer.
    action1 = float(FIXED_BRAKE_A1)

    # --------------------------- lateral --------------------------
    lateral_corr = (
        -float(lateral_kp) * state["cross_track_ft"]
        -float(lateral_kd) * state["lateral_speed_fps"]
    )
    raw_a2 = LATERAL_TRIM_ACTION + lateral_corr
    action2 = float(np.clip(raw_a2, IDENTIFIED_A2_MIN, DESCENT_TESTED_A2_MAX))

    # Mapped rudder trim already held heading in the Stage-3 teacher.
    action3 = float(RUDDER_ACTION)

    action = np.asarray([action0, action1, action2, action3], dtype=np.float32)
    return action, {
        "vs_des_fps": float(vs_des),
        "vs_error_fps": float(vs_error),
        "collective_residual": float(collective_residual),
        "fixed_brake_action1": float(action1),
        "lateral_corr": float(lateral_corr),
        "raw_action2": float(raw_a2),
    }


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
    lateral_kp: float,
    lateral_kd: float,
    candidate_index: int,
    detailed: bool = False,
):
    descent_vmax = float(LOCKED_DESCENT_VMAX)
    vs_kp = float(LOCKED_VS_KP)
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
            "lateral_kp": float(lateral_kp),
            "lateral_kd": float(lateral_kd),
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
            "lateral_kp": float(lateral_kp),
            "lateral_kd": float(lateral_kd),
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

    crossed = {200.0: False, 100.0: False, 50.0: False, 40.0: False, 35.0: False, 32.0: False}
    next_print = 0.0

    total_steps = int(DESCENT_MAX_TIME / dt)
    for step in range(total_steps):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        action, ctrl = build_stage4_teacher_action(
            state_before,
            hover_action0=hover_action0,
            descent_vmax=descent_vmax,
            vs_kp=vs_kp,
            lateral_kp=lateral_kp,
            lateral_kd=lateral_kd,
        )

        if (
            abs(ctrl["collective_residual"] - COLLECTIVE_RESIDUAL_MIN) < 1e-8
            or abs(ctrl["collective_residual"] - COLLECTIVE_RESIDUAL_MAX) < 1e-8
        ):
            residual_bound_hits += 1
        if float(action[2]) >= DESCENT_TESTED_A2_MAX - 1e-6:
            a2_upper_bound_hits += 1

        state, used_action = raw_policy_cycle(
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

        settle_hold = settle_hold + dt if stage4_settle_now(state) else 0.0
        max_settle_hold = max(max_settle_hold, settle_hold)

        trace.append({
            "candidate_index": int(candidate_index),
            "phase": "descent",
            "time_s": float(t),
            "descent_vmax_fps": float(descent_vmax),
            "vs_kp": float(vs_kp),
            "lateral_kp": float(lateral_kp),
            "lateral_kd": float(lateral_kd),
            "hover_action0": float(hover_action0),
            "action0": float(used_action[0]),
            "action1": float(used_action[1]),
            "action2": float(used_action[2]),
            "action3": float(used_action[3]),
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
                        f"A0={used_action[0]:+7.4f} dA0={ctrl['collective_residual']:+6.3f}"
                    )

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  t={t:7.2f}s | ALT={state['altitude_ft']:7.2f} | "
                f"VS={state['vertical_speed_fps']:+6.3f}/{ctrl['vs_des_fps']:+5.2f} | "
                f"FWD={state['forward_ft']:7.2f} err={state['position_error_ft']:+6.2f} "
                f"V={state['forward_speed_fps']:+6.3f} | "
                f"X={state['cross_track_ft']:+6.2f} LAT={state['lateral_speed_fps']:+6.3f} | "
                f"HDG={state['heading_error_deg']:+5.2f} | HOLD={settle_hold:4.2f}s"
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
        "lateral_kp": float(lateral_kp),
        "lateral_kd": float(lateral_kd),
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
    # Passing comes first. Among passes: horizontal quality, then vertical
    # smoothness, then time. Non-passes are ranked only for diagnosis.
    return (
        0 if row["teacher_pass"] else 1,
        0 if row["safe"] else 1,
        max(row["max_abs_cross_track_ft"], row["max_abs_position_error_ft"]),
        row.get("action2_upper_bound_fraction", 1.0),
        abs(row["final_altitude_ft"] - DESCENT_TARGET_ALT_FT),
        abs(row["final_vertical_speed_fps"]),
        row["duration_s"],
    )


# =====================================================================
# MAIN
# =====================================================================


# =====================================================================
# STAGE-4 DESCENDING-REGIME ELEVATOR AUTHORITY DIAGNOSTIC V1
# =====================================================================

RESULT_DIR = Path("results_stage4_elevator_descent_diagnostic_v1")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the best lateral diagnostic operating point from the previous sweep.
# It is not called "locked" yet because the complete descent has not passed.
DIAG_LAT_KP = 0.100
DIAG_LAT_KD = 0.35

# Trigger the authority test before the +5 ft presentation boundary, leaving
# enough margin for short pulses while staying in the actual descent regime.
ELEVATOR_ID_TRIGGER_FORWARD_FT = 302.0
ELEVATOR_ID_MIN_ALT_FT = 150.0

# Direct physical elevator-command residuals around the action[1]=-1.0 base.
# These are deliberately small compared with the JSBSim normalized command
# range.  Negative values test additional braking authority.
PULSE_ELEVATOR_DELTAS = [-0.015, -0.012, -0.009, -0.006, -0.003, 0.000, +0.003]
PULSE_SECONDS = 2.0
RECOVERY_SECONDS = 3.0

SUSTAINED_ELEVATOR_DELTAS = [-0.003, -0.006, -0.009, -0.012]
SUSTAINED_SECONDS = 12.0


def raw_cycle_with_physical_elevator_delta(
    env2,
    fdm,
    action,
    physical_elevator_delta,
    lat0,
    lon0,
    mission_heading,
):
    """Run one control cycle, overriding only physical elevator after mapping."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -1.0, +1.0).astype(np.float32)

    env2._apply_action(action)

    base_elevator = float(fdm["fcs/elevator-cmd-norm"])
    used_elevator = float(np.clip(base_elevator + float(physical_elevator_delta), -1.0, +1.0))
    fdm["fcs/elevator-cmd-norm"] = used_elevator

    for _ in range(physics_steps(env2)):
        if not fdm.run():
            raise RuntimeError("JSBSim stopped during Stage-4 elevator authority diagnostic.")

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

    return state, action, base_elevator, used_elevator


def build_current_descent_to_trigger(trigger_forward_ft=ELEVATOR_ID_TRIGGER_FORWARD_FT):
    """Rebuild Stage1->2->3, qualify Stage4 hold, then descend to a repeatable ID state."""
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

    # Qualified pre-descent endpoint hold.  Only action[1] is forced to the
    # already-qualified -1.0 braking command; remaining channels are Stage-3 PPO.
    pre_hold = 0.0
    endpoint_entered = False
    endpoint_entry_time = float("nan")
    for step in range(int(40.0 / dt)):
        state_before = snapshot(fdm, lat0, lon0, mission_heading)
        obs3 = stage3_observation(state_before)
        base, _ = stage3_model.predict(obs3, deterministic=True)
        action = np.asarray(base, dtype=np.float32).reshape(-1).copy()
        action[1] = FIXED_BRAKE_A1

        state, _ = raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading)
        t = (step + 1) * dt
        in_endpoint = endpoint_hover_now(state)
        if in_endpoint and not endpoint_entered:
            endpoint_entered = True
            endpoint_entry_time = float(t)
        pre_hold = pre_hold + dt if in_endpoint else 0.0

        reason = stage3_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(f"Pre-descent hold failed before elevator ID: {reason}")
        if endpoint_entered:
            if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
                close_handoff(start)
                raise RuntimeError("Pre-descent hold left forward corridor before elevator ID.")
            if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
                close_handoff(start)
                raise RuntimeError("Pre-descent hold left cross corridor before elevator ID.")
        if pre_hold >= PRE_DESCENT_HOLD_SECONDS:
            break

    if pre_hold < PRE_DESCENT_HOLD_SECONDS:
        close_handoff(start)
        raise RuntimeError("Stage-4 pre-descent hold did not qualify before elevator ID.")

    pre_descent_state = snapshot(fdm, lat0, lon0, mission_heading)
    obs3 = stage3_observation(pre_descent_state)
    hover_action, _ = stage3_model.predict(obs3, deterministic=True)
    hover_action = np.asarray(hover_action, dtype=np.float32).reshape(-1)
    hover_action0 = float(hover_action[0])

    # Follow the current vertical + lateral diagnostic regime until the selected
    # forward position is reached.  No physical elevator override yet.
    descent_elapsed = 0.0
    trigger_state = None
    trigger_base_elevator = float("nan")
    max_cross = abs(pre_descent_state["cross_track_ft"])
    max_pos = abs(pre_descent_state["position_error_ft"])

    for step in range(int(180.0 / dt)):
        before = snapshot(fdm, lat0, lon0, mission_heading)
        action, ctrl = build_stage4_teacher_action(
            before,
            hover_action0=hover_action0,
            descent_vmax=LOCKED_DESCENT_VMAX,
            vs_kp=LOCKED_VS_KP,
            lateral_kp=DIAG_LAT_KP,
            lateral_kd=DIAG_LAT_KD,
        )
        state, used = raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading)
        descent_elapsed = (step + 1) * dt
        max_cross = max(max_cross, abs(state["cross_track_ft"]))
        max_pos = max(max_pos, abs(state["position_error_ft"]))

        reason = stage4_teacher_safety_reason(state)
        if reason:
            close_handoff(start)
            raise RuntimeError(f"Current descent failed before elevator ID trigger: {reason}")
        if abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT:
            close_handoff(start)
            raise RuntimeError("Current descent left cross corridor before elevator ID trigger.")
        if abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT:
            close_handoff(start)
            raise RuntimeError("Current descent left forward corridor before elevator ID trigger.")

        if (
            state["forward_ft"] >= float(trigger_forward_ft)
            and state["altitude_ft"] >= ELEVATOR_ID_MIN_ALT_FT
        ):
            trigger_state = state.copy()
            trigger_base_elevator = fdm_float(fdm, "fcs/elevator-cmd-norm")
            break

    if trigger_state is None:
        close_handoff(start)
        raise RuntimeError("Elevator ID trigger was not reached in the qualified descent corridor.")

    return {
        **start,
        "dt": float(dt),
        "hover_action0": float(hover_action0),
        "pre_descent_hold_s": float(pre_hold),
        "endpoint_entry_time_s": float(endpoint_entry_time),
        "descent_elapsed_s": float(descent_elapsed),
        "trigger_state": trigger_state,
        "trigger_base_elevator_cmd": float(trigger_base_elevator),
        "max_abs_cross_to_trigger_ft": float(max_cross),
        "max_abs_position_error_to_trigger_ft": float(max_pos),
    }


def run_elevator_pulse_case(delta_phys: float):
    start = build_current_descent_to_trigger()
    env2, fdm = start["env2"], start["fdm"]
    lat0, lon0 = start["lat0"], start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]

    s0 = snapshot(fdm, lat0, lon0, mission_heading)
    rows = []
    base_vals, used_vals = [], []
    max_pitch = abs(math.degrees(s0["pitch_rad"]))
    max_roll = abs(math.degrees(s0["roll_rad"]))
    max_cross = abs(s0["cross_track_ft"])
    max_pos = abs(s0["position_error_ft"])
    min_alt = max_alt = s0["altitude_ft"]
    min_vs = max_vs = s0["vertical_speed_fps"]
    safe = True
    term = "pulse_complete"

    for step in range(int(PULSE_SECONDS / dt)):
        before = snapshot(fdm, lat0, lon0, mission_heading)
        action, ctrl = build_stage4_teacher_action(
            before,
            hover_action0=hover_action0,
            descent_vmax=LOCKED_DESCENT_VMAX,
            vs_kp=LOCKED_VS_KP,
            lateral_kp=DIAG_LAT_KP,
            lateral_kd=DIAG_LAT_KD,
        )
        state, used, base_elev, physical_elev = raw_cycle_with_physical_elevator_delta(
            env2, fdm, action, delta_phys, lat0, lon0, mission_heading
        )
        base_vals.append(base_elev)
        used_vals.append(physical_elev)
        max_pitch = max(max_pitch, abs(math.degrees(state["pitch_rad"])))
        max_roll = max(max_roll, abs(math.degrees(state["roll_rad"])))
        max_cross = max(max_cross, abs(state["cross_track_ft"]))
        max_pos = max(max_pos, abs(state["position_error_ft"]))
        min_alt = min(min_alt, state["altitude_ft"])
        max_alt = max(max_alt, state["altitude_ft"])
        min_vs = min(min_vs, state["vertical_speed_fps"])
        max_vs = max(max_vs, state["vertical_speed_fps"])
        rows.append({
            "phase": "pulse",
            "time_s": float((step + 1) * dt),
            "physical_elevator_delta": float(delta_phys),
            "base_elevator_cmd": float(base_elev),
            "physical_elevator_cmd": float(physical_elev),
            "action0": float(used[0]), "action1": float(used[1]),
            "action2": float(used[2]), "action3": float(used[3]),
            **{k: float(v) for k, v in state.items()},
        })
        reason = stage4_teacher_safety_reason(state)
        if reason:
            safe = False
            term = reason
            break

    s_pulse = snapshot(fdm, lat0, lon0, mission_heading)

    # Short recovery with no direct physical elevator residual.
    if safe:
        for step in range(int(RECOVERY_SECONDS / dt)):
            before = snapshot(fdm, lat0, lon0, mission_heading)
            action, ctrl = build_stage4_teacher_action(
                before,
                hover_action0=hover_action0,
                descent_vmax=LOCKED_DESCENT_VMAX,
                vs_kp=LOCKED_VS_KP,
                lateral_kp=DIAG_LAT_KP,
                lateral_kd=DIAG_LAT_KD,
            )
            state, used = raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading)
            rows.append({
                "phase": "recovery",
                "time_s": float(PULSE_SECONDS + (step + 1) * dt),
                "physical_elevator_delta": 0.0,
                "base_elevator_cmd": fdm_float(fdm, "fcs/elevator-cmd-norm"),
                "physical_elevator_cmd": fdm_float(fdm, "fcs/elevator-cmd-norm"),
                "action0": float(used[0]), "action1": float(used[1]),
                "action2": float(used[2]), "action3": float(used[3]),
                **{k: float(v) for k, v in state.items()},
            })
            reason = stage4_teacher_safety_reason(state)
            if reason:
                safe = False
                term = f"recovery_{reason}"
                break

    s_end = snapshot(fdm, lat0, lon0, mission_heading)
    result = {
        "physical_elevator_delta": float(delta_phys),
        "safe": bool(safe),
        "termination_reason": str(term),
        "trigger_altitude_ft": float(s0["altitude_ft"]),
        "trigger_forward_ft": float(s0["forward_ft"]),
        "trigger_forward_speed_fps": float(s0["forward_speed_fps"]),
        "trigger_cross_track_ft": float(s0["cross_track_ft"]),
        "trigger_lateral_speed_fps": float(s0["lateral_speed_fps"]),
        "base_elevator_cmd_mean": float(np.mean(base_vals)) if base_vals else float("nan"),
        "physical_elevator_cmd_mean": float(np.mean(used_vals)) if used_vals else float("nan"),
        "pulse_end_forward_ft": float(s_pulse["forward_ft"]),
        "pulse_end_forward_speed_fps": float(s_pulse["forward_speed_fps"]),
        "delta_forward_speed_2s_fps": float(s_pulse["forward_speed_fps"] - s0["forward_speed_fps"]),
        "delta_forward_2s_ft": float(s_pulse["forward_ft"] - s0["forward_ft"]),
        "delta_vertical_speed_2s_fps": float(s_pulse["vertical_speed_fps"] - s0["vertical_speed_fps"]),
        "delta_altitude_2s_ft": float(s_pulse["altitude_ft"] - s0["altitude_ft"]),
        "max_abs_pitch_deg": float(max_pitch),
        "max_abs_roll_deg": float(max_roll),
        "max_abs_cross_track_ft": float(max_cross),
        "max_abs_position_error_ft": float(max_pos),
        "min_altitude_ft": float(min_alt),
        "max_altitude_ft": float(max_alt),
        "min_vertical_speed_fps": float(min_vs),
        "max_vertical_speed_fps": float(max_vs),
        "recovery_final_forward_speed_fps": float(s_end["forward_speed_fps"]),
        "recovery_final_forward_ft": float(s_end["forward_ft"]),
    }
    close_handoff(start)
    return result, rows


def run_sustained_case(delta_phys: float):
    start = build_current_descent_to_trigger()
    env2, fdm = start["env2"], start["fdm"]
    lat0, lon0 = start["lat0"], start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]
    hover_action0 = start["hover_action0"]
    s0 = snapshot(fdm, lat0, lon0, mission_heading)

    rows = []
    safe = True
    corridor = True
    term = "time_limit"
    base_vals, used_vals = [], []
    max_pos = abs(s0["position_error_ft"])
    max_cross = abs(s0["cross_track_ft"])
    max_pitch = abs(math.degrees(s0["pitch_rad"]))
    max_roll = abs(math.degrees(s0["roll_rad"]))

    for step in range(int(SUSTAINED_SECONDS / dt)):
        before = snapshot(fdm, lat0, lon0, mission_heading)
        action, ctrl = build_stage4_teacher_action(
            before,
            hover_action0=hover_action0,
            descent_vmax=LOCKED_DESCENT_VMAX,
            vs_kp=LOCKED_VS_KP,
            lateral_kp=DIAG_LAT_KP,
            lateral_kd=DIAG_LAT_KD,
        )
        state, used, base_elev, physical_elev = raw_cycle_with_physical_elevator_delta(
            env2, fdm, action, delta_phys, lat0, lon0, mission_heading
        )
        base_vals.append(base_elev)
        used_vals.append(physical_elev)
        max_pos = max(max_pos, abs(state["position_error_ft"]))
        max_cross = max(max_cross, abs(state["cross_track_ft"]))
        max_pitch = max(max_pitch, abs(math.degrees(state["pitch_rad"])))
        max_roll = max(max_roll, abs(math.degrees(state["roll_rad"])))
        rows.append({
            "time_s": float((step + 1) * dt),
            "physical_elevator_delta": float(delta_phys),
            "base_elevator_cmd": float(base_elev),
            "physical_elevator_cmd": float(physical_elev),
            "action0": float(used[0]), "action1": float(used[1]),
            "action2": float(used[2]), "action3": float(used[3]),
            **{k: float(v) for k, v in state.items()},
        })
        reason = stage4_teacher_safety_reason(state)
        if reason:
            safe = False
            term = reason
            break
        if (
            abs(state["position_error_ft"]) > PRESENTATION_MAX_POSITION_ERROR_FT
            or abs(state["cross_track_ft"]) > PRESENTATION_MAX_CROSS_FT
        ):
            corridor = False
            term = "presentation_corridor_exit"
            break

    sf = snapshot(fdm, lat0, lon0, mission_heading)
    # Mean forward speed during the final 2 s of available trace.
    tail_n = max(1, int(2.0 / dt))
    tail = rows[-tail_n:]
    tail_v = [r["forward_speed_fps"] for r in tail]
    result = {
        "physical_elevator_delta": float(delta_phys),
        "safe": bool(safe),
        "presentation_corridor": bool(corridor),
        "termination_reason": str(term),
        "duration_s": float(len(rows) * dt),
        "trigger_forward_ft": float(s0["forward_ft"]),
        "trigger_forward_speed_fps": float(s0["forward_speed_fps"]),
        "final_forward_ft": float(sf["forward_ft"]),
        "final_position_error_ft": float(sf["position_error_ft"]),
        "final_forward_speed_fps": float(sf["forward_speed_fps"]),
        "mean_forward_speed_last2s_fps": float(np.mean(tail_v)) if tail_v else float("nan"),
        "final_altitude_ft": float(sf["altitude_ft"]),
        "final_vertical_speed_fps": float(sf["vertical_speed_fps"]),
        "final_cross_track_ft": float(sf["cross_track_ft"]),
        "max_abs_position_error_ft": float(max_pos),
        "max_abs_cross_track_ft": float(max_cross),
        "max_abs_pitch_deg": float(max_pitch),
        "max_abs_roll_deg": float(max_roll),
        "base_elevator_cmd_mean": float(np.mean(base_vals)) if base_vals else float("nan"),
        "physical_elevator_cmd_mean": float(np.mean(used_vals)) if used_vals else float("nan"),
    }
    close_handoff(start)
    return result, rows


# =====================================================================
# A — AUDIT LOCKED STAGE 1 -> 2 -> 3
# =====================================================================
rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")
qualification = build_stage4_handoff(detailed=True, require_full_hold=True)
q = qualification["state"]
print(
    f"FULL QUALIFICATION | PASS={qualification['handoff_pass']} | "
    f"same_fdm={qualification['same_fdm']} | clock_reset={qualification['clock_reset_on_attach']} | "
    f"FWD={q['forward_ft']:.3f} V={q['forward_speed_fps']:+.3f} "
    f"X={q['cross_track_ft']:+.3f} ALT={q['altitude_ft']:.3f} "
    f"HOLD={qualification['stage3_hover_hold_s']:.2f}s"
)
handoff_summary = {k: v for k, v in qualification.items() if k not in {"env1", "env2", "fdm", "state"}}
handoff_summary["state"] = {k: float(v) for k, v in q.items()}
close_handoff(qualification)


# =====================================================================
# B — REPRODUCE CURRENT DESCENT TO THE LONGITUDINAL ID STATE
# =====================================================================
rule("B — CURRENT DESCENT LONGITUDINAL TRIGGER CHARACTERIZATION")
baseline = build_current_descent_to_trigger()
bs = baseline["trigger_state"]
print(
    f"TRIGGER | FWD={bs['forward_ft']:.3f} err={bs['position_error_ft']:+.3f} "
    f"V={bs['forward_speed_fps']:+.4f} | ALT={bs['altitude_ft']:.2f} "
    f"VS={bs['vertical_speed_fps']:+.3f} | X={bs['cross_track_ft']:+.3f} "
    f"LAT={bs['lateral_speed_fps']:+.3f} | ELEVbase={baseline['trigger_base_elevator_cmd']:+.6f} | "
    f"descent_t={baseline['descent_elapsed_s']:.2f}s"
)
trigger_summary = {
    "trigger_state": {k: float(v) for k, v in bs.items()},
    "base_elevator_cmd": float(baseline["trigger_base_elevator_cmd"]),
    "descent_elapsed_s": float(baseline["descent_elapsed_s"]),
    "pre_descent_hold_s": float(baseline["pre_descent_hold_s"]),
    "max_abs_cross_to_trigger_ft": float(baseline["max_abs_cross_to_trigger_ft"]),
    "max_abs_position_error_to_trigger_ft": float(baseline["max_abs_position_error_to_trigger_ft"]),
}
close_handoff(baseline)


# =====================================================================
# C — SHORT PHYSICAL ELEVATOR AUTHORITY PULSES
# =====================================================================
rule("C — DESCENDING-REGIME PHYSICAL ELEVATOR AUTHORITY PULSES")
print("Normalized Stage-4 action[1] remains -1.0.  Only the physical elevator command is offset after the existing mapping for this diagnostic.")
print("Negative physical residual tests additional braking authority; no PPO training and no model modification.")

pulse_results = []
pulse_traces = []
for i, delta in enumerate(PULSE_ELEVATOR_DELTAS, start=1):
    result, rows = run_elevator_pulse_case(delta)
    pulse_results.append(result)
    for row in rows:
        row = dict(row)
        row["case_index"] = int(i)
        pulse_traces.append(row)
    print(
        f"Case {i:02d} dELEV={delta:+.4f} | SAFE={result['safe']} | "
        f"ELEV={result['physical_elevator_cmd_mean']:+.6f} | "
        f"dVfwd@2s={result['delta_forward_speed_2s_fps']:+.4f} | "
        f"dFWD@2s={result['delta_forward_2s_ft']:+.3f} | "
        f"dVS={result['delta_vertical_speed_2s_fps']:+.3f} dALT={result['delta_altitude_2s_ft']:+.3f} | "
        f"PITCHmax={result['max_abs_pitch_deg']:.2f} ROLLmax={result['max_abs_roll_deg']:.2f} | "
        f"Xmax={result['max_abs_cross_track_ft']:.2f} POSmax={result['max_abs_position_error_ft']:.2f}"
    )

safe_fit = [r for r in pulse_results if r["safe"]]
fit_slope = fit_intercept = fit_r2 = float("nan")
if len(safe_fit) >= 3:
    x = np.asarray([r["physical_elevator_delta"] for r in safe_fit], dtype=float)
    y = np.asarray([r["delta_forward_speed_2s_fps"] for r in safe_fit], dtype=float)
    fit_slope, fit_intercept = np.polyfit(x, y, 1)
    yhat = fit_slope * x + fit_intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

zero_case = min(pulse_results, key=lambda r: abs(r["physical_elevator_delta"]))
negative_safe = [r for r in pulse_results if r["safe"] and r["physical_elevator_delta"] < 0.0]
useful_negative = [
    r for r in negative_safe
    if r["delta_forward_speed_2s_fps"] <= zero_case["delta_forward_speed_2s_fps"] - 0.03
]

print()
print(
    f"FIT dVfwd(2s) ~= {fit_slope:+.6f} * physical_elevator_delta {fit_intercept:+.6f}   "
    f"(R^2={fit_r2:.4f})"
)
print(f"additional negative physical elevator authority useful: {bool(useful_negative)}")
if useful_negative:
    mild = sorted(useful_negative, key=lambda r: abs(r["physical_elevator_delta"]))[0]
    print(
        f"mildest useful pulse: dELEV={mild['physical_elevator_delta']:+.4f}, "
        f"dVfwd={mild['delta_forward_speed_2s_fps']:+.4f}, "
        f"ELEVmean={mild['physical_elevator_cmd_mean']:+.6f}, "
        f"pitchMax={mild['max_abs_pitch_deg']:.2f}deg"
    )


# =====================================================================
# D — SUSTAINED LOCAL CHECKS
# =====================================================================
rule("D — SUSTAINED DESCENDING-REGIME ELEVATOR CHECKS")
sustained_results = []
sustained_traces = []
for i, delta in enumerate(SUSTAINED_ELEVATOR_DELTAS, start=1):
    result, rows = run_sustained_case(delta)
    sustained_results.append(result)
    for row in rows:
        row = dict(row)
        row["case_index"] = int(i)
        sustained_traces.append(row)
    print(
        f"Case {i:02d} dELEV={delta:+.4f} | SAFE={result['safe']} CORRIDOR={result['presentation_corridor']} "
        f"TERM={result['termination_reason']} | t={result['duration_s']:.1f}s | "
        f"FWD={result['final_forward_ft']:.3f} err={result['final_position_error_ft']:+.3f} "
        f"V={result['final_forward_speed_fps']:+.4f} Vmean2={result['mean_forward_speed_last2s_fps']:+.4f} | "
        f"ALT={result['final_altitude_ft']:.2f} VS={result['final_vertical_speed_fps']:+.3f} | "
        f"X={result['final_cross_track_ft']:+.2f} | PITCHmax={result['max_abs_pitch_deg']:.2f}"
    )

usable_sustained = [
    r for r in sustained_results
    if r["safe"]
    and r["presentation_corridor"]
    and r["mean_forward_speed_last2s_fps"] <= 0.02
]

# =====================================================================
# E — CONCLUSION / SAVE EVIDENCE
# =====================================================================
rule("E — STAGE-4 DESCENT ELEVATOR AUTHORITY CONCLUSION")
authority_measurable = bool(
    len(safe_fit) >= 3
    and np.isfinite(fit_slope)
    and (max(r["delta_forward_speed_2s_fps"] for r in safe_fit)
         - min(r["delta_forward_speed_2s_fps"] for r in safe_fit)) >= 0.05
)
direction_consistent = bool(useful_negative)
ready = bool(authority_measurable and direction_consistent and len(usable_sustained) > 0)

print(f"authority measurable: {authority_measurable}")
print(f"negative physical elevator gives additional braking: {direction_consistent}")
print(f"safe sustained candidate inside 5-ft corridor: {len(usable_sustained) > 0}")
if usable_sustained:
    best_s = sorted(
        usable_sustained,
        key=lambda r: (abs(r["physical_elevator_delta"]), abs(r["mean_forward_speed_last2s_fps"]))
    )[0]
    print(
        f"preferred sustained evidence case: dELEV={best_s['physical_elevator_delta']:+.4f}, "
        f"Vmean(last2s)={best_s['mean_forward_speed_last2s_fps']:+.4f}, "
        f"finalErr={best_s['final_position_error_ft']:+.3f}, "
        f"ELEVmean={best_s['physical_elevator_cmd_mean']:+.6f}"
    )

print(f"\nREADY FOR STAGE-4 LONGITUDINAL DESCENT MAPPING CALIBRATION: {ready}")
print("Do NOT train Stage 4 here.  If READY=True, next calibrate a Stage-4-specific normalized action[1] -> physical elevator mapping within the measured safe band, then rerun the complete 300->30 ft teacher.")

write_csv(RESULT_DIR / "elevator_authority_pulse_sweep.csv", pulse_results)
write_csv(RESULT_DIR / "elevator_authority_pulse_traces.csv", pulse_traces)
write_csv(RESULT_DIR / "elevator_sustained_sweep.csv", sustained_results)
write_csv(RESULT_DIR / "elevator_sustained_traces.csv", sustained_traces)

summary = {
    "handoff": handoff_summary,
    "trigger": trigger_summary,
    "diagnostic_operating_point": {
        "descent_vmax_fps": float(LOCKED_DESCENT_VMAX),
        "vs_kp": float(LOCKED_VS_KP),
        "lateral_kp": float(DIAG_LAT_KP),
        "lateral_kd": float(DIAG_LAT_KD),
        "normalized_action1": float(FIXED_BRAKE_A1),
        "trigger_forward_ft": float(ELEVATOR_ID_TRIGGER_FORWARD_FT),
    },
    "pulse_seconds": float(PULSE_SECONDS),
    "pulse_physical_elevator_deltas": list(PULSE_ELEVATOR_DELTAS),
    "fit": {
        "dVfwd_2s_per_physical_elevator_delta": float(fit_slope),
        "intercept": float(fit_intercept),
        "r2": float(fit_r2),
    },
    "authority_measurable": bool(authority_measurable),
    "negative_extra_braking_useful": bool(direction_consistent),
    "usable_sustained_count": int(len(usable_sustained)),
    "ready_for_stage4_longitudinal_descent_mapping_calibration": bool(ready),
    "git_head": git_head(),
    "sha256": {
        "stage1": sha256_file(STAGE1_MODEL_PATH),
        "stage2": sha256_file(STAGE2_MODEL_PATH),
        "stage3": sha256_file(STAGE3_MODEL_PATH),
    },
}
with (RESULT_DIR / "diagnostic_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print("Saved:")
print(" ", RESULT_DIR / "elevator_authority_pulse_sweep.csv")
print(" ", RESULT_DIR / "elevator_authority_pulse_traces.csv")
print(" ", RESULT_DIR / "elevator_sustained_sweep.csv")
print(" ", RESULT_DIR / "elevator_sustained_traces.csv")
print(" ", RESULT_DIR / "diagnostic_summary.json")
