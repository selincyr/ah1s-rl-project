from __future__ import annotations

"""
AH-1S / JSBSim
STAGE 4 LONGITUDINAL STATION-HOLD DIAGNOSTIC V1
================================================

Purpose
-------
Calibrate a deterministic Stage-4 classical teacher for the first landing
regime: descend from the locked Stage-3 endpoint hover (~300 ft AGL) to a
stable 30 ft AGL hover while holding the endpoint XY/heading. This script
does NOT train a Stage-4 PPO and does NOT modify any locked Stage-1/2/3 model.

Evidence basis
--------------
The Stage-4 collective diagnostics established:
- negative action[0] residual produces descent;
- dVS(2 s) ~= 2.226895 * residual + 0.002683 (R^2 ~= 1.0);
- residual -0.25 gives sustained ~-0.58 ft/s;
- residual -0.45 gives sustained ~-1.07 ft/s with no action saturation.

Teacher design
--------------
1) Rebuild the true continuous locked Stage1 -> Stage2 -> Stage3 mission and
   require the 5 s endpoint-hover handoff. No FDM reset is allowed.
2) Freeze a nominal hover collective action from the locked Stage-3 policy at
   the handoff state. The Stage-4 vertical teacher then commands a bounded
   residual around that reference. Residual bounds stay inside the physically
   tested authority envelope [-0.50, +0.15].
3) Desired vertical speed is generated from altitude error to 30 ft AGL and
   tapered automatically near the target.
4) A velocity-feedback term tracks that descent-rate reference.
5) Endpoint station keeping is classical and explicit on elevator/aileron;
   rudder remains at its validated mapped trim command.
6) Sweep only the two vertical quantities justified by the ID data: maximum
   descent rate and vertical-speed feedback gain. Horizontal gains are held
   fixed so the experiment remains interpretable.
7) Select only a candidate that remains safe, stays inside the horizontal
   presentation corridor, and settles at 30 ft for 5 s. Then repeat it fresh.

This is TEACHER CALIBRATION, not reinforcement learning and not policy
distillation. Full landing/touchdown below 30 ft is intentionally deferred to
the next near-ground/ground-effect calibration after this descent regime is
validated.
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
    "results_stage4_longitudinal_diagnostic_v1"
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
ALT_TO_VS_GAIN = 0.040
MAX_UPWARD_RECOVERY_VS = 0.30

DESCENT_VMAX_CANDIDATES = [0.80, 1.00, 1.20]
VS_KP_CANDIDATES = [0.25, 0.35, 0.45]

# Endpoint station-hold teacher. Longitudinal sign/gain structure follows the
# already validated Stage-3 braking controller, but with zero brake-lead so
# 300 ft itself is the hold point. Lateral gains are the locked V3 values.
ELEVATOR_TRIM_ACTION = 0.013725
FWD_POS_GAIN = 0.040
FWD_VMAX = 0.75
FWD_SPEED_GAIN = 0.80
PITCH_RATE_DAMP = 2.0

LATERAL_TRIM_ACTION = 0.00018325989908678578
LATERAL_KP = 0.025
LATERAL_KD = 0.10
IDENTIFIED_A2_MIN = -0.630
IDENTIFIED_A2_MAX = +0.170
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

def build_stage4_handoff(detailed=False, require_full_hold=True):
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
            # Integrated Stage-4 may take over as soon as the locked Stage-3 PPO
            # first enters the validated endpoint envelope.  Stage-4 must then
            # establish its own >=5 s pre-descent hold before any descent is allowed.
            if endpoint_hover_now(state):
                break

    handoff_pass = bool(
        (hover_hold >= STOP_HOLD_SECONDS and endpoint_hover_now(state))
        if require_full_hold
        else endpoint_hover_now(state)
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
            "Locked Stage-3 PPO never entered the validated endpoint envelope. "
            "Do not run early Stage-4 transition identification."
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
# STAGE-4 LONGITUDINAL TRANSITION / HOLD DIAGNOSTIC V2
# =====================================================================
#
# V1 established two facts:
#   1) the locked Stage-3 PPO is only a finite-time endpoint stop; after the
#      accepted 5 s hold it continues to migrate forward,
#   2) the previous Stage-4 feedback grid was structurally degenerate because
#      every candidate saturated action[1] at -1.0.
#
# V2 therefore does NOT retune vertical descent.  It changes only the mission
# transition timing and identifies a saturation-aware longitudinal hold law.
# The locked Stage-3 policy remains unchanged and is first re-qualified with
# its original 5 s criterion.  For integrated Stage-4 tests, control transfers
# at the FIRST state that satisfies the already-validated endpoint envelope;
# Stage-4 must then establish its own >=5 s hold before descent would be allowed.

RESULT_DIR = Path("results_stage4_longitudinal_diagnostic_v2")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DIAG_TIME_S = 35.0
FINE_FIXED_ACTIONS = [-1.000, -0.9975, -0.9950, -0.9925, -0.9900, -0.9850, -0.9800]

# Saturation-aware law:
#   v_des = clip(Kpos * (300-FWD), -Vlim, +Vlim)
#   A1raw = -1 + release0 + Kv*(v_des - Vfwd) + Kq*q
#   A1    = clip(A1raw, -1, -0.80)
#
# The -1 lower bound is measured/qualified elevator authority.  Feedback can
# release away from that bound when the aircraft must accelerate forward; it
# never asks for unqualified action below -1.
RELEASE0_CANDIDATES = [0.005, 0.010, 0.015, 0.020]
HOLD_KPOS_CANDIDATES = [0.08, 0.12]
HOLD_KV_CANDIDATES = [0.40, 0.80]
HOLD_VDES_LIMIT_FPS = 0.25
HOLD_Q_DAMP = 2.0
HOLD_A1_MAX = -0.80

LONG_ALT_MIN = 295.0
LONG_ALT_MAX = 305.0
LONG_MAX_POS_ERR_SAFE_FT = 10.0
LONG_MAX_CROSS_SAFE_FT = 10.0
LONG_PRESENT_POS_ERR_FT = 5.0
LONG_PRESENT_CROSS_FT = 5.0
LONG_FINAL_SPEED_TOL_FPS = 0.60
LONG_FINAL_LAT_SPEED_TOL_FPS = 0.60
LONG_HEADING_TOL_DEG = 1.0
PRE_DESCENT_HOLD_SECONDS = 5.0


def close_handoff(start):
    try:
        start["env2"].fdm = None
    except Exception:
        pass
    try:
        start["env1"].close()
    except Exception:
        pass


def longitudinal_safe_reason(state):
    if not (288.0 <= state["altitude_ft"] <= 307.0):
        return "altitude_limit"
    if abs(state["position_error_ft"]) > LONG_MAX_POS_ERR_SAFE_FT:
        return "forward_position_limit"
    if abs(state["cross_track_ft"]) > LONG_MAX_CROSS_SAFE_FT:
        return "cross_track_limit"
    if abs(math.degrees(state["pitch_rad"])) > 10.0:
        return "pitch_limit"
    if abs(math.degrees(state["roll_rad"])) > 12.0:
        return "roll_limit"
    if abs(state["heading_error_deg"]) > 5.0:
        return "heading_limit"
    return ""


def hold_now(state):
    return bool(
        abs(state["position_error_ft"]) <= LONG_PRESENT_POS_ERR_FT
        and abs(state["forward_speed_fps"]) <= LONG_FINAL_SPEED_TOL_FPS
        and abs(state["cross_track_ft"]) <= LONG_PRESENT_CROSS_FT
        and abs(state["lateral_speed_fps"]) <= LONG_FINAL_LAT_SPEED_TOL_FPS
        and LONG_ALT_MIN <= state["altitude_ft"] <= LONG_ALT_MAX
        and abs(state["vertical_speed_fps"]) <= VS_TOL_FPS
        and abs(state["heading_error_deg"]) <= LONG_HEADING_TOL_DEG
    )


def summarize_trace(trace, termination_reason="time_limit", current_hold_s=0.0):
    final = trace[-1]
    max_abs_pos = max(abs(r["position_error_ft"]) for r in trace)
    max_abs_cross = max(abs(r["cross_track_ft"]) for r in trace)
    min_alt = min(r["altitude_ft"] for r in trace)
    max_alt = max(r["altitude_ft"] for r in trace)
    min_v = min(r["forward_speed_fps"] for r in trace)
    max_v = max(r["forward_speed_fps"] for r in trace)
    sat_frac = sum(1 for r in trace if bool(r.get("a1_saturated", False))) / max(1, len(trace))
    presentation = bool(
        max_abs_pos <= LONG_PRESENT_POS_ERR_FT
        and max_abs_cross <= LONG_PRESENT_CROSS_FT
    )
    final_ok = hold_now(final)
    return {
        "pass": bool(
            termination_reason == "time_limit"
            and presentation
            and final_ok
            and current_hold_s >= PRE_DESCENT_HOLD_SECONDS
        ),
        "safe": bool(termination_reason == "time_limit"),
        "termination_reason": termination_reason,
        "duration_s": float(final["time_s"]),
        "current_hold_s": float(current_hold_s),
        "final_forward_ft": float(final["forward_ft"]),
        "final_position_error_ft": float(final["position_error_ft"]),
        "final_forward_speed_fps": float(final["forward_speed_fps"]),
        "final_cross_track_ft": float(final["cross_track_ft"]),
        "final_lateral_speed_fps": float(final["lateral_speed_fps"]),
        "final_altitude_ft": float(final["altitude_ft"]),
        "final_vertical_speed_fps": float(final["vertical_speed_fps"]),
        "final_heading_error_deg": float(final["heading_error_deg"]),
        "max_abs_position_error_ft": float(max_abs_pos),
        "max_abs_cross_track_ft": float(max_abs_cross),
        "min_altitude_ft": float(min_alt),
        "max_altitude_ft": float(max_alt),
        "min_forward_speed_fps": float(min_v),
        "max_forward_speed_fps": float(max_v),
        "a1_saturation_fraction": float(sat_frac),
    }


def run_fixed_case(a1, detailed=False):
    start = build_stage4_handoff(detailed=False, require_full_hold=False)
    env2, fdm = start["env2"], start["fdm"]
    lat0, lon0, hdg = start["lat0"], start["lon0"], start["mission_heading"]
    dt = env_control_dt(env2)
    trace=[]; term="time_limit"; hold=0.0; next_print=0.0
    for step in range(int(DIAG_TIME_S/dt)):
        state_before=snapshot(fdm,lat0,lon0,hdg)
        obs3=stage3_observation(state_before)
        base,_=stage3_model.predict(obs3,deterministic=True)
        action=np.asarray(base,dtype=np.float32).reshape(-1).copy()
        action[1]=float(a1)
        state,used=raw_policy_cycle(env2,fdm,action,lat0,lon0,hdg)
        hold = hold + dt if hold_now(state) else 0.0
        t=(step+1)*dt
        trace.append({"time_s":float(t),**state,"fixed_action1":float(a1),
                      "action0":float(used[0]),"action1":float(used[1]),
                      "action2":float(used[2]),"action3":float(used[3]),
                      "a1_saturated":bool(abs(float(used[1])+1.0)<1e-7),
                      "hold_s":float(hold)})
        reason=longitudinal_safe_reason(state)
        if detailed and t+1e-9>=next_print:
            print(f"  t={t:5.2f}s FWD={state['forward_ft']:7.2f} err={state['position_error_ft']:+6.2f} "
                  f"V={state['forward_speed_fps']:+6.3f} A1={used[1]:+.4f} HOLD={hold:4.2f}s "
                  f"X={state['cross_track_ft']:+6.2f} ALT={state['altitude_ft']:7.2f}")
            next_print += 2.0
        if reason:
            term=reason; break
    s=summarize_trace(trace,term,hold)
    s.update({"fixed_action1":float(a1),
              "entry_forward_ft":float(start["state"]["forward_ft"]),
              "entry_forward_speed_fps":float(start["state"]["forward_speed_fps"]),
              "entry_stage3_hold_s":float(start["stage3_hover_hold_s"])})
    close_handoff(start)
    return s,trace


def run_release_feedback_case(release0,kpos,kv,detailed=False):
    start = build_stage4_handoff(detailed=False, require_full_hold=False)
    env2, fdm = start["env2"], start["fdm"]
    lat0, lon0, hdg = start["lat0"], start["lon0"], start["mission_heading"]
    dt = env_control_dt(env2)
    trace=[]; term="time_limit"; hold=0.0; next_print=0.0
    for step in range(int(DIAG_TIME_S/dt)):
        state_before=snapshot(fdm,lat0,lon0,hdg)
        obs3=stage3_observation(state_before)
        base,_=stage3_model.predict(obs3,deterministic=True)
        action=np.asarray(base,dtype=np.float32).reshape(-1).copy()
        v_des=float(np.clip(kpos*state_before["position_error_ft"],
                            -HOLD_VDES_LIMIT_FPS,+HOLD_VDES_LIMIT_FPS))
        raw_a1=(-1.0 + float(release0)
                + float(kv)*(v_des-state_before["forward_speed_fps"])
                + HOLD_Q_DAMP*state_before["pitch_rate_rad_s"])
        a1=float(np.clip(raw_a1,-1.0,HOLD_A1_MAX))
        action[1]=a1
        state,used=raw_policy_cycle(env2,fdm,action,lat0,lon0,hdg)
        hold = hold + dt if hold_now(state) else 0.0
        t=(step+1)*dt
        trace.append({"time_s":float(t),**state,"release0":float(release0),
                      "kpos":float(kpos),"kv":float(kv),"v_des_fps":float(v_des),
                      "raw_action1":float(raw_a1),"action0":float(used[0]),
                      "action1":float(used[1]),"action2":float(used[2]),"action3":float(used[3]),
                      "a1_saturated":bool(raw_a1<=-1.0),"hold_s":float(hold)})
        reason=longitudinal_safe_reason(state)
        if detailed and t+1e-9>=next_print:
            print(f"  t={t:5.2f}s FWD={state['forward_ft']:7.2f} err={state['position_error_ft']:+6.2f} "
                  f"V={state['forward_speed_fps']:+6.3f} Vdes={v_des:+5.2f} A1={used[1]:+.4f} "
                  f"HOLD={hold:4.2f}s X={state['cross_track_ft']:+6.2f} ALT={state['altitude_ft']:7.2f}")
            next_print += 2.0
        if reason:
            term=reason; break
    s=summarize_trace(trace,term,hold)
    s.update({"release0":float(release0),"trim_action1":float(-1.0+release0),
              "kpos":float(kpos),"kv":float(kv),
              "entry_forward_ft":float(start["state"]["forward_ft"]),
              "entry_forward_speed_fps":float(start["state"]["forward_speed_fps"]),
              "entry_stage3_hold_s":float(start["stage3_hover_hold_s"])})
    close_handoff(start)
    return s,trace


def write_csv(path,rows):
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def git_head():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:
        return "unknown"


rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")
print("First, reproduce the locked Stage-3 >=5 s endpoint hover exactly. No Stage-4 change yet.")
qualification=build_stage4_handoff(detailed=True, require_full_hold=True)
qs=qualification["state"]
print(f"FULL QUALIFICATION | PASS={qualification['handoff_pass']} | same_fdm={qualification['same_fdm']} "
      f"| clock_reset={qualification['clock_reset_on_attach']} | FWD={qs['forward_ft']:.3f} "
      f"| V={qs['forward_speed_fps']:+.3f} | X={qs['cross_track_ft']:+.3f} "
      f"| ALT={qs['altitude_ft']:.3f} | HOLD={qualification['stage3_hover_hold_s']:.2f}s")
close_handoff(qualification)

rule("B — EARLY INTEGRATED STAGE-4 ENTRY CHARACTERIZATION")
print("For integrated flight only, Stage-4 takes over at the FIRST state inside the validated endpoint envelope.")
print("Stage-4 must then hold the endpoint for >=5 s itself before any descent is allowed.")
entry=build_stage4_handoff(detailed=False, require_full_hold=False)
es=entry["state"]
print(f"EARLY ENTRY | FWD={es['forward_ft']:.3f} err={es['position_error_ft']:+.3f} "
      f"V={es['forward_speed_fps']:+.3f} | X={es['cross_track_ft']:+.3f} "
      f"LAT={es['lateral_speed_fps']:+.3f} | ALT={es['altitude_ft']:.3f} "
      f"VS={es['vertical_speed_fps']:+.3f} | Stage3Hold={entry['stage3_hover_hold_s']:.3f}s")
early_entry_summary={
    "forward_ft":float(es["forward_ft"]),"position_error_ft":float(es["position_error_ft"]),
    "forward_speed_fps":float(es["forward_speed_fps"]),"cross_track_ft":float(es["cross_track_ft"]),
    "lateral_speed_fps":float(es["lateral_speed_fps"]),"altitude_ft":float(es["altitude_ft"]),
    "vertical_speed_fps":float(es["vertical_speed_fps"]),"heading_error_deg":float(es["heading_error_deg"]),
    "stage3_hold_s":float(entry["stage3_hover_hold_s"]),"same_fdm":bool(entry["same_fdm"]),
}
close_handoff(entry)

rule("C — FINE ELEVATOR AUTHORITY SWEEP NEAR THE -1 BOUND")
print("Only action[1] is forced. Other channels remain locked Stage-3 PPO outputs.")
fixed_summaries=[]; fixed_traces=[]
for i,a1 in enumerate(FINE_FIXED_ACTIONS,1):
    s,tr=run_fixed_case(a1,detailed=False)
    fixed_summaries.append(s); fixed_traces.extend([{**r,"case":i} for r in tr])
    print(f"Case {i:02d} A1={a1:+.4f} | PASS={s['pass']} SAFE={s['safe']} TERM={s['termination_reason']} "
          f"| FWD={s['final_forward_ft']:.2f} err={s['final_position_error_ft']:+.2f} "
          f"V={s['final_forward_speed_fps']:+.3f} | MAXerr={s['max_abs_position_error_ft']:.2f} "
          f"HOLD={s['current_hold_s']:.2f}s")

rule("D — SATURATION-AWARE LONGITUDINAL HOLD SWEEP")
print("Law: v_des=clip(Kpos*(300-FWD), +/-0.25); A1=-1+release0+Kv*(v_des-Vfwd)+2*q, clipped [-1,-0.80].")
feedback_summaries=[]; feedback_traces=[]; idx=0
for release0 in RELEASE0_CANDIDATES:
    for kpos in HOLD_KPOS_CANDIDATES:
        for kv in HOLD_KV_CANDIDATES:
            idx+=1
            s,tr=run_release_feedback_case(release0,kpos,kv,detailed=False)
            feedback_summaries.append(s); feedback_traces.extend([{**r,"candidate_index":idx} for r in tr])
            print(f"C{idx:02d} trim={s['trim_action1']:+.4f} Kp={kpos:.2f} Kv={kv:.2f} "
                  f"| PASS={s['pass']} SAFE={s['safe']} TERM={s['termination_reason']} "
                  f"| FWD={s['final_forward_ft']:.2f} err={s['final_position_error_ft']:+.2f} "
                  f"V={s['final_forward_speed_fps']:+.3f} MAXerr={s['max_abs_position_error_ft']:.2f} "
                  f"XMAX={s['max_abs_cross_track_ft']:.2f} HOLD={s['current_hold_s']:.2f}s "
                  f"sat={100*s['a1_saturation_fraction']:.1f}%")

passing=[s for s in feedback_summaries if s["pass"]]
ranked=sorted(feedback_summaries,key=lambda s:(not s["pass"],not s["safe"],
                                                s["max_abs_position_error_ft"],
                                                abs(s["final_position_error_ft"]),
                                                abs(s["final_forward_speed_fps"]),
                                                s["a1_saturation_fraction"]))

rule("E — STAGE-4 LONGITUDINAL V2 CONCLUSION")
for rank,s in enumerate(ranked[:8],1):
    print(f"#{rank} trim={s['trim_action1']:+.4f} Kp={s['kpos']:.2f} Kv={s['kv']:.2f} "
          f"| PASS={s['pass']} SAFE={s['safe']} MAXerr={s['max_abs_position_error_ft']:.2f} "
          f"finalErr={s['final_position_error_ft']:+.2f} V={s['final_forward_speed_fps']:+.3f} "
          f"HOLD={s['current_hold_s']:.2f}s sat={100*s['a1_saturation_fraction']:.1f}%")

best_trace=[]
if passing:
    best=ranked[0]
    print("\nFresh detailed repeat of selected passing candidate:")
    fresh,best_trace=run_release_feedback_case(best["release0"],best["kpos"],best["kv"],detailed=True)
    ready=bool(fresh["pass"])
    print(f"FRESH BEST | PASS={fresh['pass']} SAFE={fresh['safe']} FWD={fresh['final_forward_ft']:.3f} "
          f"err={fresh['final_position_error_ft']:+.3f} V={fresh['final_forward_speed_fps']:+.3f} "
          f"X={fresh['final_cross_track_ft']:+.3f} ALT={fresh['final_altitude_ft']:.3f} "
          f"MAXerr={fresh['max_abs_position_error_ft']:.3f} HOLD={fresh['current_hold_s']:.2f}s")
else:
    best=None; fresh=None; ready=False

write_csv(RESULT_DIR/"fine_fixed_elevator_sweep.csv",fixed_summaries)
write_csv(RESULT_DIR/"fine_fixed_elevator_traces.csv",fixed_traces)
write_csv(RESULT_DIR/"feedback_hold_sweep.csv",feedback_summaries)
write_csv(RESULT_DIR/"feedback_hold_traces.csv",feedback_traces)
if best_trace:
    write_csv(RESULT_DIR/"best_feedback_trace.csv",best_trace)

summary={
    "experiment":"stage4_longitudinal_transition_hold_diagnostic_v2",
    "seed":SEED,"git_head":git_head(),
    "stage1_model":str(STAGE1_MODEL_PATH),"stage2_model":str(STAGE2_MODEL_PATH),
    "stage3_model":str(STAGE3_MODEL_PATH),
    "full_stage3_qualification":{
        "pass":bool(qualification["handoff_pass"]),"forward_ft":float(qs["forward_ft"]),
        "forward_speed_fps":float(qs["forward_speed_fps"]),"stage3_hold_s":float(qualification["stage3_hover_hold_s"]),
    },
    "early_stage4_entry":early_entry_summary,
    "fixed_elevator_sweep":fixed_summaries,
    "feedback_hold_sweep":feedback_summaries,
    "passing_feedback_candidates":len(passing),
    "best_feedback_candidate":best,
    "fresh_best_repeat":fresh,
    "ready_for_descent_teacher_retest":bool(ready),
}
(RESULT_DIR/"diagnostic_summary.json").write_text(json.dumps(summary,indent=2))

print(f"\nREADY FOR DESCENT TEACHER RETEST: {ready}")
if ready:
    print("The vertical collective identification remains locked; next retest the 300->30 ft descent using this verified pre-descent hold/transition logic.")
else:
    print("Do NOT retune vertical descent. Review near-bound elevator authority / transition behavior first.")
print("Saved:")
for name in ["fine_fixed_elevator_sweep.csv","fine_fixed_elevator_traces.csv",
             "feedback_hold_sweep.csv","feedback_hold_traces.csv","diagnostic_summary.json"]:
    print(f"  {RESULT_DIR/name}")
if best_trace:
    print(f"  {RESULT_DIR/'best_feedback_trace.csv'}")
