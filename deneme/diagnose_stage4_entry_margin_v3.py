from __future__ import annotations

"""
AH-1S / JSBSim
STAGE 4 ENTRY / LONG-HORIZON BRAKING DIAGNOSTIC V3
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



# =====================================================================
# STAGE-4 ENTRY-MARGIN / LONG-HORIZON BRAKING DIAGNOSTIC V3
# =====================================================================
# V2 proved that the useful elevator region is pinned at the measured lower
# action bound A1=-1.0.  All feedback candidates were 100% saturated, so their
# identical PASS results do not constitute a tuned feedback law.  Before the
# 300->30 ft descent (which can last several minutes), V3 checks whether an
# earlier mission-manager transfer plus the qualified maximum braking command
# can establish a long-horizon endpoint hold with real margin.

RESULT_DIR = Path("results_stage4_entry_margin_diagnostic_v3")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LONG_HORIZON_S = 300.0
ENTRY_FORWARD_CANDIDATES_FT = [294.0, 295.0, 296.0, 297.0, 298.0, 299.0]
ENTRY_MAX_SPEED_FPS = 1.00
FIXED_BRAKE_A1 = -1.0

# The transition may begin slightly before the +/-5 ft endpoint corridor, but
# descent remains prohibited until the full endpoint envelope has been entered
# and held continuously for >=5 s.  Once entered, leaving the presentation
# corridor is a failure for this diagnostic.
POST_ENTRY_MAX_POS_ERR_FT = 5.0
POST_ENTRY_MAX_CROSS_FT = 5.0


def run_long_horizon_entry_case(entry_forward_ft: float, detailed=False):
    start = build_stage4_handoff(
        detailed=False,
        require_full_hold=False,
        custom_entry_forward_ft=float(entry_forward_ft),
        custom_entry_max_speed_fps=ENTRY_MAX_SPEED_FPS,
    )
    env2, fdm = start["env2"], start["fdm"]
    lat0, lon0, hdg = start["lat0"], start["lon0"], start["mission_heading"]
    dt = env_control_dt(env2)

    trace = []
    termination_reason = "time_limit"
    current_hold = 0.0
    max_hold = 0.0
    endpoint_entered = False
    endpoint_entry_time = float("nan")
    max_abs_pos_after_entry = 0.0
    max_abs_cross_after_entry = 0.0
    next_print = 0.0

    for step in range(int(LONG_HORIZON_S / dt)):
        state_before = snapshot(fdm, lat0, lon0, hdg)
        obs3 = stage3_observation(state_before)
        base, _ = stage3_model.predict(obs3, deterministic=True)
        action = np.asarray(base, dtype=np.float32).reshape(-1).copy()

        # Isolate the longitudinal channel: use only the already-qualified
        # maximum braking action. Other channels remain the locked Stage-3 PPO.
        action[1] = FIXED_BRAKE_A1
        state, used = raw_policy_cycle(env2, fdm, action, lat0, lon0, hdg)
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

        current_hold = current_hold + dt if in_endpoint else 0.0
        max_hold = max(max_hold, current_hold)

        trace.append({
            "time_s": float(t),
            **state,
            "entry_trigger_forward_ft": float(entry_forward_ft),
            "entry_initial_forward_ft": float(start["state"]["forward_ft"]),
            "entry_initial_forward_speed_fps": float(start["state"]["forward_speed_fps"]),
            "action0": float(used[0]),
            "action1": float(used[1]),
            "action2": float(used[2]),
            "action3": float(used[3]),
            "endpoint_now": bool(in_endpoint),
            "endpoint_entered": bool(endpoint_entered),
            "hold_s": float(current_hold),
        })

        # Broad safety first.
        reason = longitudinal_safe_reason(state)
        if reason:
            termination_reason = reason
            break

        # Once the accepted endpoint envelope has been reached, it must not be
        # abandoned during this long-horizon pre-descent qualification.
        if endpoint_entered:
            if abs(state["position_error_ft"]) > POST_ENTRY_MAX_POS_ERR_FT:
                termination_reason = "left_endpoint_position_corridor"
                break
            if abs(state["cross_track_ft"]) > POST_ENTRY_MAX_CROSS_FT:
                termination_reason = "left_endpoint_cross_corridor"
                break

        if detailed and t + 1e-9 >= next_print:
            print(
                f"  t={t:6.1f}s | FWD={state['forward_ft']:7.2f} "
                f"err={state['position_error_ft']:+6.2f} | "
                f"V={state['forward_speed_fps']:+7.4f} | "
                f"X={state['cross_track_ft']:+6.2f} LAT={state['lateral_speed_fps']:+6.3f} | "
                f"ALT={state['altitude_ft']:7.2f} VS={state['vertical_speed_fps']:+6.3f} | "
                f"HOLD={current_hold:6.2f}s"
            )
            next_print += 15.0

    final = trace[-1]
    final_ok = endpoint_hover_now(final)
    pass_case = bool(
        termination_reason == "time_limit"
        and endpoint_entered
        and max_hold >= PRE_DESCENT_HOLD_SECONDS
        and current_hold >= PRE_DESCENT_HOLD_SECONDS
        and max_abs_pos_after_entry <= POST_ENTRY_MAX_POS_ERR_FT
        and max_abs_cross_after_entry <= POST_ENTRY_MAX_CROSS_FT
        and final_ok
    )

    result = {
        "pass": bool(pass_case),
        "safe": bool(termination_reason == "time_limit"),
        "termination_reason": termination_reason,
        "entry_trigger_forward_ft": float(entry_forward_ft),
        "entry_max_speed_fps": float(ENTRY_MAX_SPEED_FPS),
        "entry_forward_ft": float(start["state"]["forward_ft"]),
        "entry_position_error_ft": float(start["state"]["position_error_ft"]),
        "entry_forward_speed_fps": float(start["state"]["forward_speed_fps"]),
        "entry_cross_track_ft": float(start["state"]["cross_track_ft"]),
        "entry_altitude_ft": float(start["state"]["altitude_ft"]),
        "endpoint_entered": bool(endpoint_entered),
        "endpoint_entry_time_s": float(endpoint_entry_time),
        "max_hold_s": float(max_hold),
        "current_hold_s": float(current_hold),
        "duration_s": float(final["time_s"]),
        "final_forward_ft": float(final["forward_ft"]),
        "final_position_error_ft": float(final["position_error_ft"]),
        "final_forward_speed_fps": float(final["forward_speed_fps"]),
        "final_cross_track_ft": float(final["cross_track_ft"]),
        "final_lateral_speed_fps": float(final["lateral_speed_fps"]),
        "final_altitude_ft": float(final["altitude_ft"]),
        "final_vertical_speed_fps": float(final["vertical_speed_fps"]),
        "final_heading_error_deg": float(final["heading_error_deg"]),
        "max_abs_position_error_after_endpoint_entry_ft": float(max_abs_pos_after_entry),
        "max_abs_cross_after_endpoint_entry_ft": float(max_abs_cross_after_entry),
        "same_fdm": bool(start["same_fdm"]),
        "clock_reset_on_attach": bool(start["clock_reset_on_attach"]),
        "fixed_action1": float(FIXED_BRAKE_A1),
    }
    close_handoff(start)
    return result, trace


def long_score(r):
    # First require a genuine 300 s pass. Among passes prefer the smallest
    # long-horizon endpoint bias and most centered final position.
    return (
        0 if r["pass"] else 1,
        0 if r["safe"] else 1,
        r["max_abs_position_error_after_endpoint_entry_ft"],
        abs(r["final_position_error_ft"]),
        abs(r["final_forward_speed_fps"]),
    )


rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")
qualification = build_stage4_handoff(detailed=True, require_full_hold=True)
qs = qualification["state"]
print(
    f"FULL QUALIFICATION | PASS={qualification['handoff_pass']} | "
    f"same_fdm={qualification['same_fdm']} | clock_reset={qualification['clock_reset_on_attach']} | "
    f"FWD={qs['forward_ft']:.3f} V={qs['forward_speed_fps']:+.3f} "
    f"X={qs['cross_track_ft']:+.3f} ALT={qs['altitude_ft']:.3f} "
    f"HOLD={qualification['stage3_hover_hold_s']:.2f}s"
)
qualification_summary = {
    "pass": bool(qualification["handoff_pass"]),
    "same_fdm": bool(qualification["same_fdm"]),
    "clock_reset_on_attach": bool(qualification["clock_reset_on_attach"]),
    "stage3_hover_hold_s": float(qualification["stage3_hover_hold_s"]),
    "state": qs,
}
close_handoff(qualification)

rule("B — EARLIER STAGE-4 BRAKING-ENTRY SWEEP")
print(
    "Stage-4 may take over during the final braking approach, but descent is forbidden "
    "until the original endpoint envelope has subsequently been held for >=5 s."
)
print(
    f"Each case then holds A1={FIXED_BRAKE_A1:+.2f} for {LONG_HORIZON_S:.0f}s. "
    "Other channels remain the locked Stage-3 PPO."
)

summaries = []
all_traces = []
for i, trigger in enumerate(ENTRY_FORWARD_CANDIDATES_FT, 1):
    r, tr = run_long_horizon_entry_case(trigger, detailed=False)
    summaries.append(r)
    all_traces.extend([{**row, "case": i} for row in tr])
    print(
        f"C{i:02d} trigger={trigger:6.1f} | entry FWD={r['entry_forward_ft']:7.2f} "
        f"V={r['entry_forward_speed_fps']:+.3f} | PASS={r['pass']} SAFE={r['safe']} "
        f"TERM={r['termination_reason']} | endpoint@{r['endpoint_entry_time_s']:.2f}s | "
        f"FWDf={r['final_forward_ft']:.2f} err={r['final_position_error_ft']:+.2f} "
        f"Vf={r['final_forward_speed_fps']:+.4f} | "
        f"MAXerr_after={r['max_abs_position_error_after_endpoint_entry_ft']:.2f} "
        f"XMAX_after={r['max_abs_cross_after_endpoint_entry_ft']:.2f} | "
        f"HOLD={r['current_hold_s']:.1f}s"
    )

write_csv(RESULT_DIR / "entry_margin_sweep.csv", summaries)
write_csv(RESULT_DIR / "entry_margin_traces.csv", all_traces)

ranked = sorted(summaries, key=long_score)
passing = [r for r in ranked if r["pass"]]

rule("C — LONG-HORIZON ENTRY-MARGIN CONCLUSION")
for rank, r in enumerate(ranked, 1):
    print(
        f"#{rank} trigger={r['entry_trigger_forward_ft']:.1f} | PASS={r['pass']} SAFE={r['safe']} | "
        f"entry={r['entry_forward_ft']:.2f}/{r['entry_forward_speed_fps']:+.3f} | "
        f"final={r['final_forward_ft']:.2f}/{r['final_forward_speed_fps']:+.4f} | "
        f"MAXerr_after={r['max_abs_position_error_after_endpoint_entry_ft']:.2f} "
        f"HOLD={r['current_hold_s']:.1f}s"
    )

ready = bool(passing)
selected = passing[0] if passing else ranked[0]

if ready:
    print("\nFresh detailed repeat of selected long-horizon entry:")
    fresh, best_trace = run_long_horizon_entry_case(
        selected["entry_trigger_forward_ft"], detailed=True
    )
    write_csv(RESULT_DIR / "best_entry_trace.csv", best_trace)
    print(
        f"FRESH BEST | PASS={fresh['pass']} SAFE={fresh['safe']} | "
        f"trigger={fresh['entry_trigger_forward_ft']:.1f} entryFWD={fresh['entry_forward_ft']:.2f} "
        f"entryV={fresh['entry_forward_speed_fps']:+.3f} | "
        f"FWD={fresh['final_forward_ft']:.3f} err={fresh['final_position_error_ft']:+.3f} "
        f"V={fresh['final_forward_speed_fps']:+.4f} X={fresh['final_cross_track_ft']:+.3f} "
        f"ALT={fresh['final_altitude_ft']:.3f} HOLD={fresh['current_hold_s']:.1f}s"
    )
    selected = fresh

summary = {
    "ready_for_descent_teacher_retest": bool(ready and selected.get("pass", False)),
    "reason": (
        "At least one earlier Stage-4 braking transfer held the validated endpoint envelope for the full long horizon."
        if ready else
        "No candidate maintained the endpoint envelope for the long horizon; do not retest descent yet."
    ),
    "long_horizon_s": float(LONG_HORIZON_S),
    "entry_max_speed_fps": float(ENTRY_MAX_SPEED_FPS),
    "fixed_brake_action1": float(FIXED_BRAKE_A1),
    "qualification": qualification_summary,
    "selected": selected,
    "all_candidates": summaries,
    "git_head": git_head(),
    "stage1_sha256": sha256_file(STAGE1_MODEL_PATH),
    "stage2_sha256": sha256_file(STAGE2_MODEL_PATH),
    "stage3_sha256": sha256_file(STAGE3_MODEL_PATH),
}
with (RESULT_DIR / "diagnostic_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print(f"\nREADY FOR DESCENT TEACHER RETEST: {summary['ready_for_descent_teacher_retest']}")
print("Saved:")
print(" ", RESULT_DIR / "entry_margin_sweep.csv")
print(" ", RESULT_DIR / "entry_margin_traces.csv")
if ready:
    print(" ", RESULT_DIR / "best_entry_trace.csv")
print(" ", RESULT_DIR / "diagnostic_summary.json")
print("\nDo NOT train Stage 4 here. This script only qualifies the mission-manager entry timing and long-horizon braking margin.")
