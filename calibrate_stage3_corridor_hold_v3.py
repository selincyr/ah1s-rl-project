from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill
from helicopter_env_stage2_refine_mapped import HelicopterEnvStage2RefineMapped


# ============================================================
# LOCKED MODELS
# ============================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL_PATH = (
    "models_stage2_hybrid_final/"
    "AH1S_STAGE2_HYBRID_FINAL.zip"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path("results_stage3_corridor_hold_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "stage3_corridor_hold_v3_summary.csv"
TRACE_CSV = OUT_DIR / "stage3_corridor_hold_v3_trace.csv"


# ============================================================
# MISSION / UNCHANGED STAGE-3 LONGITUDINAL PARAMETERS
# ============================================================

TARGET_FORWARD_FT = 300.0
TARGET_ALT_FT = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
STAGE3_MAX_TIME = 75.0

HANDOFF_STABLE_TIME = 5.0
CONTROL_DT = 0.075

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040

TEACHER_ENABLE_FORWARD_FT = 80.0

# LOCKED longitudinal braking from V4 validation.
K_POS = 0.055
V_FWD_MAX = 9.0
BRAKE_LEAD_FT = 8.0
V_REV_MAX = 2.0
KV = 1.00
ELEVATOR_TRIM_ACTION = 0.013725
Q_DAMP = 2.0
MAX_ELEVATOR_RESIDUAL = 1.0

# LOCKED altitude behavior from V4 validation.
COLLECTIVE_BIAS = 0.22
ALT_KP = 0.030
VS_KD = 0.120
MAX_ALT_CORR = 0.45

# Keep Stage-2 lateral behavior untouched until the aircraft is slow.
CRUISE_AILERON_ACTION = -0.230
RUDDER_ACTION = 0.0

# Only after the aircraft is already approaching hover do we activate
# the precise endpoint lateral teacher.
LATERAL_ENABLE_SPEED_FPS = 1.25
LATERAL_ENABLE_FORWARD_FT = 292.0

# ------------------------------------------------------------
# STAGE-3 PRE-ENDPOINT LATERAL CORRIDOR HOLD
# ------------------------------------------------------------
# V2 proved the lateral correction sign and endpoint teacher, but the
# thresholded guard switched off after the first recovery and allowed a
# second drift excursion to about 9.74 ft.  In V3 we keep a mild lateral
# corridor hold active for the entire Stage-3 braking segment.  This does
# NOT change the locked longitudinal braking law; it only controls action[2].
CORRIDOR_KP = 0.020
CORRIDOR_KD = 0.180
MAX_CORRIDOR_CORR = 0.30

# Endpoint hover settings that already passed in V2.
DIAG_HOVER_TRIM = 0.00
DIAG_LATERAL_KP = 0.035
LATERAL_KD = 0.18
MAX_LATERAL_CORR = 0.55

# Presentation geometry target: do not lock Stage 3 if the braking path
# leaves this corridor, even if the final 5 s hover flag passes.
PRESENTATION_MAX_CROSS_FT = 5.0

# Acceptance.
STOP_POS_TOL_FT = 5.0
STOP_SPEED_TOL_FPS = 0.60
STOP_HOLD_SECONDS = 5.0

CROSS_TOL_FT = 5.0
LAT_SPEED_TOL_FPS = 0.60
LATERAL_HOLD_SECONDS = 5.0

ALT_MIN = 295.0
ALT_MAX = 305.0
VS_TOL_FPS = 0.75
HEADING_TOL_DEG = 1.0

# Safety.
ALT_SAFE_MIN = 288.0
ALT_SAFE_MAX = 312.0
MAX_ABS_PITCH_DEG = 8.0
MAX_ABS_ROLL_DEG = 10.0
MAX_CROSS_SAFE_FT = 15.0

EARTH_RADIUS_FT = 20_902_231.0


# ============================================================
# HELPERS
# ============================================================

def rule(text):
    print()
    print("=" * 146)
    print(text)
    print("=" * 146)


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
    if getattr(env, "fdm", None) is not None:
        return env.fdm

    base = getattr(env, "base_env", None)
    if base is not None and getattr(base, "fdm", None) is not None:
        return base.fdm

    raise RuntimeError("Active FDM not found.")


def latitude_deg(fdm):
    for key in ["position/lat-gc-deg", "position/lat-geod-deg"]:
        value = fdm_float(fdm, key)
        if np.isfinite(value):
            return value
    return float("nan")


def longitude_deg(fdm):
    return fdm_float(fdm, "position/long-gc-deg")


def heading_rad(fdm):
    for key in ["attitude/heading-true-rad", "attitude/psi-rad"]:
        value = fdm_float(fdm, key)
        if np.isfinite(value):
            return value
    return float("nan")


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


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


def geometry(fdm, lat0, lon0, mission_heading):
    north, east = local_ne_ft(
        latitude_deg(fdm),
        longitude_deg(fdm),
        lat0,
        lon0,
    )

    return mission_axes(
        north,
        east,
        mission_heading,
    )


def snapshot(fdm, lat0, lon0, mission_heading):
    forward, cross = geometry(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    heading_error = wrap_angle(
        heading_rad(fdm) - mission_heading
    )

    return {
        "forward_ft": float(forward),
        "position_error_ft": float(
            TARGET_FORWARD_FT - forward
        ),
        "cross_track_ft": float(cross),
        "altitude_ft": fdm_float(
            fdm, "position/h-agl-ft"
        ),
        "forward_speed_fps": fdm_float(
            fdm, "velocities/u-aero-fps", 0.0
        ),
        "lateral_speed_fps": fdm_float(
            fdm, "velocities/v-aero-fps", 0.0
        ),
        "vertical_speed_fps": fdm_float(
            fdm, "velocities/h-dot-fps", 0.0
        ),
        "pitch_deg": math.degrees(
            fdm_float(
                fdm, "attitude/pitch-rad", 0.0
            )
        ),
        "roll_deg": math.degrees(
            fdm_float(
                fdm, "attitude/roll-rad", 0.0
            )
        ),
        "pitch_rate_rad_s": fdm_float(
            fdm, "velocities/q-rad_sec", 0.0
        ),
        "heading_error_deg": math.degrees(
            heading_error
        ),
    }


def safe_state(state):
    return bool(
        ALT_SAFE_MIN <= state["altitude_ft"] <= ALT_SAFE_MAX
        and abs(state["pitch_deg"]) <= MAX_ABS_PITCH_DEG
        and abs(state["roll_deg"]) <= MAX_ABS_ROLL_DEG
        and abs(state["cross_track_ft"]) <= MAX_CROSS_SAFE_FT
    )


def raw_cycle(env2, action):
    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    action = np.clip(action, -1.0, +1.0)
    env2._apply_action(action)

    for _ in range(10):
        if not env2.fdm.run():
            raise RuntimeError(
                "JSBSim stopped during Stage-3 lateral calibration."
            )


# ============================================================
# MODELS
# ============================================================

stage1_model = PPO.load(STAGE1_MODEL_PATH)
stage2_model = PPO.load(STAGE2_MODEL_PATH)


# ============================================================
# BUILD LOCKED STAGE1 -> STAGE2 -> STAGE3 START
# ============================================================

def build_start():
    env1 = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs1, info1 = env1.reset()
    fdm = get_fdm(env1)
    active_id = id(fdm)
    mission_heading = heading_rad(fdm)

    dt1 = float(
        getattr(env1, "dt", CONTROL_DT) or CONTROL_DT
    )

    stable_time = 0.0

    for _ in range(int(STAGE1_MAX_TIME / dt1)):
        a1, _ = stage1_model.predict(
            obs1,
            deterministic=True,
        )

        (
            obs1,
            _,
            terminated,
            truncated,
            info1,
        ) = env1.step(a1)

        altitude = info_float(info1, "altitude")
        vertical_speed = info_float(
            info1, "vertical_speed"
        )
        vn = info_float(info1, "vn", 0.0)
        ve = info_float(info1, "ve", 0.0)
        hs = float(np.hypot(vn, ve))
        drift = info_float(
            info1, "drift", 999.0
        )

        stable = bool(
            295.0 <= altitude <= 305.0
            and abs(vertical_speed) <= 0.50
            and hs <= 1.0
            and drift <= 3.0
        )

        stable_time = (
            stable_time + dt1
            if stable
            else 0.0
        )

        if stable_time >= HANDOFF_STABLE_TIME:
            break

        if (
            terminated
            and not bool(
                info1.get("success", False)
            )
        ):
            raise RuntimeError(
                "Stage 1 failed before handoff."
            )

        if truncated:
            raise RuntimeError(
                "Stage 1 truncated before handoff."
            )

    if stable_time < HANDOFF_STABLE_TIME:
        raise RuntimeError(
            "Stable Stage-1 handoff not reached."
        )

    lat0 = latitude_deg(fdm)
    lon0 = longitude_deg(fdm)

    env2 = HelicopterEnvStage2RefineMapped(
        aileron_scale=AILERON_SCALE,
        rudder_scale=RUDDER_SCALE,
    )

    env2.reset()
    env2.fdm = fdm

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = 0.0

    if hasattr(env2, "target_heading"):
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
            setattr(env2, attr, 0)

    if id(get_fdm(env2)) != active_id:
        raise RuntimeError(
            "FDM continuity failed."
        )

    obs2 = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    dt2 = float(
        getattr(env2, "dt", CONTROL_DT) or CONTROL_DT
    )

    for _ in range(int(STAGE2_MAX_TIME / dt2)):
        a2, _ = stage2_model.predict(
            obs2,
            deterministic=True,
        )

        a2 = np.asarray(
            a2,
            dtype=np.float32,
        ).reshape(-1)

        (
            obs2,
            _,
            terminated,
            truncated,
            info2,
        ) = env2.step(a2)

        obs2 = np.asarray(
            obs2,
            dtype=np.float32,
        )

        forward, _cross = geometry(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        if forward >= TEACHER_ENABLE_FORWARD_FT:
            if hasattr(env2, "forward_distance"):
                env2.forward_distance = float(
                    forward
                )
            break

        if (
            terminated
            and not bool(
                info2.get("success", False)
            )
        ):
            raise RuntimeError(
                "Stage 2 failed before Stage-3 takeover."
            )

        if truncated:
            raise RuntimeError(
                "Stage 2 truncated before Stage-3 takeover."
            )

    return (
        env1,
        env2,
        fdm,
        obs2,
        lat0,
        lon0,
        mission_heading,
    )


# ============================================================
# LOCKED LONGITUDINAL + LATERAL-ONLY ENDPOINT TEACHER
# ============================================================

def teacher_action(
    obs,
    state,
    hover_trim,
    lateral_kp,
):
    base_action, _ = stage2_model.predict(
        obs,
        deterministic=True,
    )

    base_action = np.asarray(
        base_action,
        dtype=np.float32,
    ).reshape(-1)

    action = base_action.copy()

    # Locked altitude controller.
    altitude_error = (
        TARGET_ALT_FT - state["altitude_ft"]
    )

    col_corr = (
        COLLECTIVE_BIAS
        + ALT_KP * altitude_error
        - VS_KD * state["vertical_speed_fps"]
    )

    col_corr = float(
        np.clip(
            col_corr,
            -MAX_ALT_CORR,
            +MAX_ALT_CORR,
        )
    )

    action[0] = float(
        np.clip(
            base_action[0] + col_corr,
            -1.0,
            +1.0,
        )
    )

    # Locked V4 longitudinal controller.  DO NOT retune in this file.
    v_des = float(
        np.clip(
            K_POS
            * (
                state["position_error_ft"]
                - BRAKE_LEAD_FT
            ),
            -V_REV_MAX,
            +V_FWD_MAX,
        )
    )

    speed_error = (
        state["forward_speed_fps"] - v_des
    )

    elevator_residual = (
        -KV * speed_error
        + Q_DAMP * state["pitch_rate_rad_s"]
    )

    elevator_residual = float(
        np.clip(
            elevator_residual,
            -MAX_ELEVATOR_RESIDUAL,
            +MAX_ELEVATOR_RESIDUAL,
        )
    )

    action[1] = float(
        np.clip(
            ELEVATOR_TRIM_ACTION
            + elevator_residual,
            -1.0,
            +1.0,
        )
    )

    # Precise endpoint lateral teacher.
    lateral_active = bool(
        state["forward_ft"] >= LATERAL_ENABLE_FORWARD_FT
        and abs(state["forward_speed_fps"])
        <= LATERAL_ENABLE_SPEED_FPS
    )

    # Mild corridor hold is active throughout the complete Stage-3
    # braking segment.  This avoids the on/off threshold behavior seen in
    # V2 while leaving the longitudinal controller untouched.
    corridor_active = bool(not lateral_active)

    if lateral_active:
        lateral_corr = (
            -lateral_kp
            * state["cross_track_ft"]
            - LATERAL_KD
            * state["lateral_speed_fps"]
        )

        lateral_corr = float(
            np.clip(
                lateral_corr,
                -MAX_LATERAL_CORR,
                +MAX_LATERAL_CORR,
            )
        )

        action[2] = float(
            np.clip(
                hover_trim + lateral_corr,
                -1.0,
                +1.0,
            )
        )

    elif corridor_active:
        lateral_corr = (
            -CORRIDOR_KP
            * state["cross_track_ft"]
            - CORRIDOR_KD
            * state["lateral_speed_fps"]
        )

        lateral_corr = float(
            np.clip(
                lateral_corr,
                -MAX_CORRIDOR_CORR,
                +MAX_CORRIDOR_CORR,
            )
        )

        # Residual around the locked Stage-2 cruise aileron.
        action[2] = float(
            np.clip(
                CRUISE_AILERON_ACTION + lateral_corr,
                -1.0,
                +1.0,
            )
        )

    else:
        lateral_corr = 0.0
        action[2] = CRUISE_AILERON_ACTION

    action[3] = RUDDER_ACTION

    return (
        action.astype(np.float32),
        float(v_des),
        float(lateral_corr),
        bool(lateral_active),
        bool(corridor_active),
    )


# ============================================================
# ONE CASE
# ============================================================

def run_case(
    hover_trim,
    lateral_kp,
    detailed=False,
):
    (
        env1,
        env2,
        fdm,
        obs2,
        lat0,
        lon0,
        mission_heading,
    ) = build_start()

    start = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    trace = []
    safe = True

    stop_hold = 0.0
    lateral_hold = 0.0
    full_hover_hold = 0.0

    stop_ok = False
    lateral_ok = False
    endpoint_hover_ok = False

    max_cross = abs(
        start["cross_track_ft"]
    )
    min_alt = start["altitude_ft"]
    max_alt = start["altitude_ft"]

    final_state = start.copy()
    next_print = 0.0
    corridor_ever_active = False
    endpoint_teacher_ever_active = False

    for step in range(
        int(STAGE3_MAX_TIME / CONTROL_DT)
    ):
        state = snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        (
            action,
            v_des,
            lateral_corr,
            lateral_active,
            corridor_active,
        ) = teacher_action(
            obs2,
            state,
            hover_trim,
            lateral_kp,
        )

        raw_cycle(env2, action)

        state_after = snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        if hasattr(env2, "forward_distance"):
            env2.forward_distance = float(
                state_after["forward_ft"]
            )

        if hasattr(env2, "steps"):
            try:
                env2.steps += 1
            except Exception:
                pass

        obs2 = np.asarray(
            env2._get_obs(),
            dtype=np.float32,
        )

        t = (step + 1) * CONTROL_DT

        corridor_ever_active = (
            corridor_ever_active or bool(corridor_active)
        )
        endpoint_teacher_ever_active = (
            endpoint_teacher_ever_active or bool(lateral_active)
        )

        row = dict(state_after)
        row.update(
            {
                "time_s": float(t),
                "v_des_fps": float(v_des),
                "lateral_correction": float(
                    lateral_corr
                ),
                "lateral_active": bool(
                    lateral_active
                ),
                "corridor_active": bool(
                    corridor_active
                ),
                "action0": float(action[0]),
                "action1": float(action[1]),
                "action2": float(action[2]),
                "action3": float(action[3]),
            }
        )

        trace.append(row)
        final_state = state_after

        max_cross = max(
            max_cross,
            abs(state_after["cross_track_ft"]),
        )
        min_alt = min(
            min_alt,
            state_after["altitude_ft"],
        )
        max_alt = max(
            max_alt,
            state_after["altitude_ft"],
        )

        if not safe_state(state_after):
            safe = False
            break

        stop_now = bool(
            abs(
                state_after["position_error_ft"]
            )
            <= STOP_POS_TOL_FT
            and abs(
                state_after["forward_speed_fps"]
            )
            <= STOP_SPEED_TOL_FPS
        )

        lateral_now = bool(
            lateral_active
            and abs(
                state_after["cross_track_ft"]
            )
            <= CROSS_TOL_FT
            and abs(
                state_after["lateral_speed_fps"]
            )
            <= LAT_SPEED_TOL_FPS
        )

        hover_now = bool(
            stop_now
            and lateral_now
            and ALT_MIN
            <= state_after["altitude_ft"]
            <= ALT_MAX
            and abs(
                state_after["vertical_speed_fps"]
            )
            <= VS_TOL_FPS
            and abs(
                state_after["heading_error_deg"]
            )
            <= HEADING_TOL_DEG
        )

        stop_hold = (
            stop_hold + CONTROL_DT
            if stop_now
            else 0.0
        )

        lateral_hold = (
            lateral_hold + CONTROL_DT
            if lateral_now
            else 0.0
        )

        full_hover_hold = (
            full_hover_hold + CONTROL_DT
            if hover_now
            else 0.0
        )

        stop_ok = (
            stop_ok
            or stop_hold >= STOP_HOLD_SECONDS
        )

        lateral_ok = (
            lateral_ok
            or lateral_hold >= LATERAL_HOLD_SECONDS
        )

        if (
            full_hover_hold
            >= LATERAL_HOLD_SECONDS
        ):
            endpoint_hover_ok = True
            break

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"FWD={state_after['forward_ft']:7.2f} | "
                f"V={state_after['forward_speed_fps']:+5.2f} | "
                f"X={state_after['cross_track_ft']:+6.2f} | "
                f"LAT={state_after['lateral_speed_fps']:+5.2f} | "
                f"ALT={state_after['altitude_ft']:7.2f} | "
                f"A2={action[2]:+6.3f} | "
                f"CORR={corridor_active} | "
                f"END_LAT={lateral_active}"
            )

            next_print += 1.5

    result = {
        "hover_trim": float(hover_trim),
        "lateral_kp": float(lateral_kp),
        "safe": bool(safe),
        "persistent_stop": bool(stop_ok),
        "persistent_lateral_hold": bool(lateral_ok),
        "endpoint_hover": bool(endpoint_hover_ok),
        "corridor_ever_active": bool(corridor_ever_active),
        "endpoint_teacher_ever_active": bool(
            endpoint_teacher_ever_active
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
        "stop_hold_s": float(stop_hold),
        "final_cross_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "lateral_hold_s": float(lateral_hold),
        "max_cross_ft": float(max_cross),
        "presentation_cross_pass": bool(
            max_cross <= PRESENTATION_MAX_CROSS_FT
        ),
        "final_alt_ft": float(
            final_state["altitude_ft"]
        ),
        "final_vs_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "min_alt_ft": float(min_alt),
        "max_alt_ft": float(max_alt),
        "final_heading_error_deg": float(
            final_state["heading_error_deg"]
        ),
        "full_hover_hold_s": float(
            full_hover_hold
        ),
        "trace": trace,
    }

    env2.fdm = None
    env1.close()

    return result


# ============================================================
# SINGLE V3 CORRIDOR-HOLD VALIDATION
# ============================================================

rule("STAGE 3 — CONTINUOUS LATERAL CORRIDOR HOLD V3")

print("Stage 1 model: LOCKED")
print("Stage 2 model: HYBRID FINAL")
print("Stage-3 longitudinal parameters: UNCHANGED; re-validating under Hybrid Final handoff")
print("Altitude controller: UNCHANGED")
print("No PPO training in this diagnostic.")
print("Goal 1: keep the full braking path inside a clean lateral corridor.")
print("Goal 2: preserve the V2 endpoint hover pass and re-validate full geometry.")
print()
print(f"Corridor Kp / Kd : {CORRIDOR_KP:.3f} / {CORRIDOR_KD:.3f}")
print(f"Corridor max corr: {MAX_CORRIDOR_CORR:.2f}")
print(f"Presentation X   : <= {PRESENTATION_MAX_CROSS_FT:.1f} ft")
print(f"Endpoint trim/Kp : {DIAG_HOVER_TRIM:+.2f} / {DIAG_LATERAL_KP:.3f}")
print()

result = run_case(
    DIAG_HOVER_TRIM,
    DIAG_LATERAL_KP,
    detailed=True,
)

# Save full trace.
if result["trace"]:
    with TRACE_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(result["trace"][0].keys()),
        )
        writer.writeheader()
        writer.writerows(result["trace"])

# Save one-row summary.
summary_row = {
    key: value
    for key, value in result.items()
    if key != "trace"
}

with SUMMARY_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_row.keys()),
    )
    writer.writeheader()
    writer.writerow(summary_row)


# ============================================================
# FINAL
# ============================================================

rule("STAGE 3 CORRIDOR HOLD V3 — FINAL RESULT")

for key in [
    "safe",
    "persistent_stop",
    "persistent_lateral_hold",
    "endpoint_hover",
    "corridor_ever_active",
    "endpoint_teacher_ever_active",
    "final_forward_ft",
    "final_position_error_ft",
    "final_forward_speed_fps",
    "stop_hold_s",
    "final_cross_ft",
    "final_lateral_speed_fps",
    "lateral_hold_s",
    "max_cross_ft",
    "presentation_cross_pass",
    "final_alt_ft",
    "final_vs_fps",
    "min_alt_ft",
    "max_alt_ft",
    "final_heading_error_deg",
    "full_hover_hold_s",
]:
    print(f"{key:31s}: {result[key]}")

print()

if result["endpoint_hover"] and result["presentation_cross_pass"]:
    print("ENDPOINT HOVER: TRUE")
    print("PRESENTATION GEOMETRY: TRUE")
    print("V3 passed endpoint hover and the full <=5 ft lateral corridor target.")
elif result["endpoint_hover"]:
    print("ENDPOINT HOVER: TRUE")
    print("PRESENTATION GEOMETRY: FALSE")
    print("Functional Stage-3 pass, but do NOT lock it: max cross-track still exceeds 5 ft.")
elif result["safe"]:
    print("ENDPOINT HOVER: FALSE")
    print("PRESENTATION GEOMETRY: FALSE")
    print("The run stayed safe but Stage-3 did not satisfy the full hover criteria.")
else:
    print("ENDPOINT HOVER: FALSE")
    print("PRESENTATION GEOMETRY: FALSE")
    print("Safety was violated. Inspect the trace before changing any longitudinal parameter.")

print()
print("IMPORTANT: Lock Stage 3 only if BOTH endpoint hover and full-geometry checks pass.")
print("Saved:", SUMMARY_CSV)
print("Saved:", TRACE_CSV)
