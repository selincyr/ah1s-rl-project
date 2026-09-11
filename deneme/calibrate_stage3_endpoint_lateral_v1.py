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
    "models_stage2_distilled/"
    "AH1S_STAGE2_DISTILLED_SUCCESS.zip"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path("results_stage3_endpoint_lateral_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "endpoint_lateral_sweep.csv"
BEST_TRACE_CSV = OUT_DIR / "best_endpoint_lateral_trace.csv"


# ============================================================
# MISSION / LOCKED STAGE-3 LONGITUDINAL SOLUTION
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
# a lateral endpoint teacher. This avoids disturbing Stage-2 cruise and
# the now-locked longitudinal braking profile.
LATERAL_ENABLE_SPEED_FPS = 1.25
LATERAL_ENABLE_FORWARD_FT = 292.0

# Narrow lateral-only search.
HOVER_AILERON_TRIMS = [-0.05, 0.00, 0.05, 0.10]
LATERAL_KP_VALUES = [0.020, 0.035, 0.050]
LATERAL_KD = 0.18
MAX_LATERAL_CORR = 0.55

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

    # Locked V4 longitudinal controller.
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

    lateral_active = bool(
        state["forward_ft"]
        >= LATERAL_ENABLE_FORWARD_FT
        and abs(
            state["forward_speed_fps"]
        )
        <= LATERAL_ENABLE_SPEED_FPS
    )

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
    else:
        lateral_corr = 0.0
        action[2] = CRUISE_AILERON_ACTION

    action[3] = RUDDER_ACTION

    return (
        action.astype(np.float32),
        float(v_des),
        float(lateral_corr),
        bool(lateral_active),
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
            abs(
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
                f"LAT_ON={lateral_active}"
            )

            next_print += 1.5

    result = {
        "hover_trim": float(hover_trim),
        "lateral_kp": float(lateral_kp),
        "safe": bool(safe),
        "persistent_stop": bool(stop_ok),
        "persistent_lateral_hold": bool(lateral_ok),
        "endpoint_hover": bool(endpoint_hover_ok),
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
# SWEEP
# ============================================================

rule("STAGE 3 — ENDPOINT LATERAL HOLD V1")

print("Stage 1: LOCKED")
print("Stage 2 cruise: LOCKED")
print("Stage-3 persistent longitudinal stop: LOCKED")
print("Altitude behavior: LOCKED")
print("No PPO training.")
print("Only endpoint lateral hold is being calibrated.")
print(
    "Lateral teacher activates only near the endpoint at low forward speed."
)
print()

results = []
case_no = 0

for hover_trim in HOVER_AILERON_TRIMS:
    for lateral_kp in LATERAL_KP_VALUES:
        case_no += 1

        result = run_case(
            hover_trim,
            lateral_kp,
            detailed=False,
        )

        results.append(result)

        print(
            f"X{case_no:02d} | "
            f"TRIM={hover_trim:+.2f} | "
            f"KPX={lateral_kp:.3f} | "
            f"STOP={result['persistent_stop']} | "
            f"LAT={result['persistent_lateral_hold']} | "
            f"HOVER={result['endpoint_hover']} | "
            f"SAFE={result['safe']} | "
            f"FWD={result['final_forward_ft']:7.2f} | "
            f"V={result['final_forward_speed_fps']:+5.2f} | "
            f"X={result['final_cross_ft']:+6.2f} | "
            f"LATV={result['final_lateral_speed_fps']:+5.2f} | "
            f"ALT={result['final_alt_ft']:7.2f} | "
            f"HOLD={result['full_hover_hold_s']:4.1f}s"
        )


# ============================================================
# SAVE
# ============================================================

rows = []

for result in results:
    rows.append(
        {
            key: value
            for key, value in result.items()
            if key != "trace"
        }
    )

with SUMMARY_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SELECT
# ============================================================

def selection_key(result):
    return (
        0 if result["endpoint_hover"] else 1,
        0 if result["persistent_stop"] else 1,
        0 if result["persistent_lateral_hold"] else 1,
        0 if result["safe"] else 1,
        abs(result["final_position_error_ft"]),
        abs(result["final_forward_speed_fps"]),
        abs(result["final_cross_ft"]),
        abs(result["final_lateral_speed_fps"]),
        abs(result["final_alt_ft"] - TARGET_ALT_FT),
    )


results.sort(key=selection_key)

rule("TOP ENDPOINT LATERAL CANDIDATES")

for rank, result in enumerate(
    results[:8],
    start=1,
):
    print(
        f"{rank}. "
        f"TRIM={result['hover_trim']:+.2f} | "
        f"KPX={result['lateral_kp']:.3f} | "
        f"HOVER={result['endpoint_hover']} | "
        f"STOP={result['persistent_stop']} | "
        f"LAT={result['persistent_lateral_hold']} | "
        f"FWD={result['final_forward_ft']:.2f} | "
        f"V={result['final_forward_speed_fps']:+.2f} | "
        f"X={result['final_cross_ft']:+.2f} | "
        f"LATV={result['final_lateral_speed_fps']:+.2f} | "
        f"ALT={result['final_alt_ft']:.2f} | "
        f"HOLD={result['full_hover_hold_s']:.2f}s"
    )


best = results[0]


# ============================================================
# DETAILED REPEAT
# ============================================================

rule("BEST ENDPOINT LATERAL CASE — FULL DETAILED FLIGHT")

best_detailed = run_case(
    best["hover_trim"],
    best["lateral_kp"],
    detailed=True,
)

if best_detailed["trace"]:
    with BEST_TRACE_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                best_detailed["trace"][0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            best_detailed["trace"]
        )


# ============================================================
# FINAL
# ============================================================

rule("STAGE 3 ENDPOINT LATERAL HOLD V1 FINAL RESULT")

for key in [
    "hover_trim",
    "lateral_kp",
    "safe",
    "persistent_stop",
    "persistent_lateral_hold",
    "endpoint_hover",
    "final_forward_ft",
    "final_position_error_ft",
    "final_forward_speed_fps",
    "stop_hold_s",
    "final_cross_ft",
    "final_lateral_speed_fps",
    "lateral_hold_s",
    "max_cross_ft",
    "final_alt_ft",
    "final_vs_fps",
    "min_alt_ft",
    "max_alt_ft",
    "final_heading_error_deg",
    "full_hover_hold_s",
]:
    print(
        f"{key:31s}: "
        f"{best_detailed[key]}"
    )

print()

if best_detailed["endpoint_hover"]:
    print("ENDPOINT HOVER: TRUE")
    print(
        "Stage-3 teacher is complete: persistent stop + endpoint hover."
    )
    print(
        "NEXT: distill Stage-3 teacher into PPO and validate teacher OFF."
    )
else:
    print("ENDPOINT HOVER: FALSE")
    print(
        "Longitudinal stop remains locked. "
        "Do not modify Stage 1/2 or longitudinal braking."
    )

print()
print("Saved:", SUMMARY_CSV)
print("Saved:", BEST_TRACE_CSV)
