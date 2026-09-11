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

OUT_DIR = Path("results_stage3_brake_teacher_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "linear_position_speed_sweep.csv"
BEST_TRACE_CSV = OUT_DIR / "best_linear_brake_trace.csv"


# ============================================================
# MISSION / LOCKED CONTROL SETTINGS
# ============================================================

TARGET_FORWARD_FT = 300.0
TARGET_ALT_FT = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
STAGE3_MAX_TIME = 45.0

HANDOFF_STABLE_TIME = 5.0
CONTROL_DT = 0.075

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040

# Braking must start before the endpoint. This does NOT change the
# locked Stage-2 PPO; it only defines where Stage-3 teacher takeover
# begins in the continuous flight.
TEACHER_ENABLE_FORWARD_FT = 80.0

LOCKED_AILERON_ACTION = -0.230
LOCKED_RUDDER_ACTION = 0.000
ELEVATOR_TRIM_ACTION = 0.013725

# Keep the same altitude teacher as V1 so this experiment changes
# ONLY the longitudinal stopping profile.
ALT_KP = 0.030
VS_KD = 0.120
MAX_ALT_CORR = 0.40

# Pitch-rate damping; negative elevator residual is the identified
# braking direction.
Q_DAMP = 2.0

# V2 change:
# Desired speed is now LINEAR in signed position error:
#
#     v_des = K_POS * (300 - forward_position)
#
# This is much more aggressive close to the endpoint than the old
# sqrt stopping-speed profile and naturally commands a gentle reverse
# speed after overshoot.
K_POS_VALUES = [0.045, 0.055, 0.065, 0.075]
KV_VALUES = [0.60, 0.80, 1.00]

V_FWD_MAX = 12.0
V_REV_MAX = 2.0
MAX_ELEVATOR_RESIDUAL = 1.0

# Longitudinal acceptance.
STOP_POS_TOL_FT = 5.0
STOP_SPEED_TOL_FPS = 1.0
STOP_HOLD_SECONDS = 3.0

# Safety / geometry monitoring.
ALT_SAFE_MIN = 290.0
ALT_SAFE_MAX = 310.0
MAX_ABS_PITCH_DEG = 8.0
MAX_ABS_ROLL_DEG = 10.0
MAX_CROSS_SAFE_FT = 12.0

# Full hover readiness is reported separately. Longitudinal stop can
# pass even if hover altitude/lateral quality still needs its own step.
HOVER_CROSS_TOL_FT = 5.0
HOVER_LAT_SPEED_TOL_FPS = 1.0
HOVER_VS_TOL_FPS = 0.75
HOVER_HEADING_TOL_DEG = 1.0

EARTH_RADIUS_FT = 20_902_231.0


# ============================================================
# HELPERS
# ============================================================

def rule(text):
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


def state_snapshot(fdm, lat0, lon0, mission_heading):
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
            fdm,
            "position/h-agl-ft",
        ),
        "forward_speed_fps": fdm_float(
            fdm,
            "velocities/u-aero-fps",
            0.0,
        ),
        "lateral_speed_fps": fdm_float(
            fdm,
            "velocities/v-aero-fps",
            0.0,
        ),
        "vertical_speed_fps": fdm_float(
            fdm,
            "velocities/h-dot-fps",
            0.0,
        ),
        "pitch_deg": math.degrees(
            fdm_float(
                fdm,
                "attitude/pitch-rad",
                0.0,
            )
        ),
        "roll_deg": math.degrees(
            fdm_float(
                fdm,
                "attitude/roll-rad",
                0.0,
            )
        ),
        "pitch_rate_rad_s": fdm_float(
            fdm,
            "velocities/q-rad_sec",
            0.0,
        ),
        "heading_error_deg": math.degrees(
            heading_error
        ),
        "physical_collective": fdm_float(
            fdm,
            "fcs/collective-cmd-norm",
        ),
        "physical_elevator": fdm_float(
            fdm,
            "fcs/elevator-cmd-norm",
        ),
        "physical_aileron": fdm_float(
            fdm,
            "fcs/aileron-cmd-norm",
        ),
        "physical_rudder": fdm_float(
            fdm,
            "fcs/rudder-cmd-norm",
        ),
    }


def safe_state(state):
    return bool(
        ALT_SAFE_MIN
        <= state["altitude_ft"]
        <= ALT_SAFE_MAX
        and abs(state["pitch_deg"])
        <= MAX_ABS_PITCH_DEG
        and abs(state["roll_deg"])
        <= MAX_ABS_ROLL_DEG
        and abs(state["cross_track_ft"])
        <= MAX_CROSS_SAFE_FT
    )


def raw_control_cycle(env2, action):
    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    action = np.clip(
        action,
        -1.0,
        +1.0,
    )

    env2._apply_action(action)

    for _ in range(10):
        if not env2.fdm.run():
            raise RuntimeError(
                "JSBSim stopped during Stage-3 V2 teacher."
            )


# ============================================================
# LOAD MODELS ONCE
# ============================================================

stage1_model = PPO.load(STAGE1_MODEL_PATH)
stage2_model = PPO.load(STAGE2_MODEL_PATH)


# ============================================================
# REPRODUCE LOCKED STAGE1 -> STAGE2 TO TEACHER TAKEOVER
# ============================================================

def build_teacher_start():
    env1 = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs1, info1 = env1.reset()
    fdm = get_fdm(env1)
    active_fdm_id = id(fdm)
    mission_heading = heading_rad(fdm)

    dt1 = float(
        getattr(env1, "dt", CONTROL_DT)
        or CONTROL_DT
    )

    if not np.isfinite(dt1) or dt1 <= 0:
        dt1 = CONTROL_DT

    stable_time = 0.0

    for _step in range(
        int(STAGE1_MAX_TIME / dt1)
    ):
        action1, _ = stage1_model.predict(
            obs1,
            deterministic=True,
        )

        (
            obs1,
            _,
            terminated,
            truncated,
            info1,
        ) = env1.step(action1)

        altitude = info_float(
            info1,
            "altitude",
        )
        vertical_speed = info_float(
            info1,
            "vertical_speed",
        )
        vn = info_float(info1, "vn", 0.0)
        ve = info_float(info1, "ve", 0.0)
        horizontal_speed = float(
            np.hypot(vn, ve)
        )
        drift = info_float(
            info1,
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

    # Stage-2 forward coordinate origin is the true handoff point.
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

    if id(get_fdm(env2)) != active_fdm_id:
        raise RuntimeError(
            "Stage1->Stage2 FDM continuity failed."
        )

    obs2 = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    dt2 = float(
        getattr(env2, "dt", CONTROL_DT)
        or CONTROL_DT
    )

    if not np.isfinite(dt2) or dt2 <= 0:
        dt2 = CONTROL_DT

    for _step in range(
        int(STAGE2_MAX_TIME / dt2)
    ):
        action2, _ = stage2_model.predict(
            obs2,
            deterministic=True,
        )

        action2 = np.asarray(
            action2,
            dtype=np.float32,
        ).reshape(-1)

        (
            obs2,
            _,
            terminated,
            truncated,
            info2,
        ) = env2.step(action2)

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
# V2 TEACHER
# ============================================================

def desired_forward_speed(
    position_error_ft,
    k_pos,
):
    desired = (
        k_pos
        * position_error_ft
    )

    return float(
        np.clip(
            desired,
            -V_REV_MAX,
            +V_FWD_MAX,
        )
    )


def teacher_action(
    obs,
    state,
    k_pos,
    kv,
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

    # Same altitude support as V1.
    altitude_error = (
        TARGET_ALT_FT
        - state["altitude_ft"]
    )

    collective_correction = (
        ALT_KP
        * altitude_error
        - VS_KD
        * state["vertical_speed_fps"]
    )

    collective_correction = float(
        np.clip(
            collective_correction,
            -MAX_ALT_CORR,
            +MAX_ALT_CORR,
        )
    )

    action[0] = float(
        np.clip(
            base_action[0]
            + collective_correction,
            -1.0,
            +1.0,
        )
    )

    # New position -> desired-speed profile.
    v_des = desired_forward_speed(
        state["position_error_ft"],
        k_pos,
    )

    speed_error = (
        state["forward_speed_fps"]
        - v_des
    )

    elevator_residual = (
        -kv
        * speed_error
        + Q_DAMP
        * state["pitch_rate_rad_s"]
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

    action[2] = LOCKED_AILERON_ACTION
    action[3] = LOCKED_RUDDER_ACTION

    return (
        action.astype(np.float32),
        float(v_des),
        float(speed_error),
        float(collective_correction),
        float(elevator_residual),
    )


# ============================================================
# ONE CASE
# ============================================================

def run_case(k_pos, kv, detailed=False):
    (
        env1,
        env2,
        fdm,
        obs2,
        lat0,
        lon0,
        mission_heading,
    ) = build_teacher_start()

    start_state = state_snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    trace = []
    safe = True
    stop_hold_time = 0.0
    stop_achieved = False
    hover_ready = False

    max_cross = abs(
        start_state["cross_track_ft"]
    )
    min_alt = start_state["altitude_ft"]
    max_alt = start_state["altitude_ft"]
    max_abs_pitch = abs(
        start_state["pitch_deg"]
    )
    max_abs_roll = abs(
        start_state["roll_deg"]
    )
    max_overshoot = max(
        0.0,
        start_state["forward_ft"]
        - TARGET_FORWARD_FT,
    )

    final_state = start_state.copy()
    next_print = 0.0

    for step in range(
        int(STAGE3_MAX_TIME / CONTROL_DT)
    ):
        state = state_snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        (
            action,
            v_des,
            speed_error,
            collective_correction,
            elevator_residual,
        ) = teacher_action(
            obs2,
            state,
            k_pos,
            kv,
        )

        raw_control_cycle(
            env2,
            action,
        )

        state_after = state_snapshot(
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

        t = (
            step + 1
        ) * CONTROL_DT

        row = dict(state_after)
        row.update(
            {
                "time_s": float(t),
                "v_des_fps": float(v_des),
                "speed_error_fps": float(speed_error),
                "collective_correction": float(
                    collective_correction
                ),
                "elevator_residual": float(
                    elevator_residual
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
        max_abs_pitch = max(
            max_abs_pitch,
            abs(state_after["pitch_deg"]),
        )
        max_abs_roll = max(
            max_abs_roll,
            abs(state_after["roll_deg"]),
        )
        max_overshoot = max(
            max_overshoot,
            max(
                0.0,
                state_after["forward_ft"]
                - TARGET_FORWARD_FT,
            ),
        )

        if not safe_state(state_after):
            safe = False
            break

        longitudinal_stop_now = bool(
            abs(
                state_after["position_error_ft"]
            )
            <= STOP_POS_TOL_FT
            and abs(
                state_after["forward_speed_fps"]
            )
            <= STOP_SPEED_TOL_FPS
        )

        if longitudinal_stop_now:
            stop_hold_time += CONTROL_DT
        else:
            stop_hold_time = 0.0

        if stop_hold_time >= STOP_HOLD_SECONDS:
            stop_achieved = True

            hover_ready = bool(
                abs(
                    state_after["cross_track_ft"]
                )
                <= HOVER_CROSS_TOL_FT
                and abs(
                    state_after["lateral_speed_fps"]
                )
                <= HOVER_LAT_SPEED_TOL_FPS
                and abs(
                    state_after["vertical_speed_fps"]
                )
                <= HOVER_VS_TOL_FPS
                and abs(
                    state_after["heading_error_deg"]
                )
                <= HOVER_HEADING_TOL_DEG
                and 295.0
                <= state_after["altitude_ft"]
                <= 305.0
            )

            break

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"FWD={state_after['forward_ft']:7.2f} | "
                f"ERR={state_after['position_error_ft']:+7.2f} | "
                f"V={state_after['forward_speed_fps']:+6.2f} | "
                f"VDES={v_des:+6.2f} | "
                f"ALT={state_after['altitude_ft']:7.2f} | "
                f"VS={state_after['vertical_speed_fps']:+6.2f} | "
                f"X={state_after['cross_track_ft']:+6.2f} | "
                f"PITCH={state_after['pitch_deg']:+5.2f} | "
                f"A1={action[1]:+6.3f}"
            )
            next_print += 1.5

    result = {
        "k_pos": float(k_pos),
        "kv": float(kv),
        "safe": bool(safe),
        "stop_achieved": bool(stop_achieved),
        "full_hover_ready": bool(hover_ready),
        "start_forward_ft": float(
            start_state["forward_ft"]
        ),
        "start_speed_fps": float(
            start_state["forward_speed_fps"]
        ),
        "final_forward_ft": float(
            final_state["forward_ft"]
        ),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "stop_hold_time_s": float(
            stop_hold_time
        ),
        "max_overshoot_ft": float(
            max_overshoot
        ),
        "max_cross_ft": float(
            max_cross
        ),
        "final_cross_ft": float(
            final_state["cross_track_ft"]
        ),
        "final_lateral_speed_fps": float(
            final_state["lateral_speed_fps"]
        ),
        "min_alt_ft": float(min_alt),
        "max_alt_ft": float(max_alt),
        "final_alt_ft": float(
            final_state["altitude_ft"]
        ),
        "final_vs_fps": float(
            final_state["vertical_speed_fps"]
        ),
        "max_abs_pitch_deg": float(
            max_abs_pitch
        ),
        "max_abs_roll_deg": float(
            max_abs_roll
        ),
        "final_heading_error_deg": float(
            final_state["heading_error_deg"]
        ),
        "trace": trace,
    }

    env2.fdm = None
    env1.close()

    return result


# ============================================================
# SWEEP
# ============================================================

rule("STAGE 3 — BRAKE-TO-ENDPOINT TEACHER V2")

print("Stage 1: LOCKED")
print("Stage 2 cruise: LOCKED")
print("No PPO training.")
print("V2 changes ONLY the longitudinal stopping profile.")
print(
    "Old sqrt profile overshot by 20.22 ft; "
    "new profile uses v_des = K_POS * position_error."
)
print()

results = []
case_no = 0

for k_pos in K_POS_VALUES:
    for kv in KV_VALUES:
        case_no += 1

        result = run_case(
            k_pos,
            kv,
            detailed=False,
        )

        results.append(result)

        print(
            f"L{case_no:02d} | "
            f"KPOS={k_pos:.3f} | "
            f"KV={kv:.2f} | "
            f"STOP={result['stop_achieved']} | "
            f"HOVER={result['full_hover_ready']} | "
            f"SAFE={result['safe']} | "
            f"FWD={result['final_forward_ft']:7.2f} | "
            f"ERR={result['final_position_error_ft']:+6.2f} | "
            f"V={result['final_speed_fps']:+6.2f} | "
            f"OVR={result['max_overshoot_ft']:5.2f} | "
            f"XMAX={result['max_cross_ft']:5.2f} | "
            f"ALT=[{result['min_alt_ft']:.2f},"
            f"{result['max_alt_ft']:.2f}] | "
            f"VS={result['final_vs_fps']:+5.2f}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_rows = []

for result in results:
    summary_rows.append(
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
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# SELECT
# ============================================================

def selection_key(result):
    return (
        0 if result["stop_achieved"] else 1,
        0 if result["full_hover_ready"] else 1,
        0 if result["safe"] else 1,
        abs(result["final_position_error_ft"]),
        abs(result["final_speed_fps"]),
        result["max_overshoot_ft"],
        result["max_cross_ft"],
        abs(result["final_alt_ft"] - TARGET_ALT_FT),
        abs(result["final_vs_fps"]),
    )


results.sort(key=selection_key)

rule("TOP STAGE-3 V2 TEACHER CANDIDATES")

for rank, result in enumerate(
    results[:8],
    start=1,
):
    print(
        f"{rank}. "
        f"KPOS={result['k_pos']:.3f} | "
        f"KV={result['kv']:.2f} | "
        f"STOP={result['stop_achieved']} | "
        f"HOVER={result['full_hover_ready']} | "
        f"SAFE={result['safe']} | "
        f"FWD={result['final_forward_ft']:.2f} | "
        f"ERR={result['final_position_error_ft']:+.2f} | "
        f"V={result['final_speed_fps']:+.2f} | "
        f"OVR={result['max_overshoot_ft']:.2f} | "
        f"XMAX={result['max_cross_ft']:.2f} | "
        f"ALT={result['final_alt_ft']:.2f} | "
        f"VS={result['final_vs_fps']:+.2f}"
    )


best = results[0]


# ============================================================
# DETAILED REPEAT
# ============================================================

rule("BEST STAGE-3 V2 TEACHER — FULL DETAILED FLIGHT")

best_detailed = run_case(
    best["k_pos"],
    best["kv"],
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
# FINAL RESULT
# ============================================================

rule("STAGE 3 BRAKE-TO-ENDPOINT V2 FINAL RESULT")

for key in [
    "k_pos",
    "kv",
    "safe",
    "stop_achieved",
    "full_hover_ready",
    "start_forward_ft",
    "start_speed_fps",
    "final_forward_ft",
    "final_position_error_ft",
    "final_speed_fps",
    "stop_hold_time_s",
    "max_overshoot_ft",
    "max_cross_ft",
    "final_cross_ft",
    "final_lateral_speed_fps",
    "min_alt_ft",
    "max_alt_ft",
    "final_alt_ft",
    "final_vs_fps",
    "max_abs_pitch_deg",
    "max_abs_roll_deg",
    "final_heading_error_deg",
]:
    print(
        f"{key:30s}: "
        f"{best_detailed[key]}"
    )

print()

if best_detailed["stop_achieved"]:
    print("LONGITUDINAL ENDPOINT STOP: TRUE")

    if best_detailed["full_hover_ready"]:
        print("ENDPOINT HOVER QUALITY: TRUE")
        print(
            "Next: lock this teacher and distill Stage 3."
        )
    else:
        print("ENDPOINT HOVER QUALITY: NOT YET")
        print(
            "Longitudinal braking is solved. "
            "Next solve ONLY the remaining endpoint hover "
            "altitude/lateral hold before distillation."
        )
else:
    print("LONGITUDINAL ENDPOINT STOP: FALSE")
    print(
        "Do not distill. Use the V2 trace to make one narrow "
        "stopping-profile refinement only."
    )

print()
print("Saved:", SUMMARY_CSV)
print("Saved:", BEST_TRACE_CSV)
