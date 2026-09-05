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

OUT_DIR = Path("results_stage3_persistent_stop_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "lead_compensation_sweep.csv"
BEST_TRACE_CSV = OUT_DIR / "best_lead_compensation_trace.csv"


# ============================================================
# MISSION
# ============================================================

TARGET_FORWARD_FT = 300.0
TARGET_ALT_FT = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
STAGE3_MAX_TIME = 60.0

HANDOFF_STABLE_TIME = 5.0
CONTROL_DT = 0.075

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040

TEACHER_ENABLE_FORWARD_FT = 80.0

# Keep the already successful V2 longitudinal structure.
KV = 1.00
K_POS = 0.055
V_FWD_MAX = 9.0

# V3 finished about 8 ft late while already below the speed limit.
# V4 therefore changes only a small virtual braking lead.
# Mission acceptance remains centered at the REAL 300-ft endpoint.
BRAKE_LEAD_VALUES = [4.0, 6.0, 8.0, 10.0, 12.0]

V_REV_MAX = 2.0

ELEVATOR_TRIM_ACTION = 0.013725
Q_DAMP = 2.0
MAX_ELEVATOR_RESIDUAL = 1.0

# Keep altitude near the best hover-bias band found in the previous
# experiment. We do NOT search altitude here.
COLLECTIVE_BIAS = 0.22
ALT_KP = 0.030
VS_KD = 0.120
MAX_ALT_CORR = 0.45

# Keep one moderate lateral stabilizer only to prevent the long
# low-speed run from tripping cross-track safety. We do NOT optimize it.
AILERON_TRIM_ACTION = -0.230
LATERAL_KP = 0.035
LATERAL_KD = 0.16
MAX_AILERON_RESIDUAL = 0.65
RUDDER_ACTION = 0.0

# Persistent longitudinal stop: stricter than previous test.
STOP_POS_TOL_FT = 5.0
STOP_SPEED_TOL_FPS = 0.60
STOP_HOLD_SECONDS = 5.0

# Safety only.
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

    return mission_axes(north, east, mission_heading)


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
    }


def safe_state(state):
    return bool(
        ALT_SAFE_MIN
        <= state["altitude_ft"]
        <= ALT_SAFE_MAX
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
                "JSBSim stopped during Stage-3 persistent-stop test."
            )


# ============================================================
# MODELS
# ============================================================

stage1_model = PPO.load(STAGE1_MODEL_PATH)
stage2_model = PPO.load(STAGE2_MODEL_PATH)


# ============================================================
# LOCKED STAGE1 -> STAGE2 TO STAGE3 TAKEOVER
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
        getattr(env1, "dt", CONTROL_DT)
        or CONTROL_DT
    )

    if not np.isfinite(dt1) or dt1 <= 0:
        dt1 = CONTROL_DT

    stable_time = 0.0

    for _ in range(int(STAGE1_MAX_TIME / dt1)):
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

        altitude = info_float(info1, "altitude")
        vertical_speed = info_float(
            info1,
            "vertical_speed",
        )
        vn = info_float(info1, "vn", 0.0)
        ve = info_float(info1, "ve", 0.0)
        hs = float(np.hypot(vn, ve))
        drift = info_float(
            info1,
            "drift",
            999.0,
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

    for _ in range(int(STAGE2_MAX_TIME / dt2)):
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
# V4 TEACHER
# ============================================================

def teacher_action(
    obs,
    state,
    brake_lead_ft,
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

    # Altitude support fixed from previous experiment.
    alt_error = (
        TARGET_ALT_FT
        - state["altitude_ft"]
    )

    collective_correction = (
        COLLECTIVE_BIAS
        + ALT_KP * alt_error
        - VS_KD * state["vertical_speed_fps"]
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

    # Persistent-stop refinement: same linear position profile,
    # but with a lower forward-speed cap.
    v_des = float(
        np.clip(
            K_POS
            * (state["position_error_ft"] - brake_lead_ft),
            -V_REV_MAX,
            +V_FWD_MAX,
        )
    )

    speed_error = (
        state["forward_speed_fps"]
        - v_des
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

    # Fixed lateral stabilizer; not part of this search.
    aileron_residual = (
        -LATERAL_KP * state["cross_track_ft"]
        -LATERAL_KD * state["lateral_speed_fps"]
    )

    aileron_residual = float(
        np.clip(
            aileron_residual,
            -MAX_AILERON_RESIDUAL,
            +MAX_AILERON_RESIDUAL,
        )
    )

    action[2] = float(
        np.clip(
            AILERON_TRIM_ACTION
            + aileron_residual,
            -1.0,
            +1.0,
        )
    )

    action[3] = RUDDER_ACTION

    return (
        action.astype(np.float32),
        float(v_des),
        float(speed_error),
        float(elevator_residual),
    )


# ============================================================
# ONE CASE
# ============================================================

def run_case(brake_lead_ft, detailed=False):
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
    hold_time = 0.0
    persistent_stop = False
    ever_entered_stop = False

    min_alt = start["altitude_ft"]
    max_alt = start["altitude_ft"]
    max_cross = abs(start["cross_track_ft"])
    max_overshoot = max(
        0.0,
        start["forward_ft"] - TARGET_FORWARD_FT,
    )

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
            speed_error,
            elevator_residual,
        ) = teacher_action(
            obs2,
            state,
            brake_lead_ft,
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
                "speed_error_fps": float(speed_error),
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

        min_alt = min(
            min_alt,
            state_after["altitude_ft"],
        )
        max_alt = max(
            max_alt,
            state_after["altitude_ft"],
        )
        max_cross = max(
            max_cross,
            abs(state_after["cross_track_ft"]),
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

        if stop_now:
            ever_entered_stop = True
            hold_time += CONTROL_DT
        else:
            hold_time = 0.0

        if hold_time >= STOP_HOLD_SECONDS:
            persistent_stop = True
            break

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"FWD={state_after['forward_ft']:7.2f} | "
                f"ERR={state_after['position_error_ft']:+6.2f} | "
                f"V={state_after['forward_speed_fps']:+5.2f} | "
                f"VDES={v_des:+5.2f} | "
                f"ALT={state_after['altitude_ft']:7.2f} | "
                f"X={state_after['cross_track_ft']:+6.2f} | "
                f"A1={action[1]:+6.3f}"
            )
            next_print += 1.5

    result = {
        "brake_lead_ft": float(brake_lead_ft),
        "safe": bool(safe),
        "ever_entered_stop": bool(ever_entered_stop),
        "persistent_stop": bool(persistent_stop),
        "final_forward_ft": float(
            final_state["forward_ft"]
        ),
        "final_position_error_ft": float(
            final_state["position_error_ft"]
        ),
        "final_forward_speed_fps": float(
            final_state["forward_speed_fps"]
        ),
        "stop_hold_s": float(hold_time),
        "max_overshoot_ft": float(max_overshoot),
        "max_cross_ft": float(max_cross),
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

rule("STAGE 3 — PERSISTENT ENDPOINT STOP V4")

print("Stage 1: LOCKED")
print("Stage 2 cruise: LOCKED")
print("No PPO training.")
print("KPOS=0.055 and VMAX=9.0 are fixed from V3.")
print(
    "V3 ended about 8 ft late. V4 sweeps only a small braking lead "
    "while acceptance stays centered at the real 300-ft endpoint."
)
print(
    "Persistent-stop criterion remains strict: "
    "|error|<=5 ft and |speed|<=0.60 ft/s for 5 continuous seconds."
)
print()

results = []
case_no = 0

for brake_lead_ft in BRAKE_LEAD_VALUES:
    case_no += 1

    result = run_case(
        brake_lead_ft,
        detailed=False,
    )

    results.append(result)

    print(
        f"C{case_no:02d} | "
        f"LEAD={brake_lead_ft:4.1f}ft | "
        f"PERSIST={result['persistent_stop']} | "
        f"ENTERED={result['ever_entered_stop']} | "
        f"SAFE={result['safe']} | "
        f"FWD={result['final_forward_ft']:7.2f} | "
        f"ERR={result['final_position_error_ft']:+6.2f} | "
        f"V={result['final_forward_speed_fps']:+5.2f} | "
        f"HOLD={result['stop_hold_s']:4.1f}s | "
        f"OVR={result['max_overshoot_ft']:5.2f} | "
        f"XMAX={result['max_cross_ft']:5.2f} | "
        f"ALT={result['final_alt_ft']:7.2f}"
    )


# ============================================================
# SAVE SUMMARY
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
# SELECT BEST
# ============================================================

def selection_key(result):
    return (
        0 if result["persistent_stop"] else 1,
        0 if result["safe"] else 1,
        0 if result["ever_entered_stop"] else 1,
        abs(result["final_position_error_ft"]),
        abs(result["final_forward_speed_fps"]),
        result["max_overshoot_ft"],
        result["max_cross_ft"],
        abs(result["final_alt_ft"] - TARGET_ALT_FT),
    )


results.sort(key=selection_key)

rule("TOP PERSISTENT-STOP V4 CANDIDATES")

for rank, result in enumerate(
    results[:8],
    start=1,
):
    print(
        f"{rank}. "
        f"LEAD={result['brake_lead_ft']:.1f}ft | "
        f"PERSIST={result['persistent_stop']} | "
        f"ENTERED={result['ever_entered_stop']} | "
        f"FWD={result['final_forward_ft']:.2f} | "
        f"ERR={result['final_position_error_ft']:+.2f} | "
        f"V={result['final_forward_speed_fps']:+.2f} | "
        f"HOLD={result['stop_hold_s']:.2f}s | "
        f"OVR={result['max_overshoot_ft']:.2f} | "
        f"XMAX={result['max_cross_ft']:.2f} | "
        f"ALT={result['final_alt_ft']:.2f}"
    )


best = results[0]


# ============================================================
# DETAILED REPEAT
# ============================================================

rule("BEST PERSISTENT-STOP V4 — FULL DETAILED FLIGHT")

best_detailed = run_case(
    best["brake_lead_ft"],
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

rule("STAGE 3 PERSISTENT STOP V4 FINAL RESULT")

for key in [
    "brake_lead_ft",
    "safe",
    "ever_entered_stop",
    "persistent_stop",
    "final_forward_ft",
    "final_position_error_ft",
    "final_forward_speed_fps",
    "stop_hold_s",
    "max_overshoot_ft",
    "max_cross_ft",
    "final_cross_ft",
    "final_lateral_speed_fps",
    "min_alt_ft",
    "max_alt_ft",
    "final_alt_ft",
    "final_vs_fps",
    "final_heading_error_deg",
]:
    print(
        f"{key:30s}: "
        f"{best_detailed[key]}"
    )

print()

if best_detailed["persistent_stop"]:
    print("PERSISTENT LONGITUDINAL ENDPOINT STOP: TRUE")
    print(
        "Only now is longitudinal braking considered locked."
    )
    print(
        "Next: solve endpoint altitude/cross-track hover around this "
        "persistent stopping solution."
    )
else:
    print("PERSISTENT LONGITUDINAL ENDPOINT STOP: FALSE")
    print(
        "Do not call Stage 3 longitudinal control solved yet."
    )
    print(
        "Use this trace for one final narrow lead-band refinement only."
    )

print()
print("Saved:", SUMMARY_CSV)
print("Saved:", BEST_TRACE_CSV)
