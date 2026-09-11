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

OUT_DIR = Path("results_stage3_braking_identification_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "duration_strength_sweep.csv"
BEST_TRACE_CSV = OUT_DIR / "best_duration_strength_trace.csv"


# ============================================================
# CONSTANTS
# ============================================================

TARGET_DISTANCE = 300.0
STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
HANDOFF_STABLE_TIME = 5.0

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040
CONTROL_DT = 0.075

# We already identified the braking sign:
# negative elevator residual = nose-up tendency = braking.
ELEVATOR_DELTAS = [-0.50, -0.75, -1.00]
PULSE_DURATIONS = [2.0, 4.0, 6.0]
RECOVERY_SECONDS = 2.0

# Safety envelope for identification only.
ALT_MIN = 290.0
ALT_MAX = 310.0
MAX_ABS_PITCH_DEG = 10.0
MAX_ABS_ROLL_DEG = 10.0
MIN_FORWARD_SPEED_FPS = -2.0


# ============================================================
# HELPERS
# ============================================================

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


def rule(text):
    print()
    print("=" * 142)
    print(text)
    print("=" * 142)


def state_snapshot(fdm):
    return {
        "altitude_ft": fdm_float(fdm, "position/h-agl-ft"),
        "forward_velocity_fps": fdm_float(fdm, "velocities/u-aero-fps"),
        "lateral_velocity_fps": fdm_float(fdm, "velocities/v-aero-fps"),
        "vertical_speed_fps": fdm_float(fdm, "velocities/h-dot-fps"),
        "pitch_deg": math.degrees(
            fdm_float(fdm, "attitude/pitch-rad", 0.0)
        ),
        "roll_deg": math.degrees(
            fdm_float(fdm, "attitude/roll-rad", 0.0)
        ),
        "pitch_rate_rad_s": fdm_float(
            fdm, "velocities/q-rad_sec", 0.0
        ),
        "physical_collective": fdm_float(
            fdm, "fcs/collective-cmd-norm"
        ),
        "physical_elevator": fdm_float(
            fdm, "fcs/elevator-cmd-norm"
        ),
        "physical_aileron": fdm_float(
            fdm, "fcs/aileron-cmd-norm"
        ),
        "physical_rudder": fdm_float(
            fdm, "fcs/rudder-cmd-norm"
        ),
    }


def safety_ok(s):
    return bool(
        ALT_MIN <= s["altitude_ft"] <= ALT_MAX
        and abs(s["pitch_deg"]) <= MAX_ABS_PITCH_DEG
        and abs(s["roll_deg"]) <= MAX_ABS_ROLL_DEG
        and s["forward_velocity_fps"] >= MIN_FORWARD_SPEED_FPS
    )


def run_raw_control_cycle(env2, action):
    """
    Apply one repaired Stage-2 action and advance 10 JSBSim physics
    steps. This deliberately bypasses Stage-2 endpoint termination
    logic after 300 ft.
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    a = np.clip(a, -1.0, +1.0)

    env2._apply_action(a)

    for _ in range(10):
        if not env2.fdm.run():
            raise RuntimeError(
                "JSBSim stopped during Stage-3 identification."
            )


# ============================================================
# LOAD LOCKED POLICIES
# ============================================================

stage1_model = PPO.load(STAGE1_MODEL_PATH)
stage2_model = PPO.load(STAGE2_MODEL_PATH)


# ============================================================
# REPRODUCE EXACT STAGE-2 ENDPOINT
# ============================================================

def build_endpoint():
    env1 = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs1, info1 = env1.reset()
    fdm = get_fdm(env1)
    active_fdm_id = id(fdm)

    mission_heading = fdm_float(
        fdm, "attitude/heading-true-rad"
    )

    dt1 = float(getattr(env1, "dt", CONTROL_DT) or CONTROL_DT)
    if not np.isfinite(dt1) or dt1 <= 0:
        dt1 = CONTROL_DT

    stable_time = 0.0

    for _step in range(int(STAGE1_MAX_TIME / dt1)):
        a1, _ = stage1_model.predict(
            obs1,
            deterministic=True,
        )

        obs1, _, terminated, truncated, info1 = env1.step(a1)

        alt = info_float(info1, "altitude")
        vs = info_float(info1, "vertical_speed")
        vn = info_float(info1, "vn", 0.0)
        ve = info_float(info1, "ve", 0.0)
        hs = float(np.hypot(vn, ve))
        drift = info_float(info1, "drift", 999.0)

        stable = bool(
            295.0 <= alt <= 305.0
            and abs(vs) <= 0.50
            and hs <= 1.0
            and drift <= 3.0
        )

        stable_time = stable_time + dt1 if stable else 0.0

        if stable_time >= HANDOFF_STABLE_TIME:
            break

        if terminated and not bool(info1.get("success", False)):
            raise RuntimeError("Stage 1 failed before handoff.")

        if truncated:
            raise RuntimeError("Stage 1 truncated before handoff.")

    if stable_time < HANDOFF_STABLE_TIME:
        raise RuntimeError("Stable Stage-1 handoff not reached.")

    env2 = HelicopterEnvStage2RefineMapped(
        aileron_scale=AILERON_SCALE,
        rudder_scale=RUDDER_SCALE,
    )

    env2.reset()
    env2.fdm = fdm

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = 0.0

    if hasattr(env2, "target_heading"):
        env2.target_heading = float(mission_heading)

    for attr in [
        "steps",
        "target_hold_steps",
        "hold_steps",
        "success_hold_steps",
    ]:
        if hasattr(env2, attr):
            setattr(env2, attr, 0)

    if id(get_fdm(env2)) != active_fdm_id:
        raise RuntimeError("Stage1->Stage2 FDM continuity failed.")

    obs2 = np.asarray(env2._get_obs(), dtype=np.float32)

    dt2 = float(getattr(env2, "dt", CONTROL_DT) or CONTROL_DT)
    if not np.isfinite(dt2) or dt2 <= 0:
        dt2 = CONTROL_DT

    endpoint_action = None

    for _step in range(int(STAGE2_MAX_TIME / dt2)):
        a2, _ = stage2_model.predict(
            obs2,
            deterministic=True,
        )

        a2 = np.asarray(a2, dtype=np.float32).reshape(-1)

        obs2, _, terminated, truncated, info2 = env2.step(a2)
        obs2 = np.asarray(obs2, dtype=np.float32)

        endpoint_action = a2.copy()

        distance = info_float(
            info2,
            "forward_distance",
            getattr(env2, "forward_distance", 0.0),
        )

        if distance >= TARGET_DISTANCE:
            break

        if terminated and not bool(info2.get("success", False)):
            raise RuntimeError("Stage 2 failed before endpoint.")

        if truncated:
            raise RuntimeError("Stage 2 truncated before endpoint.")

    if endpoint_action is None:
        raise RuntimeError("No endpoint action.")

    return (
        env1,
        env2,
        fdm,
        endpoint_action,
        state_snapshot(fdm),
    )


# ============================================================
# ONE DURATION × STRENGTH CASE
# ============================================================

def run_case(delta, pulse_seconds, detailed=False):
    env1, env2, fdm, baseline_action, start = build_endpoint()

    pulse_action = baseline_action.copy()
    pulse_action[1] = float(
        np.clip(
            baseline_action[1] + delta,
            -1.0,
            +1.0,
        )
    )

    pulse_steps = int(round(pulse_seconds / CONTROL_DT))
    recovery_steps = int(round(RECOVERY_SECONDS / CONTROL_DT))

    trace = []
    safe = True

    min_alt = start["altitude_ft"]
    max_alt = start["altitude_ft"]
    min_fwd = start["forward_velocity_fps"]
    max_pitch = abs(start["pitch_deg"])

    # Pulse.
    for step in range(pulse_steps):
        run_raw_control_cycle(env2, pulse_action)
        s = state_snapshot(fdm)
        s["time_s"] = (step + 1) * CONTROL_DT
        s["phase"] = "pulse"
        trace.append(s.copy())

        min_alt = min(min_alt, s["altitude_ft"])
        max_alt = max(max_alt, s["altitude_ft"])
        min_fwd = min(min_fwd, s["forward_velocity_fps"])
        max_pitch = max(max_pitch, abs(s["pitch_deg"]))

        if not safety_ok(s):
            safe = False
            break

    pulse_end = state_snapshot(fdm)

    # Recovery at the exact Stage-2 endpoint baseline action.
    if safe:
        for step in range(recovery_steps):
            run_raw_control_cycle(env2, baseline_action)
            s = state_snapshot(fdm)
            s["time_s"] = pulse_seconds + (step + 1) * CONTROL_DT
            s["phase"] = "recovery"
            trace.append(s.copy())

            min_alt = min(min_alt, s["altitude_ft"])
            max_alt = max(max_alt, s["altitude_ft"])
            min_fwd = min(min_fwd, s["forward_velocity_fps"])
            max_pitch = max(max_pitch, abs(s["pitch_deg"]))

            if not safety_ok(s):
                safe = False
                break

    final = state_snapshot(fdm)

    pulse_brake = (
        start["forward_velocity_fps"]
        - pulse_end["forward_velocity_fps"]
    )

    total_brake = (
        start["forward_velocity_fps"]
        - final["forward_velocity_fps"]
    )

    avg_pulse_decel = (
        pulse_brake / pulse_seconds
        if pulse_seconds > 0
        else float("nan")
    )

    result = {
        "delta": float(delta),
        "pulse_seconds": float(pulse_seconds),
        "baseline_action1": float(baseline_action[1]),
        "used_action1": float(pulse_action[1]),
        "safe": bool(safe),
        "start_forward_fps": float(start["forward_velocity_fps"]),
        "pulse_end_forward_fps": float(
            pulse_end["forward_velocity_fps"]
        ),
        "final_forward_fps": float(final["forward_velocity_fps"]),
        "pulse_brake_fps": float(pulse_brake),
        "total_brake_fps": float(total_brake),
        "avg_pulse_decel_fps2": float(avg_pulse_decel),
        "min_forward_fps": float(min_fwd),
        "start_altitude_ft": float(start["altitude_ft"]),
        "min_altitude_ft": float(min_alt),
        "max_altitude_ft": float(max_alt),
        "final_altitude_ft": float(final["altitude_ft"]),
        "start_pitch_deg": float(start["pitch_deg"]),
        "pulse_end_pitch_deg": float(pulse_end["pitch_deg"]),
        "final_pitch_deg": float(final["pitch_deg"]),
        "max_abs_pitch_deg": float(max_pitch),
        "pulse_end_vs_fps": float(
            pulse_end["vertical_speed_fps"]
        ),
        "final_vs_fps": float(final["vertical_speed_fps"]),
        "physical_elevator_pulse_end": float(
            pulse_end["physical_elevator"]
        ),
        "trace": trace,
    }

    if detailed:
        print()
        print(
            f"Detailed best case: dA1={delta:+.2f}, "
            f"pulse={pulse_seconds:.1f}s"
        )

        for row in trace:
            print(
                f"t={row['time_s']:5.2f}s | "
                f"{row['phase']:8s} | "
                f"FWD={row['forward_velocity_fps']:+7.3f} | "
                f"ALT={row['altitude_ft']:7.3f} | "
                f"VS={row['vertical_speed_fps']:+6.3f} | "
                f"PITCH={row['pitch_deg']:+7.3f}deg | "
                f"ELE={row['physical_elevator']:+.6f}"
            )

    env2.fdm = None
    env1.close()

    return result


# ============================================================
# SWEEP
# ============================================================

rule("STAGE 3 — ELEVATOR BRAKING IDENTIFICATION V2")

print("NO TRAINING.")
print("Braking sign already identified: NEGATIVE elevator residual.")
print("Now measuring useful authority vs pulse duration.")
print(
    "Every row starts from a fresh locked "
    "Stage1 -> Stage2 endpoint."
)

results = []
case_no = 0

for delta in ELEVATOR_DELTAS:
    for duration in PULSE_DURATIONS:
        case_no += 1

        r = run_case(
            delta,
            duration,
            detailed=False,
        )

        results.append(r)

        print(
            f"B{case_no:02d} | "
            f"dA1={delta:+.2f} | "
            f"T={duration:3.1f}s | "
            f"A1={r['used_action1']:+.3f} | "
            f"FWD {r['start_forward_fps']:6.2f}"
            f" -> {r['pulse_end_forward_fps']:6.2f}"
            f" -> {r['final_forward_fps']:6.2f} | "
            f"BRAKE={r['pulse_brake_fps']:+6.2f} | "
            f"DECEL={r['avg_pulse_decel_fps2']:+6.3f} | "
            f"PITCH_END={r['pulse_end_pitch_deg']:+6.2f}deg | "
            f"ALT=[{r['min_altitude_ft']:.2f},"
            f"{r['max_altitude_ft']:.2f}] | "
            f"SAFE={r['safe']}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

rows = []
for r in results:
    rows.append(
        {
            k: v
            for k, v in r.items()
            if k != "trace"
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
# RANK
# ============================================================

safe = [r for r in results if r["safe"]]

if not safe:
    raise RuntimeError(
        "No safe duration/strength case found."
    )


def rank_key(r):
    # We want useful braking, but not by destroying the trajectory.
    altitude_excursion = max(
        abs(r["min_altitude_ft"] - r["start_altitude_ft"]),
        abs(r["max_altitude_ft"] - r["start_altitude_ft"]),
    )

    return (
        -r["pulse_brake_fps"],
        abs(r["pulse_end_forward_fps"]),
        r["max_abs_pitch_deg"],
        altitude_excursion,
    )


safe.sort(key=rank_key)

rule("TOP SAFE V2 BRAKING CASES")

for i, r in enumerate(safe[:6], start=1):
    print(
        f"{i}. "
        f"dA1={r['delta']:+.2f} | "
        f"T={r['pulse_seconds']:.1f}s | "
        f"A1={r['used_action1']:+.3f} | "
        f"FWD_END={r['pulse_end_forward_fps']:+.3f} | "
        f"BRAKE={r['pulse_brake_fps']:+.3f} | "
        f"DECEL={r['avg_pulse_decel_fps2']:+.3f} | "
        f"PITCH_END={r['pulse_end_pitch_deg']:+.3f}deg | "
        f"ALT_MIN={r['min_altitude_ft']:.3f}"
    )

best = safe[0]

rule("BEST V2 CASE — DETAILED REPEAT")

best_detailed = run_case(
    best["delta"],
    best["pulse_seconds"],
    detailed=True,
)

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
# FINAL DIAGNOSIS
# ============================================================

rule("STAGE 3 BRAKING AUTHORITY V2 RESULT")

print(
    "Endpoint forward speed     : "
    f"{best_detailed['start_forward_fps']:.3f} ft/s"
)

print(
    "Best elevator residual     : "
    f"{best_detailed['delta']:+.3f}"
)

print(
    "Best pulse duration        : "
    f"{best_detailed['pulse_seconds']:.3f} s"
)

print(
    "Used elevator action       : "
    f"{best_detailed['used_action1']:+.3f}"
)

print(
    "Physical elevator          : "
    f"{best_detailed['physical_elevator_pulse_end']:+.6f}"
)

print(
    "Forward speed after pulse  : "
    f"{best_detailed['pulse_end_forward_fps']:.3f} ft/s"
)

print(
    "Forward speed after recovery: "
    f"{best_detailed['final_forward_fps']:.3f} ft/s"
)

print(
    "Pulse speed reduction      : "
    f"{best_detailed['pulse_brake_fps']:.3f} ft/s"
)

print(
    "Average pulse deceleration : "
    f"{best_detailed['avg_pulse_decel_fps2']:.3f} ft/s^2"
)

print(
    "Pitch at pulse end         : "
    f"{best_detailed['pulse_end_pitch_deg']:.3f} deg"
)

print(
    "Altitude range             : "
    f"{best_detailed['min_altitude_ft']:.3f}"
    f" .. "
    f"{best_detailed['max_altitude_ft']:.3f} ft"
)

print(
    "Safe                       : "
    f"{best_detailed['safe']}"
)

print()

if (
    best_detailed["safe"]
    and best_detailed["pulse_brake_fps"] >= 1.0
):
    print("USEFUL BRAKING AUTHORITY: CONFIRMED.")
    print(
        "Next step: design the brake-to-hover teacher "
        "from forward-speed error + pitch-rate damping."
    )
else:
    print("USEFUL BRAKING AUTHORITY: STILL TOO WEAK.")
    print(
        "Next step: inspect/extend the Stage-2 elevator "
        "physical command mapping before building a teacher."
    )

print()
print("Saved:", SUMMARY_CSV)
print("Saved:", BEST_TRACE_CSV)
