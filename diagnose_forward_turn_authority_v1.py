from pathlib import Path
import csv
import json
import math

import numpy as np

BASE_SOURCE = Path("diagnose_stage4_entry_margin_v3.py")
RESULT_DIR = Path("results_forward_turn_authority_v1")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TURN_ENTRY_FORWARD_FT = 240.0
MAX_STAGE2_TIME_S = 70.0
PULSE_SECONDS = 4.0
RECOVERY_SECONDS = 4.0

CASES = [
    {"name": "baseline",          "delta_a2":  0.00, "delta_a3":  0.00},
    {"name": "rudder_m030",       "delta_a2":  0.00, "delta_a3": -0.30},
    {"name": "rudder_m045",       "delta_a2":  0.00, "delta_a3": -0.45},
    {"name": "coord_ap15_rm045",  "delta_a2": +0.15, "delta_a3": -0.45},
    {"name": "coord_am15_rm045",  "delta_a2": -0.15, "delta_a3": -0.45},
    {"name": "coord_ap30_rm060",  "delta_a2": +0.30, "delta_a3": -0.60},
    {"name": "coord_am30_rm060",  "delta_a2": -0.30, "delta_a3": -0.60},
]

SAFE_ALT_MIN_FT = 290.0
SAFE_ALT_MAX_FT = 310.0
SAFE_MAX_ABS_ROLL_DEG = 12.0
SAFE_MAX_ABS_PITCH_DEG = 10.0
SAFE_MIN_FORWARD_SPEED_FPS = 2.0
SAFE_MAX_ABS_YAW_RATE_DEG_S = 20.0

if not BASE_SOURCE.exists():
    raise FileNotFoundError(f"Missing helper source: {BASE_SOURCE}")

text = BASE_SOURCE.read_text(encoding="utf-8")
MARKER = 'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'
if MARKER not in text:
    raise RuntimeError("Could not locate definition/experiment split marker.")

prefix = text.split(MARKER, 1)[0]
ns = {"__name__": "forward_turn_authority_base", "__file__": str(BASE_SOURCE)}
exec(compile(prefix, str(BASE_SOURCE), "exec"), ns)

HelicopterEnvStage1Distill = ns["HelicopterEnvStage1Distill"]
HelicopterEnvStage2RefineMapped = ns["HelicopterEnvStage2RefineMapped"]
stage1_model = ns["stage1_model"]
stage2_model = ns["stage2_model"]
get_fdm = ns["get_fdm"]
heading_rad = ns["heading_rad"]
latitude_deg = ns["latitude_deg"]
longitude_deg = ns["longitude_deg"]
snapshot = ns["snapshot"]
env_control_dt = ns["env_control_dt"]
first_finite = ns["first_finite"]
info_float = ns["info_float"]
raw_policy_cycle = ns["raw_policy_cycle"]
AILERON_SCALE = ns["AILERON_SCALE"]
RUDDER_SCALE = ns["RUDDER_SCALE"]
HANDOFF_STABLE_TIME = ns["HANDOFF_STABLE_TIME"]
STAGE1_MAX_TIME = ns["STAGE1_MAX_TIME"]

def wrap_deg(x):
    return float((float(x) + 180.0) % 360.0 - 180.0)

def heading_delta_deg(current, initial):
    return wrap_deg(float(current) - float(initial))

def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def safety_reason(state):
    roll_deg = math.degrees(state["roll_rad"])
    pitch_deg = math.degrees(state["pitch_rad"])
    yaw_rate_deg_s = math.degrees(state["yaw_rate_rad_s"])

    if state["altitude_ft"] < SAFE_ALT_MIN_FT:
        return "altitude_low"
    if state["altitude_ft"] > SAFE_ALT_MAX_FT:
        return "altitude_high"
    if abs(roll_deg) > SAFE_MAX_ABS_ROLL_DEG:
        return "roll_limit"
    if abs(pitch_deg) > SAFE_MAX_ABS_PITCH_DEG:
        return "pitch_limit"
    if state["forward_speed_fps"] < SAFE_MIN_FORWARD_SPEED_FPS:
        return "forward_speed_low"
    if abs(yaw_rate_deg_s) > SAFE_MAX_ABS_YAW_RATE_DEG_S:
        return "yaw_rate_limit"
    return ""

def build_forward_entry():
    env1 = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )
    obs1, info1 = env1.reset()

    fdm = get_fdm(env1)
    active_fdm_id = id(fdm)
    mission_heading = heading_rad(fdm)
    dt1 = env_control_dt(env1)

    stable_time = 0.0
    for _ in range(int(STAGE1_MAX_TIME / dt1)):
        action1, _ = stage1_model.predict(obs1, deterministic=True)
        obs1, _, terminated, truncated, info1 = env1.step(action1)

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
            raise RuntimeError("Stage 1 failed before handoff.")
        if truncated:
            env1.close()
            raise RuntimeError("Stage 1 truncated before handoff.")

    if stable_time < HANDOFF_STABLE_TIME:
        env1.close()
        raise RuntimeError("Stage 1 stable handoff not reached.")

    lat0 = latitude_deg(fdm)
    lon0 = longitude_deg(fdm)

    sim_before = first_finite(fdm, ["simulation/sim-time-sec"])

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

    for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
        if hasattr(env2, attr):
            setattr(env2, attr, 0)

    if id(get_fdm(env2)) != active_fdm_id:
        env2.fdm = None
        env1.close()
        raise RuntimeError("Same-FDM continuity failed.")

    sim_after = first_finite(fdm, ["simulation/sim-time-sec"])
    if (
        np.isfinite(sim_before)
        and np.isfinite(sim_after)
        and abs(sim_after - sim_before) > 1e-9
    ):
        env2.fdm = None
        env1.close()
        raise RuntimeError("Simulation clock changed during Stage 2 attach.")

    obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
    dt2 = env_control_dt(env2)
    reached = False

    for _ in range(int(MAX_STAGE2_TIME_S / dt2)):
        action2, _ = stage2_model.predict(obs2, deterministic=True)
        action2 = np.asarray(action2, dtype=np.float32).reshape(-1)

        obs2, _, terminated, truncated, info2 = env2.step(action2)
        obs2 = np.asarray(obs2, dtype=np.float32)

        state = snapshot(fdm, lat0, lon0, mission_heading)
        if state["forward_ft"] >= TURN_ENTRY_FORWARD_FT:
            reached = True
            break

        if terminated and not bool(info2.get("success", False)):
            env2.fdm = None
            env1.close()
            raise RuntimeError("Stage 2 failed before turn-entry point.")
        if truncated:
            env2.fdm = None
            env1.close()
            raise RuntimeError("Stage 2 truncated before turn-entry point.")

    if not reached:
        env2.fdm = None
        env1.close()
        raise RuntimeError("Turn-entry point was not reached.")

    state = snapshot(fdm, lat0, lon0, mission_heading)
    return {
        "env1": env1,
        "env2": env2,
        "fdm": fdm,
        "lat0": lat0,
        "lon0": lon0,
        "mission_heading": mission_heading,
        "dt": dt2,
        "state": state,
        "same_fdm": bool(id(fdm) == active_fdm_id),
    }

def close_case(start):
    try:
        start["env2"].fdm = None
    except Exception:
        pass
    try:
        start["env1"].close()
    except Exception:
        pass

def run_case(cfg):
    start = build_forward_entry()
    try:
        env2 = start["env2"]
        fdm = start["fdm"]
        lat0 = start["lat0"]
        lon0 = start["lon0"]
        mission_heading = start["mission_heading"]
        dt = start["dt"]

        initial = snapshot(fdm, lat0, lon0, mission_heading)
        state = initial.copy()

        pulse_steps = int(round(PULSE_SECONDS / dt))
        recovery_steps = int(round(RECOVERY_SECONDS / dt))

        rows = []
        termination = "completed"

        min_alt = initial["altitude_ft"]
        max_alt = initial["altitude_ft"]
        min_fwd_speed = initial["forward_speed_fps"]
        max_abs_roll = abs(math.degrees(initial["roll_rad"]))
        max_abs_pitch = abs(math.degrees(initial["pitch_rad"]))
        max_abs_yaw_rate = abs(math.degrees(initial["yaw_rate_rad_s"]))
        pulse_end = initial.copy()

        for step in range(pulse_steps):
            obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
            base, _ = stage2_model.predict(obs2, deterministic=True)
            base = np.asarray(base, dtype=np.float32).reshape(-1)
            action = base.copy()

            action[2] = float(np.clip(action[2] + cfg["delta_a2"], -1.0, +1.0))
            action[3] = float(np.clip(action[3] + cfg["delta_a3"], -1.0, +1.0))

            state, used = raw_policy_cycle(
                env2, fdm, action, lat0, lon0, mission_heading
            )

            t = (step + 1) * dt
            roll_deg = math.degrees(state["roll_rad"])
            pitch_deg = math.degrees(state["pitch_rad"])
            yaw_rate_deg_s = math.degrees(state["yaw_rate_rad_s"])

            min_alt = min(min_alt, state["altitude_ft"])
            max_alt = max(max_alt, state["altitude_ft"])
            min_fwd_speed = min(min_fwd_speed, state["forward_speed_fps"])
            max_abs_roll = max(max_abs_roll, abs(roll_deg))
            max_abs_pitch = max(max_abs_pitch, abs(pitch_deg))
            max_abs_yaw_rate = max(max_abs_yaw_rate, abs(yaw_rate_deg_s))

            rows.append({
                "case": cfg["name"],
                "phase": "pulse",
                "time_s": float(t),
                "forward_ft": float(state["forward_ft"]),
                "forward_speed_fps": float(state["forward_speed_fps"]),
                "cross_track_ft": float(state["cross_track_ft"]),
                "lateral_speed_fps": float(state["lateral_speed_fps"]),
                "altitude_ft": float(state["altitude_ft"]),
                "vertical_speed_fps": float(state["vertical_speed_fps"]),
                "heading_change_deg": float(
                    heading_delta_deg(
                        state["heading_error_deg"],
                        initial["heading_error_deg"],
                    )
                ),
                "heading_error_deg": float(state["heading_error_deg"]),
                "roll_deg": float(roll_deg),
                "pitch_deg": float(pitch_deg),
                "yaw_rate_deg_s": float(yaw_rate_deg_s),
            })

            pulse_end = state.copy()
            reason = safety_reason(state)
            if reason:
                termination = "pulse:" + reason
                break

        if termination == "completed":
            for step in range(recovery_steps):
                obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
                base, _ = stage2_model.predict(obs2, deterministic=True)
                base = np.asarray(base, dtype=np.float32).reshape(-1)

                state, used = raw_policy_cycle(
                    env2, fdm, base, lat0, lon0, mission_heading
                )

                t = PULSE_SECONDS + (step + 1) * dt
                rows.append({
                    "case": cfg["name"],
                    "phase": "recovery",
                    "time_s": float(t),
                    "forward_ft": float(state["forward_ft"]),
                    "forward_speed_fps": float(state["forward_speed_fps"]),
                    "cross_track_ft": float(state["cross_track_ft"]),
                    "lateral_speed_fps": float(state["lateral_speed_fps"]),
                    "altitude_ft": float(state["altitude_ft"]),
                    "vertical_speed_fps": float(state["vertical_speed_fps"]),
                    "heading_change_deg": float(
                        heading_delta_deg(
                            state["heading_error_deg"],
                            initial["heading_error_deg"],
                        )
                    ),
                    "heading_error_deg": float(state["heading_error_deg"]),
                    "roll_deg": float(math.degrees(state["roll_rad"])),
                    "pitch_deg": float(math.degrees(state["pitch_rad"])),
                    "yaw_rate_deg_s": float(math.degrees(state["yaw_rate_rad_s"])),
                })

                reason = safety_reason(state)
                if reason:
                    termination = "recovery:" + reason
                    break

        final = snapshot(fdm, lat0, lon0, mission_heading)

        summary = {
            "case": cfg["name"],
            "delta_a2": float(cfg["delta_a2"]),
            "delta_a3": float(cfg["delta_a3"]),
            "same_fdm": bool(start["same_fdm"]),
            "entry_forward_ft": float(initial["forward_ft"]),
            "entry_forward_speed_fps": float(initial["forward_speed_fps"]),
            "entry_altitude_ft": float(initial["altitude_ft"]),
            "entry_vs_fps": float(initial["vertical_speed_fps"]),
            "entry_heading_error_deg": float(initial["heading_error_deg"]),
            "entry_roll_deg": float(math.degrees(initial["roll_rad"])),
            "entry_pitch_deg": float(math.degrees(initial["pitch_rad"])),
            "pulse_heading_change_deg": float(
                heading_delta_deg(
                    pulse_end["heading_error_deg"],
                    initial["heading_error_deg"],
                )
            ),
            "pulse_forward_speed_fps": float(pulse_end["forward_speed_fps"]),
            "pulse_altitude_ft": float(pulse_end["altitude_ft"]),
            "pulse_vs_fps": float(pulse_end["vertical_speed_fps"]),
            "pulse_roll_deg": float(math.degrees(pulse_end["roll_rad"])),
            "pulse_pitch_deg": float(math.degrees(pulse_end["pitch_rad"])),
            "pulse_yaw_rate_deg_s": float(math.degrees(pulse_end["yaw_rate_rad_s"])),
            "pulse_cross_track_ft": float(pulse_end["cross_track_ft"]),
            "min_forward_speed_fps": float(min_fwd_speed),
            "min_altitude_ft": float(min_alt),
            "max_altitude_ft": float(max_alt),
            "max_abs_roll_deg": float(max_abs_roll),
            "max_abs_pitch_deg": float(max_abs_pitch),
            "max_abs_yaw_rate_deg_s": float(max_abs_yaw_rate),
            "final_heading_change_deg": float(
                heading_delta_deg(
                    final["heading_error_deg"],
                    initial["heading_error_deg"],
                )
            ),
            "final_forward_speed_fps": float(final["forward_speed_fps"]),
            "final_altitude_ft": float(final["altitude_ft"]),
            "termination": termination,
            "safe": bool(termination == "completed"),
        }

        return summary, rows
    finally:
        close_case(start)

print("=" * 120)
print("FORWARD-FLIGHT TURN AUTHORITY IDENTIFICATION V1")
print("=" * 120)
print("Turn entry: ~240 ft while still moving forward.")
print("No descent. No landing. No PPO training.")
print()

summaries = []
trace_rows = []

for i, cfg in enumerate(CASES, 1):
    print(
        f"[{i}/{len(CASES)}] {cfg['name']} | "
        f"dA2={cfg['delta_a2']:+.2f} "
        f"dA3={cfg['delta_a3']:+.2f}"
    )

    summary, rows = run_case(cfg)
    summaries.append(summary)
    trace_rows.extend(rows)

    print(
        f"  entry: FWD={summary['entry_forward_ft']:.2f} ft | "
        f"V={summary['entry_forward_speed_fps']:.3f} ft/s | "
        f"ALT={summary['entry_altitude_ft']:.3f} ft | "
        f"HDG={summary['entry_heading_error_deg']:+.3f} deg"
    )
    print(
        f"  pulse: dHDG={summary['pulse_heading_change_deg']:+.4f} deg | "
        f"V={summary['pulse_forward_speed_fps']:.3f} | "
        f"ALT={summary['pulse_altitude_ft']:.3f} | "
        f"VS={summary['pulse_vs_fps']:+.3f} | "
        f"ROLL={summary['pulse_roll_deg']:+.3f} | "
        f"YAW_RATE={summary['pulse_yaw_rate_deg_s']:+.3f} | "
        f"term={summary['termination']}"
    )
    print()

save_csv(RESULT_DIR / "case_summary.csv", summaries)
save_csv(RESULT_DIR / "full_trace.csv", trace_rows)

baseline = next(s for s in summaries if s["case"] == "baseline")

effects = []
for s in summaries:
    effects.append({
        "case": s["case"],
        "delta_a2": s["delta_a2"],
        "delta_a3": s["delta_a3"],
        "net_heading_effect_deg": float(
            s["pulse_heading_change_deg"] - baseline["pulse_heading_change_deg"]
        ),
        "net_roll_effect_deg": float(
            s["pulse_roll_deg"] - baseline["pulse_roll_deg"]
        ),
        "net_speed_effect_fps": float(
            s["pulse_forward_speed_fps"] - baseline["pulse_forward_speed_fps"]
        ),
        "net_altitude_effect_ft": float(
            s["pulse_altitude_ft"] - baseline["pulse_altitude_ft"]
        ),
        "net_vs_effect_fps": float(
            s["pulse_vs_fps"] - baseline["pulse_vs_fps"]
        ),
        "safe": bool(s["safe"]),
    })

save_csv(RESULT_DIR / "baseline_subtracted_effects.csv", effects)

print("=" * 120)
print("BASELINE-SUBTRACTED FORWARD-TURN EFFECTS")
print("=" * 120)

for e in effects:
    print(
        f"{e['case']:18s} | "
        f"net dHDG={e['net_heading_effect_deg']:+.4f} deg | "
        f"net roll={e['net_roll_effect_deg']:+.4f} deg | "
        f"net dV={e['net_speed_effect_fps']:+.4f} ft/s | "
        f"net dALT={e['net_altitude_effect_ft']:+.4f} ft | "
        f"net dVS={e['net_vs_effect_fps']:+.4f} ft/s | "
        f"safe={e['safe']}"
    )

safe_effects = [e for e in effects if e["case"] != "baseline" and e["safe"]]
best = None

if safe_effects:
    best = max(
        safe_effects,
        key=lambda e: (
            e["net_heading_effect_deg"],
            -abs(e["net_altitude_effect_ft"]),
            -abs(e["net_speed_effect_fps"]),
        ),
    )

ready = bool(best is not None and best["net_heading_effect_deg"] > 0.10)

print()
print("SELECTED POSITIVE-TURN CANDIDATE:", best["case"] if best else "NONE")
print("READY FOR CLOSED-LOOP 5-DEG TURN:", ready)

final_summary = {
    "training_type": "NONE",
    "descent_used": False,
    "landing_used": False,
    "turn_entry_forward_ft": TURN_ENTRY_FORWARD_FT,
    "selected_positive_turn_candidate": best,
    "ready_for_closed_loop_5deg_turn": ready,
    "cases": summaries,
    "baseline_subtracted": effects,
}

with (RESULT_DIR / "final_summary.json").open("w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=2)

print()
print("Saved:")
print(" ", RESULT_DIR / "case_summary.csv")
print(" ", RESULT_DIR / "baseline_subtracted_effects.csv")
print(" ", RESULT_DIR / "full_trace.csv")
print(" ", RESULT_DIR / "final_summary.json")
