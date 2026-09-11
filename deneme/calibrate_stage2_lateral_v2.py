from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill
from helicopter_env_stage2_refine import HelicopterEnvStage2Refine


# ============================================================
# MODELS
# ============================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)
STAGE2_MODEL_PATH = (
    "models_stage2_refine/"
    "AH1S_STAGE2_REFINE_SUCCESS.zip"
)

# ============================================================
# LOCKED ALTITUDE TEACHER — V1 WINNER
# ============================================================

TARGET_ALT = 300.0
TARGET_DISTANCE = 300.0
COL_BIAS = 0.24
ALT_KP = 0.030
VS_KD = 0.120
COL_BIAS_FADE_DISTANCE = 240.0
MAX_COL_ACTION_CORR = 0.40

# ============================================================
# LATERAL V2
# ============================================================

# V1's feedback residuals were too small to reveal useful lateral
# authority.  V2 directly sweeps normalized PPO residual biases on
# action[2] (aileron) and action[3] (rudder), then performs a local
# refinement around the best pair.
COARSE_BIASES = [-0.80, -0.50, -0.30, 0.00, 0.30, 0.50, 0.80]
FINE_OFFSETS = [-0.16, -0.08, 0.00, 0.08, 0.16]
BIAS_RAMP_DISTANCE = 40.0

# ============================================================
# QUALITY LIMITS
# ============================================================

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
HANDOFF_STABLE_TIME = 5.0
PRESENT_MIN_ALT = 290.0
PRESENT_MAX_ALT = 310.0
PRESENT_MAX_CROSS = 5.0
CROSS_ALT_MIN = 295.0
CROSS_ALT_MAX = 305.0
CROSS_MAX_VS = 2.0
EARTH_RADIUS_FT = 20_902_231.0

# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path("results_stage2_lateral_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
COARSE_CSV = OUT_DIR / "coarse_bias_search.csv"
FINE_CSV = OUT_DIR / "fine_bias_search.csv"
BEST_TRACE_CSV = OUT_DIR / "best_lateral_trace.csv"


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
    raise RuntimeError("Active JSBSim FDM not found")


def heading_rad(fdm):
    for key in ["attitude/heading-true-rad", "attitude/psi-rad"]:
        x = fdm_float(fdm, key)
        if np.isfinite(x):
            return x
    return float("nan")


def latitude_deg(fdm):
    for key in ["position/lat-gc-deg", "position/lat-geod-deg"]:
        x = fdm_float(fdm, key)
        if np.isfinite(x):
            return x
    return float("nan")


def longitude_deg(fdm):
    return fdm_float(fdm, "position/long-gc-deg")


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


def wrap_angle(x):
    return math.atan2(math.sin(x), math.cos(x))


def print_rule(title):
    print("\n" + "=" * 138)
    print(title)
    print("=" * 138)


def save_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


stage1_model = PPO.load(STAGE1_MODEL_PATH)
stage2_model = PPO.load(STAGE2_MODEL_PATH)


# ============================================================
# TRUE STAGE-1 HANDOFF
# ============================================================

def build_handoff():
    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )
    obs, info = env.reset()
    fdm = get_fdm(env)
    mission_heading = heading_rad(fdm)
    dt = float(getattr(env, "dt", 0.075))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.075

    stable_time = 0.0
    handoff = None

    for step in range(int(STAGE1_MAX_TIME / dt)):
        action, _ = stage1_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        alt = info_float(info, "altitude")
        vs = info_float(info, "vertical_speed")
        vn = info_float(info, "vn", 0.0)
        ve = info_float(info, "ve", 0.0)
        hs = float(np.hypot(vn, ve))
        drift = info_float(info, "drift", 999.0)

        stable = (
            295.0 <= alt <= 305.0
            and abs(vs) <= 0.50
            and hs <= 1.0
            and drift <= 3.0
        )
        stable_time = stable_time + dt if stable else 0.0

        if stable_time >= HANDOFF_STABLE_TIME:
            handoff = {
                "altitude": alt,
                "vs": vs,
                "hs": hs,
                "drift": drift,
            }
            break

        if terminated and not bool(info.get("success", False)):
            break
        if truncated:
            break

    if handoff is None:
        env.close()
        raise RuntimeError("Stable Stage-1 handoff not reached")

    return env, fdm, mission_heading, handoff


# ============================================================
# ATTACH STAGE 2 TO SAME FDM
# ============================================================

def attach_stage2(fdm, mission_heading):
    env = HelicopterEnvStage2Refine()
    env.reset()  # disposable internal initialization only
    env.fdm = fdm

    if hasattr(env, "forward_distance"):
        env.forward_distance = 0.0
    if hasattr(env, "target_heading"):
        env.target_heading = float(mission_heading)

    for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
        if hasattr(env, attr):
            setattr(env, attr, 0)

    obs = np.asarray(env._get_obs(), dtype=np.float32)
    return env, obs


# ============================================================
# LOCKED ALTITUDE TEACHER
# ============================================================

def collective_correction(altitude, vs, distance):
    gate = float(np.clip(1.0 - distance / COL_BIAS_FADE_DISTANCE, 0.0, 1.0))
    corr = (
        COL_BIAS * gate
        + ALT_KP * (TARGET_ALT - altitude)
        - VS_KD * vs
    )
    return float(np.clip(corr, -MAX_COL_ACTION_CORR, MAX_COL_ACTION_CORR))


def lateral_gate(distance):
    return float(np.clip(distance / BIAS_RAMP_DISTANCE, 0.0, 1.0))


# ============================================================
# ONE FULL CONTINUOUS CASE
# ============================================================

def run_case(aileron_bias, rudder_bias, detailed=False):
    env1, fdm, mission_heading, handoff = build_handoff()
    fdm_id = id(fdm)
    lat0, lon0 = latitude_deg(fdm), longitude_deg(fdm)

    env2, obs = attach_stage2(fdm, mission_heading)
    if id(get_fdm(env2)) != fdm_id:
        raise RuntimeError("FDM continuity failed")

    dt = float(getattr(env2, "dt", 0.075))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.075

    min_alt = handoff["altitude"]
    max_alt = handoff["altitude"]
    max_cross = 0.0
    max_abs_lat = 0.0
    max_abs_heading = 0.0
    max_abs_roll = 0.0
    clipped_a2 = 0
    clipped_a3 = 0
    nsteps = 0
    used_a2_min, used_a2_max = 999.0, -999.0
    used_a3_min, used_a3_max = 999.0, -999.0
    phy_ail_min, phy_ail_max = 999.0, -999.0
    phy_rud_min, phy_rud_max = 999.0, -999.0
    crossing = None
    failure = False
    termination_reason = ""
    trace = []
    next_print = 0.0

    for step in range(int(STAGE2_MAX_TIME / dt)):
        base_action, _ = stage2_model.predict(obs, deterministic=True)
        base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
        action = base_action.copy()

        alt0 = fdm_float(fdm, "position/h-agl-ft")
        vs0 = fdm_float(fdm, "velocities/h-dot-fps")
        dist0 = float(getattr(env2, "forward_distance", 0.0))

        col_corr = collective_correction(alt0, vs0, dist0)
        action[0] = np.clip(action[0] + col_corr, -1.0, 1.0)

        gate = lateral_gate(dist0)
        req_a2 = float(base_action[2] + gate * aileron_bias)
        req_a3 = float(base_action[3] + gate * rudder_bias)
        clipped_a2 += int(req_a2 < -1.0 or req_a2 > 1.0)
        clipped_a3 += int(req_a3 < -1.0 or req_a3 > 1.0)
        action[2] = np.clip(req_a2, -1.0, 1.0)
        action[3] = np.clip(req_a3, -1.0, 1.0)

        used_a2_min = min(used_a2_min, float(action[2]))
        used_a2_max = max(used_a2_max, float(action[2]))
        used_a3_min = min(used_a3_min, float(action[3]))
        used_a3_max = max(used_a3_max, float(action[3]))

        obs, _, terminated, truncated, info = env2.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        nsteps += 1
        t = (step + 1) * dt

        alt = info_float(info, "altitude", fdm_float(fdm, "position/h-agl-ft"))
        vs = info_float(info, "vertical_speed", fdm_float(fdm, "velocities/h-dot-fps", 0.0))
        fwd = info_float(info, "forward_velocity", fdm_float(fdm, "velocities/u-aero-fps", 0.0))
        latv = info_float(info, "lateral_velocity", fdm_float(fdm, "velocities/v-aero-fps", 0.0))
        roll = info_float(info, "roll", fdm_float(fdm, "attitude/roll-rad", 0.0))
        distance = info_float(info, "forward_distance", getattr(env2, "forward_distance", 0.0))
        head_err = wrap_angle(heading_rad(fdm) - mission_heading)

        north, east = local_ne_ft(latitude_deg(fdm), longitude_deg(fdm), lat0, lon0)
        ground_fwd, cross = mission_axes(north, east, mission_heading)

        phy_ail = info_float(info, "aileron", fdm_float(fdm, "fcs/aileron-cmd-norm"))
        phy_rud = info_float(info, "rudder", fdm_float(fdm, "fcs/rudder-cmd-norm"))
        phy_ail_min, phy_ail_max = min(phy_ail_min, phy_ail), max(phy_ail_max, phy_ail)
        phy_rud_min, phy_rud_max = min(phy_rud_min, phy_rud), max(phy_rud_max, phy_rud)

        min_alt, max_alt = min(min_alt, alt), max(max_alt, alt)
        max_cross = max(max_cross, abs(cross))
        max_abs_lat = max(max_abs_lat, abs(latv))
        max_abs_heading = max(max_abs_heading, abs(head_err))
        max_abs_roll = max(max_abs_roll, abs(roll))

        row = {
            "time_s": t,
            "distance_ft": distance,
            "ground_forward_ft": ground_fwd,
            "cross_track_ft": cross,
            "altitude_ft": alt,
            "vertical_speed_fps": vs,
            "forward_speed_fps": fwd,
            "lateral_speed_fps": latv,
            "heading_error_deg": math.degrees(head_err),
            "roll_deg": math.degrees(roll),
            "base_a2": float(base_action[2]),
            "base_a3": float(base_action[3]),
            "used_a2": float(action[2]),
            "used_a3": float(action[3]),
            "physical_aileron": phy_ail,
            "physical_rudder": phy_rud,
            "col_corr": col_corr,
            "lateral_gate": gate,
        }
        trace.append(row)

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | D={distance:7.2f} | GND={ground_fwd:7.2f} | "
                f"X={cross:+7.2f} | ALT={alt:7.2f} | VS={vs:+6.2f} | "
                f"LAT={latv:+6.2f} | HEAD={math.degrees(head_err):+6.2f}deg | "
                f"ROLL={math.degrees(roll):+6.2f}deg | A2={action[2]:+6.3f} | "
                f"A3={action[3]:+6.3f} | PAIL={phy_ail:+.5f} | PRUD={phy_rud:+.5f}"
            )
            next_print += 2.5

        if distance >= TARGET_DISTANCE:
            crossing = row.copy()
            break

        if terminated:
            if not bool(info.get("success", False)):
                failure = True
                termination_reason = str(info.get("termination_reason", "terminated"))
            break

        if truncated:
            termination_reason = "truncated"
            break

    env2.fdm = None
    env1.close()

    reached = crossing is not None
    if reached:
        c_alt = crossing["altitude_ft"]
        c_vs = crossing["vertical_speed_fps"]
        c_cross = crossing["cross_track_ft"]
        c_head = crossing["heading_error_deg"]
        c_lat = crossing["lateral_speed_fps"]
        c_ground = crossing["ground_forward_ft"]
    else:
        c_alt = c_vs = c_cross = c_head = c_lat = c_ground = float("nan")

    result = {
        "aileron_bias": float(aileron_bias),
        "rudder_bias": float(rudder_bias),
        "reached_300": bool(reached),
        "failure": bool(failure),
        "termination_reason": termination_reason,
        "handoff_alt": float(handoff["altitude"]),
        "min_alt": float(min_alt),
        "max_alt": float(max_alt),
        "max_drop": float(handoff["altitude"] - min_alt),
        "max_cross": float(max_cross),
        "max_abs_lat": float(max_abs_lat),
        "max_abs_heading_deg": float(math.degrees(max_abs_heading)),
        "max_abs_roll_deg": float(math.degrees(max_abs_roll)),
        "crossing_alt": float(c_alt),
        "crossing_vs": float(c_vs),
        "crossing_cross": float(c_cross),
        "crossing_heading_deg": float(c_head),
        "crossing_lat": float(c_lat),
        "crossing_ground": float(c_ground),
        "clip_ail_fraction": clipped_a2 / max(nsteps, 1),
        "clip_rud_fraction": clipped_a3 / max(nsteps, 1),
        "min_used_a2": float(used_a2_min),
        "max_used_a2": float(used_a2_max),
        "min_used_a3": float(used_a3_min),
        "max_used_a3": float(used_a3_max),
        "min_physical_ail": float(phy_ail_min),
        "max_physical_ail": float(phy_ail_max),
        "min_physical_rud": float(phy_rud_min),
        "max_physical_rud": float(phy_rud_max),
        "presentation_pass": bool(
            reached
            and not failure
            and min_alt >= PRESENT_MIN_ALT
            and max_alt <= PRESENT_MAX_ALT
            and max_cross <= PRESENT_MAX_CROSS
            and CROSS_ALT_MIN <= c_alt <= CROSS_ALT_MAX
            and abs(c_vs) <= CROSS_MAX_VS
        ),
        "trace": trace,
    }
    return result


# ============================================================
# SCORE
# ============================================================

def score_case(r):
    if not r["reached_300"]:
        return 1_000_000.0 + 1000.0 * r["max_cross"]

    low = max(0.0, PRESENT_MIN_ALT - r["min_alt"])
    high = max(0.0, r["max_alt"] - PRESENT_MAX_ALT)

    score = (
        1200.0 * r["max_cross"]
        + 300.0 * abs(r["crossing_cross"])
        + 120.0 * r["max_abs_lat"]
        + 80.0 * r["max_abs_heading_deg"]
        + 35.0 * r["max_abs_roll_deg"]
        + 3000.0 * low
        + 2000.0 * high
        + 200.0 * abs(r["crossing_alt"] - TARGET_ALT)
        + 100.0 * abs(r["crossing_vs"])
        + 800.0 * r["clip_ail_fraction"]
        + 500.0 * r["clip_rud_fraction"]
    )
    if r["failure"]:
        score += 500_000.0
    return float(score)


# ============================================================
# PROBE ACTUATOR SCALES
# ============================================================

print_rule("STAGE 2 LATERAL V2 — DIRECT AILERON + RUDDER AUTHORITY SEARCH")
print("Stage 1: LOCKED")
print(f"Altitude teacher: B={COL_BIAS:.2f}, KP={ALT_KP:.3f}, KD={VS_KD:.3f}")

probe = HelicopterEnvStage2Refine()
probe.reset()
print("\nStage-2 actuator mapping attributes:")
for name in [
    "collective_scale", "elevator_scale", "aileron_scale", "rudder_scale",
    "base_collective", "base_elevator", "base_aileron", "base_rudder",
]:
    print(f"{name:20s}: {getattr(probe, name, 'N/A')}")
probe.close()


# ============================================================
# REFERENCE
# ============================================================

print_rule("REFERENCE — ALTITUDE TEACHER ONLY")
reference = run_case(0.0, 0.0)
for key in [
    "reached_300", "min_alt", "max_alt", "max_drop", "max_cross",
    "max_abs_lat", "max_abs_heading_deg", "max_abs_roll_deg",
    "crossing_alt", "crossing_vs", "crossing_cross",
    "crossing_heading_deg", "crossing_lat",
]:
    print(f"{key:27s}: {reference[key]}")


# ============================================================
# COARSE SEARCH
# ============================================================

print_rule("PHASE 1 — COARSE DIRECT BIAS SWEEP")
coarse_rows = []
case = 0
for ail in COARSE_BIASES:
    for rud in COARSE_BIASES:
        case += 1
        r = run_case(ail, rud)
        row = {k: v for k, v in r.items() if k != "trace"}
        row["score"] = score_case(r)
        row["case"] = case
        coarse_rows.append(row)
        print(
            f"C{case:02d} | AIL={ail:+.2f} RUD={rud:+.2f} | "
            f"MAX_X={r['max_cross']:6.2f} | CROSS_X={r['crossing_cross']:+6.2f} | "
            f"LAT={r['max_abs_lat']:5.2f} | HEAD={r['max_abs_heading_deg']:5.2f}deg | "
            f"MIN_ALT={r['min_alt']:6.2f} | "
            f"CLIP=({100*r['clip_ail_fraction']:4.1f}%,{100*r['clip_rud_fraction']:4.1f}%) | "
            f"REACH={r['reached_300']} | SCORE={row['score']:9.1f}"
        )

save_rows(COARSE_CSV, coarse_rows)
coarse_sorted = sorted(coarse_rows, key=lambda x: x["score"])

print_rule("TOP 12 COARSE LATERAL CANDIDATES")
for i, row in enumerate(coarse_sorted[:12], 1):
    print(
        f"{i:2d}. AIL={row['aileron_bias']:+.2f} RUD={row['rudder_bias']:+.2f} | "
        f"MAX_X={row['max_cross']:6.2f} | CROSS_X={row['crossing_cross']:+6.2f} | "
        f"LAT={row['max_abs_lat']:5.2f} | HEAD={row['max_abs_heading_deg']:5.2f}deg | "
        f"ALT={row['crossing_alt']:6.2f} | VS={row['crossing_vs']:+5.2f} | "
        f"SCORE={row['score']:9.1f}"
    )


# ============================================================
# FINE SEARCH
# ============================================================

best_coarse = coarse_sorted[0]
ail_values = sorted({
    float(np.clip(best_coarse["aileron_bias"] + x, -1.0, 1.0))
    for x in FINE_OFFSETS
})
rud_values = sorted({
    float(np.clip(best_coarse["rudder_bias"] + x, -1.0, 1.0))
    for x in FINE_OFFSETS
})

print_rule("PHASE 2 — FINE SEARCH AROUND BEST COARSE PAIR")
print("Aileron:", ail_values)
print("Rudder :", rud_values)

fine_rows = []
case = 0
for ail in ail_values:
    for rud in rud_values:
        case += 1
        r = run_case(ail, rud)
        row = {k: v for k, v in r.items() if k != "trace"}
        row["score"] = score_case(r)
        row["case"] = case
        fine_rows.append(row)
        print(
            f"F{case:02d} | AIL={ail:+.3f} RUD={rud:+.3f} | "
            f"MAX_X={r['max_cross']:6.2f} | CROSS_X={r['crossing_cross']:+6.2f} | "
            f"LAT={r['max_abs_lat']:5.2f} | HEAD={r['max_abs_heading_deg']:5.2f}deg | "
            f"MIN_ALT={r['min_alt']:6.2f} | REACH={r['reached_300']} | SCORE={row['score']:9.1f}"
        )

save_rows(FINE_CSV, fine_rows)
fine_sorted = sorted(fine_rows, key=lambda x: x["score"])

print_rule("TOP 12 FINE LATERAL CANDIDATES")
for i, row in enumerate(fine_sorted[:12], 1):
    print(
        f"{i:2d}. AIL={row['aileron_bias']:+.3f} RUD={row['rudder_bias']:+.3f} | "
        f"MAX_X={row['max_cross']:6.2f} | CROSS_X={row['crossing_cross']:+6.2f} | "
        f"LAT={row['max_abs_lat']:5.2f} | HEAD={row['max_abs_heading_deg']:5.2f}deg | "
        f"MIN_ALT={row['min_alt']:6.2f} | ALT={row['crossing_alt']:6.2f} | "
        f"VS={row['crossing_vs']:+5.2f} | SCORE={row['score']:9.1f}"
    )


# ============================================================
# BEST DETAILED RUN
# ============================================================

best = fine_sorted[0]
print_rule("BEST V2 LATERAL CONFIG — FULL DETAILED FLIGHT")
print(f"AILERON BIAS = {best['aileron_bias']:+.3f}")
print(f"RUDDER BIAS  = {best['rudder_bias']:+.3f}\n")

best_result = run_case(
    best["aileron_bias"],
    best["rudder_bias"],
    detailed=True,
)
save_rows(BEST_TRACE_CSV, best_result["trace"])

print_rule("STAGE 2 LATERAL V2 FINAL RESULT")
for key in [
    "aileron_bias", "rudder_bias", "reached_300", "failure",
    "termination_reason", "handoff_alt", "min_alt", "max_alt", "max_drop",
    "max_cross", "max_abs_lat", "max_abs_heading_deg", "max_abs_roll_deg",
    "crossing_alt", "crossing_vs", "crossing_cross", "crossing_heading_deg",
    "crossing_lat", "crossing_ground", "clip_ail_fraction", "clip_rud_fraction",
    "min_used_a2", "max_used_a2", "min_used_a3", "max_used_a3",
    "min_physical_ail", "max_physical_ail", "min_physical_rud", "max_physical_rud",
    "presentation_pass",
]:
    print(f"{key:27s}: {best_result[key]}")

improvement = reference["max_cross"] - best_result["max_cross"]
print(f"\nCross-track improvement     : {improvement:+.3f} ft")

if best_result["presentation_pass"]:
    print("PRESENTATION QUALITY         : TRUE")
    print("Next: distill altitude + lateral teacher behavior into Stage-2 PPO.")
elif improvement > 10.0:
    print("PRESENTATION QUALITY         : FALSE")
    print("Control direction is identified. Next: small feedback around this feed-forward pair.")
else:
    print("PRESENTATION QUALITY         : FALSE")
    print("Direct aileron/rudder biases still do not materially affect cross-track.")
    print("Next: explicit control-effectiveness pulse identification; no more blind gain sweeps.")

print("\nSaved:", COARSE_CSV)
print("Saved:", FINE_CSV)
print("Saved:", BEST_TRACE_CSV)
