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

OUT_DIR = Path("results_stage3_brake_teacher_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "brake_teacher_sweep.csv"
BEST_TRACE_CSV = OUT_DIR / "best_brake_teacher_trace.csv"


# ============================================================
# MISSION
# ============================================================

TARGET_FORWARD_FT = 300.0
TARGET_ALT_FT = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
STAGE3_MAX_TIME = 35.0

HANDOFF_STABLE_TIME = 5.0
CONTROL_DT = 0.075

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040

# We do not touch Stage 1 or the locked Stage-2 cruise model.
# The braking teacher is allowed to take over only late in Stage 2.
TEACHER_ENABLE_FORWARD_FT = 120.0

# Final Stage-2 distilled lateral/yaw outputs.
LOCKED_AILERON_ACTION = -0.230
LOCKED_RUDDER_ACTION = 0.000

# The endpoint PPO elevator action observed in the locked flight.
ELEVATOR_TRIM_ACTION = 0.013725

# Altitude support during braking. This is TEACHER logic only and
# will later be distilled; it is not a final runtime controller.
ALT_KP = 0.030
VS_KD = 0.120
MAX_ALT_CORR = 0.40

# Pitch-rate damping sign:
# negative elevator residual produced nose-up/braking.
# Positive q is therefore opposed with a positive elevator correction.
Q_DAMP = 2.0

# The V2 authority experiment measured ~0.60 ft/s^2 at full braking.
# Search slightly below that so the velocity profile is achievable.
A_DES_VALUES = [0.35, 0.45, 0.55]

# Converts velocity-profile error to normalized elevator residual.
KV_VALUES = [0.25, 0.40, 0.60]

MAX_ELEVATOR_RESIDUAL = 1.0
V_DES_MAX = 12.0

# Longitudinal stop must be held, not crossed momentarily.
STOP_POS_TOL_FT = 5.0
STOP_SPEED_TOL_FPS = 1.0
STOP_HOLD_SECONDS = 3.0

# Safety / quality.
ALT_SAFE_MIN = 290.0
ALT_SAFE_MAX = 310.0
MAX_ABS_PITCH_DEG = 8.0
MAX_ABS_ROLL_DEG = 10.0
MAX_CROSS_SAFE_FT = 12.0

# Bonus "endpoint hover ready" criterion. Stage-3 V1 primarily solves
# longitudinal stopping; these fields tell us whether lateral hover
# is already good enough without another teacher.
HOVER_CROSS_TOL_FT = 5.0
HOVER_LAT_SPEED_TOL_FPS = 1.0
HOVER_VS_TOL_FPS = 0.75
HOVER_HEADING_TOL_DEG = 1.0


# ============================================================
# GEOMETRY
# ============================================================

EARTH_RADIUS_FT = 20_902_231.0


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


def wrap_angle(x):
    return math.atan2(math.sin(x), math.cos(x))


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


# ============================================================
# STATE
# ============================================================

def state_snapshot(
    fdm,
    lat0,
    lon0,
    mission_heading,
):
    forward, cross = geometry(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    h_now = heading_rad(fdm)
    h_err = wrap_angle(
        h_now - mission_heading
    )

    return {
        "forward_ft":
            forward,

        "position_error_ft":
            TARGET_FORWARD_FT - forward,

        "cross_track_ft":
            cross,

        "altitude_ft":
            fdm_float(
                fdm,
                "position/h-agl-ft",
            ),

        "forward_speed_fps":
            fdm_float(
                fdm,
                "velocities/u-aero-fps",
                0.0,
            ),

        "lateral_speed_fps":
            fdm_float(
                fdm,
                "velocities/v-aero-fps",
                0.0,
            ),

        "vertical_speed_fps":
            fdm_float(
                fdm,
                "velocities/h-dot-fps",
                0.0,
            ),

        "pitch_deg":
            math.degrees(
                fdm_float(
                    fdm,
                    "attitude/pitch-rad",
                    0.0,
                )
            ),

        "roll_deg":
            math.degrees(
                fdm_float(
                    fdm,
                    "attitude/roll-rad",
                    0.0,
                )
            ),

        "pitch_rate_rad_s":
            fdm_float(
                fdm,
                "velocities/q-rad_sec",
                0.0,
            ),

        "heading_error_deg":
            math.degrees(
                h_err
            ),

        "physical_collective":
            fdm_float(
                fdm,
                "fcs/collective-cmd-norm",
            ),

        "physical_elevator":
            fdm_float(
                fdm,
                "fcs/elevator-cmd-norm",
            ),

        "physical_aileron":
            fdm_float(
                fdm,
                "fcs/aileron-cmd-norm",
            ),

        "physical_rudder":
            fdm_float(
                fdm,
                "fcs/rudder-cmd-norm",
            ),
    }


def safe_state(s):
    return bool(
        ALT_SAFE_MIN
        <= s["altitude_ft"]
        <= ALT_SAFE_MAX

        and abs(s["pitch_deg"])
        <= MAX_ABS_PITCH_DEG

        and abs(s["roll_deg"])
        <= MAX_ABS_ROLL_DEG

        and abs(s["cross_track_ft"])
        <= MAX_CROSS_SAFE_FT
    )


# ============================================================
# RAW CONTINUOUS PHYSICS AFTER TEACHER TAKEOVER
# ============================================================

def raw_control_cycle(env2, action):
    a = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    a = np.clip(a, -1.0, +1.0)

    env2._apply_action(a)

    for _ in range(10):
        if not env2.fdm.run():
            raise RuntimeError(
                "JSBSim stopped during Stage-3 teacher."
            )


# ============================================================
# MODELS
# ============================================================

stage1_model = PPO.load(STAGE1_MODEL_PATH)
stage2_model = PPO.load(STAGE2_MODEL_PATH)


# ============================================================
# BUILD LOCKED STAGE1 -> STAGE2 CRUISE TO TEACHER TAKEOVER
# ============================================================

def build_teacher_start():
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

    for _step in range(
        int(STAGE1_MAX_TIME / dt1)
    ):
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

    # Stage-2 geometry origin = actual handoff position.
    stage2_lat0 = latitude_deg(fdm)
    stage2_lon0 = longitude_deg(fdm)

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
        getattr(env2, "dt", CONTROL_DT)
        or CONTROL_DT
    )

    if not np.isfinite(dt2) or dt2 <= 0:
        dt2 = CONTROL_DT

    # Cruise with the LOCKED Stage-2 PPO only.
    for _step in range(
        int(STAGE2_MAX_TIME / dt2)
    ):
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
            stage2_lat0,
            stage2_lon0,
            mission_heading,
        )

        if forward >= TEACHER_ENABLE_FORWARD_FT:
            # From this point onward we control the raw continuous
            # FDM ourselves so Stage-2's 300-ft termination cannot
            # stop the braking/hover experiment.
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
                "Stage 2 failed before teacher takeover."
            )

        if truncated:
            raise RuntimeError(
                "Stage 2 truncated before teacher takeover."
            )

    return (
        env1,
        env2,
        fdm,
        obs2,
        stage2_lat0,
        stage2_lon0,
        mission_heading,
    )


# ============================================================
# TEACHER
# ============================================================

def desired_forward_speed(
    remaining_ft,
    a_des,
):
    # Signed stopping-speed profile:
    #
    #   v_des = sign(remaining) * sqrt(2*a*|remaining|)
    #
    # It naturally starts braking before 300 ft and can also
    # correct an overshoot by requesting a small backward speed.
    if abs(remaining_ft) < 1e-9:
        return 0.0

    magnitude = math.sqrt(
        max(
            0.0,
            2.0
            * a_des
            * abs(remaining_ft),
        )
    )

    magnitude = min(
        magnitude,
        V_DES_MAX,
    )

    return (
        magnitude
        if remaining_ft > 0.0
        else -magnitude
    )


def teacher_action(
    obs,
    s,
    a_des,
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

    # -------------------------
    # Collective: altitude hold
    # -------------------------
    alt_error = (
        TARGET_ALT_FT
        - s["altitude_ft"]
    )

    col_corr = (
        ALT_KP * alt_error
        - VS_KD * s["vertical_speed_fps"]
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
            base_action[0]
            + col_corr,
            -1.0,
            +1.0,
        )
    )

    # -------------------------
    # Elevator: position-aware
    # braking velocity profile
    # -------------------------
    remaining = (
        TARGET_FORWARD_FT
        - s["forward_ft"]
    )

    v_des = desired_forward_speed(
        remaining,
        a_des,
    )

    speed_error = (
        s["forward_speed_fps"]
        - v_des
    )

    elevator_residual = (
        -kv * speed_error
        + Q_DAMP
        * s["pitch_rate_rad_s"]
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

    # Keep the already calibrated lateral/yaw teacher values locked.
    action[2] = LOCKED_AILERON_ACTION
    action[3] = LOCKED_RUDDER_ACTION

    return (
        action.astype(np.float32),
        float(v_des),
        float(speed_error),
        float(col_corr),
        float(elevator_residual),
    )


# ============================================================
# ONE TEACHER CASE
# ============================================================

def run_case(
    a_des,
    kv,
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
    full_hover_ready = False

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

    next_print = 0.0

    final_state = start_state.copy()

    for step in range(
        int(STAGE3_MAX_TIME / CONTROL_DT)
    ):
        s = state_snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        (
            action,
            v_des,
            speed_error,
            col_corr,
            elevator_residual,
        ) = teacher_action(
            obs2,
            s,
            a_des,
            kv,
        )

        raw_control_cycle(
            env2,
            action,
        )

        # Update physical forward distance for any Stage-2 observation
        # component that depends on the environment's distance state.
        s_after = state_snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        if hasattr(env2, "forward_distance"):
            env2.forward_distance = float(
                s_after["forward_ft"]
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

        row = dict(s_after)
        row.update(
            {
                "time_s":
                    float(t),

                "v_des_fps":
                    float(v_des),

                "speed_error_fps":
                    float(speed_error),

                "collective_correction":
                    float(col_corr),

                "elevator_residual":
                    float(elevator_residual),

                "action0":
                    float(action[0]),

                "action1":
                    float(action[1]),

                "action2":
                    float(action[2]),

                "action3":
                    float(action[3]),
            }
        )

        trace.append(row)

        final_state = s_after

        max_cross = max(
            max_cross,
            abs(
                s_after["cross_track_ft"]
            ),
        )

        min_alt = min(
            min_alt,
            s_after["altitude_ft"],
        )

        max_alt = max(
            max_alt,
            s_after["altitude_ft"],
        )

        max_abs_pitch = max(
            max_abs_pitch,
            abs(
                s_after["pitch_deg"]
            ),
        )

        max_abs_roll = max(
            max_abs_roll,
            abs(
                s_after["roll_deg"]
            ),
        )

        max_overshoot = max(
            max_overshoot,
            max(
                0.0,
                s_after["forward_ft"]
                - TARGET_FORWARD_FT,
            ),
        )

        if not safe_state(s_after):
            safe = False
            break

        stop_now = bool(
            abs(
                s_after["position_error_ft"]
            )
            <= STOP_POS_TOL_FT

            and abs(
                s_after["forward_speed_fps"]
            )
            <= STOP_SPEED_TOL_FPS
        )

        if stop_now:
            stop_hold_time += CONTROL_DT
        else:
            stop_hold_time = 0.0

        if stop_hold_time >= STOP_HOLD_SECONDS:
            stop_achieved = True

            full_hover_ready = bool(
                abs(
                    s_after["cross_track_ft"]
                )
                <= HOVER_CROSS_TOL_FT

                and abs(
                    s_after["lateral_speed_fps"]
                )
                <= HOVER_LAT_SPEED_TOL_FPS

                and abs(
                    s_after["vertical_speed_fps"]
                )
                <= HOVER_VS_TOL_FPS

                and abs(
                    s_after["heading_error_deg"]
                )
                <= HOVER_HEADING_TOL_DEG

                and 295.0
                <= s_after["altitude_ft"]
                <= 305.0
            )

            break

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"FWD={s_after['forward_ft']:7.2f} | "
                f"ERR={s_after['position_error_ft']:+7.2f} | "
                f"V={s_after['forward_speed_fps']:+6.2f} | "
                f"VDES={v_des:+6.2f} | "
                f"ALT={s_after['altitude_ft']:7.2f} | "
                f"VS={s_after['vertical_speed_fps']:+6.2f} | "
                f"X={s_after['cross_track_ft']:+6.2f} | "
                f"PITCH={s_after['pitch_deg']:+5.2f} | "
                f"A1={action[1]:+6.3f}"
            )

            next_print += 1.5

    result = {
        "a_des":
            float(a_des),

        "kv":
            float(kv),

        "safe":
            bool(safe),

        "stop_achieved":
            bool(stop_achieved),

        "full_hover_ready":
            bool(full_hover_ready),

        "start_forward_ft":
            float(
                start_state["forward_ft"]
            ),

        "start_speed_fps":
            float(
                start_state["forward_speed_fps"]
            ),

        "final_forward_ft":
            float(
                final_state["forward_ft"]
            ),

        "final_position_error_ft":
            float(
                final_state["position_error_ft"]
            ),

        "final_speed_fps":
            float(
                final_state["forward_speed_fps"]
            ),

        "stop_hold_time_s":
            float(stop_hold_time),

        "max_overshoot_ft":
            float(max_overshoot),

        "max_cross_ft":
            float(max_cross),

        "final_cross_ft":
            float(
                final_state["cross_track_ft"]
            ),

        "final_lateral_speed_fps":
            float(
                final_state["lateral_speed_fps"]
            ),

        "min_alt_ft":
            float(min_alt),

        "max_alt_ft":
            float(max_alt),

        "final_alt_ft":
            float(
                final_state["altitude_ft"]
            ),

        "final_vs_fps":
            float(
                final_state["vertical_speed_fps"]
            ),

        "max_abs_pitch_deg":
            float(max_abs_pitch),

        "max_abs_roll_deg":
            float(max_abs_roll),

        "final_heading_error_deg":
            float(
                final_state["heading_error_deg"]
            ),

        "trace":
            trace,
    }

    env2.fdm = None
    env1.close()

    return result


# ============================================================
# SWEEP
# ============================================================

rule("STAGE 3 — BRAKE-TO-ENDPOINT TEACHER V1")

print("Stage 1: LOCKED")
print("Stage 2 cruise: LOCKED")
print("No PPO training.")
print("Teacher goal: arrive at 300 ft AND reduce forward speed to hover.")
print(
    "Important: braking begins automatically from a "
    "position-dependent stopping-speed profile."
)
print()

results = []
case_no = 0

for a_des in A_DES_VALUES:
    for kv in KV_VALUES:
        case_no += 1

        r = run_case(
            a_des,
            kv,
            detailed=False,
        )

        results.append(r)

        print(
            f"T{case_no:02d} | "
            f"A_DES={a_des:.2f} | "
            f"KV={kv:.2f} | "
            f"STOP={r['stop_achieved']} | "
            f"HOVER={r['full_hover_ready']} | "
            f"SAFE={r['safe']} | "
            f"FWD={r['final_forward_ft']:7.2f} | "
            f"ERR={r['final_position_error_ft']:+6.2f} | "
            f"V={r['final_speed_fps']:+6.2f} | "
            f"OVR={r['max_overshoot_ft']:5.2f} | "
            f"XMAX={r['max_cross_ft']:5.2f} | "
            f"ALT=[{r['min_alt_ft']:.2f},{r['max_alt_ft']:.2f}] | "
            f"VS={r['final_vs_fps']:+5.2f}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_rows = []

for r in results:
    summary_rows.append(
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
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# SELECT
# ============================================================

def selection_key(r):
    return (
        0 if r["stop_achieved"] else 1,
        0 if r["full_hover_ready"] else 1,
        0 if r["safe"] else 1,
        abs(r["final_position_error_ft"]),
        abs(r["final_speed_fps"]),
        r["max_overshoot_ft"],
        r["max_cross_ft"],
        abs(r["final_alt_ft"] - TARGET_ALT_FT),
        abs(r["final_vs_fps"]),
    )


results.sort(
    key=selection_key
)

rule("TOP STAGE-3 TEACHER CANDIDATES")

for rank, r in enumerate(
    results[:6],
    start=1,
):
    print(
        f"{rank}. "
        f"A_DES={r['a_des']:.2f} | "
        f"KV={r['kv']:.2f} | "
        f"STOP={r['stop_achieved']} | "
        f"HOVER={r['full_hover_ready']} | "
        f"FWD={r['final_forward_ft']:.2f} | "
        f"ERR={r['final_position_error_ft']:+.2f} | "
        f"V={r['final_speed_fps']:+.2f} | "
        f"XMAX={r['max_cross_ft']:.2f} | "
        f"ALT={r['final_alt_ft']:.2f} | "
        f"VS={r['final_vs_fps']:+.2f}"
    )


best = results[0]


# ============================================================
# DETAILED REPEAT
# ============================================================

rule("BEST STAGE-3 TEACHER — FULL DETAILED FLIGHT")

best_detailed = run_case(
    best["a_des"],
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
# FINAL
# ============================================================

rule("STAGE 3 BRAKE-TO-ENDPOINT V1 FINAL RESULT")

for key in [
    "a_des",
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
            "Next: lock this Stage-3 teacher and distill it into PPO."
        )
    else:
        print("ENDPOINT HOVER QUALITY: NOT YET")
        print(
            "Next: keep this longitudinal brake teacher locked and "
            "solve only the remaining endpoint hover axis/altitude issue."
        )
else:
    print("LONGITUDINAL ENDPOINT STOP: FALSE")
    print(
        "Do not distill yet. Use the reported trace to refine only "
        "the Stage-3 stopping profile."
    )

print()
print("Saved:", SUMMARY_CSV)
print("Saved:", BEST_TRACE_CSV)
