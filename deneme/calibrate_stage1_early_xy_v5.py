import json
import os
import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill


# ============================================================
# FINAL STAGE-1 PPO
# ============================================================

MODEL_PATH = (
    "models_stage1_final_distilled/"
    "AH1S_STAGE1_FINAL_DISTILLED.zip"
)

model = PPO.load(MODEL_PATH)


# ============================================================
# IDENTIFIED XY CONTROL EFFECTIVENESS
# ============================================================

B_INV = np.array(
    [
        [-0.20338265, -0.00859620],
        [ 0.01952073, -0.14631468],
    ],
    dtype=np.float64,
)

CYCLIC_AUTHORITY = 0.026


# ============================================================
# EXISTING V3 OUTER XY TEACHER
#
# Keep the part that already worked well.
# Collective and rudder remain PPO.
# ============================================================

XY_KP = 0.000
XY_KD = 0.50
XY_ALPHA = 0.90

FF_ELE = -0.026

FF_HOLD = 0.75
FF_FADE = 1.50

TEACHER_FULL_ALT = 100.0
TEACHER_OFF_ALT = 140.0


# ============================================================
# V5 SEARCH
#
# Key discovery from V4B:
#
# Initial roll  ~ -0.31 deg
# Roll overshoot ~ -3.84 deg
# East drift grows at the same time.
#
# V5 therefore:
#   1) searches BOTH signs for initial aileron feed-forward,
#   2) adds one-sided negative-roll-rate damping,
#   3) adds a one-sided roll-floor guard only when roll becomes
#      more negative than a selected threshold.
#
# We DO NOT force roll to 0 deg.
# That could destroy the AH-1S natural hover bank.
# ============================================================

FF_AIL_VALUES = [
    -0.026,
    -0.013,
     0.000,
    +0.013,
    +0.026,
]

ROLL_RATE_GAIN_VALUES = [
    0.0,
    1.5,
    3.0,
    4.5,
    6.0,
]

ROLL_FLOOR_DEG_VALUES = [
    -2.4,
    -2.7,
    -3.0,
    -3.3,
]

ROLL_ANGLE_GAIN_VALUES = [
    0.0,
    4.0,
    8.0,
    12.0,
]


# ============================================================
# HELPERS
# ============================================================

def teacher_gate(altitude):
    if altitude <= TEACHER_FULL_ALT:
        return 1.0

    if altitude >= TEACHER_OFF_ALT:
        return 0.0

    return float(
        (TEACHER_OFF_ALT - altitude)
        /
        (TEACHER_OFF_ALT - TEACHER_FULL_ALT)
    )


def ff_scale(t):
    if t <= FF_HOLD:
        return 1.0

    end = FF_HOLD + FF_FADE

    if t >= end:
        return 0.0

    return float(
        (end - t)
        /
        FF_FADE
    )


def raw_fdm_state(env):
    return {
        "altitude": float(
            env.fdm["position/h-agl-ft"]
        ),
        "pitch": float(
            env.fdm["attitude/pitch-rad"]
        ),
        "roll": float(
            env.fdm["attitude/roll-rad"]
        ),
        "p": float(
            env.fdm["velocities/p-rad_sec"]
        ),
        "q": float(
            env.fdm["velocities/q-rad_sec"]
        ),
        "collective": float(
            env.fdm["fcs/collective-cmd-norm"]
        ),
        "elevator": float(
            env.fdm["fcs/elevator-cmd-norm"]
        ),
        "aileron": float(
            env.fdm["fcs/aileron-cmd-norm"]
        ),
        "rudder": float(
            env.fdm["fcs/rudder-cmd-norm"]
        ),
    }


def velocity_sign(value, threshold=0.08):
    if value > threshold:
        return 1

    if value < -threshold:
        return -1

    return 0


def score_result(r):
    slow_penalty = max(
        0.0,
        r["time100"] - 18.0,
    )

    # Roll overshoot is now included in the score,
    # but trajectory still dominates.
    roll_overshoot_penalty = max(
        0.0,
        r["max_negative_roll_deg"] - 3.2,
    )

    return (
        800.0 * r["early_max"]
        + 230.0 * r["path100"]
        + 260.0 * r["s_index"]
        + 100.0 * r["hs100"]
        + 45.0 * r["max_n"]
        + 55.0 * r["max_e"]
        + 35.0 * r["reversals"]
        + 120.0 * slow_penalty
        + 180.0 * roll_overshoot_penalty
    )


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    ff_ail,
    roll_rate_gain,
    roll_floor_deg,
    roll_angle_gain,
    total_time=24.0,
    detailed=False,
):
    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env.reset()

    dt = float(env.dt)
    max_steps = int(total_time / dt)

    teacher_prev = np.zeros(
        2,
        dtype=np.float64,
    )

    early_max = 0.0
    path100 = None
    drift100 = None
    hs100 = None
    time100 = None

    max_n = 0.0
    max_e = 0.0

    reversals_n = 0
    reversals_e = 0
    last_sign_n = 0
    last_sign_e = 0

    max_pitch100 = 0.0
    max_roll100 = 0.0
    most_negative_roll = 0.0
    max_abs_p100 = 0.0

    max_guard = 0.0
    guard_active_steps = 0
    teacher_steps = 0

    next_print = 0.0
    failed = False
    last_info = info

    for step in range(max_steps):
        t_before = step * dt

        # ----------------------------------------------------
        # PPO
        # ----------------------------------------------------

        ppo_action, _ = model.predict(
            obs,
            deterministic=True,
        )

        ppo_action = np.asarray(
            ppo_action,
            dtype=np.float32,
        ).reshape(-1)

        action = ppo_action.copy()

        raw_before = raw_fdm_state(env)

        altitude = raw_before["altitude"]
        roll = raw_before["roll"]
        roll_rate = raw_before["p"]

        north = float(
            info.get("north", 0.0)
        )
        east = float(
            info.get("east", 0.0)
        )
        vn = float(
            info.get("vn", 0.0)
        )
        ve = float(
            info.get("ve", 0.0)
        )

        gate = teacher_gate(
            altitude
        )

        teacher_norm = np.zeros(
            2,
            dtype=np.float64,
        )

        guard = 0.0
        rate_guard = 0.0
        angle_guard = 0.0

        # ----------------------------------------------------
        # DIRECT CYCLIC TEACHER
        # ----------------------------------------------------

        if gate > 0.0:
            teacher_steps += 1

            desired_accel = np.array(
                [
                    -XY_KP * north
                    -
                    XY_KD * vn,

                    -XY_KP * east
                    -
                    XY_KD * ve,
                ],
                dtype=np.float64,
            )

            desired_accel = np.clip(
                desired_accel,
                -0.22,
                +0.22,
            )

            feedback_delta = (
                B_INV
                @
                desired_accel
            )

            ff_delta = np.array(
                [
                    FF_ELE
                    *
                    ff_scale(t_before),

                    ff_ail
                    *
                    ff_scale(t_before),
                ],
                dtype=np.float64,
            )

            requested_delta = (
                feedback_delta
                +
                ff_delta
            )

            requested_delta = np.clip(
                requested_delta,
                -CYCLIC_AUTHORITY,
                +CYCLIC_AUTHORITY,
            )

            raw_teacher_norm = (
                requested_delta
                /
                CYCLIC_AUTHORITY
            )

            raw_teacher_norm = np.clip(
                raw_teacher_norm,
                -1.0,
                +1.0,
            )

            # ------------------------------------------------
            # V5 ONE-SIDED ROLL GUARD
            #
            # Empirically, positive aileron action produces
            # positive roll-rate in the V4B trajectory.
            #
            # So:
            #   negative roll-rate -> positive damping command
            #   excessive negative roll -> positive correction
            #
            # No correction is added for positive roll-rate or
            # roll values above the floor.
            # ------------------------------------------------

            rate_guard = (
                roll_rate_gain
                *
                max(
                    0.0,
                    -roll_rate,
                )
            )

            roll_floor = np.radians(
                roll_floor_deg
            )

            angle_guard = (
                roll_angle_gain
                *
                max(
                    0.0,
                    roll_floor
                    -
                    roll,
                )
            )

            guard = (
                rate_guard
                +
                angle_guard
            )

            # Limit guard so it stabilizes rather than
            # completely replacing the outer-loop controller.
            guard = float(
                np.clip(
                    guard,
                    0.0,
                    0.75,
                )
            )

            if guard > 1e-6:
                guard_active_steps += 1

            max_guard = max(
                max_guard,
                guard,
            )

            # Add guard ONLY to aileron normalized action.
            raw_teacher_norm[1] += guard

            raw_teacher_norm = np.clip(
                raw_teacher_norm,
                -1.0,
                +1.0,
            )

            # Smooth teacher.
            teacher_norm = (
                (1.0 - XY_ALPHA)
                *
                teacher_prev
                +
                XY_ALPHA
                *
                raw_teacher_norm
            )

            teacher_prev = (
                teacher_norm.copy()
            )

            teacher_norm = np.clip(
                teacher_norm,
                -1.0,
                +1.0,
            )

            action[1] = float(
                gate
                *
                teacher_norm[0]
                +
                (1.0 - gate)
                *
                ppo_action[1]
            )

            action[2] = float(
                gate
                *
                teacher_norm[1]
                +
                (1.0 - gate)
                *
                ppo_action[2]
            )

        # Collective and rudder stay PPO.
        action = np.clip(
            action,
            -1.0,
            +1.0,
        ).astype(np.float32)

        # ----------------------------------------------------
        # STEP
        # ----------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        last_info = info

        raw_after = raw_fdm_state(env)

        t = (step + 1) * dt

        altitude_now = raw_after["altitude"]

        north_now = float(
            info.get("north", 0.0)
        )
        east_now = float(
            info.get("east", 0.0)
        )
        vn_now = float(
            info.get("vn", 0.0)
        )
        ve_now = float(
            info.get("ve", 0.0)
        )

        drift = float(
            info.get("drift", 0.0)
        )
        path = float(
            info.get("path", 0.0)
        )

        hspeed = float(
            np.hypot(
                vn_now,
                ve_now,
            )
        )

        # ----------------------------------------------------
        # FIRST 100 FT METRICS
        # ----------------------------------------------------

        if altitude_now <= 100.0:
            early_max = max(
                early_max,
                drift,
            )

            max_n = max(
                max_n,
                abs(north_now),
            )

            max_e = max(
                max_e,
                abs(east_now),
            )

            max_pitch100 = max(
                max_pitch100,
                abs(
                    raw_after["pitch"]
                ),
            )

            max_roll100 = max(
                max_roll100,
                abs(
                    raw_after["roll"]
                ),
            )

            most_negative_roll = min(
                most_negative_roll,
                raw_after["roll"],
            )

            max_abs_p100 = max(
                max_abs_p100,
                abs(
                    raw_after["p"]
                ),
            )

            sn = velocity_sign(
                vn_now
            )
            se = velocity_sign(
                ve_now
            )

            if sn != 0:
                if (
                    last_sign_n != 0
                    and
                    sn != last_sign_n
                ):
                    reversals_n += 1

                last_sign_n = sn

            if se != 0:
                if (
                    last_sign_e != 0
                    and
                    se != last_sign_e
                ):
                    reversals_e += 1

                last_sign_e = se

        if (
            path100 is None
            and
            altitude_now >= 100.0
        ):
            path100 = path
            drift100 = drift
            hs100 = hspeed
            time100 = t

        # ----------------------------------------------------
        # DETAILED OUTPUT
        # ----------------------------------------------------

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"ALT={altitude_now:7.2f} | "
                f"N={north_now:+6.2f} | "
                f"E={east_now:+6.2f} | "
                f"DRIFT={drift:5.2f} | "
                f"ROLL={np.degrees(raw_after['roll']):+7.3f}deg | "
                f"P={np.degrees(raw_after['p']):+7.3f}deg/s | "
                f"GUARD={guard:+5.3f} | "
                f"RATE={rate_guard:+5.3f} | "
                f"ANGLE={angle_guard:+5.3f} | "
                f"T_AIL={teacher_norm[1]:+6.3f} | "
                f"OUT_AIL={action[2]:+6.3f} | "
                f"FDM_AIL={raw_after['aileron']:+.5f}"
            )

            next_print += 0.5

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------

        if (
            terminated
            and
            not bool(
                info.get(
                    "success",
                    False,
                )
            )
        ):
            failed = True
            break

        if drift > 15.0:
            failed = True
            break

        if altitude_now > 350.0:
            failed = True
            break

    env.close()

    if path100 is None:
        path100 = float(
            last_info.get(
                "path",
                999.0,
            )
        )

        drift100 = float(
            last_info.get(
                "drift",
                999.0,
            )
        )

        hs100 = float(
            np.hypot(
                float(
                    last_info.get(
                        "vn",
                        999.0,
                    )
                ),
                float(
                    last_info.get(
                        "ve",
                        999.0,
                    )
                ),
            )
        )

        time100 = total_time
        failed = True

    s_index = max(
        0.0,
        path100 - drift100,
    )

    result = {
        "ff_ail":
            float(ff_ail),

        "roll_rate_gain":
            float(roll_rate_gain),

        "roll_floor_deg":
            float(roll_floor_deg),

        "roll_angle_gain":
            float(roll_angle_gain),

        "failed":
            bool(failed),

        "early_max":
            float(early_max),

        "path100":
            float(path100),

        "drift100":
            float(drift100),

        "s_index":
            float(s_index),

        "hs100":
            float(hs100),

        "time100":
            float(time100),

        "max_n":
            float(max_n),

        "max_e":
            float(max_e),

        "reversals_n":
            int(reversals_n),

        "reversals_e":
            int(reversals_e),

        "reversals":
            int(
                reversals_n
                +
                reversals_e
            ),

        "max_pitch_deg":
            float(
                np.degrees(
                    max_pitch100
                )
            ),

        "max_roll_deg":
            float(
                np.degrees(
                    max_roll100
                )
            ),

        "max_negative_roll_deg":
            float(
                abs(
                    np.degrees(
                        most_negative_roll
                    )
                )
            ),

        "max_roll_rate_deg_s":
            float(
                np.degrees(
                    max_abs_p100
                )
            ),

        "max_guard":
            float(max_guard),

        "guard_fraction":
            float(
                guard_active_steps
                /
                max(
                    teacher_steps,
                    1,
                )
            ),

        "total_max":
            float(
                last_info.get(
                    "max_drift",
                    999.0,
                )
            ),

        "final_drift":
            float(
                last_info.get(
                    "drift",
                    999.0,
                )
            ),

        "final_path":
            float(
                last_info.get(
                    "path",
                    999.0,
                )
            ),

        "final_alt":
            float(
                last_info.get(
                    "altitude",
                    999.0,
                )
            ),

        "final_vs":
            float(
                last_info.get(
                    "vertical_speed",
                    999.0,
                )
            ),
    }

    result["score"] = (
        score_result(result)
        if not failed
        else 1e9
    )

    return result


# ============================================================
# HEADER
# ============================================================

print("=" * 150)
print(
    "STAGE 1 EARLY XY V5 — ROLL-OVERSHOOT GUARD"
)
print("=" * 150)

print("\nModel:")
print(MODEL_PATH)

print(
    "\nArchitecture:"
)
print(
    "  Collective : PPO"
)
print(
    "  Rudder     : PPO"
)
print(
    "  Elevator   : V3 XY teacher"
)
print(
    "  Aileron    : V3 XY teacher + one-sided roll guard"
)
print(
    "  >140 ft    : PPO only"
)


# ============================================================
# PASS 1
#
# Find correct initial aileron feed-forward direction and
# useful roll-rate damping.
# No roll-angle guard yet.
# ============================================================

print("\n" + "=" * 150)
print(
    "PASS 1 — AILERON FEED-FORWARD SIGN + ROLL-RATE DAMPING"
)
print("=" * 150)

pass1 = []

total1 = (
    len(FF_AIL_VALUES)
    *
    len(ROLL_RATE_GAIN_VALUES)
)

counter = 0

for ff_ail in FF_AIL_VALUES:
    for rate_gain in ROLL_RATE_GAIN_VALUES:
        counter += 1

        r = run_case(
            ff_ail=ff_ail,
            roll_rate_gain=rate_gain,
            roll_floor_deg=-3.0,
            roll_angle_gain=0.0,
            total_time=24.0,
            detailed=False,
        )

        pass1.append(r)

        print(
            f"{counter:2d}/{total1} | "
            f"FFA={ff_ail:+.3f} | "
            f"KR={rate_gain:.1f} | "
            f"MAX100={r['early_max']:.3f} | "
            f"PATH100={r['path100']:.3f} | "
            f"DRIFT100={r['drift100']:.3f} | "
            f"S={r['s_index']:.3f} | "
            f"MAXE={r['max_e']:.3f} | "
            f"ROLLMIN=-{r['max_negative_roll_deg']:.3f}deg | "
            f"REV={r['reversals']}"
        )


pass1.sort(
    key=lambda x: x["score"]
)

print("\nTOP 8 PASS-1:")

for i, r in enumerate(
    pass1[:8],
    start=1,
):
    print(
        f"{i:2d}. "
        f"FFA={r['ff_ail']:+.3f} | "
        f"KR={r['roll_rate_gain']:.1f} | "
        f"MAX100={r['early_max']:.3f} | "
        f"PATH100={r['path100']:.3f} | "
        f"DRIFT100={r['drift100']:.3f} | "
        f"S={r['s_index']:.3f} | "
        f"MAXE={r['max_e']:.3f} | "
        f"ROLLMIN=-{r['max_negative_roll_deg']:.3f}deg"
    )


# ============================================================
# PASS 2
#
# Use top 3 Pass-1 controllers.
# Add one-sided roll-angle floor.
# ============================================================

print("\n" + "=" * 150)
print(
    "PASS 2 — ROLL FLOOR / OVERSHOOT GUARD"
)
print("=" * 150)

seeds = pass1[:3]
pass2 = []

total2 = (
    len(seeds)
    *
    len(ROLL_FLOOR_DEG_VALUES)
    *
    len(ROLL_ANGLE_GAIN_VALUES)
)

counter = 0

for seed_i, seed in enumerate(
    seeds,
    start=1,
):
    for floor_deg in ROLL_FLOOR_DEG_VALUES:
        for angle_gain in ROLL_ANGLE_GAIN_VALUES:
            counter += 1

            r = run_case(
                ff_ail=seed["ff_ail"],
                roll_rate_gain=
                    seed["roll_rate_gain"],
                roll_floor_deg=floor_deg,
                roll_angle_gain=angle_gain,
                total_time=24.0,
                detailed=False,
            )

            r["seed"] = seed_i

            pass2.append(r)

            if (
                counter % 8 == 0
                or
                r["early_max"] <= 2.0
            ):
                print(
                    f"{counter:2d}/{total2} | "
                    f"SEED={seed_i} | "
                    f"FFA={r['ff_ail']:+.3f} | "
                    f"KR={r['roll_rate_gain']:.1f} | "
                    f"FLOOR={floor_deg:+.1f}deg | "
                    f"KA={angle_gain:.1f} | "
                    f"MAX100={r['early_max']:.3f} | "
                    f"PATH100={r['path100']:.3f} | "
                    f"DRIFT100={r['drift100']:.3f} | "
                    f"MAXE={r['max_e']:.3f} | "
                    f"ROLLMIN=-{r['max_negative_roll_deg']:.3f}deg"
                )


pass2.sort(
    key=lambda x: x["score"]
)


# ============================================================
# TOP 20
# ============================================================

print("\n" + "=" * 150)
print(
    "TOP 20 V5 CANDIDATES"
)
print("=" * 150)

for i, r in enumerate(
    pass2[:20],
    start=1,
):
    print(
        f"{i:2d}. "
        f"FFA={r['ff_ail']:+.3f} | "
        f"KR={r['roll_rate_gain']:.1f} | "
        f"FLOOR={r['roll_floor_deg']:+.1f}deg | "
        f"KA={r['roll_angle_gain']:.1f} | "
        f"MAX100={r['early_max']:.3f} | "
        f"PATH100={r['path100']:.3f} | "
        f"DRIFT100={r['drift100']:.3f} | "
        f"S={r['s_index']:.3f} | "
        f"HS100={r['hs100']:.3f} | "
        f"T100={r['time100']:.2f}s | "
        f"MAXN={r['max_n']:.3f} | "
        f"MAXE={r['max_e']:.3f} | "
        f"ROLLMIN=-{r['max_negative_roll_deg']:.3f}deg | "
        f"REV={r['reversals']}"
    )


# ============================================================
# FULL 120 SECOND VALIDATION
# ============================================================

best = pass2[0]

print("\n" + "=" * 150)
print(
    "BEST V5 — FULL 120 SECOND VALIDATION"
)
print("=" * 150)

print(
    f"FF AIL          = {best['ff_ail']:+.3f}"
)
print(
    f"ROLL RATE GAIN  = {best['roll_rate_gain']:.1f}"
)
print(
    f"ROLL FLOOR      = {best['roll_floor_deg']:+.1f} deg"
)
print(
    f"ROLL ANGLE GAIN = {best['roll_angle_gain']:.1f}"
)
print()

full = run_case(
    ff_ail=best["ff_ail"],
    roll_rate_gain=
        best["roll_rate_gain"],
    roll_floor_deg=
        best["roll_floor_deg"],
    roll_angle_gain=
        best["roll_angle_gain"],
    total_time=120.0,
    detailed=True,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 150)
print(
    "V5 BEST FULL RESULT"
)
print("=" * 150)

print(
    f"EARLY MAX DRIFT (<100 ft) : "
    f"{full['early_max']:.3f} ft"
)

print(
    f"XY PATH AT 100 ft         : "
    f"{full['path100']:.3f} ft"
)

print(
    f"DRIFT AT 100 ft           : "
    f"{full['drift100']:.3f} ft"
)

print(
    f"S-INDEX                    : "
    f"{full['s_index']:.3f} ft"
)

print(
    f"HORIZONTAL SPEED @100 ft  : "
    f"{full['hs100']:.3f} ft/s"
)

print(
    f"TIME TO 100 ft             : "
    f"{full['time100']:.3f} s"
)

print(
    f"MAX |NORTH| <100 ft       : "
    f"{full['max_n']:.3f} ft"
)

print(
    f"MAX |EAST| <100 ft        : "
    f"{full['max_e']:.3f} ft"
)

print(
    f"DIRECTION REVERSALS        : "
    f"{full['reversals']} "
    f"(N={full['reversals_n']}, "
    f"E={full['reversals_e']})"
)

print()
print(
    "TRUE JSBSIM ATTITUDE (<100 ft)"
)

print(
    f"MAX |PITCH|                : "
    f"{full['max_pitch_deg']:.4f} deg"
)

print(
    f"MAX |ROLL|                 : "
    f"{full['max_roll_deg']:.4f} deg"
)

print(
    f"MOST NEGATIVE ROLL         : "
    f"-{full['max_negative_roll_deg']:.4f} deg"
)

print(
    f"MAX |ROLL RATE|            : "
    f"{full['max_roll_rate_deg_s']:.4f} deg/s"
)

print(
    f"MAX ROLL-GUARD COMMAND     : "
    f"{full['max_guard']:.4f}"
)

print(
    f"ROLL-GUARD ACTIVE FRACTION : "
    f"{100.0*full['guard_fraction']:.2f}%"
)

print()
print(
    f"TOTAL MAX DRIFT            : "
    f"{full['total_max']:.3f} ft"
)

print(
    f"FINAL DRIFT                : "
    f"{full['final_drift']:.3f} ft"
)

print(
    f"FINAL XY PATH              : "
    f"{full['final_path']:.3f} ft"
)

print(
    f"FINAL ALTITUDE             : "
    f"{full['final_alt']:.3f} ft"
)

print(
    f"FINAL VS                   : "
    f"{full['final_vs']:.3f} ft/s"
)


presentation_quality = bool(
    not full["failed"]
    and full["early_max"] <= 2.0
    and full["path100"] <= 5.0
    and full["drift100"] <= 1.5
    and full["hs100"] <= 0.75
    and full["time100"] <= 22.0
    and full["total_max"] <= 8.0
    and 295.0 <= full["final_alt"] <= 305.0
    and abs(full["final_vs"]) <= 0.75
)

print()

print(
    "EARLY TAKEOFF PRESENTATION QUALITY :",
    presentation_quality,
)


# ============================================================
# SAVE BEST CONFIG
# ============================================================

OUTPUT_DIR = (
    "results_stage1_early_xy_v5"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

config = {
    "model_path":
        MODEL_PATH,

    "teacher": {
        "xy_kp":
            XY_KP,

        "xy_kd":
            XY_KD,

        "xy_alpha":
            XY_ALPHA,

        "ff_ele":
            FF_ELE,

        "ff_ail":
            full["ff_ail"],

        "roll_rate_gain":
            full["roll_rate_gain"],

        "roll_floor_deg":
            full["roll_floor_deg"],

        "roll_angle_gain":
            full["roll_angle_gain"],

        "teacher_full_alt":
            TEACHER_FULL_ALT,

        "teacher_off_alt":
            TEACHER_OFF_ALT,
    },

    "metrics": {
        "early_max_drift_ft":
            full["early_max"],

        "path_at_100_ft":
            full["path100"],

        "drift_at_100_ft":
            full["drift100"],

        "s_index_ft":
            full["s_index"],

        "max_east_ft":
            full["max_e"],

        "most_negative_roll_deg":
            full["max_negative_roll_deg"],

        "final_altitude_ft":
            full["final_alt"],

        "final_vs_ft_s":
            full["final_vs"],
    },

    "presentation_quality":
        presentation_quality,
}

config_path = os.path.join(
    OUTPUT_DIR,
    "best_v5_teacher_config.json",
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )

print(
    f"\nSaved: {config_path}"
)

if presentation_quality:
    print(
        "\nSUCCESS."
    )
    print(
        "Next step: distill the corrected early cyclic "
        "teacher into PPO elevator/aileron outputs."
    )
else:
    print(
        "\nStill not presentation-clean."
    )
    print(
        "Inspect whether EAST drift fell strongly while NORTH "
        "became dominant. If so, V6 should target pitch/north only."
    )

print("=" * 150)
