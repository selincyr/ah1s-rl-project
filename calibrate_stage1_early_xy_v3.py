import json
import os
import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    "models_stage1_final_distilled/"
    "AH1S_STAGE1_FINAL_DISTILLED.zip"
)

model = PPO.load(MODEL_PATH)


# ============================================================
# IDENTIFIED HORIZONTAL CONTROL EFFECTIVENESS
# ============================================================

B_INV = np.array(
    [
        [-0.20338265, -0.00859620],
        [ 0.01952073, -0.14631468],
    ],
    dtype=np.float64,
)

# Physical cyclic delta corresponding approximately to a
# normalized action magnitude of 1.0 in the Stage-1 env.
CYCLIC_AUTHORITY = 0.026


# ============================================================
# TEACHER AUTHORITY WINDOW
#
# Below 100 ft:
#   teacher owns elevator + aileron completely.
#
# 100 -> 140 ft:
#   smooth blend from teacher back to PPO.
#
# Above 140 ft:
#   PPO owns elevator + aileron completely.
# ============================================================

TEACHER_FULL_ALT = 100.0
TEACHER_OFF_ALT = 140.0


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


# ============================================================
# FEED-FORWARD
#
# Unlike V2, feed-forward can start immediately. No 0.35 s
# ramp-in. The goal is to counter the repeatable takeoff
# transient BEFORE a visible drift develops.
# ============================================================

FF_FADE_TIME = 1.5


def ff_scale(t, hold_end):
    if t <= hold_end:
        return 1.0

    fade_end = hold_end + FF_FADE_TIME

    if t >= fade_end:
        return 0.0

    return float(
        (fade_end - t)
        /
        FF_FADE_TIME
    )


# ============================================================
# METRICS
# ============================================================

def velocity_sign(value, threshold=0.08):
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def evaluate_metrics(
    early_max,
    path100,
    drift100,
    hs100,
    max_n,
    max_e,
    reversals,
    failed,
):
    s_index = max(
        0.0,
        path100 - drift100,
    )

    score = (
        500.0 * early_max
        + 160.0 * path100
        + 180.0 * s_index
        + 100.0 * hs100
        + 35.0 * max_n
        + 35.0 * max_e
        + 40.0 * reversals
    )

    if failed:
        score += 1e9

    return score, s_index


# ============================================================
# RUN ONE FLIGHT
# ============================================================

def run_case(
    kp=0.0,
    kd=0.0,
    alpha=0.70,
    ff_ele=0.0,
    ff_ail=0.0,
    ff_hold=0.0,
    total_time=22.0,
    use_teacher=True,
    detailed=False,
):
    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env.reset()

    dt = float(env.dt)
    max_steps = int(total_time / dt)

    # Smoothed DIRECT teacher actions for:
    # [elevator_action, aileron_action]
    teacher_action_prev = np.zeros(
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

    last_sign_n = 0
    last_sign_e = 0
    reversals_n = 0
    reversals_e = 0

    last_info = info
    failed = False
    next_print = 0.0

    # For diagnostics.
    saturation_count = 0
    teacher_steps = 0

    for step in range(max_steps):
        t_before = step * dt

        # ----------------------------------------------------
        # PPO POLICY
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

        # ----------------------------------------------------
        # CURRENT STATE
        # ----------------------------------------------------
        altitude = float(
            info.get("altitude", 0.0)
        )

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

        gate = (
            teacher_gate(altitude)
            if use_teacher
            else 0.0
        )

        teacher_norm = np.zeros(
            2,
            dtype=np.float64,
        )

        feedback_delta = np.zeros(
            2,
            dtype=np.float64,
        )

        ff_delta = np.zeros(
            2,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # DIRECT CYCLIC TEACHER
        # ----------------------------------------------------
        if use_teacher and gate > 0.0:
            teacher_steps += 1

            # Desired horizontal acceleration.
            desired_accel = np.array(
                [
                    -kp * north - kd * vn,
                    -kp * east  - kd * ve,
                ],
                dtype=np.float64,
            )

            # Keep acceleration request bounded.
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

            scale = ff_scale(
                t_before,
                ff_hold,
            )

            ff_delta = np.array(
                [
                    ff_ele * scale,
                    ff_ail * scale,
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

            if (
                abs(raw_teacher_norm[0]) > 0.985
                or
                abs(raw_teacher_norm[1]) > 0.985
            ):
                saturation_count += 1

            # Smooth the teacher command itself.
            teacher_action = (
                (1.0 - alpha)
                * teacher_action_prev
                +
                alpha
                * raw_teacher_norm
            )

            teacher_action_prev = (
                teacher_action.copy()
            )

            teacher_norm = np.clip(
                teacher_action,
                -1.0,
                +1.0,
            )

            # ------------------------------------------------
            # KEY DIFFERENCE FROM V1/V2:
            #
            # NOT:
            #   PPO cyclic + teacher residual
            #
            # BUT:
            #   direct teacher cyclic below 100 ft
            #   smooth teacher -> PPO blend from 100 to 140 ft
            # ------------------------------------------------
            action[1] = float(
                gate * teacher_norm[0]
                +
                (1.0 - gate)
                * ppo_action[1]
            )

            action[2] = float(
                gate * teacher_norm[1]
                +
                (1.0 - gate)
                * ppo_action[2]
            )

            action[1] = float(
                np.clip(
                    action[1],
                    -1.0,
                    +1.0,
                )
            )

            action[2] = float(
                np.clip(
                    action[2],
                    -1.0,
                    +1.0,
                )
            )

        # Collective and rudder remain PPO outputs.
        # action[0] = PPO collective
        # action[3] = PPO rudder

        # ----------------------------------------------------
        # STEP
        # ----------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        last_info = info

        t = (step + 1) * dt

        altitude_now = float(
            info.get("altitude", 0.0)
        )

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

            sn = velocity_sign(vn_now)
            se = velocity_sign(ve_now)

            if sn != 0:
                if (
                    last_sign_n != 0
                    and sn != last_sign_n
                ):
                    reversals_n += 1

                last_sign_n = sn

            if se != 0:
                if (
                    last_sign_e != 0
                    and se != last_sign_e
                ):
                    reversals_e += 1

                last_sign_e = se

        if (
            path100 is None
            and altitude_now >= 100.0
        ):
            path100 = path
            drift100 = drift
            hs100 = hspeed
            time100 = t

        # ----------------------------------------------------
        # DETAILED LOG
        # ----------------------------------------------------
        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"ALT={altitude_now:7.2f} | "
                f"N={north_now:+6.2f} | "
                f"E={east_now:+6.2f} | "
                f"DRIFT={drift:5.2f} | "
                f"PATH={path:5.2f} | "
                f"VN={vn_now:+6.3f} | "
                f"VE={ve_now:+6.3f} | "
                f"GATE={gate:4.2f} | "
                f"PPOcyc=({ppo_action[1]:+5.2f},"
                f"{ppo_action[2]:+5.2f}) | "
                f"Tcyc=({teacher_norm[0]:+5.2f},"
                f"{teacher_norm[1]:+5.2f}) | "
                f"OUT=({action[1]:+5.2f},"
                f"{action[2]:+5.2f})"
            )

            next_print += 1.0

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------
        if (
            terminated
            and not bool(
                info.get("success", False)
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

    reversals = (
        reversals_n
        +
        reversals_e
    )

    score, s_index = evaluate_metrics(
        early_max=early_max,
        path100=path100,
        drift100=drift100,
        hs100=hs100,
        max_n=max_n,
        max_e=max_e,
        reversals=reversals,
        failed=failed,
    )

    sat_fraction = (
        saturation_count
        /
        max(
            teacher_steps,
            1,
        )
    )

    return {
        "kp": float(kp),
        "kd": float(kd),
        "alpha": float(alpha),
        "ff_ele": float(ff_ele),
        "ff_ail": float(ff_ail),
        "ff_hold": float(ff_hold),
        "score": float(score),
        "failed": bool(failed),
        "early_max": float(early_max),
        "path100": float(path100),
        "drift100": float(drift100),
        "s_index": float(s_index),
        "hs100": float(hs100),
        "time100": float(time100),
        "max_n": float(max_n),
        "max_e": float(max_e),
        "reversals_n": int(reversals_n),
        "reversals_e": int(reversals_e),
        "reversals": int(reversals),
        "sat_fraction": float(sat_fraction),
        "total_max": float(
            last_info.get(
                "max_drift",
                999.0,
            )
        ),
        "final_drift": float(
            last_info.get(
                "drift",
                999.0,
            )
        ),
        "final_path": float(
            last_info.get(
                "path",
                999.0,
            )
        ),
        "final_alt": float(
            last_info.get(
                "altitude",
                999.0,
            )
        ),
        "final_vs": float(
            last_info.get(
                "vertical_speed",
                999.0,
            )
        ),
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 150)
print("STAGE 1 EARLY XY V3 — DIRECT PHASE-BASED CYCLIC TEACHER")
print("=" * 150)

print("\nPPO:")
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
    "  Elevator   : direct teacher below 100 ft"
)
print(
    "  Aileron    : direct teacher below 100 ft"
)
print(
    "  100-140 ft : smooth teacher -> PPO cyclic handoff"
)
print(
    "  >140 ft    : PPO only"
)


# ============================================================
# BASELINE
# ============================================================

print("\n")
print("=" * 150)
print("BASELINE — FINAL PPO ONLY")
print("=" * 150)

baseline = run_case(
    total_time=22.0,
    use_teacher=False,
    detailed=False,
)

print(
    f"MAX100={baseline['early_max']:.3f} | "
    f"PATH100={baseline['path100']:.3f} | "
    f"DRIFT100={baseline['drift100']:.3f} | "
    f"S={baseline['s_index']:.3f} | "
    f"HS100={baseline['hs100']:.3f} | "
    f"REV={baseline['reversals']}"
)


# ============================================================
# PASS 1
#
# Search direct feedback controller WITHOUT feed-forward.
# ============================================================

KP_VALUES = [
    0.000,
    0.004,
    0.008,
    0.012,
    0.016,
]

KD_VALUES = [
    0.18,
    0.26,
    0.34,
    0.42,
    0.50,
    0.60,
]

ALPHA_VALUES = [
    0.45,
    0.70,
    0.90,
]


print("\n")
print("=" * 150)
print("PASS 1 — DIRECT FEEDBACK SEARCH")
print("=" * 150)

pass1 = []

total1 = (
    len(KP_VALUES)
    *
    len(KD_VALUES)
    *
    len(ALPHA_VALUES)
)

counter = 0

for kp in KP_VALUES:
    for kd in KD_VALUES:
        for alpha in ALPHA_VALUES:
            counter += 1

            r = run_case(
                kp=kp,
                kd=kd,
                alpha=alpha,
                ff_ele=0.0,
                ff_ail=0.0,
                ff_hold=0.0,
                total_time=22.0,
                use_teacher=True,
                detailed=False,
            )

            pass1.append(r)

            if (
                counter % 10 == 0
                or r["early_max"] <= 2.0
            ):
                print(
                    f"{counter:3d}/{total1} | "
                    f"KP={kp:.3f} "
                    f"KD={kd:.2f} "
                    f"A={alpha:.2f} | "
                    f"MAX100={r['early_max']:.3f} | "
                    f"PATH100={r['path100']:.3f} | "
                    f"DRIFT100={r['drift100']:.3f} | "
                    f"S={r['s_index']:.3f} | "
                    f"HS100={r['hs100']:.3f} | "
                    f"REV={r['reversals']}"
                )


pass1.sort(
    key=lambda x: x["score"]
)

print("\nTOP 10 PASS-1:")

for i, r in enumerate(
    pass1[:10],
    start=1,
):
    print(
        f"{i:2d}. "
        f"KP={r['kp']:.3f} | "
        f"KD={r['kd']:.2f} | "
        f"A={r['alpha']:.2f} | "
        f"MAX100={r['early_max']:.3f} | "
        f"PATH100={r['path100']:.3f} | "
        f"DRIFT100={r['drift100']:.3f} | "
        f"S={r['s_index']:.3f} | "
        f"HS100={r['hs100']:.3f} | "
        f"REV={r['reversals']} | "
        f"SAT={100*r['sat_fraction']:.1f}%"
    )


# ============================================================
# PASS 2
#
# Take the three best direct-feedback controllers and search
# a short anticipatory feed-forward pulse.
# ============================================================

FF_VALUES = [
    -0.026,
    -0.018,
    -0.010,
    0.000,
]

FF_HOLD_VALUES = [
    0.75,
    1.50,
    2.50,
]


print("\n")
print("=" * 150)
print("PASS 2 — SHORT PHASE FEED-FORWARD SEARCH")
print("=" * 150)

pass2 = []

seed_controllers = pass1[:3]

total2 = (
    len(seed_controllers)
    *
    len(FF_VALUES)
    *
    len(FF_VALUES)
    *
    len(FF_HOLD_VALUES)
)

counter = 0

for seed_i, seed in enumerate(
    seed_controllers,
    start=1,
):
    for ff_ele in FF_VALUES:
        for ff_ail in FF_VALUES:
            for ff_hold in FF_HOLD_VALUES:
                counter += 1

                r = run_case(
                    kp=seed["kp"],
                    kd=seed["kd"],
                    alpha=seed["alpha"],
                    ff_ele=ff_ele,
                    ff_ail=ff_ail,
                    ff_hold=ff_hold,
                    total_time=22.0,
                    use_teacher=True,
                    detailed=False,
                )

                r["seed_rank"] = seed_i

                pass2.append(r)

                if (
                    counter % 12 == 0
                    or r["early_max"] <= 2.0
                ):
                    print(
                        f"{counter:3d}/{total2} | "
                        f"SEED={seed_i} "
                        f"FFE={ff_ele:+.3f} "
                        f"FFA={ff_ail:+.3f} "
                        f"HOLD={ff_hold:.2f} | "
                        f"MAX100={r['early_max']:.3f} | "
                        f"PATH100={r['path100']:.3f} | "
                        f"DRIFT100={r['drift100']:.3f} | "
                        f"S={r['s_index']:.3f} | "
                        f"HS100={r['hs100']:.3f} | "
                        f"REV={r['reversals']}"
                    )


pass2.sort(
    key=lambda x: x["score"]
)

print("\n")
print("=" * 150)
print("TOP 20 V3 CANDIDATES")
print("=" * 150)

for i, r in enumerate(
    pass2[:20],
    start=1,
):
    print(
        f"{i:2d}. "
        f"KP={r['kp']:.3f} | "
        f"KD={r['kd']:.2f} | "
        f"A={r['alpha']:.2f} | "
        f"FFE={r['ff_ele']:+.3f} | "
        f"FFA={r['ff_ail']:+.3f} | "
        f"HOLD={r['ff_hold']:.2f} | "
        f"MAX100={r['early_max']:.3f} | "
        f"PATH100={r['path100']:.3f} | "
        f"DRIFT100={r['drift100']:.3f} | "
        f"S={r['s_index']:.3f} | "
        f"HS100={r['hs100']:.3f} | "
        f"MAXN={r['max_n']:.3f} | "
        f"MAXE={r['max_e']:.3f} | "
        f"REV={r['reversals']} | "
        f"SAT={100*r['sat_fraction']:.1f}%"
    )


# ============================================================
# BEST FULL 120 SECOND VALIDATION
# ============================================================

best = pass2[0]

print("\n")
print("=" * 150)
print("BEST V3 — FULL 120 SECOND VALIDATION")
print("=" * 150)

print(
    f"KP       = {best['kp']:.3f}"
)
print(
    f"KD       = {best['kd']:.2f}"
)
print(
    f"ALPHA    = {best['alpha']:.2f}"
)
print(
    f"FF ELE   = {best['ff_ele']:+.3f}"
)
print(
    f"FF AIL   = {best['ff_ail']:+.3f}"
)
print(
    f"FF HOLD  = {best['ff_hold']:.2f} s"
)
print()


full = run_case(
    kp=best["kp"],
    kd=best["kd"],
    alpha=best["alpha"],
    ff_ele=best["ff_ele"],
    ff_ail=best["ff_ail"],
    ff_hold=best["ff_hold"],
    total_time=120.0,
    use_teacher=True,
    detailed=True,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 150)
print("V3 BEST FULL RESULT")
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

print(
    f"CYCLIC SATURATION FRACTION : "
    f"{100*full['sat_fraction']:.1f}%"
)

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
    and full["total_max"] <= 8.0
)

print()
print(
    "EARLY TAKEOFF PRESENTATION QUALITY :",
    presentation_quality,
)


# ============================================================
# SAVE BEST CONFIG FOR DISTILLATION
# ============================================================

os.makedirs(
    "results_stage1_early_xy_v3",
    exist_ok=True,
)

best_config = {
    "model_path": MODEL_PATH,
    "kp": full["kp"],
    "kd": full["kd"],
    "alpha": full["alpha"],
    "ff_ele": full["ff_ele"],
    "ff_ail": full["ff_ail"],
    "ff_hold": full["ff_hold"],
    "teacher_full_alt": TEACHER_FULL_ALT,
    "teacher_off_alt": TEACHER_OFF_ALT,
    "cyclic_authority": CYCLIC_AUTHORITY,
    "metrics": {
        "early_max_drift_ft": full["early_max"],
        "path_at_100_ft": full["path100"],
        "drift_at_100_ft": full["drift100"],
        "s_index_ft": full["s_index"],
        "horizontal_speed_at_100_ft_s": full["hs100"],
        "direction_reversals": full["reversals"],
        "total_max_drift_ft": full["total_max"],
        "final_drift_ft": full["final_drift"],
        "final_altitude_ft": full["final_alt"],
        "final_vs_ft_s": full["final_vs"],
    },
    "presentation_quality": presentation_quality,
}

config_path = (
    "results_stage1_early_xy_v3/"
    "best_teacher_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        best_config,
        f,
        indent=2,
    )

print(
    f"\nSaved best config: {config_path}"
)


if presentation_quality:
    print(
        "\nSUCCESS."
    )
    print(
        "Next step: distill ONLY the early elevator/aileron "
        "teacher behavior into the PPO cyclic outputs."
    )
else:
    print(
        "\nNot ready for distillation yet."
    )
    print(
        "Use the detailed PPOcyc / Tcyc / OUT log to determine "
        "whether remaining error is authority-limited or "
        "requires a collective soft-start."
    )

print("=" * 150)
