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
# IDENTIFIED XY CONTROL EFFECTIVENESS
# ============================================================

B_INV = np.array(
    [
        [-0.20338265, -0.00859620],
        [ 0.01952073, -0.14631468],
    ],
    dtype=np.float64,
)

ACTION_CYCLIC_AUTHORITY = 0.026


# ============================================================
# BEST FEEDBACK CONTROLLER FROM V1
# ============================================================

KP = 0.010
KD = 0.46
A_MAX = 0.18
DELTA_MAX = 0.026


# ============================================================
# ALTITUDE GATE
# ============================================================

FULL_CORRECTION_ALT = 100.0
FADE_OUT_ALT = 140.0


def altitude_gate(altitude):
    if altitude <= FULL_CORRECTION_ALT:
        return 1.0

    if altitude >= FADE_OUT_ALT:
        return 0.0

    return float(
        (FADE_OUT_ALT - altitude)
        /
        (FADE_OUT_ALT - FULL_CORRECTION_ALT)
    )


# ============================================================
# FEED-FORWARD PROFILE
#
# Purpose:
# The V1 controller reacts only AFTER horizontal motion starts.
# This feed-forward pulse begins immediately, before the drift
# develops, then fades away.
# ============================================================

FF_ELE_VALUES = [
    -0.026,
    -0.020,
    -0.014,
    -0.008,
    0.000,
]

FF_AIL_VALUES = [
    -0.026,
    -0.020,
    -0.014,
    -0.008,
    0.000,
]

FF_HOLD_END_VALUES = [
    2.5,
    4.0,
    5.5,
]

# How long after hold_end to fade to zero.
FF_FADE_TIME = 2.0

# Smooth total cyclic residual. Higher = faster response.
SMOOTH_ALPHA_VALUES = [
    0.35,
    0.65,
]


def feedforward_scale(t, hold_end):
    # Gentle ramp-in to avoid an instantaneous cyclic step.
    ramp_time = 0.35

    if t <= 0.0:
        return 0.0

    if t < ramp_time:
        return float(t / ramp_time)

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
# RUN ONE CASE
# ============================================================

def run_case(
    ff_ele,
    ff_ail,
    ff_hold_end,
    smooth_alpha,
    total_time=22.0,
    detailed=False,
):
    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env.reset()

    dt = float(env.dt)
    max_steps = int(total_time / dt)

    prev_delta = np.zeros(
        2,
        dtype=np.float64,
    )

    early_max_drift = 0.0
    path100 = None
    drift100 = None
    hs100 = None
    time100 = None

    # Track trajectory shape more explicitly.
    max_abs_north_100 = 0.0
    max_abs_east_100 = 0.0

    next_print = 0.0
    failed = False
    last_info = info

    for step in range(max_steps):
        t_before = step * dt

        # ----------------------------------------------------
        # FINAL PPO
        # ----------------------------------------------------
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        # ----------------------------------------------------
        # CURRENT NAV STATE FROM INFO
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

        gate = altitude_gate(altitude)

        # ----------------------------------------------------
        # FEEDBACK PART
        # ----------------------------------------------------
        desired_accel = np.array(
            [
                -KP * north - KD * vn,
                -KP * east - KD * ve,
            ],
            dtype=np.float64,
        )

        desired_accel = np.clip(
            desired_accel,
            -A_MAX,
            +A_MAX,
        )

        feedback_delta = B_INV @ desired_accel

        # ----------------------------------------------------
        # ANTICIPATORY FEED-FORWARD PART
        # ----------------------------------------------------
        ff_scale = feedforward_scale(
            t_before,
            ff_hold_end,
        )

        ff_delta = np.array(
            [
                ff_ele * ff_scale,
                ff_ail * ff_scale,
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # TOTAL PHYSICAL CYCLIC CORRECTION
        # ----------------------------------------------------
        requested_delta = (
            feedback_delta
            +
            ff_delta
        )

        requested_delta = np.clip(
            requested_delta,
            -DELTA_MAX,
            +DELTA_MAX,
        )

        # Low-pass smoothing prevents hard sign-flip chatter.
        smoothed_delta = (
            (1.0 - smooth_alpha) * prev_delta
            +
            smooth_alpha * requested_delta
        )

        prev_delta = smoothed_delta.copy()

        # Altitude fade-out.
        applied_delta = (
            smoothed_delta
            *
            gate
        )

        delta_ele = float(applied_delta[0])
        delta_ail = float(applied_delta[1])

        # ----------------------------------------------------
        # PHYSICAL RESIDUAL -> NORMALIZED ACTION RESIDUAL
        # ----------------------------------------------------
        action[1] += (
            delta_ele
            /
            ACTION_CYCLIC_AUTHORITY
        )

        action[2] += (
            delta_ail
            /
            ACTION_CYCLIC_AUTHORITY
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
        # EARLY METRICS
        # ----------------------------------------------------
        if altitude_now <= 100.0:
            early_max_drift = max(
                early_max_drift,
                drift,
            )

            max_abs_north_100 = max(
                max_abs_north_100,
                abs(north_now),
            )

            max_abs_east_100 = max(
                max_abs_east_100,
                abs(east_now),
            )

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
                f"FF={ff_scale:4.2f} | "
                f"dELE={delta_ele:+.5f} | "
                f"dAIL={delta_ail:+.5f}"
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
            last_info.get("path", 999.0)
        )

        drift100 = float(
            last_info.get("drift", 999.0)
        )

        hs100 = float(
            np.hypot(
                float(
                    last_info.get("vn", 999.0)
                ),
                float(
                    last_info.get("ve", 999.0)
                ),
            )
        )

        time100 = total_time
        failed = True

    s_index = max(
        0.0,
        path100 - drift100,
    )

    total_max_drift = float(
        last_info.get(
            "max_drift",
            999.0,
        )
    )

    final_drift = float(
        last_info.get(
            "drift",
            999.0,
        )
    )

    final_path = float(
        last_info.get(
            "path",
            999.0,
        )
    )

    final_alt = float(
        last_info.get(
            "altitude",
            999.0,
        )
    )

    # Strongly prioritize the visible shape.
    score = (
        350.0 * early_max_drift
        + 120.0 * path100
        + 140.0 * s_index
        + 80.0 * hs100
        + 25.0 * max_abs_north_100
        + 25.0 * max_abs_east_100
    )

    if failed:
        score += 1e9

    return {
        "ff_ele": ff_ele,
        "ff_ail": ff_ail,
        "ff_hold_end": ff_hold_end,
        "alpha": smooth_alpha,
        "score": score,
        "failed": failed,
        "early_max": early_max_drift,
        "path100": path100,
        "drift100": drift100,
        "s_index": s_index,
        "hs100": hs100,
        "time100": time100,
        "max_n": max_abs_north_100,
        "max_e": max_abs_east_100,
        "total_max": total_max_drift,
        "final_drift": final_drift,
        "final_path": final_path,
        "final_alt": final_alt,
    }


# ============================================================
# SEARCH
# ============================================================

print("=" * 140)
print("STAGE 1 EARLY XY V2 — FEED-FORWARD + DAMPED FEEDBACK")
print("=" * 140)

print("\nFinal PPO:")
print(MODEL_PATH)

print(
    "\nFixed feedback gains from V1:"
)
print(
    f"KP={KP}, KD={KD}, A_MAX={A_MAX}, DELTA_MAX={DELTA_MAX}"
)

total_candidates = (
    len(FF_ELE_VALUES)
    *
    len(FF_AIL_VALUES)
    *
    len(FF_HOLD_END_VALUES)
    *
    len(SMOOTH_ALPHA_VALUES)
)

print(
    f"\nSearching {total_candidates} feed-forward candidates..."
)

results = []
counter = 0

for ff_ele in FF_ELE_VALUES:
    for ff_ail in FF_AIL_VALUES:
        for hold_end in FF_HOLD_END_VALUES:
            for alpha in SMOOTH_ALPHA_VALUES:
                counter += 1

                r = run_case(
                    ff_ele=ff_ele,
                    ff_ail=ff_ail,
                    ff_hold_end=hold_end,
                    smooth_alpha=alpha,
                    total_time=22.0,
                    detailed=False,
                )

                results.append(r)

                if (
                    counter % 15 == 0
                    or r["early_max"] <= 2.0
                ):
                    print(
                        f"{counter:3d}/{total_candidates} | "
                        f"FFE={ff_ele:+.3f} "
                        f"FFA={ff_ail:+.3f} "
                        f"HOLD={hold_end:.1f} "
                        f"A={alpha:.2f} | "
                        f"MAX100={r['early_max']:.3f} | "
                        f"PATH100={r['path100']:.3f} | "
                        f"DRIFT100={r['drift100']:.3f} | "
                        f"S={r['s_index']:.3f} | "
                        f"HS100={r['hs100']:.3f}"
                    )


results.sort(
    key=lambda x: x["score"]
)


# ============================================================
# TOP RESULTS
# ============================================================

print("\n")
print("=" * 140)
print("TOP 20 V2 CANDIDATES")
print("=" * 140)

for i, r in enumerate(
    results[:20],
    start=1,
):
    print(
        f"{i:2d}. "
        f"FFE={r['ff_ele']:+.3f} | "
        f"FFA={r['ff_ail']:+.3f} | "
        f"HOLD={r['ff_hold_end']:.1f} | "
        f"ALPHA={r['alpha']:.2f} | "
        f"MAX100={r['early_max']:.3f} | "
        f"PATH100={r['path100']:.3f} | "
        f"DRIFT100={r['drift100']:.3f} | "
        f"S={r['s_index']:.3f} | "
        f"HS100={r['hs100']:.3f} | "
        f"MAXN={r['max_n']:.3f} | "
        f"MAXE={r['max_e']:.3f}"
    )


# ============================================================
# FULL 120 s TEST OF BEST
# ============================================================

best = results[0]

print("\n")
print("=" * 140)
print("BEST CANDIDATE — 120 SECOND VALIDATION")
print("=" * 140)

print(
    f"FF ELE    = {best['ff_ele']:+.3f}"
)
print(
    f"FF AIL    = {best['ff_ail']:+.3f}"
)
print(
    f"HOLD END  = {best['ff_hold_end']:.1f} s"
)
print(
    f"ALPHA     = {best['alpha']:.2f}"
)
print()


full = run_case(
    ff_ele=best["ff_ele"],
    ff_ail=best["ff_ail"],
    ff_hold_end=best["ff_hold_end"],
    smooth_alpha=best["alpha"],
    total_time=120.0,
    detailed=True,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 140)
print("V2 BEST FULL RESULT")
print("=" * 140)

print(
    "EARLY MAX DRIFT (<100 ft) :",
    f"{full['early_max']:.3f} ft",
)

print(
    "XY PATH AT 100 ft         :",
    f"{full['path100']:.3f} ft",
)

print(
    "DRIFT AT 100 ft           :",
    f"{full['drift100']:.3f} ft",
)

print(
    "S-INDEX                    :",
    f"{full['s_index']:.3f} ft",
)

print(
    "HORIZONTAL SPEED @100 ft  :",
    f"{full['hs100']:.3f} ft/s",
)

print(
    "MAX |NORTH| <100 ft       :",
    f"{full['max_n']:.3f} ft",
)

print(
    "MAX |EAST| <100 ft        :",
    f"{full['max_e']:.3f} ft",
)

print(
    "TOTAL MAX DRIFT            :",
    f"{full['total_max']:.3f} ft",
)

print(
    "FINAL DRIFT                :",
    f"{full['final_drift']:.3f} ft",
)

print(
    "FINAL XY PATH              :",
    f"{full['final_path']:.3f} ft",
)

print(
    "FINAL ALTITUDE             :",
    f"{full['final_alt']:.3f} ft",
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

if presentation_quality:
    print(
        "\nSUCCESS."
    )
    print(
        "This teacher is clean enough to distill into "
        "the PPO cyclic outputs."
    )
else:
    print(
        "\nStill not clean enough."
    )
    print(
        "Do NOT distill yet."
    )
    print(
        "Next step would be a time-varying / phase-based "
        "early controller rather than more gain."
    )

print("=" * 140)
