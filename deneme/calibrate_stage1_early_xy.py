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
#
# From the JSBSim AH-1S pulse tests:
#
# [dELE]   = B_INV @ [aN]
# [dAIL]             [aE]
#
# Units here are the same identified local control units used
# in the successful XY teacher.
# ============================================================

B_INV = np.array(
    [
        [-0.20338265, -0.00859620],
        [ 0.01952073, -0.14631468],
    ],
    dtype=np.float64,
)


# ============================================================
# ENV ACTION AUTHORITY
#
# In HelicopterEnvStage1Distill:
# elevator residual physical authority ~ 0.026
# aileron  residual physical authority ~ 0.026
# ============================================================

ACTION_CYCLIC_AUTHORITY = 0.026


# ============================================================
# EARLY-TAKEOFF RESIDUAL CONTROLLER WINDOW
#
# Full correction below 100 ft.
# Smoothly fades out between 100 and 140 ft.
# Above 140 ft, final PPO flies completely alone.
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
# GRID
#
# The old successful XY teacher used approximately:
# KP=0.016, KD=0.200, A_MAX=0.12, DELTA_MAX=0.026
#
# Now the final PPO already has its own XY policy, so this is
# a RESIDUAL correction on top of it. We sweep stronger
# damping/gain but keep bounded authority.
# ============================================================

KP_VALUES = [
    0.010,
    0.016,
    0.022,
    0.030,
    0.040,
]

KD_VALUES = [
    0.15,
    0.20,
    0.28,
    0.36,
    0.46,
]

A_MAX_VALUES = [
    0.08,
    0.12,
    0.18,
]

DELTA_MAX_VALUES = [
    0.012,
    0.018,
    0.026,
]


# ============================================================
# QUICK SEARCH SETTINGS
# ============================================================

QUICK_TIME = 40.0

TARGET_ALT_FOR_EARLY_METRICS = 100.0


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def run_candidate(
    kp,
    kd,
    a_max,
    delta_max,
    total_time=QUICK_TIME,
    detailed=False,
):
    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env.reset()

    dt = float(env.dt)
    max_steps = int(total_time / dt)

    next_print = 0.0

    early_max_drift = 0.0
    early_path_at_100 = None
    drift_at_100 = None
    horizontal_speed_at_100 = None
    time_at_100 = None

    # Used to quantify how much meandering occurred.
    # If path is much larger than displacement, the trajectory
    # is curving / reversing direction.
    early_s_index = None

    physical_failure = False

    last_info = info
    altitude = float(info.get("altitude", 0.0))

    for step in range(max_steps):
        # ----------------------------------------------------
        # FINAL SINGLE PPO
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
        # CURRENT STATE
        # ----------------------------------------------------
        state = env._state()

        altitude = float(state["altitude"])
        north = float(state["north"])
        east = float(state["east"])
        vn = float(state["vn"])
        ve = float(state["ve"])

        gate = altitude_gate(altitude)

        # ----------------------------------------------------
        # EARLY XY POSITION + VELOCITY FEEDBACK
        #
        # Desired horizontal acceleration:
        #
        # a_des = -Kp * position - Kd * velocity
        # ----------------------------------------------------
        desired_accel = np.array(
            [
                -kp * north - kd * vn,
                -kp * east  - kd * ve,
            ],
            dtype=np.float64,
        )

        desired_accel = np.clip(
            desired_accel,
            -a_max,
            +a_max,
        )

        # ----------------------------------------------------
        # DECOUPLE INTO PHYSICAL ELEVATOR / AILERON DELTAS
        # ----------------------------------------------------
        delta_cyclic = B_INV @ desired_accel

        delta_cyclic = np.clip(
            delta_cyclic,
            -delta_max,
            +delta_max,
        )

        delta_ele = float(delta_cyclic[0]) * gate
        delta_ail = float(delta_cyclic[1]) * gate

        # ----------------------------------------------------
        # PHYSICAL CYCLIC DELTA -> NORMALIZED PPO ACTION DELTA
        #
        # env:
        # physical_delta ~= 0.026 * action
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
            np.clip(action[1], -1.0, 1.0)
        )

        action[2] = float(
            np.clip(action[2], -1.0, 1.0)
        )

        # ----------------------------------------------------
        # STEP
        # ----------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        last_info = info

        t = (step + 1) * dt

        altitude = float(info["altitude"])
        drift = float(info["drift"])
        path = float(info["path"])
        vn_now = float(info["vn"])
        ve_now = float(info["ve"])

        horizontal_speed = float(
            np.hypot(
                vn_now,
                ve_now,
            )
        )

        # ----------------------------------------------------
        # FIRST 100 FT METRICS
        # ----------------------------------------------------
        if altitude <= TARGET_ALT_FOR_EARLY_METRICS:
            early_max_drift = max(
                early_max_drift,
                drift,
            )

        if (
            early_path_at_100 is None
            and altitude >= TARGET_ALT_FOR_EARLY_METRICS
        ):
            early_path_at_100 = path
            drift_at_100 = drift
            horizontal_speed_at_100 = horizontal_speed
            time_at_100 = t

            early_s_index = max(
                0.0,
                early_path_at_100
                -
                drift_at_100,
            )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------
        if detailed and t >= next_print:
            print(
                f"t={t:6.1f}s | "
                f"ALT={altitude:7.2f} | "
                f"DRIFT={drift:5.2f} | "
                f"PATH={path:5.2f} | "
                f"VN={vn_now:+6.3f} | "
                f"VE={ve_now:+6.3f} | "
                f"GATE={gate:4.2f} | "
                f"dELE={delta_ele:+.5f} | "
                f"dAIL={delta_ail:+.5f}"
            )

            next_print += 2.5

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------
        if terminated and not info["success"]:
            physical_failure = True
            break

        if drift > 15.0:
            physical_failure = True
            break

        if altitude > 350.0:
            physical_failure = True
            break

    env.close()

    # Fallback in case 100 ft was never reached.
    if early_path_at_100 is None:
        early_path_at_100 = float(
            last_info.get("path", 999.0)
        )
        drift_at_100 = float(
            last_info.get("drift", 999.0)
        )
        horizontal_speed_at_100 = float(
            np.hypot(
                float(last_info.get("vn", 999.0)),
                float(last_info.get("ve", 999.0)),
            )
        )
        time_at_100 = total_time
        early_s_index = 999.0
        physical_failure = True

    final_drift = float(
        last_info.get("drift", 999.0)
    )

    max_drift_total = float(
        last_info.get("max_drift", 999.0)
    )

    final_path = float(
        last_info.get("path", 999.0)
    )

    # --------------------------------------------------------
    # SCORE
    #
    # Main target is a genuinely straight first 100 ft:
    #
    # 1) low max displacement
    # 2) low traveled XY path
    # 3) low S/meandering index
    # 4) low residual horizontal speed at 100 ft
    # --------------------------------------------------------
    score = (
        200.0 * early_max_drift
        + 80.0 * early_path_at_100
        + 100.0 * early_s_index
        + 60.0 * horizontal_speed_at_100
        + 10.0 * max_drift_total
    )

    if physical_failure:
        score += 1e9

    return {
        "kp": kp,
        "kd": kd,
        "a_max": a_max,
        "delta_max": delta_max,
        "failed": physical_failure,
        "score": score,
        "early_max_drift": early_max_drift,
        "path100": early_path_at_100,
        "drift100": drift_at_100,
        "hs100": horizontal_speed_at_100,
        "s_index": early_s_index,
        "time100": time_at_100,
        "max_drift_total": max_drift_total,
        "final_drift": final_drift,
        "final_path": final_path,
        "final_alt": altitude,
    }


# ============================================================
# BASELINE
# ============================================================

print("=" * 130)
print("STAGE 1 EARLY-TAKEOFF XY STRAIGHTNESS CALIBRATION")
print("=" * 130)

print("\nBase model:")
print(MODEL_PATH)

print(
    "\nGoal: remove the visible S-shaped motion "
    "during the first 100 ft."
)

print(
    "Main metrics: MAX DRIFT <100 ft, XY PATH at 100 ft, "
    "S-INDEX = path100 - drift100."
)


# ============================================================
# GRID SEARCH
# ============================================================

results = []

total_candidates = (
    len(KP_VALUES)
    *
    len(KD_VALUES)
    *
    len(A_MAX_VALUES)
    *
    len(DELTA_MAX_VALUES)
)

print(
    f"\nTesting {total_candidates} candidates..."
)
print()


counter = 0

for kp in KP_VALUES:
    for kd in KD_VALUES:
        for a_max in A_MAX_VALUES:
            for delta_max in DELTA_MAX_VALUES:
                counter += 1

                r = run_candidate(
                    kp=kp,
                    kd=kd,
                    a_max=a_max,
                    delta_max=delta_max,
                    total_time=QUICK_TIME,
                    detailed=False,
                )

                results.append(r)

                if (
                    counter % 15 == 0
                    or r["early_max_drift"] < 2.0
                ):
                    print(
                        f"{counter:3d}/{total_candidates} | "
                        f"KP={kp:.3f} "
                        f"KD={kd:.2f} "
                        f"AMAX={a_max:.2f} "
                        f"DMAX={delta_max:.3f} | "
                        f"EARLY_MAX={r['early_max_drift']:.3f} | "
                        f"PATH100={r['path100']:.3f} | "
                        f"DRIFT100={r['drift100']:.3f} | "
                        f"S={r['s_index']:.3f} | "
                        f"HS100={r['hs100']:.3f}"
                    )


# ============================================================
# RANK
# ============================================================

results.sort(
    key=lambda r: r["score"]
)

print("\n")
print("=" * 130)
print("TOP 15 EARLY-TAKEOFF CONTROLLERS")
print("=" * 130)

for i, r in enumerate(
    results[:15],
    start=1,
):
    print(
        f"{i:2d}. "
        f"KP={r['kp']:.3f} | "
        f"KD={r['kd']:.2f} | "
        f"AMAX={r['a_max']:.2f} | "
        f"DMAX={r['delta_max']:.3f} | "
        f"EARLY_MAX={r['early_max_drift']:.3f} | "
        f"PATH100={r['path100']:.3f} | "
        f"DRIFT100={r['drift100']:.3f} | "
        f"S={r['s_index']:.3f} | "
        f"HS100={r['hs100']:.3f} | "
        f"TOTAL_MAX={r['max_drift_total']:.3f}"
    )


# ============================================================
# DETAILED BEST — 120 SEC
#
# This verifies that the early correction fades out cleanly and
# does not damage the already-solved hover.
# ============================================================

best = results[0]

print("\n")
print("=" * 130)
print("DETAILED BEST — FULL 120 SECOND VALIDATION")
print("=" * 130)

print(
    f"KP        = {best['kp']}"
)
print(
    f"KD        = {best['kd']}"
)
print(
    f"A_MAX     = {best['a_max']}"
)
print(
    f"DELTA_MAX = {best['delta_max']}"
)
print()


best_full = run_candidate(
    kp=best["kp"],
    kd=best["kd"],
    a_max=best["a_max"],
    delta_max=best["delta_max"],
    total_time=120.0,
    detailed=True,
)


print("\n")
print("=" * 130)
print("BEST FULL RESULT")
print("=" * 130)

print(
    "EARLY MAX DRIFT (<100 ft) :",
    round(
        best_full["early_max_drift"],
        3,
    ),
    "ft",
)

print(
    "XY PATH AT 100 ft         :",
    round(
        best_full["path100"],
        3,
    ),
    "ft",
)

print(
    "DRIFT AT 100 ft           :",
    round(
        best_full["drift100"],
        3,
    ),
    "ft",
)

print(
    "S-INDEX                    :",
    round(
        best_full["s_index"],
        3,
    ),
    "ft",
)

print(
    "HORIZONTAL SPEED @100 ft  :",
    round(
        best_full["hs100"],
        3,
    ),
    "ft/s",
)

print(
    "TOTAL MAX DRIFT            :",
    round(
        best_full["max_drift_total"],
        3,
    ),
    "ft",
)

print(
    "FINAL DRIFT                :",
    round(
        best_full["final_drift"],
        3,
    ),
    "ft",
)

print(
    "FINAL XY PATH              :",
    round(
        best_full["final_path"],
        3,
    ),
    "ft",
)

print(
    "FINAL ALTITUDE             :",
    round(
        best_full["final_alt"],
        3,
    ),
    "ft",
)


# ============================================================
# PRESENTATION-QUALITY EARLY TAKEOFF TARGET
# ============================================================

presentation_quality = bool(
    not best_full["failed"]
    and best_full["early_max_drift"] <= 2.0
    and best_full["path100"] <= 5.0
    and best_full["drift100"] <= 1.5
    and best_full["hs100"] <= 0.75
    and best_full["max_drift_total"] <= 8.0
)

print()
print(
    "EARLY TAKEOFF PRESENTATION QUALITY :",
    presentation_quality,
)

if presentation_quality:
    print(
        "\nSUCCESS: early S-motion is sufficiently reduced."
    )
    print(
        "Next step: distill this early XY teacher into "
        "the PPO cyclic outputs."
    )
else:
    print(
        "\nNot yet clean enough."
    )
    print(
        "Do NOT distill yet; refine the early controller first."
    )

print("=" * 130)
