import numpy as np

from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill
)


# ============================================================
# CURRENT BEST SINGLE 4-ACTION PPO
# ============================================================

MODEL_PATH = (
    "models_stage1_final/"
    "AH1S_STAGE1_FINAL_300FT.zip"
)

model = PPO.load(MODEL_PATH)


# ============================================================
# TARGET
# ============================================================

TARGET_ALT = 300.0

TOTAL_TIME = 120.0

HOVER_START = 60.0


# ============================================================
# CORRECTION GATE
#
# Climb is already excellent.
# Do not touch it below 280 ft.
# ============================================================

GATE_START = 280.0
GATE_FULL = 295.0


def gate_value(altitude):

    if altitude <= GATE_START:
        return 0.0

    if altitude >= GATE_FULL:
        return 1.0

    return float(
        (altitude - GATE_START)
        /
        (GATE_FULL - GATE_START)
    )


# ============================================================
# PHYSICAL COLLECTIVE CORRECTION
#
# correction =
#
# BASE_BIAS
# +
# KP * altitude_error
# -
# KD * vertical_speed
#
# altitude_error = target - altitude
#
# If above 300:
# error negative -> collective decreases.
#
# If climbing:
# VS positive -> collective decreases.
# ============================================================


BASE_BIASES = [
    -0.0015,
    -0.0020,
    -0.0025,
    -0.0030,
]


KP_VALUES = [
    0.00008,
    0.00012,
    0.00016,
    0.00020,
]


KD_VALUES = [
    0.0006,
    0.0009,
    0.0012,
]


MAX_NEGATIVE_CORRECTION = -0.0050
MAX_POSITIVE_CORRECTION = +0.0025


# ============================================================
# RUN ONE CONTROLLER
# ============================================================

def run_controller(
    base_bias,
    kp,
    kd,
    detailed=False
):

    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False
    )

    obs, info = env.reset()

    dt = env.dt

    max_steps = int(
        TOTAL_TIME / dt
    )

    hover_alts = []
    hover_vs = []

    altitude_violations = 0
    vs_violations = 0

    physical_failure = False

    next_print = 0.0


    for step in range(max_steps):

        t = (
            step
            *
            dt
        )


        # ====================================================
        # SINGLE PPO OUTPUT
        # ====================================================

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        action = np.asarray(
            action,
            dtype=np.float32
        ).copy()


        # ====================================================
        # CURRENT STATE
        # ====================================================

        s = env._state()

        altitude = float(
            s["altitude"]
        )

        vertical_speed = float(
            s["vertical_speed"]
        )


        altitude_error = (
            TARGET_ALT
            -
            altitude
        )


        gate = gate_value(
            altitude
        )


        # ====================================================
        # PHYSICAL COLLECTIVE PD
        # ====================================================

        correction = (
            base_bias
            +
            kp
            *
            altitude_error
            -
            kd
            *
            vertical_speed
        )


        correction = float(
            np.clip(
                correction,
                MAX_NEGATIVE_CORRECTION,
                MAX_POSITIVE_CORRECTION
            )
        )


        correction *= gate


        # ====================================================
        # PHYSICAL COLLECTIVE -> NORMALIZED PPO ACTION
        #
        # COL = 0.620 + 0.030*a0
        # ====================================================

        action[0] += (
            correction
            /
            0.030
        )


        action[0] = float(
            np.clip(
                action[0],
                -1.0,
                1.0
            )
        )


        # ====================================================
        # STEP
        # ====================================================

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )


        t = (
            (step + 1)
            *
            dt
        )


        altitude = float(
            info["altitude"]
        )

        vertical_speed = float(
            info["vertical_speed"]
        )

        altitude_error = abs(
            TARGET_ALT
            -
            altitude
        )


        # ====================================================
        # HOVER METRICS
        # ====================================================

        if t >= HOVER_START:

            hover_alts.append(
                altitude
            )

            hover_vs.append(
                vertical_speed
            )


            if altitude_error > 5.0:

                altitude_violations += 1


            if abs(vertical_speed) > 0.75:

                vs_violations += 1


        # ====================================================
        # DETAIL
        # ====================================================

        if (
            detailed
            and
            t >= next_print
        ):

            print(
                f"t={t:6.1f}s | "
                f"ALT={altitude:7.2f} | "
                f"ERR={altitude_error:5.2f} | "
                f"VS={vertical_speed:6.3f} | "
                f"DRIFT={info['drift']:5.2f} | "
                f"MAX={info['max_drift']:5.2f} | "
                f"PATH={info['path']:5.2f} | "
                f"COL={info['collective']:.5f} | "
                f"CORR={correction:+.5f}"
            )

            next_print += 5.0


        # ====================================================
        # PARENT SUCCESS IS NOT A STOP CONDITION
        # ====================================================

        if (
            terminated
            and
            not info["success"]
        ):

            physical_failure = True
            break


        if altitude > 340.0:

            physical_failure = True
            break


        if (
            altitude < 270.0
            and
            t > 70.0
        ):

            physical_failure = True
            break


        if info["max_drift"] > 12.0:

            physical_failure = True
            break


    env.close()


    hover_alts = np.asarray(
        hover_alts,
        dtype=np.float64
    )

    hover_vs = np.asarray(
        hover_vs,
        dtype=np.float64
    )


    if len(hover_alts) == 0:

        return {
            "base_bias": base_bias,
            "kp": kp,
            "kd": kd,
            "failed": True,
            "success": False,
            "score": 1e12
        }


    mean_alt = float(
        np.mean(
            hover_alts
        )
    )

    std_alt = float(
        np.std(
            hover_alts
        )
    )

    min_alt = float(
        np.min(
            hover_alts
        )
    )

    max_alt = float(
        np.max(
            hover_alts
        )
    )

    max_error = float(
        np.max(
            np.abs(
                hover_alts
                -
                TARGET_ALT
            )
        )
    )

    max_abs_vs = float(
        np.max(
            np.abs(
                hover_vs
            )
        )
    )


    sustained_success = bool(

        not physical_failure

        and

        altitude_violations == 0

        and

        vs_violations == 0

        and

        info["max_drift"] <= 8.0

        and

        info["drift"] <= 5.0

        and

        info["path"] <= 25.0
    )


    # ========================================================
    # RANKING
    # ========================================================

    score = (

        40.0
        *
        abs(
            mean_alt
            -
            TARGET_ALT
        )

        +

        25.0
        *
        max_error

        +

        10.0
        *
        std_alt

        +

        10.0
        *
        max_abs_vs

        +

        4.0
        *
        info["max_drift"]

        +

        0.5
        *
        info["path"]

        +

        500.0
        *
        altitude_violations

        +

        500.0
        *
        vs_violations
    )


    return {

        "base_bias":
            base_bias,

        "kp":
            kp,

        "kd":
            kd,

        "failed":
            physical_failure,

        "success":
            sustained_success,

        "mean_alt":
            mean_alt,

        "std_alt":
            std_alt,

        "min_alt":
            min_alt,

        "max_alt":
            max_alt,

        "max_error":
            max_error,

        "max_abs_vs":
            max_abs_vs,

        "alt_violations":
            altitude_violations,

        "vs_violations":
            vs_violations,

        "max_drift":
            float(
                info["max_drift"]
            ),

        "final_drift":
            float(
                info["drift"]
            ),

        "path":
            float(
                info["path"]
            ),

        "final_alt":
            altitude,

        "final_vs":
            vertical_speed,

        "score":
            score,
    }


# ============================================================
# GRID SEARCH
# ============================================================

print("=" * 125)
print("AH-1S STAGE 1 — SUSTAINED HOVER PD CALIBRATION")
print("=" * 125)

print()
print("Teacher              : runtime test only")
print("Classical XY control : OFF")
print("Base policy          : SINGLE 4-ACTION PPO")
print("Validation           : 120 seconds")
print()


results = []


for base_bias in BASE_BIASES:

    for kp in KP_VALUES:

        for kd in KD_VALUES:

            r = run_controller(
                base_bias,
                kp,
                kd,
                detailed=False
            )

            results.append(
                r
            )


            if r.get(
                "success",
                False
            ):

                icon = "🏆"

            elif r.get(
                "failed",
                False
            ):

                icon = "❌"

            else:

                icon = "✅"


            if "mean_alt" in r:

                print(
                    f"{icon} "
                    f"B={base_bias:+.4f} | "
                    f"KP={kp:.5f} | "
                    f"KD={kd:.4f} | "
                    f"MEAN={r['mean_alt']:7.2f} | "
                    f"RANGE=["
                    f"{r['min_alt']:.2f},"
                    f"{r['max_alt']:.2f}] | "
                    f"MAXERR={r['max_error']:5.2f} | "
                    f"MAXVS={r['max_abs_vs']:5.3f} | "
                    f"DRIFT={r['max_drift']:5.2f}"
                )

            else:

                print(
                    f"{icon} "
                    f"B={base_bias:+.4f} | "
                    f"KP={kp:.5f} | "
                    f"KD={kd:.4f} | FAILED"
                )


# ============================================================
# RANK
# ============================================================

results.sort(
    key=lambda x: x["score"]
)


print("\n")
print("=" * 125)
print("TOP 10 CONTROLLERS")
print("=" * 125)


for i, r in enumerate(
    results[:10],
    start=1
):

    print(
        f"{i:2d}. "
        f"SUCCESS={str(r['success']):5s} | "
        f"B={r['base_bias']:+.4f} | "
        f"KP={r['kp']:.5f} | "
        f"KD={r['kd']:.4f} | "
        f"MEAN={r['mean_alt']:7.3f} | "
        f"RANGE=["
        f"{r['min_alt']:.3f},"
        f"{r['max_alt']:.3f}] | "
        f"MAXERR={r['max_error']:.3f} | "
        f"MAXVS={r['max_abs_vs']:.3f} | "
        f"MAXDRIFT={r['max_drift']:.3f} | "
        f"PATH={r['path']:.3f}"
    )


best = results[0]


# ============================================================
# DETAILED BEST
# ============================================================

print("\n")
print("=" * 125)
print("DETAILED BEST PD CONTROLLER")
print("=" * 125)

print(
    "BASE BIAS =",
    best["base_bias"]
)

print(
    "KP        =",
    best["kp"]
)

print(
    "KD        =",
    best["kd"]
)

print()


detailed = run_controller(
    best["base_bias"],
    best["kp"],
    best["kd"],
    detailed=True
)


print("\n")
print("=" * 125)
print("BEST RESULT")
print("=" * 125)

print(
    "SUSTAINED SUCCESS :",
    detailed["success"]
)

print(
    "MEAN ALTITUDE     :",
    round(
        detailed["mean_alt"],
        3
    ),
    "ft"
)

print(
    "STD ALTITUDE      :",
    round(
        detailed["std_alt"],
        3
    ),
    "ft"
)

print(
    "MIN ALTITUDE      :",
    round(
        detailed["min_alt"],
        3
    ),
    "ft"
)

print(
    "MAX ALTITUDE      :",
    round(
        detailed["max_alt"],
        3
    ),
    "ft"
)

print(
    "MAX ALT ERROR     :",
    round(
        detailed["max_error"],
        3
    ),
    "ft"
)

print(
    "MAX |VS|          :",
    round(
        detailed["max_abs_vs"],
        3
    ),
    "ft/s"
)

print(
    "ALT VIOLATIONS    :",
    detailed["alt_violations"]
)

print(
    "VS VIOLATIONS     :",
    detailed["vs_violations"]
)

print(
    "MAX DRIFT         :",
    round(
        detailed["max_drift"],
        3
    ),
    "ft"
)

print(
    "FINAL DRIFT       :",
    round(
        detailed["final_drift"],
        3
    ),
    "ft"
)

print(
    "PATH              :",
    round(
        detailed["path"],
        3
    ),
    "ft"
)

print(
    "FINAL ALTITUDE    :",
    round(
        detailed["final_alt"],
        3
    ),
    "ft"
)

print(
    "FINAL VS          :",
    round(
        detailed["final_vs"],
        3
    ),
    "ft/s"
)

print("=" * 125)
