import numpy as np

from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    "models_stage1_final/"
    "AH1S_STAGE1_FINAL_300FT.zip"
)

model = PPO.load(MODEL_PATH)


# ============================================================
# MISSION
# ============================================================

TARGET_ALT = 300.0

TOTAL_TIME = 120.0

HOVER_START = 60.0


# ============================================================
# GATE
#
# Leave the already-good climb alone.
# Start damping near hover.
# ============================================================

GATE_START = 285.0
GATE_FULL = 298.0


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
# FINER / STRONGER PD SEARCH
#
# Previous search stopped at KP=0.00020.
#
# Trend showed that increasing KP was STILL improving
# sustained altitude.
# ============================================================

BASE_BIASES = [
    -0.0015,
    -0.0017,
    -0.0019,
    -0.0021,
    -0.0023,
]


KP_VALUES = [
    0.00022,
    0.00026,
    0.00030,
    0.00034,
    0.00038,
    0.00042,
    0.00046,
    0.00050,
]


KD_VALUES = [
    0.0008,
    0.0011,
    0.0014,
    0.0017,
]


# More negative authority allowed for late overshoot.
MAX_NEGATIVE_CORRECTION = -0.0060

MAX_POSITIVE_CORRECTION = +0.0030


# ============================================================
# RUN
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


    hover_altitudes = []
    hover_vertical_speeds = []


    altitude_violations = 0
    vs_violations = 0


    next_print = 0.0

    physical_failure = False


    for step in range(max_steps):

        t = (
            step
            *
            dt
        )


        # ====================================================
        # ORIGINAL SINGLE PPO
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
        # STATE
        # ====================================================

        state = env._state()

        altitude = float(
            state["altitude"]
        )

        vertical_speed = float(
            state["vertical_speed"]
        )


        signed_error = (
            TARGET_ALT
            -
            altitude
        )


        # ====================================================
        # PD
        #
        # Positive altitude error:
        # helicopter below target
        # -> add collective.
        #
        # Negative altitude error:
        # helicopter above target
        # -> remove collective.
        #
        # Positive VS:
        # climbing
        # -> remove collective.
        # ====================================================

        correction = (
            base_bias
            +
            kp
            *
            signed_error
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


        gate = gate_value(
            altitude
        )


        correction *= gate


        # ====================================================
        # PHYSICAL COLLECTIVE CORRECTION
        # ->
        # NORMALIZED ACTION CORRECTION
        #
        # collective = 0.620 + 0.030*a0
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
        # PHYSICS
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
        # HOVER WINDOW
        # ====================================================

        if t >= HOVER_START:

            hover_altitudes.append(
                altitude
            )

            hover_vertical_speeds.append(
                vertical_speed
            )


            if altitude_error > 5.0:

                altitude_violations += 1


            if abs(vertical_speed) > 0.75:

                vs_violations += 1


        # ====================================================
        # LOG
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
        # FAILURE
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


    hover_altitudes = np.asarray(
        hover_altitudes,
        dtype=np.float64
    )

    hover_vertical_speeds = np.asarray(
        hover_vertical_speeds,
        dtype=np.float64
    )


    if len(hover_altitudes) == 0:

        return {
            "base_bias": base_bias,
            "kp": kp,
            "kd": kd,
            "failed": True,
            "success": False,
            "score": 1e12,
        }


    mean_alt = float(
        np.mean(
            hover_altitudes
        )
    )


    std_alt = float(
        np.std(
            hover_altitudes
        )
    )


    min_alt = float(
        np.min(
            hover_altitudes
        )
    )


    max_alt = float(
        np.max(
            hover_altitudes
        )
    )


    max_error = float(
        np.max(
            np.abs(
                hover_altitudes
                -
                TARGET_ALT
            )
        )
    )


    max_abs_vs = float(
        np.max(
            np.abs(
                hover_vertical_speeds
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
    # SCORE
    #
    # Any violation gets huge penalty.
    # Then rank by maximum error, mean error and oscillation.
    # ============================================================

    score = (

        10000.0
        *
        altitude_violations

        +

        10000.0
        *
        vs_violations

        +

        100.0
        *
        max_error

        +

        40.0
        *
        abs(
            mean_alt
            -
            TARGET_ALT
        )

        +

        20.0
        *
        std_alt

        +

        10.0
        *
        max_abs_vs

        +

        3.0
        *
        info["max_drift"]

        +

        info["path"]
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
# SEARCH
# ============================================================

print("=" * 125)

print(
    "STAGE 1 SUSTAINED HOVER — PD V2"
)

print("=" * 125)


print(
    "\nBase policy:"
)

print(
    MODEL_PATH
)


print(
    "\nTeacher/controller used only "
    "for calibration."
)

print(
    "XY controller: OFF"
)

print(
    "Validation: 120 seconds"
)

print()


results = []


for base_bias in BASE_BIASES:

    for kp in KP_VALUES:

        for kd in KD_VALUES:

            result = run_controller(
                base_bias,
                kp,
                kd,
                detailed=False
            )


            results.append(
                result
            )


            if result["success"]:

                icon = "🏆"

            elif result["failed"]:

                icon = "❌"

            else:

                icon = "•"


            print(
                f"{icon} "
                f"B={base_bias:+.4f} | "
                f"KP={kp:.5f} | "
                f"KD={kd:.4f} | "
                f"MEAN={result['mean_alt']:7.2f} | "
                f"RANGE=["
                f"{result['min_alt']:.2f},"
                f"{result['max_alt']:.2f}] | "
                f"MAXERR={result['max_error']:5.2f} | "
                f"MAXVS={result['max_abs_vs']:5.3f} | "
                f"VIOL={result['alt_violations']:4d} | "
                f"DRIFT={result['max_drift']:5.2f}"
            )


# ============================================================
# RANK
# ============================================================

results.sort(
    key=lambda r: r["score"]
)


print("\n")
print("=" * 125)
print("TOP 15")
print("=" * 125)


for i, result in enumerate(
    results[:15],
    start=1
):

    print(
        f"{i:2d}. "
        f"SUCCESS={str(result['success']):5s} | "
        f"B={result['base_bias']:+.4f} | "
        f"KP={result['kp']:.5f} | "
        f"KD={result['kd']:.4f} | "
        f"MEAN={result['mean_alt']:7.3f} | "
        f"RANGE=["
        f"{result['min_alt']:.3f},"
        f"{result['max_alt']:.3f}] | "
        f"MAXERR={result['max_error']:.3f} | "
        f"MAXVS={result['max_abs_vs']:.3f} | "
        f"VIOL={result['alt_violations']} | "
        f"MAXDRIFT={result['max_drift']:.3f} | "
        f"PATH={result['path']:.3f}"
    )


# ============================================================
# BEST DETAIL
# ============================================================

best = results[0]


print("\n")
print("=" * 125)
print("DETAILED BEST")
print("=" * 125)

print(
    "BASE BIAS:",
    best["base_bias"]
)

print(
    "KP:",
    best["kp"]
)

print(
    "KD:",
    best["kd"]
)

print()


best = run_controller(
    best["base_bias"],
    best["kp"],
    best["kd"],
    detailed=True
)


print("\n")
print("=" * 125)
print("FINAL BEST RESULT")
print("=" * 125)


print(
    "SUSTAINED SUCCESS :",
    best["success"]
)

print(
    "MEAN ALTITUDE     :",
    round(
        best["mean_alt"],
        3
    ),
    "ft"
)

print(
    "STD ALTITUDE      :",
    round(
        best["std_alt"],
        3
    ),
    "ft"
)

print(
    "MIN ALTITUDE      :",
    round(
        best["min_alt"],
        3
    ),
    "ft"
)

print(
    "MAX ALTITUDE      :",
    round(
        best["max_alt"],
        3
    ),
    "ft"
)

print(
    "MAX ALT ERROR     :",
    round(
        best["max_error"],
        3
    ),
    "ft"
)

print(
    "MAX |VS|          :",
    round(
        best["max_abs_vs"],
        3
    ),
    "ft/s"
)

print(
    "ALT VIOLATIONS    :",
    best["alt_violations"]
)

print(
    "VS VIOLATIONS     :",
    best["vs_violations"]
)

print(
    "MAX DRIFT         :",
    round(
        best["max_drift"],
        3
    ),
    "ft"
)

print(
    "FINAL DRIFT       :",
    round(
        best["final_drift"],
        3
    ),
    "ft"
)

print(
    "PATH              :",
    round(
        best["path"],
        3
    ),
    "ft"
)

print(
    "FINAL ALTITUDE    :",
    round(
        best["final_alt"],
        3
    ),
    "ft"
)

print(
    "FINAL VS          :",
    round(
        best["final_vs"],
        3
    ),
    "ft/s"
)


print("=" * 125)
