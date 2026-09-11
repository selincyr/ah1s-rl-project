import numpy as np

from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill
)


# ============================================================
# FINAL MODEL
# ============================================================

MODEL_PATH = (
    "models_stage1_final/"
    "AH1S_STAGE1_FINAL_300FT.zip"
)


model = PPO.load(
    MODEL_PATH
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

TOTAL_TIME = 100.0

HOVER_START = 60.0

TARGET_ALTITUDE = 300.0

ALTITUDE_BAND = 5.0

MAX_VERTICAL_SPEED = 0.75

MAX_DRIFT_LIMIT = 8.0

FINAL_DRIFT_LIMIT = 5.0

PATH_LIMIT = 25.0


# ============================================================
# ENV
#
# Teacher OFF
# Controller OFF
# Runtime bias OFF
# ============================================================

env = HelicopterEnvStage1Distill(
    teacher_model_path=None,
    training_mode=False
)

obs, info = env.reset()

dt = env.dt

max_steps = int(
    TOTAL_TIME / dt
)


# ============================================================
# METRICS
# ============================================================

hover_altitudes = []
hover_vertical_speeds = []

hover_errors = []

altitude_violations = 0
vs_violations = 0

next_print = 0.0

physical_failure = False


print("=" * 120)
print("FINAL STAGE 1 — 100 SECOND SUSTAINED VALIDATION")
print("=" * 120)

print("\nModel:")
print(MODEL_PATH)

print("\nTeacher              : OFF")
print("Classical controller : OFF")
print("Runtime bias         : OFF")
print("Policy               : SINGLE 4-ACTION PPO")

print("\nHover validation:")
print("60 s -> 100 s")
print("Altitude band: 295 -> 305 ft")
print()


# ============================================================
# FLIGHT
# ============================================================

for step in range(max_steps):

    action, _ = model.predict(
        obs,
        deterministic=True
    )

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

    drift = float(
        info["drift"]
    )

    max_drift = float(
        info["max_drift"]
    )

    path = float(
        info["path"]
    )


    # ========================================================
    # HOVER WINDOW
    # ========================================================

    if t >= HOVER_START:

        altitude_error = abs(
            TARGET_ALTITUDE
            -
            altitude
        )

        hover_altitudes.append(
            altitude
        )

        hover_vertical_speeds.append(
            vertical_speed
        )

        hover_errors.append(
            altitude_error
        )


        if altitude_error > ALTITUDE_BAND:

            altitude_violations += 1


        if abs(vertical_speed) > MAX_VERTICAL_SPEED:

            vs_violations += 1


    # ========================================================
    # PRINT EVERY 5 SEC
    # ========================================================

    if t >= next_print:

        print(
            f"t={t:6.1f}s | "
            f"ALT={altitude:7.2f} | "
            f"ERR={abs(300.0-altitude):5.2f} | "
            f"VS={vertical_speed:6.2f} | "
            f"DRIFT={drift:5.2f} | "
            f"MAX={max_drift:5.2f} | "
            f"PATH={path:5.2f} | "
            f"N={info['north']:6.2f} | "
            f"E={info['east']:6.2f} | "
            f"COL={info['collective']:.5f}"
        )

        next_print += 5.0


    # ========================================================
    # IMPORTANT
    #
    # Environment returns terminated=True when its original
    # success condition is reached.
    #
    # For this validation we intentionally KEEP FLYING.
    # ========================================================

    if (
        terminated
        and
        not info["success"]
    ):

        physical_failure = True

        print(
            "\n❌ Physical failure termination."
        )

        break


    # Additional safety
    if altitude > 360.0:

        physical_failure = True
        break


    if max_drift > 15.0:

        physical_failure = True
        break


# ============================================================
# SUMMARY
# ============================================================

env.close()


hover_altitudes = np.asarray(
    hover_altitudes,
    dtype=np.float64
)

hover_vertical_speeds = np.asarray(
    hover_vertical_speeds,
    dtype=np.float64
)

hover_errors = np.asarray(
    hover_errors,
    dtype=np.float64
)


mean_altitude = float(
    np.mean(
        hover_altitudes
    )
)

std_altitude = float(
    np.std(
        hover_altitudes
    )
)

min_altitude = float(
    np.min(
        hover_altitudes
    )
)

max_altitude = float(
    np.max(
        hover_altitudes
    )
)

max_alt_error = float(
    np.max(
        hover_errors
    )
)

mean_alt_error = float(
    np.mean(
        hover_errors
    )
)

max_abs_vs = float(
    np.max(
        np.abs(
            hover_vertical_speeds
        )
    )
)


final_altitude = altitude

final_error = abs(
    TARGET_ALTITUDE
    -
    final_altitude
)

final_drift = drift


# ============================================================
# FINAL PRESENTATION-QUALITY PASS
# ============================================================

sustained_success = bool(

    not physical_failure

    and

    altitude_violations == 0

    and

    vs_violations == 0

    and

    max_drift
    <=
    MAX_DRIFT_LIMIT

    and

    final_drift
    <=
    FINAL_DRIFT_LIMIT

    and

    path
    <=
    PATH_LIMIT
)


print("\n")
print("=" * 120)
print("SUSTAINED HOVER SUMMARY")
print("=" * 120)

print(
    "HOVER WINDOW      :",
    f"{HOVER_START:.0f} -> {TOTAL_TIME:.0f} s"
)

print(
    "MEAN ALTITUDE     :",
    round(
        mean_altitude,
        3
    ),
    "ft"
)

print(
    "STD ALTITUDE      :",
    round(
        std_altitude,
        3
    ),
    "ft"
)

print(
    "MIN ALTITUDE      :",
    round(
        min_altitude,
        3
    ),
    "ft"
)

print(
    "MAX ALTITUDE      :",
    round(
        max_altitude,
        3
    ),
    "ft"
)

print(
    "MEAN ALT ERROR    :",
    round(
        mean_alt_error,
        3
    ),
    "ft"
)

print(
    "MAX ALT ERROR     :",
    round(
        max_alt_error,
        3
    ),
    "ft"
)

print(
    "MAX |VS|          :",
    round(
        max_abs_vs,
        3
    ),
    "ft/s"
)

print(
    "ALT VIOLATIONS    :",
    altitude_violations
)

print(
    "VS VIOLATIONS     :",
    vs_violations
)


print("\n")
print("=" * 120)
print("XY SUMMARY")
print("=" * 120)

print(
    "MAX DRIFT         :",
    round(
        max_drift,
        3
    ),
    "ft"
)

print(
    "FINAL DRIFT       :",
    round(
        final_drift,
        3
    ),
    "ft"
)

print(
    "PATH              :",
    round(
        path,
        3
    ),
    "ft"
)

print(
    "FINAL NORTH       :",
    round(
        float(
            info["north"]
        ),
        3
    ),
    "ft"
)

print(
    "FINAL EAST        :",
    round(
        float(
            info["east"]
        ),
        3
    ),
    "ft"
)


print("\n")
print("=" * 120)
print("FINAL VALIDATION")
print("=" * 120)

print(
    "SUSTAINED SUCCESS :",
    sustained_success
)

print(
    "FINAL ALTITUDE    :",
    round(
        final_altitude,
        3
    ),
    "ft"
)

print(
    "FINAL ALT ERROR   :",
    round(
        final_error,
        3
    ),
    "ft"
)


if sustained_success:

    print("\n🏆 STAGE 1 PRESENTATION-QUALITY COMPLETE")

    print(
        "Single PPO"
    )

    print(
        "4 actions"
    )

    print(
        "Teacher OFF"
    )

    print(
        "Classical controller OFF"
    )

    print(
        "Runtime bias OFF"
    )

    print(
        "Sustained 300 ± 5 ft hover validated."
    )

else:

    print(
        "\n⚠️ Short-term success passed, "
        "but sustained hover still needs refinement."
    )


print("=" * 120)
