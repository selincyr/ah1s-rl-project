import math

from stable_baselines3 import PPO
from helicopter_env_9 import HelicopterEnv


MODEL_PATH = "models/task9_hover300/best/best_model"
OUTPUT_FILE = "ah1s_v9_best.acmi"

# RL step yaklaşık 0.075 saniye
DT = 0.075

# ACMI object id
OBJECT_ID = "100"


def rad_to_deg(value):
    return math.degrees(float(value))


def ft_to_m(value):
    return float(value) * 0.3048


def normalize_heading_deg(value):
    value = value % 360.0

    if value < 0.0:
        value += 360.0

    return value


def safe_get(fdm, property_name, default=0.0):
    try:
        return float(
            fdm[property_name]
        )
    except Exception:
        return float(default)


def main():

    print("=" * 80)
    print("V9 BEST -> ACMI WRITER")
    print("=" * 80)

    env = HelicopterEnv()

    model = PPO.load(
        MODEL_PATH
    )

    obs, info = env.reset()

    print()
    print("Model loaded:")
    print(MODEL_PATH)

    # ----------------------------------------------------------
    # İlk JSBSim konumunu kontrol ediyoruz
    # ----------------------------------------------------------

    initial_latitude = safe_get(
        env.fdm,
        "position/lat-gc-deg"
    )

    initial_longitude = safe_get(
        env.fdm,
        "position/long-gc-deg"
    )

    initial_altitude_ft = safe_get(
        env.fdm,
        "position/h-sl-ft"
    )

    print()
    print(
        "Initial latitude:",
        initial_latitude
    )

    print(
        "Initial longitude:",
        initial_longitude
    )

    print(
        "Initial MSL altitude:",
        initial_altitude_ft,
        "ft"
    )

    # ----------------------------------------------------------
    # ACMI dosyasını oluştur
    # ----------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as acmi:

        # ======================================================
        # ACMI HEADER
        # ======================================================

        acmi.write(
            "FileType=text/acmi/tacview\n"
        )

        acmi.write(
            "FileVersion=2.2\n"
        )

        acmi.write(
            "0,ReferenceTime=2026-09-02T00:00:00Z\n"
        )

        acmi.write(
            "0,Title=AH-1S RL V9 BEST\n"
        )

        acmi.write(
            "0,Author=AH1S RL Project\n"
        )

        acmi.write(
            "0,Comments=Ground to 300 ft hover training - PPO V9 BEST\n"
        )

        # ======================================================
        # OBJECT INFORMATION
        # ======================================================

        acmi.write(
            f"{OBJECT_ID},"
            "Type=Air+Rotorcraft,"
            "Name=AH-1S,"
            "Pilot=PPO V9 BEST,"
            "Coalition=Blue,"
            "Color=Blue\n"
        )

        # ======================================================
        # SIMULATION
        # ======================================================

        for step in range(1, 5001):

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

            # --------------------------------------------------
            # TIME
            # --------------------------------------------------

            sim_time = step * DT

            # --------------------------------------------------
            # POSITION
            # --------------------------------------------------

            latitude = safe_get(
                env.fdm,
                "position/lat-gc-deg",
                initial_latitude
            )

            longitude = safe_get(
                env.fdm,
                "position/long-gc-deg",
                initial_longitude
            )

            altitude_ft = safe_get(
                env.fdm,
                "position/h-sl-ft"
            )

            altitude_m = ft_to_m(
                altitude_ft
            )

            # --------------------------------------------------
            # ATTITUDE
            # --------------------------------------------------

            roll_rad = safe_get(
                env.fdm,
                "attitude/roll-rad"
            )

            pitch_rad = safe_get(
                env.fdm,
                "attitude/pitch-rad"
            )

            heading_rad = safe_get(
                env.fdm,
                "attitude/heading-true-rad"
            )

            roll_deg = rad_to_deg(
                roll_rad
            )

            pitch_deg = rad_to_deg(
                pitch_rad
            )

            heading_deg = normalize_heading_deg(
                rad_to_deg(
                    heading_rad
                )
            )

            # --------------------------------------------------
            # ACMI FRAME
            # --------------------------------------------------

            acmi.write(
                f"#{sim_time:.3f}\n"
            )

            acmi.write(
                f"{OBJECT_ID},"
                f"T={longitude:.8f}|"
                f"{latitude:.8f}|"
                f"{altitude_m:.3f}|"
                f"{roll_deg:.3f}|"
                f"{pitch_deg:.3f}|"
                f"{heading_deg:.3f}\n"
            )

            # --------------------------------------------------
            # DEBUG
            # --------------------------------------------------

            if step % 100 == 0:

                print(
                    f"Step {step:4d} | "
                    f"Time {sim_time:7.2f}s | "
                    f"Alt AGL {info['altitude']:7.2f} ft | "
                    f"Lat {latitude:.6f} | "
                    f"Lon {longitude:.6f} | "
                    f"Heading {heading_deg:7.2f}"
                )

            # --------------------------------------------------
            # END
            # --------------------------------------------------

            if terminated or truncated:

                print()
                print(
                    "Episode ended at step:",
                    step
                )

                print(
                    "Termination reason:",
                    info.get(
                        "termination_reason"
                    )
                )

                print(
                    "Final AGL altitude:",
                    f"{info['altitude']:.2f} ft"
                )

                break

        # Object removal at end
        acmi.write(
            f"-{OBJECT_ID}\n"
        )

    env.close()

    print()
    print("=" * 80)
    print("ACMI CREATED")
    print("=" * 80)

    print(
        "File:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
