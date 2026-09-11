from stable_baselines3 import PPO

from helicopter_env_3_v2 import HelicopterEnv


def test_model(model_path, model_name):

    print()
    print("=" * 65)
    print("MODEL:", model_name)
    print("=" * 65)

    env = HelicopterEnv()

    model = PPO.load(
        model_path,
        env=env
    )

    obs, info = env.reset()

    max_altitude = info["altitude"]
    max_forward = abs(info["forward_velocity"])
    max_lateral = abs(info["lateral_velocity"])
    max_heading_error = abs(info["heading_error"])
    max_roll = abs(info["roll"])
    max_pitch = abs(info["pitch"])
    max_vertical_speed = abs(info["vertical_speed"])

    max_hold = 0

    reached_takeoff = False
    reached_climb = False
    reached_approach = False
    reached_forward = False

    first_900_step = None
    first_970_step = None
    first_1000_step = None

    final_info = info

    for step in range(5000):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        final_info = info

        altitude = info["altitude"]

        max_altitude = max(
            max_altitude,
            altitude
        )

        max_forward = max(
            max_forward,
            abs(info["forward_velocity"])
        )

        max_lateral = max(
            max_lateral,
            abs(info["lateral_velocity"])
        )

        max_heading_error = max(
            max_heading_error,
            abs(info["heading_error"])
        )

        max_roll = max(
            max_roll,
            abs(info["roll"])
        )

        max_pitch = max(
            max_pitch,
            abs(info["pitch"])
        )

        max_vertical_speed = max(
            max_vertical_speed,
            abs(info["vertical_speed"])
        )

        max_hold = max(
            max_hold,
            info["target_hold_steps"]
        )

        if altitude >= 15.0:
            reached_takeoff = True

        if altitude >= 30.0:
            reached_climb = True

        if altitude >= 900.0:
            reached_approach = True

            if first_900_step is None:
                first_900_step = step

        if altitude >= 970.0:
            reached_forward = True

            if first_970_step is None:
                first_970_step = step

        if altitude >= 1000.0:
            if first_1000_step is None:
                first_1000_step = step

        if step % 100 == 0:

            print(
                f"Step {step:4d} | "
                f"{info['phase']:8s} | "
                f"Alt {info['altitude']:8.2f} | "
                f"VS {info['vertical_speed']:7.2f} | "
                f"Fwd {info['forward_velocity']:7.2f} | "
                f"Lat {info['lateral_velocity']:7.2f} | "
                f"HeadErr {info['heading_error']:7.3f} | "
                f"Roll {info['roll']:7.3f} | "
                f"Pitch {info['pitch']:7.3f} | "
                f"Hold {info['target_hold_steps']:3d}"
            )

        if terminated or truncated:

            print()
            print(
                "Episode ended at step:",
                step
            )

            break

    print()
    print("=" * 65)
    print("TEST OZETI")
    print("=" * 65)

    print("Model:", model_name)

    print(
        "Max altitude:",
        round(max_altitude, 2)
    )

    print(
        "Max |vertical speed|:",
        round(max_vertical_speed, 2)
    )

    print(
        "Max |forward velocity|:",
        round(max_forward, 2)
    )

    print(
        "Max |lateral velocity|:",
        round(max_lateral, 2)
    )

    print(
        "Max |heading error|:",
        round(max_heading_error, 3)
    )

    print(
        "Max |roll|:",
        round(max_roll, 3)
    )

    print(
        "Max |pitch|:",
        round(max_pitch, 3)
    )

    print(
        "Max hold:",
        max_hold,
        "/ 100"
    )

    print()
    print("PHASES")

    print(
        "Takeoff reached:",
        reached_takeoff
    )

    print(
        "Climb reached:",
        reached_climb
    )

    print(
        "Approach reached:",
        reached_approach
    )

    print(
        "Forward reached:",
        reached_forward
    )

    print()
    print("MILESTONES")

    print(
        "900 ft step:",
        first_900_step
    )

    print(
        "970 ft step:",
        first_970_step
    )

    print(
        "1000 ft step:",
        first_1000_step
    )

    print()
    print("FINAL STATE")

    print(
        "Phase:",
        final_info["phase"]
    )

    print(
        "Altitude:",
        round(
            final_info["altitude"],
            2
        )
    )

    print(
        "Vertical speed:",
        round(
            final_info["vertical_speed"],
            2
        )
    )

    print(
        "Forward velocity:",
        round(
            final_info["forward_velocity"],
            2
        )
    )

    print(
        "Target forward:",
        round(
            final_info["target_forward_velocity"],
            2
        )
    )

    print(
        "Lateral velocity:",
        round(
            final_info["lateral_velocity"],
            2
        )
    )

    print(
        "Heading error:",
        round(
            final_info["heading_error"],
            3
        )
    )

    print(
        "Roll:",
        round(
            final_info["roll"],
            3
        )
    )

    print(
        "Pitch:",
        round(
            final_info["pitch"],
            3
        )
    )

    print(
        "Collective:",
        round(
            final_info["collective"],
            4
        )
    )

    print(
        "Trim collective:",
        round(
            final_info["trim_collective"],
            4
        )
    )

    print(
        "Rudder:",
        round(
            final_info["rudder"],
            4
        )
    )

    print(
        "Trim rudder:",
        round(
            final_info["trim_rudder"],
            4
        )
    )

    print(
        "Success:",
        final_info["success"]
    )

    env.close()


def main():

    test_model(
        "models/task3_trim_v2/best/best_model",
        "DYNAMIC TRIM - BEST"
    )

    test_model(
        "models/task3_trim_v2/"
        "ppo_ah1s_task3_trim_v2_final",
        "DYNAMIC TRIM - FINAL"
    )


if __name__ == "__main__":
    main()
