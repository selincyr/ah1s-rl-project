from stable_baselines3 import PPO

from helicopter_env_3 import HelicopterEnv


def test_model(model_path, model_name):

    print()
    print("=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

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
    max_hold = 0

    reached_takeoff = False
    reached_climb = False
    reached_approach = False
    reached_forward = False

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

        if altitude >= 970.0:
            reached_forward = True

        if step % 200 == 0:

            print(
                f"Step {step:4d} | "
                f"Phase {info['phase']:8s} | "
                f"Alt {altitude:8.2f} | "
                f"VSpeed {info['vertical_speed']:7.2f} | "
                f"Fwd {info['forward_velocity']:7.2f} | "
                f"Lat {info['lateral_velocity']:7.2f} | "
                f"HeadingErr {info['heading_error']:6.3f} | "
                f"Hold {info['target_hold_steps']:3d}"
            )

        if terminated or truncated:
            break

    print()
    print("============= TEST OZETI =============")

    print("Model:", model_name)

    print(
        "Max altitude:",
        round(max_altitude, 2)
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
        "Forward phase reached:",
        reached_forward
    )

    print()
    print("FINAL")

    print(
        "Final phase:",
        final_info["phase"]
    )

    print(
        "Final altitude:",
        round(
            final_info["altitude"],
            2
        )
    )

    print(
        "Final vertical speed:",
        round(
            final_info["vertical_speed"],
            2
        )
    )

    print(
        "Final forward velocity:",
        round(
            final_info["forward_velocity"],
            2
        )
    )

    print(
        "Final lateral velocity:",
        round(
            final_info["lateral_velocity"],
            2
        )
    )

    print(
        "Final heading error:",
        round(
            final_info["heading_error"],
            3
        )
    )

    print(
        "Final roll:",
        round(
            final_info["roll"],
            3
        )
    )

    print(
        "Final pitch:",
        round(
            final_info["pitch"],
            3
        )
    )

    print(
        "Success:",
        final_info["success"]
    )

    print("=" * 60)

    env.close()


def main():

    test_model(
        "models/task3/best/best_model",
        "BEST MODEL"
    )

    test_model(
        "models/task3/ppo_ah1s_task3_final",
        "FINAL MODEL"
    )


if __name__ == "__main__":
    main()
