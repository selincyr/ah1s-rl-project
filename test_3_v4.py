from stable_baselines3 import PPO

from helicopter_env_3_v4 import HelicopterEnv


def test_model(model_path, model_name):

    print()
    print("=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    env = HelicopterEnv()

    model = PPO.load(
        model_path,
        env=env
    )

    obs, info = env.reset()

    max_altitude = info["altitude"]
    max_vertical_speed = abs(info["vertical_speed"])
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

    first_800_step = None
    first_900_step = None
    first_970_step = None
    first_1000_step = None

    ended_step = None
    terminated_value = False
    truncated_value = False

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

        max_vertical_speed = max(
            max_vertical_speed,
            abs(info["vertical_speed"])
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

        # ------------------------------------------
        # MILESTONES
        # ------------------------------------------

        if altitude >= 15.0:
            reached_takeoff = True

        if altitude >= 30.0:
            reached_climb = True

        if altitude >= 800.0:
            reached_approach = True

            if first_800_step is None:
                first_800_step = step

        if altitude >= 900.0:

            if first_900_step is None:
                first_900_step = step

        if altitude >= 970.0:
            reached_forward = True

            if first_970_step is None:
                first_970_step = step

        if altitude >= 1000.0:

            if first_1000_step is None:
                first_1000_step = step

        # ------------------------------------------
        # PRINT EVERY 50 STEPS
        # ------------------------------------------

        if step % 50 == 0:

            print(
                f"Step {step:4d} | "
                f"{info['phase']:8s} | "
                f"Alt {info['altitude']:8.2f} | "
                f"TargetVS {info['target_vertical_speed']:5.2f} | "
                f"VS {info['vertical_speed']:7.2f} | "
                f"Fwd {info['forward_velocity']:7.2f} | "
                f"Lat {info['lateral_velocity']:7.2f} | "
                f"HeadErr {info['heading_error']:7.3f} | "
                f"Roll {info['roll']:7.3f} | "
                f"Pitch {info['pitch']:7.3f} | "
                f"Hold {info['target_hold_steps']:3d}"
            )

        # ------------------------------------------
        # EPISODE END
        # ------------------------------------------

        if terminated or truncated:

            ended_step = step
            terminated_value = terminated
            truncated_value = truncated

            print()
            print(
                "Episode ended at step:",
                step
            )

            print(
                "terminated:",
                terminated
            )

            print(
                "truncated:",
                truncated
            )

            break

    # ==============================================
    # FINAL TRIM
    # ==============================================

    (
        trim_collective,
        trim_elevator,
        trim_aileron,
        trim_rudder
    ) = env._get_trim_controls(
        final_info["altitude"]
    )

    # ==============================================
    # SUMMARY
    # ==============================================

    print()
    print("=" * 70)
    print("TEST OZETI")
    print("=" * 70)

    print(
        "Model:",
        model_name
    )

    print(
        "Episode end step:",
        ended_step
    )

    print(
        "Terminated:",
        terminated_value
    )

    print(
        "Truncated:",
        truncated_value
    )

    print()

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

    # ==============================================
    # PHASES
    # ==============================================

    print()
    print("PHASES")
    print("-" * 40)

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

    # ==============================================
    # MILESTONES
    # ==============================================

    print()
    print("MILESTONES")
    print("-" * 40)

    print(
        "800 ft step:",
        first_800_step
    )

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

    # ==============================================
    # FINAL STATE
    # ==============================================

    print()
    print("FINAL STATE")
    print("-" * 40)

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
        "Target vertical speed:",
        round(
            final_info["target_vertical_speed"],
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
        "Target forward velocity:",
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

    print()

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
            trim_collective,
            4
        )
    )

    print(
        "Elevator:",
        round(
            final_info["elevator"],
            4
        )
    )

    print(
        "Trim elevator:",
        round(
            trim_elevator,
            4
        )
    )

    print(
        "Aileron:",
        round(
            final_info["aileron"],
            4
        )
    )

    print(
        "Trim aileron:",
        round(
            trim_aileron,
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
            trim_rudder,
            4
        )
    )

    print()

    print(
        "Success:",
        final_info["success"]
    )

    env.close()


def main():

    test_model(
        "models/task3_trim_v4/best/best_model",
        "V4 - BEST MODEL"
    )

    test_model(
        "models/task3_trim_v4/"
        "ppo_ah1s_task3_trim_v4_final",
        "V4 - FINAL MODEL"
    )


if __name__ == "__main__":
    main()
