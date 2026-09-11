from stable_baselines3 import PPO

from helicopter_env_4 import HelicopterEnv


MODEL_PATHS = {
    "BEST": "models/task4_hover/best/best_model",
    "FINAL": "models/task4_hover/ppo_ah1s_hover_final",
}


def test_model(name, model_path):

    print()
    print("=" * 70)
    print(f"{name} MODEL TEST")
    print("=" * 70)

    env = HelicopterEnv()

    model = PPO.load(
        model_path,
        env=env
    )

    obs, info = env.reset()

    max_altitude = info["altitude"]

    max_abs_vertical_speed = 0.0
    max_abs_forward_velocity = 0.0
    max_abs_lateral_velocity = 0.0

    max_abs_heading_error = 0.0

    max_abs_roll = 0.0
    max_abs_pitch = 0.0

    max_hold_steps = 0

    reached_850 = False
    reached_900 = False
    reached_950 = False
    reached_1000 = False

    success = False
    termination_reason = None

    final_info = info

    for step in range(5000):

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
        ) = env.step(action)

        final_info = info

        altitude = info["altitude"]

        vertical_speed = info[
            "vertical_speed"
        ]

        forward_velocity = info[
            "forward_velocity"
        ]

        lateral_velocity = info[
            "lateral_velocity"
        ]

        heading_error = info[
            "heading_error"
        ]

        roll = info["roll"]
        pitch = info["pitch"]

        hold_steps = info[
            "target_hold_steps"
        ]

        # --------------------------------------
        # MAX VALUES
        # --------------------------------------

        max_altitude = max(
            max_altitude,
            altitude
        )

        max_abs_vertical_speed = max(
            max_abs_vertical_speed,
            abs(vertical_speed)
        )

        max_abs_forward_velocity = max(
            max_abs_forward_velocity,
            abs(forward_velocity)
        )

        max_abs_lateral_velocity = max(
            max_abs_lateral_velocity,
            abs(lateral_velocity)
        )

        max_abs_heading_error = max(
            max_abs_heading_error,
            abs(heading_error)
        )

        max_abs_roll = max(
            max_abs_roll,
            abs(roll)
        )

        max_abs_pitch = max(
            max_abs_pitch,
            abs(pitch)
        )

        max_hold_steps = max(
            max_hold_steps,
            hold_steps
        )

        # --------------------------------------
        # ALTITUDE MILESTONES
        # --------------------------------------

        if altitude >= 850.0:
            reached_850 = True

        if altitude >= 900.0:
            reached_900 = True

        if altitude >= 950.0:
            reached_950 = True

        if altitude >= 1000.0:
            reached_1000 = True

        # --------------------------------------
        # LIVE OUTPUT
        # --------------------------------------

        if step % 50 == 0:

            print(
                f"Step {step:4d} | "
                f"Phase {info['phase']:7s} | "
                f"Alt {altitude:8.2f} ft | "
                f"VS {vertical_speed:7.2f} | "
                f"Fwd {forward_velocity:7.2f} | "
                f"Lat {lateral_velocity:7.2f} | "
                f"HeadErr {heading_error:7.3f} | "
                f"Roll {roll:7.3f} | "
                f"Pitch {pitch:7.3f} | "
                f"Hold {hold_steps:3d}"
            )

        # --------------------------------------
        # EPISODE END
        # --------------------------------------

        if terminated or truncated:

            success = info.get(
                "success",
                False
            )

            termination_reason = info.get(
                "termination_reason",
                None
            )

            print()
            print(
                f"Episode ended at step {step}"
            )

            print(
                "Terminated:",
                terminated
            )

            print(
                "Truncated:",
                truncated
            )

            print(
                "Termination reason:",
                termination_reason
            )

            break

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("-" * 70)
    print(f"{name} SUMMARY")
    print("-" * 70)

    print(
        f"Max altitude: "
        f"{max_altitude:.2f} ft"
    )

    print(
        f"Max |vertical speed|: "
        f"{max_abs_vertical_speed:.2f} ft/s"
    )

    print(
        f"Max |forward velocity|: "
        f"{max_abs_forward_velocity:.2f} ft/s"
    )

    print(
        f"Max |lateral velocity|: "
        f"{max_abs_lateral_velocity:.2f} ft/s"
    )

    print(
        f"Max |heading error|: "
        f"{max_abs_heading_error:.3f} rad"
    )

    print(
        f"Max |roll|: "
        f"{max_abs_roll:.3f} rad"
    )

    print(
        f"Max |pitch|: "
        f"{max_abs_pitch:.3f} rad"
    )

    print(
        f"Max hold: "
        f"{max_hold_steps}/"
        f"{env.required_hold_steps}"
    )

    print()
    print(
        "Reached 850 ft:",
        reached_850
    )

    print(
        "Reached 900 ft:",
        reached_900
    )

    print(
        "Reached 950 ft:",
        reached_950
    )

    print(
        "Reached 1000 ft:",
        reached_1000
    )

    print()
    print("FINAL STATE")

    print(
        f"Phase: "
        f"{final_info['phase']}"
    )

    print(
        f"Altitude: "
        f"{final_info['altitude']:.2f} ft"
    )

    print(
        f"Target VS: "
        f"{final_info['target_vertical_speed']:.2f} ft/s"
    )

    print(
        f"Vertical speed: "
        f"{final_info['vertical_speed']:.2f} ft/s"
    )

    print(
        f"Forward velocity: "
        f"{final_info['forward_velocity']:.2f} ft/s"
    )

    print(
        f"Lateral velocity: "
        f"{final_info['lateral_velocity']:.2f} ft/s"
    )

    print(
        f"Heading error: "
        f"{final_info['heading_error']:.3f} rad"
    )

    print(
        f"Roll: "
        f"{final_info['roll']:.3f} rad"
    )

    print(
        f"Pitch: "
        f"{final_info['pitch']:.3f} rad"
    )

    print(
        f"Hold: "
        f"{final_info['target_hold_steps']}/"
        f"{env.required_hold_steps}"
    )

    print()
    print(
        "Success:",
        success
    )

    print(
        "Termination reason:",
        termination_reason
    )

    # --------------------------------------
    # CURRENT TRIM
    # --------------------------------------

    try:

        (
            trim_collective,
            trim_elevator,
            trim_aileron,
            trim_rudder
        ) = env._get_trim_controls(
            final_info["altitude"]
        )

        print()
        print("TRIM AT FINAL ALTITUDE")

        print(
            f"Trim collective: "
            f"{trim_collective:.4f}"
        )

        print(
            f"Trim elevator: "
            f"{trim_elevator:.4f}"
        )

        print(
            f"Trim aileron: "
            f"{trim_aileron:.4f}"
        )

        print(
            f"Trim rudder: "
            f"{trim_rudder:.4f}"
        )

    except Exception as e:

        print()
        print(
            "Trim bilgisi okunamadi:",
            e
        )

    env.close()


def main():

    for name, path in MODEL_PATHS.items():

        try:

            test_model(
                name,
                path
            )

        except Exception as e:

            print()
            print("=" * 70)
            print(
                f"{name} model testinde hata:"
            )

            print(e)

            print("=" * 70)


if __name__ == "__main__":
    main()
