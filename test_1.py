from stable_baselines3 import PPO
from helicopter_env_1 import HelicopterEnv


def main():

    print("Takeoff-straight test baslatiliyor...")

    env = HelicopterEnv()

    model = PPO.load(
        "ppo_ah1s_takeoff_v2",
        env=env
    )

    obs, info = env.reset()

    print("Model yuklendi.")
    print("Test basladi.")
    print()

    max_altitude = info["altitude"]

    max_abs_x_velocity = 0.0
    max_abs_y_velocity = 0.0

    max_hold = 0

    for step in range(1000):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        altitude = info["altitude"]

        forward_velocity = info[
            "forward_velocity"
        ]

        lateral_velocity = info[
            "lateral_velocity"
        ]

        vertical_speed = info[
            "vertical_speed"
        ]

        roll = info["roll"]

        pitch = info["pitch"]

        heading_error = info[
            "heading_error"
        ]

        roll_rate = info[
            "roll_rate"
        ]

        pitch_rate = info[
            "pitch_rate"
        ]

        hold = info[
            "target_hold_steps"
        ]

        max_altitude = max(
            max_altitude,
            altitude
        )

        max_abs_x_velocity = max(
            max_abs_x_velocity,
            abs(forward_velocity)
        )

        max_abs_y_velocity = max(
            max_abs_y_velocity,
            abs(lateral_velocity)
        )

        max_hold = max(
            max_hold,
            hold
        )

        # Her 20 stepte bir yazdır.
        if step % 20 == 0:

            print(
                f"Step: {step:4d} | "
                f"Alt: {altitude:6.2f} ft | "
                f"VSpeed: {vertical_speed:6.2f} | "
                f"Forward: {forward_velocity:6.2f} | "
                f"Lateral: {lateral_velocity:6.2f} | "
                f"Roll: {roll:6.3f} | "
                f"Pitch: {pitch:6.3f} | "
                f"HeadErr: {heading_error:6.3f} | "
                f"Hold: {hold:3d}"
            )

        if terminated or truncated:

            print()
            print("Episode bitti.")

            print(
                "Success:",
                info["success"]
            )

            break

    print()
    print("============= TEST OZETI =============")

    print(
        f"Max altitude: {max_altitude:.2f} ft"
    )

    print(
        f"Max |Forward velocity|: "
        f"{max_abs_x_velocity:.2f} ft/s"
    )

    print(
        f"Max |Lateral velocity|: "
        f"{max_abs_y_velocity:.2f} ft/s"
    )

    print(
        f"Max hold: {max_hold} / 50"
    )

    print(
        f"Final altitude: "
        f"{info['altitude']:.2f} ft"
    )

    print(
        f"Final forward velocity: "
        f"{info['forward_velocity']:.2f} ft/s"
    )

    print(
        f"Final lateral velocity: "
        f"{info['lateral_velocity']:.2f} ft/s"
    )

    print(
        f"Final vertical speed: "
        f"{info['vertical_speed']:.2f} ft/s"
    )

    print(
        f"Final roll: "
        f"{info['roll']:.3f} rad"
    )

    print(
        f"Final pitch: "
        f"{info['pitch']:.3f} rad"
    )

    print(
        f"Final heading error: "
        f"{info['heading_error']:.3f} rad"
    )

    print(
        f"Success: "
        f"{info['success']}"
    )

    print("======================================")

    env.close()


if __name__ == "__main__":
    main()
