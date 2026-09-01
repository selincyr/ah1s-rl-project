from stable_baselines3 import PPO

from helicopter_env_2 import HelicopterEnv


def main():

    print("Task 2 - 1000 ft climb test baslatiliyor...")

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    env = HelicopterEnv()

    # =====================================================
    # MODEL
    # =====================================================

    model = PPO.load(
        "ppo_ah1s_1000ft_v3",
        env=env
    )

    print("Model yuklendi.")

    # =====================================================
    # RESET
    # =====================================================

    obs, info = env.reset()

    # =====================================================
    # METRICS
    # =====================================================

    max_altitude = info["altitude"]

    max_vertical_speed = abs(
        info["vertical_speed"]
    )

    max_forward_velocity = abs(
        info["forward_velocity"]
    )

    max_lateral_velocity = abs(
        info["lateral_velocity"]
    )

    max_heading_error = abs(
        info["heading_error"]
    )

    max_roll = abs(
        info["roll"]
    )

    max_pitch = abs(
        info["pitch"]
    )

    max_hold = 0

    reached_takeoff = False
    reached_climb = False
    reached_approach = False
    reached_hover = False

    final_info = info

    # =====================================================
    # TEST LOOP
    # =====================================================

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
        vertical_speed = info["vertical_speed"]

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

        phase = info["phase"]

        hold = info[
            "target_hold_steps"
        ]

        # =================================================
        # METRIC UPDATE
        # =================================================

        max_altitude = max(
            max_altitude,
            altitude
        )

        max_vertical_speed = max(
            max_vertical_speed,
            abs(
                vertical_speed
            )
        )

        max_forward_velocity = max(
            max_forward_velocity,
            abs(
                forward_velocity
            )
        )

        max_lateral_velocity = max(
            max_lateral_velocity,
            abs(
                lateral_velocity
            )
        )

        max_heading_error = max(
            max_heading_error,
            abs(
                heading_error
            )
        )

        max_roll = max(
            max_roll,
            abs(
                roll
            )
        )

        max_pitch = max(
            max_pitch,
            abs(
                pitch
            )
        )

        max_hold = max(
            max_hold,
            hold
        )

        # =================================================
        # PHASE TRACKING
        # =================================================

        if altitude >= 15.0:
            reached_takeoff = True

        if altitude >= 30.0:
            reached_climb = True

        if altitude >= 850.0:
            reached_approach = True

        if altitude >= 970.0:
            reached_hover = True

        # =================================================
        # LIVE OUTPUT
        # =================================================

        if step % 100 == 0:

            print(
                f"Step: {step:4d} | "
                f"Phase: {phase:8s} | "
                f"Alt: {altitude:8.2f} ft | "
                f"VSpeed: {vertical_speed:7.2f} ft/s | "
                f"Target VS: "
                f"{info['target_vertical_speed']:6.2f} | "
                f"Fwd: {forward_velocity:7.2f} | "
                f"Lat: {lateral_velocity:7.2f} | "
                f"Heading err: {heading_error:6.3f} | "
                f"Hold: {hold:2d}/50"
            )

        # =================================================
        # END
        # =================================================

        if terminated or truncated:

            print()
            print(
                "Episode bitti. "
                f"Step: {step}"
            )

            break

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("============= TEST OZETI =============")

    print(
        f"Max altitude: "
        f"{max_altitude:.2f} ft"
    )

    print(
        f"Max |vertical speed|: "
        f"{max_vertical_speed:.2f} ft/s"
    )

    print(
        f"Max |forward velocity|: "
        f"{max_forward_velocity:.2f} ft/s"
    )

    print(
        f"Max |lateral velocity|: "
        f"{max_lateral_velocity:.2f} ft/s"
    )

    print(
        f"Max |heading error|: "
        f"{max_heading_error:.3f} rad"
    )

    print(
        f"Max |roll|: "
        f"{max_roll:.3f} rad"
    )

    print(
        f"Max |pitch|: "
        f"{max_pitch:.3f} rad"
    )

    print(
        f"Max hold: "
        f"{max_hold} / 50"
    )

    print()
    print("------------- PHASES ----------------")

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
        "Hover zone reached:",
        reached_hover
    )

    print()
    print("------------- FINAL -----------------")

    print(
        f"Final phase: "
        f"{final_info['phase']}"
    )

    print(
        f"Final altitude: "
        f"{final_info['altitude']:.2f} ft"
    )

    print(
        f"Final vertical speed: "
        f"{final_info['vertical_speed']:.2f} ft/s"
    )

    print(
        f"Final target vertical speed: "
        f"{final_info['target_vertical_speed']:.2f} ft/s"
    )

    print(
        f"Final forward velocity: "
        f"{final_info['forward_velocity']:.2f} ft/s"
    )

    print(
        f"Final lateral velocity: "
        f"{final_info['lateral_velocity']:.2f} ft/s"
    )

    print(
        f"Final heading error: "
        f"{final_info['heading_error']:.3f} rad"
    )

    print(
        f"Final roll: "
        f"{final_info['roll']:.3f} rad"
    )

    print(
        f"Final pitch: "
        f"{final_info['pitch']:.3f} rad"
    )

    print(
        f"Final rotor RPM: "
        f"{final_info['rotor_rpm']:.2f}"
    )

    print(
        f"Final collective: "
        f"{final_info['collective']:.3f}"
    )

    print(
        f"Success: "
        f"{final_info['success']}"
    )

    print("======================================")

    env.close()


if __name__ == "__main__":
    main()
