import numpy as np
from stable_baselines3 import PPO

from helicopter_env_5 import HelicopterEnv


MODEL_PATHS = {
    "BEST": "models/task5_hover/best/best_model",
    "FINAL": "models/task5_hover/ppo_ah1s_hover_v5_final",
}


def test_model(name, model_path):

    print()
    print("=" * 90)
    print(f"{name} MODEL TEST")
    print("=" * 90)

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

    reached_500 = False
    reached_700 = False
    reached_850 = False
    reached_900 = False
    reached_950 = False
    reached_1000 = False

    action_abs_sum = np.zeros(4, dtype=np.float64)
    action_abs_max = np.zeros(4, dtype=np.float64)

    steps_run = 0

    final_info = info

    success = False
    termination_reason = None

    for step in range(5000):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action_abs_sum += np.abs(action)

        action_abs_max = np.maximum(
            action_abs_max,
            np.abs(action)
        )

        steps_run += 1

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

        # ==================================================
        # MAX VALUES
        # ==================================================

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

        # ==================================================
        # ALTITUDE MILESTONES
        # ==================================================

        if altitude >= 500.0:
            reached_500 = True

        if altitude >= 700.0:
            reached_700 = True

        if altitude >= 850.0:
            reached_850 = True

        if altitude >= 900.0:
            reached_900 = True

        if altitude >= 950.0:
            reached_950 = True

        if altitude >= 1000.0:
            reached_1000 = True

        # ==================================================
        # LIVE OUTPUT
        # ==================================================

        if step % 50 == 0:

            print(
                f"Step {step:4d} | "
                f"Phase {info['phase']:7s} | "
                f"Alt {altitude:8.2f} | "
                f"VS {vertical_speed:7.2f} | "
                f"Fwd {forward_velocity:7.2f} | "
                f"Lat {lateral_velocity:7.2f} | "
                f"HeadErr {heading_error:7.3f} | "
                f"Roll {roll:7.3f} | "
                f"Pitch {pitch:7.3f} | "
                f"A0 {action[0]:+.2f} | "
                f"A2 {action[2]:+.2f} | "
                f"Hold {hold_steps:3d}"
            )

        # ==================================================
        # END
        # ==================================================

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
    # ACTION SUMMARY
    # ======================================================

    mean_abs_action = (
        action_abs_sum / max(steps_run, 1)
    )

    action_names = [
        "Collective",
        "Elevator",
        "Aileron",
        "Rudder"
    ]

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("-" * 90)
    print(f"{name} SUMMARY")
    print("-" * 90)

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
    print("ALTITUDE MILESTONES")

    print(
        "Reached 500 ft:",
        reached_500
    )

    print(
        "Reached 700 ft:",
        reached_700
    )

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
    print("ACTION SUMMARY")

    for i, action_name in enumerate(
        action_names
    ):

        print(
            f"{action_name:10s} | "
            f"Mean |action| = "
            f"{mean_abs_action[i]:.3f} | "
            f"Max |action| = "
            f"{action_abs_max[i]:.3f}"
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
        f"Heading sin: "
        f"{final_info['heading_error_sin']:.3f}"
    )

    print(
        f"Heading cos: "
        f"{final_info['heading_error_cos']:.3f}"
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
            print("=" * 90)

            print(
                f"{name} model testinde hata:"
            )

            print(e)

            print("=" * 90)


if __name__ == "__main__":
    main()
