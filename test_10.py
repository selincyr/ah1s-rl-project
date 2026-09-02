import numpy as np

from stable_baselines3 import PPO
from helicopter_env_10 import HelicopterEnv


MODEL_PATHS = [
    (
        "BEST",
        "models/task10_hover300/best/best_model"
    ),
    (
        "FINAL",
        "models/task10_hover300/ppo_ah1s_hover300_v10_final"
    )
]


def test_model(model_path, label):

    print()
    print("=" * 90)
    print(f"{label} MODEL TEST")
    print("=" * 90)

    env = HelicopterEnv()

    model = PPO.load(
        model_path
    )

    obs, info = env.reset()

    max_altitude = -np.inf
    max_abs_vertical_speed = 0.0
    max_abs_forward_velocity = 0.0
    max_abs_lateral_velocity = 0.0
    max_abs_heading_error = 0.0
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    max_hold = 0

    collective_actions = []
    elevator_actions = []
    aileron_actions = []
    rudder_actions = []

    reached_200 = False
    reached_230 = False
    reached_260 = False
    reached_280 = False
    reached_285 = False
    reached_295 = False
    reached_300 = False

    final_info = info

    for step in range(1, 5001):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        final_info = info

        altitude = float(info["altitude"])
        vertical_speed = float(info["vertical_speed"])
        forward_velocity = float(info["forward_velocity"])
        lateral_velocity = float(info["lateral_velocity"])
        heading_error = float(info["heading_error"])
        roll = float(info["roll"])
        pitch = float(info["pitch"])

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

        max_hold = max(
            max_hold,
            int(info["target_hold_steps"])
        )

        collective_actions.append(
            abs(float(action[0]))
        )

        elevator_actions.append(
            abs(float(action[1]))
        )

        aileron_actions.append(
            abs(float(action[2]))
        )

        rudder_actions.append(
            abs(float(action[3]))
        )

        if altitude >= 200.0:
            reached_200 = True

        if altitude >= 230.0:
            reached_230 = True

        if altitude >= 260.0:
            reached_260 = True

        if altitude >= 280.0:
            reached_280 = True

        if altitude >= 285.0:
            reached_285 = True

        if altitude >= 295.0:
            reached_295 = True

        if altitude >= 300.0:
            reached_300 = True

        if step % 25 == 0:

            print(
                f"Step {step:4d} | "
                f"Phase {info['phase']:<7} | "
                f"Alt {altitude:7.2f} | "
                f"VS {vertical_speed:7.2f} | "
                f"TgtVS {info['target_vertical_speed']:5.2f} | "
                f"Fwd {forward_velocity:7.2f} | "
                f"Lat {lateral_velocity:7.2f} | "
                f"Head {heading_error:7.3f} | "
                f"Roll {roll:7.3f} | "
                f"Pitch {pitch:7.3f} | "
                f"Hold {info['target_hold_steps']:3d} | "
                f"A0 {float(action[0]):+.2f} | "
                f"A1 {float(action[1]):+.2f} | "
                f"A2 {float(action[2]):+.2f} | "
                f"A3 {float(action[3]):+.2f}"
            )

        if terminated or truncated:

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
                info.get("termination_reason")
            )

            break

    print()
    print("-" * 90)
    print(f"{label} SUMMARY")
    print("-" * 90)

    print(
        f"Max altitude: {max_altitude:.2f} ft"
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
        f"{max_hold}/100"
    )

    print()
    print("ALTITUDE MILESTONES")

    print(
        "Reached 200 ft:",
        reached_200
    )

    print(
        "Reached 230 ft:",
        reached_230
    )

    print(
        "Reached 260 ft:",
        reached_260
    )

    print(
        "Reached 280 ft:",
        reached_280
    )

    print(
        "Reached 285 ft:",
        reached_285
    )

    print(
        "Reached 295 ft:",
        reached_295
    )

    print(
        "Reached 300 ft:",
        reached_300
    )

    print()
    print("ACTION SUMMARY")

    print(
        "Collective | "
        f"Mean |action| = "
        f"{np.mean(collective_actions):.3f} | "
        f"Max |action| = "
        f"{np.max(collective_actions):.3f}"
    )

    print(
        "Elevator   | "
        f"Mean |action| = "
        f"{np.mean(elevator_actions):.3f} | "
        f"Max |action| = "
        f"{np.max(elevator_actions):.3f}"
    )

    print(
        "Aileron    | "
        f"Mean |action| = "
        f"{np.mean(aileron_actions):.3f} | "
        f"Max |action| = "
        f"{np.max(aileron_actions):.3f}"
    )

    print(
        "Rudder     | "
        f"Mean |action| = "
        f"{np.mean(rudder_actions):.3f} | "
        f"Max |action| = "
        f"{np.max(rudder_actions):.3f}"
    )

    print()
    print("FINAL STATE")

    print(
        "Phase:",
        final_info["phase"]
    )

    print(
        "Altitude:",
        f"{final_info['altitude']:.2f} ft"
    )

    print(
        "Target VS:",
        f"{final_info['target_vertical_speed']:.2f} ft/s"
    )

    print(
        "Vertical speed:",
        f"{final_info['vertical_speed']:.2f} ft/s"
    )

    print(
        "Forward velocity:",
        f"{final_info['forward_velocity']:.2f} ft/s"
    )

    print(
        "Lateral velocity:",
        f"{final_info['lateral_velocity']:.2f} ft/s"
    )

    print(
        "Heading error:",
        f"{final_info['heading_error']:.3f} rad"
    )

    print(
        "Roll:",
        f"{final_info['roll']:.3f} rad"
    )

    print(
        "Pitch:",
        f"{final_info['pitch']:.3f} rad"
    )

    print(
        "Hold:",
        f"{final_info['target_hold_steps']}/100"
    )

    print()
    print(
        "Success:",
        final_info["success"]
    )

    print(
        "Termination reason:",
        final_info.get("termination_reason")
    )

    env.close()


def main():

    for label, model_path in MODEL_PATHS:

        test_model(
            model_path,
            label
        )


if __name__ == "__main__":
    main()
