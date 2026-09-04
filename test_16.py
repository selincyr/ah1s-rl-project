import os
import numpy as np

from stable_baselines3 import PPO

from helicopter_env_16 import HelicopterEnv


MODEL_DIR = "models/task16a_vertical300"

BEST_MODEL_PATH = f"{MODEL_DIR}/best/best_model"
FINAL_MODEL_PATH = (
    f"{MODEL_DIR}/ppo_ah1s_vertical300_v16a_final"
)

MAX_STEPS = 4000


def evaluate_model(model_path, label):
    print()
    print("=" * 90)
    print(label)
    print("=" * 90)

    if not (
        os.path.exists(model_path)
        or os.path.exists(model_path + ".zip")
    ):
        print("MODEL NOT FOUND:")
        print(model_path)
        return

    env = HelicopterEnv()
    obs, info = env.reset(seed=42)

    model = PPO.load(
        model_path,
        env=env,
        device="auto",
    )

    max_altitude = float(info["altitude"])
    max_horizontal_distance = float(
        info["horizontal_distance"]
    )

    initial_horizontal_speed = np.hypot(
        float(info["forward_velocity"]),
        float(info["lateral_velocity"]),
    )

    max_horizontal_speed = float(
        initial_horizontal_speed
    )

    initial_vs_error = abs(
        float(info["vertical_speed"])
        - float(info["target_vertical_speed"])
    )

    max_vs_tracking_error = float(
        initial_vs_error
    )

    max_hold = int(
        info.get("hold_steps", 0)
    )

    best_snapshot = None
    best_snapshot_score = float("inf")

    reached_30 = False
    reached_100 = False
    reached_200 = False
    reached_280 = False
    reached_300 = False

    total_reward = 0.0

    terminated = False
    truncated = False

    step = 0

    while (
        not terminated
        and not truncated
        and step < MAX_STEPS
    ):
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        step += 1
        total_reward += float(reward)

        altitude = float(info["altitude"])

        vertical_speed = float(
            info["vertical_speed"]
        )

        target_vs = float(
            info["target_vertical_speed"]
        )

        forward_velocity = float(
            info["forward_velocity"]
        )

        lateral_velocity = float(
            info["lateral_velocity"]
        )

        heading_error = float(
            info["heading_error"]
        )

        roll = float(info["roll"])
        pitch = float(info["pitch"])

        horizontal_distance = float(
            info["horizontal_distance"]
        )

        horizontal_speed = float(
            np.hypot(
                forward_velocity,
                lateral_velocity,
            )
        )

        vs_tracking_error = abs(
            vertical_speed - target_vs
        )

        hold_steps = int(
            info.get("hold_steps", 0)
        )

        max_altitude = max(
            max_altitude,
            altitude,
        )

        max_horizontal_distance = max(
            max_horizontal_distance,
            horizontal_distance,
        )

        max_horizontal_speed = max(
            max_horizontal_speed,
            horizontal_speed,
        )

        max_vs_tracking_error = max(
            max_vs_tracking_error,
            vs_tracking_error,
        )

        max_hold = max(
            max_hold,
            hold_steps,
        )

        reached_30 |= altitude >= 30.0
        reached_100 |= altitude >= 100.0
        reached_200 |= altitude >= 200.0
        reached_280 |= altitude >= 280.0
        reached_300 |= altitude >= 300.0

        # Best physically useful snapshot around target altitude.
        if 280.0 <= altitude <= 320.0:
            snapshot_score = (
                abs(altitude - 300.0) / 20.0
                + abs(vertical_speed) / 5.0
                + horizontal_distance / 20.0
                + horizontal_speed / 10.0
                + abs(heading_error) / 0.30
            )

            if snapshot_score < best_snapshot_score:
                best_snapshot_score = snapshot_score

                best_snapshot = {
                    "step": step,
                    "altitude": altitude,
                    "vertical_speed": vertical_speed,
                    "target_vs": target_vs,
                    "forward_velocity": forward_velocity,
                    "lateral_velocity": lateral_velocity,
                    "horizontal_speed": horizontal_speed,
                    "horizontal_distance": horizontal_distance,
                    "east": float(
                        info["relative_east"]
                    ),
                    "north": float(
                        info["relative_north"]
                    ),
                    "heading_error": heading_error,
                    "roll": roll,
                    "pitch": pitch,
                    "hold_steps": hold_steps,
                }

        if step % 25 == 0:
            print(
                f"step={step:4d} | "
                f"phase={info['phase']:7s} | "
                f"alt={altitude:7.2f} | "
                f"VS={vertical_speed:6.2f} | "
                f"targetVS={target_vs:5.2f} | "
                f"fwd={forward_velocity:7.2f} | "
                f"lat={lateral_velocity:7.2f} | "
                f"dist={horizontal_distance:7.2f} | "
                f"hold={hold_steps:3d}"
            )

    overshoot = max(
        0.0,
        max_altitude - 300.0,
    )

    print()
    print("-" * 90)
    print("PHYSICAL BENCHMARK SUMMARY")
    print("-" * 90)

    print(f"Steps: {step}")
    print(f"Total reward: {total_reward:.2f}")

    print(
        f"Max altitude: "
        f"{max_altitude:.2f} ft"
    )

    print(
        f"Altitude overshoot: "
        f"{overshoot:.2f} ft"
    )

    print(
        f"Max horizontal distance: "
        f"{max_horizontal_distance:.2f} ft"
    )

    print(
        f"Max horizontal speed: "
        f"{max_horizontal_speed:.2f} ft/s"
    )

    print(
        f"Max VS tracking error: "
        f"{max_vs_tracking_error:.2f} ft/s"
    )

    print(
        f"Max hold: "
        f"{max_hold}/100"
    )

    print()
    print("ALTITUDE MILESTONES")
    print(f"Reached 30 ft : {reached_30}")
    print(f"Reached 100 ft: {reached_100}")
    print(f"Reached 200 ft: {reached_200}")
    print(f"Reached 280 ft: {reached_280}")
    print(f"Reached 300 ft: {reached_300}")

    print()
    print("BEST 280-320 FT SNAPSHOT")

    if best_snapshot is None:
        print(
            "No sample entered the "
            "280-320 ft region."
        )
    else:
        for key, value in best_snapshot.items():
            if isinstance(value, float):
                print(
                    f"{key}: {value:.3f}"
                )
            else:
                print(
                    f"{key}: {value}"
                )

    print()
    print("FINAL STATE")

    print(
        f"Altitude: "
        f"{float(info['altitude']):.2f} ft"
    )

    print(
        f"Horizontal distance: "
        f"{float(info['horizontal_distance']):.2f} ft"
    )

    print(
        f"Hold: "
        f"{int(info.get('hold_steps', 0))}/100"
    )

    print(
        "Success:",
        bool(info.get("success", False)),
    )

    print(
        "Termination reason:",
        info.get(
            "termination_reason",
            None,
        ),
    )

    env.close()


def main():
    evaluate_model(
        BEST_MODEL_PATH,
        "V16-A BEST MODEL",
    )

    evaluate_model(
        FINAL_MODEL_PATH,
        "V16-A FINAL MODEL",
    )


if __name__ == "__main__":
    main()
