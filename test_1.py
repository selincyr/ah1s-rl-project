from stable_baselines3 import PPO
from helicopter_env_1 import HelicopterEnv
import matplotlib.pyplot as plt


def main():

    env = HelicopterEnv()

    model = PPO.load(
        "ppo_ah1s_task1_takeoff",
        env=env
    )

    obs, info = env.reset()

    print("\nTASK 1 TAKEOFF TEST BASLADI")
    print("----------------------------")

    total_reward = 0.0

    altitudes = []
    vertical_speeds = []
    collectives = []
    steps_list = []

    for step in range(5000):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        altitudes.append(
            info["altitude"]
        )

        vertical_speeds.append(
            info["vertical_speed"]
        )

        collectives.append(
            info["collective"]
        )

        steps_list.append(
            step
        )

        if step % 20 == 0:

            print(
                f"Step: {step:4d} | "
                f"Altitude: {info['altitude']:8.2f} ft | "
                f"VSpeed: {info['vertical_speed']:7.2f} ft/s | "
                f"Velocity: {info['forward_velocity']:7.2f} ft/s | "
                f"Pitch: {info['pitch']:6.3f} | "
                f"Roll: {info['roll']:6.3f} | "
                f"Collective: {info['collective']:.3f} | "
                f"Hold: {info['target_hold_steps']:3d}"
            )

        if terminated or truncated:

            print("\nEpisode bitti.")

            print(
                "Success:",
                info["success"]
            )

            print(
                "Son altitude:",
                info["altitude"]
            )

            print(
                "Son vertical speed:",
                info["vertical_speed"]
            )

            print(
                "Toplam reward:",
                total_reward
            )

            print(
                "Toplam step:",
                step + 1
            )

            break

    env.close()

    # ==========================================
    # ALTITUDE GRAFIGI
    # ==========================================

    plt.figure()

    plt.plot(
        steps_list,
        altitudes
    )

    plt.axhline(
        y=1000,
        linestyle="--"
    )

    plt.xlabel(
        "RL Step"
    )

    plt.ylabel(
        "Altitude AGL (ft)"
    )

    plt.title(
        "Task 1 - Altitude"
    )

    plt.grid()

    plt.show()

    # ==========================================
    # VERTICAL SPEED GRAFIGI
    # ==========================================

    plt.figure()

    plt.plot(
        steps_list,
        vertical_speeds
    )

    plt.xlabel(
        "RL Step"
    )

    plt.ylabel(
        "Vertical Speed (ft/s)"
    )

    plt.title(
        "Task 1 - Vertical Speed"
    )

    plt.grid()

    plt.show()

    # ==========================================
    # COLLECTIVE GRAFIGI
    # ==========================================

    plt.figure()

    plt.plot(
        steps_list,
        collectives
    )

    plt.xlabel(
        "RL Step"
    )

    plt.ylabel(
        "Collective"
    )

    plt.title(
        "Task 1 - Collective"
    )

    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
