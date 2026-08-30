from stable_baselines3 import PPO
from helicopter_env_1 import HelicopterEnv


def main():

    env = HelicopterEnv()

    model = PPO.load(
        "ppo_ah1s_task1",
        env=env
    )

    obs, info = env.reset()

    print("\nTASK 1 TEST BASLADI")
    print("--------------------")

    total_reward = 0

    for step in range(2000):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        # Her 20 stepte bir bilgi yaz
        if step % 20 == 0:

            print(
                f"Step: {step:4d} | "
                f"Phase: {info['phase']:8s} | "
                f"Altitude: {info['altitude']:8.2f} ft | "
                f"Velocity: {info['forward_velocity']:7.2f} ft/s | "
                f"VSpeed: {info['vertical_speed']:7.2f} ft/s | "
                f"Pitch: {info['pitch']:6.3f} | "
                f"Roll: {info['roll']:6.3f}"
            )

        if terminated or truncated:

            print("\nEpisode bitti.")

            print(
                "Son phase:",
                info["phase"]
            )

            print(
                "Son altitude:",
                info["altitude"]
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


if __name__ == "__main__":
    main()
