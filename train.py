
from stable_baselines3 import PPO
from helicopter_env import HelicopterEnv


def main():
    env = HelicopterEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99
    )

    model.learn(total_timesteps=10_000)

    model.save("ppo_ah1s")

    print("Training tamamlandı")


if __name__ == "__main__":
    main()
