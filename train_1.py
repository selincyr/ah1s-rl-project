from stable_baselines3 import PPO
from helicopter_env_1 import HelicopterEnv


def main():

    env = HelicopterEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=64
    )

    model.learn(
        total_timesteps=50_000
    )

    model.save(
        "ppo_ah1s_task1_rates"
    )

    env.close()

    print(
        "Task 1 rate-aware training tamamlandı."
    )


if __name__ == "__main__":
    main()
