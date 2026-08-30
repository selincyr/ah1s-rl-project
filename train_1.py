from stable_baselines3 import PPO
from helicopter_env_1 import HelicopterEnv


def main():

    env = HelicopterEnv()

    model = PPO.load(
        "ppo_ah1s_task1_takeoff",
        env=env
    )

    model.learn(
        total_timesteps=30_000,
        reset_num_timesteps=False
    )

    model.save(
        "ppo_ah1s_task1_takeoff"
    )

    env.close()

    print(
        "Task 1 TAKEOFF ek training tamamlandı."
    )


if __name__ == "__main__":
    main()
