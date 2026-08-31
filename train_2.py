from stable_baselines3 import PPO

from helicopter_env_2 import HelicopterEnv


def main():

    print("Task 2 - 1000 ft climb training baslatiliyor...")

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    env = HelicopterEnv()

    print("Environment olusturuldu.")
    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space.shape)

    # =====================================================
    # PPO MODEL
    # =====================================================

    model = PPO(
        "MlpPolicy",
        env,

        learning_rate=3e-4,

        gamma=0.995,

        n_steps=2048,

        batch_size=64,

        verbose=1
    )

    print("PPO model olusturuldu.")

    # =====================================================
    # TRAINING
    # =====================================================

    model.learn(
        total_timesteps=200_000
    )

    # =====================================================
    # SAVE
    # =====================================================

    model.save(
        "ppo_ah1s_1000ft_v1"
    )

    print(
        "Model kaydedildi: ppo_ah1s_1000ft_v1.zip"
    )

    env.close()

    print("Training tamamlandi.")


if __name__ == "__main__":
    main()
