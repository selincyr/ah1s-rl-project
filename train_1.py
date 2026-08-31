from stable_baselines3 import PPO

from helicopter_env_1 import HelicopterEnv


def main():

    print(
        "Takeoff-hover training baslatiliyor..."
    )

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    env = HelicopterEnv()

    print(
        "Environment olusturuldu."
    )

    print(
        "Observation space:",
        env.observation_space.shape
    )

    print(
        "Action space:",
        env.action_space.shape
    )

    # =====================================================
    # PPO MODEL
    # =====================================================

    model = PPO(
        "MlpPolicy",
        env,

        learning_rate=3e-4,

        gamma=0.99,

        n_steps=1024,

        batch_size=64,

        verbose=1
    )

    print(
        "PPO model olusturuldu."
    )

    # =====================================================
    # TRAINING
    # =====================================================

    model.learn(
        total_timesteps=50_000
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model.save(
        "ppo_ah1s_takeoff_hover"
    )

    env.close()

    print()
    print(
        "Training tamamlandi."
    )

    print(
        "Model kaydedildi:"
    )

    print(
        "ppo_ah1s_takeoff_hover.zip"
    )


if __name__ == "__main__":
    main()
