import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from helicopter_env_15 import HelicopterEnv


MODEL_DIR = "models/task15_vertical300"
LOG_DIR = "logs/task15_vertical300"

TOTAL_TIMESTEPS = 400_000


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/best", exist_ok=True)

    train_env = Monitor(HelicopterEnv())
    eval_env = Monitor(HelicopterEnv())

    # Observation is still 15-D, but V13 learned the wrong local optimum
    # (stay on the ground). Start V14 from scratch with corrected shaping.
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_ah1s_vertical300_v15",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/best",
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    print("=" * 80)
    print("V15 TRAINING FROM SCRATCH")
    print("Goal: liftoff first, then progressively tighten the X/Y corridor")
    print("Observation size: 15")
    print("Total timesteps:", TOTAL_TIMESTEPS)
    print("=" * 80)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_model_path = f"{MODEL_DIR}/ppo_ah1s_vertical300_v15_final"
    model.save(final_model_path)

    train_env.close()
    eval_env.close()

    print("=" * 80)
    print("V15 TRAINING COMPLETED")
    print("Best :", f"{MODEL_DIR}/best/best_model")
    print("Final:", final_model_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
