from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from helicopter_env_v2 import HelicopterEnvV2


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models_v2")
LOG_DIR = Path("logs_v2")
BEST_MODEL_DIR = MODEL_DIR / "best"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
BEST_MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# ENVIRONMENTS
# ============================================================

print("Training environment oluşturuluyor...")

train_env = HelicopterEnvV2()
train_env = Monitor(
    train_env,
    filename=str(LOG_DIR / "train_monitor.csv")
)

print("Evaluation environment oluşturuluyor...")

eval_env = HelicopterEnvV2()
eval_env = Monitor(
    eval_env,
    filename=str(LOG_DIR / "eval_monitor.csv")
)


# ============================================================
# CALLBACKS
# ============================================================

# Her 10k step'te ara model kaydet
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=str(MODEL_DIR),
    name_prefix="ppo_ah1s_stage1_checkpoint"
)

# Belirli aralıklarla deterministic evaluation
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(BEST_MODEL_DIR),
    log_path=str(LOG_DIR / "eval"),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

callbacks = CallbackList(
    [
        checkpoint_callback,
        eval_callback,
    ]
)


# ============================================================
# PPO MODEL
# ============================================================

print("\nPPO modeli oluşturuluyor...")

model = PPO(
    policy="MlpPolicy",

    env=train_env,

    # Öğrenme
    learning_rate=3e-4,

    # Rollout
    n_steps=2048,
    batch_size=64,
    n_epochs=10,

    # RL
    gamma=0.995,
    gae_lambda=0.95,

    # PPO clipping
    clip_range=0.2,

    # Biraz exploration
    ent_coef=0.005,

    vf_coef=0.5,

    max_grad_norm=0.5,

    # Ağ yapısı
    policy_kwargs=dict(
        net_arch=dict(
            pi=[128, 128],
            vf=[128, 128]
        )
    ),

    # Debug / logs
    verbose=1,

    tensorboard_log=str(LOG_DIR / "tensorboard"),

    seed=42,

    device="auto",
)


# ============================================================
# TRAIN
# ============================================================

TOTAL_TIMESTEPS = 100_000

print("\n" + "=" * 70)
print("AH-1S PPO STAGE 1 TRAINING")
print("=" * 70)

print("Görev:")
print("Motor hazır")
print("-> kalkış")
print("-> 300 ft")
print("-> 10 saniye stabil hover")

print("\nTotal timesteps:", TOTAL_TIMESTEPS)

print("=" * 70 + "\n")


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    progress_bar=True,
    tb_log_name="AH1S_STAGE1_V2"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_path = MODEL_DIR / "ppo_ah1s_stage1_v2_final"

model.save(
    str(final_path)
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Final model:",
    str(final_path) + ".zip"
)

print(
    "Best model:",
    str(BEST_MODEL_DIR / "best_model.zip")
)

print("=" * 70)


# ============================================================
# CLEANUP
# ============================================================

train_env.close()
eval_env.close()
