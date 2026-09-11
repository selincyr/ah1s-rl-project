#############teacherdan topladığımımz veriyi alıp ilk student modelini eğite dosya ##########################
from pathlib import Path
import json
import csv
import math
import random
import numpy as np

import torch
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO


# =====================================================================
# STAGE-4 POLICY DISTILLATION / BEHAVIORAL CLONING V1
# =====================================================================
#
# IMPORTANT METHODOLOGY:
#   - This is supervised policy distillation / behavioral cloning.
#   - This is NOT reinforcement learning.
#   - The locked Stage-4 teacher remains unchanged.
#   - PPO is used here only as the neural policy architecture/container.
#   - Reward-based PPO fine-tuning comes later, only after teacher-off
#     validation of the distilled student.
#
# Dataset:
#   results_stage4_teacher_rollouts_v4/
#
# Student interface:
#   observation: normalized 26-D Stage-4 observation
#   action: 4-D normalized [-1, +1]
#   runtime mapping: normalized action -> physical actuator commands
#
# The physical mapping is fixed so all Stage-4 phases use ONE consistent
# neural action representation.

SEED = 42

DATA_DIR = Path("results_stage4_teacher_rollouts_v4")
MODEL_DIR = Path("models_stage4_distilled_bc_v1")
RESULT_DIR = Path("results_stage4_distilled_bc_v1")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OBS_PATH = DATA_DIR / "stage4_obs_raw.npy"
ACT_PATH = DATA_DIR / "stage4_actions_physical.npy"
PHASE_PATH = DATA_DIR / "stage4_phase_ids.npy"
ROLLOUT_PATH = DATA_DIR / "stage4_rollout_ids.npy"
WEIGHT_PATH = DATA_DIR / "stage4_sample_weights.npy"
META_PATH = DATA_DIR / "dataset_metadata.json"

MODEL_STEM = MODEL_DIR / "AH1S_STAGE4_BC_DISTILLED"
BEST_MODEL_ZIP = Path(str(MODEL_STEM) + ".zip")

# Fixed, evidence-compatible physical action envelope.
# All collected teacher targets lie strictly inside these bounds.
ACTION_LOW = np.asarray(
    [0.5360, -0.1830, 0.1800, 0.3890],
    dtype=np.float32,
)
ACTION_HIGH = np.asarray(
    [0.6070, -0.1780, 0.2020, 0.3910],
    dtype=np.float32,
)

ACTION_NAMES = [
    "collective",
    "elevator",
    "aileron",
    "rudder",
]

HIDDEN = [128, 128]

MAX_EPOCHS = 360
BATCH_SIZE = 512
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
PATIENCE = 55
MIN_DELTA = 1.0e-7

# Avoid exploding normalization on nearly constant channels.
OBS_STD_FLOOR = 1.0e-3

# Hold out the latest / most forward entry rollout (299 ft) as validation.
VALIDATION_ROLLOUT_ID = 5

# This only gates whether the student is worth running in JSBSim.
# It does NOT qualify the policy physically.
MAX_ACCEPTABLE_VAL_NORM_RMSE = 0.10
MIN_NOVEL_VALIDATION_SAMPLES = 50


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


class Stage4BCDummyEnv(gym.Env):
    metadata = {}

    def __init__(self, obs_dim):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return (
            np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            ),
            {},
        )

    def step(self, action):
        obs = np.zeros(
            self.observation_space.shape,
            dtype=np.float32,
        )
        reward = 0.0
        terminated = False
        truncated = True
        info = {}
        return obs, reward, terminated, truncated, info


def normalize_action_physical(actions):
    scale = ACTION_HIGH - ACTION_LOW
    z = 2.0 * (actions - ACTION_LOW) / scale - 1.0
    return z.astype(np.float32)


def denormalize_action(z):
    z = np.asarray(z, dtype=np.float32)
    return (
        ACTION_LOW
        + 0.5 * (z + 1.0) * (ACTION_HIGH - ACTION_LOW)
    ).astype(np.float32)


def exact_deduplicate(obs, act, phase, weights):
    combined = np.concatenate(
        [
            np.asarray(obs, dtype=np.float32),
            np.asarray(act, dtype=np.float32),
            np.asarray(phase, dtype=np.float32).reshape(-1, 1),
        ],
        axis=1,
    )

    _, first_idx = np.unique(
        combined,
        axis=0,
        return_index=True,
    )
    idx = np.sort(first_idx)

    return (
        obs[idx],
        act[idx],
        phase[idx],
        weights[idx],
        idx,
    )


def actor_mean(policy, obs_tensor):
    features = policy.extract_features(obs_tensor)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_pi)


@torch.no_grad()
def predict_normalized(policy, obs_norm, device, batch_size=4096):
    outs = []

    for start in range(0, len(obs_norm), batch_size):
        x = torch.as_tensor(
            obs_norm[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        pred = actor_mean(policy, x)
        pred = torch.clamp(pred, -1.0, 1.0)
        outs.append(pred.cpu().numpy())

    return np.concatenate(outs, axis=0)


def weighted_mse_numpy(pred, target, weights):
    per = np.mean((pred - target) ** 2, axis=1)
    w = weights / max(1e-12, float(np.mean(weights)))
    return float(np.mean(per * w))


def physical_metrics(pred_norm, target_physical):
    pred_phys = denormalize_action(pred_norm)
    err = pred_phys - target_physical

    metrics = {}

    for j, name in enumerate(ACTION_NAMES):
        e = err[:, j]
        metrics[name] = {
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "max_abs": float(np.max(np.abs(e))),
            "p95_abs": float(
                np.percentile(np.abs(e), 95.0)
            ),
        }

    return pred_phys, metrics


def phase_metrics(
    pred_norm,
    target_norm,
    target_phys,
    phase_ids,
    phase_names,
):
    out = {}

    for pid in sorted(int(x) for x in np.unique(phase_ids)):
        mask = phase_ids == pid

        if not np.any(mask):
            continue

        norm_rmse = float(
            np.sqrt(
                np.mean(
                    (pred_norm[mask] - target_norm[mask]) ** 2
                )
            )
        )

        _, phys = physical_metrics(
            pred_norm[mask],
            target_phys[mask],
        )

        name = (
            phase_names[pid]
            if 0 <= pid < len(phase_names)
            else f"phase_{pid}"
        )

        out[name] = {
            "samples": int(np.sum(mask)),
            "normalized_rmse": norm_rmse,
            "physical": phys,
        }

    return out


def write_history(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# A — LOAD / VERIFY DATASET
# =====================================================================

print("=" * 120)
print("A — LOAD VERIFIED STAGE-4 TEACHER DATASET")
print("=" * 120)

for p in [
    OBS_PATH,
    ACT_PATH,
    PHASE_PATH,
    ROLLOUT_PATH,
    WEIGHT_PATH,
    META_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required Stage-4 dataset file: {p}"
        )

obs_raw = np.load(OBS_PATH).astype(np.float32)
actions_phys = np.load(ACT_PATH).astype(np.float32)
phase_ids = np.load(PHASE_PATH).astype(np.int64)
rollout_ids = np.load(ROLLOUT_PATH).astype(np.int64)
sample_weights = np.load(WEIGHT_PATH).astype(np.float32)

with open(META_PATH) as f:
    metadata = json.load(f)

if not metadata.get("rollout_diversity_ok", False):
    raise RuntimeError(
        "Dataset metadata says rollout diversity is NOT qualified."
    )

if not metadata.get("ready_for_bc", False):
    raise RuntimeError(
        "Dataset metadata says READY FOR BC is false."
    )

if obs_raw.ndim != 2 or obs_raw.shape[1] != 26:
    raise RuntimeError(
        f"Unexpected Stage-4 observation shape: {obs_raw.shape}"
    )

if actions_phys.shape != (len(obs_raw), 4):
    raise RuntimeError(
        f"Unexpected Stage-4 action shape: {actions_phys.shape}"
    )

if not np.all(actions_phys >= ACTION_LOW - 1e-7):
    raise RuntimeError(
        "Collected teacher action is below fixed physical action envelope."
    )

if not np.all(actions_phys <= ACTION_HIGH + 1e-7):
    raise RuntimeError(
        "Collected teacher action is above fixed physical action envelope."
    )

print(f"dataset obs      : {obs_raw.shape}")
print(f"dataset actions  : {actions_phys.shape}")
print(
    f"rollout diversity: "
    f"{metadata['distinct_observation_rollouts']}/"
    f"{metadata['passing_rollouts']}"
)
print(f"validation rollout id: {VALIDATION_ROLLOUT_ID}")


# =====================================================================
# B — GROUPED TRAIN / VALIDATION SPLIT
# =====================================================================

print()
print("=" * 120)
print("B — GROUPED TRAIN / VALIDATION SPLIT")
print("=" * 120)

train_mask = rollout_ids != VALIDATION_ROLLOUT_ID
val_mask = rollout_ids == VALIDATION_ROLLOUT_ID

if not np.any(train_mask):
    raise RuntimeError("Empty training split.")

if not np.any(val_mask):
    raise RuntimeError("Empty validation split.")

train_obs_raw_all = obs_raw[train_mask]
train_act_phys_all = actions_phys[train_mask]
train_phase_all = phase_ids[train_mask]
train_weights_all = sample_weights[train_mask]

val_obs_raw = obs_raw[val_mask]
val_act_phys = actions_phys[val_mask]
val_phase = phase_ids[val_mask]
val_weights = sample_weights[val_mask]

(
    train_obs_raw,
    train_act_phys,
    train_phase,
    train_weights,
    train_unique_idx,
) = exact_deduplicate(
    train_obs_raw_all,
    train_act_phys_all,
    train_phase_all,
    train_weights_all,
)

train_weights = (
    train_weights
    / max(1e-12, float(np.mean(train_weights)))
).astype(np.float32)

val_weights = (
    val_weights
    / max(1e-12, float(np.mean(val_weights)))
).astype(np.float32)

print(
    f"train raw rows       : {len(train_obs_raw_all)}"
)
print(
    f"train exact-unique   : {len(train_obs_raw)}"
)
print(
    f"removed train repeats: "
    f"{len(train_obs_raw_all) - len(train_obs_raw)}"
)
print(f"validation rows      : {len(val_obs_raw)}")

# Strictly novel validation observations, for a less optimistic metric.
train_obs_keys = {
    np.ascontiguousarray(row).tobytes()
    for row in train_obs_raw
}

novel_val_mask = np.asarray(
    [
        np.ascontiguousarray(row).tobytes()
        not in train_obs_keys
        for row in val_obs_raw
    ],
    dtype=bool,
)

print(
    f"strict novel validation observations: "
    f"{int(np.sum(novel_val_mask))}/{len(val_obs_raw)}"
)


# =====================================================================
# C — NORMALIZATION
# =====================================================================

print()
print("=" * 120)
print("C — BUILD FIXED STUDENT NORMALIZATION")
print("=" * 120)

obs_mean = np.mean(
    train_obs_raw,
    axis=0,
).astype(np.float32)

obs_std_raw = np.std(
    train_obs_raw,
    axis=0,
).astype(np.float32)

obs_std = np.maximum(
    obs_std_raw,
    OBS_STD_FLOOR,
).astype(np.float32)

train_obs = (
    (train_obs_raw - obs_mean) / obs_std
).astype(np.float32)

val_obs = (
    (val_obs_raw - obs_mean) / obs_std
).astype(np.float32)

train_target = normalize_action_physical(
    train_act_phys
)

val_target = normalize_action_physical(
    val_act_phys
)

print(
    f"normalized train obs range: "
    f"[{np.min(train_obs):+.3f}, {np.max(train_obs):+.3f}]"
)
print(
    f"normalized target range: "
    f"[{np.min(train_target):+.3f}, "
    f"{np.max(train_target):+.3f}]"
)

np.savez(
    MODEL_DIR / "stage4_obs_normalization.npz",
    mean=obs_mean,
    std=obs_std,
    std_raw=obs_std_raw,
)

np.savez(
    MODEL_DIR / "stage4_action_mapping.npz",
    physical_low=ACTION_LOW,
    physical_high=ACTION_HIGH,
)


# =====================================================================
# D — FRESH PPO POLICY CONTAINER
# =====================================================================

print()
print("=" * 120)
print("D — CREATE FRESH STAGE-4 PPO NEURAL POLICY")
print("=" * 120)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

env = Stage4BCDummyEnv(
    obs_dim=train_obs.shape[1]
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=64,
    batch_size=64,
    n_epochs=1,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        "net_arch": dict(
            pi=HIDDEN,
            vf=HIDDEN,
        ),
    },
    verbose=0,
    seed=SEED,
    device=device,
)

policy = model.policy
policy.train()

optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

print(f"device: {device}")
print(f"policy hidden layers: {HIDDEN}")
print(
    "Training mode: supervised BC / policy distillation "
    "(NOT reinforcement learning)"
)


# =====================================================================
# E — WEIGHTED BEHAVIORAL CLONING
# =====================================================================

print()
print("=" * 120)
print("E — WEIGHTED STAGE-4 POLICY DISTILLATION")
print("=" * 120)

history = []

best_val = float("inf")
best_epoch = -1
epochs_without_improvement = 0

n_train = len(train_obs)

for epoch in range(1, MAX_EPOCHS + 1):
    policy.train()

    perm = np.random.permutation(n_train)

    epoch_loss_num = 0.0
    epoch_weight_sum = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]

        x = torch.as_tensor(
            train_obs[idx],
            dtype=torch.float32,
            device=device,
        )
        y = torch.as_tensor(
            train_target[idx],
            dtype=torch.float32,
            device=device,
        )
        w = torch.as_tensor(
            train_weights[idx],
            dtype=torch.float32,
            device=device,
        )

        pred = actor_mean(policy, x)

        per_sample = torch.mean(
            (pred - y) ** 2,
            dim=1,
        )

        loss = torch.sum(
            per_sample * w
        ) / torch.sum(w)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        epoch_loss_num += float(
            torch.sum(per_sample.detach() * w).cpu()
        )
        epoch_weight_sum += float(
            torch.sum(w).detach().cpu()
        )

    train_loss = (
        epoch_loss_num
        / max(1e-12, epoch_weight_sum)
    )

    policy.eval()

    pred_val = predict_normalized(
        policy,
        val_obs,
        device,
    )

    val_loss = weighted_mse_numpy(
        pred_val,
        val_target,
        val_weights,
    )

    val_norm_rmse = float(
        np.sqrt(
            np.mean(
                (pred_val - val_target) ** 2
            )
        )
    )

    if np.any(novel_val_mask):
        novel_norm_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        pred_val[novel_val_mask]
                        - val_target[novel_val_mask]
                    ) ** 2
                )
            )
        )
    else:
        novel_norm_rmse = float("nan")

    history.append({
        "epoch": int(epoch),
        "train_weighted_mse": float(train_loss),
        "val_weighted_mse": float(val_loss),
        "val_normalized_rmse": float(val_norm_rmse),
        "novel_val_normalized_rmse": float(
            novel_norm_rmse
        ),
    })

    improved = (
        val_loss < best_val - MIN_DELTA
    )

    if improved:
        best_val = float(val_loss)
        best_epoch = int(epoch)
        epochs_without_improvement = 0
        model.save(MODEL_STEM)
    else:
        epochs_without_improvement += 1

    if (
        epoch == 1
        or epoch % 10 == 0
        or improved and epoch <= 10
    ):
        print(
            f"epoch {epoch:03d} | "
            f"trainMSE={train_loss:.8f} | "
            f"valMSE={val_loss:.8f} | "
            f"valRMSE={val_norm_rmse:.6f} | "
            f"novelRMSE={novel_norm_rmse:.6f} | "
            f"best={best_val:.8f}@{best_epoch}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(
            f"Early stopping at epoch {epoch}; "
            f"best epoch={best_epoch}"
        )
        break


# =====================================================================
# F — LOAD BEST MODEL / FINAL BC METRICS
# =====================================================================

print()
print("=" * 120)
print("F — BEST DISTILLED POLICY VALIDATION")
print("=" * 120)

if not BEST_MODEL_ZIP.exists():
    raise RuntimeError(
        f"Best BC model was not saved: {BEST_MODEL_ZIP}"
    )

best_model = PPO.load(
    str(BEST_MODEL_ZIP),
    env=env,
    device=device,
)
best_policy = best_model.policy
best_policy.eval()

pred_train = predict_normalized(
    best_policy,
    train_obs,
    device,
)

pred_val = predict_normalized(
    best_policy,
    val_obs,
    device,
)

train_norm_rmse = float(
    np.sqrt(
        np.mean(
            (pred_train - train_target) ** 2
        )
    )
)

val_norm_rmse = float(
    np.sqrt(
        np.mean(
            (pred_val - val_target) ** 2
        )
    )
)

if np.any(novel_val_mask):
    novel_val_norm_rmse = float(
        np.sqrt(
            np.mean(
                (
                    pred_val[novel_val_mask]
                    - val_target[novel_val_mask]
                ) ** 2
            )
        )
    )
else:
    novel_val_norm_rmse = float("nan")

_, train_phys_metrics = physical_metrics(
    pred_train,
    train_act_phys,
)

val_pred_phys, val_phys_metrics = physical_metrics(
    pred_val,
    val_act_phys,
)

phase_names = metadata.get(
    "phase_names",
    [f"phase_{i}" for i in range(6)],
)

val_phase_metrics = phase_metrics(
    pred_val,
    val_target,
    val_act_phys,
    val_phase,
    phase_names,
)

print(
    f"best epoch                 : {best_epoch}"
)
print(
    f"train normalized RMSE      : {train_norm_rmse:.6f}"
)
print(
    f"validation normalized RMSE : {val_norm_rmse:.6f}"
)
print(
    f"novel validation RMSE      : {novel_val_norm_rmse:.6f}"
)
print(
    f"novel validation samples   : "
    f"{int(np.sum(novel_val_mask))}"
)

print()
print("validation physical actuator errors:")

for name in ACTION_NAMES:
    m = val_phys_metrics[name]
    print(
        f"  {name:10s} | "
        f"MAE={m['mae']:.8f} | "
        f"RMSE={m['rmse']:.8f} | "
        f"P95={m['p95_abs']:.8f} | "
        f"MAX={m['max_abs']:.8f}"
    )

print()
print("validation error by decision phase:")

for phase_name, pm in val_phase_metrics.items():
    print(
        f"  {phase_name:30s} | "
        f"n={pm['samples']:5d} | "
        f"normRMSE={pm['normalized_rmse']:.6f}"
    )


# =====================================================================
# G — SAVE EVIDENCE
# =====================================================================

write_history(
    RESULT_DIR / "training_history.csv",
    history,
)

np.save(
    RESULT_DIR / "validation_pred_normalized.npy",
    pred_val.astype(np.float32),
)

np.save(
    RESULT_DIR / "validation_pred_physical.npy",
    val_pred_phys.astype(np.float32),
)

ready_teacher_off = bool(
    np.isfinite(val_norm_rmse)
    and val_norm_rmse
        <= MAX_ACCEPTABLE_VAL_NORM_RMSE
    and int(np.sum(novel_val_mask))
        >= MIN_NOVEL_VALIDATION_SAMPLES
)

summary = {
    "training_type": (
        "supervised behavioral cloning / policy distillation"
    ),
    "reinforcement_learning_used": False,
    "ppo_container_only": True,
    "seed": SEED,
    "dataset_dir": str(DATA_DIR),
    "validation_rollout_id": int(
        VALIDATION_ROLLOUT_ID
    ),
    "train_rows_before_dedup": int(
        len(train_obs_raw_all)
    ),
    "train_rows_after_exact_dedup": int(
        len(train_obs_raw)
    ),
    "validation_rows": int(
        len(val_obs_raw)
    ),
    "novel_validation_rows": int(
        np.sum(novel_val_mask)
    ),
    "obs_dim": int(train_obs.shape[1]),
    "action_dim": 4,
    "hidden_layers": HIDDEN,
    "action_low_physical": ACTION_LOW.tolist(),
    "action_high_physical": ACTION_HIGH.tolist(),
    "best_epoch": int(best_epoch),
    "best_val_weighted_mse": float(best_val),
    "train_normalized_rmse": float(
        train_norm_rmse
    ),
    "validation_normalized_rmse": float(
        val_norm_rmse
    ),
    "novel_validation_normalized_rmse": float(
        novel_val_norm_rmse
    ),
    "train_physical_metrics": train_phys_metrics,
    "validation_physical_metrics": val_phys_metrics,
    "validation_phase_metrics": val_phase_metrics,
    "model_zip": str(BEST_MODEL_ZIP),
    "obs_normalization": str(
        MODEL_DIR / "stage4_obs_normalization.npz"
    ),
    "action_mapping": str(
        MODEL_DIR / "stage4_action_mapping.npz"
    ),
    "ready_for_teacher_off_validation": bool(
        ready_teacher_off
    ),
}

with open(
    RESULT_DIR / "final_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

with open(
    MODEL_DIR / "stage4_policy_interface.json",
    "w",
) as f:
    json.dump(
        {
            "observation_dim": 26,
            "observation_input": (
                "raw Stage-4 26-D observation normalized by "
                "stage4_obs_normalization.npz"
            ),
            "network_action_space": (
                "4-D normalized [-1,+1]"
            ),
            "physical_action_mapping": {
                name: {
                    "low": float(ACTION_LOW[i]),
                    "high": float(ACTION_HIGH[i]),
                }
                for i, name in enumerate(ACTION_NAMES)
            },
            "note": (
                "BC/distillation only; no reward-based PPO update "
                "has occurred yet."
            ),
        },
        f,
        indent=2,
    )


print()
print("=" * 120)
print("G — STAGE-4 BC / DISTILLATION V1 CONCLUSION")
print("=" * 120)

print(
    "BC / POLICY DISTILLATION COMPLETE: True"
)
print(
    "REWARD-BASED PPO TRAINING USED: False"
)
print(
    "READY FOR STAGE-4 TEACHER-OFF VALIDATION: "
    f"{ready_teacher_off}"
)

print("Saved:")
print(f"  {BEST_MODEL_ZIP}")
print(
    f"  {MODEL_DIR / 'stage4_obs_normalization.npz'}"
)
print(
    f"  {MODEL_DIR / 'stage4_action_mapping.npz'}"
)
print(
    f"  {MODEL_DIR / 'stage4_policy_interface.json'}"
)
print(
    f"  {RESULT_DIR / 'training_history.csv'}"
)
print(
    f"  {RESULT_DIR / 'final_summary.json'}"
)

print()
print(
    "Do NOT call this RL training. "
    "The next required step is a fresh teacher-OFF JSBSim validation "
    "where this student policy alone commands the Stage-4 physical actuators."
)
