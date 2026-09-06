
from pathlib import Path
import json
import csv
import copy
import numpy as np
import torch

from stable_baselines3 import PPO


# =====================================================================
# STAGE-4 CORRECTIVE DISTILLATION V2
# DAgger corrective data + locked nominal teacher replay
# =====================================================================
#
# METHODOLOGY:
#   - NOT reinforcement learning.
#   - Start from the already-distilled Stage-4 BC V1 policy.
#   - Freeze the entire actor representation and action channels 1..3.
#   - Refit ONLY the collective actor-head row (row 0).
#   - Use nominal teacher replay to preserve the qualified teacher manifold.
#   - Use student-visited DAgger shadow labels to correct closed-loop descent.
#
# This is intentionally narrow because the teacher-off failure was isolated to
# collective under-command while longitudinal/lateral geometry remained clean.

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

NOMINAL_DIR = Path("results_stage4_teacher_rollouts_v4")
DAGGER_DIR = Path("results_stage4_dagger_shadow_v1")
BASE_MODEL_DIR = Path("models_stage4_distilled_bc_v1")

OUT_MODEL_DIR = Path("models_stage4_corrective_distilled_v2")
OUT_RESULT_DIR = Path("results_stage4_corrective_distilled_v2")

OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = BASE_MODEL_DIR / "AH1S_STAGE4_BC_DISTILLED.zip"
OBS_NORM_PATH = BASE_MODEL_DIR / "stage4_obs_normalization.npz"
ACTION_MAP_PATH = BASE_MODEL_DIR / "stage4_action_mapping.npz"

NOMINAL_OBS = NOMINAL_DIR / "stage4_obs_raw.npy"
NOMINAL_ACT = NOMINAL_DIR / "stage4_actions_physical.npy"
NOMINAL_ROLLOUT = NOMINAL_DIR / "stage4_rollout_ids.npy"
NOMINAL_WEIGHT = NOMINAL_DIR / "stage4_sample_weights.npy"

DAGGER_OBS = DAGGER_DIR / "dagger_obs_raw.npy"
DAGGER_ACT = DAGGER_DIR / "dagger_teacher_actions_physical.npy"
DAGGER_ROLLOUT = DAGGER_DIR / "dagger_rollout_ids.npy"
DAGGER_WEIGHT = DAGGER_DIR / "dagger_priority_weights.npy"

VALIDATION_ROLLOUT_ID = 5

# Conservative weighting grid. DAgger samples are fewer but specifically target
# the closed-loop failure region, so their aggregate influence must be meaningful.
DAGGER_GLOBAL_WEIGHTS = [1.0, 2.0, 4.0, 6.0]

# Ridge is centered on the ORIGINAL collective head, not zero. This directly
# penalizes unnecessary departure from the BC V1 policy.
RIDGE_FACTORS = [
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
]

# A candidate is rejected if it damages nominal collective imitation too much.
MAX_NOMINAL_VAL_COLLECTIVE_NORM_RMSE = 0.006

# Require a meaningful improvement on the held-out corrective trajectory.
MIN_DAGGER_VAL_IMPROVEMENT_FRACTION = 0.25

ACTION_NAMES = [
    "collective",
    "elevator",
    "aileron",
    "rudder",
]


def load_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def normalize_action_physical(actions, low, high):
    return (
        2.0 * (actions - low) / (high - low) - 1.0
    ).astype(np.float32)


def denormalize_action(z, low, high):
    return (
        low + 0.5 * (z + 1.0) * (high - low)
    ).astype(np.float32)


@torch.no_grad()
def actor_latent(policy, obs_norm, device, batch_size=4096):
    chunks = []

    for start in range(0, len(obs_norm), batch_size):
        x = torch.as_tensor(
            obs_norm[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        features = policy.extract_features(x)
        latent = policy.mlp_extractor.forward_actor(features)
        chunks.append(latent.cpu().numpy())

    return np.concatenate(chunks, axis=0).astype(np.float64)


@torch.no_grad()
def actor_output(policy, obs_norm, device, batch_size=4096):
    chunks = []

    for start in range(0, len(obs_norm), batch_size):
        x = torch.as_tensor(
            obs_norm[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        features = policy.extract_features(x)
        latent = policy.mlp_extractor.forward_actor(features)
        y = policy.action_net(latent)
        y = torch.clamp(y, -1.0, 1.0)
        chunks.append(y.cpu().numpy())

    return np.concatenate(chunks, axis=0).astype(np.float32)


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def mae(x):
    return float(np.mean(np.abs(x)))


def fit_centered_weighted_ridge(
    X,
    y,
    weights,
    beta0,
    ridge_factor,
):
    """
    Solve:
        min_beta sum_i w_i (x_i beta - y_i)^2
                 + lambda ||beta - beta0||^2

    X includes the bias column.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    beta0 = np.asarray(beta0, dtype=np.float64)

    weights = weights / max(1e-12, float(np.mean(weights)))

    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Xw = X * sqrt_w
    yw = y * sqrt_w.reshape(-1)

    xtx = Xw.T @ Xw
    xty = Xw.T @ yw

    scale = float(np.mean(np.diag(xtx)))
    lam = max(1e-12, ridge_factor * scale)

    A = xtx + lam * np.eye(X.shape[1], dtype=np.float64)
    b = xty + lam * beta0

    beta = np.linalg.solve(A, b)

    return beta, lam


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# A — LOAD DATA / BASE POLICY
# =====================================================================

print("=" * 120)
print("A — LOAD BASE BC V1 + NOMINAL + DAgger CORRECTIVE DATA")
print("=" * 120)

for p in [
    BASE_MODEL,
    OBS_NORM_PATH,
    ACTION_MAP_PATH,
    NOMINAL_OBS,
    NOMINAL_ACT,
    NOMINAL_ROLLOUT,
    NOMINAL_WEIGHT,
    DAGGER_OBS,
    DAGGER_ACT,
    DAGGER_ROLLOUT,
    DAGGER_WEIGHT,
]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required input: {p}")

norm_pack = np.load(OBS_NORM_PATH)
obs_mean = norm_pack["mean"].astype(np.float32)
obs_std = norm_pack["std"].astype(np.float32)

map_pack = np.load(ACTION_MAP_PATH)
action_low = map_pack["physical_low"].astype(np.float32)
action_high = map_pack["physical_high"].astype(np.float32)

nom_obs_raw = np.load(NOMINAL_OBS).astype(np.float32)
nom_act_phys = np.load(NOMINAL_ACT).astype(np.float32)
nom_rollout = np.load(NOMINAL_ROLLOUT).astype(np.int64)
nom_weight = np.load(NOMINAL_WEIGHT).astype(np.float32)

dag_obs_raw = np.load(DAGGER_OBS).astype(np.float32)
dag_act_phys = np.load(DAGGER_ACT).astype(np.float32)
dag_rollout = np.load(DAGGER_ROLLOUT).astype(np.int64)
dag_weight = np.load(DAGGER_WEIGHT).astype(np.float32)

nom_obs = ((nom_obs_raw - obs_mean) / obs_std).astype(np.float32)
dag_obs = ((dag_obs_raw - obs_mean) / obs_std).astype(np.float32)

nom_target = normalize_action_physical(
    nom_act_phys,
    action_low,
    action_high,
)
dag_target = normalize_action_physical(
    dag_act_phys,
    action_low,
    action_high,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

base_model = PPO.load(
    str(BASE_MODEL),
    device=device,
)
base_model.policy.eval()

print(f"device               : {device}")
print(f"nominal samples      : {len(nom_obs)}")
print(f"DAgger samples       : {len(dag_obs)}")
print(f"validation rollout id: {VALIDATION_ROLLOUT_ID}")


# =====================================================================
# B — GROUPED SPLITS
# =====================================================================

print()
print("=" * 120)
print("B — GROUPED NOMINAL / CORRECTIVE SPLITS")
print("=" * 120)

nom_train_mask = nom_rollout != VALIDATION_ROLLOUT_ID
nom_val_mask = nom_rollout == VALIDATION_ROLLOUT_ID

dag_train_mask = dag_rollout != VALIDATION_ROLLOUT_ID
dag_val_mask = dag_rollout == VALIDATION_ROLLOUT_ID

if not np.any(nom_val_mask):
    raise RuntimeError("Nominal validation rollout missing.")

if not np.any(dag_val_mask):
    raise RuntimeError("DAgger validation rollout missing.")

print(
    f"nominal train/val : "
    f"{int(np.sum(nom_train_mask))}/"
    f"{int(np.sum(nom_val_mask))}"
)
print(
    f"DAgger train/val  : "
    f"{int(np.sum(dag_train_mask))}/"
    f"{int(np.sum(dag_val_mask))}"
)


# =====================================================================
# C — FREEZE ACTOR REPRESENTATION / EXTRACT LATENTS
# =====================================================================

print()
print("=" * 120)
print("C — EXTRACT FIXED ACTOR LATENTS")
print("=" * 120)

nom_latent = actor_latent(
    base_model.policy,
    nom_obs,
    device,
)

dag_latent = actor_latent(
    base_model.policy,
    dag_obs,
    device,
)

# Append bias feature.
nom_X = np.concatenate(
    [
        nom_latent,
        np.ones((len(nom_latent), 1), dtype=np.float64),
    ],
    axis=1,
)

dag_X = np.concatenate(
    [
        dag_latent,
        np.ones((len(dag_latent), 1), dtype=np.float64),
    ],
    axis=1,
)

head_weight = (
    base_model.policy.action_net.weight.detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)

head_bias = (
    base_model.policy.action_net.bias.detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)

beta0 = np.concatenate(
    [
        head_weight[0],
        np.asarray([head_bias[0]], dtype=np.float64),
    ]
)

print(f"latent dim          : {nom_latent.shape[1]}")
print(
    "trainable parameter scope: "
    "collective action-head row only"
)


# =====================================================================
# D — BASELINE HELD-OUT METRICS
# =====================================================================

print()
print("=" * 120)
print("D — BASELINE BC V1 HELD-OUT METRICS")
print("=" * 120)

base_nom_pred = actor_output(
    base_model.policy,
    nom_obs[nom_val_mask],
    device,
)

base_dag_pred = actor_output(
    base_model.policy,
    dag_obs[dag_val_mask],
    device,
)

base_nom_coll_rmse = rmse(
    base_nom_pred[:, 0]
    - nom_target[nom_val_mask, 0]
)

base_dag_coll_rmse = rmse(
    base_dag_pred[:, 0]
    - dag_target[dag_val_mask, 0]
)

base_dag_coll_phys = denormalize_action(
    base_dag_pred,
    action_low,
    action_high,
)[:, 0]

base_dag_phys_rmse = rmse(
    base_dag_coll_phys
    - dag_act_phys[dag_val_mask, 0]
)

print(
    f"baseline nominal val collective norm RMSE: "
    f"{base_nom_coll_rmse:.6f}"
)
print(
    f"baseline DAgger val collective norm RMSE : "
    f"{base_dag_coll_rmse:.6f}"
)
print(
    f"baseline DAgger val collective phys RMSE : "
    f"{base_dag_phys_rmse:.8f}"
)


# =====================================================================
# E — CENTERED WEIGHTED RIDGE GRID
# =====================================================================

print()
print("=" * 120)
print("E — CORRECTIVE COLLECTIVE-HEAD GRID")
print("=" * 120)

X_nom_train = nom_X[nom_train_mask]
y_nom_train = nom_target[nom_train_mask, 0].astype(np.float64)
w_nom_train = nom_weight[nom_train_mask].astype(np.float64)

X_dag_train = dag_X[dag_train_mask]
y_dag_train = dag_target[dag_train_mask, 0].astype(np.float64)
w_dag_train_raw = dag_weight[dag_train_mask].astype(np.float64)

X_nom_val = nom_X[nom_val_mask]
y_nom_val = nom_target[nom_val_mask, 0].astype(np.float64)

X_dag_val = dag_X[dag_val_mask]
y_dag_val = dag_target[dag_val_mask, 0].astype(np.float64)

rows = []
best = None

for dagger_global_weight in DAGGER_GLOBAL_WEIGHTS:
    # Normalize each source to mean 1 before the source-level multiplier.
    wn = w_nom_train / max(
        1e-12,
        float(np.mean(w_nom_train)),
    )
    wd = w_dag_train_raw / max(
        1e-12,
        float(np.mean(w_dag_train_raw)),
    )
    wd = wd * float(dagger_global_weight)

    X_train = np.concatenate(
        [X_nom_train, X_dag_train],
        axis=0,
    )
    y_train = np.concatenate(
        [y_nom_train, y_dag_train],
        axis=0,
    )
    w_train = np.concatenate(
        [wn, wd],
        axis=0,
    )

    for ridge_factor in RIDGE_FACTORS:
        beta, lam = fit_centered_weighted_ridge(
            X_train,
            y_train,
            w_train,
            beta0,
            ridge_factor,
        )

        pred_nom = np.clip(
            X_nom_val @ beta,
            -1.0,
            +1.0,
        )
        pred_dag = np.clip(
            X_dag_val @ beta,
            -1.0,
            +1.0,
        )

        nom_rmse = rmse(
            pred_nom - y_nom_val
        )
        dag_rmse = rmse(
            pred_dag - y_dag_val
        )

        improvement = (
            1.0
            - dag_rmse
            / max(1e-12, base_dag_coll_rmse)
        )

        # Favor DAgger correction while retaining the nominal manifold.
        score = (
            dag_rmse
            + 0.50 * nom_rmse
        )

        feasible = bool(
            nom_rmse
            <= MAX_NOMINAL_VAL_COLLECTIVE_NORM_RMSE
            and improvement
            >= MIN_DAGGER_VAL_IMPROVEMENT_FRACTION
        )

        row = {
            "dagger_global_weight": float(
                dagger_global_weight
            ),
            "ridge_factor": float(
                ridge_factor
            ),
            "lambda": float(lam),
            "nominal_val_collective_norm_rmse": float(
                nom_rmse
            ),
            "dagger_val_collective_norm_rmse": float(
                dag_rmse
            ),
            "dagger_improvement_fraction": float(
                improvement
            ),
            "score": float(score),
            "feasible": bool(feasible),
        }
        rows.append(row)

        if feasible:
            if best is None or score < best["score"]:
                best = {
                    **row,
                    "beta": beta,
                }

        print(
            f"DAGw={dagger_global_weight:3.1f} | "
            f"ridge={ridge_factor:.1e} | "
            f"nomRMSE={nom_rmse:.6f} | "
            f"dagRMSE={dag_rmse:.6f} | "
            f"improve={100*improvement:6.2f}% | "
            f"feasible={feasible}"
        )

write_csv(
    OUT_RESULT_DIR / "candidate_grid.csv",
    rows,
)

if best is None:
    raise RuntimeError(
        "No corrective collective-head candidate satisfied the "
        "nominal-preservation + DAgger-improvement gates."
    )


# =====================================================================
# F — WRITE ONLY COLLECTIVE HEAD ROW
# =====================================================================

print()
print("=" * 120)
print("F — BUILD CORRECTIVE DISTILLED V2 MODEL")
print("=" * 120)

corrected_model = PPO.load(
    str(BASE_MODEL),
    device=device,
)

before_weight = (
    corrected_model.policy.action_net.weight.detach()
    .cpu()
    .numpy()
    .copy()
)
before_bias = (
    corrected_model.policy.action_net.bias.detach()
    .cpu()
    .numpy()
    .copy()
)

beta = best["beta"]

with torch.no_grad():
    corrected_model.policy.action_net.weight[0].copy_(
        torch.as_tensor(
            beta[:-1],
            dtype=corrected_model.policy.action_net.weight.dtype,
            device=corrected_model.policy.action_net.weight.device,
        )
    )
    corrected_model.policy.action_net.bias[0].copy_(
        torch.as_tensor(
            beta[-1],
            dtype=corrected_model.policy.action_net.bias.dtype,
            device=corrected_model.policy.action_net.bias.device,
        )
    )

after_weight = (
    corrected_model.policy.action_net.weight.detach()
    .cpu()
    .numpy()
    .copy()
)
after_bias = (
    corrected_model.policy.action_net.bias.detach()
    .cpu()
    .numpy()
    .copy()
)

# Hard guarantee: rows 1..3 are byte-for-byte unchanged.
if not np.array_equal(
    before_weight[1:],
    after_weight[1:],
):
    raise RuntimeError(
        "Non-collective actor-head weights changed unexpectedly."
    )

if not np.array_equal(
    before_bias[1:],
    after_bias[1:],
):
    raise RuntimeError(
        "Non-collective actor-head biases changed unexpectedly."
    )

OUT_MODEL_STEM = (
    OUT_MODEL_DIR
    / "AH1S_STAGE4_CORRECTIVE_DISTILLED_V2"
)
corrected_model.save(OUT_MODEL_STEM)
OUT_MODEL_ZIP = Path(str(OUT_MODEL_STEM) + ".zip")

# Reuse the exact same normalization / physical action mapping.
np.savez(
    OUT_MODEL_DIR / "stage4_obs_normalization.npz",
    mean=obs_mean,
    std=obs_std,
    std_raw=norm_pack["std_raw"]
    if "std_raw" in norm_pack.files
    else obs_std,
)

np.savez(
    OUT_MODEL_DIR / "stage4_action_mapping.npz",
    physical_low=action_low,
    physical_high=action_high,
)


# =====================================================================
# G — FINAL HELD-OUT COMPARISON
# =====================================================================

print()
print("=" * 120)
print("G — FINAL V2 HELD-OUT COMPARISON")
print("=" * 120)

corrected_model.policy.eval()

v2_nom_pred = actor_output(
    corrected_model.policy,
    nom_obs[nom_val_mask],
    device,
)

v2_dag_pred = actor_output(
    corrected_model.policy,
    dag_obs[dag_val_mask],
    device,
)

v2_nom_coll_rmse = rmse(
    v2_nom_pred[:, 0]
    - nom_target[nom_val_mask, 0]
)

v2_dag_coll_rmse = rmse(
    v2_dag_pred[:, 0]
    - dag_target[dag_val_mask, 0]
)

v2_dag_phys = denormalize_action(
    v2_dag_pred,
    action_low,
    action_high,
)

v2_dag_phys_rmse = rmse(
    v2_dag_phys[:, 0]
    - dag_act_phys[dag_val_mask, 0]
)

# Non-collective network outputs must remain exactly identical to BC V1 because
# the representation + action rows 1..3 were frozen.
nom_noncoll_max_delta = float(
    np.max(
        np.abs(
            v2_nom_pred[:, 1:]
            - base_nom_pred[:, 1:]
        )
    )
)

dag_noncoll_max_delta = float(
    np.max(
        np.abs(
            v2_dag_pred[:, 1:]
            - base_dag_pred[:, 1:]
        )
    )
)

print(
    f"selected DAgger global weight : "
    f"{best['dagger_global_weight']}"
)
print(
    f"selected ridge factor         : "
    f"{best['ridge_factor']:.1e}"
)
print(
    f"nominal collective RMSE "
    f"{base_nom_coll_rmse:.6f} -> "
    f"{v2_nom_coll_rmse:.6f}"
)
print(
    f"DAgger collective RMSE  "
    f"{base_dag_coll_rmse:.6f} -> "
    f"{v2_dag_coll_rmse:.6f}"
)
print(
    f"DAgger physical coll RMSE "
    f"{base_dag_phys_rmse:.8f} -> "
    f"{v2_dag_phys_rmse:.8f}"
)
print(
    f"non-collective max output delta nominal: "
    f"{nom_noncoll_max_delta:.12g}"
)
print(
    f"non-collective max output delta DAgger : "
    f"{dag_noncoll_max_delta:.12g}"
)

ready_teacher_off = bool(
    v2_nom_coll_rmse
    <= MAX_NOMINAL_VAL_COLLECTIVE_NORM_RMSE
    and v2_dag_coll_rmse
    <= base_dag_coll_rmse
    * (1.0 - MIN_DAGGER_VAL_IMPROVEMENT_FRACTION)
    and nom_noncoll_max_delta <= 1e-9
    and dag_noncoll_max_delta <= 1e-9
)

summary = {
    "training_type": (
        "DAgger-style corrective policy distillation / "
        "centered weighted ridge collective-head refit"
    ),
    "reinforcement_learning_used": False,
    "base_model": str(BASE_MODEL),
    "output_model": str(OUT_MODEL_ZIP),
    "changed_actor_scope": "action_net row 0 only (collective)",
    "non_collective_actor_rows_changed": False,
    "normalization_changed": False,
    "physical_action_mapping_changed": False,
    "validation_rollout_id": int(
        VALIDATION_ROLLOUT_ID
    ),
    "selected": {
        k: v
        for k, v in best.items()
        if k != "beta"
    },
    "baseline": {
        "nominal_val_collective_norm_rmse": float(
            base_nom_coll_rmse
        ),
        "dagger_val_collective_norm_rmse": float(
            base_dag_coll_rmse
        ),
        "dagger_val_collective_physical_rmse": float(
            base_dag_phys_rmse
        ),
    },
    "corrected": {
        "nominal_val_collective_norm_rmse": float(
            v2_nom_coll_rmse
        ),
        "dagger_val_collective_norm_rmse": float(
            v2_dag_coll_rmse
        ),
        "dagger_val_collective_physical_rmse": float(
            v2_dag_phys_rmse
        ),
        "nominal_noncollective_max_output_delta": float(
            nom_noncoll_max_delta
        ),
        "dagger_noncollective_max_output_delta": float(
            dag_noncoll_max_delta
        ),
    },
    "ready_for_teacher_off_validation": bool(
        ready_teacher_off
    ),
}

with open(
    OUT_RESULT_DIR / "final_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

with open(
    OUT_MODEL_DIR / "stage4_policy_interface.json",
    "w",
) as f:
    json.dump(
        {
            "observation_dim": 26,
            "normalization": "unchanged from BC V1",
            "action_mapping": "unchanged from BC V1",
            "corrective_change": (
                "collective actor-head row 0 only"
            ),
            "reinforcement_learning_used": False,
        },
        f,
        indent=2,
    )


print()
print("=" * 120)
print("H — CORRECTIVE DISTILLATION V2 CONCLUSION")
print("=" * 120)

print(
    "CORRECTIVE DISTILLATION COMPLETE: True"
)
print(
    "REWARD-BASED PPO TRAINING USED: False"
)
print(
    "NON-COLLECTIVE ACTOR OUTPUTS PRESERVED: "
    f"{nom_noncoll_max_delta <= 1e-9 and dag_noncoll_max_delta <= 1e-9}"
)
print(
    "READY FOR FRESH TEACHER-OFF VALIDATION: "
    f"{ready_teacher_off}"
)
print("Saved:")
print(f"  {OUT_MODEL_ZIP}")
print(
    f"  {OUT_MODEL_DIR / 'stage4_obs_normalization.npz'}"
)
print(
    f"  {OUT_MODEL_DIR / 'stage4_action_mapping.npz'}"
)
print(
    f"  {OUT_MODEL_DIR / 'stage4_policy_interface.json'}"
)
print(
    f"  {OUT_RESULT_DIR / 'candidate_grid.csv'}"
)
print(
    f"  {OUT_RESULT_DIR / 'final_summary.json'}"
)
print()
print(
    "Do NOT call this PPO/RL training. "
    "The next required step is a fresh teacher-OFF JSBSim run using V2."
)
