
from pathlib import Path
import csv
import json
import numpy as np
import torch

from stable_baselines3 import PPO


# =====================================================================
# STAGE-4 CORRECTIVE DISTILLATION V3
# nominal teacher replay + high-alt DAgger + low-alt DAgger
# =====================================================================
#
# METHODOLOGY:
#   - NOT reinforcement learning.
#   - Start from corrective distilled V2.
#   - Freeze the actor representation and action channels 1..3.
#   - Refit ONLY the collective actor-head row (row 0).
#   - Preserve:
#       1) nominal teacher manifold,
#       2) high-altitude DAgger correction,
#     while adding:
#       3) low-altitude 70->24 ft corrective labels.
#
# Goal:
#   make the student satisfy the 30-ft settle transition instead of
#   overshooting the settle window.

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_MODEL_DIR = Path("models_stage4_corrective_distilled_v2")
BASE_MODEL = (
    BASE_MODEL_DIR
    / "AH1S_STAGE4_CORRECTIVE_DISTILLED_V2.zip"
)

OBS_NORM_PATH = (
    BASE_MODEL_DIR
    / "stage4_obs_normalization.npz"
)
ACTION_MAP_PATH = (
    BASE_MODEL_DIR
    / "stage4_action_mapping.npz"
)

NOMINAL_DIR = Path(
    "results_stage4_teacher_rollouts_v4"
)
HIGH_DAGGER_DIR = Path(
    "results_stage4_dagger_shadow_v1"
)
LOW_DAGGER_DIR = Path(
    "results_stage4_dagger_lowalt_v2"
)

OUT_MODEL_DIR = Path(
    "models_stage4_corrective_distilled_v3"
)
OUT_RESULT_DIR = Path(
    "results_stage4_corrective_distilled_v3"
)

OUT_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
OUT_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VALIDATION_ROLLOUT_ID = 5

# Source-level weights.
# High-alt DAgger was already successful in V2 and must remain protected.
HIGH_DAGGER_GLOBAL_WEIGHTS = [2.0, 4.0, 6.0]
LOW_DAGGER_GLOBAL_WEIGHTS = [2.0, 4.0, 6.0, 8.0, 10.0]

RIDGE_FACTORS = [
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
]

# Candidate gates.
MAX_NOMINAL_VAL_RMSE = 0.0065
MAX_HIGH_DAGGER_DEGRADATION = 1.15
MIN_LOW_DAGGER_IMPROVEMENT = 0.35

# Files
NOMINAL_OBS = NOMINAL_DIR / "stage4_obs_raw.npy"
NOMINAL_ACT = (
    NOMINAL_DIR
    / "stage4_actions_physical.npy"
)
NOMINAL_ROLLOUT = (
    NOMINAL_DIR
    / "stage4_rollout_ids.npy"
)
NOMINAL_WEIGHT = (
    NOMINAL_DIR
    / "stage4_sample_weights.npy"
)

HIGH_OBS = (
    HIGH_DAGGER_DIR
    / "dagger_obs_raw.npy"
)
HIGH_ACT = (
    HIGH_DAGGER_DIR
    / "dagger_teacher_actions_physical.npy"
)
HIGH_ROLLOUT = (
    HIGH_DAGGER_DIR
    / "dagger_rollout_ids.npy"
)
HIGH_WEIGHT = (
    HIGH_DAGGER_DIR
    / "dagger_priority_weights.npy"
)

LOW_OBS = (
    LOW_DAGGER_DIR
    / "dagger_obs_raw.npy"
)
LOW_ACT = (
    LOW_DAGGER_DIR
    / "dagger_teacher_actions_physical.npy"
)
LOW_ROLLOUT = (
    LOW_DAGGER_DIR
    / "dagger_rollout_ids.npy"
)
LOW_WEIGHT = (
    LOW_DAGGER_DIR
    / "dagger_priority_weights.npy"
)


def normalize_action_physical(
    actions,
    low,
    high,
):
    return (
        2.0
        * (actions - low)
        / (high - low)
        - 1.0
    ).astype(np.float32)


def denormalize_action(
    z,
    low,
    high,
):
    return (
        low
        + 0.5
        * (z + 1.0)
        * (high - low)
    ).astype(np.float32)


@torch.no_grad()
def actor_latent(
    policy,
    obs_norm,
    device,
    batch_size=4096,
):
    chunks = []

    for start in range(
        0,
        len(obs_norm),
        batch_size,
    ):
        x = torch.as_tensor(
            obs_norm[
                start:start + batch_size
            ],
            dtype=torch.float32,
            device=device,
        )

        features = policy.extract_features(
            x
        )

        latent = (
            policy.mlp_extractor
            .forward_actor(features)
        )

        chunks.append(
            latent.cpu().numpy()
        )

    return np.concatenate(
        chunks,
        axis=0,
    ).astype(np.float64)


@torch.no_grad()
def actor_output(
    policy,
    obs_norm,
    device,
    batch_size=4096,
):
    chunks = []

    for start in range(
        0,
        len(obs_norm),
        batch_size,
    ):
        x = torch.as_tensor(
            obs_norm[
                start:start + batch_size
            ],
            dtype=torch.float32,
            device=device,
        )

        features = policy.extract_features(
            x
        )

        latent = (
            policy.mlp_extractor
            .forward_actor(features)
        )

        y = policy.action_net(
            latent
        )

        y = torch.clamp(
            y,
            -1.0,
            +1.0,
        )

        chunks.append(
            y.cpu().numpy()
        )

    return np.concatenate(
        chunks,
        axis=0,
    ).astype(np.float32)


def rmse(x):
    return float(
        np.sqrt(
            np.mean(
                np.square(x)
            )
        )
    )


def fit_centered_weighted_ridge(
    X,
    y,
    weights,
    beta0,
    ridge_factor,
):
    X = np.asarray(
        X,
        dtype=np.float64,
    )
    y = np.asarray(
        y,
        dtype=np.float64,
    )
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )
    beta0 = np.asarray(
        beta0,
        dtype=np.float64,
    )

    weights = (
        weights
        / max(
            1e-12,
            float(
                np.mean(weights)
            ),
        )
    )

    sqrt_w = np.sqrt(
        weights
    ).reshape(-1, 1)

    Xw = X * sqrt_w
    yw = y * sqrt_w.reshape(-1)

    xtx = Xw.T @ Xw
    xty = Xw.T @ yw

    scale = float(
        np.mean(
            np.diag(xtx)
        )
    )

    lam = max(
        1e-12,
        ridge_factor * scale,
    )

    A = (
        xtx
        + lam
        * np.eye(
            X.shape[1],
            dtype=np.float64,
        )
    )

    b = (
        xty
        + lam * beta0
    )

    beta = np.linalg.solve(
        A,
        b,
    )

    return beta, lam


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# A — LOAD
# =====================================================================

print("=" * 120)
print("A — LOAD V2 BASE + THREE DATA SOURCES")
print("=" * 120)

required = [
    BASE_MODEL,
    OBS_NORM_PATH,
    ACTION_MAP_PATH,
    NOMINAL_OBS,
    NOMINAL_ACT,
    NOMINAL_ROLLOUT,
    NOMINAL_WEIGHT,
    HIGH_OBS,
    HIGH_ACT,
    HIGH_ROLLOUT,
    HIGH_WEIGHT,
    LOW_OBS,
    LOW_ACT,
    LOW_ROLLOUT,
    LOW_WEIGHT,
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required file: {p}"
        )

norm_pack = np.load(
    OBS_NORM_PATH
)
obs_mean = (
    norm_pack["mean"]
    .astype(np.float32)
)
obs_std = (
    norm_pack["std"]
    .astype(np.float32)
)

map_pack = np.load(
    ACTION_MAP_PATH
)
action_low = (
    map_pack["physical_low"]
    .astype(np.float32)
)
action_high = (
    map_pack["physical_high"]
    .astype(np.float32)
)

nom_obs_raw = np.load(
    NOMINAL_OBS
).astype(np.float32)
nom_act_phys = np.load(
    NOMINAL_ACT
).astype(np.float32)
nom_rollout = np.load(
    NOMINAL_ROLLOUT
).astype(np.int64)
nom_weight = np.load(
    NOMINAL_WEIGHT
).astype(np.float32)

high_obs_raw = np.load(
    HIGH_OBS
).astype(np.float32)
high_act_phys = np.load(
    HIGH_ACT
).astype(np.float32)
high_rollout = np.load(
    HIGH_ROLLOUT
).astype(np.int64)
high_weight = np.load(
    HIGH_WEIGHT
).astype(np.float32)

low_obs_raw = np.load(
    LOW_OBS
).astype(np.float32)
low_act_phys = np.load(
    LOW_ACT
).astype(np.float32)
low_rollout = np.load(
    LOW_ROLLOUT
).astype(np.int64)
low_weight = np.load(
    LOW_WEIGHT
).astype(np.float32)

for name, arr in [
    ("nominal", nom_act_phys),
    ("high_dagger", high_act_phys),
    ("low_dagger", low_act_phys),
]:
    if not np.all(
        arr >= action_low - 1e-7
    ):
        raise RuntimeError(
            f"{name} target below action map."
        )

    if not np.all(
        arr <= action_high + 1e-7
    ):
        raise RuntimeError(
            f"{name} target above action map."
        )

nom_obs = (
    (nom_obs_raw - obs_mean)
    / obs_std
).astype(np.float32)

high_obs = (
    (high_obs_raw - obs_mean)
    / obs_std
).astype(np.float32)

low_obs = (
    (low_obs_raw - obs_mean)
    / obs_std
).astype(np.float32)

nom_target = (
    normalize_action_physical(
        nom_act_phys,
        action_low,
        action_high,
    )
)

high_target = (
    normalize_action_physical(
        high_act_phys,
        action_low,
        action_high,
    )
)

low_target = (
    normalize_action_physical(
        low_act_phys,
        action_low,
        action_high,
    )
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

base_model = PPO.load(
    str(BASE_MODEL),
    device=device,
)
base_model.policy.eval()

print(f"device            : {device}")
print(f"nominal samples   : {len(nom_obs)}")
print(f"high DAgger       : {len(high_obs)}")
print(f"low DAgger        : {len(low_obs)}")


# =====================================================================
# B — GROUPED SPLITS
# =====================================================================

print()
print("=" * 120)
print("B — GROUPED TRAIN / HELD-OUT SPLITS")
print("=" * 120)

nom_train = (
    nom_rollout
    != VALIDATION_ROLLOUT_ID
)
nom_val = (
    nom_rollout
    == VALIDATION_ROLLOUT_ID
)

high_train = (
    high_rollout
    != VALIDATION_ROLLOUT_ID
)
high_val = (
    high_rollout
    == VALIDATION_ROLLOUT_ID
)

low_train = (
    low_rollout
    != VALIDATION_ROLLOUT_ID
)
low_val = (
    low_rollout
    == VALIDATION_ROLLOUT_ID
)

for name, mask in [
    ("nominal val", nom_val),
    ("high val", high_val),
    ("low val", low_val),
]:
    if not np.any(mask):
        raise RuntimeError(
            f"Missing {name} split."
        )

print(
    f"nominal train/val : "
    f"{int(np.sum(nom_train))}/"
    f"{int(np.sum(nom_val))}"
)
print(
    f"high train/val    : "
    f"{int(np.sum(high_train))}/"
    f"{int(np.sum(high_val))}"
)
print(
    f"low train/val     : "
    f"{int(np.sum(low_train))}/"
    f"{int(np.sum(low_val))}"
)


# =====================================================================
# C — FIXED LATENT REPRESENTATION
# =====================================================================

print()
print("=" * 120)
print("C — EXTRACT FROZEN ACTOR LATENTS")
print("=" * 120)

nom_latent = actor_latent(
    base_model.policy,
    nom_obs,
    device,
)
high_latent = actor_latent(
    base_model.policy,
    high_obs,
    device,
)
low_latent = actor_latent(
    base_model.policy,
    low_obs,
    device,
)

def with_bias(latent):
    return np.concatenate(
        [
            latent,
            np.ones(
                (len(latent), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

nom_X = with_bias(
    nom_latent
)
high_X = with_bias(
    high_latent
)
low_X = with_bias(
    low_latent
)

base_weight = (
    base_model.policy.action_net
    .weight.detach()
    .cpu().numpy()
    .astype(np.float64)
)

base_bias = (
    base_model.policy.action_net
    .bias.detach()
    .cpu().numpy()
    .astype(np.float64)
)

beta0 = np.concatenate(
    [
        base_weight[0],
        np.asarray(
            [base_bias[0]],
            dtype=np.float64,
        ),
    ]
)

print(
    f"latent dim: "
    f"{nom_latent.shape[1]}"
)
print(
    "trainable scope: "
    "collective head row 0 only"
)


# =====================================================================
# D — V2 BASELINE HELD-OUT METRICS
# =====================================================================

print()
print("=" * 120)
print("D — CORRECTIVE V2 BASELINE METRICS")
print("=" * 120)

base_nom_pred = actor_output(
    base_model.policy,
    nom_obs[nom_val],
    device,
)
base_high_pred = actor_output(
    base_model.policy,
    high_obs[high_val],
    device,
)
base_low_pred = actor_output(
    base_model.policy,
    low_obs[low_val],
    device,
)

base_nom_rmse = rmse(
    base_nom_pred[:, 0]
    - nom_target[nom_val, 0]
)

base_high_rmse = rmse(
    base_high_pred[:, 0]
    - high_target[high_val, 0]
)

base_low_rmse = rmse(
    base_low_pred[:, 0]
    - low_target[low_val, 0]
)

base_low_phys = (
    denormalize_action(
        base_low_pred,
        action_low,
        action_high,
    )[:, 0]
)

base_low_phys_rmse = rmse(
    base_low_phys
    - low_act_phys[low_val, 0]
)

print(
    f"V2 nominal val collective RMSE : "
    f"{base_nom_rmse:.6f}"
)
print(
    f"V2 high DAgger collective RMSE : "
    f"{base_high_rmse:.6f}"
)
print(
    f"V2 low DAgger collective RMSE  : "
    f"{base_low_rmse:.6f}"
)
print(
    f"V2 low DAgger physical RMSE    : "
    f"{base_low_phys_rmse:.8f}"
)


# =====================================================================
# E — THREE-SOURCE GRID
# =====================================================================

print()
print("=" * 120)
print("E — THREE-SOURCE CORRECTIVE COLLECTIVE GRID")
print("=" * 120)

X_nom_train = nom_X[nom_train]
y_nom_train = (
    nom_target[
        nom_train,
        0,
    ].astype(np.float64)
)
w_nom_train = (
    nom_weight[
        nom_train
    ].astype(np.float64)
)

X_high_train = (
    high_X[high_train]
)
y_high_train = (
    high_target[
        high_train,
        0,
    ].astype(np.float64)
)
w_high_train = (
    high_weight[
        high_train
    ].astype(np.float64)
)

X_low_train = (
    low_X[low_train]
)
y_low_train = (
    low_target[
        low_train,
        0,
    ].astype(np.float64)
)
w_low_train = (
    low_weight[
        low_train
    ].astype(np.float64)
)

X_nom_val = nom_X[nom_val]
y_nom_val = (
    nom_target[
        nom_val,
        0,
    ].astype(np.float64)
)

X_high_val = high_X[high_val]
y_high_val = (
    high_target[
        high_val,
        0,
    ].astype(np.float64)
)

X_low_val = low_X[low_val]
y_low_val = (
    low_target[
        low_val,
        0,
    ].astype(np.float64)
)

# Normalize each source internally first.
wn = (
    w_nom_train
    / max(
        1e-12,
        float(
            np.mean(
                w_nom_train
            )
        ),
    )
)

wh0 = (
    w_high_train
    / max(
        1e-12,
        float(
            np.mean(
                w_high_train
            )
        ),
    )
)

wl0 = (
    w_low_train
    / max(
        1e-12,
        float(
            np.mean(
                w_low_train
            )
        ),
    )
)

rows = []
best = None

for high_global in (
    HIGH_DAGGER_GLOBAL_WEIGHTS
):
    for low_global in (
        LOW_DAGGER_GLOBAL_WEIGHTS
    ):
        wh = (
            wh0
            * float(high_global)
        )
        wl = (
            wl0
            * float(low_global)
        )

        X_train = np.concatenate(
            [
                X_nom_train,
                X_high_train,
                X_low_train,
            ],
            axis=0,
        )

        y_train = np.concatenate(
            [
                y_nom_train,
                y_high_train,
                y_low_train,
            ],
            axis=0,
        )

        w_train = np.concatenate(
            [
                wn,
                wh,
                wl,
            ],
            axis=0,
        )

        for ridge_factor in (
            RIDGE_FACTORS
        ):
            beta, lam = (
                fit_centered_weighted_ridge(
                    X_train,
                    y_train,
                    w_train,
                    beta0,
                    ridge_factor,
                )
            )

            pred_nom = np.clip(
                X_nom_val @ beta,
                -1.0,
                +1.0,
            )

            pred_high = np.clip(
                X_high_val @ beta,
                -1.0,
                +1.0,
            )

            pred_low = np.clip(
                X_low_val @ beta,
                -1.0,
                +1.0,
            )

            nom_rmse = rmse(
                pred_nom
                - y_nom_val
            )
            high_rmse = rmse(
                pred_high
                - y_high_val
            )
            low_rmse = rmse(
                pred_low
                - y_low_val
            )

            low_improvement = (
                1.0
                - low_rmse
                / max(
                    1e-12,
                    base_low_rmse,
                )
            )

            high_ratio = (
                high_rmse
                / max(
                    1e-12,
                    base_high_rmse,
                )
            )

            feasible = bool(
                nom_rmse
                <= MAX_NOMINAL_VAL_RMSE
                and high_ratio
                <= MAX_HIGH_DAGGER_DEGRADATION
                and low_improvement
                >= MIN_LOW_DAGGER_IMPROVEMENT
            )

            # Low-alt correction is primary, but high-alt and nominal
            # preservation remain significant.
            score = (
                low_rmse
                + 0.50 * high_rmse
                + 0.35 * nom_rmse
            )

            row = {
                "high_dagger_global_weight": float(
                    high_global
                ),
                "low_dagger_global_weight": float(
                    low_global
                ),
                "ridge_factor": float(
                    ridge_factor
                ),
                "lambda": float(lam),
                "nominal_val_rmse": float(
                    nom_rmse
                ),
                "high_dagger_val_rmse": float(
                    high_rmse
                ),
                "high_dagger_ratio_vs_v2": float(
                    high_ratio
                ),
                "low_dagger_val_rmse": float(
                    low_rmse
                ),
                "low_dagger_improvement_fraction": float(
                    low_improvement
                ),
                "score": float(
                    score
                ),
                "feasible": bool(
                    feasible
                ),
            }

            rows.append(row)

            if feasible:
                if (
                    best is None
                    or score
                    < best["score"]
                ):
                    best = {
                        **row,
                        "beta": beta,
                    }

            print(
                f"H={high_global:3.1f} "
                f"L={low_global:4.1f} "
                f"ridge={ridge_factor:.1e} | "
                f"nom={nom_rmse:.6f} | "
                f"high={high_rmse:.6f} "
                f"({high_ratio:.3f}x) | "
                f"low={low_rmse:.6f} "
                f"improve={100*low_improvement:6.2f}% | "
                f"ok={feasible}"
            )

write_csv(
    OUT_RESULT_DIR
    / "candidate_grid.csv",
    rows,
)

if best is None:
    raise RuntimeError(
        "No V3 collective-head candidate satisfied "
        "nominal + high-alt preservation and low-alt improvement gates."
    )


# =====================================================================
# F — WRITE ROW 0 ONLY
# =====================================================================

print()
print("=" * 120)
print("F — BUILD CORRECTIVE DISTILLED V3")
print("=" * 120)

model_v3 = PPO.load(
    str(BASE_MODEL),
    device=device,
)

before_weight = (
    model_v3.policy.action_net
    .weight.detach()
    .cpu().numpy()
    .copy()
)

before_bias = (
    model_v3.policy.action_net
    .bias.detach()
    .cpu().numpy()
    .copy()
)

beta = best["beta"]

with torch.no_grad():
    model_v3.policy.action_net.weight[
        0
    ].copy_(
        torch.as_tensor(
            beta[:-1],
            dtype=(
                model_v3.policy
                .action_net.weight.dtype
            ),
            device=(
                model_v3.policy
                .action_net.weight.device
            ),
        )
    )

    model_v3.policy.action_net.bias[
        0
    ].copy_(
        torch.as_tensor(
            beta[-1],
            dtype=(
                model_v3.policy
                .action_net.bias.dtype
            ),
            device=(
                model_v3.policy
                .action_net.bias.device
            ),
        )
    )

after_weight = (
    model_v3.policy.action_net
    .weight.detach()
    .cpu().numpy()
    .copy()
)

after_bias = (
    model_v3.policy.action_net
    .bias.detach()
    .cpu().numpy()
    .copy()
)

if not np.array_equal(
    before_weight[1:],
    after_weight[1:],
):
    raise RuntimeError(
        "Rows 1..3 changed unexpectedly."
    )

if not np.array_equal(
    before_bias[1:],
    after_bias[1:],
):
    raise RuntimeError(
        "Bias rows 1..3 changed unexpectedly."
    )

OUT_MODEL_STEM = (
    OUT_MODEL_DIR
    / "AH1S_STAGE4_CORRECTIVE_DISTILLED_V3"
)

model_v3.save(
    OUT_MODEL_STEM
)

OUT_MODEL_ZIP = Path(
    str(OUT_MODEL_STEM)
    + ".zip"
)

# Preserve exact V2 normalization/action interface.
np.savez(
    OUT_MODEL_DIR
    / "stage4_obs_normalization.npz",
    mean=obs_mean,
    std=obs_std,
    std_raw=(
        norm_pack["std_raw"]
        if "std_raw"
        in norm_pack.files
        else obs_std
    ),
)

np.savez(
    OUT_MODEL_DIR
    / "stage4_action_mapping.npz",
    physical_low=action_low,
    physical_high=action_high,
)


# =====================================================================
# G — FINAL HELD-OUT COMPARISON
# =====================================================================

print()
print("=" * 120)
print("G — FINAL V3 HELD-OUT COMPARISON")
print("=" * 120)

model_v3.policy.eval()

v3_nom_pred = actor_output(
    model_v3.policy,
    nom_obs[nom_val],
    device,
)
v3_high_pred = actor_output(
    model_v3.policy,
    high_obs[high_val],
    device,
)
v3_low_pred = actor_output(
    model_v3.policy,
    low_obs[low_val],
    device,
)

v3_nom_rmse = rmse(
    v3_nom_pred[:, 0]
    - nom_target[nom_val, 0]
)

v3_high_rmse = rmse(
    v3_high_pred[:, 0]
    - high_target[high_val, 0]
)

v3_low_rmse = rmse(
    v3_low_pred[:, 0]
    - low_target[low_val, 0]
)

v3_low_phys = (
    denormalize_action(
        v3_low_pred,
        action_low,
        action_high,
    )[:, 0]
)

v3_low_phys_rmse = rmse(
    v3_low_phys
    - low_act_phys[low_val, 0]
)

nom_noncoll_delta = float(
    np.max(
        np.abs(
            v3_nom_pred[:, 1:]
            - base_nom_pred[:, 1:]
        )
    )
)

high_noncoll_delta = float(
    np.max(
        np.abs(
            v3_high_pred[:, 1:]
            - base_high_pred[:, 1:]
        )
    )
)

low_noncoll_delta = float(
    np.max(
        np.abs(
            v3_low_pred[:, 1:]
            - base_low_pred[:, 1:]
        )
    )
)

print(
    f"selected high weight : "
    f"{best['high_dagger_global_weight']}"
)
print(
    f"selected low weight  : "
    f"{best['low_dagger_global_weight']}"
)
print(
    f"selected ridge       : "
    f"{best['ridge_factor']:.1e}"
)

print(
    f"nominal RMSE "
    f"{base_nom_rmse:.6f} -> "
    f"{v3_nom_rmse:.6f}"
)
print(
    f"high DAgger RMSE "
    f"{base_high_rmse:.6f} -> "
    f"{v3_high_rmse:.6f}"
)
print(
    f"low DAgger RMSE "
    f"{base_low_rmse:.6f} -> "
    f"{v3_low_rmse:.6f}"
)
print(
    f"low physical collective RMSE "
    f"{base_low_phys_rmse:.8f} -> "
    f"{v3_low_phys_rmse:.8f}"
)

print(
    "non-collective max output delta | "
    f"nominal={nom_noncoll_delta:.12g} "
    f"high={high_noncoll_delta:.12g} "
    f"low={low_noncoll_delta:.12g}"
)

ready = bool(
    v3_nom_rmse
    <= MAX_NOMINAL_VAL_RMSE
    and v3_high_rmse
    <= (
        base_high_rmse
        * MAX_HIGH_DAGGER_DEGRADATION
    )
    and v3_low_rmse
    <= (
        base_low_rmse
        * (
            1.0
            - MIN_LOW_DAGGER_IMPROVEMENT
        )
    )
    and nom_noncoll_delta
    <= 1e-9
    and high_noncoll_delta
    <= 1e-9
    and low_noncoll_delta
    <= 1e-9
)

summary = {
    "training_type": (
        "corrective policy distillation / centered weighted ridge"
    ),
    "reinforcement_learning_used": False,
    "base_model": str(BASE_MODEL),
    "output_model": str(
        OUT_MODEL_ZIP
    ),
    "changed_actor_scope": (
        "action_net row 0 only (collective)"
    ),
    "non_collective_actor_rows_changed": False,
    "validation_rollout_id": int(
        VALIDATION_ROLLOUT_ID
    ),
    "baseline_v2": {
        "nominal_val_collective_rmse": float(
            base_nom_rmse
        ),
        "high_dagger_val_collective_rmse": float(
            base_high_rmse
        ),
        "low_dagger_val_collective_rmse": float(
            base_low_rmse
        ),
        "low_dagger_val_collective_physical_rmse": float(
            base_low_phys_rmse
        ),
    },
    "selected": {
        k: v
        for k, v in best.items()
        if k != "beta"
    },
    "corrected_v3": {
        "nominal_val_collective_rmse": float(
            v3_nom_rmse
        ),
        "high_dagger_val_collective_rmse": float(
            v3_high_rmse
        ),
        "low_dagger_val_collective_rmse": float(
            v3_low_rmse
        ),
        "low_dagger_val_collective_physical_rmse": float(
            v3_low_phys_rmse
        ),
        "nominal_noncollective_max_output_delta": float(
            nom_noncoll_delta
        ),
        "high_noncollective_max_output_delta": float(
            high_noncoll_delta
        ),
        "low_noncollective_max_output_delta": float(
            low_noncoll_delta
        ),
    },
    "ready_for_fresh_teacher_off_validation": bool(
        ready
    ),
}

with open(
    OUT_RESULT_DIR
    / "final_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

with open(
    OUT_MODEL_DIR
    / "stage4_policy_interface.json",
    "w",
) as f:
    json.dump(
        {
            "observation_dim": 26,
            "normalization": "unchanged from corrective V2",
            "action_mapping": "unchanged from corrective V2",
            "corrective_change": (
                "collective actor-head row 0 only; "
                "nominal + high DAgger + low DAgger"
            ),
            "reinforcement_learning_used": False,
        },
        f,
        indent=2,
    )


print()
print("=" * 120)
print("H — CORRECTIVE DISTILLATION V3 CONCLUSION")
print("=" * 120)

print(
    "CORRECTIVE DISTILLATION V3 COMPLETE: True"
)
print(
    "REWARD-BASED PPO TRAINING USED: False"
)
print(
    "NON-COLLECTIVE ACTOR OUTPUTS PRESERVED: "
    f"{nom_noncoll_delta <= 1e-9 and high_noncoll_delta <= 1e-9 and low_noncoll_delta <= 1e-9}"
)
print(
    "READY FOR FRESH TEACHER-OFF VALIDATION: "
    f"{ready}"
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
    "Do NOT call this RL. "
    "Next: fresh teacher-OFF JSBSim validation with V3."
)
