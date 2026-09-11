from pathlib import Path
import csv
import json
import shutil
import hashlib

import numpy as np
import torch
from stable_baselines3 import PPO


# =====================================================================
# STAGE-4 V5 NEAR-GROUND LOCALIZED CORRECTIVE DISTILLATION
# =====================================================================
#
# IMPORTANT METHODOLOGY
#   - NOT reinforcement learning.
#   - Locked V2 PPO weights remain unchanged.
#   - Locked V4 descent adapter remains unchanged.
#   - A SECOND additive collective residual adapter is learned.
#   - V5 adapter is active ONLY in:
#         near_ground_30_to_native_eq
#   - Elevator / aileron / rudder remain exactly unchanged.
#
# Runtime after this script:
#
#   base_z = locked V2 PPO actor
#
#   z0 =
#       base_z0
#       + V4_descent_gate(raw_obs) * V4_residual(latent)
#       + V5_nearground_gate(raw_obs) * V5_residual(latent)
#
#   z1..z3 = base_z1..z3
#
# V4 and V5 phase gates are mutually exclusive.
# =====================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# LOCKED V4 INPUTS
# ---------------------------------------------------------------------

V4_MODEL_DIR = Path(
    "models_stage4_localized_v4"
)

BASE_MODEL = (
    V4_MODEL_DIR
    / "AH1S_STAGE4_LOCALIZED_V4_BASE.zip"
)

OBS_NORM_PATH = (
    V4_MODEL_DIR
    / "stage4_obs_normalization.npz"
)

ACTION_MAP_PATH = (
    V4_MODEL_DIR
    / "stage4_action_mapping.npz"
)

V4_ADAPTER_PATH = (
    V4_MODEL_DIR
    / "stage4_localized_collective_adapter.npz"
)

V4_INTERFACE_PATH = (
    V4_MODEL_DIR
    / "stage4_policy_interface.json"
)

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

NOMINAL_DIR = Path(
    "results_stage4_teacher_rollouts_v4"
)

V5_DAGGER_DIR = Path(
    "results_stage4_dagger_nearground_v5"
)

# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------

OUT_MODEL_DIR = Path(
    "models_stage4_localized_v5"
)

OUT_RESULT_DIR = Path(
    "results_stage4_localized_v5"
)

OUT_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------------------
# STAGE-4 OBSERVATION CONTRACT
# ---------------------------------------------------------------------
#
# raw[0]  = altitude_ft
#
# phase one-hot starts at raw[20]:
#   20 pre_descent_hold
#   21 descent_300_to_30
#   22 near_ground_30_to_native_eq
#   23 capture_to_9ft
#   24 continuous_touchdown
#   25 landed_hold
# ---------------------------------------------------------------------

ALTITUDE_INDEX = 0
NEAR_GROUND_PHASE_INDEX = 22

VALIDATION_ROLLOUT_ID = 5

# ---------------------------------------------------------------------
# PRESERVATION / IMPROVEMENT GATES
# ---------------------------------------------------------------------
#
# Nominal Stage-4 teacher replay in the near-ground phase is already
# imitated very accurately by the locked student. V5 is therefore
# rejected if it damages that behavior materially.
#
# DAgger data is the real student-visited covariate-shift region and
# must improve strongly.
# ---------------------------------------------------------------------

MAX_NOMINAL_NEARGROUND_ABS_RMSE = 0.0065
MAX_NOMINAL_NEARGROUND_DEGRADATION = 1.35

MIN_DAGGER_IMPROVEMENT = 0.50
MIN_FLOOR_DAGGER_IMPROVEMENT = 0.50

# Teacher floor observed in the qualified near-ground controller.
TEACHER_FLOOR_PHYSICAL = 0.5900000
TEACHER_FLOOR_TOL = 2.0e-5

# Candidate grid.
DAGGER_GLOBAL_WEIGHTS = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    4.00,
    6.00,
    8.00,
]

RIDGE_FACTORS = [
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
]


def rule(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def rmse(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )
    return float(
        np.sqrt(
            np.mean(
                np.square(x)
            )
        )
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
    ).astype(
        np.float32
    )


def denormalize_action(
    z,
    low,
    high,
):
    return (
        low
        +
        0.5
        * (z + 1.0)
        * (high - low)
    ).astype(
        np.float32
    )


def write_csv(
    path,
    rows,
):
    if not rows:
        raise RuntimeError(
            f"No rows to write: {path}"
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def phase_gate_nearground(
    raw_obs,
):
    return (
        raw_obs[
            :,
            NEAR_GROUND_PHASE_INDEX
        ]
        >
        0.5
    ).astype(
        np.float64
    )


def smoothstep_descent_gate(
    raw_obs,
    v4_adapter,
):
    alt = raw_obs[
        :,
        int(
            v4_adapter[
                "altitude_index"
            ]
        ),
    ]

    phase = (
        raw_obs[
            :,
            int(
                v4_adapter[
                    "phase_descent_index"
                ]
            ),
        ]
        >
        0.5
    ).astype(
        np.float64
    )

    full = float(
        v4_adapter[
            "gate_full_below_ft"
        ]
    )

    zero = float(
        v4_adapter[
            "gate_zero_above_ft"
        ]
    )

    x = np.clip(
        (zero - alt)
        /
        max(
            1e-12,
            zero - full,
        ),
        0.0,
        1.0,
    )

    smooth = (
        x
        * x
        * (
            3.0
            -
            2.0 * x
        )
    )

    return (
        phase
        *
        smooth
    ).astype(
        np.float64
    )


@torch.no_grad()
def latent_and_base_action(
    policy,
    obs_norm,
    device,
    batch_size=4096,
):
    latents = []
    actions = []

    for start in range(
        0,
        len(obs_norm),
        batch_size,
    ):
        x = torch.as_tensor(
            obs_norm[
                start:
                start + batch_size
            ],
            dtype=torch.float32,
            device=device,
        )

        features = (
            policy.extract_features(
                x
            )
        )

        latent = (
            policy
            .mlp_extractor
            .forward_actor(
                features
            )
        )

        action = (
            policy.action_net(
                latent
            )
        )

        action = torch.clamp(
            action,
            -1.0,
            +1.0,
        )

        latents.append(
            latent
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        actions.append(
            action
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    return (
        np.concatenate(
            latents,
            axis=0,
        ),
        np.concatenate(
            actions,
            axis=0,
        ),
    )


def phi(
    latent,
):
    return np.c_[
        latent,
        np.ones(
            len(latent),
            dtype=np.float64,
        ),
    ]


def normalize_weights(
    weights,
):
    w = np.asarray(
        weights,
        dtype=np.float64,
    )

    return (
        w
        /
        max(
            1e-12,
            float(
                np.mean(w)
            ),
        )
    )


def fit_zero_centered_weighted_ridge(
    X,
    y,
    weights,
    ridge_factor,
):
    """
    Solve:

        min_beta
            sum_i w_i (X_i beta - y_i)^2
            + lambda ||beta||^2

    Zero-centered regularization matters here:
    beta=0 means "do not change the locked policy".
    """

    w = normalize_weights(
        weights
    )

    sw = np.sqrt(
        w
    )[:, None]

    Xw = X * sw
    yw = y * sw[:, 0]

    XTX = (
        Xw.T
        @
        Xw
    )

    XTy = (
        Xw.T
        @
        yw
    )

    scale = max(
        1e-12,
        float(
            np.mean(
                np.diag(
                    XTX
                )
            )
        ),
    )

    lam = float(
        ridge_factor
        *
        scale
    )

    beta = np.linalg.solve(
        XTX
        +
        lam
        *
        np.eye(
            X.shape[1],
            dtype=np.float64,
        ),
        XTy,
    )

    return (
        beta,
        lam,
    )


def load_v4_adapter():
    pack = np.load(
        V4_ADAPTER_PATH
    )

    return {
        "beta": (
            pack[
                "beta"
            ]
            .astype(
                np.float64
            )
        ),
        "gate_full_below_ft": float(
            pack[
                "gate_full_below_ft"
            ][0]
        ),
        "gate_zero_above_ft": float(
            pack[
                "gate_zero_above_ft"
            ][0]
        ),
        "altitude_index": int(
            pack[
                "altitude_index"
            ][0]
        ),
        "phase_descent_index": int(
            pack[
                "phase_descent_index"
            ][0]
        ),
    }


def apply_v4(
    base_action,
    Phi,
    raw_obs,
    v4_adapter,
):
    out = base_action.copy()

    g = smoothstep_descent_gate(
        raw_obs,
        v4_adapter,
    )

    correction = (
        g
        *
        (
            Phi
            @
            v4_adapter[
                "beta"
            ]
        )
    )

    out[
        :,
        0,
    ] = np.clip(
        out[
            :,
            0,
        ]
        +
        correction,
        -1.0,
        +1.0,
    )

    return (
        out,
        correction,
    )


def apply_v5_on_top_of_v4(
    v4_action,
    Phi,
    raw_obs,
    beta,
):
    out = v4_action.copy()

    gate = phase_gate_nearground(
        raw_obs
    )

    correction = (
        gate
        *
        (
            Phi
            @
            beta
        )
    )

    out[
        :,
        0,
    ] = np.clip(
        out[
            :,
            0,
        ]
        +
        correction,
        -1.0,
        +1.0,
    )

    return (
        out,
        correction,
    )


# =====================================================================
# A — LOAD LOCKED V4 + NOMINAL + V5 DAgger
# =====================================================================

rule(
    "A — LOAD LOCKED V4 + NOMINAL + V5 NEAR-GROUND DAgger"
)

required = [
    BASE_MODEL,
    OBS_NORM_PATH,
    ACTION_MAP_PATH,
    V4_ADAPTER_PATH,

    NOMINAL_DIR
    / "stage4_obs_raw.npy",

    NOMINAL_DIR
    / "stage4_actions_physical.npy",

    NOMINAL_DIR
    / "stage4_phase_ids.npy",

    NOMINAL_DIR
    / "stage4_rollout_ids.npy",

    NOMINAL_DIR
    / "stage4_sample_weights.npy",

    V5_DAGGER_DIR
    / "dagger_obs_raw.npy",

    V5_DAGGER_DIR
    / "dagger_teacher_actions_physical.npy",

    V5_DAGGER_DIR
    / "dagger_student_actions_physical.npy",

    V5_DAGGER_DIR
    / "dagger_phase_ids.npy",

    V5_DAGGER_DIR
    / "dagger_rollout_ids.npy",

    V5_DAGGER_DIR
    / "dagger_priority_weights.npy",
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

norm_pack = np.load(
    OBS_NORM_PATH
)

obs_mean = (
    norm_pack[
        "mean"
    ]
    .astype(
        np.float32
    )
)

obs_std = (
    norm_pack[
        "std"
    ]
    .astype(
        np.float32
    )
)

action_pack = np.load(
    ACTION_MAP_PATH
)

physical_low = (
    action_pack[
        "physical_low"
    ]
    .astype(
        np.float32
    )
)

physical_high = (
    action_pack[
        "physical_high"
    ]
    .astype(
        np.float32
    )
)

v4_adapter = (
    load_v4_adapter()
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = PPO.load(
    str(
        BASE_MODEL
    ),
    device=device,
)

model.policy.eval()

nom_raw_all = np.load(
    NOMINAL_DIR
    / "stage4_obs_raw.npy"
).astype(
    np.float32
)

nom_act_all = np.load(
    NOMINAL_DIR
    / "stage4_actions_physical.npy"
).astype(
    np.float32
)

nom_phase_all = np.load(
    NOMINAL_DIR
    / "stage4_phase_ids.npy"
).astype(
    np.int64
)

nom_rid_all = np.load(
    NOMINAL_DIR
    / "stage4_rollout_ids.npy"
).astype(
    np.int64
)

nom_w_all = np.load(
    NOMINAL_DIR
    / "stage4_sample_weights.npy"
).astype(
    np.float32
)

# Near-ground nominal replay only.
nom_mask = (
    nom_phase_all
    ==
    2
)

if not np.any(
    nom_mask
):
    raise RuntimeError(
        "Nominal near-ground phase_id=2 samples missing."
    )

nom_raw = (
    nom_raw_all[
        nom_mask
    ]
)

nom_act_phys = (
    nom_act_all[
        nom_mask
    ]
)

nom_rid = (
    nom_rid_all[
        nom_mask
    ]
)

nom_w = (
    nom_w_all[
        nom_mask
    ]
)

dag_raw = np.load(
    V5_DAGGER_DIR
    / "dagger_obs_raw.npy"
).astype(
    np.float32
)

dag_act_phys_raw = np.load(
    V5_DAGGER_DIR
    / "dagger_teacher_actions_physical.npy"
).astype(
    np.float32
)

dag_student_phys = np.load(
    V5_DAGGER_DIR
    / "dagger_student_actions_physical.npy"
).astype(
    np.float32
)

dag_phase = np.load(
    V5_DAGGER_DIR
    / "dagger_phase_ids.npy"
).astype(
    np.int64
)

dag_rid = np.load(
    V5_DAGGER_DIR
    / "dagger_rollout_ids.npy"
).astype(
    np.int64
)

dag_w = np.load(
    V5_DAGGER_DIR
    / "dagger_priority_weights.npy"
).astype(
    np.float32
)

if (
    nom_raw.ndim != 2
    or
    nom_raw.shape[1] != 26
    or
    dag_raw.ndim != 2
    or
    dag_raw.shape[1] != 26
):
    raise RuntimeError(
        "Stage-4 observation must be 26-D."
    )

if not np.all(
    dag_phase
    ==
    2
):
    raise RuntimeError(
        "V5 DAgger must contain only phase_id=2."
    )

if not np.all(
    dag_raw[
        :,
        NEAR_GROUND_PHASE_INDEX
    ]
    >
    0.5
):
    raise RuntimeError(
        "V5 DAgger observation phase one-hot does not match near-ground."
    )

# Nominal teacher targets are a hard interface contract.
if (
    not np.all(
        nom_act_phys
        >=
        physical_low
        -
        1e-7
    )
    or
    not np.all(
        nom_act_phys
        <=
        physical_high
        +
        1e-7
    )
):
    raise RuntimeError(
        "Nominal near-ground targets violate locked action map."
    )

# V5 corrective teacher collective is expected to remain inside the
# locked policy envelope. We do not silently widen that interface.
if (
    np.min(
        dag_act_phys_raw[
            :,
            0
        ]
    )
    <
    physical_low[0]
    -
    1e-7
    or
    np.max(
        dag_act_phys_raw[
            :,
            0
        ]
    )
    >
    physical_high[0]
    +
    1e-7
):
    raise RuntimeError(
        "V5 DAgger collective target violates locked action map."
    )

nom_obs = (
    (
        nom_raw
        -
        obs_mean
    )
    /
    obs_std
).astype(
    np.float32
)

dag_obs = (
    (
        dag_raw
        -
        obs_mean
    )
    /
    obs_std
).astype(
    np.float32
)

nom_target = (
    normalize_action_physical(
        nom_act_phys,
        physical_low,
        physical_high,
    )
)

dag_target = (
    normalize_action_physical(
        dag_act_phys_raw,
        physical_low,
        physical_high,
    )
)

nom_latent, nom_base = (
    latent_and_base_action(
        model.policy,
        nom_obs,
        device,
    )
)

dag_latent, dag_base = (
    latent_and_base_action(
        model.policy,
        dag_obs,
        device,
    )
)

nom_phi = phi(
    nom_latent
)

dag_phi = phi(
    dag_latent
)

nom_v4, nom_v4_corr = (
    apply_v4(
        nom_base,
        nom_phi,
        nom_raw,
        v4_adapter,
    )
)

dag_v4, dag_v4_corr = (
    apply_v4(
        dag_base,
        dag_phi,
        dag_raw,
        v4_adapter,
    )
)

# V4 adapter MUST be off in near-ground.
if (
    np.max(
        np.abs(
            nom_v4_corr
        )
    )
    >
    1e-12
    or
    np.max(
        np.abs(
            dag_v4_corr
        )
    )
    >
    1e-12
):
    raise RuntimeError(
        "Locked V4 descent adapter unexpectedly active in near-ground."
    )

# Verify the collected DAgger student is actually the same locked V4
# policy/interface that we are about to correct.
dag_v4_phys = (
    denormalize_action(
        dag_v4,
        physical_low,
        physical_high,
    )
)

stored_student_delta = float(
    np.max(
        np.abs(
            dag_v4_phys
            -
            dag_student_phys
        )
    )
)

print(
    "device:",
    device,
)

print(
    "nominal near-ground samples:",
    len(
        nom_raw
    ),
)

print(
    "V5 DAgger samples:",
    len(
        dag_raw
    ),
)

print(
    "nominal train/val:",
    int(
        np.sum(
            nom_rid
            !=
            VALIDATION_ROLLOUT_ID
        )
    ),
    "/",
    int(
        np.sum(
            nom_rid
            ==
            VALIDATION_ROLLOUT_ID
        )
    ),
)

print(
    "DAgger train/val:",
    int(
        np.sum(
            dag_rid
            !=
            VALIDATION_ROLLOUT_ID
        )
    ),
    "/",
    int(
        np.sum(
            dag_rid
            ==
            VALIDATION_ROLLOUT_ID
        )
    ),
)

print(
    "V4 vs stored V5-DAgger student max physical delta:",
    f"{stored_student_delta:.8f}",
)

if (
    stored_student_delta
    >
    2e-4
):
    raise RuntimeError(
        "V5 DAgger student and locked V4 runtime do not match."
    )

# Mutually exclusive phase audit.
if np.any(
    (
        nom_raw[
            :,
            int(
                v4_adapter[
                    "phase_descent_index"
                ]
            ),
        ]
        >
        0.5
    )
    &
    (
        nom_raw[
            :,
            NEAR_GROUND_PHASE_INDEX
        ]
        >
        0.5
    )
):
    raise RuntimeError(
        "Descent and near-ground phase one-hot unexpectedly overlap."
    )


# =====================================================================
# B — LOCKED V4 BASELINE METRICS
# =====================================================================

rule(
    "B — LOCKED V4 BASELINE METRICS"
)

nom_train = (
    nom_rid
    !=
    VALIDATION_ROLLOUT_ID
)

nom_val = (
    nom_rid
    ==
    VALIDATION_ROLLOUT_ID
)

dag_train = (
    dag_rid
    !=
    VALIDATION_ROLLOUT_ID
)

dag_val = (
    dag_rid
    ==
    VALIDATION_ROLLOUT_ID
)

if (
    not np.any(
        nom_train
    )
    or
    not np.any(
        nom_val
    )
    or
    not np.any(
        dag_train
    )
    or
    not np.any(
        dag_val
    )
):
    raise RuntimeError(
        "Train/validation split is incomplete."
    )

base_nom_rmse = rmse(
    nom_v4[
        nom_val,
        0
    ]
    -
    nom_target[
        nom_val,
        0
    ]
)

base_dag_rmse = rmse(
    dag_v4[
        dag_val,
        0
    ]
    -
    dag_target[
        dag_val,
        0
    ]
)

base_nom_phys = (
    denormalize_action(
        nom_v4[
            nom_val
        ],
        physical_low,
        physical_high,
    )[
        :,
        0
    ]
)

base_dag_phys = (
    denormalize_action(
        dag_v4[
            dag_val
        ],
        physical_low,
        physical_high,
    )[
        :,
        0
    ]
)

base_nom_phys_rmse = rmse(
    base_nom_phys
    -
    nom_act_phys[
        nom_val,
        0
    ]
)

base_dag_phys_rmse = rmse(
    base_dag_phys
    -
    dag_act_phys_raw[
        dag_val,
        0
    ]
)

dag_floor = (
    dag_act_phys_raw[
        :,
        0
    ]
    <=
    TEACHER_FLOOR_PHYSICAL
    +
    TEACHER_FLOOR_TOL
)

dag_floor_val = (
    dag_val
    &
    dag_floor
)

if not np.any(
    dag_floor_val
):
    raise RuntimeError(
        "Held-out V5 DAgger has no teacher-floor samples."
    )

base_floor_rmse = rmse(
    dag_v4[
        dag_floor_val,
        0
    ]
    -
    dag_target[
        dag_floor_val,
        0
    ]
)

print(
    f"V4 nominal near-ground RMSE : "
    f"{base_nom_rmse:.6f}"
)

print(
    f"V4 nominal physical RMSE    : "
    f"{base_nom_phys_rmse:.8f}"
)

print(
    f"V4 V5-DAgger RMSE           : "
    f"{base_dag_rmse:.6f}"
)

print(
    f"V4 V5-DAgger physical RMSE  : "
    f"{base_dag_phys_rmse:.8f}"
)

print(
    f"V4 teacher-floor RMSE       : "
    f"{base_floor_rmse:.6f}"
)

print(
    "held-out floor samples:",
    int(
        np.sum(
            dag_floor_val
        )
    ),
)

nominal_allowed = min(
    MAX_NOMINAL_NEARGROUND_ABS_RMSE,
    max(
        1e-12,
        MAX_NOMINAL_NEARGROUND_DEGRADATION
        *
        base_nom_rmse,
    ),
)

print(
    f"V5 nominal preservation limit: "
    f"{nominal_allowed:.6f}"
)


# =====================================================================
# C — FIT PHASE-LOCALIZED V5 RESIDUAL
# =====================================================================

rule(
    "C — FIT PHASE-LOCALIZED V5 RESIDUAL"
)

# Current V4 -> teacher residual targets.
nom_y = (
    nom_target[
        :,
        0
    ]
    -
    nom_v4[
        :,
        0
    ]
).astype(
    np.float64
)

dag_y = (
    dag_target[
        :,
        0
    ]
    -
    dag_v4[
        :,
        0
    ]
).astype(
    np.float64
)

# Phase gate is exactly 1 for these two training sources.
nom_gate = phase_gate_nearground(
    nom_raw
)

dag_gate = phase_gate_nearground(
    dag_raw
)

if (
    not np.all(
        nom_gate
        ==
        1.0
    )
    or
    not np.all(
        dag_gate
        ==
        1.0
    )
):
    raise RuntimeError(
        "Near-ground fit sources must have gate=1."
    )

X_nom = (
    nom_gate[
        nom_train,
        None
    ]
    *
    nom_phi[
        nom_train
    ]
)

y_nom = (
    nom_y[
        nom_train
    ]
)

w_nom = normalize_weights(
    nom_w[
        nom_train
    ]
)

X_dag = (
    dag_gate[
        dag_train,
        None
    ]
    *
    dag_phi[
        dag_train
    ]
)

y_dag = (
    dag_y[
        dag_train
    ]
)

w_dag_base = normalize_weights(
    dag_w[
        dag_train
    ]
)

rows = []
best = None

for dagger_weight in DAGGER_GLOBAL_WEIGHTS:

    X = np.concatenate(
        [
            X_nom,
            X_dag,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            y_nom,
            y_dag,
        ],
        axis=0,
    )

    w = np.concatenate(
        [
            w_nom,
            w_dag_base
            *
            float(
                dagger_weight
            ),
        ],
        axis=0,
    )

    for ridge_factor in RIDGE_FACTORS:

        beta, lam = (
            fit_zero_centered_weighted_ridge(
                X,
                y,
                w,
                ridge_factor,
            )
        )

        nom_pred, _ = (
            apply_v5_on_top_of_v4(
                nom_v4[
                    nom_val
                ],
                nom_phi[
                    nom_val
                ],
                nom_raw[
                    nom_val
                ],
                beta,
            )
        )

        dag_pred, _ = (
            apply_v5_on_top_of_v4(
                dag_v4[
                    dag_val
                ],
                dag_phi[
                    dag_val
                ],
                dag_raw[
                    dag_val
                ],
                beta,
            )
        )

        floor_pred, _ = (
            apply_v5_on_top_of_v4(
                dag_v4[
                    dag_floor_val
                ],
                dag_phi[
                    dag_floor_val
                ],
                dag_raw[
                    dag_floor_val
                ],
                beta,
            )
        )

        nom_rmse = rmse(
            nom_pred[
                :,
                0
            ]
            -
            nom_target[
                nom_val,
                0
            ]
        )

        dag_rmse = rmse(
            dag_pred[
                :,
                0
            ]
            -
            dag_target[
                dag_val,
                0
            ]
        )

        floor_rmse = rmse(
            floor_pred[
                :,
                0
            ]
            -
            dag_target[
                dag_floor_val,
                0
            ]
        )

        nominal_ratio = (
            nom_rmse
            /
            max(
                1e-12,
                base_nom_rmse,
            )
        )

        dagger_improvement = (
            1.0
            -
            dag_rmse
            /
            max(
                1e-12,
                base_dag_rmse,
            )
        )

        floor_improvement = (
            1.0
            -
            floor_rmse
            /
            max(
                1e-12,
                base_floor_rmse,
            )
        )

        feasible = bool(
            nom_rmse
            <=
            nominal_allowed
            and
            nominal_ratio
            <=
            MAX_NOMINAL_NEARGROUND_DEGRADATION
            +
            1e-12
            and
            dagger_improvement
            >=
            MIN_DAGGER_IMPROVEMENT
            and
            floor_improvement
            >=
            MIN_FLOOR_DAGGER_IMPROVEMENT
        )

        score = float(
            dag_rmse
            +
            0.60
            *
            floor_rmse
            +
            0.40
            *
            nom_rmse
        )

        row = {
            "dagger_global_weight": float(
                dagger_weight
            ),
            "ridge_factor": float(
                ridge_factor
            ),
            "lambda": float(
                lam
            ),
            "nominal_nearground_rmse": float(
                nom_rmse
            ),
            "nominal_ratio_vs_v4": float(
                nominal_ratio
            ),
            "dagger_rmse": float(
                dag_rmse
            ),
            "dagger_improvement_fraction": float(
                dagger_improvement
            ),
            "floor_rmse": float(
                floor_rmse
            ),
            "floor_improvement_fraction": float(
                floor_improvement
            ),
            "score": float(
                score
            ),
            "feasible": bool(
                feasible
            ),
        }

        rows.append(
            row
        )

        if feasible:
            if (
                best is None
                or
                score
                <
                best[
                    "score"
                ]
            ):
                best = {
                    **row,
                    "beta": beta.copy(),
                }

write_csv(
    OUT_RESULT_DIR
    /
    "candidate_grid.csv",
    rows,
)

print(
    "candidate count:",
    len(
        rows
    ),
)

print(
    "feasible count :",
    sum(
        bool(
            r[
                "feasible"
            ]
        )
        for r in rows
    ),
)

safe = [
    r
    for r in rows
    if (
        r[
            "nominal_nearground_rmse"
        ]
        <=
        nominal_allowed
        and
        r[
            "nominal_ratio_vs_v4"
        ]
        <=
        MAX_NOMINAL_NEARGROUND_DEGRADATION
        +
        1e-12
    )
]

safe.sort(
    key=lambda r: (
        -r[
            "floor_improvement_fraction"
        ],
        -r[
            "dagger_improvement_fraction"
        ],
        r[
            "score"
        ],
    )
)

print()
print(
    "Best nominal-preserving candidates:"
)

for r in safe[
    :15
]:
    print(
        f"D={r['dagger_global_weight']:4.2f} "
        f"ridge={r['ridge_factor']:.1e} | "
        f"nom={r['nominal_nearground_rmse']:.6f} "
        f"({r['nominal_ratio_vs_v4']:.3f}x) | "
        f"dag={r['dagger_rmse']:.6f} "
        f"improve={100*r['dagger_improvement_fraction']:.2f}% | "
        f"floor={r['floor_rmse']:.6f} "
        f"floorImprove={100*r['floor_improvement_fraction']:.2f}% | "
        f"ok={r['feasible']}"
    )

if best is None:
    raise RuntimeError(
        "No V5 near-ground adapter satisfied nominal preservation "
        "+ DAgger improvement + teacher-floor improvement gates."
    )


# =====================================================================
# D — FINAL V5 AUDIT
# =====================================================================

rule(
    "D — FINAL V5 AUDIT"
)

beta = best[
    "beta"
]

nom_v5, nom_corr = (
    apply_v5_on_top_of_v4(
        nom_v4,
        nom_phi,
        nom_raw,
        beta,
    )
)

dag_v5, dag_corr = (
    apply_v5_on_top_of_v4(
        dag_v4,
        dag_phi,
        dag_raw,
        beta,
    )
)

final_nom_rmse = rmse(
    nom_v5[
        nom_val,
        0
    ]
    -
    nom_target[
        nom_val,
        0
    ]
)

final_dag_rmse = rmse(
    dag_v5[
        dag_val,
        0
    ]
    -
    dag_target[
        dag_val,
        0
    ]
)

final_floor_rmse = rmse(
    dag_v5[
        dag_floor_val,
        0
    ]
    -
    dag_target[
        dag_floor_val,
        0
    ]
)

final_dagger_improvement = (
    1.0
    -
    final_dag_rmse
    /
    max(
        1e-12,
        base_dag_rmse,
    )
)

final_floor_improvement = (
    1.0
    -
    final_floor_rmse
    /
    max(
        1e-12,
        base_floor_rmse,
    )
)

final_nom_ratio = (
    final_nom_rmse
    /
    max(
        1e-12,
        base_nom_rmse,
    )
)

final_dag_phys = (
    denormalize_action(
        dag_v5[
            dag_val
        ],
        physical_low,
        physical_high,
    )[
        :,
        0
    ]
)

final_dag_phys_rmse = rmse(
    final_dag_phys
    -
    dag_act_phys_raw[
        dag_val,
        0
    ]
)

# Non-collective outputs must remain exact.
noncollective_delta = max(
    float(
        np.max(
            np.abs(
                nom_v5[
                    :,
                    1:
                ]
                -
                nom_v4[
                    :,
                    1:
                ]
            )
        )
    ),
    float(
        np.max(
            np.abs(
                dag_v5[
                    :,
                    1:
                ]
                -
                dag_v4[
                    :,
                    1:
                ]
            )
        )
    ),
)

# V5 phase gate is exactly zero outside near-ground by construction.
# Audit against the COMPLETE nominal Stage-4 dataset.
all_obs = (
    (
        nom_raw_all
        -
        obs_mean
    )
    /
    obs_std
).astype(
    np.float32
)

all_latent, all_base = (
    latent_and_base_action(
        model.policy,
        all_obs,
        device,
    )
)

all_phi = phi(
    all_latent
)

all_v4, _ = (
    apply_v4(
        all_base,
        all_phi,
        nom_raw_all,
        v4_adapter,
    )
)

all_v5, all_corr = (
    apply_v5_on_top_of_v4(
        all_v4,
        all_phi,
        nom_raw_all,
        beta,
    )
)

outside_near = (
    nom_raw_all[
        :,
        NEAR_GROUND_PHASE_INDEX
    ]
    <=
    0.5
)

outside_phase_max_correction = float(
    np.max(
        np.abs(
            all_corr[
                outside_near
            ]
        )
    )
    if np.any(
        outside_near
    )
    else
    0.0
)

all_noncollective_delta = float(
    np.max(
        np.abs(
            all_v5[
                :,
                1:
            ]
            -
            all_v4[
                :,
                1:
            ]
        )
    )
)

max_correction_norm = float(
    np.max(
        np.abs(
            dag_corr
        )
    )
)

max_correction_phys = float(
    0.5
    *
    (
        physical_high[0]
        -
        physical_low[0]
    )
    *
    max_correction_norm
)

min_correction_phys = float(
    0.5
    *
    (
        physical_high[0]
        -
        physical_low[0]
    )
    *
    np.min(
        dag_corr
    )
)

max_positive_correction_phys = float(
    0.5
    *
    (
        physical_high[0]
        -
        physical_low[0]
    )
    *
    np.max(
        dag_corr
    )
)

ready = bool(
    final_nom_rmse
    <=
    nominal_allowed
    and
    final_nom_ratio
    <=
    MAX_NOMINAL_NEARGROUND_DEGRADATION
    +
    1e-12
    and
    final_dagger_improvement
    >=
    MIN_DAGGER_IMPROVEMENT
    and
    final_floor_improvement
    >=
    MIN_FLOOR_DAGGER_IMPROVEMENT
    and
    outside_phase_max_correction
    <=
    1e-12
    and
    noncollective_delta
    <=
    1e-12
    and
    all_noncollective_delta
    <=
    1e-12
)

print(
    "selected D weight:",
    best[
        "dagger_global_weight"
    ],
)

print(
    "selected ridge   :",
    f"{best['ridge_factor']:.1e}",
)

print(
    f"nominal near-ground RMSE : "
    f"{base_nom_rmse:.6f} "
    f"-> {final_nom_rmse:.6f} "
    f"({final_nom_ratio:.3f}x)"
)

print(
    f"DAgger RMSE              : "
    f"{base_dag_rmse:.6f} "
    f"-> {final_dag_rmse:.6f} "
    f"(improve={100*final_dagger_improvement:.2f}%)"
)

print(
    f"DAgger physical RMSE     : "
    f"{base_dag_phys_rmse:.8f} "
    f"-> {final_dag_phys_rmse:.8f}"
)

print(
    f"teacher-floor RMSE       : "
    f"{base_floor_rmse:.6f} "
    f"-> {final_floor_rmse:.6f} "
    f"(improve={100*final_floor_improvement:.2f}%)"
)

print(
    f"outside-near-ground correction : "
    f"{outside_phase_max_correction:.12g}"
)

print(
    f"non-collective delta            : "
    f"{noncollective_delta:.12g}"
)

print(
    f"all-phase non-collective delta  : "
    f"{all_noncollective_delta:.12g}"
)

print(
    f"max |V5 correction| norm        : "
    f"{max_correction_norm:.8f}"
)

print(
    f"max |V5 correction| physical    : "
    f"{max_correction_phys:.8f}"
)

print(
    f"most negative correction phys   : "
    f"{min_correction_phys:+.8f}"
)

print(
    f"max positive correction phys    : "
    f"{max_positive_correction_phys:+.8f}"
)


# =====================================================================
# E — SAVE COMBINED V4 + V5 POLICY PACKAGE
# =====================================================================

rule(
    "E — SAVE COMBINED V4 + V5 POLICY PACKAGE"
)

out_base = (
    OUT_MODEL_DIR
    /
    "AH1S_STAGE4_LOCALIZED_V5_BASE.zip"
)

shutil.copy2(
    BASE_MODEL,
    out_base,
)

shutil.copy2(
    OBS_NORM_PATH,
    OUT_MODEL_DIR
    /
    "stage4_obs_normalization.npz",
)

shutil.copy2(
    ACTION_MAP_PATH,
    OUT_MODEL_DIR
    /
    "stage4_action_mapping.npz",
)

# Preserve V4 adapter byte-for-byte.
out_v4_adapter = (
    OUT_MODEL_DIR
    /
    "stage4_localized_collective_adapter_v4.npz"
)

shutil.copy2(
    V4_ADAPTER_PATH,
    out_v4_adapter,
)

v4_sha_before = (
    file_sha256(
        V4_ADAPTER_PATH
    )
)

v4_sha_after = (
    file_sha256(
        out_v4_adapter
    )
)

if (
    v4_sha_before
    !=
    v4_sha_after
):
    raise RuntimeError(
        "Copied V4 adapter is not byte-identical."
    )

# New V5 adapter.
v5_adapter_path = (
    OUT_MODEL_DIR
    /
    "stage4_nearground_collective_adapter_v5.npz"
)

np.savez(
    v5_adapter_path,
    beta=beta.astype(
        np.float64
    ),
    phase_nearground_index=np.asarray(
        [
            NEAR_GROUND_PHASE_INDEX
        ],
        dtype=np.int64,
    ),
    altitude_index=np.asarray(
        [
            ALTITUDE_INDEX
        ],
        dtype=np.int64,
    ),
    activation=np.asarray(
        [
            "phase_one_hot"
        ],
        dtype="<U32",
    ),
)

interface = {
    "architecture": (
        "locked V2 PPO + locked V4 descent collective adapter "
        "+ V5 near-ground collective adapter"
    ),
    "training_type": (
        "corrective policy distillation; NOT reinforcement learning"
    ),
    "base_model_parameters_changed": False,
    "observation_dim": 26,
    "action_dim": 4,
    "normalization": (
        "unchanged from locked V4/V2"
    ),
    "action_mapping": (
        "unchanged from locked V4/V2"
    ),
    "v4_adapter": {
        "status": "LOCKED AND BYTE-IDENTICAL",
        "file": (
            "stage4_localized_collective_adapter_v4.npz"
        ),
        "phase": (
            "descent_300_to_30"
        ),
        "sha256": (
            v4_sha_after
        ),
    },
    "v5_adapter": {
        "input": (
            "frozen V2 actor latent 128-D + bias"
        ),
        "output": (
            "additive normalized collective residual only"
        ),
        "phase": (
            "near_ground_30_to_native_eq"
        ),
        "phase_obs_index": (
            NEAR_GROUND_PHASE_INDEX
        ),
        "activation": (
            "phase one-hot only"
        ),
    },
    "runtime": (
        "base_z = locked PPO actor; "
        "z0=clip(base_z0 + v4_descent_residual + "
        "v5_nearground_residual,-1,+1); "
        "z1..z3=base_z1..z3; "
        "V4 and V5 gates are phase-mutually-exclusive"
    ),
    "teacher_runtime": False,
    "stage4_classical_controller_runtime": False,
    "reward_based_ppo_fine_tuning_used": False,
}

with open(
    OUT_MODEL_DIR
    /
    "stage4_policy_interface.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        interface,
        f,
        indent=2,
    )

summary = {
    "reinforcement_learning_used": False,
    "base_model_parameters_changed": False,
    "locked_v4_adapter_changed": False,
    "targeted_phase": (
        "near_ground_30_to_native_eq"
    ),
    "baseline_v4": {
        "nominal_nearground_rmse": float(
            base_nom_rmse
        ),
        "nominal_nearground_physical_rmse": float(
            base_nom_phys_rmse
        ),
        "dagger_rmse": float(
            base_dag_rmse
        ),
        "dagger_physical_rmse": float(
            base_dag_phys_rmse
        ),
        "teacher_floor_rmse": float(
            base_floor_rmse
        ),
    },
    "gates": {
        "nominal_abs_rmse_limit": float(
            MAX_NOMINAL_NEARGROUND_ABS_RMSE
        ),
        "nominal_degradation_limit": float(
            MAX_NOMINAL_NEARGROUND_DEGRADATION
        ),
        "effective_nominal_rmse_limit": float(
            nominal_allowed
        ),
        "min_dagger_improvement_fraction": float(
            MIN_DAGGER_IMPROVEMENT
        ),
        "min_floor_improvement_fraction": float(
            MIN_FLOOR_DAGGER_IMPROVEMENT
        ),
    },
    "selected": {
        k: value
        for k, value
        in best.items()
        if k != "beta"
    },
    "v5": {
        "nominal_nearground_rmse": float(
            final_nom_rmse
        ),
        "nominal_ratio_vs_v4": float(
            final_nom_ratio
        ),
        "dagger_rmse": float(
            final_dag_rmse
        ),
        "dagger_improvement_fraction": float(
            final_dagger_improvement
        ),
        "dagger_physical_rmse": float(
            final_dag_phys_rmse
        ),
        "teacher_floor_rmse": float(
            final_floor_rmse
        ),
        "teacher_floor_improvement_fraction": float(
            final_floor_improvement
        ),
        "outside_nearground_max_correction": float(
            outside_phase_max_correction
        ),
        "non_collective_max_delta": float(
            max(
                noncollective_delta,
                all_noncollective_delta,
            )
        ),
        "max_adapter_normalized_correction": float(
            max_correction_norm
        ),
        "max_adapter_physical_equivalent": float(
            max_correction_phys
        ),
        "most_negative_adapter_physical_equivalent": float(
            min_correction_phys
        ),
        "max_positive_adapter_physical_equivalent": float(
            max_positive_correction_phys
        ),
    },
    "v4_adapter_sha256": (
        v4_sha_after
    ),
    "ready_for_fresh_teacher_off_validation": bool(
        ready
    ),
}

with open(
    OUT_RESULT_DIR
    /
    "final_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print()
print(
    "V5 NEAR-GROUND LOCALIZED DISTILLATION COMPLETE:",
    True,
)

print(
    "REWARD-BASED PPO TRAINING USED:",
    False,
)

print(
    "BASE PPO PARAMETERS CHANGED:",
    False,
)

print(
    "LOCKED V4 DESCENT ADAPTER CHANGED:",
    False,
)

print(
    "OUTSIDE NEAR-GROUND V5 CORRECTION ZERO:",
    outside_phase_max_correction
    <=
    1e-12,
)

print(
    "NON-COLLECTIVE OUTPUTS PRESERVED:",
    max(
        noncollective_delta,
        all_noncollective_delta,
    )
    <=
    1e-12,
)

print(
    "READY FOR FRESH V5 TEACHER-OFF VALIDATION:",
    ready,
)

print()
print(
    "Saved:"
)

for path in [
    out_base,
    out_v4_adapter,
    v5_adapter_path,
    OUT_MODEL_DIR
    /
    "stage4_obs_normalization.npz",
    OUT_MODEL_DIR
    /
    "stage4_action_mapping.npz",
    OUT_MODEL_DIR
    /
    "stage4_policy_interface.json",
    OUT_RESULT_DIR
    /
    "candidate_grid.csv",
    OUT_RESULT_DIR
    /
    "final_summary.json",
]:
    print(
        " ",
        path,
    )

print()
print(
    "Next: fresh teacher-OFF JSBSim validation with BOTH "
    "locked V4 descent adapter and V5 near-ground adapter."
)

print(
    "Do NOT call this RL."
)
