import os
import copy
import numpy as np
import torch

from stable_baselines3 import PPO
from helicopter_env_stage1_distill import HelicopterEnvStage1Distill


# ============================================================
# PATHS
# ============================================================

SOURCE_MODEL = (
    "models_stage1_final_distilled/"
    "AH1S_STAGE1_FINAL_DISTILLED.zip"
)

OUTPUT_DIR = "models_stage1_early_distilled"
OUTPUT_MODEL = (
    OUTPUT_DIR
    + "/AH1S_STAGE1_EARLY_DISTILLED"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# FINAL CHOICE:
# USE THE CLEAN V3 EARLY CYCLIC TEACHER
#
# We intentionally DO NOT use V5 roll guard.
# V5 made max drift worse.
#
# We intentionally DO NOT distill V4B collective soft-start.
# It only produced a very small improvement and we want to
# protect the already-good altitude/hover head.
#
# Therefore:
#
# action[0] collective -> untouched
# action[1] elevator   -> distilled
# action[2] aileron    -> distilled
# action[3] rudder     -> untouched
# ============================================================

B_INV = np.array(
    [
        [-0.20338265, -0.00859620],
        [ 0.01952073, -0.14631468],
    ],
    dtype=np.float64,
)

CYCLIC_AUTHORITY = 0.026

XY_KP = 0.000
XY_KD = 0.50
XY_ALPHA = 0.90

FF_ELE = -0.026
FF_AIL = -0.026

FF_HOLD = 0.75
FF_FADE = 1.50

TEACHER_FULL_ALT = 100.0
TEACHER_OFF_ALT = 140.0

TOTAL_TIME = 120.0
HOVER_START_TIME = 60.0


# ============================================================
# DATA DIVERSITY
#
# Small initial XY offsets teach the frozen policy head a local
# neighborhood rather than one single trajectory.
# ============================================================

OFFSETS = [
    ( 0.0,  0.0),
    (+0.5,  0.0),
    (-0.5,  0.0),
    ( 0.0, +0.5),
    ( 0.0, -0.5),
    (+1.0, +1.0),
    (-1.0, -1.0),
    (+1.0, -1.0),
    (-1.0, +1.0),
]


# ============================================================
# RIDGE SEARCH
# ============================================================

RIDGE_VALUES = [
    0.001,
    0.003,
    0.010,
    0.030,
    0.100,
    0.300,
    1.000,
    3.000,
    10.000,
    30.000,
    100.000,
    300.000,
    1000.000,
]


# ============================================================
# HELPERS
# ============================================================

def teacher_gate(altitude):
    if altitude <= TEACHER_FULL_ALT:
        return 1.0

    if altitude >= TEACHER_OFF_ALT:
        return 0.0

    return float(
        (TEACHER_OFF_ALT - altitude)
        /
        (TEACHER_OFF_ALT - TEACHER_FULL_ALT)
    )


def ff_scale(t):
    if t <= FF_HOLD:
        return 1.0

    end = FF_HOLD + FF_FADE

    if t >= end:
        return 0.0

    return float(
        (end - t)
        /
        FF_FADE
    )


def build_teacher_action(
    original_action,
    altitude,
    north,
    east,
    vn,
    ve,
    t,
    teacher_prev,
):
    """
    Reproduces the selected V3 direct cyclic teacher.

    Returns:
        teacher_action
        new_teacher_prev
        gate
    """

    original_action = np.asarray(
        original_action,
        dtype=np.float32,
    ).reshape(-1)

    teacher_action = (
        original_action.copy()
    )

    gate = teacher_gate(
        altitude
    )

    if gate <= 0.0:
        return (
            teacher_action,
            teacher_prev,
            0.0,
        )

    desired_accel = np.array(
        [
            -XY_KP * north
            -
            XY_KD * vn,

            -XY_KP * east
            -
            XY_KD * ve,
        ],
        dtype=np.float64,
    )

    desired_accel = np.clip(
        desired_accel,
        -0.22,
        +0.22,
    )

    feedback_delta = (
        B_INV
        @
        desired_accel
    )

    ff_delta = np.array(
        [
            FF_ELE * ff_scale(t),
            FF_AIL * ff_scale(t),
        ],
        dtype=np.float64,
    )

    requested_delta = (
        feedback_delta
        +
        ff_delta
    )

    requested_delta = np.clip(
        requested_delta,
        -CYCLIC_AUTHORITY,
        +CYCLIC_AUTHORITY,
    )

    raw_teacher_norm = (
        requested_delta
        /
        CYCLIC_AUTHORITY
    )

    raw_teacher_norm = np.clip(
        raw_teacher_norm,
        -1.0,
        +1.0,
    )

    cyclic_teacher = (
        (1.0 - XY_ALPHA)
        *
        teacher_prev
        +
        XY_ALPHA
        *
        raw_teacher_norm
    )

    cyclic_teacher = np.clip(
        cyclic_teacher,
        -1.0,
        +1.0,
    )

    new_teacher_prev = (
        cyclic_teacher.copy()
    )

    # Direct teacher below 100 ft.
    # Smooth teacher -> PPO blend from 100 to 140 ft.
    teacher_action[1] = float(
        gate
        *
        cyclic_teacher[0]
        +
        (1.0 - gate)
        *
        original_action[1]
    )

    teacher_action[2] = float(
        gate
        *
        cyclic_teacher[1]
        +
        (1.0 - gate)
        *
        original_action[2]
    )

    teacher_action = np.clip(
        teacher_action,
        -1.0,
        +1.0,
    ).astype(np.float32)

    return (
        teacher_action,
        new_teacher_prev,
        gate,
    )


def collect_metrics(
    model,
    use_teacher=False,
    detailed=False,
):
    """
    120 s validation.

    If use_teacher=True:
      V3 teacher owns cyclic below 100 ft and fades out by 140 ft.

    If use_teacher=False:
      single PPO flies completely alone.
    """

    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env.reset()

    dt = float(env.dt)
    max_steps = int(
        TOTAL_TIME
        /
        dt
    )

    teacher_prev = np.zeros(
        2,
        dtype=np.float64,
    )

    early_max = 0.0
    path100 = None
    drift100 = None
    hs100 = None
    time100 = None

    max_n100 = 0.0
    max_e100 = 0.0

    hover_altitudes = []
    hover_vs = []

    physical_failure = False

    next_print = 0.0

    last_info = info

    for step in range(max_steps):
        t_before = step * dt

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if use_teacher:
            altitude = float(
                info.get(
                    "altitude",
                    0.0,
                )
            )

            north = float(
                info.get(
                    "north",
                    0.0,
                )
            )

            east = float(
                info.get(
                    "east",
                    0.0,
                )
            )

            vn = float(
                info.get(
                    "vn",
                    0.0,
                )
            )

            ve = float(
                info.get(
                    "ve",
                    0.0,
                )
            )

            action, teacher_prev, _ = (
                build_teacher_action(
                    original_action=action,
                    altitude=altitude,
                    north=north,
                    east=east,
                    vn=vn,
                    ve=ve,
                    t=t_before,
                    teacher_prev=teacher_prev,
                )
            )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        last_info = info

        t = (step + 1) * dt

        altitude_now = float(
            info["altitude"]
        )

        vs_now = float(
            info["vertical_speed"]
        )

        north_now = float(
            info["north"]
        )

        east_now = float(
            info["east"]
        )

        vn_now = float(
            info["vn"]
        )

        ve_now = float(
            info["ve"]
        )

        drift_now = float(
            info["drift"]
        )

        path_now = float(
            info["path"]
        )

        hspeed_now = float(
            np.hypot(
                vn_now,
                ve_now,
            )
        )

        # ----------------------------------------------------
        # FIRST 100 FT
        # ----------------------------------------------------

        if altitude_now <= 100.0:
            early_max = max(
                early_max,
                drift_now,
            )

            max_n100 = max(
                max_n100,
                abs(north_now),
            )

            max_e100 = max(
                max_e100,
                abs(east_now),
            )

        if (
            path100 is None
            and
            altitude_now >= 100.0
        ):
            path100 = path_now
            drift100 = drift_now
            hs100 = hspeed_now
            time100 = t

        # ----------------------------------------------------
        # HOVER
        # ----------------------------------------------------

        if t >= HOVER_START_TIME:
            hover_altitudes.append(
                altitude_now
            )

            hover_vs.append(
                vs_now
            )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        if (
            detailed
            and
            t >= next_print
        ):
            label = (
                "TEACHER"
                if use_teacher
                else "PPO"
            )

            print(
                f"{label:7s} | "
                f"t={t:6.1f}s | "
                f"ALT={altitude_now:7.2f} | "
                f"VS={vs_now:+6.3f} | "
                f"N={north_now:+6.2f} | "
                f"E={east_now:+6.2f} | "
                f"DRIFT={drift_now:5.2f} | "
                f"MAX={info['max_drift']:5.2f} | "
                f"PATH={path_now:6.2f}"
            )

            next_print += 5.0

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------

        if (
            terminated
            and
            not bool(
                info.get(
                    "success",
                    False,
                )
            )
        ):
            physical_failure = True
            break

        if altitude_now > 340.0:
            physical_failure = True
            break

        if (
            t > 70.0
            and
            altitude_now < 270.0
        ):
            physical_failure = True
            break

        if (
            float(
                info.get(
                    "max_drift",
                    999.0,
                )
            )
            >
            12.0
        ):
            physical_failure = True
            break

    env.close()

    if path100 is None:
        path100 = float(
            last_info.get(
                "path",
                999.0,
            )
        )

        drift100 = float(
            last_info.get(
                "drift",
                999.0,
            )
        )

        hs100 = float(
            np.hypot(
                float(
                    last_info.get(
                        "vn",
                        999.0,
                    )
                ),
                float(
                    last_info.get(
                        "ve",
                        999.0,
                    )
                ),
            )
        )

        time100 = TOTAL_TIME
        physical_failure = True

    hover_altitudes = np.asarray(
        hover_altitudes,
        dtype=np.float64,
    )

    hover_vs = np.asarray(
        hover_vs,
        dtype=np.float64,
    )

    if len(hover_altitudes) == 0:
        mean_alt = np.nan
        std_alt = np.nan
        min_alt = np.nan
        max_alt = np.nan
        max_abs_vs = np.inf
    else:
        mean_alt = float(
            np.mean(
                hover_altitudes
            )
        )

        std_alt = float(
            np.std(
                hover_altitudes
            )
        )

        min_alt = float(
            np.min(
                hover_altitudes
            )
        )

        max_alt = float(
            np.max(
                hover_altitudes
            )
        )

        max_abs_vs = float(
            np.max(
                np.abs(
                    hover_vs
                )
            )
        )

    s_index = max(
        0.0,
        path100 - drift100,
    )

    total_max = float(
        last_info.get(
            "max_drift",
            999.0,
        )
    )

    final_drift = float(
        last_info.get(
            "drift",
            999.0,
        )
    )

    final_path = float(
        last_info.get(
            "path",
            999.0,
        )
    )

    final_alt = float(
        last_info.get(
            "altitude",
            999.0,
        )
    )

    final_vs = float(
        last_info.get(
            "vertical_speed",
            999.0,
        )
    )

    # --------------------------------------------------------
    # LOCK CRITERIA
    #
    # Teacher itself is around MAX100=2.76 ft.
    # Distillation does not need to beat an impossible 2 ft
    # target. It must reproduce a clearly improved trajectory
    # while preserving the 300 ft hover.
    # --------------------------------------------------------

    success = bool(
        not physical_failure
        and early_max <= 3.20
        and path100 <= 6.50
        and drift100 <= 2.00
        and hs100 <= 0.80
        and time100 <= 22.0
        and total_max <= 6.0
        and final_drift <= 2.5
        and 295.0 <= final_alt <= 305.0
        and abs(final_vs) <= 0.75
        and min_alt >= 295.0
        and max_alt <= 305.0
        and max_abs_vs <= 0.75
    )

    score = (
        1000.0
        *
        early_max

        +
        250.0
        *
        path100

        +
        300.0
        *
        s_index

        +
        100.0
        *
        drift100

        +
        50.0
        *
        max_n100

        +
        50.0
        *
        max_e100

        +
        150.0
        *
        abs(
            mean_alt - 300.0
        )

        +
        100.0
        *
        std_alt

        +
        50.0
        *
        max_abs_vs
    )

    if not success:
        score += 1e7

    return {
        "success":
            success,

        "score":
            float(score),

        "early_max":
            float(early_max),

        "path100":
            float(path100),

        "drift100":
            float(drift100),

        "s_index":
            float(s_index),

        "hs100":
            float(hs100),

        "time100":
            float(time100),

        "max_n100":
            float(max_n100),

        "max_e100":
            float(max_e100),

        "mean_alt":
            float(mean_alt),

        "std_alt":
            float(std_alt),

        "min_alt":
            float(min_alt),

        "max_alt":
            float(max_alt),

        "max_abs_vs":
            float(max_abs_vs),

        "total_max":
            float(total_max),

        "final_drift":
            float(final_drift),

        "final_path":
            float(final_path),

        "final_alt":
            float(final_alt),

        "final_vs":
            float(final_vs),
    }


def print_result(
    label,
    r,
):
    print(
        f"{label:18s} | "
        f"OK={str(r['success']):5s} | "
        f"MAX100={r['early_max']:5.3f} | "
        f"PATH100={r['path100']:5.3f} | "
        f"D100={r['drift100']:5.3f} | "
        f"S={r['s_index']:5.3f} | "
        f"HS100={r['hs100']:5.3f} | "
        f"FINALALT={r['final_alt']:7.3f} | "
        f"FINALVS={r['final_vs']:+6.3f} | "
        f"FINALDRIFT={r['final_drift']:5.3f}"
    )


# ============================================================
# LOAD SOURCE
# ============================================================

print("=" * 140)
print(
    "STAGE 1 FINAL EARLY-CYCLIC DISTILLATION"
)
print("=" * 140)

print("\nSource model:")
print(SOURCE_MODEL)

base_model = PPO.load(
    SOURCE_MODEL
)

print(
    "\nSource PPO loaded."
)

print(
    "Only action-head rows 1 and 2 may change."
)
print(
    "Rows 0 and 3 will be verified bit-for-bit unchanged."
)


# ============================================================
# BASELINE + TEACHER REFERENCES
# ============================================================

print(
    "\n"
    +
    "=" * 140
)
print(
    "REFERENCE FLIGHTS"
)
print(
    "=" * 140
)

baseline = collect_metrics(
    base_model,
    use_teacher=False,
    detailed=False,
)

teacher_reference = collect_metrics(
    base_model,
    use_teacher=True,
    detailed=False,
)

print_result(
    "BASE PPO",
    baseline,
)

print_result(
    "V3 TEACHER",
    teacher_reference,
)

print(
    "\nExpected direction:"
)
print(
    "  Base PPO early max drift should be around the old ~4.65 ft."
)
print(
    "  V3 teacher early max drift should be around ~2.76 ft."
)


# ============================================================
# COLLECT TEACHER DATA
# ============================================================

print(
    "\n"
    +
    "=" * 140
)
print(
    "COLLECTING EARLY-CYCLIC TEACHER DATA"
)
print(
    "=" * 140
)

observations = []
target_elevator = []
target_aileron = []
sample_weights = []

for episode_index, (
    offset_n,
    offset_e,
) in enumerate(
    OFFSETS,
    start=1,
):
    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env.reset()

    # Small local XY diversity.
    if hasattr(
        env,
        "north",
    ):
        env.north = float(
            offset_n
        )

    if hasattr(
        env,
        "east",
    ):
        env.east = float(
            offset_e
        )

    if (
        hasattr(
            env,
            "_get_obs",
        )
        and
        (
            abs(offset_n) > 0.0
            or
            abs(offset_e) > 0.0
        )
    ):
        obs = env._get_obs()

        # Keep info synchronized with modified local position.
        info = dict(info)
        info["north"] = float(
            offset_n
        )
        info["east"] = float(
            offset_e
        )

    dt = float(
        env.dt
    )

    max_steps = int(
        TOTAL_TIME
        /
        dt
    )

    teacher_prev = np.zeros(
        2,
        dtype=np.float64,
    )

    episode_samples = 0
    early_samples = 0
    transition_samples = 0
    protection_samples = 0

    for step in range(
        max_steps
    ):
        t_before = step * dt

        original_action, _ = (
            base_model.predict(
                obs,
                deterministic=True,
            )
        )

        original_action = np.asarray(
            original_action,
            dtype=np.float32,
        ).reshape(-1)

        altitude = float(
            info.get(
                "altitude",
                0.0,
            )
        )

        north = float(
            info.get(
                "north",
                0.0,
            )
        )

        east = float(
            info.get(
                "east",
                0.0,
            )
        )

        vn = float(
            info.get(
                "vn",
                0.0,
            )
        )

        ve = float(
            info.get(
                "ve",
                0.0,
            )
        )

        (
            teacher_action,
            teacher_prev,
            gate,
        ) = build_teacher_action(
            original_action=
                original_action,

            altitude=
                altitude,

            north=
                north,

            east=
                east,

            vn=
                vn,

            ve=
                ve,

            t=
                t_before,

            teacher_prev=
                teacher_prev,
        )

        # ----------------------------------------------------
        # DATA WEIGHTING
        #
        # <=100 ft     : main target
        # 100-140 ft   : handoff target
        # >140 ft      : preserve original PPO
        #
        # We keep fewer late samples because otherwise the
        # 120-second dataset would be dominated by hover.
        # ----------------------------------------------------

        keep_sample = True

        if altitude <= 100.0:
            weight = 10.0
            early_samples += 1

        elif altitude < 140.0:
            weight = 6.0
            transition_samples += 1

        else:
            # Only keep every 5th late sample.
            keep_sample = (
                step % 5 == 0
            )

            weight = 1.0

            if keep_sample:
                protection_samples += 1

        if keep_sample:
            observations.append(
                np.asarray(
                    obs,
                    dtype=np.float32,
                ).copy()
            )

            target_elevator.append(
                float(
                    teacher_action[1]
                )
            )

            target_aileron.append(
                float(
                    teacher_action[2]
                )
            )

            sample_weights.append(
                float(weight)
            )

            episode_samples += 1

        # Teacher flies the data-collection trajectory.
        obs, reward, terminated, truncated, info = env.step(
            teacher_action
        )

        if (
            terminated
            and
            not bool(
                info.get(
                    "success",
                    False,
                )
            )
        ):
            print(
                f"WARNING: episode {episode_index} "
                "ended with physical failure."
            )
            break

    env.close()

    print(
        f"Episode {episode_index:2d} | "
        f"offset=({offset_n:+.1f},{offset_e:+.1f}) | "
        f"stored={episode_samples:5d} | "
        f"early={early_samples:4d} | "
        f"transition={transition_samples:4d} | "
        f"late-protect={protection_samples:4d}"
    )


observations = np.asarray(
    observations,
    dtype=np.float32,
)

target_elevator = np.asarray(
    target_elevator,
    dtype=np.float64,
)

target_aileron = np.asarray(
    target_aileron,
    dtype=np.float64,
)

sample_weights = np.asarray(
    sample_weights,
    dtype=np.float64,
)

print(
    "\nDataset:"
)
print(
    "Observations :",
    observations.shape,
)
print(
    "Elevator y   :",
    target_elevator.shape,
)
print(
    "Aileron y    :",
    target_aileron.shape,
)
print(
    "Weights      :",
    sample_weights.shape,
)


# ============================================================
# EXTRACT FROZEN ACTOR LATENT FEATURES
# ============================================================

print(
    "\n"
    +
    "=" * 140
)
print(
    "EXTRACTING FROZEN PPO ACTOR FEATURES"
)
print(
    "=" * 140
)

device = (
    base_model.device
)

obs_tensor = torch.as_tensor(
    observations,
    dtype=torch.float32,
    device=device,
)

latent_batches = []
BATCH_SIZE = 512

with torch.no_grad():
    for start in range(
        0,
        len(observations),
        BATCH_SIZE,
    ):
        batch_obs = obs_tensor[
            start:
            start + BATCH_SIZE
        ]

        features = (
            base_model.policy.extract_features(
                batch_obs
            )
        )

        if isinstance(
            features,
            tuple,
        ):
            actor_features = (
                features[0]
            )
        else:
            actor_features = (
                features
            )

        latent_pi = (
            base_model.policy.mlp_extractor.forward_actor(
                actor_features
            )
        )

        latent_batches.append(
            latent_pi
            .detach()
            .cpu()
            .numpy()
        )


latent = np.concatenate(
    latent_batches,
    axis=0,
).astype(
    np.float64
)

print(
    "Actor latent:",
    latent.shape,
)


# ============================================================
# DESIGN MATRIX
# ============================================================

ones = np.ones(
    (
        latent.shape[0],
        1,
    ),
    dtype=np.float64,
)

X = np.concatenate(
    [
        latent,
        ones,
    ],
    axis=1,
)

sqrt_w = np.sqrt(
    sample_weights
).reshape(
    -1,
    1,
)

Xw = (
    X
    *
    sqrt_w
)

y_ele_w = (
    target_elevator
    *
    sqrt_w[:, 0]
)

y_ail_w = (
    target_aileron
    *
    sqrt_w[:, 0]
)


# ============================================================
# ORIGINAL HEAD PARAMETERS
# ============================================================

action_weight = (
    base_model
    .policy
    .action_net
    .weight
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float64
    )
)

action_bias = (
    base_model
    .policy
    .action_net
    .bias
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float64
    )
)

original_rows = {}

for row in [1, 2]:
    original_rows[row] = (
        np.concatenate(
            [
                action_weight[row],
                np.array(
                    [
                        action_bias[row]
                    ],
                    dtype=np.float64,
                ),
            ]
        )
    )


print(
    "\nAction head shape:",
    action_weight.shape,
)

print(
    "Elevator parameters:",
    original_rows[1].shape,
)

print(
    "Aileron parameters :",
    original_rows[2].shape,
)


# ============================================================
# WEIGHTED RIDGE FIT
#
# Solve separately for action rows 1 and 2:
#
# min sum_i w_i (X_i beta - y_i)^2
#     + lambda ||beta - beta_original||^2
#
# Rows 0 and 3 are never touched.
# ============================================================

print(
    "\n"
    +
    "=" * 140
)
print(
    "WEIGHTED RIDGE DISTILLATION — ACTION ROWS 1 & 2"
)
print(
    "=" * 140
)

XtX = (
    Xw.T
    @
    Xw
)

Xty_ele = (
    Xw.T
    @
    y_ele_w
)

Xty_ail = (
    Xw.T
    @
    y_ail_w
)

identity = np.eye(
    X.shape[1],
    dtype=np.float64,
)

results = []

# Store source rows 0 and 3 for exact verification.
source_row0_w = (
    base_model
    .policy
    .action_net
    .weight[0]
    .detach()
    .cpu()
    .clone()
)

source_row0_b = (
    base_model
    .policy
    .action_net
    .bias[0]
    .detach()
    .cpu()
    .clone()
)

source_row3_w = (
    base_model
    .policy
    .action_net
    .weight[3]
    .detach()
    .cpu()
    .clone()
)

source_row3_b = (
    base_model
    .policy
    .action_net
    .bias[3]
    .detach()
    .cpu()
    .clone()
)


for ridge in RIDGE_VALUES:
    A = (
        XtX
        +
        ridge
        *
        identity
    )

    b_ele = (
        Xty_ele
        +
        ridge
        *
        original_rows[1]
    )

    b_ail = (
        Xty_ail
        +
        ridge
        *
        original_rows[2]
    )

    fitted_ele = np.linalg.solve(
        A,
        b_ele,
    )

    fitted_ail = np.linalg.solve(
        A,
        b_ail,
    )

    candidate = PPO.load(
        SOURCE_MODEL
    )

    with torch.no_grad():
        candidate.policy.action_net.weight[1].copy_(
            torch.as_tensor(
                fitted_ele[:-1],
                dtype=(
                    candidate
                    .policy
                    .action_net
                    .weight
                    .dtype
                ),
                device=(
                    candidate
                    .policy
                    .action_net
                    .weight
                    .device
                ),
            )
        )

        candidate.policy.action_net.bias[1].copy_(
            torch.tensor(
                float(
                    fitted_ele[-1]
                ),
                dtype=(
                    candidate
                    .policy
                    .action_net
                    .bias
                    .dtype
                ),
                device=(
                    candidate
                    .policy
                    .action_net
                    .bias
                    .device
                ),
            )
        )

        candidate.policy.action_net.weight[2].copy_(
            torch.as_tensor(
                fitted_ail[:-1],
                dtype=(
                    candidate
                    .policy
                    .action_net
                    .weight
                    .dtype
                ),
                device=(
                    candidate
                    .policy
                    .action_net
                    .weight
                    .device
                ),
            )
        )

        candidate.policy.action_net.bias[2].copy_(
            torch.tensor(
                float(
                    fitted_ail[-1]
                ),
                dtype=(
                    candidate
                    .policy
                    .action_net
                    .bias
                    .dtype
                ),
                device=(
                    candidate
                    .policy
                    .action_net
                    .bias
                    .device
                ),
            )
        )

    # --------------------------------------------------------
    # VERIFY COLLECTIVE + RUDDER HEADS ARE UNCHANGED
    # --------------------------------------------------------

    row0_same = bool(
        torch.equal(
            candidate
            .policy
            .action_net
            .weight[0]
            .detach()
            .cpu(),
            source_row0_w,
        )
        and
        torch.equal(
            candidate
            .policy
            .action_net
            .bias[0]
            .detach()
            .cpu(),
            source_row0_b,
        )
    )

    row3_same = bool(
        torch.equal(
            candidate
            .policy
            .action_net
            .weight[3]
            .detach()
            .cpu(),
            source_row3_w,
        )
        and
        torch.equal(
            candidate
            .policy
            .action_net
            .bias[3]
            .detach()
            .cpu(),
            source_row3_b,
        )
    )

    if not row0_same:
        raise RuntimeError(
            "Collective head changed unexpectedly."
        )

    if not row3_same:
        raise RuntimeError(
            "Rudder head changed unexpectedly."
        )

    validation = collect_metrics(
        candidate,
        use_teacher=False,
        detailed=False,
    )

    validation["ridge"] = (
        ridge
    )

    validation["row0_same"] = (
        row0_same
    )

    validation["row3_same"] = (
        row3_same
    )

    results.append(
        (
            validation,
            candidate,
        )
    )

    status = (
        "PASS"
        if validation["success"]
        else "----"
    )

    print(
        f"{status:4s} | "
        f"ridge={ridge:8.3f} | "
        f"MAX100={validation['early_max']:5.3f} | "
        f"PATH100={validation['path100']:5.3f} | "
        f"D100={validation['drift100']:5.3f} | "
        f"S={validation['s_index']:5.3f} | "
        f"HS100={validation['hs100']:5.3f} | "
        f"ALT={validation['final_alt']:7.3f} | "
        f"VS={validation['final_vs']:+6.3f} | "
        f"FDRIFT={validation['final_drift']:5.3f}"
    )


# ============================================================
# SELECT BEST
# ============================================================

passing = [
    pair
    for pair in results
    if pair[0]["success"]
]

if len(passing) > 0:
    passing.sort(
        key=lambda pair:
            pair[0]["score"]
    )

    best_result, best_model = (
        passing[0]
    )

    best_is_passing = True

else:
    results.sort(
        key=lambda pair:
            pair[0]["score"]
    )

    best_result, best_model = (
        results[0]
    )

    best_is_passing = False


print(
    "\n"
    +
    "=" * 140
)
print(
    "BEST DISTILLED CANDIDATE"
)
print(
    "=" * 140
)

print(
    "RIDGE       :",
    best_result["ridge"],
)

print_result(
    "BEST",
    best_result,
)

print(
    "Collective row unchanged:",
    best_result["row0_same"],
)

print(
    "Rudder row unchanged    :",
    best_result["row3_same"],
)


# ============================================================
# FINAL 120 s CONTROLLER-FREE VALIDATION
# ============================================================

print(
    "\n"
    +
    "=" * 140
)
print(
    "FINAL 120 SECOND CONTROLLER-FREE VALIDATION"
)
print(
    "=" * 140
)

print(
    "Teacher              : OFF"
)
print(
    "PD controller        : OFF"
)
print(
    "Classical XY runtime : OFF"
)
print(
    "Roll guard           : OFF"
)
print(
    "Collective softstart : OFF"
)
print(
    "Runtime bias         : OFF"
)
print(
    "Policy               : SINGLE 4-ACTION PPO"
)
print()

final_result = collect_metrics(
    best_model,
    use_teacher=False,
    detailed=True,
)


print(
    "\n"
    +
    "=" * 140
)
print(
    "STAGE 1 FINAL LOCK RESULT"
)
print(
    "=" * 140
)

print(
    f"LOCK SUCCESS                 : "
    f"{final_result['success']}"
)

print(
    f"EARLY MAX DRIFT (<100 ft)   : "
    f"{final_result['early_max']:.3f} ft"
)

print(
    f"XY PATH AT 100 ft           : "
    f"{final_result['path100']:.3f} ft"
)

print(
    f"DRIFT AT 100 ft             : "
    f"{final_result['drift100']:.3f} ft"
)

print(
    f"S-INDEX                      : "
    f"{final_result['s_index']:.3f} ft"
)

print(
    f"HORIZONTAL SPEED @100 ft    : "
    f"{final_result['hs100']:.3f} ft/s"
)

print(
    f"TIME TO 100 ft              : "
    f"{final_result['time100']:.3f} s"
)

print(
    f"MAX |NORTH| <100 ft         : "
    f"{final_result['max_n100']:.3f} ft"
)

print(
    f"MAX |EAST| <100 ft          : "
    f"{final_result['max_e100']:.3f} ft"
)

print()
print(
    f"HOVER MEAN ALT              : "
    f"{final_result['mean_alt']:.3f} ft"
)

print(
    f"HOVER STD ALT               : "
    f"{final_result['std_alt']:.3f} ft"
)

print(
    f"HOVER RANGE                 : "
    f"{final_result['min_alt']:.3f} -> "
    f"{final_result['max_alt']:.3f} ft"
)

print(
    f"HOVER MAX |VS|              : "
    f"{final_result['max_abs_vs']:.3f} ft/s"
)

print()
print(
    f"TOTAL MAX DRIFT             : "
    f"{final_result['total_max']:.3f} ft"
)

print(
    f"FINAL DRIFT                 : "
    f"{final_result['final_drift']:.3f} ft"
)

print(
    f"FINAL XY PATH               : "
    f"{final_result['final_path']:.3f} ft"
)

print(
    f"FINAL ALTITUDE              : "
    f"{final_result['final_alt']:.3f} ft"
)

print(
    f"FINAL VS                    : "
    f"{final_result['final_vs']:.3f} ft/s"
)


# ============================================================
# COMPARE AGAINST ORIGINAL BASE PPO
# ============================================================

improvement = (
    baseline["early_max"]
    -
    final_result["early_max"]
)

path_improvement = (
    baseline["path100"]
    -
    final_result["path100"]
)

print()
print(
    f"EARLY MAX IMPROVEMENT       : "
    f"{improvement:+.3f} ft"
)

print(
    f"PATH@100 IMPROVEMENT        : "
    f"{path_improvement:+.3f} ft"
)


# ============================================================
# SAVE ONLY IF IT PASSES
# ============================================================

if (
    best_is_passing
    and
    final_result["success"]
):
    best_model.save(
        OUTPUT_MODEL
    )

    print(
        "\n"
        +
        "=" * 140
    )

    print(
        "STAGE 1 LOCKED"
    )

    print(
        "=" * 140
    )

    print(
        "Single PPO: YES"
    )

    print(
        "4 actions: YES"
    )

    print(
        "Teacher at runtime: OFF"
    )

    print(
        "PD at runtime: OFF"
    )

    print(
        "Classical XY runtime: OFF"
    )

    print(
        "Collective soft-start runtime: OFF"
    )

    print(
        "Roll guard runtime: OFF"
    )

    print(
        "\nSaved NEW model:"
    )

    print(
        OUTPUT_MODEL
        +
        ".zip"
    )

    print(
        "\nOriginal golden model remains untouched:"
    )

    print(
        SOURCE_MODEL
    )

    print(
        "\nNEXT STEP: STAGE 2 CONTINUOUS HANDOFF VALIDATION."
    )

else:
    print(
        "\n"
        +
        "=" * 140
    )

    print(
        "DISTILLATION DID NOT PASS — SOURCE MODEL PRESERVED"
    )

    print(
        "=" * 140
    )

    print(
        "No new model was accepted."
    )

    print(
        "Do not start another V6 controller search."
    )

    print(
        "If no ridge candidate passes, keep the existing "
        "golden Stage-1 PPO and move to Stage 2."
    )


print(
    "=" * 140
)
