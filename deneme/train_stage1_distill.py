import os

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill
)


SOURCE_MODEL = (
    "models_v2/"
    "AH1S_STAGE1_SUCCESS.zip"
)

OUTPUT_DIR = (
    "models_stage1_distill"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SOURCE MODEL
# ============================================================

print("=" * 110)
print("LOADING ORIGINAL STAGE 1")
print("=" * 110)


source_model = PPO.load(
    SOURCE_MODEL
)


print(
    "✅ Source model loaded"
)


# ============================================================
# TRAIN ENV
# ============================================================

raw_train_env = (
    HelicopterEnvStage1Distill(
        teacher_model_path=
            SOURCE_MODEL,

        training_mode=True
    )
)


train_env = Monitor(
    raw_train_env
)


# ============================================================
# NEW SINGLE 4-ACTION PPO
# ============================================================

model = PPO(
    "MlpPolicy",

    train_env,

    learning_rate=3e-5,

    n_steps=1024,

    batch_size=64,

    n_epochs=10,

    gamma=0.995,

    gae_lambda=0.95,

    clip_range=0.12,

    ent_coef=0.0003,

    vf_coef=0.5,

    max_grad_norm=0.5,

    target_kl=0.015,

    policy_kwargs=dict(
        net_arch=[
            128,
            128
        ],

        log_std_init=-1.7
    ),

    verbose=1,

    tensorboard_log=(
        "logs_stage1_distill/"
        "tensorboard/"
    ),
)


# ============================================================
# SELECTIVE OLD STAGE1 TRANSFER
# ============================================================

print("\n")
print("=" * 110)
print("SELECTIVE WEIGHT TRANSFER")
print("=" * 110)


source_state = (
    source_model
    .policy
    .state_dict()
)


target_state = (
    model
    .policy
    .state_dict()
)


for key in target_state.keys():

    if key not in source_state:

        continue


    src = (
        source_state[key]
    )


    dst = (
        target_state[key]
    )


    # ========================================================
    # ACTION HEAD
    #
    # only collective old output mattered
    # ========================================================

    if key == "action_net.weight":

        new = dst.clone()

        new[0, :] = (
            src[0, :]
        )

        new[1:, :] = 0.0

        target_state[key] = new

        continue


    if key == "action_net.bias":

        new = dst.clone()

        new[0] = src[0]

        new[1:] = 0.0

        target_state[key] = new

        continue


    # ========================================================
    # EXPLORATION
    # ========================================================

    if key == "log_std":

        new = dst.clone()

        new[0] = src[0]

        new[1] = -1.8
        new[2] = -1.8
        new[3] = -2.2

        target_state[key] = new

        continue


    # ========================================================
    # SAME SHAPE
    # ========================================================

    if (
        src.shape
        ==
        dst.shape
    ):

        target_state[key] = (
            src.clone()
        )

        continue


    # ========================================================
    # 12 OBS -> 18 OBS
    # ========================================================

    if (
        src.ndim == 2
        and
        dst.ndim == 2
        and
        src.shape[0]
        ==
        dst.shape[0]
        and
        src.shape[1]
        <
        dst.shape[1]
    ):

        new = dst.clone()

        old_n = (
            src.shape[1]
        )


        new[
            :,
            :old_n
        ] = src


        new[
            :,
            old_n:
        ] = 0.0


        target_state[key] = new


model.policy.load_state_dict(
    target_state
)


print(
    "✅ Selective transfer complete"
)


# ============================================================
# COLLECT TEACHER DATA
# ============================================================

print("\n")
print("=" * 110)
print("COLLECTING TEACHER DATA")
print("=" * 110)


teacher_env = (
    HelicopterEnvStage1Distill(
        teacher_model_path=
            SOURCE_MODEL,

        training_mode=True
    )
)


teacher_env.teacher_blend = 1.0


observations = []
actions = []


# Add a few small virtual XY offsets
# so policy sees recovery states too.
OFFSETS = [
    (0.0, 0.0),

    (2.0, 0.0),
    (-2.0, 0.0),

    (0.0, 2.0),
    (0.0, -2.0),

    (2.0, 2.0),
    (-2.0, -2.0),
]


for episode_index, (
    offset_n,
    offset_e
) in enumerate(
    OFFSETS
):

    obs, info = (
        teacher_env.reset()
    )


    teacher_env.north = (
        offset_n
    )

    teacher_env.east = (
        offset_e
    )


    obs = (
        teacher_env._get_obs()
    )


    episode_samples = 0


    while True:

        teacher_action = (
            teacher_env
            .get_teacher_action()
        )


        observations.append(
            obs.copy()
        )


        actions.append(
            teacher_action.copy()
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = teacher_env.step(
            teacher_action
        )


        episode_samples += 1


        if (
            terminated
            or
            truncated
        ):

            break


    print(
        f"Teacher episode "
        f"{episode_index + 1}: "
        f"{episode_samples} samples"
    )


teacher_env.close()


observations = np.asarray(
    observations,
    dtype=np.float32
)


actions = np.asarray(
    actions,
    dtype=np.float32
)


print(
    "\nDataset observations:",
    observations.shape
)

print(
    "Dataset actions     :",
    actions.shape
)


# ============================================================
# BEHAVIOR CLONING PRETRAIN
# ============================================================

print("\n")
print("=" * 110)
print("BEHAVIOR CLONING PRETRAIN")
print("=" * 110)


device = (
    model.device
)


optimizer = torch.optim.Adam(
    model.policy.parameters(),
    lr=1e-4
)


obs_tensor = torch.as_tensor(
    observations,
    dtype=torch.float32,
    device=device
)


action_tensor = torch.as_tensor(
    actions,
    dtype=torch.float32,
    device=device
)


BC_EPOCHS = 120
BC_BATCH = 128


n_samples = (
    observations.shape[0]
)


for epoch in range(
    BC_EPOCHS
):

    permutation = np.random.permutation(
        n_samples
    )


    epoch_loss = 0.0
    batches = 0


    for start in range(
        0,
        n_samples,
        BC_BATCH
    ):

        idx = (
            permutation[
                start:
                start + BC_BATCH
            ]
        )


        batch_obs = (
            obs_tensor[idx]
        )


        batch_actions = (
            action_tensor[idx]
        )


        # PPO Gaussian distribution
        distribution = (
            model.policy
            .get_distribution(
                batch_obs
            )
        )


        predicted_mean = (
            distribution
            .distribution
            .mean
        )


        loss = torch.mean(
            (
                predicted_mean
                -
                batch_actions
            ) ** 2
        )


        optimizer.zero_grad()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.policy.parameters(),
            1.0
        )


        optimizer.step()


        epoch_loss += float(
            loss.item()
        )

        batches += 1


    if (
        epoch == 0
        or
        (epoch + 1) % 10 == 0
    ):

        print(
            f"BC epoch "
            f"{epoch + 1:3d}/"
            f"{BC_EPOCHS} | "
            f"loss="
            f"{epoch_loss / batches:.8f}"
        )


model.save(
    f"{OUTPUT_DIR}/"
    "AH1S_STAGE1_BC_PRETRAIN"
)


print(
    "\n✅ BC pretrained model saved"
)


# ============================================================
# TEACHER-OFF EVALUATION
# ============================================================

def evaluate(
    model,
    label
):

    env = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False
        )
    )


    obs, info = env.reset()

    next_print = 0.0


    while True:

        action, _ = (
            model.predict(
                obs,
                deterministic=True
            )
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )


        t = (
            env.steps
            *
            env.dt
        )


        if t >= next_print:

            print(
                f"{label} | "
                f"t={t:6.1f}s | "
                f"ALT={info['altitude']:7.2f} | "
                f"VS={info['vertical_speed']:6.2f} | "
                f"DRIFT={info['drift']:6.2f} | "
                f"MAX={info['max_drift']:6.2f} | "
                f"PATH={info['path']:6.2f} | "
                f"N={info['north']:6.2f} | "
                f"E={info['east']:6.2f} | "
                f"COL={info['collective']:.4f} | "
                f"ELE={info['elevator']:.5f} | "
                f"AIL={info['aileron']:.5f} | "
                f"RUD={info['rudder']:.5f}"
            )


            next_print += 10.0


        if (
            terminated
            or
            truncated
        ):

            break


    result = {
        "success":
            info["success"],

        "altitude":
            info["altitude"],

        "altitude_error":
            info[
                "altitude_error"
            ],

        "vs":
            info[
                "vertical_speed"
            ],

        "max":
            info["max_drift"],

        "final":
            info["drift"],

        "path":
            info["path"],

        "north":
            info["north"],

        "east":
            info["east"],
    }


    env.close()

    return result


# ============================================================
# BC TEST
# ============================================================

print("\n")
print("=" * 110)
print("BC TEACHER-OFF TEST")
print("=" * 110)


bc_result = evaluate(
    model,
    "BC"
)


print(
    "\nBC RESULT:",
    bc_result
)


# ============================================================
# PPO DISTILLATION CURRICULUM
# ============================================================

print("\n")
print("=" * 110)
print("PPO DISTILLATION")
print("=" * 110)


# teacher physical blend, timesteps
SCHEDULE = [

    (1.00, 10_240),

    (0.75, 10_240),

    (0.50, 15_360),

    (0.25, 15_360),

    (0.10, 15_360),

    (0.00, 30_720),
]


total_trained = 0

best_score = float(
    "inf"
)

success_found = False


for blend, timesteps in SCHEDULE:

    print("\n")
    print("=" * 110)

    print(
        f"TEACHER BLEND = "
        f"{blend:.2f}"
    )

    print("=" * 110)


    raw_train_env.teacher_blend = (
        blend
    )


    remaining = timesteps


    while remaining > 0:

        chunk = min(
            5_120,
            remaining
        )


        model.learn(
            total_timesteps=chunk,

            reset_num_timesteps=False,

            progress_bar=True
        )


        remaining -= chunk
        total_trained += chunk


        print("\n")
        print("=" * 110)

        print(
            f"TEACHER-OFF EVALUATION | "
            f"TRAINED={total_trained:,} | "
            f"BLEND={blend:.2f}"
        )

        print("=" * 110)


        r = evaluate(
            model,
            f"{total_trained // 1000}K"
        )


        print("\n")

        print(
            "SUCCESS   :",
            r["success"]
        )

        print(
            "ALT       :",
            round(
                r["altitude"],
                2
            )
        )

        print(
            "ALT ERROR :",
            round(
                r[
                    "altitude_error"
                ],
                2
            )
        )

        print(
            "VS        :",
            round(
                r["vs"],
                2
            )
        )

        print(
            "MAX DRIFT :",
            round(
                r["max"],
                2
            )
        )

        print(
            "FINAL     :",
            round(
                r["final"],
                2
            )
        )

        print(
            "PATH      :",
            round(
                r["path"],
                2
            )
        )


        # ================================================
        # SAVE CHECKPOINT
        # ================================================

        model.save(
            f"{OUTPUT_DIR}/"
            f"stage1_distill_"
            f"{total_trained}_steps"
        )


        # ================================================
        # BEST SCORE
        # ================================================

        score = (

            10.0
            *
            r["max"]

            +

            3.0
            *
            r["altitude_error"]

            +

            r["path"]

            +

            2.0
            *
            r["final"]
        )


        if score < best_score:

            best_score = score

            model.save(
                f"{OUTPUT_DIR}/"
                "AH1S_STAGE1_DISTILL_BEST"
            )

            print(
                "✅ New best teacher-off model"
            )


        # ================================================
        # STRICT SUCCESS
        #
        # Evaluation itself has teacher OFF.
        # ================================================

        if r["success"]:

            model.save(
                f"{OUTPUT_DIR}/"
                "AH1S_STAGE1_DISTILL_SUCCESS"
            )


            print("\n")
            print(
                "🏆 SINGLE 4-ACTION PPO SUCCESS"
            )

            print(
                "Teacher OFF"
            )

            print(
                "Classical XY controller OFF"
            )

            print(
                "Model:"
            )

            print(
                f"{OUTPUT_DIR}/"
                "AH1S_STAGE1_DISTILL_SUCCESS.zip"
            )


            success_found = True

            break


    if success_found:

        break


# ============================================================
# FINAL SAVE
# ============================================================

model.save(
    f"{OUTPUT_DIR}/"
    "AH1S_STAGE1_DISTILL_FINAL"
)


print("\n")
print("=" * 110)
print("DISTILLATION FINISHED")
print("=" * 110)


if success_found:

    print(
        "✅ SINGLE PPO STRICT SUCCESS"
    )

else:

    print(
        "⚠️ Strict success not found."
    )

    print(
        "Best:"
    )

    print(
        f"{OUTPUT_DIR}/"
        "AH1S_STAGE1_DISTILL_BEST.zip"
    )


print("=" * 110)
