import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_stage1_curriculum_v3 import (
    HelicopterEnvStage1CurriculumV3
)


SOURCE_MODEL = (
    "models_v2/"
    "AH1S_STAGE1_SUCCESS.zip"
)

OUTPUT_DIR = (
    "models_stage1_curriculum_v3"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CREATE ENV
# ============================================================

def make_env(
    phase
):

    return Monitor(
        HelicopterEnvStage1CurriculumV3(
            phase=phase,
            teacher_model_path=SOURCE_MODEL
        )
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
    "✅ Original Stage 1 loaded"
)


# ============================================================
# PHASE A ENV
# ============================================================

train_env = make_env(
    HelicopterEnvStage1CurriculumV3.PHASE_XY_EARLY
)


# ============================================================
# NEW 4 ACTION PPO
# ============================================================

model = PPO(
    "MlpPolicy",

    train_env,

    learning_rate=7e-5,

    n_steps=1024,

    batch_size=64,

    n_epochs=10,

    gamma=0.995,

    gae_lambda=0.95,

    clip_range=0.15,

    ent_coef=0.001,

    vf_coef=0.5,

    max_grad_norm=0.5,

    target_kl=0.02,

    policy_kwargs=dict(
        net_arch=[
            128,
            128
        ],

        log_std_init=-1.3
    ),

    verbose=1,

    tensorboard_log=(
        "logs_stage1_curriculum_v3/"
        "tensorboard/"
    ),
)


# ============================================================
# SELECTIVE TRANSFER
# ============================================================

source_state = (
    source_model.policy.state_dict()
)

target_state = (
    model.policy.state_dict()
)


for key in target_state.keys():

    if key not in source_state:

        continue


    source_tensor = (
        source_state[key]
    )

    target_tensor = (
        target_state[key]
    )


    # --------------------------------------------------------
    # Action head
    #
    # only collective transferred
    # --------------------------------------------------------

    if key == "action_net.weight":

        new_tensor = (
            target_tensor.clone()
        )

        new_tensor[0, :] = (
            source_tensor[0, :]
        )

        new_tensor[1:, :] = 0.0

        target_state[key] = (
            new_tensor
        )

        continue


    if key == "action_net.bias":

        new_tensor = (
            target_tensor.clone()
        )

        new_tensor[0] = (
            source_tensor[0]
        )

        new_tensor[1:] = 0.0

        target_state[key] = (
            new_tensor
        )

        continue


    # --------------------------------------------------------
    # Exploration
    # --------------------------------------------------------

    if key == "log_std":

        new_tensor = (
            target_tensor.clone()
        )

        new_tensor[0] = (
            source_tensor[0]
        )

        # More cyclic exploration than V2.
        new_tensor[1] = -1.2
        new_tensor[2] = -1.2

        # Rudder smaller
        new_tensor[3] = -1.8

        target_state[key] = (
            new_tensor
        )

        continue


    # --------------------------------------------------------
    # Exact shapes
    # --------------------------------------------------------

    if (
        source_tensor.shape
        ==
        target_tensor.shape
    ):

        target_state[key] = (
            source_tensor.clone()
        )

        continue


    # --------------------------------------------------------
    # Old 12 obs -> new 18 obs
    # --------------------------------------------------------

    if (
        source_tensor.ndim == 2
        and
        target_tensor.ndim == 2
        and
        source_tensor.shape[0]
        ==
        target_tensor.shape[0]
        and
        source_tensor.shape[1]
        <
        target_tensor.shape[1]
    ):

        new_tensor = (
            target_tensor.clone()
        )

        old_n = (
            source_tensor.shape[1]
        )

        new_tensor[
            :,
            :old_n
        ] = source_tensor

        new_tensor[
            :,
            old_n:
        ] = 0.0

        target_state[key] = (
            new_tensor
        )


model.policy.load_state_dict(
    target_state
)

print(
    "✅ Selective transfer complete"
)


# ============================================================
# EVALUATION A
#
# Teacher collective ON.
# We evaluate only first 35 seconds.
# ============================================================

def evaluate_xy_early(
    model,
    label
):

    env = (
        HelicopterEnvStage1CurriculumV3(
            phase=(
                HelicopterEnvStage1CurriculumV3
                .PHASE_XY_EARLY
            ),

            teacher_model_path=
                SOURCE_MODEL
        )
    )

    obs, info = env.reset()

    total_reward = 0.0

    next_print = 0.0


    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
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

        total_reward += reward

        t = (
            env.steps
            *
            env.dt
        )


        if t >= next_print:

            print(
                f"{label} | "
                f"t={t:5.1f} | "
                f"ALT={info['altitude']:7.2f} | "
                f"DRIFT={info['drift']:6.2f} | "
                f"MAX={info['max_drift']:6.2f} | "
                f"PATH={info['path']:6.2f} | "
                f"N={info['north']:6.2f} | "
                f"E={info['east']:6.2f} | "
                f"VN={info['vn']:6.2f} | "
                f"VE={info['ve']:6.2f} | "
                f"ELE={info['elevator']:.5f} | "
                f"AIL={info['aileron']:.5f}"
            )

            next_print += 5.0


        if (
            terminated
            or
            truncated
        ):

            break


    result = {
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

        "reward":
            total_reward,
    }

    env.close()

    return result


# ============================================================
# PHASE A TRAINING
# ============================================================

print("\n")
print("=" * 110)
print("PHASE A — EARLY VERTICAL LINE")
print("=" * 110)

print(
    "Teacher: collective only"
)

print(
    "PPO: elevator + aileron + rudder"
)

print(
    "Goal: first 35 sec MAX DRIFT <= 8 ft"
)


PHASE_A_CHUNK = 5_120
PHASE_A_LIMIT = 81_920

phase_a_steps = 0

best_a = float(
    "inf"
)


while (
    phase_a_steps
    <
    PHASE_A_LIMIT
):

    model.learn(
        total_timesteps=
            PHASE_A_CHUNK,

        reset_num_timesteps=False,

        progress_bar=True
    )

    phase_a_steps += (
        PHASE_A_CHUNK
    )


    print("\n")
    print("=" * 110)

    print(
        f"PHASE A EVALUATION "
        f"@ {phase_a_steps:,}"
    )

    print("=" * 110)


    r = evaluate_xy_early(
        model,
        "A"
    )


    print(
        "\nMAX  :",
        round(
            r["max"],
            2
        )
    )

    print(
        "FINAL:",
        round(
            r["final"],
            2
        )
    )

    print(
        "PATH :",
        round(
            r["path"],
            2
        )
    )


    score = (
        10.0
        *
        r["max"]

        +
        r["path"]

        +
        2.0
        *
        r["final"]
    )


    if score < best_a:

        best_a = score

        model.save(
            f"{OUTPUT_DIR}/"
            "PHASE_A_BEST"
        )

        print(
            "✅ New Phase A best"
        )


    if (
        r["max"] <= 8.0
        and
        r["final"] <= 6.0
    ):

        print(
            "🏆 PHASE A PASSED"
        )

        model.save(
            f"{OUTPUT_DIR}/"
            "PHASE_A_SUCCESS"
        )

        break


# ============================================================
# PHASE B
# ============================================================

print("\n")
print("=" * 110)
print("PHASE B — FULL VERTICAL TAKEOFF")
print("=" * 110)


phase_b_env = make_env(
    HelicopterEnvStage1CurriculumV3.PHASE_XY_FULL
)

model.set_env(
    phase_b_env
)


def evaluate_full_teacher(
    model,
    label
):

    env = (
        HelicopterEnvStage1CurriculumV3(
            phase=(
                HelicopterEnvStage1CurriculumV3
                .PHASE_XY_FULL
            ),

            teacher_model_path=
                SOURCE_MODEL
        )
    )

    obs, info = env.reset()

    next_print = 0.0


    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
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
                f"DRIFT={info['drift']:6.2f} | "
                f"MAX={info['max_drift']:6.2f} | "
                f"PATH={info['path']:6.2f} | "
                f"N={info['north']:6.2f} | "
                f"E={info['east']:6.2f}"
            )

            next_print += 10.0


        if terminated or truncated:

            break


    result = {
        "success":
            info["success"],

        "alt":
            info["altitude"],

        "max":
            info["max_drift"],

        "final":
            info["drift"],

        "path":
            info["path"],
    }

    env.close()

    return result


PHASE_B_CHUNK = 5_120
PHASE_B_LIMIT = 102_400

phase_b_steps = 0

best_b = float(
    "inf"
)


while (
    phase_b_steps
    <
    PHASE_B_LIMIT
):

    model.learn(
        total_timesteps=
            PHASE_B_CHUNK,

        reset_num_timesteps=False,

        progress_bar=True
    )

    phase_b_steps += (
        PHASE_B_CHUNK
    )


    print("\n")
    print("=" * 110)

    print(
        f"PHASE B EVALUATION "
        f"@ {phase_b_steps:,}"
    )

    print("=" * 110)


    r = evaluate_full_teacher(
        model,
        "B"
    )


    print(
        "\nSUCCESS:",
        r["success"]
    )

    print(
        "ALT    :",
        round(
            r["alt"],
            2
        )
    )

    print(
        "MAX    :",
        round(
            r["max"],
            2
        )
    )

    print(
        "FINAL  :",
        round(
            r["final"],
            2
        )
    )

    print(
        "PATH   :",
        round(
            r["path"],
            2
        )
    )


    score = (
        10.0
        *
        r["max"]

        +
        r["path"]

        +
        2.0
        *
        r["final"]
    )


    if score < best_b:

        best_b = score

        model.save(
            f"{OUTPUT_DIR}/"
            "PHASE_B_BEST"
        )

        print(
            "✅ New Phase B best"
        )


    if r["success"]:

        print(
            "🏆 PHASE B PASSED"
        )

        model.save(
            f"{OUTPUT_DIR}/"
            "PHASE_B_SUCCESS"
        )

        break


# ============================================================
# PHASE C
#
# SAME PPO.
# Teacher fades.
# ============================================================

print("\n")
print("=" * 110)
print("PHASE C — JOINT 4-ACTION PPO")
print("=" * 110)


phase_c_env = make_env(
    HelicopterEnvStage1CurriculumV3.PHASE_JOINT
)

model.set_env(
    phase_c_env
)


# ============================================================
# FINAL TEACHER-OFF EVALUATOR
# ============================================================

def evaluate_final(
    model,
    label
):

    env = (
        HelicopterEnvStage1CurriculumV3(
            phase=(
                HelicopterEnvStage1CurriculumV3
                .PHASE_FINAL
            ),

            teacher_model_path=None
        )
    )

    obs, info = env.reset()

    next_print = 0.0


    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
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


        if terminated or truncated:

            break


    result = {

        "success":
            info["success"],

        "alt":
            info["altitude"],

        "alt_err":
            info["altitude_error"],

        "vs":
            info["vertical_speed"],

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


PHASE_C_CHUNK = 5_120
PHASE_C_LIMIT = 153_600

phase_c_steps = 0

best_c = float(
    "inf"
)

success_found = False


while (
    phase_c_steps
    <
    PHASE_C_LIMIT
):

    model.learn(
        total_timesteps=
            PHASE_C_CHUNK,

        reset_num_timesteps=False,

        progress_bar=True
    )

    phase_c_steps += (
        PHASE_C_CHUNK
    )


    print("\n")
    print("=" * 110)

    print(
        f"FINAL TEACHER-OFF "
        f"EVALUATION @ "
        f"{phase_c_steps:,}"
    )

    print("=" * 110)


    r = evaluate_final(
        model,
        "FINAL"
    )


    print(
        "\nSUCCESS   :",
        r["success"]
    )

    print(
        "ALT       :",
        round(
            r["alt"],
            2
        )
    )

    print(
        "ALT ERROR :",
        round(
            r["alt_err"],
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


    score = (

        10.0
        *
        r["max"]

        +

        3.0
        *
        r["alt_err"]

        +

        r["path"]

        +

        2.0
        *
        r["final"]
    )


    if score < best_c:

        best_c = score

        model.save(
            f"{OUTPUT_DIR}/"
            "AH1S_STAGE1_V3_BEST"
        )

        print(
            "✅ New final best"
        )


    if r["success"]:

        model.save(
            f"{OUTPUT_DIR}/"
            "AH1S_STAGE1_V3_SUCCESS"
        )

        print(
            "\n🏆🏆🏆 STRICT 4-ACTION "
            "VERTICAL TAKEOFF SUCCESS"
        )

        success_found = True

        break


# ============================================================
# SAVE
# ============================================================

model.save(
    f"{OUTPUT_DIR}/"
    "AH1S_STAGE1_V3_FINAL"
)


print("\n")
print("=" * 110)
print("V3 CURRICULUM FINISHED")
print("=" * 110)

if success_found:

    print(
        "✅ FINAL MODEL:"
    )

    print(
        f"{OUTPUT_DIR}/"
        "AH1S_STAGE1_V3_SUCCESS.zip"
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
        "AH1S_STAGE1_V3_BEST.zip"
    )

print("=" * 110)
