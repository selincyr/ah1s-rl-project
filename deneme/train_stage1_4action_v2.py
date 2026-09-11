import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_stage1_4action_v2 import (
    HelicopterEnvStage1FourActionV2
)


SOURCE_MODEL = (
    "models_v2/"
    "AH1S_STAGE1_SUCCESS.zip"
)

OUTPUT_DIR = (
    "models_stage1_4action_v2"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRAIN ENV
# ============================================================

train_env = Monitor(
    HelicopterEnvStage1FourActionV2(
        teacher_model_path=SOURCE_MODEL,
        training_mode=True
    )
)


# ============================================================
# ORIGINAL SUCCESSFUL STAGE 1
# ============================================================

print("=" * 110)
print("LOADING ORIGINAL STAGE 1")
print("=" * 110)

source_model = PPO.load(
    SOURCE_MODEL
)

print("✅ Stage 1 teacher loaded")


# ============================================================
# NEW SINGLE PPO - 4 ACTIONS
# ============================================================

model = PPO(
    "MlpPolicy",

    train_env,

    learning_rate=4e-5,

    n_steps=1024,

    batch_size=64,

    n_epochs=10,

    gamma=0.995,

    gae_lambda=0.95,

    clip_range=0.12,

    ent_coef=0.0005,

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
        "logs_stage1_4action_v2/"
        "tensorboard/"
    ),
)


# ============================================================
# SELECTIVE TRANSFER
# ============================================================

print("\n")
print("=" * 110)
print("SELECTIVE TRANSFER")
print("=" * 110)

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
    # ACTION OUTPUT
    #
    # only collective old output is meaningful
    # --------------------------------------------------------

    if key == "action_net.weight":

        new_tensor = (
            target_tensor.clone()
        )

        new_tensor[0, :] = (
            source_tensor[0, :]
        )

        new_tensor[1:, :] = 0.0

        target_state[key] = new_tensor

        continue


    if key == "action_net.bias":

        new_tensor = (
            target_tensor.clone()
        )

        new_tensor[0] = (
            source_tensor[0]
        )

        new_tensor[1:] = 0.0

        target_state[key] = new_tensor

        continue


    # --------------------------------------------------------
    # STD
    # --------------------------------------------------------

    if key == "log_std":

        new_tensor = (
            target_tensor.clone()
        )

        new_tensor[0] = (
            source_tensor[0]
        )

        new_tensor[1] = -1.7
        new_tensor[2] = -1.7
        new_tensor[3] = -2.0

        target_state[key] = new_tensor

        continue


    # --------------------------------------------------------
    # EXACT
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
    # 12 OBS -> 18 OBS
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

        old_inputs = (
            source_tensor.shape[1]
        )

        new_tensor[
            :,
            :old_inputs
        ] = source_tensor

        new_tensor[
            :,
            old_inputs:
        ] = 0.0

        target_state[key] = new_tensor


model.policy.load_state_dict(
    target_state
)

print("✅ Selective transfer complete")


# ============================================================
# EVALUATION
#
# IMPORTANT:
# teacher completely OFF.
# ============================================================

def evaluate(model, label):

    env = (
        HelicopterEnvStage1FourActionV2(
            teacher_model_path=None,
            training_mode=False
        )
    )

    obs, info = env.reset()

    total_reward = 0.0
    next_print = 0.0

    for step in range(
        env.max_steps
    ):

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
        ) = env.step(action)

        total_reward += reward

        t = (
            (step + 1)
            *
            env.dt
        )

        if t >= next_print:

            hdg_deg = (
                info["heading_error"]
                *
                180.0
                /
                np.pi
            )

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
                f"HDG={hdg_deg:5.2f}° | "
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

        "time":
            t,

        "altitude":
            info["altitude"],

        "altitude_error":
            info["altitude_error"],

        "vertical_speed":
            info["vertical_speed"],

        "max_drift":
            info["max_drift"],

        "final_drift":
            info["drift"],

        "path":
            info["path"],

        "north":
            info["north"],

        "east":
            info["east"],

        "heading":
            (
                info["heading_error"]
                *
                180.0
                /
                np.pi
            ),

        "reward":
            total_reward,
    }

    env.close()

    return result


# ============================================================
# BASELINE
# ============================================================

print("\n")
print("=" * 110)
print("V2 BASELINE - TEACHER OFF")
print("=" * 110)

baseline = evaluate(
    model,
    "BASE"
)

print(
    "\nBASELINE:",
    baseline
)


# ============================================================
# TRAIN
# ============================================================

CHUNK = 5_120
TOTAL_LIMIT = 122_880

trained = 0

best_score = float("inf")
success_found = False


while trained < TOTAL_LIMIT:

    print("\n")
    print("=" * 110)

    print(
        f"TRAINING "
        f"{trained:,} -> "
        f"{trained + CHUNK:,}"
    )

    print("=" * 110)

    model.learn(
        total_timesteps=CHUNK,
        reset_num_timesteps=False,
        progress_bar=True
    )

    trained += CHUNK

    checkpoint = (
        f"{OUTPUT_DIR}/"
        f"stage1_4action_v2_"
        f"{trained}_steps"
    )

    model.save(checkpoint)


    print("\n")
    print("=" * 110)

    print(
        f"V2 EVALUATION @ "
        f"{trained:,} STEPS "
        f"(TEACHER OFF)"
    )

    print("=" * 110)

    result = evaluate(
        model,
        f"{trained // 1000}K"
    )


    print("\n")

    print(
        "SUCCESS   :",
        result["success"]
    )

    print(
        "ALT       :",
        round(
            result["altitude"],
            2
        ),
        "ft"
    )

    print(
        "ALT ERROR :",
        round(
            result["altitude_error"],
            2
        ),
        "ft"
    )

    print(
        "VS        :",
        round(
            result["vertical_speed"],
            2
        ),
        "ft/s"
    )

    print(
        "MAX DRIFT :",
        round(
            result["max_drift"],
            2
        ),
        "ft"
    )

    print(
        "FINAL     :",
        round(
            result["final_drift"],
            2
        ),
        "ft"
    )

    print(
        "PATH      :",
        round(
            result["path"],
            2
        ),
        "ft"
    )

    print(
        "NORTH     :",
        round(
            result["north"],
            2
        ),
        "ft"
    )

    print(
        "EAST      :",
        round(
            result["east"],
            2
        ),
        "ft"
    )

    print(
        "HEADING   :",
        round(
            result["heading"],
            2
        ),
        "deg"
    )

    print(
        "REWARD    :",
        round(
            result["reward"],
            2
        )
    )


    # ========================================================
    # CANDIDATE SCORE
    # ========================================================

    score = (

        10.0
        *
        result["max_drift"]

        +

        3.0
        *
        result["altitude_error"]

        +

        1.0
        *
        result["path"]

        +

        2.0
        *
        result["final_drift"]
    )


    if score < best_score:

        best_score = score

        model.save(
            f"{OUTPUT_DIR}/"
            "AH1S_STAGE1_4ACTION_V2_BEST"
        )

        print(
            "✅ New best V2 candidate saved."
        )


    if result["success"]:

        model.save(
            f"{OUTPUT_DIR}/"
            "AH1S_STAGE1_4ACTION_V2_SUCCESS"
        )

        success_found = True

        print("\n")
        print(
            "🏆 STRICT STRAIGHT TAKEOFF SUCCESS"
        )

        print(
            f"{OUTPUT_DIR}/"
            "AH1S_STAGE1_4ACTION_V2_SUCCESS.zip"
        )

        print(
            "Teacher is OFF in this evaluation."
        )

        break


# ============================================================
# FINAL
# ============================================================

model.save(
    f"{OUTPUT_DIR}/"
    "AH1S_STAGE1_4ACTION_V2_FINAL"
)

print("\n")
print("=" * 110)
print("V2 TRAINING FINISHED")
print("=" * 110)

if success_found:

    print(
        "✅ SUCCESS"
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
        "AH1S_STAGE1_4ACTION_V2_BEST.zip"
    )

print("=" * 110)
