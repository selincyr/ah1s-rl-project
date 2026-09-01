import numpy as np
from stable_baselines3 import PPO

from helicopter_env_4 import HelicopterEnv


MODEL_PATH = "models/task4_hover/best/best_model"


env = HelicopterEnv()

model = PPO.load(
    MODEL_PATH,
    env=env
)

obs, info = env.reset()


max_abs_action = np.zeros(
    4,
    dtype=np.float32
)

sum_abs_action = np.zeros(
    4,
    dtype=np.float64
)

steps_run = 0


print()
print("=" * 90)
print("ACTION DIAGNOSTIC")
print("=" * 90)

print(
    "Action sirasi:"
)

print(
    "[collective, elevator, aileron, rudder]"
)

print()


for step in range(5000):

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    action = np.asarray(
        action,
        dtype=np.float32
    )

    max_abs_action = np.maximum(
        max_abs_action,
        np.abs(action)
    )

    sum_abs_action += np.abs(action)

    steps_run += 1

    (
        obs,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    if step % 25 == 0:

        print(
            f"Step {step:4d} | "
            f"Alt {info['altitude']:7.1f} | "
            f"Lat {info['lateral_velocity']:7.2f} | "
            f"HeadErr {info['heading_error']:7.3f} | "
            f"Roll {info['roll']:7.3f} | "
            f"YawRate {info['yaw_rate']:7.3f} | "
            f"A = "
            f"[{action[0]:+.3f}, "
            f"{action[1]:+.3f}, "
            f"{action[2]:+.3f}, "
            f"{action[3]:+.3f}]"
        )

    if (
        abs(info["lateral_velocity"]) > 40.0
        or abs(info["heading_error"]) > 2.0
    ):

        print()
        print("!!! KONTROL KAYBI BOLGESI !!!")

        print(
            f"Step: {step}"
        )

        print(
            f"Altitude: "
            f"{info['altitude']:.2f}"
        )

        print(
            f"Lateral velocity: "
            f"{info['lateral_velocity']:.2f}"
        )

        print(
            f"Heading error: "
            f"{info['heading_error']:.3f}"
        )

        print(
            f"Roll: "
            f"{info['roll']:.3f}"
        )

        print(
            f"Pitch: "
            f"{info['pitch']:.3f}"
        )

        print(
            f"Yaw rate: "
            f"{info['yaw_rate']:.3f}"
        )

        print(
            "Action:",
            action
        )

        print()

    if terminated or truncated:

        print()
        print("=" * 90)

        print(
            f"Episode ended at step {step}"
        )

        print(
            "Reason:",
            info.get(
                "termination_reason"
            )
        )

        break


mean_abs_action = (
    sum_abs_action / steps_run
)


print()
print("=" * 90)
print("ACTION SUMMARY")
print("=" * 90)

names = [
    "Collective",
    "Elevator",
    "Aileron",
    "Rudder"
]

for i, name in enumerate(names):

    print(
        f"{name:10s} | "
        f"Mean |action| = "
        f"{mean_abs_action[i]:.3f} | "
        f"Max |action| = "
        f"{max_abs_action[i]:.3f}"
    )


print()
print("FINAL STATE")

print(
    "Altitude:",
    round(info["altitude"], 2)
)

print(
    "Lateral velocity:",
    round(
        info["lateral_velocity"],
        2
    )
)

print(
    "Heading error:",
    round(
        info["heading_error"],
        3
    )
)

print(
    "Roll:",
    round(
        info["roll"],
        3
    )
)

print(
    "Pitch:",
    round(
        info["pitch"],
        3
    )
)

print(
    "Termination:",
    info.get(
        "termination_reason"
    )
)

env.close()
