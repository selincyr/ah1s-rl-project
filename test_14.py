import numpy as np
from stable_baselines3 import PPO
from helicopter_env_14 import HelicopterEnv


MODEL_PATHS = [
    ("BEST", "models/task14_vertical300/best/best_model"),
    ("FINAL", "models/task14_vertical300/ppo_ah1s_vertical300_v14_final"),
]


def test_model(label, path):
    print("\n" + "=" * 100)
    print(f"{label} MODEL TEST")
    print("=" * 100)

    env = HelicopterEnv()
    model = PPO.load(path)
    obs, info = env.reset()

    max_alt = 0.0
    max_vs = 0.0
    max_fwd = 0.0
    max_lat = 0.0
    max_dist = 0.0
    max_heading = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    max_hold = 0

    best_snapshot = None
    final_info = info

    actions = [[], [], [], []]

    for step in range(1, 5001):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info

        alt = float(info["altitude"])
        vs = float(info["vertical_speed"])
        fwd = float(info["forward_velocity"])
        lat = float(info["lateral_velocity"])
        east = float(info["relative_east"])
        north = float(info["relative_north"])
        dist = float(info["horizontal_distance"])
        head = float(info["heading_error"])
        roll = float(info["roll"])
        pitch = float(info["pitch"])
        hold = int(info["target_hold_steps"])

        max_alt = max(max_alt, alt)
        max_vs = max(max_vs, abs(vs))
        max_fwd = max(max_fwd, abs(fwd))
        max_lat = max(max_lat, abs(lat))
        max_dist = max(max_dist, dist)
        max_heading = max(max_heading, abs(head))
        max_roll = max(max_roll, abs(roll))
        max_pitch = max(max_pitch, abs(pitch))
        max_hold = max(max_hold, hold)

        for i in range(4):
            actions[i].append(abs(float(action[i])))

        # Main V13 objective score: near 300 ft AND near takeoff point.
        score = (
            abs(alt - 300.0) / 10.0
            + abs(vs) / 2.0
            + abs(fwd) / 6.0
            + abs(lat) / 6.0
            + dist / 5.0
            + abs(head) / 0.25
            + abs(roll) / 0.20
            + abs(pitch) / 0.20
        )

        if 280.0 <= alt <= 320.0:
            if best_snapshot is None or score < best_snapshot["score"]:
                best_snapshot = {
                    "score": score,
                    "step": step,
                    "alt": alt,
                    "vs": vs,
                    "fwd": fwd,
                    "lat": lat,
                    "east": east,
                    "north": north,
                    "dist": dist,
                    "head": head,
                    "roll": roll,
                    "pitch": pitch,
                    "hold": hold,
                }

        if step % 25 == 0:
            print(
                f"Step {step:4d} | {info['phase']:<7} | "
                f"Alt {alt:7.2f} | VS {vs:6.2f} | TgtVS {info['target_vertical_speed']:6.2f} | "
                f"Fwd {fwd:6.2f} | Lat {lat:6.2f} | "
                f"E {east:7.2f} | N {north:7.2f} | Dist {dist:6.2f} | "
                f"Head {head:6.3f} | Roll {roll:6.3f} | Pitch {pitch:6.3f} | Hold {hold:3d}"
            )

        if terminated or truncated:
            print(
                f"Episode ended at step {step} | "
                f"reason={info.get('termination_reason')} | "
                f"terminated={terminated} truncated={truncated}"
            )
            break

    print("\n" + "-" * 100)
    print(f"{label} SUMMARY")
    print("-" * 100)
    print(f"Max altitude             : {max_alt:.2f} ft")
    print(f"Max |vertical speed|     : {max_vs:.2f} ft/s")
    print(f"Max |forward velocity|   : {max_fwd:.2f} ft/s")
    print(f"Max |lateral velocity|   : {max_lat:.2f} ft/s")
    print(f"Max horizontal distance  : {max_dist:.2f} ft")
    print(f"Max |heading error|      : {max_heading:.3f} rad")
    print(f"Max |roll|               : {max_roll:.3f} rad")
    print(f"Max |pitch|              : {max_pitch:.3f} rad")
    print(f"Max hold                 : {max_hold}/100")

    print("\nACTION SUMMARY")
    names = ["Collective", "Elevator", "Aileron", "Rudder"]
    for name, values in zip(names, actions):
        print(
            f"{name:<10} mean |a|={np.mean(values):.3f} "
            f"max |a|={np.max(values):.3f}"
        )

    print("\nBEST VERTICAL-HOVER SNAPSHOT")
    if best_snapshot is None:
        print("No sample entered 280-320 ft.")
    else:
        b = best_snapshot
        print(f"Step                : {b['step']}")
        print(f"Score               : {b['score']:.3f}")
        print(f"Altitude            : {b['alt']:.2f} ft")
        print(f"Vertical speed      : {b['vs']:.2f} ft/s")
        print(f"Forward velocity    : {b['fwd']:.2f} ft/s")
        print(f"Lateral velocity    : {b['lat']:.2f} ft/s")
        print(f"East displacement   : {b['east']:.2f} ft")
        print(f"North displacement  : {b['north']:.2f} ft")
        print(f"Horizontal distance : {b['dist']:.2f} ft")
        print(f"Heading error       : {b['head']:.3f} rad")
        print(f"Roll                : {b['roll']:.3f} rad")
        print(f"Pitch               : {b['pitch']:.3f} rad")
        print(f"Hold                : {b['hold']}/100")

    print("\nFINAL STATE")
    print(f"Phase               : {final_info['phase']}")
    print(f"Altitude            : {final_info['altitude']:.2f} ft")
    print(f"Vertical speed      : {final_info['vertical_speed']:.2f} ft/s")
    print(f"Forward velocity    : {final_info['forward_velocity']:.2f} ft/s")
    print(f"Lateral velocity    : {final_info['lateral_velocity']:.2f} ft/s")
    print(f"East displacement   : {final_info['relative_east']:.2f} ft")
    print(f"North displacement  : {final_info['relative_north']:.2f} ft")
    print(f"Horizontal distance : {final_info['horizontal_distance']:.2f} ft")
    print(f"Heading error       : {final_info['heading_error']:.3f} rad")
    print(f"Hold                : {final_info['target_hold_steps']}/100")
    print(f"Success             : {final_info['success']}")
    print(f"Termination reason  : {final_info.get('termination_reason')}")

    env.close()


def main():
    for label, path in MODEL_PATHS:
        test_model(label, path)


if __name__ == "__main__":
    main()
