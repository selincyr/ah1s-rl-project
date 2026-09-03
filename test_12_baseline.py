import os
import numpy as np
from stable_baselines3 import PPO
from helicopter_env_12 import HelicopterEnv


MODEL_PATHS = [
    ("BEST", "models/task12_vertical300/best/best_model"),
    ("FINAL", "models/task12_vertical300/ppo_ah1s_vertical300_v12_final"),
]

EARTH_RADIUS_FT = 20_925_524.9
TARGET_ALTITUDE_FT = 300.0


def get_relative_position(env, initial_latitude_deg, initial_longitude_deg):
    """Return true east/north displacement from the takeoff point in feet.

    This is only a test-time measurement. It does not change V12 observation
    space, reward, policy input, or the trained model.
    """
    latitude_deg = float(env.fdm["position/lat-geod-deg"])
    longitude_deg = float(env.fdm["position/long-gc-deg"])

    lat0_rad = np.deg2rad(initial_latitude_deg)
    dlat = np.deg2rad(latitude_deg - initial_latitude_deg)
    dlon = np.deg2rad(longitude_deg - initial_longitude_deg)

    north_ft = EARTH_RADIUS_FT * dlat
    east_ft = EARTH_RADIUS_FT * np.cos(lat0_rad) * dlon
    distance_ft = float(np.hypot(east_ft, north_ft))

    return float(east_ft), float(north_ft), distance_ft


def model_exists(path):
    return os.path.exists(path) or os.path.exists(path + ".zip")


def test_model(label, path):
    print("\n" + "=" * 105)
    print(f"V12 {label} BASELINE TEST")
    print("=" * 105)

    if not model_exists(path):
        print(f"Model bulunamadi: {path}")
        print("Bu modeli atliyorum.")
        return

    env = HelicopterEnv()
    model = PPO.load(path)
    obs, info = env.reset()

    # Takeoff origin. This lets us measure the real trajectory of V12 without
    # adding east/north to its 13-dimensional observation.
    initial_latitude_deg = float(env.fdm["position/lat-geod-deg"])
    initial_longitude_deg = float(env.fdm["position/long-gc-deg"])

    max_altitude = float(info["altitude"])
    max_abs_vertical_speed = abs(float(info["vertical_speed"]))
    max_abs_forward_velocity = abs(float(info["forward_velocity"]))
    max_abs_lateral_velocity = abs(float(info["lateral_velocity"]))
    max_horizontal_distance = 0.0
    max_horizontal_speed = 0.0
    max_vs_tracking_error = abs(
        float(info["vertical_speed"]) - float(info["target_vertical_speed"])
    )
    max_abs_heading_error = abs(float(info["heading_error"]))
    max_abs_roll = abs(float(info["roll"]))
    max_abs_pitch = abs(float(info["pitch"]))
    max_hold = int(info["target_hold_steps"])

    entered_280_320 = False
    best_snapshot = None
    final_info = info
    final_east = 0.0
    final_north = 0.0
    final_distance = 0.0

    actions = [[], [], [], []]

    for step in range(1, 5001):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info

        altitude = float(info["altitude"])
        vertical_speed = float(info["vertical_speed"])
        target_vertical_speed = float(info["target_vertical_speed"])
        forward_velocity = float(info["forward_velocity"])
        lateral_velocity = float(info["lateral_velocity"])
        heading_error = float(info["heading_error"])
        roll = float(info["roll"])
        pitch = float(info["pitch"])
        hold = int(info["target_hold_steps"])

        east_ft, north_ft, horizontal_distance = get_relative_position(
            env,
            initial_latitude_deg,
            initial_longitude_deg,
        )

        horizontal_speed = float(
            np.hypot(forward_velocity, lateral_velocity)
        )

        vs_tracking_error = abs(
            vertical_speed - target_vertical_speed
        )

        max_altitude = max(max_altitude, altitude)
        max_abs_vertical_speed = max(
            max_abs_vertical_speed,
            abs(vertical_speed),
        )
        max_abs_forward_velocity = max(
            max_abs_forward_velocity,
            abs(forward_velocity),
        )
        max_abs_lateral_velocity = max(
            max_abs_lateral_velocity,
            abs(lateral_velocity),
        )
        max_horizontal_distance = max(
            max_horizontal_distance,
            horizontal_distance,
        )
        max_horizontal_speed = max(
            max_horizontal_speed,
            horizontal_speed,
        )
        max_vs_tracking_error = max(
            max_vs_tracking_error,
            vs_tracking_error,
        )
        max_abs_heading_error = max(
            max_abs_heading_error,
            abs(heading_error),
        )
        max_abs_roll = max(max_abs_roll, abs(roll))
        max_abs_pitch = max(max_abs_pitch, abs(pitch))
        max_hold = max(max_hold, hold)

        final_east = east_ft
        final_north = north_ft
        final_distance = horizontal_distance

        for i in range(4):
            actions[i].append(abs(float(action[i])))

        if 280.0 <= altitude <= 320.0:
            entered_280_320 = True

            # This score is TEST-ONLY. It does not affect training.
            # It is used simply to choose a representative near-target sample.
            snapshot_score = (
                abs(altitude - TARGET_ALTITUDE_FT) / 10.0
                + abs(vertical_speed) / 2.0
                + horizontal_speed / 10.0
                + horizontal_distance / 20.0
                + abs(heading_error) / 0.30
                + abs(roll) / 0.20
                + abs(pitch) / 0.20
            )

            if (
                best_snapshot is None
                or snapshot_score < best_snapshot["score"]
            ):
                best_snapshot = {
                    "score": snapshot_score,
                    "step": step,
                    "altitude": altitude,
                    "vertical_speed": vertical_speed,
                    "target_vertical_speed": target_vertical_speed,
                    "vs_tracking_error": vs_tracking_error,
                    "forward_velocity": forward_velocity,
                    "lateral_velocity": lateral_velocity,
                    "horizontal_speed": horizontal_speed,
                    "east": east_ft,
                    "north": north_ft,
                    "distance": horizontal_distance,
                    "heading_error": heading_error,
                    "roll": roll,
                    "pitch": pitch,
                    "hold": hold,
                }

        if step % 25 == 0:
            print(
                f"Step {step:4d} | {info['phase']:<7} | "
                f"Alt {altitude:7.2f} | "
                f"VS {vertical_speed:6.2f} | "
                f"TgtVS {target_vertical_speed:6.2f} | "
                f"VSErr {vs_tracking_error:6.2f} | "
                f"Fwd {forward_velocity:6.2f} | "
                f"Lat {lateral_velocity:6.2f} | "
                f"HSpd {horizontal_speed:6.2f} | "
                f"E {east_ft:7.2f} | "
                f"N {north_ft:7.2f} | "
                f"Dist {horizontal_distance:7.2f} | "
                f"Head {heading_error:6.3f} | "
                f"Roll {roll:6.3f} | "
                f"Pitch {pitch:6.3f} | "
                f"Hold {hold:3d}"
            )

        if terminated or truncated:
            print(
                f"Episode ended at step {step} | "
                f"reason={info.get('termination_reason')} | "
                f"terminated={terminated} truncated={truncated}"
            )
            break

    altitude_overshoot = max(
        0.0,
        max_altitude - TARGET_ALTITUDE_FT,
    )

    print("\n" + "-" * 105)
    print(f"V12 {label} BASELINE SUMMARY")
    print("-" * 105)
    print(f"Max altitude              : {max_altitude:.2f} ft")
    print(f"Altitude overshoot         : {altitude_overshoot:.2f} ft")
    print(f"Max |vertical speed|      : {max_abs_vertical_speed:.2f} ft/s")
    print(f"Max VS tracking error     : {max_vs_tracking_error:.2f} ft/s")
    print(f"Max |forward velocity|    : {max_abs_forward_velocity:.2f} ft/s")
    print(f"Max |lateral velocity|    : {max_abs_lateral_velocity:.2f} ft/s")
    print(f"Max horizontal speed      : {max_horizontal_speed:.2f} ft/s")
    print(f"Max horizontal distance   : {max_horizontal_distance:.2f} ft")
    print(f"Max |heading error|       : {max_abs_heading_error:.3f} rad")
    print(f"Max |roll|                : {max_abs_roll:.3f} rad")
    print(f"Max |pitch|               : {max_abs_pitch:.3f} rad")
    print(f"Max hold                  : {max_hold}/100")
    print(f"Entered 280-320 ft        : {entered_280_320}")

    print("\nACTION SUMMARY")
    names = ["Collective", "Elevator", "Aileron", "Rudder"]
    for name, values in zip(names, actions):
        if values:
            print(
                f"{name:<10} mean |a|={np.mean(values):.3f} "
                f"max |a|={np.max(values):.3f}"
            )

    print("\nBEST 280-320 FT SNAPSHOT")
    if best_snapshot is None:
        print("No sample entered 280-320 ft.")
    else:
        b = best_snapshot
        print(f"Step                 : {b['step']}")
        print(f"Test-only score      : {b['score']:.3f}")
        print(f"Altitude             : {b['altitude']:.2f} ft")
        print(f"Vertical speed       : {b['vertical_speed']:.2f} ft/s")
        print(f"Target vertical speed: {b['target_vertical_speed']:.2f} ft/s")
        print(f"VS tracking error    : {b['vs_tracking_error']:.2f} ft/s")
        print(f"Forward velocity     : {b['forward_velocity']:.2f} ft/s")
        print(f"Lateral velocity     : {b['lateral_velocity']:.2f} ft/s")
        print(f"Horizontal speed     : {b['horizontal_speed']:.2f} ft/s")
        print(f"East displacement    : {b['east']:.2f} ft")
        print(f"North displacement   : {b['north']:.2f} ft")
        print(f"Horizontal distance  : {b['distance']:.2f} ft")
        print(f"Heading error        : {b['heading_error']:.3f} rad")
        print(f"Roll                 : {b['roll']:.3f} rad")
        print(f"Pitch                : {b['pitch']:.3f} rad")
        print(f"Hold                 : {b['hold']}/100")

    print("\nFINAL STATE")
    print(f"Phase                : {final_info['phase']}")
    print(f"Altitude             : {final_info['altitude']:.2f} ft")
    print(f"Vertical speed       : {final_info['vertical_speed']:.2f} ft/s")
    print(f"Forward velocity     : {final_info['forward_velocity']:.2f} ft/s")
    print(f"Lateral velocity     : {final_info['lateral_velocity']:.2f} ft/s")
    print(f"East displacement    : {final_east:.2f} ft")
    print(f"North displacement   : {final_north:.2f} ft")
    print(f"Horizontal distance  : {final_distance:.2f} ft")
    print(f"Heading error        : {final_info['heading_error']:.3f} rad")
    print(f"Hold                 : {final_info['target_hold_steps']}/100")
    print(f"Success              : {final_info['success']}")
    print(f"Termination reason   : {final_info.get('termination_reason')}")

    env.close()


def main():
    for label, path in MODEL_PATHS:
        test_model(label, path)


if __name__ == "__main__":
    main()
