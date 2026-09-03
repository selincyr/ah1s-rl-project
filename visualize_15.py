import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stable_baselines3 import PPO
from helicopter_env_15 import HelicopterEnv


MODEL_PATH = "models/task15_vertical300/best/best_model"
OUTPUT_GIF = "ah1s_v15_best.gif"
MAX_STEPS = 5000


env = HelicopterEnv()
model = PPO.load(MODEL_PATH)
obs, info = env.reset()

xs = [float(info["relative_east"])]
ys = [float(info["relative_north"])]
zs = [float(info["altitude"])]
rolls = [float(info["roll"])]
pitches = [float(info["pitch"])]
headings = [float(info["heading"])]
vs_values = [float(info["vertical_speed"])]
fwd_values = [float(info["forward_velocity"])]
lat_values = [float(info["lateral_velocity"])]
dist_values = [float(info["horizontal_distance"])]
hold_values = [int(info["target_hold_steps"])]

for step in range(MAX_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    xs.append(float(info["relative_east"]))
    ys.append(float(info["relative_north"]))
    zs.append(float(info["altitude"]))
    rolls.append(float(info["roll"]))
    pitches.append(float(info["pitch"]))
    headings.append(float(info["heading"]))
    vs_values.append(float(info["vertical_speed"]))
    fwd_values.append(float(info["forward_velocity"]))
    lat_values.append(float(info["lateral_velocity"]))
    dist_values.append(float(info["horizontal_distance"]))
    hold_values.append(int(info["target_hold_steps"]))

    if step % 100 == 0:
        print(
            f"Step {step:4d} | Alt={info['altitude']:.1f} ft | "
            f"Dist={info['horizontal_distance']:.1f} ft | "
            f"VS={info['vertical_speed']:.1f} | Hold={info['target_hold_steps']}"
        )

    if terminated or truncated:
        print("Episode ended:", info.get("termination_reason"))
        break

env.close()

xs = np.asarray(xs)
ys = np.asarray(ys)
zs = np.asarray(zs)
rolls = np.asarray(rolls)
pitches = np.asarray(pitches)
headings = np.asarray(headings)
vs_values = np.asarray(vs_values)
fwd_values = np.asarray(fwd_values)
lat_values = np.asarray(lat_values)
dist_values = np.asarray(dist_values)
hold_values = np.asarray(hold_values)


def rotation_matrix(roll, pitch, yaw):
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    return rz @ ry @ rx


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

xy_limit = max(30.0, float(np.max(np.abs(np.concatenate([xs, ys])))) + 15.0)
ax.set_xlim(-xy_limit, xy_limit)
ax.set_ylim(-xy_limit, xy_limit)
ax.set_zlim(0.0, max(350.0, float(np.max(zs)) + 20.0))
ax.set_xlabel("East displacement from takeoff (ft)")
ax.set_ylabel("North displacement from takeoff (ft)")
ax.set_zlabel("Altitude AGL (ft)")
ax.set_title("AH-1S RL - V15 BEST - TRUE TAKEOFF-RELATIVE POSITION")

# Vertical reference line: the trajectory we actually want.
ax.plot([0, 0], [0, 0], [0, 300], linestyle="--", linewidth=1.5)

trail, = ax.plot([], [], [], linewidth=2)
info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

body_collection = None
nose_line = None
rotor_line = None


def update(frame):
    global body_collection, nose_line, rotor_line

    if body_collection is not None:
        body_collection.remove()
    if nose_line is not None:
        nose_line.remove()
    if rotor_line is not None:
        rotor_line.remove()

    x, y, z = xs[frame], ys[frame], zs[frame]
    roll, pitch, yaw = rolls[frame], pitches[frame], headings[frame]
    r = rotation_matrix(roll, pitch, yaw)

    trail.set_data(xs[:frame + 1], ys[:frame + 1])
    trail.set_3d_properties(zs[:frame + 1])

    length, width, height = 18.0, 5.0, 4.0
    vertices = np.array([
        [-length/2, -width/2, -height/2],
        [ length/2, -width/2, -height/2],
        [ length/2,  width/2, -height/2],
        [-length/2,  width/2, -height/2],
        [-length/2, -width/2,  height/2],
        [ length/2, -width/2,  height/2],
        [ length/2,  width/2,  height/2],
        [-length/2,  width/2,  height/2],
    ])
    vertices = vertices @ r.T
    vertices[:, 0] += x
    vertices[:, 1] += y
    vertices[:, 2] += z

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
    ]
    body_collection = Poly3DCollection(faces, alpha=0.8)
    ax.add_collection3d(body_collection)

    nose = r @ np.array([15.0, 0.0, 0.0])
    nose_line, = ax.plot(
        [x, x + nose[0]],
        [y, y + nose[1]],
        [z, z + nose[2]],
        linewidth=3,
    )

    rotor_a = r @ np.array([0.0, -14.0, 2.5])
    rotor_b = r @ np.array([0.0, 14.0, 2.5])
    rotor_line, = ax.plot(
        [x + rotor_a[0], x + rotor_b[0]],
        [y + rotor_a[1], y + rotor_b[1]],
        [z + rotor_a[2], z + rotor_b[2]],
        linewidth=2,
    )

    info_text.set_text(
        f"Step: {frame}\n"
        f"Altitude: {z:.1f} ft\n"
        f"Distance from takeoff: {dist_values[frame]:.1f} ft\n"
        f"VS: {vs_values[frame]:.1f} ft/s\n"
        f"Forward: {fwd_values[frame]:.1f} ft/s\n"
        f"Lateral: {lat_values[frame]:.1f} ft/s\n"
        f"Hold: {hold_values[frame]}/100"
    )

    return trail, body_collection, nose_line, rotor_line, info_text


frames = range(0, len(xs), 3)
animation = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
animation.save(OUTPUT_GIF, writer=PillowWriter(fps=20))
plt.close(fig)

print("GIF created:", OUTPUT_GIF)
