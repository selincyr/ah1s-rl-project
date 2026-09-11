import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stable_baselines3 import PPO
from helicopter_env_12 import HelicopterEnv


# ==============================================================
# SETTINGS
# ==============================================================

MODEL_PATH = "models/task12_hover300/best/best_model"

OUTPUT_GIF = "ah1s_v12_best.gif"

DT = 0.075

MAX_STEPS = 5000


# ==============================================================
# LOAD ENVIRONMENT + MODEL
# ==============================================================

env = HelicopterEnv()

model = PPO.load(
    MODEL_PATH
)

obs, info = env.reset()


# ==============================================================
# STORAGE
# ==============================================================

x_positions = [0.0]
y_positions = [0.0]
z_positions = [float(info["altitude"])]

roll_values = [float(info["roll"])]
pitch_values = [float(info["pitch"])]
heading_values = [float(info["heading"])]

forward_values = [float(info["forward_velocity"])]
lateral_values = [float(info["lateral_velocity"])]
vertical_values = [float(info["vertical_speed"])]

hold_values = [int(info["target_hold_steps"])]


# ==============================================================
# RUN MODEL
# ==============================================================

print()
print("=" * 80)
print("RUNNING V12 BEST MODEL")
print("=" * 80)
print()


for step in range(MAX_STEPS):

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    altitude = float(
        info["altitude"]
    )

    forward_velocity = float(
        info["forward_velocity"]
    )

    lateral_velocity = float(
        info["lateral_velocity"]
    )

    vertical_speed = float(
        info["vertical_speed"]
    )

    heading = float(
        info["heading"]
    )

    roll = float(
        info["roll"]
    )

    pitch = float(
        info["pitch"]
    )

    hold = int(
        info["target_hold_steps"]
    )

    # ==========================================================
    # APPROXIMATE WORLD POSITION
    # ==========================================================

    previous_x = x_positions[-1]
    previous_y = y_positions[-1]

    vx_world = (
        forward_velocity * np.cos(heading)
        -
        lateral_velocity * np.sin(heading)
    )

    vy_world = (
        forward_velocity * np.sin(heading)
        +
        lateral_velocity * np.cos(heading)
    )

    new_x = previous_x + vx_world * DT
    new_y = previous_y + vy_world * DT

    x_positions.append(
        new_x
    )

    y_positions.append(
        new_y
    )

    z_positions.append(
        altitude
    )

    roll_values.append(
        roll
    )

    pitch_values.append(
        pitch
    )

    heading_values.append(
        heading
    )

    forward_values.append(
        forward_velocity
    )

    lateral_values.append(
        lateral_velocity
    )

    vertical_values.append(
        vertical_speed
    )

    hold_values.append(
        hold
    )

    if step % 100 == 0:

        print(
            f"Step {step:4d} | "
            f"Alt {altitude:7.2f} ft | "
            f"VS {vertical_speed:6.2f} | "
            f"Fwd {forward_velocity:6.2f} | "
            f"Lat {lateral_velocity:6.2f} | "
            f"Hold {hold:3d}"
        )

    if terminated or truncated:

        print()
        print(
            "Episode ended at step:",
            step
        )

        print(
            "Reason:",
            info.get("termination_reason")
        )

        break


env.close()


# ==============================================================
# CONVERT TO NUMPY
# ==============================================================

x_positions = np.array(
    x_positions
)

y_positions = np.array(
    y_positions
)

z_positions = np.array(
    z_positions
)

roll_values = np.array(
    roll_values
)

pitch_values = np.array(
    pitch_values
)

heading_values = np.array(
    heading_values
)

forward_values = np.array(
    forward_values
)

lateral_values = np.array(
    lateral_values
)

vertical_values = np.array(
    vertical_values
)

hold_values = np.array(
    hold_values
)


# ==============================================================
# HELICOPTER BODY
# ==============================================================

def create_helicopter_vertices(
    x,
    y,
    z,
    roll,
    pitch,
    yaw
):

    # Simple AH-1-like body box

    length = 18.0
    width = 5.0
    height = 4.0

    vertices = np.array([
        [-length / 2, -width / 2, -height / 2],
        [ length / 2, -width / 2, -height / 2],
        [ length / 2,  width / 2, -height / 2],
        [-length / 2,  width / 2, -height / 2],

        [-length / 2, -width / 2,  height / 2],
        [ length / 2, -width / 2,  height / 2],
        [ length / 2,  width / 2,  height / 2],
        [-length / 2,  width / 2,  height / 2]
    ])

    # Rotation matrices

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    rotation = (
        Rz @ Ry @ Rx
    )

    vertices = (
        vertices @ rotation.T
    )

    vertices[:, 0] += x
    vertices[:, 1] += y
    vertices[:, 2] += z

    return vertices, rotation


# ==============================================================
# FIGURE
# ==============================================================

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ==============================================================
# AXIS LIMITS
# ==============================================================

margin_xy = 30.0

x_min = np.min(x_positions) - margin_xy
x_max = np.max(x_positions) + margin_xy

y_min = np.min(y_positions) - margin_xy
y_max = np.max(y_positions) + margin_xy

z_min = 0.0

z_max = max(
    350.0,
    np.max(z_positions) + 20.0
)


ax.set_xlim(
    x_min,
    x_max
)

ax.set_ylim(
    y_min,
    y_max
)

ax.set_zlim(
    z_min,
    z_max
)


ax.set_xlabel(
    "X Position (ft)"
)

ax.set_ylabel(
    "Y Position (ft)"
)

ax.set_zlabel(
    "Altitude AGL (ft)"
)

ax.set_title(
    "AH-1S RL - V12 BEST"
)


# ==============================================================
# 300 FT TARGET PLANE
# ==============================================================

plane_x = np.array([
    [x_min, x_max],
    [x_min, x_max]
])

plane_y = np.array([
    [y_min, y_min],
    [y_max, y_max]
])

plane_z = np.full(
    (2, 2),
    300.0
)

ax.plot_surface(
    plane_x,
    plane_y,
    plane_z,
    alpha=0.12
)


# ==============================================================
# TRAJECTORY
# ==============================================================

trail, = ax.plot(
    [],
    [],
    [],
    linewidth=2
)


# ==============================================================
# TEXT
# ==============================================================

info_text = ax.text2D(
    0.02,
    0.95,
    "",
    transform=ax.transAxes
)


# ==============================================================
# ANIMATION
# ==============================================================

body_collection = None

nose_line = None

rotor_line = None


def update(frame):

    global body_collection
    global nose_line
    global rotor_line

    x = x_positions[frame]
    y = y_positions[frame]
    z = z_positions[frame]

    roll = roll_values[frame]
    pitch = pitch_values[frame]
    yaw = heading_values[frame]

    # ==========================================================
    # TRAIL
    # ==========================================================

    trail.set_data(
        x_positions[:frame + 1],
        y_positions[:frame + 1]
    )

    trail.set_3d_properties(
        z_positions[:frame + 1]
    )

    # ==========================================================
    # REMOVE OLD HELICOPTER
    # ==========================================================

    if body_collection is not None:

        body_collection.remove()

    if nose_line is not None:

        nose_line.remove()

    if rotor_line is not None:

        rotor_line.remove()

    # ==========================================================
    # BODY
    # ==========================================================

    vertices, rotation = (
        create_helicopter_vertices(
            x,
            y,
            z,
            roll,
            pitch,
            yaw
        )
    )

    faces = [
        [
            vertices[0],
            vertices[1],
            vertices[2],
            vertices[3]
        ],
        [
            vertices[4],
            vertices[5],
            vertices[6],
            vertices[7]
        ],
        [
            vertices[0],
            vertices[1],
            vertices[5],
            vertices[4]
        ],
        [
            vertices[2],
            vertices[3],
            vertices[7],
            vertices[6]
        ],
        [
            vertices[1],
            vertices[2],
            vertices[6],
            vertices[5]
        ],
        [
            vertices[0],
            vertices[3],
            vertices[7],
            vertices[4]
        ]
    ]

    body_collection = Poly3DCollection(
        faces,
        alpha=0.8
    )

    ax.add_collection3d(
        body_collection
    )

    # ==========================================================
    # NOSE DIRECTION
    # ==========================================================

    nose_local = np.array([
        15.0,
        0.0,
        0.0
    ])

    nose_world = (
        rotation @ nose_local
    )

    nose_line, = ax.plot(
        [
            x,
            x + nose_world[0]
        ],
        [
            y,
            y + nose_world[1]
        ],
        [
            z,
            z + nose_world[2]
        ],
        linewidth=3
    )

    # ==========================================================
    # MAIN ROTOR
    # ==========================================================

    rotor_radius = 14.0

    rotor_local_1 = np.array([
        0.0,
        -rotor_radius,
        2.5
    ])

    rotor_local_2 = np.array([
        0.0,
        rotor_radius,
        2.5
    ])

    rotor_world_1 = (
        rotation @ rotor_local_1
    )

    rotor_world_2 = (
        rotation @ rotor_local_2
    )

    rotor_line, = ax.plot(
        [
            x + rotor_world_1[0],
            x + rotor_world_2[0]
        ],
        [
            y + rotor_world_1[1],
            y + rotor_world_2[1]
        ],
        [
            z + rotor_world_1[2],
            z + rotor_world_2[2]
        ],
        linewidth=2
    )

    # ==========================================================
    # INFO
    # ==========================================================

    info_text.set_text(
        f"Step: {frame}\n"
        f"Altitude: {z:.1f} ft\n"
        f"Vertical Speed: {vertical_values[frame]:.1f} ft/s\n"
        f"Forward: {forward_values[frame]:.1f} ft/s\n"
        f"Lateral: {lateral_values[frame]:.1f} ft/s\n"
        f"Roll: {roll:.3f} rad\n"
        f"Pitch: {pitch:.3f} rad\n"
        f"Heading: {yaw:.3f} rad\n"
        f"Hold: {hold_values[frame]}/100"
    )

    return (
        trail,
        body_collection,
        nose_line,
        rotor_line,
        info_text
    )


# ==============================================================
# FRAME SKIP
# ==============================================================

frame_skip = 3

frames = range(
    0,
    len(x_positions),
    frame_skip
)


animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=40,
    blit=False
)


# ==============================================================
# SAVE GIF
# ==============================================================

print()
print("=" * 80)
print("CREATING GIF")
print("=" * 80)
print()

writer = PillowWriter(
    fps=20
)

animation.save(
    OUTPUT_GIF,
    writer=writer
)

plt.close(
    fig
)

print()
print(
    "GIF created:",
    OUTPUT_GIF
)
