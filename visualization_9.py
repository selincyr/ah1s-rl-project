import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stable_baselines3 import PPO
from helicopter_env_9 import HelicopterEnv


MODEL_PATH = "models/task9_hover300/best/best_model"


def cuboid_vertices(center, size, yaw):

    cx, cy, cz = center

    length, width, height = size

    x = length / 2.0
    y = width / 2.0
    z = height / 2.0

    points = np.array(
        [
            [-x, -y, -z],
            [ x, -y, -z],
            [ x,  y, -z],
            [-x,  y, -z],

            [-x, -y,  z],
            [ x, -y,  z],
            [ x,  y,  z],
            [-x,  y,  z],
        ]
    )

    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0,          0.0,         1.0]
        ]
    )

    rotated = points @ rotation.T

    rotated[:, 0] += cx
    rotated[:, 1] += cy
    rotated[:, 2] += cz

    return rotated


def cuboid_faces(vertices):

    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
    ]


env = HelicopterEnv()

model = PPO.load(
    MODEL_PATH
)

obs, info = env.reset()


x = 0.0
y = 0.0

dt = 0.075


x_history = []
y_history = []
z_history = []

heading_history = []
roll_history = []
pitch_history = []

forward_history = []
lateral_history = []

phase_history = []


for step in range(5000):

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    altitude = info["altitude"]

    forward_velocity = info["forward_velocity"]

    lateral_velocity = info["lateral_velocity"]

    heading_error = info["heading_error"]

    roll = info["roll"]

    pitch = info["pitch"]

    heading = -heading_error


    dx_body = forward_velocity * dt

    dy_body = lateral_velocity * dt


    dx_world = (
        dx_body * np.cos(heading)
        -
        dy_body * np.sin(heading)
    )

    dy_world = (
        dx_body * np.sin(heading)
        +
        dy_body * np.cos(heading)
    )


    x += dx_world

    y += dy_world


    x_history.append(
        x
    )

    y_history.append(
        y
    )

    z_history.append(
        altitude
    )

    heading_history.append(
        heading
    )

    roll_history.append(
        roll
    )

    pitch_history.append(
        pitch
    )

    forward_history.append(
        forward_velocity
    )

    lateral_history.append(
        lateral_velocity
    )

    phase_history.append(
        info["phase"]
    )


    if terminated or truncated:

        print(
            "Episode ended:",
            info.get(
                "termination_reason"
            )
        )

        print(
            "Step:",
            step + 1
        )

        print(
            "Altitude:",
            altitude
        )

        break


env.close()


x_history = np.array(
    x_history
)

y_history = np.array(
    y_history
)

z_history = np.array(
    z_history
)


fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


max_horizontal = max(
    np.max(
        np.abs(
            x_history
        )
    ),
    np.max(
        np.abs(
            y_history
        )
    ),
    50.0
)


ax.set_xlim(
    -max_horizontal,
    max_horizontal
)

ax.set_ylim(
    -max_horizontal,
    max_horizontal
)

ax.set_zlim(
    0,
    max(
        350,
        np.max(
            z_history
        ) + 20
    )
)


ax.set_xlabel(
    "X Forward (ft)"
)

ax.set_ylabel(
    "Y Lateral (ft)"
)

ax.set_zlabel(
    "Altitude AGL (ft)"
)

ax.set_title(
    "AH-1S RL - V9 BEST"
)


trail, = ax.plot(
    [],
    [],
    [],
    linewidth=2
)


body_collection = Poly3DCollection(
    [],
    alpha=0.8
)

ax.add_collection3d(
    body_collection
)


nose_line, = ax.plot(
    [],
    [],
    [],
    linewidth=3
)


rotor_line, = ax.plot(
    [],
    [],
    [],
    linewidth=2
)


status_text = ax.text2D(
    0.02,
    0.95,
    "",
    transform=ax.transAxes
)


def update(frame):

    current_x = x_history[frame]

    current_y = y_history[frame]

    current_z = z_history[frame]

    heading = heading_history[frame]


    trail.set_data(
        x_history[:frame + 1],
        y_history[:frame + 1]
    )

    trail.set_3d_properties(
        z_history[:frame + 1]
    )


    helicopter_size = (
        16.0,
        5.0,
        4.0
    )


    vertices = cuboid_vertices(
        (
            current_x,
            current_y,
            current_z
        ),
        helicopter_size,
        heading
    )

    faces = cuboid_faces(
        vertices
    )

    body_collection.set_verts(
        faces
    )


    nose_length = 12.0

    nose_x = (
        current_x
        +
        nose_length
        *
        np.cos(
            heading
        )
    )

    nose_y = (
        current_y
        +
        nose_length
        *
        np.sin(
            heading
        )
    )


    nose_line.set_data(
        [
            current_x,
            nose_x
        ],
        [
            current_y,
            nose_y
        ]
    )

    nose_line.set_3d_properties(
        [
            current_z,
            current_z
        ]
    )


    rotor_radius = 10.0

    rotor_dx = (
        rotor_radius
        *
        np.cos(
            heading
            +
            np.pi / 2.0
        )
    )

    rotor_dy = (
        rotor_radius
        *
        np.sin(
            heading
            +
            np.pi / 2.0
        )
    )


    rotor_line.set_data(
        [
            current_x - rotor_dx,
            current_x + rotor_dx
        ],
        [
            current_y - rotor_dy,
            current_y + rotor_dy
        ]
    )

    rotor_line.set_3d_properties(
        [
            current_z + 2.5,
            current_z + 2.5
        ]
    )


    status_text.set_text(
        f"Step: {frame}\n"
        f"Phase: {phase_history[frame]}\n"
        f"Altitude: {current_z:.1f} ft\n"
        f"Forward: {forward_history[frame]:.1f} ft/s\n"
        f"Lateral: {lateral_history[frame]:.1f} ft/s\n"
        f"Roll: {roll_history[frame]:.3f} rad\n"
        f"Pitch: {pitch_history[frame]:.3f} rad"
    )


    return (
        trail,
        body_collection,
        nose_line,
        rotor_line,
        status_text
    )


animation = FuncAnimation(
    fig,
    update,
    frames=len(
        x_history
    ),
    interval=40,
    blit=False
)


writer = PillowWriter(
    fps=25
)


output_file = "ah1s_v9_best.gif"


animation.save(
    output_file,
    writer=writer
)


print(
    "Visualization saved:",
    output_file
)


plt.close(
    fig
)
