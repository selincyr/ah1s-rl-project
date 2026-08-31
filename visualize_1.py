#####%%writefile /content/ah1s-rl-project/visualize_1.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
from helicopter_env_1 import HelicopterEnv

def create_box():
    vertices = np.array([
        [-2, -1, -0.5],
        [2, -1, -0.5],
        [2, 1, -0.5],
        [-2, 1, -0.5],
        [-2, -1, 0.5],
        [2, -1, 0.5],
        [2, 1, 0.5],
        [-2, 1, 0.5]
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return vertices, edges

def rotation_matrix(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])
    return rz @ ry @ rx

def main():
    print("mainnnnnnnnnnnnnn")
    env = HelicopterEnv()
    print("env oluştu")
    model = PPO.load(
        "ppo_ah1s_task1_rates", env=env
    )
    print( "model yüklendi")
    obs, info = env.reset()
    print("reset oldu")

    positions = []
    rolls = []
    pitches = []
    headings = []

    x = 0.0
    y = 0.0
    dt = 0.075

    for step in range(1200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        u = info["forward_velocity"]
        v = float(env.fdm["velocities/v-aero-fps"])
        heading = info["heading"]

        dx = (u * np.cos(heading) - v * np.sin(heading)) * dt
        dy = (u * np.sin(heading) + v * np.cos(heading)) * dt
        x += dx
        y += dy
        z = info["altitude"]

        positions.append((x, y, z))
        rolls.append(info["roll"])
        pitches.append(info["pitch"])
        headings.append(info["heading"])

        if terminated or truncated:
            break

    env.close()

    positions = np.array(positions)
    vertices, edges = create_box()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_min = positions[:, 0].min()
    x_max = positions[:, 0].max()
    y_min = positions[:, 1].min()
    y_max = positions[:, 1].max()
    z_max = max(1000, positions[:, 2].max())

    ax.set_xlim(x_min - 20, x_max + 20)
    ax.set_ylim(y_min - 20, y_max + 20)
    ax.set_zlim(0, z_max + 50)

    ax.set_xlabel("X - Forward (ft)")
    ax.set_ylabel("Y - Lateral (ft)")
    ax.set_zlabel("Altitude AGL (ft)")
    ax.set_title("AH-1S RL Flight Visualization")
    info_text = ax.text2D(0.02, 0.70, "", transform=ax.transAxes)

    ax.plot(
        [x_min - 20, x_max + 20], [0, 0], [1000, 1000], linestyle="--"
    )

    trail, = ax.plot([], [], [])
    box_lines = []
    for _ in edges:
        line, = ax.plot([], [], [])
        box_lines.append(line)

    def update(frame):
        position = positions[frame]
        roll = rolls[frame]
        pitch = pitches[frame]
        yaw = headings[frame]

        rotation = rotation_matrix(roll, pitch, yaw)
        rotated_vertices = (vertices @ rotation.T)
        scale = 40.0
        rotated_vertices *= scale
        rotated_vertices[:, 0] += position[0]
        rotated_vertices[:, 1] += position[1]
        rotated_vertices[:, 2] += position[2]

        for line, edge in zip(box_lines, edges):
            p1 = rotated_vertices[edge[0]]
            p2 = rotated_vertices[edge[1]]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])

        trail.set_data(
            positions[:frame + 1, 0],
            positions[:frame + 1, 1]
        )
        trail.set_3d_properties(
            positions[:frame + 1, 2]
        )

        info_text.set_text(
            f"X: {position[0]:.1f} ft\n"
            f"Y: {position[1]:.1f} ft\n"
            f"Altitude: {position[2]:.1f} ft\n\n"
            f"Roll: {roll:.3f} rad\n"
            f"Pitch: {pitch:.3f} rad\n"
            f"Heading: {yaw:.3f} rad"
        )
        return box_lines + [trail, info_text]

    animation = FuncAnimation(fig, update, frames=len(positions), interval=40, blit=False)

    print("Görsel oluşturuluyor")
    animation.save("task1_flight.gif", writer=PillowWriter(fps=20))
    print("task_flight.gif oluşturuldu")
    plt.close()

if __name__ == "__main__":
    main()
