import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
from helicopter_env_1 import HelicopterEnv


# =========================================================
# HELIKOPTER GÖVDESI
# =========================================================

def create_helicopter():

    # Basit fakat yönü anlaşılır bir gövde.
    # X ekseni burun-kuyruk yönü.

    vertices = np.array([
        [-3.0, -1.5, -0.8],
        [ 3.0, -1.5, -0.8],
        [ 3.0,  1.5, -0.8],
        [-3.0,  1.5, -0.8],

        [-3.0, -1.5,  0.8],
        [ 3.0, -1.5,  0.8],
        [ 3.0,  1.5,  0.8],
        [-3.0,  1.5,  0.8]
    ])

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),

        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),

        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]

    return vertices, edges


# =========================================================
# ROTATION MATRIX
# =========================================================

def rotation_matrix(roll, pitch, yaw):

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    roll_matrix = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    pitch_matrix = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    yaw_matrix = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    return (
        yaw_matrix
        @ pitch_matrix
        @ roll_matrix
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("Visualization baslatiliyor...")

    # -----------------------------------------------------
    # ENVIRONMENT
    # -----------------------------------------------------

    env = HelicopterEnv()

    print("Environment olusturuldu.")

    # -----------------------------------------------------
    # TAKEOFF-ONLY PPO MODEL
    # -----------------------------------------------------

    model = PPO.load(
        "ppo_ah1s_takeoff_only",
        env=env
    )

    print("PPO model yuklendi.")

    obs, info = env.reset()

    print("Environment reset tamamlandi.")

    # -----------------------------------------------------
    # VERI LISTELERI
    # -----------------------------------------------------

    positions = []

    rolls = []
    pitches = []
    headings = []

    vertical_speeds = []
    forward_velocities = []

    collectives = []

    hold_steps = []

    # -----------------------------------------------------
    # BASLANGIC KONUMU
    # -----------------------------------------------------

    x = 0.0
    y = 0.0

    # 10 JSBSim physics step:
    #
    # 10 * 0.0075 = 0.075 saniye

    dt = 0.075

    # =====================================================
    # SIMULASYON
    # =====================================================

    for step in range(600):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        # ---------------------------------------------
        # BODY VELOCITIES
        # ---------------------------------------------

        u = info[
            "forward_velocity"
        ]

        v = float(
            env.fdm[
                "velocities/v-aero-fps"
            ]
        )

        heading = info[
            "heading"
        ]

        # ---------------------------------------------
        # APPROXIMATE XY POSITION
        # ---------------------------------------------

        dx = (
            u * np.cos(heading)
            - v * np.sin(heading)
        ) * dt

        dy = (
            u * np.sin(heading)
            + v * np.cos(heading)
        ) * dt

        x += dx
        y += dy

        altitude = info[
            "altitude"
        ]

        # ---------------------------------------------
        # SAVE DATA
        # ---------------------------------------------

        positions.append(
            [
                x,
                y,
                altitude
            ]
        )

        rolls.append(
            info["roll"]
        )

        pitches.append(
            info["pitch"]
        )

        headings.append(
            info["heading"]
        )

        vertical_speeds.append(
            info["vertical_speed"]
        )

        forward_velocities.append(
            info["forward_velocity"]
        )

        collectives.append(
            info["collective"]
        )

        hold_steps.append(
            info["target_hold_steps"]
        )

        if terminated or truncated:

            print(
                "Episode bitti."
            )

            print(
                "Success:",
                info["success"]
            )

            print(
                "Final altitude:",
                info["altitude"]
            )

            print(
                "Final hold:",
                info["target_hold_steps"]
            )

            break

    env.close()

    # -----------------------------------------------------
    # NUMPY
    # -----------------------------------------------------

    positions = np.array(
        positions
    )

    # =====================================================
    # FIGURE
    # =====================================================

    vertices, edges = create_helicopter()

    fig = plt.figure(
        figsize=(12, 8)
    )

    # Grafiğe sağ tarafta bilgi paneli için yer bırak.
    fig.subplots_adjust(
        left=0.05,
        right=0.72,
        bottom=0.08,
        top=0.90
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # =====================================================
    # AXIS LIMITS
    # =====================================================

    x_min = positions[:, 0].min()
    x_max = positions[:, 0].max()

    y_min = positions[:, 1].min()
    y_max = positions[:, 1].max()

    # Çok dar eksen oluşmasını engelle.

    x_range = max(
        x_max - x_min,
        40.0
    )

    y_range = max(
        y_max - y_min,
        40.0
    )

    x_center = (
        x_min + x_max
    ) / 2.0

    y_center = (
        y_min + y_max
    ) / 2.0

    ax.set_xlim(
        x_center - x_range / 2,
        x_center + x_range / 2
    )

    ax.set_ylim(
        y_center - y_range / 2,
        y_center + y_range / 2
    )

    # Takeoff-only görev:
    # hedef 20 ft.
    # Görselde 0-40 ft yeterli.

    z_max = max(
        40.0,
        positions[:, 2].max() + 5.0
    )

    ax.set_zlim(
        0,
        z_max
    )

    # =====================================================
    # LABELS
    # =====================================================

    ax.set_xlabel(
        "X - Forward (ft)",
        labelpad=10
    )

    ax.set_ylabel(
        "Y - Lateral (ft)",
        labelpad=10
    )

    ax.set_zlabel(
        "Altitude AGL (ft)",
        labelpad=10
    )

    ax.set_title(
        "AH-1S Takeoff-Only Flight Visualization",
        pad=20
    )

    # =====================================================
    # TARGET ALTITUDE PLANE / LINE
    # =====================================================

    target_altitude = 20.0

    target_line, = ax.plot(
        [
            x_center - x_range / 2,
            x_center + x_range / 2
        ],
        [
            y_center,
            y_center
        ],
        [
            target_altitude,
            target_altitude
        ],
        linestyle="--",
        linewidth=2,
        label="20 ft target"
    )

    ax.legend(
        loc="lower left"
    )

    # =====================================================
    # FLIGHT TRAIL
    # =====================================================

    trail, = ax.plot(
        [],
        [],
        [],
        linewidth=2
    )

    # =====================================================
    # HELICOPTER BODY
    # =====================================================

    body_lines = []

    for _ in edges:

        line, = ax.plot(
            [],
            [],
            [],
            linewidth=3
        )

        body_lines.append(
            line
        )

    # =====================================================
    # NOSE DIRECTION LINE
    # =====================================================

    nose_line, = ax.plot(
        [],
        [],
        [],
        linewidth=4
    )

    # =====================================================
    # ROTOR LINE
    # =====================================================

    rotor_line, = ax.plot(
        [],
        [],
        [],
        linewidth=3
    )

    # =====================================================
    # INFO PANEL
    # =====================================================

    # Grafiğin üstüne değil, FIGURE'ın sağ tarafına yazıyoruz.
    # Böylece title ile çakışmayacak.

    info_text = fig.text(
        0.75,
        0.82,
        "",
        fontsize=11,
        verticalalignment="top",
        family="monospace"
    )

    explanation_text = fig.text(
        0.75,
        0.34,

        "TARGET\n"
        "------\n"
        "Altitude : 20 ft\n"
        "VSpeed   : ~0 at target\n"
        "Roll     : near 0\n"
        "Pitch    : near 0\n"
        "X/Y drift: low\n"
        "Hold     : 50\n",

        fontsize=10,
        verticalalignment="top",
        family="monospace"
    )

    # =====================================================
    # UPDATE
    # =====================================================

    def update(frame):

        position = positions[
            frame
        ]

        roll = rolls[
            frame
        ]

        pitch = pitches[
            frame
        ]

        yaw = headings[
            frame
        ]

        rotation = rotation_matrix(
            roll,
            pitch,
            yaw
        )

        # -------------------------------------------------
        # BODY ROTATION
        # -------------------------------------------------

        rotated_vertices = (
            vertices
            @ rotation.T
        )

        # MODEL BÜYÜKLÜĞÜ
        #
        # Burayı özellikle büyük tutuyoruz.
        #
        # Eski 5.0 yerine 25.0.

        body_scale = 25.0

        rotated_vertices *= (
            body_scale
        )

        rotated_vertices[:, 0] += (
            position[0]
        )

        rotated_vertices[:, 1] += (
            position[1]
        )

        rotated_vertices[:, 2] += (
            position[2]
        )

        # -------------------------------------------------
        # BODY EDGES
        # -------------------------------------------------

        for line, edge in zip(
            body_lines,
            edges
        ):

            p1 = rotated_vertices[
                edge[0]
            ]

            p2 = rotated_vertices[
                edge[1]
            ]

            line.set_data(
                [
                    p1[0],
                    p2[0]
                ],
                [
                    p1[1],
                    p2[1]
                ]
            )

            line.set_3d_properties(
                [
                    p1[2],
                    p2[2]
                ]
            )

        # -------------------------------------------------
        # NOSE INDICATOR
        # -------------------------------------------------

        # Gövdenin +X yönünü burun kabul ediyoruz.

        nose_local = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0]
        ])

        nose_world = (
            nose_local
            @ rotation.T
        )

        nose_world *= (
            body_scale
        )

        nose_world[:, 0] += (
            position[0]
        )

        nose_world[:, 1] += (
            position[1]
        )

        nose_world[:, 2] += (
            position[2]
        )

        nose_line.set_data(
            nose_world[:, 0],
            nose_world[:, 1]
        )

        nose_line.set_3d_properties(
            nose_world[:, 2]
        )

        # -------------------------------------------------
        # MAIN ROTOR
        # -------------------------------------------------

        rotor_local = np.array([
            [0.0, -5.0, 1.0],
            [0.0,  5.0, 1.0]
        ])

        rotor_world = (
            rotor_local
            @ rotation.T
        )

        rotor_world *= (
            body_scale
        )

        rotor_world[:, 0] += (
            position[0]
        )

        rotor_world[:, 1] += (
            position[1]
        )

        rotor_world[:, 2] += (
            position[2]
        )

        rotor_line.set_data(
            rotor_world[:, 0],
            rotor_world[:, 1]
        )

        rotor_line.set_3d_properties(
            rotor_world[:, 2]
        )

        # -------------------------------------------------
        # TRAIL
        # -------------------------------------------------

        trail.set_data(
            positions[
                :frame + 1,
                0
            ],
            positions[
                :frame + 1,
                1
            ]
        )

        trail.set_3d_properties(
            positions[
                :frame + 1,
                2
            ]
        )

        # -------------------------------------------------
        # INFO
        # -------------------------------------------------

        info_text.set_text(

            "FLIGHT DATA\n"
            "===========\n\n"

            f"Frame     : {frame:4d}\n\n"

            f"X         : {position[0]:8.2f} ft\n"
            f"Y         : {position[1]:8.2f} ft\n"
            f"Altitude  : {position[2]:8.2f} ft\n\n"

            f"Roll      : {roll:8.3f} rad\n"
            f"Pitch     : {pitch:8.3f} rad\n"
            f"Heading   : {yaw:8.3f} rad\n\n"

            f"VSpeed    : {vertical_speeds[frame]:8.2f} ft/s\n"
            f"Forward V : {forward_velocities[frame]:8.2f} ft/s\n\n"

            f"Collective: {collectives[frame]:8.3f}\n"
            f"Hold      : {hold_steps[frame]:4d} / 50"
        )

        return (
            body_lines
            + [
                trail,
                nose_line,
                rotor_line,
                info_text
            ]
        )

    # =====================================================
    # ANIMATION
    # =====================================================

    # Her fizik/RL frame'ini GIF'e koymak çok yavaş.
    # 5 step'te bir frame alıyoruz.

    frame_indices = range(
        0,
        len(positions),
        5
    )

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=100,
        blit=False
    )

    # =====================================================
    # SAVE
    # =====================================================

    print(
        "GIF olusturuluyor..."
    )

    animation.save(
        "takeoff_visualization.gif",
        writer=PillowWriter(
            fps=10
        ),
        dpi=100
    )

    plt.close(
        fig
    )

    print(
        "takeoff_visualization.gif olusturuldu."
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()
