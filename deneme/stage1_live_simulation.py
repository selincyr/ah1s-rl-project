import os
import shutil
import numpy as np

from stable_baselines3 import PPO
from helicopter_env_stage1_distill import HelicopterEnvStage1Distill

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    "models_stage1_final_distilled/"
    "AH1S_STAGE1_FINAL_DISTILLED.zip"
)

TOTAL_TIME = 120.0

# Record every physics control step, but animate a subset.
# env.dt is ~0.075 s, so 4 => ~0.30 s/frame.
SAMPLE_EVERY = 4

OUTPUT_DIR = "results_stage1_final"
OUTPUT_HTML = os.path.join(
    OUTPUT_DIR,
    "stage1_live_simulation.html"
)

# Optional GitHub Pages copy.
PAGES_DIR = "docs/stage1"
PAGES_HTML = os.path.join(
    PAGES_DIR,
    "index.html"
)

TARGET_ALT = 300.0
HOVER_LOW = 295.0
HOVER_HIGH = 305.0


# ============================================================
# HELPERS
# ============================================================

def get_info(info, key, default=0.0):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def phase_name(altitude, vertical_speed):
    if altitude < 285.0:
        return "CLIMB"

    if altitude < HOVER_LOW:
        return "TRANSITION"

    if HOVER_LOW <= altitude <= HOVER_HIGH and abs(vertical_speed) <= 0.75:
        return "HOVER"

    return "STABILIZING"


def telemetry_text(i, data):
    return (
        "<b>LIVE TELEMETRY</b><br><br>"
        f"<b>Time</b> ............ {data['time'][i]:7.2f} s<br>"
        f"<b>Phase</b> ........... {data['phase'][i]}<br><br>"
        f"<b>Altitude</b> ........ {data['altitude'][i]:7.2f} ft<br>"
        f"<b>Vertical speed</b> .. {data['vs'][i]:7.3f} ft/s<br>"
        f"<b>Altitude error</b> .. {abs(TARGET_ALT-data['altitude'][i]):7.2f} ft<br><br>"
        f"<b>North</b> ........... {data['north'][i]:7.2f} ft<br>"
        f"<b>East</b> ............ {data['east'][i]:7.2f} ft<br>"
        f"<b>Drift</b> ........... {data['drift'][i]:7.2f} ft<br>"
        f"<b>Max drift</b> ....... {data['max_drift'][i]:7.2f} ft<br>"
        f"<b>XY path</b> ......... {data['path'][i]:7.2f} ft<br><br>"
        f"<b>Collective</b> ...... {data['collective'][i]:7.5f}<br>"
        f"<b>PPO a0</b> .......... {data['a0'][i]:+7.4f}<br>"
        f"<b>PPO a1</b> .......... {data['a1'][i]:+7.4f}<br>"
        f"<b>PPO a2</b> .......... {data['a2'][i]:+7.4f}<br>"
        f"<b>PPO a3</b> .......... {data['a3'][i]:+7.4f}<br><br>"
        "<b>Controller state</b><br>"
        "Teacher: OFF<br>"
        "PD controller: OFF<br>"
        "Classical XY: OFF<br>"
        "Runtime bias: OFF<br>"
        "Policy: SINGLE 4-ACTION PPO"
    )


# ============================================================
# RUN THE REAL JSBSIM + PPO FLIGHT
# ============================================================

print("=" * 90)
print("AH-1S STAGE 1 LIVE SIMULATION GENERATOR")
print("=" * 90)

print("\nLoading model:")
print(MODEL_PATH)

model = PPO.load(MODEL_PATH)

env = HelicopterEnvStage1Distill(
    teacher_model_path=None,
    training_mode=False,
)

obs, info = env.reset()

dt = float(env.dt)
max_steps = int(TOTAL_TIME / dt)

print(f"\nControl dt   : {dt:.4f} s")
print(f"Flight time  : {TOTAL_TIME:.1f} s")
print(f"Control steps: {max_steps}")
print("\nRunning JSBSim flight...")

data = {
    "time": [],
    "altitude": [],
    "vs": [],
    "north": [],
    "east": [],
    "drift": [],
    "max_drift": [],
    "path": [],
    "collective": [],
    "a0": [],
    "a1": [],
    "a2": [],
    "a3": [],
    "phase": [],
}

physical_failure = False

for step in range(max_steps):
    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    t = (step + 1) * dt

    altitude = get_info(info, "altitude")
    vs = get_info(info, "vertical_speed")
    north = get_info(info, "north")
    east = get_info(info, "east")
    drift = get_info(info, "drift")
    max_drift = get_info(info, "max_drift")
    path = get_info(info, "path")
    collective = get_info(info, "collective")

    # Record at selected frame rate and always record last frame.
    if (
        step % SAMPLE_EVERY == 0
        or step == max_steps - 1
    ):
        data["time"].append(t)
        data["altitude"].append(altitude)
        data["vs"].append(vs)
        data["north"].append(north)
        data["east"].append(east)
        data["drift"].append(drift)
        data["max_drift"].append(max_drift)
        data["path"].append(path)
        data["collective"].append(collective)

        data["a0"].append(float(action[0]))
        data["a1"].append(float(action[1]))
        data["a2"].append(float(action[2]))
        data["a3"].append(float(action[3]))

        data["phase"].append(
            phase_name(
                altitude,
                vs,
            )
        )

    # Environment's old success termination is intentionally ignored.
    if terminated and not bool(info.get("success", False)):
        physical_failure = True
        print(
            f"\nPhysical failure at t={t:.2f} s"
        )
        break

env.close()

for key in [
    "time",
    "altitude",
    "vs",
    "north",
    "east",
    "drift",
    "max_drift",
    "path",
    "collective",
    "a0",
    "a1",
    "a2",
    "a3",
]:
    data[key] = np.asarray(
        data[key],
        dtype=np.float64,
    )

n_frames = len(data["time"])

if n_frames == 0:
    raise RuntimeError(
        "No flight data was recorded."
    )

print(f"Recorded frames: {n_frames}")

if physical_failure:
    print(
        "WARNING: flight ended with a physical failure."
    )


# ============================================================
# PLOT LIMITS
# ============================================================

east_margin = 2.5
north_margin = 2.5

east_min = min(
    -6.0,
    float(np.min(data["east"])) - east_margin,
)
east_max = max(
    +6.0,
    float(np.max(data["east"])) + east_margin,
)

north_min = min(
    -6.0,
    float(np.min(data["north"])) - north_margin,
)
north_max = max(
    +6.0,
    float(np.max(data["north"])) + north_margin,
)

z_max = max(
    320.0,
    float(np.max(data["altitude"])) + 10.0,
)


# ============================================================
# TARGET HOVER PLANE
# ============================================================

plane_x = np.array([
    [east_min, east_max],
    [east_min, east_max],
])

plane_y = np.array([
    [north_min, north_min],
    [north_max, north_max],
])

plane_z = np.full(
    (2, 2),
    TARGET_ALT,
)


# ============================================================
# FIGURE
# ============================================================

fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.72, 0.28],
    specs=[
        [
            {"type": "scene"},
            {"type": "xy"},
        ]
    ],
    horizontal_spacing=0.02,
)


# 0) Target plane
fig.add_trace(
    go.Surface(
        x=plane_x,
        y=plane_y,
        z=plane_z,
        opacity=0.18,
        showscale=False,
        name="300 ft target",
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)


# 1) Full reference trajectory (faint)
fig.add_trace(
    go.Scatter3d(
        x=data["east"],
        y=data["north"],
        z=data["altitude"],
        mode="lines",
        line=dict(
            width=2,
        ),
        opacity=0.18,
        name="Full Stage 1 path",
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)


# 2) Animated flown tail
fig.add_trace(
    go.Scatter3d(
        x=[data["east"][0]],
        y=[data["north"][0]],
        z=[data["altitude"][0]],
        mode="lines",
        line=dict(
            width=7,
        ),
        name="Flown path",
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)


# 3) Helicopter body
fig.add_trace(
    go.Scatter3d(
        x=[data["east"][0]],
        y=[data["north"][0]],
        z=[data["altitude"][0]],
        mode="markers",
        marker=dict(
            size=8,
            symbol="diamond",
        ),
        name="AH-1S",
        hovertemplate=(
            "AH-1S<br>"
            "E=%{x:.2f} ft<br>"
            "N=%{y:.2f} ft<br>"
            "Alt=%{z:.2f} ft"
            "<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)


# 4) Main rotor line East-West
rotor_half = 1.1

fig.add_trace(
    go.Scatter3d(
        x=[
            data["east"][0] - rotor_half,
            data["east"][0] + rotor_half,
        ],
        y=[
            data["north"][0],
            data["north"][0],
        ],
        z=[
            data["altitude"][0],
            data["altitude"][0],
        ],
        mode="lines",
        line=dict(
            width=8,
        ),
        name="Rotor",
        hoverinfo="skip",
        showlegend=False,
    ),
    row=1,
    col=1,
)


# 5) Main rotor line North-South
fig.add_trace(
    go.Scatter3d(
        x=[
            data["east"][0],
            data["east"][0],
        ],
        y=[
            data["north"][0] - rotor_half,
            data["north"][0] + rotor_half,
        ],
        z=[
            data["altitude"][0],
            data["altitude"][0],
        ],
        mode="lines",
        line=dict(
            width=8,
        ),
        name="Rotor",
        hoverinfo="skip",
        showlegend=False,
    ),
    row=1,
    col=1,
)


# 6) Start point
fig.add_trace(
    go.Scatter3d(
        x=[data["east"][0]],
        y=[data["north"][0]],
        z=[0.0],
        mode="markers+text",
        marker=dict(
            size=5,
        ),
        text=["START"],
        textposition="bottom center",
        name="Start",
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)


# 7) Telemetry text panel
fig.add_trace(
    go.Scatter(
        x=[0.02],
        y=[0.98],
        mode="text",
        text=[telemetry_text(0, data)],
        textposition="top left",
        textfont=dict(
            family="Courier New, monospace",
            size=15,
        ),
        hoverinfo="skip",
        showlegend=False,
    ),
    row=1,
    col=2,
)


# Hide telemetry panel axes.
fig.update_xaxes(
    visible=False,
    range=[0, 1],
    row=1,
    col=2,
)

fig.update_yaxes(
    visible=False,
    range=[0, 1],
    row=1,
    col=2,
)


# ============================================================
# ANIMATION FRAMES
# ============================================================

frames = []

for i in range(n_frames):
    e = float(data["east"][i])
    n = float(data["north"][i])
    z = float(data["altitude"][i])

    frame = go.Frame(
        name=str(i),
        data=[
            # Trace 2: tail
            go.Scatter3d(
                x=data["east"][: i + 1],
                y=data["north"][: i + 1],
                z=data["altitude"][: i + 1],
            ),

            # Trace 3: helicopter body
            go.Scatter3d(
                x=[e],
                y=[n],
                z=[z],
            ),

            # Trace 4: EW rotor
            go.Scatter3d(
                x=[
                    e - rotor_half,
                    e + rotor_half,
                ],
                y=[
                    n,
                    n,
                ],
                z=[
                    z,
                    z,
                ],
            ),

            # Trace 5: NS rotor
            go.Scatter3d(
                x=[
                    e,
                    e,
                ],
                y=[
                    n - rotor_half,
                    n + rotor_half,
                ],
                z=[
                    z,
                    z,
                ],
            ),

            # Trace 7: telemetry panel
            go.Scatter(
                x=[0.02],
                y=[0.98],
                text=[
                    telemetry_text(
                        i,
                        data,
                    )
                ],
            ),
        ],
        traces=[
            2,
            3,
            4,
            5,
            7,
        ],
    )

    frames.append(frame)

fig.frames = frames


# ============================================================
# SLIDER
# ============================================================

slider_steps = []

# Keep slider manageable: label roughly every 5 s.
for i in range(n_frames):
    show_label = (
        i == 0
        or i == n_frames - 1
        or int(data["time"][i]) % 5 == 0
    )

    slider_steps.append(
        {
            "args": [
                [str(i)],
                {
                    "frame": {
                        "duration": 0,
                        "redraw": True,
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 0,
                    },
                },
            ],
            "label": (
                f"{data['time'][i]:.1f}"
                if show_label
                else ""
            ),
            "method": "animate",
        }
    )


# ============================================================
# PLAYBACK BUTTONS
# ============================================================

# One frame corresponds to dt*SAMPLE_EVERY seconds.
frame_seconds = dt * SAMPLE_EVERY

duration_1x_ms = max(
    30,
    int(frame_seconds * 1000.0),
)

duration_4x_ms = max(
    20,
    int(duration_1x_ms / 4),
)


fig.update_layout(
    title=dict(
        text=(
            "AH-1S Stage 1 — Live PPO Flight Simulation"
            "<br><sup>"
            "Vertical takeoff → 300 ft → sustained hover | "
            "Single 4-action PPO | No runtime helper"
            "</sup>"
        ),
        x=0.43,
    ),

    height=760,

    margin=dict(
        l=10,
        r=10,
        t=95,
        b=95,
    ),

    scene=dict(
        xaxis=dict(
            title="East (ft)",
            range=[east_min, east_max],
            showbackground=True,
            gridwidth=1,
        ),
        yaxis=dict(
            title="North (ft)",
            range=[north_min, north_max],
            showbackground=True,
            gridwidth=1,
        ),
        zaxis=dict(
            title="Altitude (ft)",
            range=[0, z_max],
            showbackground=True,
            gridwidth=1,
        ),
        aspectmode="manual",
        aspectratio=dict(
            x=0.55,
            y=0.55,
            z=2.5,
        ),
        camera=dict(
            eye=dict(
                x=1.45,
                y=1.45,
                z=0.95,
            )
        ),
    ),

    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.02,
            "y": -0.06,
            "showactive": False,
            "buttons": [
                {
                    "label": "▶ Play 1×",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": duration_1x_ms,
                                "redraw": True,
                            },
                            "fromcurrent": True,
                            "transition": {
                                "duration": 0,
                            },
                        },
                    ],
                },
                {
                    "label": "▶ Play 4×",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": duration_4x_ms,
                                "redraw": True,
                            },
                            "fromcurrent": True,
                            "transition": {
                                "duration": 0,
                            },
                        },
                    ],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {
                                "duration": 0,
                                "redraw": False,
                            },
                            "mode": "immediate",
                            "transition": {
                                "duration": 0,
                            },
                        },
                    ],
                },
                {
                    "label": "↺ Restart",
                    "method": "animate",
                    "args": [
                        ["0"],
                        {
                            "frame": {
                                "duration": 0,
                                "redraw": True,
                            },
                            "mode": "immediate",
                            "transition": {
                                "duration": 0,
                            },
                        },
                    ],
                },
            ],
        }
    ],

    sliders=[
        {
            "active": 0,
            "x": 0.02,
            "y": -0.13,
            "len": 0.94,
            "currentvalue": {
                "prefix": "Simulation time: ",
                "suffix": " s",
                "font": {
                    "size": 15,
                },
            },
            "steps": slider_steps,
        }
    ],

    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor="rgba(255,255,255,0.75)",
    ),
)


# ============================================================
# WRITE HTML
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

# include_plotlyjs=True => self-contained file.
fig.write_html(
    OUTPUT_HTML,
    include_plotlyjs=True,
    full_html=True,
    auto_play=False,
)

os.makedirs(
    PAGES_DIR,
    exist_ok=True,
)

shutil.copyfile(
    OUTPUT_HTML,
    PAGES_HTML,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("LIVE SIMULATION CREATED")
print("=" * 90)

print("\nFinal flight metrics:")
print(
    f"Final altitude : {data['altitude'][-1]:.3f} ft"
)
print(
    f"Final VS       : {data['vs'][-1]:.3f} ft/s"
)
print(
    f"Max drift      : {data['max_drift'][-1]:.3f} ft"
)
print(
    f"Final drift    : {data['drift'][-1]:.3f} ft"
)
print(
    f"XY path        : {data['path'][-1]:.3f} ft"
)

print("\nHTML:")
print(OUTPUT_HTML)

print("\nGitHub Pages copy:")
print(PAGES_HTML)

print(
    "\nOpen the HTML in Colab/browser "
    "to watch the helicopter move."
)

print("=" * 90)
