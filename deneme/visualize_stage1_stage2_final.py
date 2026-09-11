from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill,
)
from helicopter_env_stage2_refine_mapped import (
    HelicopterEnvStage2RefineMapped,
)


# ============================================================
# FINAL LOCKED MODELS
# ============================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL_PATH = (
    "models_stage2_distilled/"
    "AH1S_STAGE2_DISTILLED_SUCCESS.zip"
)

OUTPUT_DIR = Path(
    "results_stage1_stage2_visual"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HTML_PATH = (
    OUTPUT_DIR
    /
    "stage1_stage2_final_visualization.html"
)

JSON_PATH = (
    OUTPUT_DIR
    /
    "stage1_stage2_final_trajectory.json"
)


# ============================================================
# CONSTANTS
# ============================================================

EARTH_RADIUS_FT = 20_902_231.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0

HANDOFF_STABLE_TIME = 5.0

TARGET_DISTANCE = 300.0

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040


# ============================================================
# HELPERS
# ============================================================

def fdm_float(
    fdm,
    key,
    default=float("nan"),
):
    try:
        return float(
            fdm[key]
        )
    except Exception:
        return float(
            default
        )


def info_float(
    info,
    key,
    default=float("nan"),
):
    try:
        return float(
            info.get(
                key,
                default,
            )
        )
    except Exception:
        return float(
            default
        )


def get_fdm(
    env,
):
    direct = getattr(
        env,
        "fdm",
        None,
    )

    if direct is not None:
        return direct

    base = getattr(
        env,
        "base_env",
        None,
    )

    if base is not None:
        nested = getattr(
            base,
            "fdm",
            None,
        )

        if nested is not None:
            return nested

    raise RuntimeError(
        "Active FDM not found."
    )


def latitude_deg(
    fdm,
):
    for key in [
        "position/lat-gc-deg",
        "position/lat-geod-deg",
    ]:
        x = fdm_float(
            fdm,
            key,
        )

        if np.isfinite(
            x
        ):
            return x

    return float("nan")


def longitude_deg(
    fdm,
):
    return fdm_float(
        fdm,
        "position/long-gc-deg",
    )


def heading_rad(
    fdm,
):
    for key in [
        "attitude/heading-true-rad",
        "attitude/psi-rad",
    ]:
        x = fdm_float(
            fdm,
            key,
        )

        if np.isfinite(
            x
        ):
            return x

    return float("nan")


def wrap_angle(
    x,
):
    return math.atan2(
        math.sin(
            x
        ),
        math.cos(
            x
        ),
    )


def local_ne_ft(
    lat,
    lon,
    lat0,
    lon0,
):
    dlat = math.radians(
        lat - lat0
    )

    dlon = math.radians(
        lon - lon0
    )

    north = (
        EARTH_RADIUS_FT
        *
        dlat
    )

    east = (
        EARTH_RADIUS_FT
        *
        math.cos(
            math.radians(
                lat0
            )
        )
        *
        dlon
    )

    return (
        float(
            north
        ),
        float(
            east
        ),
    )


def mission_axes(
    north,
    east,
    mission_heading,
):
    c = math.cos(
        mission_heading
    )

    s = math.sin(
        mission_heading
    )

    forward = (
        north
        *
        c
        +
        east
        *
        s
    )

    cross = (
        -north
        *
        s
        +
        east
        *
        c
    )

    return (
        float(
            forward
        ),
        float(
            cross
        ),
    )


def aircraft_point(
    phase,
    t,
    fdm,
    info,
    origin_lat,
    origin_lon,
    mission_heading,
    action,
):
    lat = latitude_deg(
        fdm
    )

    lon = longitude_deg(
        fdm
    )

    north, east = local_ne_ft(
        lat,
        lon,
        origin_lat,
        origin_lon,
    )

    forward, cross = mission_axes(
        north,
        east,
        mission_heading,
    )

    altitude = info_float(
        info,
        "altitude",
        fdm_float(
            fdm,
            "position/h-agl-ft",
        ),
    )

    vs = info_float(
        info,
        "vertical_speed",
        fdm_float(
            fdm,
            "velocities/h-dot-fps",
            0.0,
        ),
    )

    forward_velocity = info_float(
        info,
        "forward_velocity",
        fdm_float(
            fdm,
            "velocities/u-aero-fps",
            0.0,
        ),
    )

    lateral_velocity = info_float(
        info,
        "lateral_velocity",
        fdm_float(
            fdm,
            "velocities/v-aero-fps",
            0.0,
        ),
    )

    roll = info_float(
        info,
        "roll",
        fdm_float(
            fdm,
            "attitude/roll-rad",
            0.0,
        ),
    )

    pitch = info_float(
        info,
        "pitch",
        fdm_float(
            fdm,
            "attitude/pitch-rad",
            0.0,
        ),
    )

    heading_now = heading_rad(
        fdm
    )

    heading_error = wrap_angle(
        heading_now
        -
        mission_heading
    )

    return {
        "phase":
            phase,

        "time_s":
            float(
                t
            ),

        "north_ft":
            float(
                north
            ),

        "east_ft":
            float(
                east
            ),

        "forward_ft":
            float(
                forward
            ),

        "cross_track_ft":
            float(
                cross
            ),

        "altitude_ft":
            float(
                altitude
            ),

        "vertical_speed_fps":
            float(
                vs
            ),

        "forward_speed_fps":
            float(
                forward_velocity
            ),

        "lateral_speed_fps":
            float(
                lateral_velocity
            ),

        "roll_deg":
            float(
                math.degrees(
                    roll
                )
            ),

        "pitch_deg":
            float(
                math.degrees(
                    pitch
                )
            ),

        "heading_error_deg":
            float(
                math.degrees(
                    heading_error
                )
            ),

        "action0":
            float(
                action[0]
            ),

        "action1":
            float(
                action[1]
            ),

        "action2":
            float(
                action[2]
            ),

        "action3":
            float(
                action[3]
            ),
    }


# ============================================================
# LOAD FINAL POLICIES
# ============================================================

stage1_model = PPO.load(
    STAGE1_MODEL_PATH
)

stage2_model = PPO.load(
    STAGE2_MODEL_PATH
)


# ============================================================
# STAGE 1 — REAL LOCKED TAKEOFF
# ============================================================

print(
    "=" * 120
)

print(
    "FINAL VISUAL REPLAY — STAGE 1 + STAGE 2"
)

print(
    "=" * 120
)

print(
    "Stage 1 teacher/controller: OFF"
)

print(
    "Stage 2 teacher/controller: OFF"
)

print(
    "Continuous JSBSim FDM handoff: ON"
)

print()

env1 = (
    HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )
)

obs1, info1 = env1.reset()

fdm = get_fdm(
    env1
)

active_fdm_id = id(
    fdm
)

origin_lat = latitude_deg(
    fdm
)

origin_lon = longitude_deg(
    fdm
)

mission_heading = heading_rad(
    fdm
)

dt1 = float(
    getattr(
        env1,
        "dt",
        0.075,
    )
)

if (
    not np.isfinite(
        dt1
    )
    or
    dt1 <= 0.0
):
    dt1 = 0.075

trajectory = []

stable_time = 0.0
handoff_found = False

for step in range(
    int(
        STAGE1_MAX_TIME
        /
        dt1
    )
):
    action1, _ = (
        stage1_model.predict(
            obs1,
            deterministic=True,
        )
    )

    action1 = np.asarray(
        action1,
        dtype=np.float32,
    ).reshape(-1)

    (
        obs1,
        _,
        terminated1,
        truncated1,
        info1,
    ) = env1.step(
        action1
    )

    t1 = (
        step + 1
    ) * dt1

    point = aircraft_point(
        "STAGE 1 — VERTICAL TAKEOFF",
        t1,
        fdm,
        info1,
        origin_lat,
        origin_lon,
        mission_heading,
        action1,
    )

    trajectory.append(
        point
    )

    altitude = point[
        "altitude_ft"
    ]

    vs = point[
        "vertical_speed_fps"
    ]

    drift = float(
        np.hypot(
            point[
                "north_ft"
            ],
            point[
                "east_ft"
            ],
        )
    )

    hs = float(
        np.hypot(
            info_float(
                info1,
                "vn",
                0.0,
            ),
            info_float(
                info1,
                "ve",
                0.0,
            ),
        )
    )

    stable = bool(
        295.0
        <=
        altitude
        <=
        305.0

        and
        abs(
            vs
        )
        <=
        0.50

        and
        hs
        <=
        1.0

        and
        drift
        <=
        3.0
    )

    stable_time = (
        stable_time
        +
        dt1
        if stable
        else 0.0
    )

    if (
        stable_time
        >=
        HANDOFF_STABLE_TIME
    ):
        handoff_found = True

        break

    if (
        terminated1
        and
        not bool(
            info1.get(
                "success",
                False,
            )
        )
    ):
        raise RuntimeError(
            "Stage 1 failed before handoff."
        )

    if truncated1:
        raise RuntimeError(
            "Stage 1 truncated before handoff."
        )


if not handoff_found:
    raise RuntimeError(
        "Stable Stage-1 handoff was not found."
    )


# ============================================================
# STAGE 2 — SAME FDM, FINAL DISTILLED PPO
# ============================================================

env2 = (
    HelicopterEnvStage2RefineMapped(
        aileron_scale=
            AILERON_SCALE,

        rudder_scale=
            RUDDER_SCALE,
    )
)

# Disposable reset for Python bookkeeping only.
env2.reset()

env2.fdm = (
    fdm
)

if hasattr(
    env2,
    "forward_distance",
):
    env2.forward_distance = (
        0.0
    )

if hasattr(
    env2,
    "target_heading",
):
    env2.target_heading = float(
        mission_heading
    )

for attr in [
    "steps",
    "target_hold_steps",
    "hold_steps",
    "success_hold_steps",
]:
    if hasattr(
        env2,
        attr,
    ):
        setattr(
            env2,
            attr,
            0,
        )

if (
    id(
        get_fdm(
            env2
        )
    )
    !=
    active_fdm_id
):
    raise RuntimeError(
        "Continuous FDM handoff failed."
    )

obs2 = np.asarray(
    env2._get_obs(),
    dtype=np.float32,
)

dt2 = float(
    getattr(
        env2,
        "dt",
        0.075,
    )
)

if (
    not np.isfinite(
        dt2
    )
    or
    dt2 <= 0.0
):
    dt2 = 0.075

stage2_start_global_time = (
    trajectory[
        -1
    ][
        "time_s"
    ]
)

for step in range(
    int(
        STAGE2_MAX_TIME
        /
        dt2
    )
):
    action2, _ = (
        stage2_model.predict(
            obs2,
            deterministic=True,
        )
    )

    action2 = np.asarray(
        action2,
        dtype=np.float32,
    ).reshape(-1)

    (
        obs2,
        _,
        terminated2,
        truncated2,
        info2,
    ) = env2.step(
        action2
    )

    obs2 = np.asarray(
        obs2,
        dtype=np.float32,
    )

    global_t = (
        stage2_start_global_time
        +
        (
            step + 1
        )
        *
        dt2
    )

    point = aircraft_point(
        "STAGE 2 — STRAIGHT FORWARD",
        global_t,
        fdm,
        info2,
        origin_lat,
        origin_lon,
        mission_heading,
        action2,
    )

    trajectory.append(
        point
    )

    distance = info_float(
        info2,
        "forward_distance",
        getattr(
            env2,
            "forward_distance",
            0.0,
        ),
    )

    if (
        distance
        >=
        TARGET_DISTANCE
    ):
        break

    if (
        terminated2
        and
        not bool(
            info2.get(
                "success",
                False,
            )
        )
    ):
        raise RuntimeError(
            "Stage 2 failed before 300-ft crossing."
        )

    if truncated2:
        raise RuntimeError(
            "Stage 2 truncated before 300-ft crossing."
        )


# Detach shared FDM before closing Stage 1.
env2.fdm = None
env1.close()


# ============================================================
# TRAJECTORY SUMMARY
# ============================================================

stage1_points = [
    p
    for p in trajectory
    if p[
        "phase"
    ].startswith(
        "STAGE 1"
    )
]

stage2_points = [
    p
    for p in trajectory
    if p[
        "phase"
    ].startswith(
        "STAGE 2"
    )
]

stage1_max_drift = max(
    math.hypot(
        p[
            "north_ft"
        ],
        p[
            "east_ft"
        ],
    )
    for p in stage1_points
)

stage2_max_cross = max(
    abs(
        p[
            "cross_track_ft"
        ]
    )
    for p in stage2_points
)

final = trajectory[
    -1
]

summary = {
    "stage1_max_horizontal_drift_ft":
        stage1_max_drift,

    "stage2_max_cross_track_ft":
        stage2_max_cross,

    "final_forward_ft":
        final[
            "forward_ft"
        ],

    "final_cross_track_ft":
        final[
            "cross_track_ft"
        ],

    "final_altitude_ft":
        final[
            "altitude_ft"
        ],

    "final_vertical_speed_fps":
        final[
            "vertical_speed_fps"
        ],

    "total_time_s":
        final[
            "time_s"
        ],
}

JSON_PATH.write_text(
    json.dumps(
        {
            "summary":
                summary,

            "trajectory":
                trajectory,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "Stage1 max horizontal drift:",
    f"{stage1_max_drift:.3f} ft"
)

print(
    "Stage2 max cross-track:",
    f"{stage2_max_cross:.3f} ft"
)

print(
    "Final forward:",
    f"{final['forward_ft']:.3f} ft"
)

print(
    "Final cross-track:",
    f"{final['cross_track_ft']:.3f} ft"
)

print(
    "Final altitude:",
    f"{final['altitude_ft']:.3f} ft"
)


# ============================================================
# HTML
# ============================================================

data_json = json.dumps(
    trajectory
)

summary_json = json.dumps(
    summary
)

html = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AH-1S RL — Stage 1 + Stage 2 Final Replay</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    color-scheme: dark;
    --bg:#0b0d10;
    --panel:#12161b;
    --line:#2a3139;
    --text:#e7edf3;
    --muted:#9ba8b5;
    --accent:#f1f5f9;
  }
  * { box-sizing:border-box; }
  body {
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  }
  header {
    padding:18px 22px 10px 22px;
    border-bottom:1px solid var(--line);
  }
  h1 { margin:0 0 5px 0; font-size:20px; font-weight:700; }
  .sub { color:var(--muted); font-size:13px; }
  .grid {
    display:grid;
    grid-template-columns: 1.35fr 1fr;
    gap:12px;
    padding:12px;
  }
  .panel {
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:12px;
    overflow:hidden;
  }
  .panel-title {
    padding:10px 13px 0 13px;
    font-size:12px;
    color:var(--muted);
    font-weight:600;
    letter-spacing:.04em;
    text-transform:uppercase;
  }
  #scene3d { height:520px; }
  #topview,#sideview { height:250px; }
  #telemetry {
    padding:14px;
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:8px;
  }
  .metric {
    padding:10px;
    border:1px solid var(--line);
    border-radius:9px;
    min-width:0;
  }
  .metric .k { color:var(--muted); font-size:11px; margin-bottom:3px; }
  .metric .v { font-size:17px; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .controls {
    display:flex;
    align-items:center;
    gap:10px;
    padding:12px 14px 16px 14px;
  }
  button {
    border:1px solid var(--line);
    background:#1b222a;
    color:var(--text);
    border-radius:8px;
    padding:8px 12px;
    cursor:pointer;
  }
  input[type=range] { width:100%; }
  .legend {
    color:var(--muted);
    font-size:12px;
    padding:0 14px 12px 14px;
  }
  @media(max-width:900px){
    .grid { grid-template-columns:1fr; }
    #telemetry { grid-template-columns:repeat(2,minmax(0,1fr)); }
  }
</style>
</head>
<body>
<header>
  <h1>AH-1S RL — Final Stage 1 + Stage 2 Replay</h1>
  <div class="sub">True continuous JSBSim handoff · teacher/controller OFF · Stage 1 locked PPO → Stage 2 distilled PPO</div>
</header>

<div class="grid">
  <div class="panel">
    <div class="panel-title">3D Mission Replay</div>
    <div id="scene3d"></div>
    <div class="controls">
      <button id="playBtn">Play</button>
      <button id="pauseBtn">Pause</button>
      <input id="slider" type="range" min="0" max="1" value="0" step="1"/>
      <span id="timeLabel">0.0 s</span>
    </div>
    <div class="legend">Stage 1 = vertical takeoff / hover · Stage 2 = straight forward flight</div>
  </div>

  <div>
    <div class="panel" style="margin-bottom:12px">
      <div class="panel-title">Top View — Equal Scale</div>
      <div id="topview"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Side View</div>
      <div id="sideview"></div>
    </div>
  </div>
</div>

<div class="panel" style="margin:0 12px 12px 12px">
  <div class="panel-title">Live Telemetry</div>
  <div id="telemetry"></div>
</div>

<script>
const data = __DATA__;
const summary = __SUMMARY__;

const x = data.map(d => d.forward_ft);
const y = data.map(d => d.cross_track_ft);
const z = data.map(d => d.altitude_ft);
const t = data.map(d => d.time_s);

const stage1Idx = data.map((d,i)=>d.phase.startsWith("STAGE 1")?i:null).filter(i=>i!==null);
const stage2Idx = data.map((d,i)=>d.phase.startsWith("STAGE 2")?i:null).filter(i=>i!==null);

function subset(arr, idxs){ return idxs.map(i=>arr[i]); }

const stage1Trace3d = {
  type:'scatter3d',
  mode:'lines',
  name:'Stage 1',
  x:subset(x,stage1Idx),
  y:subset(y,stage1Idx),
  z:subset(z,stage1Idx),
  line:{width:6}
};

const stage2Trace3d = {
  type:'scatter3d',
  mode:'lines',
  name:'Stage 2',
  x:subset(x,stage2Idx),
  y:subset(y,stage2Idx),
  z:subset(z,stage2Idx),
  line:{width:6}
};

const heli3d = {
  type:'scatter3d',
  mode:'markers',
  name:'Aircraft',
  x:[x[0]], y:[y[0]], z:[z[0]],
  marker:{size:7, symbol:'diamond'}
};

const target3d = {
  type:'scatter3d',
  mode:'markers+text',
  name:'300 ft forward target',
  x:[300], y:[0], z:[300],
  text:['Target'],
  textposition:'top center',
  marker:{size:5, symbol:'circle-open'}
};

Plotly.newPlot('scene3d',
  [stage1Trace3d, stage2Trace3d, heli3d, target3d],
  {
    margin:{l:0,r:0,b:0,t:0},
    paper_bgcolor:'#12161b',
    plot_bgcolor:'#12161b',
    font:{color:'#e7edf3'},
    scene:{
      xaxis:{title:'Forward (ft)',gridcolor:'#2a3139',zerolinecolor:'#4b5563'},
      yaxis:{title:'Cross-track (ft)',gridcolor:'#2a3139',zerolinecolor:'#4b5563'},
      zaxis:{title:'Altitude AGL (ft)',gridcolor:'#2a3139',zerolinecolor:'#4b5563'},
      aspectmode:'data',
      camera:{eye:{x:1.6,y:1.4,z:0.85}}
    },
    legend:{orientation:'h',x:0,y:1.03}
  },
  {responsive:true}
);

const topStage1 = {
  type:'scatter', mode:'lines', name:'Stage 1',
  x:subset(x,stage1Idx), y:subset(y,stage1Idx), line:{width:3}
};
const topStage2 = {
  type:'scatter', mode:'lines', name:'Stage 2',
  x:subset(x,stage2Idx), y:subset(y,stage2Idx), line:{width:3}
};
const topHeli = {
  type:'scatter', mode:'markers', name:'Aircraft',
  x:[x[0]], y:[y[0]], marker:{size:10, symbol:'diamond'}
};

Plotly.newPlot('topview',
  [topStage1,topStage2,topHeli],
  {
    margin:{l:55,r:15,b:45,t:10},
    paper_bgcolor:'#12161b',
    plot_bgcolor:'#12161b',
    font:{color:'#e7edf3'},
    xaxis:{title:'Forward (ft)',gridcolor:'#2a3139',zerolinecolor:'#4b5563'},
    yaxis:{
      title:'Cross-track (ft)',
      gridcolor:'#2a3139',
      zerolinecolor:'#4b5563',
      scaleanchor:'x',
      scaleratio:1
    },
    showlegend:false
  },
  {responsive:true}
);

const sideStage1 = {
  type:'scatter', mode:'lines', name:'Stage 1',
  x:subset(x,stage1Idx), y:subset(z,stage1Idx), line:{width:3}
};
const sideStage2 = {
  type:'scatter', mode:'lines', name:'Stage 2',
  x:subset(x,stage2Idx), y:subset(z,stage2Idx), line:{width:3}
};
const sideHeli = {
  type:'scatter', mode:'markers', name:'Aircraft',
  x:[x[0]], y:[z[0]], marker:{size:10, symbol:'diamond'}
};

Plotly.newPlot('sideview',
  [sideStage1,sideStage2,sideHeli],
  {
    margin:{l:55,r:15,b:45,t:10},
    paper_bgcolor:'#12161b',
    plot_bgcolor:'#12161b',
    font:{color:'#e7edf3'},
    xaxis:{title:'Forward (ft)',gridcolor:'#2a3139',zerolinecolor:'#4b5563'},
    yaxis:{title:'Altitude AGL (ft)',gridcolor:'#2a3139',zerolinecolor:'#4b5563'},
    showlegend:false
  },
  {responsive:true}
);

const telemetry = document.getElementById('telemetry');

function metric(k,v){
  return `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`;
}

function updateTelemetry(i){
  const d = data[i];
  telemetry.innerHTML =
      metric('Phase', d.phase.replace('STAGE 1 — ','').replace('STAGE 2 — ',''))
    + metric('Time', d.time_s.toFixed(2)+' s')
    + metric('Forward', d.forward_ft.toFixed(2)+' ft')
    + metric('Cross-track', d.cross_track_ft.toFixed(2)+' ft')
    + metric('Altitude', d.altitude_ft.toFixed(2)+' ft')
    + metric('Vertical speed', d.vertical_speed_fps.toFixed(2)+' ft/s')
    + metric('Forward speed', d.forward_speed_fps.toFixed(2)+' ft/s')
    + metric('Lateral speed', d.lateral_speed_fps.toFixed(2)+' ft/s')
    + metric('Pitch', d.pitch_deg.toFixed(2)+'°')
    + metric('Roll', d.roll_deg.toFixed(2)+'°')
    + metric('Heading error', d.heading_error_deg.toFixed(2)+'°')
    + metric('Actions', `[${d.action0.toFixed(2)}, ${d.action1.toFixed(2)}, ${d.action2.toFixed(2)}, ${d.action3.toFixed(2)}]`);
}

const slider = document.getElementById('slider');
const timeLabel = document.getElementById('timeLabel');
slider.max = data.length - 1;

let current = 0;
let timer = null;

function setFrame(i){
  current = Math.max(0, Math.min(data.length-1, i));
  slider.value = current;
  timeLabel.textContent = data[current].time_s.toFixed(1)+' s';

  Plotly.restyle('scene3d',
    {x:[[x[current]]], y:[[y[current]]], z:[[z[current]]]},
    [2]
  );

  Plotly.restyle('topview',
    {x:[[x[current]]], y:[[y[current]]]},
    [2]
  );

  Plotly.restyle('sideview',
    {x:[[x[current]]], y:[[z[current]]]},
    [2]
  );

  updateTelemetry(current);
}

function play(){
  if(timer) return;
  timer = setInterval(()=>{
    if(current >= data.length-1){
      clearInterval(timer);
      timer = null;
      return;
    }
    setFrame(current+2);
  }, 30);
}

function pause(){
  if(timer){
    clearInterval(timer);
    timer = null;
  }
}

document.getElementById('playBtn').onclick = play;
document.getElementById('pauseBtn').onclick = pause;
slider.oninput = e => setFrame(Number(e.target.value));

setFrame(0);
</script>
</body>
</html>
"""

html = (
    html
    .replace(
        "__DATA__",
        data_json,
    )
    .replace(
        "__SUMMARY__",
        summary_json,
    )
)

HTML_PATH.write_text(
    html,
    encoding="utf-8",
)

print()
print(
    "Saved trajectory:",
    JSON_PATH,
)

print(
    "Saved interactive replay:",
    HTML_PATH,
)

print()
print(
    "Open in Colab with:"
)

print(
    'from IPython.display import HTML, display'
)

print(
    f'display(HTML(filename="{HTML_PATH}"))'
)
