from pathlib import Path
import csv
import json
import math

import numpy as np


# =====================================================================
# AH-1S / JSBSim
# TRUE MISSION — FORWARD-FLIGHT TURN ENTRY DIAGNOSTIC V1
# =====================================================================
#
# Corrected mission:
#   Stage 1 : rotor already running -> takeoff -> 300 ft hover
#   Stage 2 : straight forward flight while holding ~300 ft
#   NEW     : 45-degree coordinated turn WHILE FORWARD FLIGHT CONTINUES
#   FINAL   : continue forward on the new heading while holding ~300 ft
#
# This script ONLY finds a clean state during locked Stage-2 straight
# forward flight from which the new turn maneuver should begin.
#
# NO descent.
# NO landing.
# NO turn command yet.
# NO training.
#
# =====================================================================


BASE_SOURCE = Path(
    "diagnose_stage4_entry_margin_v3.py"
)

RESULT_DIR = Path(
    "results_turn_entry_forward_v1"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# LOAD LOCKED DEFINITIONS ONLY
# =====================================================================

if not BASE_SOURCE.exists():
    raise FileNotFoundError(
        f"Missing locked helper source: {BASE_SOURCE}"
    )

text = BASE_SOURCE.read_text(
    encoding="utf-8"
)

MARKER = (
    'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'
)

if MARKER not in text:
    raise RuntimeError(
        "Could not locate definition/experiment split marker."
    )

prefix = text.split(
    MARKER,
    1,
)[0]

ns = {
    "__name__": "turn_entry_forward_base",
    "__file__": str(BASE_SOURCE),
}

exec(
    compile(
        prefix,
        str(BASE_SOURCE),
        "exec",
    ),
    ns,
)


HelicopterEnvStage1Distill = ns[
    "HelicopterEnvStage1Distill"
]

HelicopterEnvStage2RefineMapped = ns[
    "HelicopterEnvStage2RefineMapped"
]

stage1_model = ns[
    "stage1_model"
]

stage2_model = ns[
    "stage2_model"
]

get_fdm = ns[
    "get_fdm"
]

heading_rad = ns[
    "heading_rad"
]

latitude_deg = ns[
    "latitude_deg"
]

longitude_deg = ns[
    "longitude_deg"
]

geometry = ns[
    "geometry"
]

snapshot = ns[
    "snapshot"
]

env_control_dt = ns[
    "env_control_dt"
]

first_finite = ns[
    "first_finite"
]

info_float = ns[
    "info_float"
]

AILERON_SCALE = ns[
    "AILERON_SCALE"
]

RUDDER_SCALE = ns[
    "RUDDER_SCALE"
]

HANDOFF_STABLE_TIME = ns[
    "HANDOFF_STABLE_TIME"
]

STAGE1_MAX_TIME = ns[
    "STAGE1_MAX_TIME"
]


# =====================================================================
# FORWARD-FLIGHT MILESTONES
# =====================================================================

MILESTONES_FT = [
    50.0,
    80.0,
    120.0,
    160.0,
    200.0,
    240.0,
    270.0,
]

MAX_STAGE2_TIME_S = 70.0

# Only diagnostic quality limits.
ALT_MIN_FT = 295.0
ALT_MAX_FT = 305.0
MAX_ABS_VS_FPS = 2.0
MAX_ABS_ROLL_DEG = 8.0
MAX_ABS_PITCH_DEG = 8.0
MAX_ABS_HEADING_DEG = 2.0
MAX_ABS_CROSS_FT = 5.0

# A useful turn-entry state must still be genuinely moving forward.
MIN_FORWARD_SPEED_FPS = 3.0


def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def state_to_row(
    t,
    state,
):
    return {
        "time_s": float(t),

        "forward_ft": float(
            state["forward_ft"]
        ),

        "forward_speed_fps": float(
            state["forward_speed_fps"]
        ),

        "cross_track_ft": float(
            state["cross_track_ft"]
        ),

        "lateral_speed_fps": float(
            state["lateral_speed_fps"]
        ),

        "altitude_ft": float(
            state["altitude_ft"]
        ),

        "vertical_speed_fps": float(
            state["vertical_speed_fps"]
        ),

        "heading_error_deg": float(
            state["heading_error_deg"]
        ),

        "roll_deg": float(
            math.degrees(
                state["roll_rad"]
            )
        ),

        "pitch_deg": float(
            math.degrees(
                state["pitch_rad"]
            )
        ),

        "yaw_rate_deg_s": float(
            math.degrees(
                state["yaw_rate_rad_s"]
            )
        ),

        "physical_collective_cmd": float(
            state["physical_collective_cmd"]
        ),

        "physical_elevator_cmd": float(
            state["physical_elevator_cmd"]
        ),

        "physical_aileron_cmd": float(
            state["physical_aileron_cmd"]
        ),

        "physical_rudder_cmd": float(
            state["physical_rudder_cmd"]
        ),
    }


def entry_quality(
    row,
):
    return bool(
        ALT_MIN_FT
        <=
        row["altitude_ft"]
        <=
        ALT_MAX_FT

        and
        abs(
            row["vertical_speed_fps"]
        )
        <=
        MAX_ABS_VS_FPS

        and
        row["forward_speed_fps"]
        >=
        MIN_FORWARD_SPEED_FPS

        and
        abs(
            row["cross_track_ft"]
        )
        <=
        MAX_ABS_CROSS_FT

        and
        abs(
            row["heading_error_deg"]
        )
        <=
        MAX_ABS_HEADING_DEG

        and
        abs(
            row["roll_deg"]
        )
        <=
        MAX_ABS_ROLL_DEG

        and
        abs(
            row["pitch_deg"]
        )
        <=
        MAX_ABS_PITCH_DEG
    )


# =====================================================================
# STAGE 1 — LOCKED TAKEOFF + 300 FT HOVER
# =====================================================================

print("=" * 120)
print("TRUE MISSION — FORWARD-FLIGHT TURN ENTRY DIAGNOSTIC V1")
print("=" * 120)
print("No descent. No landing. No turn command. No training.")
print()

env1 = HelicopterEnvStage1Distill(
    teacher_model_path=None,
    training_mode=False,
)

obs1, info1 = env1.reset()

fdm = get_fdm(env1)
active_fdm_id = id(fdm)

mission_heading = heading_rad(
    fdm
)

dt1 = env_control_dt(
    env1
)

stable_time = 0.0
stage1_elapsed = 0.0

for _ in range(
    int(
        STAGE1_MAX_TIME
        /
        dt1
    )
):
    action1, _ = stage1_model.predict(
        obs1,
        deterministic=True,
    )

    obs1, _, terminated, truncated, info1 = env1.step(
        action1
    )

    stage1_elapsed += dt1

    altitude = info_float(
        info1,
        "altitude"
    )

    vertical_speed = info_float(
        info1,
        "vertical_speed"
    )

    vn = info_float(
        info1,
        "vn",
        0.0,
    )

    ve = info_float(
        info1,
        "ve",
        0.0,
    )

    horizontal_speed = float(
        np.hypot(
            vn,
            ve,
        )
    )

    drift = info_float(
        info1,
        "drift",
        999.0,
    )

    stable = bool(
        295.0
        <=
        altitude
        <=
        305.0

        and
        abs(
            vertical_speed
        )
        <=
        0.50

        and
        horizontal_speed
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
        else
        0.0
    )

    if (
        stable_time
        >=
        HANDOFF_STABLE_TIME
    ):
        break

    if (
        terminated
        and
        not bool(
            info1.get(
                "success",
                False,
            )
        )
    ):
        env1.close()
        raise RuntimeError(
            "Stage 1 failed before stable handoff."
        )

    if truncated:
        env1.close()
        raise RuntimeError(
            "Stage 1 truncated before stable handoff."
        )


if (
    stable_time
    <
    HANDOFF_STABLE_TIME
):
    env1.close()
    raise RuntimeError(
        "Stage 1 stable 300-ft handoff was not reached."
    )


lat0 = latitude_deg(
    fdm
)

lon0 = longitude_deg(
    fdm
)


print(
    f"Stage1 PASS | "
    f"t={stage1_elapsed:.2f}s | "
    f"ALT={altitude:.2f} ft | "
    f"VS={vertical_speed:+.3f} ft/s"
)


# =====================================================================
# ATTACH LOCKED STAGE 2 TO SAME FDM
# =====================================================================

sim_before = first_finite(
    fdm,
    [
        "simulation/sim-time-sec"
    ],
)

env2 = HelicopterEnvStage2RefineMapped(
    aileron_scale=AILERON_SCALE,
    rudder_scale=RUDDER_SCALE,
)

env2.reset()
env2.fdm = fdm

if hasattr(
    env2,
    "forward_distance",
):
    env2.forward_distance = 0.0

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
    env2.fdm = None
    env1.close()
    raise RuntimeError(
        "Same-FDM continuity failed."
    )


sim_after = first_finite(
    fdm,
    [
        "simulation/sim-time-sec"
    ],
)

if (
    np.isfinite(
        sim_before
    )
    and
    np.isfinite(
        sim_after
    )
    and
    abs(
        sim_after
        -
        sim_before
    )
    >
    1e-9
):
    env2.fdm = None
    env1.close()
    raise RuntimeError(
        "Simulation clock changed during Stage2 attach."
    )


obs2 = np.asarray(
    env2._get_obs(),
    dtype=np.float32,
)

dt2 = env_control_dt(
    env2
)

print(
    "Stage2 attach PASS | same_fdm=True | clock_reset=False"
)

print()


# =====================================================================
# RUN STRAIGHT FLIGHT AND RECORD ENTRY CANDIDATES
# =====================================================================

trace = []
milestone_rows = []

pending = list(
    MILESTONES_FT
)

termination = "time_limit"
elapsed = 0.0

for step in range(
    int(
        MAX_STAGE2_TIME_S
        /
        dt2
    )
):
    action2, _ = stage2_model.predict(
        obs2,
        deterministic=True,
    )

    action2 = np.asarray(
        action2,
        dtype=np.float32,
    ).reshape(-1)

    obs2, _, terminated, truncated, info2 = env2.step(
        action2
    )

    obs2 = np.asarray(
        obs2,
        dtype=np.float32,
    )

    elapsed = (
        step
        +
        1
    ) * dt2

    state = snapshot(
        fdm,
        lat0,
        lon0,
        mission_heading,
    )

    row = state_to_row(
        elapsed,
        state,
    )

    trace.append(
        row
    )

    while (
        pending
        and
        state["forward_ft"]
        >=
        pending[0]
    ):
        target = pending.pop(
            0
        )

        m = dict(
            row
        )

        m[
            "milestone_ft"
        ] = float(
            target
        )

        m[
            "entry_quality_pass"
        ] = bool(
            entry_quality(
                row
            )
        )

        milestone_rows.append(
            m
        )

        print(
            f"{target:6.1f} ft | "
            f"Vfwd={row['forward_speed_fps']:+7.3f} | "
            f"ALT={row['altitude_ft']:7.3f} | "
            f"VS={row['vertical_speed_fps']:+6.3f} | "
            f"X={row['cross_track_ft']:+6.3f} | "
            f"HDG={row['heading_error_deg']:+6.3f}° | "
            f"ROLL={row['roll_deg']:+6.3f}° | "
            f"PITCH={row['pitch_deg']:+6.3f}° | "
            f"entry={m['entry_quality_pass']}"
        )

    if not pending:
        termination = "all_milestones_reached"
        break

    if (
        terminated
        and
        not bool(
            info2.get(
                "success",
                False,
            )
        )
    ):
        termination = "stage2_failed"
        break

    if truncated:
        termination = "stage2_truncated"
        break


save_csv(
    RESULT_DIR
    /
    "straight_flight_trace.csv",
    trace,
)

save_csv(
    RESULT_DIR
    /
    "milestone_summary.csv",
    milestone_rows,
)


passing = [
    r
    for r
    in milestone_rows
    if r[
        "entry_quality_pass"
    ]
]


# Prefer a state with useful forward speed and plenty of straight-flight
# distance already demonstrated. Among passing milestones, choose the
# latest one that is not yet too close to 300 ft.
selected = None

if passing:
    eligible = [
        r
        for r
        in passing
        if r[
            "milestone_ft"
        ]
        <=
        240.0
    ]

    pool = (
        eligible
        if eligible
        else passing
    )

    selected = max(
        pool,
        key=lambda r: (
            r[
                "milestone_ft"
            ],
            r[
                "forward_speed_fps"
            ],
        ),
    )


ready = bool(
    selected is not None
)


print()
print("=" * 120)
print("TURN ENTRY CONCLUSION")
print("=" * 120)

print(
    "termination:",
    termination
)

print(
    "passing milestones:",
    len(
        passing
    ),
    "/",
    len(
        milestone_rows
    ),
)

if selected is not None:

    print(
        "SELECTED TURN ENTRY:"
    )

    print(
        f"  forward = "
        f"{selected['forward_ft']:.3f} ft"
    )

    print(
        f"  speed   = "
        f"{selected['forward_speed_fps']:.3f} ft/s"
    )

    print(
        f"  altitude= "
        f"{selected['altitude_ft']:.3f} ft"
    )

    print(
        f"  VS      = "
        f"{selected['vertical_speed_fps']:+.3f} ft/s"
    )

    print(
        f"  heading = "
        f"{selected['heading_error_deg']:+.3f} deg"
    )

    print(
        f"  roll    = "
        f"{selected['roll_deg']:+.3f} deg"
    )

    print(
        f"  pitch   = "
        f"{selected['pitch_deg']:+.3f} deg"
    )

else:

    print(
        "SELECTED TURN ENTRY: NONE"
    )


print(
    "READY FOR FORWARD-FLIGHT TURN AUTHORITY TEST:",
    ready
)


summary = {
    "training_type": "NONE",
    "descent_used": False,
    "landing_used": False,
    "turn_command_used": False,
    "same_fdm": True,
    "mission_definition": (
        "takeoff -> 300 ft hover -> straight forward flight -> "
        "45-degree turn while moving -> forward flight on new heading"
    ),
    "termination": termination,
    "milestones_requested_ft": MILESTONES_FT,
    "milestones": milestone_rows,
    "selected_turn_entry": selected,
    "ready_for_forward_flight_turn_authority_test": ready,
}


with (
    RESULT_DIR
    /
    "final_summary.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("Saved:")
print(
    " ",
    RESULT_DIR
    /
    "straight_flight_trace.csv"
)
print(
    " ",
    RESULT_DIR
    /
    "milestone_summary.csv"
)
print(
    " ",
    RESULT_DIR
    /
    "final_summary.json"
)

print()
print(
    "Next only after reading this result: "
    "apply small coordinated turn commands at the selected "
    "FORWARD-FLIGHT state, not from hover."
)


env2.fdm = None
env1.close()
