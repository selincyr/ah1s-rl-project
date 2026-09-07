from pathlib import Path
import csv
import json
import math

import numpy as np


# =====================================================================
# AH-1S / JSBSim
# TRUE MISSION STAGE-4 TURN COUPLING SWEEP V2
# =====================================================================
#
# Mission branch:
#   Stage 1 -> takeoff / 300 ft
#   Stage 2 -> forward flight
#   Stage 3 -> stop / stabilized endpoint
#   Stage 4 -> 45 deg turn while holding ~300 ft
#
# THIS SCRIPT:
#   - does NOT descend
#   - does NOT land
#   - does NOT train PPO
#   - does NOT yet attempt the full 45 deg turn
#
# V1 evidence:
#   rudder_minus -> positive heading change
#   rudder_plus  -> negative heading change
#   aileron effect on heading was tiny at +/-0.15 for 1.5 s
#   altitude coupling was negligible in that small-pulse test
#
# V2 purpose:
#   Test stronger/longer positive-turn authority and aileron/rudder coupling.
#   Collective is deliberately left under the locked Stage-3 policy.
#
# =====================================================================


BASE_SOURCE = Path(
    "diagnose_stage4_entry_margin_v3.py"
)

RESULT_DIR = Path(
    "results_stage4_turn_coupling_v2"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# LOAD ONLY DEFINITIONS FROM LOCKED MISSION SOURCE
# =====================================================================

if not BASE_SOURCE.exists():
    raise FileNotFoundError(
        f"Missing locked mission helper source: {BASE_SOURCE}"
    )

source_text = BASE_SOURCE.read_text(
    encoding="utf-8"
)

MARKER = (
    'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'
)

if MARKER not in source_text:
    raise RuntimeError(
        "Could not locate experiment marker in locked mission source."
    )

prefix = source_text.split(
    MARKER,
    1,
)[0]

ns = {
    "__name__": "stage4_turn_coupling_base",
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

build_stage4_handoff = ns["build_stage4_handoff"]
close_handoff = ns["close_handoff"]
snapshot = ns["snapshot"]
stage3_observation = ns["stage3_observation"]
raw_policy_cycle = ns["raw_policy_cycle"]
env_control_dt = ns["env_control_dt"]
stage3_model = ns["stage3_model"]


# =====================================================================
# EXPERIMENT
# =====================================================================

PULSE_SECONDS = 4.0
RECOVERY_SECONDS = 6.0

# Positive-turn direction from V1:
#     delta_a3 < 0  -> positive heading change
#
# aileron-plus is paired with it for the first coordinated-turn sweep,
# because V1 showed aileron_plus moved roll in the positive direction
# relative to baseline.
#
# No collective offset yet.
CASES = [
    {
        "name": "baseline",
        "delta_a2": 0.00,
        "delta_a3": 0.00,
    },
    {
        "name": "rudder_m030",
        "delta_a2": 0.00,
        "delta_a3": -0.30,
    },
    {
        "name": "rudder_m045",
        "delta_a2": 0.00,
        "delta_a3": -0.45,
    },
    {
        "name": "rudder_m060",
        "delta_a2": 0.00,
        "delta_a3": -0.60,
    },
    {
        "name": "coord_soft",
        "delta_a2": +0.15,
        "delta_a3": -0.30,
    },
    {
        "name": "coord_mid",
        "delta_a2": +0.30,
        "delta_a3": -0.45,
    },
    {
        "name": "coord_strong",
        "delta_a2": +0.45,
        "delta_a3": -0.60,
    },
]


# Diagnostic safety only, not final mission acceptance.
SAFE_ALT_MIN_FT = 290.0
SAFE_ALT_MAX_FT = 310.0
SAFE_MAX_ABS_ROLL_DEG = 12.0
SAFE_MAX_ABS_PITCH_DEG = 10.0
SAFE_MAX_ABS_YAW_RATE_DEG_S = 15.0

# A candidate is considered useful for the next closed-loop turn test if:
MIN_USEFUL_NET_HEADING_DEG = 1.0
MAX_USEFUL_ALT_DEVIATION_FT = 2.0
MAX_USEFUL_ABS_ROLL_DEG = 8.0


def wrap_deg(x):
    return float(
        (
            float(x)
            + 180.0
        )
        % 360.0
        - 180.0
    )


def heading_delta_deg(current, initial):
    return wrap_deg(
        float(current)
        -
        float(initial)
    )


def safety_reason(state):
    roll_deg = math.degrees(
        state["roll_rad"]
    )

    pitch_deg = math.degrees(
        state["pitch_rad"]
    )

    yaw_rate_deg_s = math.degrees(
        state["yaw_rate_rad_s"]
    )

    if state["altitude_ft"] < SAFE_ALT_MIN_FT:
        return "altitude_low"

    if state["altitude_ft"] > SAFE_ALT_MAX_FT:
        return "altitude_high"

    if abs(roll_deg) > SAFE_MAX_ABS_ROLL_DEG:
        return "roll_limit"

    if abs(pitch_deg) > SAFE_MAX_ABS_PITCH_DEG:
        return "pitch_limit"

    if abs(yaw_rate_deg_s) > SAFE_MAX_ABS_YAW_RATE_DEG_S:
        return "yaw_rate_limit"

    return ""


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


def run_case(cfg):
    start = build_stage4_handoff(
        detailed=False,
        require_full_hold=True,
    )

    try:
        if not start["handoff_pass"]:
            raise RuntimeError(
                "Locked Stage-3 endpoint handoff did not pass."
            )

        if not start["same_fdm"]:
            raise RuntimeError(
                "FDM continuity failed."
            )

        env2 = start["env2"]
        fdm = start["fdm"]
        lat0 = start["lat0"]
        lon0 = start["lon0"]
        mission_heading = start["mission_heading"]

        dt = env_control_dt(env2)

        initial = snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        state = initial.copy()

        rows = []
        termination = "completed"

        pulse_steps = int(
            round(
                PULSE_SECONDS / dt
            )
        )

        recovery_steps = int(
            round(
                RECOVERY_SECONDS / dt
            )
        )

        max_abs_roll = abs(
            math.degrees(
                initial["roll_rad"]
            )
        )

        max_abs_pitch = abs(
            math.degrees(
                initial["pitch_rad"]
            )
        )

        max_abs_yaw_rate = abs(
            math.degrees(
                initial["yaw_rate_rad_s"]
            )
        )

        min_alt = float(
            initial["altitude_ft"]
        )

        max_alt = float(
            initial["altitude_ft"]
        )

        min_vs = float(
            initial["vertical_speed_fps"]
        )

        max_vs = float(
            initial["vertical_speed_fps"]
        )

        min_used_a2 = +999.0
        max_used_a2 = -999.0
        min_used_a3 = +999.0
        max_used_a3 = -999.0

        min_phys_ail = +999.0
        max_phys_ail = -999.0
        min_phys_rud = +999.0
        max_phys_rud = -999.0

        pulse_end = initial.copy()

        # -------------------------------------------------------------
        # PULSE / COUPLING TEST
        # -------------------------------------------------------------

        for step in range(pulse_steps):

            before = snapshot(
                fdm,
                lat0,
                lon0,
                mission_heading,
            )

            obs3 = stage3_observation(
                before
            )

            base, _ = stage3_model.predict(
                obs3,
                deterministic=True,
            )

            base = np.asarray(
                base,
                dtype=np.float32,
            ).reshape(-1)

            action = base.copy()

            # Keep Stage-3 collective/elevator exactly as predicted.
            # Only lateral + yaw residuals are perturbed.
            action[2] = float(
                np.clip(
                    action[2]
                    +
                    cfg["delta_a2"],
                    -1.0,
                    +1.0,
                )
            )

            action[3] = float(
                np.clip(
                    action[3]
                    +
                    cfg["delta_a3"],
                    -1.0,
                    +1.0,
                )
            )

            state, used = raw_policy_cycle(
                env2,
                fdm,
                action,
                lat0,
                lon0,
                mission_heading,
            )

            t = (step + 1) * dt

            roll_deg = math.degrees(
                state["roll_rad"]
            )

            pitch_deg = math.degrees(
                state["pitch_rad"]
            )

            yaw_rate_deg_s = math.degrees(
                state["yaw_rate_rad_s"]
            )

            max_abs_roll = max(
                max_abs_roll,
                abs(roll_deg),
            )

            max_abs_pitch = max(
                max_abs_pitch,
                abs(pitch_deg),
            )

            max_abs_yaw_rate = max(
                max_abs_yaw_rate,
                abs(yaw_rate_deg_s),
            )

            min_alt = min(
                min_alt,
                state["altitude_ft"],
            )

            max_alt = max(
                max_alt,
                state["altitude_ft"],
            )

            min_vs = min(
                min_vs,
                state["vertical_speed_fps"],
            )

            max_vs = max(
                max_vs,
                state["vertical_speed_fps"],
            )

            min_used_a2 = min(
                min_used_a2,
                float(used[2]),
            )

            max_used_a2 = max(
                max_used_a2,
                float(used[2]),
            )

            min_used_a3 = min(
                min_used_a3,
                float(used[3]),
            )

            max_used_a3 = max(
                max_used_a3,
                float(used[3]),
            )

            min_phys_ail = min(
                min_phys_ail,
                state["physical_aileron_cmd"],
            )

            max_phys_ail = max(
                max_phys_ail,
                state["physical_aileron_cmd"],
            )

            min_phys_rud = min(
                min_phys_rud,
                state["physical_rudder_cmd"],
            )

            max_phys_rud = max(
                max_phys_rud,
                state["physical_rudder_cmd"],
            )

            rows.append({
                "case": cfg["name"],
                "phase": "pulse",
                "time_s": float(t),

                "heading_change_deg": float(
                    heading_delta_deg(
                        state["heading_error_deg"],
                        initial["heading_error_deg"],
                    )
                ),

                "heading_error_deg": float(
                    state["heading_error_deg"]
                ),

                "altitude_ft": float(
                    state["altitude_ft"]
                ),

                "vertical_speed_fps": float(
                    state["vertical_speed_fps"]
                ),

                "roll_deg": float(
                    roll_deg
                ),

                "pitch_deg": float(
                    pitch_deg
                ),

                "yaw_rate_deg_s": float(
                    yaw_rate_deg_s
                ),

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

                "base_a0": float(base[0]),
                "base_a1": float(base[1]),
                "base_a2": float(base[2]),
                "base_a3": float(base[3]),

                "used_a0": float(used[0]),
                "used_a1": float(used[1]),
                "used_a2": float(used[2]),
                "used_a3": float(used[3]),

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
            })

            pulse_end = state.copy()

            reason = safety_reason(state)

            if reason:
                termination = (
                    "pulse:"
                    +
                    reason
                )
                break

        # -------------------------------------------------------------
        # RECOVERY — LOCKED STAGE-3 ONLY
        # -------------------------------------------------------------

        if termination == "completed":

            for step in range(recovery_steps):

                before = snapshot(
                    fdm,
                    lat0,
                    lon0,
                    mission_heading,
                )

                obs3 = stage3_observation(
                    before
                )

                base, _ = stage3_model.predict(
                    obs3,
                    deterministic=True,
                )

                base = np.asarray(
                    base,
                    dtype=np.float32,
                ).reshape(-1)

                state, used = raw_policy_cycle(
                    env2,
                    fdm,
                    base,
                    lat0,
                    lon0,
                    mission_heading,
                )

                t = (
                    PULSE_SECONDS
                    +
                    (step + 1) * dt
                )

                rows.append({
                    "case": cfg["name"],
                    "phase": "recovery",
                    "time_s": float(t),

                    "heading_change_deg": float(
                        heading_delta_deg(
                            state["heading_error_deg"],
                            initial["heading_error_deg"],
                        )
                    ),

                    "heading_error_deg": float(
                        state["heading_error_deg"]
                    ),

                    "altitude_ft": float(
                        state["altitude_ft"]
                    ),

                    "vertical_speed_fps": float(
                        state["vertical_speed_fps"]
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

                    "base_a0": float(base[0]),
                    "base_a1": float(base[1]),
                    "base_a2": float(base[2]),
                    "base_a3": float(base[3]),

                    "used_a0": float(used[0]),
                    "used_a1": float(used[1]),
                    "used_a2": float(used[2]),
                    "used_a3": float(used[3]),

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
                })

                reason = safety_reason(state)

                if reason:
                    termination = (
                        "recovery:"
                        +
                        reason
                    )
                    break

        final = snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        return {
            "case": cfg["name"],
            "delta_a2": float(cfg["delta_a2"]),
            "delta_a3": float(cfg["delta_a3"]),

            "same_fdm": bool(start["same_fdm"]),
            "stage3_handoff_pass": bool(
                start["handoff_pass"]
            ),
            "stage3_hover_hold_s": float(
                start["stage3_hover_hold_s"]
            ),

            "initial_altitude_ft": float(
                initial["altitude_ft"]
            ),

            "initial_heading_error_deg": float(
                initial["heading_error_deg"]
            ),

            "pulse_heading_change_deg": float(
                heading_delta_deg(
                    pulse_end["heading_error_deg"],
                    initial["heading_error_deg"],
                )
            ),

            "pulse_end_roll_deg": float(
                math.degrees(
                    pulse_end["roll_rad"]
                )
            ),

            "pulse_end_yaw_rate_deg_s": float(
                math.degrees(
                    pulse_end["yaw_rate_rad_s"]
                )
            ),

            "pulse_altitude_change_ft": float(
                pulse_end["altitude_ft"]
                -
                initial["altitude_ft"]
            ),

            "pulse_end_vs_fps": float(
                pulse_end["vertical_speed_fps"]
            ),

            "max_abs_roll_deg": float(
                max_abs_roll
            ),

            "max_abs_pitch_deg": float(
                max_abs_pitch
            ),

            "max_abs_yaw_rate_deg_s": float(
                max_abs_yaw_rate
            ),

            "min_altitude_ft": float(
                min_alt
            ),

            "max_altitude_ft": float(
                max_alt
            ),

            "max_abs_altitude_deviation_from_300_ft": float(
                max(
                    abs(min_alt - 300.0),
                    abs(max_alt - 300.0),
                )
            ),

            "min_vs_fps": float(
                min_vs
            ),

            "max_vs_fps": float(
                max_vs
            ),

            "min_used_a2": float(
                min_used_a2
            ),

            "max_used_a2": float(
                max_used_a2
            ),

            "min_used_a3": float(
                min_used_a3
            ),

            "max_used_a3": float(
                max_used_a3
            ),

            "min_physical_aileron": float(
                min_phys_ail
            ),

            "max_physical_aileron": float(
                max_phys_ail
            ),

            "min_physical_rudder": float(
                min_phys_rud
            ),

            "max_physical_rudder": float(
                max_phys_rud
            ),

            "final_heading_change_deg": float(
                heading_delta_deg(
                    final["heading_error_deg"],
                    initial["heading_error_deg"],
                )
            ),

            "final_altitude_ft": float(
                final["altitude_ft"]
            ),

            "final_vs_fps": float(
                final["vertical_speed_fps"]
            ),

            "final_roll_deg": float(
                math.degrees(
                    final["roll_rad"]
                )
            ),

            "termination": str(
                termination
            ),

            "safe": bool(
                termination
                ==
                "completed"
            ),
        }, rows

    finally:
        close_handoff(start)


# =====================================================================
# RUN SWEEP
# =====================================================================

print("=" * 120)
print("TRUE MISSION STAGE-4 — TURN COUPLING SWEEP V2")
print("=" * 120)
print("Target direction for this diagnostic: positive heading change.")
print("No descent. No landing. No PPO training.")
print("Collective remains under the locked Stage-3 policy.")
print()

summaries = []
trace_rows = []

for i, cfg in enumerate(CASES, 1):

    print(
        f"[{i}/{len(CASES)}] "
        f"{cfg['name']} | "
        f"dA2={cfg['delta_a2']:+.2f} "
        f"dA3={cfg['delta_a3']:+.2f}"
    )

    summary, rows = run_case(cfg)

    summaries.append(summary)
    trace_rows.extend(rows)

    print(
        f"  pulse ΔHDG="
        f"{summary['pulse_heading_change_deg']:+.4f} deg | "
        f"yawRate_end="
        f"{summary['pulse_end_yaw_rate_deg_s']:+.3f} deg/s | "
        f"roll_end="
        f"{summary['pulse_end_roll_deg']:+.3f} deg | "
        f"max|roll|="
        f"{summary['max_abs_roll_deg']:.3f} deg"
    )

    print(
        f"  pulse ΔALT="
        f"{summary['pulse_altitude_change_ft']:+.4f} ft | "
        f"max|ALT-300|="
        f"{summary['max_abs_altitude_deviation_from_300_ft']:.4f} ft | "
        f"final ΔHDG="
        f"{summary['final_heading_change_deg']:+.4f} deg | "
        f"term={summary['termination']}"
    )

    print()


save_csv(
    RESULT_DIR / "case_summary.csv",
    summaries,
)

save_csv(
    RESULT_DIR / "full_trace.csv",
    trace_rows,
)


# =====================================================================
# BASELINE SUBTRACTION / CANDIDATE SELECTION
# =====================================================================

baseline = next(
    s for s in summaries
    if s["case"] == "baseline"
)

effects = []

for s in summaries:

    net_heading = (
        s["pulse_heading_change_deg"]
        -
        baseline["pulse_heading_change_deg"]
    )

    net_alt = (
        s["pulse_altitude_change_ft"]
        -
        baseline["pulse_altitude_change_ft"]
    )

    net_vs = (
        s["pulse_end_vs_fps"]
        -
        baseline["pulse_end_vs_fps"]
    )

    net_roll = (
        s["pulse_end_roll_deg"]
        -
        baseline["pulse_end_roll_deg"]
    )

    useful = bool(
        s["safe"]
        and
        net_heading
        >=
        MIN_USEFUL_NET_HEADING_DEG
        and
        s[
            "max_abs_altitude_deviation_from_300_ft"
        ]
        <=
        MAX_USEFUL_ALT_DEVIATION_FT
        and
        s["max_abs_roll_deg"]
        <=
        MAX_USEFUL_ABS_ROLL_DEG
    )

    effects.append({
        "case": s["case"],
        "delta_a2": s["delta_a2"],
        "delta_a3": s["delta_a3"],

        "net_heading_effect_deg": float(
            net_heading
        ),

        "net_roll_end_effect_deg": float(
            net_roll
        ),

        "net_altitude_effect_ft": float(
            net_alt
        ),

        "net_vs_end_effect_fps": float(
            net_vs
        ),

        "max_abs_roll_deg": float(
            s["max_abs_roll_deg"]
        ),

        "max_abs_altitude_deviation_from_300_ft": float(
            s[
                "max_abs_altitude_deviation_from_300_ft"
            ]
        ),

        "safe": bool(
            s["safe"]
        ),

        "useful_for_closed_loop_turn": bool(
            useful
        ),
    })


save_csv(
    RESULT_DIR
    /
    "baseline_subtracted_effects.csv",
    effects,
)


print("=" * 120)
print("BASELINE-SUBTRACTED COUPLING RESULTS")
print("=" * 120)

for e in effects:

    print(
        f"{e['case']:14s} | "
        f"dA2={e['delta_a2']:+.2f} "
        f"dA3={e['delta_a3']:+.2f} | "
        f"net ΔHDG="
        f"{e['net_heading_effect_deg']:+.4f} deg | "
        f"net roll="
        f"{e['net_roll_end_effect_deg']:+.4f} deg | "
        f"net ΔALT="
        f"{e['net_altitude_effect_ft']:+.4f} ft | "
        f"net ΔVS="
        f"{e['net_vs_end_effect_fps']:+.4f} ft/s | "
        f"useful={e['useful_for_closed_loop_turn']}"
    )


useful = [
    e for e in effects
    if e[
        "useful_for_closed_loop_turn"
    ]
]

if useful:
    selected = max(
        useful,
        key=lambda x: (
            x["net_heading_effect_deg"],
            -abs(
                x["net_altitude_effect_ft"]
            ),
            -x["max_abs_roll_deg"],
        ),
    )
else:
    selected = None


all_safe = bool(
    all(
        s["safe"]
        for s in summaries
    )
)

positive_authority_monotonic = bool(
    next(
        e["net_heading_effect_deg"]
        for e in effects
        if e["case"] == "rudder_m030"
    )
    >
    0.0
    and
    next(
        e["net_heading_effect_deg"]
        for e in effects
        if e["case"] == "rudder_m045"
    )
    >
    next(
        e["net_heading_effect_deg"]
        for e in effects
        if e["case"] == "rudder_m030"
    )
    and
    next(
        e["net_heading_effect_deg"]
        for e in effects
        if e["case"] == "rudder_m060"
    )
    >
    next(
        e["net_heading_effect_deg"]
        for e in effects
        if e["case"] == "rudder_m045"
    )
)


ready = bool(
    all_safe
    and
    selected is not None
)


print()
print("ALL CASES SAFE:", all_safe)

print(
    "POSITIVE RUDDER AUTHORITY MONOTONIC:",
    positive_authority_monotonic,
)

print(
    "SELECTED TURN-COUPLING CASE:",
    (
        selected["case"]
        if selected is not None
        else "NONE"
    ),
)

print(
    "READY FOR CLOSED-LOOP 10-DEG TURN:",
    ready,
)


final_summary = {
    "training_type": "NONE",
    "descent_used": False,
    "landing_used": False,
    "target_direction": (
        "positive relative heading"
    ),
    "pulse_seconds": float(
        PULSE_SECONDS
    ),
    "recovery_seconds": float(
        RECOVERY_SECONDS
    ),
    "all_cases_safe": bool(
        all_safe
    ),
    "positive_rudder_authority_monotonic": bool(
        positive_authority_monotonic
    ),
    "selected_case": selected,
    "ready_for_closed_loop_10deg_turn": bool(
        ready
    ),
    "cases": summaries,
    "baseline_subtracted": effects,
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
        final_summary,
        f,
        indent=2,
    )


print()
print("Saved:")
print(" ", RESULT_DIR / "case_summary.csv")
print(" ", RESULT_DIR / "baseline_subtracted_effects.csv")
print(" ", RESULT_DIR / "full_trace.csv")
print(" ", RESULT_DIR / "final_summary.json")

print()
print("Next only after reading this result:")
print(
    "build a closed-loop 10-degree heading-change teacher, "
    "then scale the qualified controller to 45 degrees."
)
