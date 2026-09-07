from pathlib import Path
import csv
import json
import math

import numpy as np


# =====================================================================
# AH-1S / JSBSim
# TRUE MISSION STAGE-4 TURN AUTHORITY IDENTIFICATION V1
# =====================================================================
#
# REAL MISSION:
#   Stage 1 : takeoff + climb to 300 ft
#   Stage 2 : forward flight
#   Stage 3 : stop / stabilize near 300 ft endpoint
#   Stage 4 : NEW — 45 degree turn while holding ~300 ft
#   Stage 5 : NEW — forward flight on new heading
#
# THIS SCRIPT:
#   - DOES NOT DESCEND
#   - DOES NOT LAND
#   - DOES NOT TRAIN PPO
#   - DOES NOT BUILD A TURN CONTROLLER YET
#
# It only identifies the local turn authority of:
#   action[2] = lateral / aileron channel
#   action[3] = yaw / rudder channel
#
# Each experiment:
#   1) rebuilds the locked continuous Stage1 -> Stage2 -> Stage3 mission,
#   2) waits for the already-qualified 5 s Stage-3 endpoint hover,
#   3) applies a small temporary normalized-action delta for 1.5 s,
#   4) returns control to the locked Stage-3 policy for 6 s,
#   5) records heading / roll / yaw-rate / altitude effects.
#
# =====================================================================


BASE_SOURCE = Path(
    "diagnose_stage4_entry_margin_v3.py"
)

RESULT_DIR = Path(
    "results_stage4_turn_authority_v1"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Load ONLY the definitions from the already-qualified mission source.
#
# Important:
# diagnose_stage4_entry_margin_v3.py has executable experiments at its
# bottom and no __main__ guard. Importing it normally would rerun those
# old Stage-4 experiments. Therefore we execute only the definitions
# above its experiment section.
# ---------------------------------------------------------------------

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
        "Could not locate experiment marker in "
        "diagnose_stage4_entry_margin_v3.py"
    )

prefix = source_text.split(
    MARKER,
    1,
)[0]

ns = {
    "__name__": "stage4_turn_authority_base",
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


# Locked helpers / models.
build_stage4_handoff = ns[
    "build_stage4_handoff"
]

close_handoff = ns[
    "close_handoff"
]

snapshot = ns[
    "snapshot"
]

stage3_observation = ns[
    "stage3_observation"
]

raw_policy_cycle = ns[
    "raw_policy_cycle"
]

env_control_dt = ns[
    "env_control_dt"
]

stage3_model = ns[
    "stage3_model"
]


# =====================================================================
# IDENTIFICATION SETTINGS
# =====================================================================

PULSE_SECONDS = 1.50
RECOVERY_SECONDS = 6.00

# First-pass normalized-action perturbation.
# This is deliberately small relative to the full [-1,+1] policy range.
DELTA = 0.15

CASES = [
    {
        "name": "baseline",
        "delta_a2": 0.0,
        "delta_a3": 0.0,
    },
    {
        "name": "aileron_plus",
        "delta_a2": +DELTA,
        "delta_a3": 0.0,
    },
    {
        "name": "aileron_minus",
        "delta_a2": -DELTA,
        "delta_a3": 0.0,
    },
    {
        "name": "rudder_plus",
        "delta_a2": 0.0,
        "delta_a3": +DELTA,
    },
    {
        "name": "rudder_minus",
        "delta_a2": 0.0,
        "delta_a3": -DELTA,
    },
]


# Conservative diagnostic abort envelope.
# These are NOT final mission acceptance criteria.
SAFE_ALT_MIN_FT = 288.0
SAFE_ALT_MAX_FT = 312.0
SAFE_MAX_ABS_ROLL_DEG = 12.0
SAFE_MAX_ABS_PITCH_DEG = 10.0
SAFE_MAX_ABS_YAW_RATE_DEG_S = 15.0


def wrap_deg(x):
    return float(
        (
            float(x)
            +
            180.0
        )
        %
        360.0
        -
        180.0
    )


def heading_delta_deg(
    current_heading_error_deg,
    initial_heading_error_deg,
):
    return wrap_deg(
        float(
            current_heading_error_deg
        )
        -
        float(
            initial_heading_error_deg
        )
    )


def diagnostic_safety_reason(
    state,
):
    roll_deg = math.degrees(
        state[
            "roll_rad"
        ]
    )

    pitch_deg = math.degrees(
        state[
            "pitch_rad"
        ]
    )

    yaw_rate_deg_s = math.degrees(
        state[
            "yaw_rate_rad_s"
        ]
    )

    if (
        state[
            "altitude_ft"
        ]
        <
        SAFE_ALT_MIN_FT
    ):
        return "altitude_low"

    if (
        state[
            "altitude_ft"
        ]
        >
        SAFE_ALT_MAX_FT
    ):
        return "altitude_high"

    if (
        abs(
            roll_deg
        )
        >
        SAFE_MAX_ABS_ROLL_DEG
    ):
        return "roll_limit"

    if (
        abs(
            pitch_deg
        )
        >
        SAFE_MAX_ABS_PITCH_DEG
    ):
        return "pitch_limit"

    if (
        abs(
            yaw_rate_deg_s
        )
        >
        SAFE_MAX_ABS_YAW_RATE_DEG_S
    ):
        return "yaw_rate_limit"

    return ""


def state_row(
    case_name,
    phase,
    time_s,
    state,
    initial,
    base_action,
    used_action,
):
    return {
        "case": str(
            case_name
        ),
        "phase": str(
            phase
        ),
        "time_s": float(
            time_s
        ),

        "altitude_ft": float(
            state[
                "altitude_ft"
            ]
        ),

        "altitude_error_from_300_ft": float(
            state[
                "altitude_ft"
            ]
            -
            300.0
        ),

        "vertical_speed_fps": float(
            state[
                "vertical_speed_fps"
            ]
        ),

        "heading_error_deg": float(
            state[
                "heading_error_deg"
            ]
        ),

        "heading_change_from_start_deg": float(
            heading_delta_deg(
                state[
                    "heading_error_deg"
                ],
                initial[
                    "heading_error_deg"
                ],
            )
        ),

        "roll_deg": float(
            math.degrees(
                state[
                    "roll_rad"
                ]
            )
        ),

        "pitch_deg": float(
            math.degrees(
                state[
                    "pitch_rad"
                ]
            )
        ),

        "roll_rate_deg_s": float(
            math.degrees(
                state[
                    "roll_rate_rad_s"
                ]
            )
        ),

        "yaw_rate_deg_s": float(
            math.degrees(
                state[
                    "yaw_rate_rad_s"
                ]
            )
        ),

        "forward_ft": float(
            state[
                "forward_ft"
            ]
        ),

        "forward_speed_fps": float(
            state[
                "forward_speed_fps"
            ]
        ),

        "cross_track_ft": float(
            state[
                "cross_track_ft"
            ]
        ),

        "lateral_speed_fps": float(
            state[
                "lateral_speed_fps"
            ]
        ),

        "physical_collective_cmd": float(
            state[
                "physical_collective_cmd"
            ]
        ),

        "physical_elevator_cmd": float(
            state[
                "physical_elevator_cmd"
            ]
        ),

        "physical_aileron_cmd": float(
            state[
                "physical_aileron_cmd"
            ]
        ),

        "physical_rudder_cmd": float(
            state[
                "physical_rudder_cmd"
            ]
        ),

        "base_action_0": float(
            base_action[0]
        ),

        "base_action_1": float(
            base_action[1]
        ),

        "base_action_2": float(
            base_action[2]
        ),

        "base_action_3": float(
            base_action[3]
        ),

        "used_action_0": float(
            used_action[0]
        ),

        "used_action_1": float(
            used_action[1]
        ),

        "used_action_2": float(
            used_action[2]
        ),

        "used_action_3": float(
            used_action[3]
        ),
    }


def save_csv(
    path,
    rows,
):
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
        writer.writerows(
            rows
        )


def run_case(
    cfg,
):
    start = build_stage4_handoff(
        detailed=False,
        require_full_hold=True,
    )

    try:
        if not bool(
            start[
                "handoff_pass"
            ]
        ):
            raise RuntimeError(
                "Locked Stage-3 handoff did not pass."
            )

        if not bool(
            start[
                "same_fdm"
            ]
        ):
            raise RuntimeError(
                "FDM continuity failed."
            )

        env2 = start[
            "env2"
        ]

        fdm = start[
            "fdm"
        ]

        lat0 = start[
            "lat0"
        ]

        lon0 = start[
            "lon0"
        ]

        mission_heading = start[
            "mission_heading"
        ]

        dt = env_control_dt(
            env2
        )

        state = snapshot(
            fdm,
            lat0,
            lon0,
            mission_heading,
        )

        initial = state.copy()

        rows = []

        termination = (
            "completed"
        )

        pulse_steps = int(
            round(
                PULSE_SECONDS
                /
                dt
            )
        )

        recovery_steps = int(
            round(
                RECOVERY_SECONDS
                /
                dt
            )
        )

        # Metrics over pulse period.
        max_abs_roll = abs(
            math.degrees(
                state[
                    "roll_rad"
                ]
            )
        )

        max_abs_pitch = abs(
            math.degrees(
                state[
                    "pitch_rad"
                ]
            )
        )

        max_abs_yaw_rate = abs(
            math.degrees(
                state[
                    "yaw_rate_rad_s"
                ]
            )
        )

        min_alt = float(
            state[
                "altitude_ft"
            ]
        )

        max_alt = float(
            state[
                "altitude_ft"
            ]
        )

        min_vs = float(
            state[
                "vertical_speed_fps"
            ]
        )

        max_vs = float(
            state[
                "vertical_speed_fps"
            ]
        )

        pulse_end = (
            state.copy()
        )

        # -------------------------------------------------------------
        # PULSE
        # -------------------------------------------------------------

        for step in range(
            pulse_steps
        ):
            before = snapshot(
                fdm,
                lat0,
                lon0,
                mission_heading,
            )

            obs3 = (
                stage3_observation(
                    before
                )
            )

            base, _ = (
                stage3_model.predict(
                    obs3,
                    deterministic=True,
                )
            )

            base = np.asarray(
                base,
                dtype=np.float32,
            ).reshape(-1)

            action = base.copy()

            action[2] = float(
                np.clip(
                    action[2]
                    +
                    float(
                        cfg[
                            "delta_a2"
                        ]
                    ),
                    -1.0,
                    +1.0,
                )
            )

            action[3] = float(
                np.clip(
                    action[3]
                    +
                    float(
                        cfg[
                            "delta_a3"
                        ]
                    ),
                    -1.0,
                    +1.0,
                )
            )

            state, used = (
                raw_policy_cycle(
                    env2,
                    fdm,
                    action,
                    lat0,
                    lon0,
                    mission_heading,
                )
            )

            t = (
                step
                +
                1
            ) * dt

            rows.append(
                state_row(
                    cfg[
                        "name"
                    ],
                    "pulse",
                    t,
                    state,
                    initial,
                    base,
                    used,
                )
            )

            roll_deg = math.degrees(
                state[
                    "roll_rad"
                ]
            )

            pitch_deg = math.degrees(
                state[
                    "pitch_rad"
                ]
            )

            yaw_rate_deg_s = (
                math.degrees(
                    state[
                        "yaw_rate_rad_s"
                    ]
                )
            )

            max_abs_roll = max(
                max_abs_roll,
                abs(
                    roll_deg
                ),
            )

            max_abs_pitch = max(
                max_abs_pitch,
                abs(
                    pitch_deg
                ),
            )

            max_abs_yaw_rate = max(
                max_abs_yaw_rate,
                abs(
                    yaw_rate_deg_s
                ),
            )

            min_alt = min(
                min_alt,
                state[
                    "altitude_ft"
                ],
            )

            max_alt = max(
                max_alt,
                state[
                    "altitude_ft"
                ],
            )

            min_vs = min(
                min_vs,
                state[
                    "vertical_speed_fps"
                ],
            )

            max_vs = max(
                max_vs,
                state[
                    "vertical_speed_fps"
                ],
            )

            pulse_end = (
                state.copy()
            )

            reason = (
                diagnostic_safety_reason(
                    state
                )
            )

            if reason:
                termination = (
                    "pulse:"
                    +
                    reason
                )
                break

        # -------------------------------------------------------------
        # RECOVERY — LOCKED STAGE-3 POLICY ONLY
        # -------------------------------------------------------------

        if termination == "completed":

            for step in range(
                recovery_steps
            ):
                before = snapshot(
                    fdm,
                    lat0,
                    lon0,
                    mission_heading,
                )

                obs3 = (
                    stage3_observation(
                        before
                    )
                )

                base, _ = (
                    stage3_model.predict(
                        obs3,
                        deterministic=True,
                    )
                )

                base = np.asarray(
                    base,
                    dtype=np.float32,
                ).reshape(-1)

                state, used = (
                    raw_policy_cycle(
                        env2,
                        fdm,
                        base,
                        lat0,
                        lon0,
                        mission_heading,
                    )
                )

                t = (
                    PULSE_SECONDS
                    +
                    (
                        step
                        +
                        1
                    )
                    *
                    dt
                )

                rows.append(
                    state_row(
                        cfg[
                            "name"
                        ],
                        "recovery",
                        t,
                        state,
                        initial,
                        base,
                        used,
                    )
                )

                reason = (
                    diagnostic_safety_reason(
                        state
                    )
                )

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

        pulse_heading_delta = (
            heading_delta_deg(
                pulse_end[
                    "heading_error_deg"
                ],
                initial[
                    "heading_error_deg"
                ],
            )
        )

        final_heading_delta = (
            heading_delta_deg(
                final[
                    "heading_error_deg"
                ],
                initial[
                    "heading_error_deg"
                ],
            )
        )

        summary = {
            "case": str(
                cfg[
                    "name"
                ]
            ),

            "delta_a2": float(
                cfg[
                    "delta_a2"
                ]
            ),

            "delta_a3": float(
                cfg[
                    "delta_a3"
                ]
            ),

            "same_fdm": bool(
                start[
                    "same_fdm"
                ]
            ),

            "stage3_handoff_pass": bool(
                start[
                    "handoff_pass"
                ]
            ),

            "stage3_hover_hold_s": float(
                start[
                    "stage3_hover_hold_s"
                ]
            ),

            "initial_altitude_ft": float(
                initial[
                    "altitude_ft"
                ]
            ),

            "initial_vs_fps": float(
                initial[
                    "vertical_speed_fps"
                ]
            ),

            "initial_heading_error_deg": float(
                initial[
                    "heading_error_deg"
                ]
            ),

            "initial_roll_deg": float(
                math.degrees(
                    initial[
                        "roll_rad"
                    ]
                )
            ),

            "pulse_heading_change_deg": float(
                pulse_heading_delta
            ),

            "pulse_end_roll_deg": float(
                math.degrees(
                    pulse_end[
                        "roll_rad"
                    ]
                )
            ),

            "pulse_end_yaw_rate_deg_s": float(
                math.degrees(
                    pulse_end[
                        "yaw_rate_rad_s"
                    ]
                )
            ),

            "pulse_end_altitude_ft": float(
                pulse_end[
                    "altitude_ft"
                ]
            ),

            "pulse_end_vs_fps": float(
                pulse_end[
                    "vertical_speed_fps"
                ]
            ),

            "pulse_altitude_change_ft": float(
                pulse_end[
                    "altitude_ft"
                ]
                -
                initial[
                    "altitude_ft"
                ]
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

            "min_vs_fps": float(
                min_vs
            ),

            "max_vs_fps": float(
                max_vs
            ),

            "final_heading_change_deg": float(
                final_heading_delta
            ),

            "final_altitude_ft": float(
                final[
                    "altitude_ft"
                ]
            ),

            "final_vs_fps": float(
                final[
                    "vertical_speed_fps"
                ]
            ),

            "final_roll_deg": float(
                math.degrees(
                    final[
                        "roll_rad"
                    ]
                )
            ),

            "final_cross_track_ft": float(
                final[
                    "cross_track_ft"
                ]
            ),

            "termination": str(
                termination
            ),

            "safe": bool(
                termination
                ==
                "completed"
            ),
        }

        return (
            summary,
            rows,
        )

    finally:
        close_handoff(
            start
        )


# =====================================================================
# RUN
# =====================================================================

print(
    "=" * 120
)

print(
    "TRUE MISSION STAGE-4 — TURN AUTHORITY IDENTIFICATION V1"
)

print(
    "=" * 120
)

print(
    "No descent. No landing. No training."
)

print(
    "Each case starts from the locked Stage1->Stage2->Stage3 "
    "300-ft endpoint hover."
)

print()

all_summaries = []
all_trace_rows = []

for i, cfg in enumerate(
    CASES,
    start=1,
):
    print(
        f"[{i}/{len(CASES)}] "
        f"{cfg['name']} | "
        f"dA2={cfg['delta_a2']:+.3f} "
        f"dA3={cfg['delta_a3']:+.3f}"
    )

    summary, rows = (
        run_case(
            cfg
        )
    )

    all_summaries.append(
        summary
    )

    all_trace_rows.extend(
        rows
    )

    print(
        f"  pulse ΔHDG="
        f"{summary['pulse_heading_change_deg']:+.4f} deg | "
        f"roll_end="
        f"{summary['pulse_end_roll_deg']:+.3f} deg | "
        f"max|roll|="
        f"{summary['max_abs_roll_deg']:.3f} deg | "
        f"yawRate_end="
        f"{summary['pulse_end_yaw_rate_deg_s']:+.3f} deg/s"
    )

    print(
        f"  pulse ΔALT="
        f"{summary['pulse_altitude_change_ft']:+.4f} ft | "
        f"VS_end="
        f"{summary['pulse_end_vs_fps']:+.4f} ft/s | "
        f"final ΔHDG="
        f"{summary['final_heading_change_deg']:+.4f} deg | "
        f"term={summary['termination']}"
    )

    print()


save_csv(
    RESULT_DIR
    /
    "case_summary.csv",
    all_summaries,
)

save_csv(
    RESULT_DIR
    /
    "full_trace.csv",
    all_trace_rows,
)


# ---------------------------------------------------------------------
# Compare each pulse against the baseline pulse.
# ---------------------------------------------------------------------

baseline = next(
    x
    for x in all_summaries
    if x[
        "case"
    ]
    ==
    "baseline"
)

comparison = []

for s in all_summaries:

    comparison.append({
        "case": s[
            "case"
        ],

        "delta_a2": s[
            "delta_a2"
        ],

        "delta_a3": s[
            "delta_a3"
        ],

        "net_heading_effect_vs_baseline_deg": float(
            s[
                "pulse_heading_change_deg"
            ]
            -
            baseline[
                "pulse_heading_change_deg"
            ]
        ),

        "net_roll_end_effect_vs_baseline_deg": float(
            s[
                "pulse_end_roll_deg"
            ]
            -
            baseline[
                "pulse_end_roll_deg"
            ]
        ),

        "net_altitude_effect_vs_baseline_ft": float(
            s[
                "pulse_altitude_change_ft"
            ]
            -
            baseline[
                "pulse_altitude_change_ft"
            ]
        ),

        "net_vs_end_effect_vs_baseline_fps": float(
            s[
                "pulse_end_vs_fps"
            ]
            -
            baseline[
                "pulse_end_vs_fps"
            ]
        ),

        "safe": bool(
            s[
                "safe"
            ]
        ),
    })


save_csv(
    RESULT_DIR
    /
    "baseline_subtracted_effects.csv",
    comparison,
)


print(
    "=" * 120
)

print(
    "BASELINE-SUBTRACTED TURN AUTHORITY"
)

print(
    "=" * 120
)

for c in comparison:

    print(
        f"{c['case']:14s} | "
        f"dA2={c['delta_a2']:+.3f} "
        f"dA3={c['delta_a3']:+.3f} | "
        f"net ΔHDG="
        f"{c['net_heading_effect_vs_baseline_deg']:+.5f} deg | "
        f"net roll="
        f"{c['net_roll_end_effect_vs_baseline_deg']:+.5f} deg | "
        f"net ΔALT="
        f"{c['net_altitude_effect_vs_baseline_ft']:+.5f} ft | "
        f"net ΔVS="
        f"{c['net_vs_end_effect_vs_baseline_fps']:+.5f} ft/s | "
        f"safe={c['safe']}"
    )


safe_cases = bool(
    all(
        x[
            "safe"
        ]
        for x in all_summaries
    )
)

nonzero_heading_response = bool(
    max(
        abs(
            c[
                "net_heading_effect_vs_baseline_deg"
            ]
        )
        for c in comparison
        if c[
            "case"
        ]
        !=
        "baseline"
    )
    >
    0.01
)

ready = bool(
    safe_cases
    and
    nonzero_heading_response
)

summary = {
    "training_type": "NONE",
    "mission_stage": (
        "NEW Stage 4: 45-degree turn at ~300 ft"
    ),
    "descent_used": False,
    "landing_used": False,
    "pulse_seconds": float(
        PULSE_SECONDS
    ),
    "recovery_seconds": float(
        RECOVERY_SECONDS
    ),
    "normalized_action_delta": float(
        DELTA
    ),
    "cases": all_summaries,
    "baseline_subtracted": comparison,
    "all_cases_safe": bool(
        safe_cases
    ),
    "measurable_heading_authority": bool(
        nonzero_heading_response
    ),
    "ready_for_turn_coupling_sweep": bool(
        ready
    ),
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

print(
    "ALL CASES SAFE:",
    safe_cases
)

print(
    "MEASURABLE HEADING AUTHORITY:",
    nonzero_heading_response
)

print(
    "READY FOR TURN COUPLING SWEEP:",
    ready
)

print()

print(
    "Saved:"
)

print(
    " ",
    RESULT_DIR
    /
    "case_summary.csv"
)

print(
    " ",
    RESULT_DIR
    /
    "baseline_subtracted_effects.csv"
)

print(
    " ",
    RESULT_DIR
    /
    "full_trace.csv"
)

print(
    " ",
    RESULT_DIR
    /
    "final_summary.json"
)

print()

print(
    "Next step only after reading this result:"
)

print(
    "identify the correct turn direction and test coordinated "
    "aileron+rudder+collective compensation."
)
