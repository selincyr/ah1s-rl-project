from __future__ import annotations

import csv
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
# LOCKED MODELS
# ============================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL_PATH = (
    "models_stage2_distilled/"
    "AH1S_STAGE2_DISTILLED_SUCCESS.zip"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path(
    "results_stage3_braking_identification"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = OUT_DIR / "elevator_pulse_summary.csv"
BEST_TRACE_CSV = OUT_DIR / "best_elevator_pulse_trace.csv"


# ============================================================
# CONSTANTS
# ============================================================

TARGET_DISTANCE = 300.0

STAGE1_MAX_TIME = 120.0
STAGE2_MAX_TIME = 55.0
HANDOFF_STABLE_TIME = 5.0

AILERON_SCALE = 0.026
RUDDER_SCALE = 0.040

# One Stage-2 control cycle is ~0.075 s.
CONTROL_DT = 0.075

# Keep the pulse short enough to be an identification experiment,
# not a hand-written braking controller.
PULSE_SECONDS = 1.00
RECOVERY_SECONDS = 1.50

# Test normalized elevator RESIDUALS around the actual Stage-2
# endpoint PPO action[1].
ELEVATOR_DELTAS = [
    -0.50,
    -0.35,
    -0.20,
    -0.10,
     0.00,
    +0.10,
    +0.20,
    +0.35,
    +0.50,
]

# Conservative safety envelope for this identification only.
ALT_MIN = 285.0
ALT_MAX = 315.0
MAX_ABS_PITCH_DEG = 15.0
MAX_ABS_ROLL_DEG = 10.0
MIN_FORWARD_SPEED_FPS = -5.0


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


def get_fdm(env):
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


def print_rule(text):
    print()
    print("=" * 136)
    print(text)
    print("=" * 136)


def run_physics_cycle(
    env2,
    action,
):
    """
    Apply one Stage-2 action using the repaired mapping, then advance
    JSBSim by the same 10 raw physics steps used by the environment.

    We deliberately bypass env.step() after the 300-ft endpoint so
    Stage-2 success/termination logic cannot stop the pulse experiment.
    """

    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    action = np.clip(
        action,
        -1.0,
        +1.0,
    )

    env2._apply_action(
        action
    )

    for _ in range(10):
        if not env2.fdm.run():
            raise RuntimeError(
                "JSBSim stopped during braking identification."
            )


def state_snapshot(
    fdm,
):
    return {
        "altitude_ft":
            fdm_float(
                fdm,
                "position/h-agl-ft",
            ),

        "forward_velocity_fps":
            fdm_float(
                fdm,
                "velocities/u-aero-fps",
            ),

        "lateral_velocity_fps":
            fdm_float(
                fdm,
                "velocities/v-aero-fps",
            ),

        "vertical_speed_fps":
            fdm_float(
                fdm,
                "velocities/h-dot-fps",
            ),

        "pitch_deg":
            math.degrees(
                fdm_float(
                    fdm,
                    "attitude/pitch-rad",
                    0.0,
                )
            ),

        "roll_deg":
            math.degrees(
                fdm_float(
                    fdm,
                    "attitude/roll-rad",
                    0.0,
                )
            ),

        "pitch_rate_rad_s":
            fdm_float(
                fdm,
                "velocities/q-rad_sec",
                0.0,
            ),

        "physical_collective":
            fdm_float(
                fdm,
                "fcs/collective-cmd-norm",
            ),

        "physical_elevator":
            fdm_float(
                fdm,
                "fcs/elevator-cmd-norm",
            ),

        "physical_aileron":
            fdm_float(
                fdm,
                "fcs/aileron-cmd-norm",
            ),

        "physical_rudder":
            fdm_float(
                fdm,
                "fcs/rudder-cmd-norm",
            ),
    }


def safety_ok(state):
    return bool(
        ALT_MIN
        <=
        state["altitude_ft"]
        <=
        ALT_MAX

        and
        abs(
            state["pitch_deg"]
        )
        <=
        MAX_ABS_PITCH_DEG

        and
        abs(
            state["roll_deg"]
        )
        <=
        MAX_ABS_ROLL_DEG

        and
        state["forward_velocity_fps"]
        >=
        MIN_FORWARD_SPEED_FPS
    )


# ============================================================
# LOAD MODELS ONCE
# ============================================================

stage1_model = PPO.load(
    STAGE1_MODEL_PATH
)

stage2_model = PPO.load(
    STAGE2_MODEL_PATH
)


# ============================================================
# BUILD EXACT LOCKED STAGE-2 ENDPOINT
# ============================================================

def build_endpoint():
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

    active_id = id(
        fdm
    )

    mission_heading = fdm_float(
        fdm,
        "attitude/heading-true-rad",
    )

    dt1 = float(
        getattr(
            env1,
            "dt",
            CONTROL_DT,
        )
    )

    if (
        not np.isfinite(dt1)
        or
        dt1 <= 0.0
    ):
        dt1 = CONTROL_DT

    stable_time = 0.0

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

        (
            obs1,
            _,
            terminated,
            truncated,
            info1,
        ) = env1.step(
            action1
        )

        altitude = info_float(
            info1,
            "altitude",
        )

        vs = info_float(
            info1,
            "vertical_speed",
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

        hs = float(
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
            abs(vs)
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
            stable_time + dt1
            if stable
            else 0.0
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
            raise RuntimeError(
                "Stage 1 failed before handoff."
            )

        if truncated:
            raise RuntimeError(
                "Stage 1 truncated before handoff."
            )

    if (
        stable_time
        <
        HANDOFF_STABLE_TIME
    ):
        raise RuntimeError(
            "Stable Stage-1 handoff not reached."
        )

    env2 = (
        HelicopterEnvStage2RefineMapped(
            aileron_scale=
                AILERON_SCALE,

            rudder_scale=
                RUDDER_SCALE,
        )
    )

    env2.reset()

    env2.fdm = (
        fdm
    )

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
        active_id
    ):
        raise RuntimeError(
            "FDM continuity failed."
        )

    obs2 = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    dt2 = float(
        getattr(
            env2,
            "dt",
            CONTROL_DT,
        )
    )

    if (
        not np.isfinite(dt2)
        or
        dt2 <= 0.0
    ):
        dt2 = CONTROL_DT

    last_action = None

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
            terminated,
            truncated,
            info2,
        ) = env2.step(
            action2
        )

        obs2 = np.asarray(
            obs2,
            dtype=np.float32,
        )

        last_action = (
            action2.copy()
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
            terminated
            and
            not bool(
                info2.get(
                    "success",
                    False,
                )
            )
        ):
            raise RuntimeError(
                "Stage 2 failed before endpoint."
            )

        if truncated:
            raise RuntimeError(
                "Stage 2 truncated before endpoint."
            )

    if last_action is None:
        raise RuntimeError(
            "No Stage-2 endpoint action available."
        )

    return (
        env1,
        env2,
        fdm,
        last_action,
        state_snapshot(
            fdm
        ),
    )


# ============================================================
# ONE PULSE CASE
# ============================================================

def run_case(
    elevator_delta,
    detailed=False,
):
    (
        env1,
        env2,
        fdm,
        endpoint_action,
        start_state,
    ) = build_endpoint()

    baseline_action = (
        endpoint_action.copy()
    )

    pulse_action = (
        baseline_action.copy()
    )

    pulse_action[1] = float(
        np.clip(
            baseline_action[1]
            +
            elevator_delta,
            -1.0,
            +1.0,
        )
    )

    trace = []

    pulse_steps = int(
        round(
            PULSE_SECONDS
            /
            CONTROL_DT
        )
    )

    recovery_steps = int(
        round(
            RECOVERY_SECONDS
            /
            CONTROL_DT
        )
    )

    min_forward = (
        start_state[
            "forward_velocity_fps"
        ]
    )

    max_forward = (
        start_state[
            "forward_velocity_fps"
        ]
    )

    min_alt = (
        start_state[
            "altitude_ft"
        ]
    )

    max_alt = (
        start_state[
            "altitude_ft"
        ]
    )

    max_abs_pitch = abs(
        start_state[
            "pitch_deg"
        ]
    )

    safe = True

    # --------------------------------------------------------
    # PULSE
    # --------------------------------------------------------

    for step in range(
        pulse_steps
    ):
        run_physics_cycle(
            env2,
            pulse_action,
        )

        state = state_snapshot(
            fdm
        )

        t = (
            step + 1
        ) * CONTROL_DT

        state[
            "time_s"
        ] = float(
            t
        )

        state[
            "phase"
        ] = "pulse"

        trace.append(
            state.copy()
        )

        min_forward = min(
            min_forward,
            state[
                "forward_velocity_fps"
            ],
        )

        max_forward = max(
            max_forward,
            state[
                "forward_velocity_fps"
            ],
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

        max_abs_pitch = max(
            max_abs_pitch,
            abs(
                state[
                    "pitch_deg"
                ]
            ),
        )

        if not safety_ok(
            state
        ):
            safe = False
            break

    pulse_end_state = state_snapshot(
        fdm
    )

    # --------------------------------------------------------
    # RECOVERY AT EXACT ENDPOINT BASE ACTION
    # --------------------------------------------------------

    if safe:
        for step in range(
            recovery_steps
        ):
            run_physics_cycle(
                env2,
                baseline_action,
            )

            state = state_snapshot(
                fdm
            )

            t = (
                PULSE_SECONDS
                +
                (
                    step + 1
                )
                *
                CONTROL_DT
            )

            state[
                "time_s"
            ] = float(
                t
            )

            state[
                "phase"
            ] = "recovery"

            trace.append(
                state.copy()
            )

            min_forward = min(
                min_forward,
                state[
                    "forward_velocity_fps"
                ],
            )

            max_forward = max(
                max_forward,
                state[
                    "forward_velocity_fps"
                ],
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

            max_abs_pitch = max(
                max_abs_pitch,
                abs(
                    state[
                        "pitch_deg"
                    ]
                ),
            )

            if not safety_ok(
                state
            ):
                safe = False
                break

    final_state = state_snapshot(
        fdm
    )

    # Positive means actual braking.
    pulse_braking = (
        start_state[
            "forward_velocity_fps"
        ]
        -
        pulse_end_state[
            "forward_velocity_fps"
        ]
    )

    total_braking = (
        start_state[
            "forward_velocity_fps"
        ]
        -
        final_state[
            "forward_velocity_fps"
        ]
    )

    result = {
        "elevator_delta":
            float(
                elevator_delta
            ),

        "baseline_action1":
            float(
                baseline_action[1]
            ),

        "used_action1":
            float(
                pulse_action[1]
            ),

        "safe":
            bool(
                safe
            ),

        "start_forward_fps":
            float(
                start_state[
                    "forward_velocity_fps"
                ]
            ),

        "pulse_end_forward_fps":
            float(
                pulse_end_state[
                    "forward_velocity_fps"
                ]
            ),

        "final_forward_fps":
            float(
                final_state[
                    "forward_velocity_fps"
                ]
            ),

        "pulse_braking_fps":
            float(
                pulse_braking
            ),

        "total_braking_fps":
            float(
                total_braking
            ),

        "min_forward_fps":
            float(
                min_forward
            ),

        "max_forward_fps":
            float(
                max_forward
            ),

        "start_altitude_ft":
            float(
                start_state[
                    "altitude_ft"
                ]
            ),

        "min_altitude_ft":
            float(
                min_alt
            ),

        "max_altitude_ft":
            float(
                max_alt
            ),

        "final_altitude_ft":
            float(
                final_state[
                    "altitude_ft"
                ]
            ),

        "start_pitch_deg":
            float(
                start_state[
                    "pitch_deg"
                ]
            ),

        "pulse_end_pitch_deg":
            float(
                pulse_end_state[
                    "pitch_deg"
                ]
            ),

        "final_pitch_deg":
            float(
                final_state[
                    "pitch_deg"
                ]
            ),

        "max_abs_pitch_deg":
            float(
                max_abs_pitch
            ),

        "pulse_end_vs_fps":
            float(
                pulse_end_state[
                    "vertical_speed_fps"
                ]
            ),

        "final_vs_fps":
            float(
                final_state[
                    "vertical_speed_fps"
                ]
            ),

        "physical_elevator_start":
            float(
                start_state[
                    "physical_elevator"
                ]
            ),

        "physical_elevator_pulse_end":
            float(
                pulse_end_state[
                    "physical_elevator"
                ]
            ),

        "physical_elevator_final":
            float(
                final_state[
                    "physical_elevator"
                ]
            ),

        "trace":
            trace,
    }

    if detailed:
        print()
        print(
            f"Detailed case delta={elevator_delta:+.3f}"
        )

        for row in trace:
            print(
                f"t={row['time_s']:5.2f}s | "
                f"{row['phase']:8s} | "
                f"FWD={row['forward_velocity_fps']:+7.3f} | "
                f"ALT={row['altitude_ft']:7.3f} | "
                f"VS={row['vertical_speed_fps']:+6.3f} | "
                f"PITCH={row['pitch_deg']:+7.3f}deg | "
                f"ELE={row['physical_elevator']:+.6f}"
            )

    env2.fdm = None
    env1.close()

    return result


# ============================================================
# RUN SEARCH
# ============================================================

print_rule(
    "STAGE 3 — ELEVATOR BRAKING AUTHORITY IDENTIFICATION"
)

print(
    "NO TRAINING."
)

print(
    "Every case starts from a fresh locked Stage1->Stage2 endpoint."
)

print(
    "Only elevator action[1] is pulsed."
)

print(
    f"Pulse: {PULSE_SECONDS:.2f}s | "
    f"Recovery: {RECOVERY_SECONDS:.2f}s"
)

print()

results = []

for index, delta in enumerate(
    ELEVATOR_DELTAS,
    start=1,
):
    result = run_case(
        delta,
        detailed=False,
    )

    results.append(
        result
    )

    print(
        f"E{index:02d} | "
        f"dA1={delta:+.2f} | "
        f"A1={result['used_action1']:+.3f} | "
        f"FWD {result['start_forward_fps']:6.2f}"
        f" -> {result['pulse_end_forward_fps']:6.2f}"
        f" -> {result['final_forward_fps']:6.2f} | "
        f"BRAKE pulse={result['pulse_braking_fps']:+6.2f} "
        f"total={result['total_braking_fps']:+6.2f} | "
        f"PITCH_END={result['pulse_end_pitch_deg']:+6.2f}deg | "
        f"ALT=[{result['min_altitude_ft']:.2f},"
        f"{result['max_altitude_ft']:.2f}] | "
        f"SAFE={result['safe']}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_rows = []

for result in results:
    row = {
        key:
            value
        for key, value
        in result.items()
        if key != "trace"
    }

    summary_rows.append(
        row
    )

with SUMMARY_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


# ============================================================
# RANK SAFE BRAKING CASES
# ============================================================

safe_results = [
    r
    for r in results
    if r[
        "safe"
    ]
]

if not safe_results:
    raise RuntimeError(
        "No safe elevator pulse case found."
    )


def rank_key(r):
    # Prefer strong braking, then smaller pitch/altitude disturbance.
    altitude_excursion = max(
        abs(
            r[
                "min_altitude_ft"
            ]
            -
            r[
                "start_altitude_ft"
            ]
        ),
        abs(
            r[
                "max_altitude_ft"
            ]
            -
            r[
                "start_altitude_ft"
            ]
        ),
    )

    return (
        -r[
            "pulse_braking_fps"
        ],
        -r[
            "total_braking_fps"
        ],
        r[
            "max_abs_pitch_deg"
        ],
        altitude_excursion,
    )


safe_results.sort(
    key=rank_key
)

print_rule(
    "TOP SAFE BRAKING PULSES"
)

for rank, r in enumerate(
    safe_results[:5],
    start=1,
):
    print(
        f"{rank}. "
        f"dA1={r['elevator_delta']:+.2f} | "
        f"A1={r['used_action1']:+.3f} | "
        f"PULSE_BRAKE={r['pulse_braking_fps']:+.3f} fps | "
        f"TOTAL_BRAKE={r['total_braking_fps']:+.3f} fps | "
        f"PITCH_END={r['pulse_end_pitch_deg']:+.3f}deg | "
        f"ALT_FINAL={r['final_altitude_ft']:.3f}ft"
    )


best = safe_results[0]

print_rule(
    "BEST IDENTIFICATION CASE — DETAILED REPEAT"
)

best_detailed = run_case(
    best[
        "elevator_delta"
    ],
    detailed=True,
)

with BEST_TRACE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            best_detailed[
                "trace"
            ][0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        best_detailed[
            "trace"
        ]
    )


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print_rule(
    "STAGE 3 BRAKING AUTHORITY RESULT"
)

print(
    "Baseline Stage-2 endpoint forward speed:",
    f"{best_detailed['start_forward_fps']:.3f} ft/s"
)

print(
    "Best elevator residual:",
    f"{best_detailed['elevator_delta']:+.3f}"
)

print(
    "Used normalized elevator action:",
    f"{best_detailed['used_action1']:+.3f}"
)

print(
    "Physical elevator at pulse end:",
    f"{best_detailed['physical_elevator_pulse_end']:+.6f}"
)

print(
    "Forward speed after pulse:",
    f"{best_detailed['pulse_end_forward_fps']:.3f} ft/s"
)

print(
    "Forward speed after recovery:",
    f"{best_detailed['final_forward_fps']:.3f} ft/s"
)

print(
    "Pulse braking:",
    f"{best_detailed['pulse_braking_fps']:+.3f} ft/s"
)

print(
    "Total braking:",
    f"{best_detailed['total_braking_fps']:+.3f} ft/s"
)

print(
    "Pitch at pulse end:",
    f"{best_detailed['pulse_end_pitch_deg']:+.3f} deg"
)

print(
    "Altitude range:",
    f"{best_detailed['min_altitude_ft']:.3f}"
    f" .. "
    f"{best_detailed['max_altitude_ft']:.3f} ft"
)

print(
    "Safe:",
    best_detailed[
        "safe"
    ]
)

print()

positive = [
    r
    for r in safe_results
    if r[
        "pulse_braking_fps"
    ]
    >
    0.25
]

negative = [
    r
    for r in safe_results
    if r[
        "pulse_braking_fps"
    ]
    <
    -0.25
]

if positive:
    print(
        "BRAKING DIRECTION IDENTIFIED."
    )

    print(
        "Next: build a Stage-3 brake-to-hover teacher using the "
        "identified elevator sign, with forward-speed and pitch-rate damping."
    )

else:
    print(
        "BRAKING DIRECTION NOT STRONG ENOUGH YET."
    )

    print(
        "Next: extend pulse duration slightly before designing a teacher."
    )

print()
print(
    "Saved:",
    SUMMARY_CSV
)

print(
    "Saved:",
    BEST_TRACE_CSV
)
