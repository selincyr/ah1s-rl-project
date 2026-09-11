import json
import os
import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill


# ============================================================
# FINAL STAGE-1 PPO
# ============================================================

MODEL_PATH = (
    "models_stage1_final_distilled/"
    "AH1S_STAGE1_FINAL_DISTILLED.zip"
)

model = PPO.load(MODEL_PATH)


# ============================================================
# IDENTIFIED XY CONTROL EFFECTIVENESS
# ============================================================

B_INV = np.array(
    [
        [-0.20338265, -0.00859620],
        [ 0.01952073, -0.14631468],
    ],
    dtype=np.float64,
)

CYCLIC_AUTHORITY = 0.026


# ============================================================
# BEST V3 DIRECT CYCLIC TEACHER
# ============================================================

XY_KP = 0.000
XY_KD = 0.50
XY_ALPHA = 0.90

FF_ELE = -0.026
FF_AIL = -0.026
FF_HOLD = 0.75
FF_FADE = 1.50

TEACHER_FULL_ALT = 100.0
TEACHER_OFF_ALT = 140.0


# ============================================================
# V4 IDEA
#
# V3 showed that early cyclic is authority-limited:
# teacher is already near +/-1 normalized cyclic.
#
# So V4 does NOT ask for more cyclic.
#
# Instead:
#   1) keep the best V3 direct cyclic teacher,
#   2) temporarily reduce PPO collective during first seconds,
#   3) give XY controller time to settle,
#   4) smoothly return collective to PPO,
#   5) after 140 ft everything is PPO again.
#
# IMPORTANT:
# COLLECTIVE_REDUCTION is in PHYSICAL collective command units,
# not PPO normalized-action units.
# ============================================================

COLLECTIVE_REDUCTION_VALUES = [
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
    0.050,
]

COLLECTIVE_HOLD_VALUES = [
    0.75,
    1.50,
    2.50,
    3.50,
]

COLLECTIVE_FADE_VALUES = [
    1.0,
    2.0,
    3.0,
]


# ============================================================
# HELPERS
# ============================================================

def teacher_gate(altitude):
    if altitude <= TEACHER_FULL_ALT:
        return 1.0

    if altitude >= TEACHER_OFF_ALT:
        return 0.0

    return float(
        (TEACHER_OFF_ALT - altitude)
        /
        (TEACHER_OFF_ALT - TEACHER_FULL_ALT)
    )


def ff_scale(t):
    if t <= FF_HOLD:
        return 1.0

    end = FF_HOLD + FF_FADE

    if t >= end:
        return 0.0

    return float(
        (end - t)
        /
        FF_FADE
    )


def collective_soft_gate(
    t,
    hold_time,
    fade_time,
):
    """
    1.0 = full collective reduction
    0.0 = original PPO collective
    """
    if t <= hold_time:
        return 1.0

    end = hold_time + fade_time

    if t >= end:
        return 0.0

    return float(
        (end - t)
        /
        fade_time
    )


def get_collective_mapping(
    env,
    altitude,
):
    """
    Returns:
        trim_collective,
        collective_scale

    Works with the Stage-1 altitude-dependent trim env.
    Includes fallback for older residual env variants.
    """
    if hasattr(
        env,
        "_get_trim_controls",
    ):
        trims = env._get_trim_controls(
            altitude
        )

        trim_collective = float(
            trims[0]
        )
    else:
        trim_collective = float(
            getattr(
                env,
                "base_collective",
                0.56,
            )
        )

    collective_scale = float(
        getattr(
            env,
            "collective_scale",
            0.12,
        )
    )

    if abs(collective_scale) < 1e-9:
        raise RuntimeError(
            "collective_scale is zero."
        )

    return (
        trim_collective,
        collective_scale,
    )


def action0_to_physical(
    env,
    altitude,
    action0,
):
    trim, scale = get_collective_mapping(
        env,
        altitude,
    )

    return float(
        np.clip(
            trim
            +
            scale
            *
            float(action0),
            0.0,
            1.0,
        )
    )


def physical_to_action0(
    env,
    altitude,
    physical_collective,
):
    trim, scale = get_collective_mapping(
        env,
        altitude,
    )

    return float(
        np.clip(
            (
                float(physical_collective)
                -
                trim
            )
            /
            scale,
            -1.0,
            +1.0,
        )
    )


def velocity_sign(
    value,
    threshold=0.08,
):
    if value > threshold:
        return 1

    if value < -threshold:
        return -1

    return 0


# ============================================================
# SCORE
# ============================================================

def make_score(
    early_max,
    path100,
    drift100,
    hs100,
    max_n,
    max_e,
    reversals,
    time100,
    failed,
):
    s_index = max(
        0.0,
        path100 - drift100,
    )

    # We want a straight takeoff, but we also reject solutions
    # that simply "cheat" by staying near the ground forever.
    slow_penalty = max(
        0.0,
        time100 - 18.0,
    )

    score = (
        650.0 * early_max
        + 190.0 * path100
        + 210.0 * s_index
        + 90.0 * hs100
        + 40.0 * max_n
        + 40.0 * max_e
        + 35.0 * reversals
        + 120.0 * slow_penalty
    )

    if failed:
        score += 1e9

    return (
        score,
        s_index,
    )


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def run_case(
    collective_reduction,
    collective_hold,
    collective_fade,
    total_time=28.0,
    detailed=False,
    use_soft_collective=True,
):
    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env.reset()

    dt = float(env.dt)
    max_steps = int(
        total_time / dt
    )

    teacher_prev = np.zeros(
        2,
        dtype=np.float64,
    )

    early_max = 0.0
    path100 = None
    drift100 = None
    hs100 = None
    time100 = None

    max_n = 0.0
    max_e = 0.0

    last_sign_n = 0
    last_sign_e = 0

    reversals_n = 0
    reversals_e = 0

    failed = False
    last_info = info

    next_print = 0.0

    min_collective = 999.0
    max_collective = -999.0

    max_abs_pitch_100 = 0.0
    max_abs_roll_100 = 0.0

    for step in range(max_steps):
        t_before = step * dt

        # ----------------------------------------------------
        # PPO
        # ----------------------------------------------------
        ppo_action, _ = model.predict(
            obs,
            deterministic=True,
        )

        ppo_action = np.asarray(
            ppo_action,
            dtype=np.float32,
        ).reshape(-1)

        action = ppo_action.copy()

        # ----------------------------------------------------
        # CURRENT STATE
        # ----------------------------------------------------
        altitude = float(
            info.get(
                "altitude",
                0.0,
            )
        )

        north = float(
            info.get(
                "north",
                0.0,
            )
        )

        east = float(
            info.get(
                "east",
                0.0,
            )
        )

        vn = float(
            info.get(
                "vn",
                0.0,
            )
        )

        ve = float(
            info.get(
                "ve",
                0.0,
            )
        )

        # ====================================================
        # 1) DIRECT CYCLIC TEACHER FROM V3
        # ====================================================

        xy_gate = teacher_gate(
            altitude
        )

        teacher_norm = np.zeros(
            2,
            dtype=np.float64,
        )

        if xy_gate > 0.0:
            desired_accel = np.array(
                [
                    -XY_KP * north
                    -
                    XY_KD * vn,

                    -XY_KP * east
                    -
                    XY_KD * ve,
                ],
                dtype=np.float64,
            )

            desired_accel = np.clip(
                desired_accel,
                -0.22,
                +0.22,
            )

            feedback_delta = (
                B_INV
                @
                desired_accel
            )

            ff = np.array(
                [
                    FF_ELE
                    *
                    ff_scale(
                        t_before
                    ),

                    FF_AIL
                    *
                    ff_scale(
                        t_before
                    ),
                ],
                dtype=np.float64,
            )

            requested_delta = (
                feedback_delta
                +
                ff
            )

            requested_delta = np.clip(
                requested_delta,
                -CYCLIC_AUTHORITY,
                +CYCLIC_AUTHORITY,
            )

            raw_teacher_norm = (
                requested_delta
                /
                CYCLIC_AUTHORITY
            )

            raw_teacher_norm = np.clip(
                raw_teacher_norm,
                -1.0,
                +1.0,
            )

            teacher_norm = (
                (1.0 - XY_ALPHA)
                *
                teacher_prev
                +
                XY_ALPHA
                *
                raw_teacher_norm
            )

            teacher_prev = (
                teacher_norm.copy()
            )

            teacher_norm = np.clip(
                teacher_norm,
                -1.0,
                +1.0,
            )

            action[1] = float(
                xy_gate
                *
                teacher_norm[0]
                +
                (1.0 - xy_gate)
                *
                ppo_action[1]
            )

            action[2] = float(
                xy_gate
                *
                teacher_norm[1]
                +
                (1.0 - xy_gate)
                *
                ppo_action[2]
            )

        # ====================================================
        # 2) COLLECTIVE SOFT START
        # ====================================================

        ppo_collective = (
            action0_to_physical(
                env,
                altitude,
                ppo_action[0],
            )
        )

        soft_gate = 0.0

        commanded_collective = (
            ppo_collective
        )

        if use_soft_collective:
            soft_gate = (
                collective_soft_gate(
                    t_before,
                    collective_hold,
                    collective_fade,
                )
            )

            # Reduce PPO physical collective by the requested
            # amount. Never raise collective above PPO.
            commanded_collective = float(
                np.clip(
                    ppo_collective
                    -
                    soft_gate
                    *
                    collective_reduction,
                    0.0,
                    1.0,
                )
            )

            action[0] = (
                physical_to_action0(
                    env,
                    altitude,
                    commanded_collective,
                )
            )

        action = np.clip(
            action,
            -1.0,
            +1.0,
        )

        # ----------------------------------------------------
        # STEP
        # ----------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        last_info = info

        t = (step + 1) * dt

        altitude_now = float(
            info.get(
                "altitude",
                0.0,
            )
        )

        north_now = float(
            info.get(
                "north",
                0.0,
            )
        )

        east_now = float(
            info.get(
                "east",
                0.0,
            )
        )

        vn_now = float(
            info.get(
                "vn",
                0.0,
            )
        )

        ve_now = float(
            info.get(
                "ve",
                0.0,
            )
        )

        drift = float(
            info.get(
                "drift",
                0.0,
            )
        )

        path = float(
            info.get(
                "path",
                0.0,
            )
        )

        actual_collective = float(
            info.get(
                "collective",
                commanded_collective,
            )
        )

        vs_now = float(
            info.get(
                "vertical_speed",
                0.0,
            )
        )

        pitch_now = float(
            info.get(
                "pitch",
                0.0,
            )
        )

        roll_now = float(
            info.get(
                "roll",
                0.0,
            )
        )

        hspeed = float(
            np.hypot(
                vn_now,
                ve_now,
            )
        )

        min_collective = min(
            min_collective,
            actual_collective,
        )

        max_collective = max(
            max_collective,
            actual_collective,
        )

        # ----------------------------------------------------
        # FIRST 100 FT METRICS
        # ----------------------------------------------------
        if altitude_now <= 100.0:
            early_max = max(
                early_max,
                drift,
            )

            max_n = max(
                max_n,
                abs(
                    north_now
                ),
            )

            max_e = max(
                max_e,
                abs(
                    east_now
                ),
            )

            max_abs_pitch_100 = max(
                max_abs_pitch_100,
                abs(
                    pitch_now
                ),
            )

            max_abs_roll_100 = max(
                max_abs_roll_100,
                abs(
                    roll_now
                ),
            )

            sn = velocity_sign(
                vn_now
            )

            se = velocity_sign(
                ve_now
            )

            if sn != 0:
                if (
                    last_sign_n != 0
                    and
                    sn != last_sign_n
                ):
                    reversals_n += 1

                last_sign_n = sn

            if se != 0:
                if (
                    last_sign_e != 0
                    and
                    se != last_sign_e
                ):
                    reversals_e += 1

                last_sign_e = se

        if (
            path100 is None
            and
            altitude_now >= 100.0
        ):
            path100 = path
            drift100 = drift
            hs100 = hspeed
            time100 = t

        # ----------------------------------------------------
        # DETAILED LOG
        # ----------------------------------------------------
        if (
            detailed
            and
            t >= next_print
        ):
            print(
                f"t={t:6.2f}s | "
                f"ALT={altitude_now:7.2f} | "
                f"VS={vs_now:+6.2f} | "
                f"N={north_now:+6.2f} | "
                f"E={east_now:+6.2f} | "
                f"DRIFT={drift:5.2f} | "
                f"PATH={path:5.2f} | "
                f"PPOcol={ppo_collective:.5f} | "
                f"OUTcol={actual_collective:.5f} | "
                f"SOFT={soft_gate:4.2f} | "
                f"Tcyc=({teacher_norm[0]:+5.2f},"
                f"{teacher_norm[1]:+5.2f}) | "
                f"OUTcyc=({action[1]:+5.2f},"
                f"{action[2]:+5.2f})"
            )

            next_print += 1.0

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------
        if (
            terminated
            and
            not bool(
                info.get(
                    "success",
                    False,
                )
            )
        ):
            failed = True
            break

        if drift > 15.0:
            failed = True
            break

        if altitude_now > 350.0:
            failed = True
            break

        # If a very low collective candidate simply sits on the
        # ground too long, reject it.
        if (
            t > 12.0
            and
            altitude_now < 12.0
        ):
            failed = True
            break

    env.close()

    # --------------------------------------------------------
    # FALLBACK IF 100 FT WAS NOT REACHED
    # --------------------------------------------------------
    if path100 is None:
        path100 = float(
            last_info.get(
                "path",
                999.0,
            )
        )

        drift100 = float(
            last_info.get(
                "drift",
                999.0,
            )
        )

        hs100 = float(
            np.hypot(
                float(
                    last_info.get(
                        "vn",
                        999.0,
                    )
                ),
                float(
                    last_info.get(
                        "ve",
                        999.0,
                    )
                ),
            )
        )

        time100 = total_time
        failed = True

    reversals = (
        reversals_n
        +
        reversals_e
    )

    score, s_index = (
        make_score(
            early_max=early_max,
            path100=path100,
            drift100=drift100,
            hs100=hs100,
            max_n=max_n,
            max_e=max_e,
            reversals=reversals,
            time100=time100,
            failed=failed,
        )
    )

    return {
        "collective_reduction":
            float(
                collective_reduction
            ),

        "collective_hold":
            float(
                collective_hold
            ),

        "collective_fade":
            float(
                collective_fade
            ),

        "failed":
            bool(
                failed
            ),

        "score":
            float(
                score
            ),

        "early_max":
            float(
                early_max
            ),

        "path100":
            float(
                path100
            ),

        "drift100":
            float(
                drift100
            ),

        "s_index":
            float(
                s_index
            ),

        "hs100":
            float(
                hs100
            ),

        "time100":
            float(
                time100
            ),

        "max_n":
            float(
                max_n
            ),

        "max_e":
            float(
                max_e
            ),

        "reversals":
            int(
                reversals
            ),

        "reversals_n":
            int(
                reversals_n
            ),

        "reversals_e":
            int(
                reversals_e
            ),

        "max_pitch100":
            float(
                max_abs_pitch_100
            ),

        "max_roll100":
            float(
                max_abs_roll_100
            ),

        "min_collective":
            float(
                min_collective
            ),

        "max_collective":
            float(
                max_collective
            ),

        "total_max":
            float(
                last_info.get(
                    "max_drift",
                    999.0,
                )
            ),

        "final_drift":
            float(
                last_info.get(
                    "drift",
                    999.0,
                )
            ),

        "final_path":
            float(
                last_info.get(
                    "path",
                    999.0,
                )
            ),

        "final_alt":
            float(
                last_info.get(
                    "altitude",
                    999.0,
                )
            ),

        "final_vs":
            float(
                last_info.get(
                    "vertical_speed",
                    999.0,
                )
            ),
    }


# ============================================================
# HEADER
# ============================================================

print(
    "="
    *
    150
)

print(
    "STAGE 1 EARLY XY V4 — COLLECTIVE SOFT-START + V3 CYCLIC TEACHER"
)

print(
    "="
    *
    150
)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nV3 cyclic teacher:"
)

print(
    f"KP={XY_KP} | "
    f"KD={XY_KD} | "
    f"ALPHA={XY_ALPHA} | "
    f"FF=({FF_ELE:+.3f},"
    f"{FF_AIL:+.3f})"
)

print(
    "\nV4 search varies only the early physical collective reduction."
)


# ============================================================
# RUNTIME MAPPING DIAGNOSTIC
# ============================================================

diag_env = HelicopterEnvStage1Distill(
    teacher_model_path=None,
    training_mode=False,
)

diag_obs, diag_info = (
    diag_env.reset()
)

diag_alt = float(
    diag_info.get(
        "altitude",
        0.0,
    )
)

diag_action, _ = model.predict(
    diag_obs,
    deterministic=True,
)

diag_action = np.asarray(
    diag_action,
    dtype=np.float32,
).reshape(-1)

diag_trim, diag_scale = (
    get_collective_mapping(
        diag_env,
        diag_alt,
    )
)

diag_ppo_collective = (
    action0_to_physical(
        diag_env,
        diag_alt,
        diag_action[0],
    )
)

diag_env.close()

print(
    "\nDetected collective mapping:"
)

print(
    f"Initial altitude       : "
    f"{diag_alt:.3f} ft"
)

print(
    f"Trim collective        : "
    f"{diag_trim:.5f}"
)

print(
    f"Collective scale       : "
    f"{diag_scale:.5f}"
)

print(
    f"Initial PPO action[0]  : "
    f"{float(diag_action[0]):+.5f}"
)

print(
    f"Initial PPO collective : "
    f"{diag_ppo_collective:.5f}"
)


# ============================================================
# BASELINE V3
# ============================================================

print(
    "\n"
    +
    "="
    *
    150
)

print(
    "REFERENCE — V3 WITHOUT COLLECTIVE SOFT-START"
)

print(
    "="
    *
    150
)

reference = run_case(
    collective_reduction=0.0,
    collective_hold=0.0,
    collective_fade=1.0,
    total_time=28.0,
    detailed=False,
    use_soft_collective=False,
)

print(
    f"MAX100={reference['early_max']:.3f} | "
    f"PATH100={reference['path100']:.3f} | "
    f"DRIFT100={reference['drift100']:.3f} | "
    f"S={reference['s_index']:.3f} | "
    f"HS100={reference['hs100']:.3f} | "
    f"T100={reference['time100']:.2f}s"
)


# ============================================================
# GRID SEARCH
# ============================================================

print(
    "\n"
    +
    "="
    *
    150
)

print(
    "V4 COLLECTIVE SOFT-START SEARCH"
)

print(
    "="
    *
    150
)

results = []

total_candidates = (
    len(
        COLLECTIVE_REDUCTION_VALUES
    )
    *
    len(
        COLLECTIVE_HOLD_VALUES
    )
    *
    len(
        COLLECTIVE_FADE_VALUES
    )
)

print(
    f"\nTesting {total_candidates} candidates...\n"
)

counter = 0

for reduction in (
    COLLECTIVE_REDUCTION_VALUES
):
    for hold in (
        COLLECTIVE_HOLD_VALUES
    ):
        for fade in (
            COLLECTIVE_FADE_VALUES
        ):
            counter += 1

            r = run_case(
                collective_reduction=
                    reduction,

                collective_hold=
                    hold,

                collective_fade=
                    fade,

                total_time=28.0,

                detailed=False,

                use_soft_collective=True,
            )

            results.append(
                r
            )

            if (
                counter % 8 == 0
                or
                r["early_max"] <= 2.0
            ):
                print(
                    f"{counter:3d}/"
                    f"{total_candidates} | "
                    f"D={reduction:.3f} "
                    f"H={hold:.2f} "
                    f"F={fade:.1f} | "
                    f"MAX100={r['early_max']:.3f} | "
                    f"PATH100={r['path100']:.3f} | "
                    f"DRIFT100={r['drift100']:.3f} | "
                    f"S={r['s_index']:.3f} | "
                    f"HS100={r['hs100']:.3f} | "
                    f"T100={r['time100']:.2f}"
                )


results.sort(
    key=lambda x: x["score"]
)


# ============================================================
# TOP 20
# ============================================================

print(
    "\n"
    +
    "="
    *
    150
)

print(
    "TOP 20 V4 CANDIDATES"
)

print(
    "="
    *
    150
)

for i, r in enumerate(
    results[:20],
    start=1,
):
    print(
        f"{i:2d}. "
        f"D={r['collective_reduction']:.3f} | "
        f"HOLD={r['collective_hold']:.2f} | "
        f"FADE={r['collective_fade']:.1f} | "
        f"MAX100={r['early_max']:.3f} | "
        f"PATH100={r['path100']:.3f} | "
        f"DRIFT100={r['drift100']:.3f} | "
        f"S={r['s_index']:.3f} | "
        f"HS100={r['hs100']:.3f} | "
        f"T100={r['time100']:.2f}s | "
        f"MAXN={r['max_n']:.3f} | "
        f"MAXE={r['max_e']:.3f} | "
        f"REV={r['reversals']} | "
        f"COLMIN={r['min_collective']:.5f}"
    )


# ============================================================
# FULL 120 SECOND VALIDATION
# ============================================================

best = results[0]

print(
    "\n"
    +
    "="
    *
    150
)

print(
    "BEST V4 — FULL 120 SECOND VALIDATION"
)

print(
    "="
    *
    150
)

print(
    f"COLLECTIVE REDUCTION = "
    f"{best['collective_reduction']:.3f}"
)

print(
    f"HOLD                 = "
    f"{best['collective_hold']:.2f} s"
)

print(
    f"FADE                 = "
    f"{best['collective_fade']:.1f} s"
)

print()


full = run_case(
    collective_reduction=
        best[
            "collective_reduction"
        ],

    collective_hold=
        best[
            "collective_hold"
        ],

    collective_fade=
        best[
            "collective_fade"
        ],

    total_time=120.0,

    detailed=True,

    use_soft_collective=True,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    +
    "="
    *
    150
)

print(
    "V4 BEST FULL RESULT"
)

print(
    "="
    *
    150
)

print(
    f"EARLY MAX DRIFT (<100 ft) : "
    f"{full['early_max']:.3f} ft"
)

print(
    f"XY PATH AT 100 ft         : "
    f"{full['path100']:.3f} ft"
)

print(
    f"DRIFT AT 100 ft           : "
    f"{full['drift100']:.3f} ft"
)

print(
    f"S-INDEX                    : "
    f"{full['s_index']:.3f} ft"
)

print(
    f"HORIZONTAL SPEED @100 ft  : "
    f"{full['hs100']:.3f} ft/s"
)

print(
    f"TIME TO 100 ft             : "
    f"{full['time100']:.3f} s"
)

print(
    f"MAX |NORTH| <100 ft       : "
    f"{full['max_n']:.3f} ft"
)

print(
    f"MAX |EAST| <100 ft        : "
    f"{full['max_e']:.3f} ft"
)

print(
    f"DIRECTION REVERSALS        : "
    f"{full['reversals']} "
    f"(N={full['reversals_n']}, "
    f"E={full['reversals_e']})"
)

print(
    f"MAX |PITCH| <100 ft       : "
    f"{full['max_pitch100']:.5f} rad"
)

print(
    f"MAX |ROLL| <100 ft        : "
    f"{full['max_roll100']:.5f} rad"
)

print(
    f"MIN COLLECTIVE             : "
    f"{full['min_collective']:.5f}"
)

print(
    f"MAX COLLECTIVE             : "
    f"{full['max_collective']:.5f}"
)

print(
    f"TOTAL MAX DRIFT            : "
    f"{full['total_max']:.3f} ft"
)

print(
    f"FINAL DRIFT                : "
    f"{full['final_drift']:.3f} ft"
)

print(
    f"FINAL XY PATH              : "
    f"{full['final_path']:.3f} ft"
)

print(
    f"FINAL ALTITUDE             : "
    f"{full['final_alt']:.3f} ft"
)

print(
    f"FINAL VS                   : "
    f"{full['final_vs']:.3f} ft/s"
)


presentation_quality = bool(
    not full["failed"]

    and
    full["early_max"] <= 2.0

    and
    full["path100"] <= 5.0

    and
    full["drift100"] <= 1.5

    and
    full["hs100"] <= 0.75

    and
    full["time100"] <= 22.0

    and
    full["total_max"] <= 8.0

    and
    295.0
    <=
    full["final_alt"]
    <=
    305.0

    and
    abs(
        full["final_vs"]
    )
    <=
    0.75
)

print()

print(
    "EARLY TAKEOFF PRESENTATION QUALITY :",
    presentation_quality,
)


# ============================================================
# SAVE BEST CONFIG
# ============================================================

OUTPUT_DIR = (
    "results_stage1_early_xy_v4"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

config = {
    "model_path":
        MODEL_PATH,

    "cyclic_teacher": {
        "kp":
            XY_KP,

        "kd":
            XY_KD,

        "alpha":
            XY_ALPHA,

        "ff_ele":
            FF_ELE,

        "ff_ail":
            FF_AIL,

        "ff_hold":
            FF_HOLD,

        "teacher_full_alt":
            TEACHER_FULL_ALT,

        "teacher_off_alt":
            TEACHER_OFF_ALT,
    },

    "collective_soft_start": {
        "physical_reduction":
            full[
                "collective_reduction"
            ],

        "hold_seconds":
            full[
                "collective_hold"
            ],

        "fade_seconds":
            full[
                "collective_fade"
            ],
    },

    "metrics": {
        "early_max_drift_ft":
            full[
                "early_max"
            ],

        "path_at_100_ft":
            full[
                "path100"
            ],

        "drift_at_100_ft":
            full[
                "drift100"
            ],

        "s_index_ft":
            full[
                "s_index"
            ],

        "horizontal_speed_at_100":
            full[
                "hs100"
            ],

        "time_to_100_ft_s":
            full[
                "time100"
            ],

        "final_altitude_ft":
            full[
                "final_alt"
            ],

        "final_vs_ft_s":
            full[
                "final_vs"
            ],

        "final_drift_ft":
            full[
                "final_drift"
            ],
    },

    "presentation_quality":
        presentation_quality,
}

config_path = os.path.join(
    OUTPUT_DIR,
    "best_teacher_config.json",
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )

print(
    f"\nSaved: {config_path}"
)


if presentation_quality:
    print(
        "\nSUCCESS."
    )

    print(
        "Next: distill BOTH the early collective soft-start "
        "and the direct cyclic teacher into the single PPO."
    )
else:
    print(
        "\nStill not presentation-clean."
    )

    print(
        "If V4 remains near ~2.5-3 ft despite slower collective, "
        "the next step is explicit pitch/roll attitude teacher "
        "during liftoff rather than more XY gain."
    )

print(
    "="
    *
    150
)
