import json
import os
import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill


# ============================================================
# FINAL PPO
# ============================================================

MODEL_PATH = (
    "models_stage1_final_distilled/"
    "AH1S_STAGE1_FINAL_DISTILLED.zip"
)

model = PPO.load(MODEL_PATH)


# ============================================================
# V3 CYCLIC TEACHER
# ============================================================

B_INV = np.array(
    [
        [-0.20338265, -0.00859620],
        [ 0.01952073, -0.14631468],
    ],
    dtype=np.float64,
)

CYCLIC_AUTHORITY = 0.026

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
# V4B: NORMALIZED PPO COLLECTIVE SOFT START
#
# IMPORTANT:
# We do NOT attempt to convert action[0] into a physical
# collective value before env.step().
#
# We directly reduce PPO action[0].
# JSBSim's true applied collective is then read from:
#     fcs/collective-cmd-norm
#
# This avoids the incorrect mapping used in V4.
# ============================================================

A0_REDUCTION_VALUES = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.65,
]

HOLD_VALUES = [
    0.75,
    1.50,
    2.50,
    3.50,
]

FADE_VALUES = [
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


def soft_gate(t, hold, fade):
    if t <= hold:
        return 1.0

    end = hold + fade

    if t >= end:
        return 0.0

    return float(
        (end - t)
        /
        fade
    )


def raw_fdm_state(env):
    """
    Read the real JSBSim state directly.

    This deliberately avoids info.get(..., 0) for attitude.
    """
    return {
        "altitude": float(
            env.fdm["position/h-agl-ft"]
        ),
        "pitch": float(
            env.fdm["attitude/pitch-rad"]
        ),
        "roll": float(
            env.fdm["attitude/roll-rad"]
        ),
        "p": float(
            env.fdm["velocities/p-rad_sec"]
        ),
        "q": float(
            env.fdm["velocities/q-rad_sec"]
        ),
        "r": float(
            env.fdm["velocities/r-rad_sec"]
        ),
        "collective": float(
            env.fdm["fcs/collective-cmd-norm"]
        ),
        "elevator": float(
            env.fdm["fcs/elevator-cmd-norm"]
        ),
        "aileron": float(
            env.fdm["fcs/aileron-cmd-norm"]
        ),
        "rudder": float(
            env.fdm["fcs/rudder-cmd-norm"]
        ),
    }


def velocity_sign(value, threshold=0.08):
    if value > threshold:
        return 1

    if value < -threshold:
        return -1

    return 0


def score_result(r):
    slow_penalty = max(
        0.0,
        r["time100"] - 18.0,
    )

    return (
        700.0 * r["early_max"]
        + 210.0 * r["path100"]
        + 230.0 * r["s_index"]
        + 100.0 * r["hs100"]
        + 40.0 * r["max_n"]
        + 40.0 * r["max_e"]
        + 35.0 * r["reversals"]
        + 120.0 * slow_penalty
    )


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    a0_reduction,
    hold,
    fade,
    total_time=25.0,
    use_soft=True,
    detailed=False,
):
    env = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )

    obs, info = env.reset()

    dt = float(env.dt)
    max_steps = int(total_time / dt)

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

    reversals_n = 0
    reversals_e = 0
    last_sign_n = 0
    last_sign_e = 0

    max_pitch100 = 0.0
    max_roll100 = 0.0
    max_p100 = 0.0
    max_q100 = 0.0

    min_collective100 = 999.0
    max_collective100 = -999.0

    next_print = 0.0
    failed = False
    last_info = info

    # True initial state, directly from JSBSim.
    raw = raw_fdm_state(env)

    initial_pitch = raw["pitch"]
    initial_roll = raw["roll"]

    for step in range(max_steps):
        t_before = step * dt

        ppo_action, _ = model.predict(
            obs,
            deterministic=True,
        )

        ppo_action = np.asarray(
            ppo_action,
            dtype=np.float32,
        ).reshape(-1)

        action = ppo_action.copy()

        raw_before = raw_fdm_state(env)

        altitude = raw_before["altitude"]

        north = float(
            info.get("north", 0.0)
        )
        east = float(
            info.get("east", 0.0)
        )
        vn = float(
            info.get("vn", 0.0)
        )
        ve = float(
            info.get("ve", 0.0)
        )

        # ----------------------------------------------------
        # V3 DIRECT CYCLIC TEACHER
        # ----------------------------------------------------

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

            ff_delta = np.array(
                [
                    FF_ELE * ff_scale(t_before),
                    FF_AIL * ff_scale(t_before),
                ],
                dtype=np.float64,
            )

            requested_delta = (
                feedback_delta
                +
                ff_delta
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
                xy_gate * teacher_norm[0]
                +
                (1.0 - xy_gate)
                * ppo_action[1]
            )

            action[2] = float(
                xy_gate * teacher_norm[1]
                +
                (1.0 - xy_gate)
                * ppo_action[2]
            )

        # ----------------------------------------------------
        # CORRECT COLLECTIVE SOFT START
        #
        # Directly modify normalized PPO action[0].
        # ----------------------------------------------------

        sg = 0.0

        if use_soft:
            sg = soft_gate(
                t_before,
                hold,
                fade,
            )

            action[0] = (
                float(ppo_action[0])
                -
                sg
                *
                a0_reduction
            )

        action = np.clip(
            action,
            -1.0,
            +1.0,
        ).astype(np.float32)

        # ----------------------------------------------------
        # STEP
        # ----------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        last_info = info

        raw_after = raw_fdm_state(env)

        t = (step + 1) * dt

        altitude_now = raw_after["altitude"]

        north_now = float(
            info.get("north", 0.0)
        )
        east_now = float(
            info.get("east", 0.0)
        )
        vn_now = float(
            info.get("vn", 0.0)
        )
        ve_now = float(
            info.get("ve", 0.0)
        )

        drift = float(
            info.get("drift", 0.0)
        )
        path = float(
            info.get("path", 0.0)
        )

        vs_now = float(
            info.get(
                "vertical_speed",
                env.fdm["velocities/h-dot-fps"],
            )
        )

        hspeed = float(
            np.hypot(
                vn_now,
                ve_now,
            )
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
                abs(north_now),
            )

            max_e = max(
                max_e,
                abs(east_now),
            )

            max_pitch100 = max(
                max_pitch100,
                abs(raw_after["pitch"]),
            )

            max_roll100 = max(
                max_roll100,
                abs(raw_after["roll"]),
            )

            max_p100 = max(
                max_p100,
                abs(raw_after["p"]),
            )

            max_q100 = max(
                max_q100,
                abs(raw_after["q"]),
            )

            min_collective100 = min(
                min_collective100,
                raw_after["collective"],
            )

            max_collective100 = max(
                max_collective100,
                raw_after["collective"],
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
        # DETAILED OUTPUT
        # ----------------------------------------------------

        if detailed and t >= next_print:
            print(
                f"t={t:6.2f}s | "
                f"ALT={altitude_now:7.2f} | "
                f"VS={vs_now:+6.2f} | "
                f"N={north_now:+6.2f} | "
                f"E={east_now:+6.2f} | "
                f"DRIFT={drift:5.2f} | "
                f"PITCH={np.degrees(raw_after['pitch']):+7.3f}deg | "
                f"ROLL={np.degrees(raw_after['roll']):+7.3f}deg | "
                f"P={np.degrees(raw_after['p']):+7.3f}deg/s | "
                f"Q={np.degrees(raw_after['q']):+7.3f}deg/s | "
                f"PPOa0={float(ppo_action[0]):+6.3f} | "
                f"OUTa0={float(action[0]):+6.3f} | "
                f"COL={raw_after['collective']:.5f} | "
                f"SOFT={sg:4.2f}"
            )

            next_print += 0.5

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------

        if (
            terminated
            and
            not bool(
                info.get("success", False)
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

    env.close()

    if path100 is None:
        path100 = float(
            last_info.get("path", 999.0)
        )
        drift100 = float(
            last_info.get("drift", 999.0)
        )
        hs100 = float(
            np.hypot(
                float(last_info.get("vn", 999.0)),
                float(last_info.get("ve", 999.0)),
            )
        )
        time100 = total_time
        failed = True

    s_index = max(
        0.0,
        path100 - drift100,
    )

    result = {
        "a0_reduction":
            float(a0_reduction),

        "hold":
            float(hold),

        "fade":
            float(fade),

        "failed":
            bool(failed),

        "early_max":
            float(early_max),

        "path100":
            float(path100),

        "drift100":
            float(drift100),

        "s_index":
            float(s_index),

        "hs100":
            float(hs100),

        "time100":
            float(time100),

        "max_n":
            float(max_n),

        "max_e":
            float(max_e),

        "reversals_n":
            int(reversals_n),

        "reversals_e":
            int(reversals_e),

        "reversals":
            int(
                reversals_n
                +
                reversals_e
            ),

        "initial_pitch":
            float(initial_pitch),

        "initial_roll":
            float(initial_roll),

        "max_pitch100":
            float(max_pitch100),

        "max_roll100":
            float(max_roll100),

        "max_p100":
            float(max_p100),

        "max_q100":
            float(max_q100),

        "min_collective100":
            float(min_collective100),

        "max_collective100":
            float(max_collective100),

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

    result["score"] = (
        score_result(result)
        if not failed
        else 1e9
    )

    return result


# ============================================================
# HEADER
# ============================================================

print("=" * 150)
print(
    "STAGE 1 EARLY XY V4B — CORRECT NORMALIZED COLLECTIVE SOFT-START"
)
print("=" * 150)

print("\nModel:")
print(MODEL_PATH)

print(
    "\nImportant fixes:"
)
print(
    "  1) No guessed action->physical collective conversion."
)
print(
    "  2) Soft-start directly modifies PPO action[0]."
)
print(
    "  3) True collective is read from JSBSim FDM."
)
print(
    "  4) True pitch/roll/p/q are read directly from JSBSim FDM."
)


# ============================================================
# TRUE INITIAL STATE
# ============================================================

probe_env = HelicopterEnvStage1Distill(
    teacher_model_path=None,
    training_mode=False,
)

probe_obs, probe_info = probe_env.reset()
probe = raw_fdm_state(probe_env)

probe_action, _ = model.predict(
    probe_obs,
    deterministic=True,
)

probe_action = np.asarray(
    probe_action,
    dtype=np.float32,
).reshape(-1)

print("\nTRUE INITIAL JSBSIM STATE:")
print(
    f"ALT        : {probe['altitude']:.4f} ft"
)
print(
    f"PITCH      : {np.degrees(probe['pitch']):+.4f} deg"
)
print(
    f"ROLL       : {np.degrees(probe['roll']):+.4f} deg"
)
print(
    f"P          : {np.degrees(probe['p']):+.4f} deg/s"
)
print(
    f"Q          : {np.degrees(probe['q']):+.4f} deg/s"
)
print(
    f"PPO a0     : {float(probe_action[0]):+.5f}"
)
print(
    f"FDM COLLECT: {probe['collective']:.5f}"
)

probe_env.close()


# ============================================================
# REFERENCE: V3, NO COLLECTIVE SOFT START
# ============================================================

print("\n" + "=" * 150)
print(
    "REFERENCE — V3 CYCLIC TEACHER, ORIGINAL PPO COLLECTIVE"
)
print("=" * 150)

reference = run_case(
    a0_reduction=0.0,
    hold=0.0,
    fade=1.0,
    total_time=25.0,
    use_soft=False,
    detailed=False,
)

print(
    f"MAX100={reference['early_max']:.3f} | "
    f"PATH100={reference['path100']:.3f} | "
    f"DRIFT100={reference['drift100']:.3f} | "
    f"S={reference['s_index']:.3f} | "
    f"T100={reference['time100']:.2f}s | "
    f"MAXPITCH={np.degrees(reference['max_pitch100']):.3f}deg | "
    f"MAXROLL={np.degrees(reference['max_roll100']):.3f}deg | "
    f"COL=[{reference['min_collective100']:.5f},"
    f"{reference['max_collective100']:.5f}]"
)


# ============================================================
# GRID SEARCH
# ============================================================

print("\n" + "=" * 150)
print(
    "V4B SEARCH"
)
print("=" * 150)

results = []

total = (
    len(A0_REDUCTION_VALUES)
    *
    len(HOLD_VALUES)
    *
    len(FADE_VALUES)
)

print(
    f"\nTesting {total} candidates...\n"
)

counter = 0

for reduction in A0_REDUCTION_VALUES:
    for hold in HOLD_VALUES:
        for fade in FADE_VALUES:
            counter += 1

            r = run_case(
                a0_reduction=reduction,
                hold=hold,
                fade=fade,
                total_time=25.0,
                use_soft=True,
                detailed=False,
            )

            results.append(r)

            if (
                counter % 8 == 0
                or
                r["early_max"] <= 2.0
            ):
                print(
                    f"{counter:3d}/{total} | "
                    f"dA0={reduction:.2f} "
                    f"H={hold:.2f} "
                    f"F={fade:.1f} | "
                    f"MAX100={r['early_max']:.3f} | "
                    f"PATH100={r['path100']:.3f} | "
                    f"DRIFT100={r['drift100']:.3f} | "
                    f"S={r['s_index']:.3f} | "
                    f"HS100={r['hs100']:.3f} | "
                    f"T100={r['time100']:.2f} | "
                    f"COLMIN={r['min_collective100']:.5f}"
                )


results.sort(
    key=lambda x: x["score"]
)


# ============================================================
# TOP 20
# ============================================================

print("\n" + "=" * 150)
print(
    "TOP 20 V4B CANDIDATES"
)
print("=" * 150)

for i, r in enumerate(
    results[:20],
    start=1,
):
    print(
        f"{i:2d}. "
        f"dA0={r['a0_reduction']:.2f} | "
        f"HOLD={r['hold']:.2f} | "
        f"FADE={r['fade']:.1f} | "
        f"MAX100={r['early_max']:.3f} | "
        f"PATH100={r['path100']:.3f} | "
        f"DRIFT100={r['drift100']:.3f} | "
        f"S={r['s_index']:.3f} | "
        f"HS100={r['hs100']:.3f} | "
        f"T100={r['time100']:.2f}s | "
        f"REV={r['reversals']} | "
        f"PITCH={np.degrees(r['max_pitch100']):.3f}deg | "
        f"ROLL={np.degrees(r['max_roll100']):.3f}deg | "
        f"COLMIN={r['min_collective100']:.5f}"
    )


# ============================================================
# FULL 120 SECOND VALIDATION
# ============================================================

best = results[0]

print("\n" + "=" * 150)
print(
    "BEST V4B — FULL 120 SECOND VALIDATION"
)
print("=" * 150)

print(
    f"A0 REDUCTION = {best['a0_reduction']:.2f}"
)
print(
    f"HOLD         = {best['hold']:.2f} s"
)
print(
    f"FADE         = {best['fade']:.1f} s"
)
print()

full = run_case(
    a0_reduction=best["a0_reduction"],
    hold=best["hold"],
    fade=best["fade"],
    total_time=120.0,
    use_soft=True,
    detailed=True,
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 150)
print(
    "V4B BEST FULL RESULT"
)
print("=" * 150)

print(
    f"EARLY MAX DRIFT (<100 ft) : {full['early_max']:.3f} ft"
)
print(
    f"XY PATH AT 100 ft         : {full['path100']:.3f} ft"
)
print(
    f"DRIFT AT 100 ft           : {full['drift100']:.3f} ft"
)
print(
    f"S-INDEX                    : {full['s_index']:.3f} ft"
)
print(
    f"HORIZONTAL SPEED @100 ft  : {full['hs100']:.3f} ft/s"
)
print(
    f"TIME TO 100 ft             : {full['time100']:.3f} s"
)
print(
    f"MAX |NORTH| <100 ft       : {full['max_n']:.3f} ft"
)
print(
    f"MAX |EAST| <100 ft        : {full['max_e']:.3f} ft"
)
print(
    f"DIRECTION REVERSALS        : {full['reversals']} "
    f"(N={full['reversals_n']}, E={full['reversals_e']})"
)

print()
print(
    "TRUE JSBSIM ATTITUDE (<100 ft)"
)
print(
    f"INITIAL PITCH              : "
    f"{np.degrees(full['initial_pitch']):+.4f} deg"
)
print(
    f"INITIAL ROLL               : "
    f"{np.degrees(full['initial_roll']):+.4f} deg"
)
print(
    f"MAX |PITCH|                : "
    f"{np.degrees(full['max_pitch100']):.4f} deg"
)
print(
    f"MAX |ROLL|                 : "
    f"{np.degrees(full['max_roll100']):.4f} deg"
)
print(
    f"MAX |ROLL RATE|            : "
    f"{np.degrees(full['max_p100']):.4f} deg/s"
)
print(
    f"MAX |PITCH RATE|           : "
    f"{np.degrees(full['max_q100']):.4f} deg/s"
)

print()
print(
    f"TRUE COLLECTIVE RANGE <100 : "
    f"{full['min_collective100']:.5f} -> "
    f"{full['max_collective100']:.5f}"
)

print(
    f"TOTAL MAX DRIFT            : {full['total_max']:.3f} ft"
)
print(
    f"FINAL DRIFT                : {full['final_drift']:.3f} ft"
)
print(
    f"FINAL XY PATH              : {full['final_path']:.3f} ft"
)
print(
    f"FINAL ALTITUDE             : {full['final_alt']:.3f} ft"
)
print(
    f"FINAL VS                   : {full['final_vs']:.3f} ft/s"
)


presentation_quality = bool(
    not full["failed"]
    and full["early_max"] <= 2.0
    and full["path100"] <= 5.0
    and full["drift100"] <= 1.5
    and full["hs100"] <= 0.75
    and full["time100"] <= 22.0
    and full["total_max"] <= 8.0
    and 295.0 <= full["final_alt"] <= 305.0
    and abs(full["final_vs"]) <= 0.75
)

print()
print(
    "EARLY TAKEOFF PRESENTATION QUALITY :",
    presentation_quality,
)


# ============================================================
# SAVE CONFIG
# ============================================================

OUTPUT_DIR = (
    "results_stage1_early_xy_v4b"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

config = {
    "model_path":
        MODEL_PATH,

    "a0_reduction":
        full["a0_reduction"],

    "hold_seconds":
        full["hold"],

    "fade_seconds":
        full["fade"],

    "metrics": {
        "early_max_drift_ft":
            full["early_max"],

        "path_at_100_ft":
            full["path100"],

        "drift_at_100_ft":
            full["drift100"],

        "s_index_ft":
            full["s_index"],

        "time_to_100_ft_s":
            full["time100"],

        "max_pitch_deg":
            float(
                np.degrees(
                    full["max_pitch100"]
                )
            ),

        "max_roll_deg":
            float(
                np.degrees(
                    full["max_roll100"]
                )
            ),

        "true_collective_min":
            full["min_collective100"],

        "true_collective_max":
            full["max_collective100"],

        "final_altitude_ft":
            full["final_alt"],

        "final_vs_ft_s":
            full["final_vs"],
    },

    "presentation_quality":
        presentation_quality,
}

config_path = os.path.join(
    OUTPUT_DIR,
    "best_v4b_config.json",
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
        "The corrected collective soft-start solved the early trajectory."
    )
    print(
        "Next step: distill cyclic + collective behavior into the PPO."
    )
else:
    print(
        "\nV4B did not reach presentation quality."
    )
    print(
        "Now the TRUE pitch/roll diagnostics are valid."
    )
    print(
        "Use them to design the V5 liftoff attitude controller."
    )

print("=" * 150)
