from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from helicopter_env_stage1_distill import (
    HelicopterEnvStage1Distill,
)
from helicopter_env_stage2_refine import (
    HelicopterEnvStage2Refine,
)


# =====================================================================
# PATHS
# =====================================================================

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

SOURCE_STAGE2_MODEL = (
    "models_stage2_refine/"
    "AH1S_STAGE2_REFINE_SUCCESS.zip"
)

OUTPUT_DIR = Path(
    "models_stage2_quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUCCESS_MODEL_PATH = (
    OUTPUT_DIR
    / "AH1S_STAGE2_QUALITY_SUCCESS"
)

FINAL_MODEL_PATH = (
    OUTPUT_DIR
    / "AH1S_STAGE2_QUALITY_FINAL"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# TRAINING
# =====================================================================

TOTAL_TIMESTEPS = 60_000
EVAL_EVERY = 5_000

LEARNING_RATE = 2.0e-5

TARGET_ALTITUDE = 300.0
TARGET_FORWARD_DISTANCE = 300.0

# The old Stage-2 policy was already close in endpoint terms.
# We are not relearning forward flight from scratch.
# We are teaching:
#
#   - do not dive while accelerating
#   - do not build lateral velocity
#   - do not accumulate cross-track
#   - arrive at 300 ft forward while still near 300 ft altitude


# =====================================================================
# STRICT QUALITY ENVIRONMENT
# =====================================================================

class HelicopterEnvStage2Quality(
    HelicopterEnvStage2Refine
):
    """
    Same Stage-2 dynamics, observation space and action space.

    Only reward / mission-quality criteria are strengthened.

    This is deliberate:
      - existing PPO knowledge is preserved
      - no architecture change
      - no observation change
      - no action change
    """

    def __init__(self):
        super().__init__()

        self.quality_cross_track_est = 0.0
        self.quality_prev_forward_distance = 0.0

        self.quality_dt = float(
            getattr(
                self,
                "dt",
                0.075,
            )
        )

        if (
            not np.isfinite(
                self.quality_dt
            )
            or
            self.quality_dt <= 0.0
        ):
            self.quality_dt = 0.075


    def reset(
        self,
        seed=None,
        options=None,
    ):
        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        self.quality_cross_track_est = 0.0

        self.quality_prev_forward_distance = float(
            info.get(
                "forward_distance",
                getattr(
                    self,
                    "forward_distance",
                    0.0,
                ),
            )
        )

        return obs, info


    def step(
        self,
        action,
    ):
        (
            obs,
            base_reward,
            terminated,
            truncated,
            info,
        ) = super().step(
            action
        )

        reward = float(
            base_reward
        )

        altitude = float(
            info.get(
                "altitude",
                300.0,
            )
        )

        vertical_speed = float(
            info.get(
                "vertical_speed",
                0.0,
            )
        )

        lateral_velocity = float(
            info.get(
                "lateral_velocity",
                0.0,
            )
        )

        forward_velocity = float(
            info.get(
                "forward_velocity",
                0.0,
            )
        )

        forward_distance = float(
            info.get(
                "forward_distance",
                getattr(
                    self,
                    "forward_distance",
                    0.0,
                ),
            )
        )

        altitude_error = (
            altitude
            -
            TARGET_ALTITUDE
        )

        # -------------------------------------------------------------
        # LATERAL-DISPLACEMENT PROXY
        #
        # Body lateral speed is not exactly inertial cross-track,
        # but while heading is held approximately constant it is an
        # excellent training signal.
        # -------------------------------------------------------------

        self.quality_cross_track_est += (
            lateral_velocity
            *
            self.quality_dt
        )

        # -------------------------------------------------------------
        # FORWARD PROGRESS
        # -------------------------------------------------------------

        delta_forward = (
            forward_distance
            -
            self.quality_prev_forward_distance
        )

        self.quality_prev_forward_distance = (
            forward_distance
        )

        # =============================================================
        # QUALITY REWARD
        # =============================================================

        # 1) ALTITUDE HOLD
        #
        # Old policy accepted ~35 ft of altitude loss because it could
        # recover near the end. That behavior is now expensive.
        reward -= (
            0.30
            *
            abs(
                altitude_error
            )
        )

        # Extra penalty once outside presentation corridor.
        if altitude < 295.0:
            reward -= (
                0.70
                *
                (
                    295.0
                    -
                    altitude
                )
            )

        if altitude > 305.0:
            reward -= (
                0.70
                *
                (
                    altitude
                    -
                    305.0
                )
            )

        # 2) VERTICAL SPEED
        reward -= (
            1.20
            *
            abs(
                vertical_speed
            )
        )

        # Descent during forward acceleration is specifically the old
        # failure mode, so descent receives an extra penalty.
        if vertical_speed < 0.0:
            reward -= (
                2.00
                *
                abs(
                    vertical_speed
                )
            )

        # 3) LATERAL VELOCITY
        reward -= (
            2.00
            *
            abs(
                lateral_velocity
            )
        )

        # 4) INTEGRATED CROSS-TRACK PROXY
        reward -= (
            0.12
            *
            abs(
                self.quality_cross_track_est
            )
        )

        # 5) KEEP USEFUL FORWARD PROGRESS
        if delta_forward > 0.0:
            reward += (
                0.50
                *
                delta_forward
            )

        # Prevent "solve altitude by refusing to move forward".
        if (
            forward_distance < 250.0
            and
            forward_velocity < 4.0
        ):
            reward -= (
                0.20
                *
                (
                    4.0
                    -
                    forward_velocity
                )
            )

        # 6) QUALITY CORRIDOR BONUSES
        if (
            295.0
            <=
            altitude
            <=
            305.0
        ):
            reward += 2.5

        if (
            297.0
            <=
            altitude
            <=
            303.0
            and
            abs(
                vertical_speed
            )
            <=
            1.0
        ):
            reward += 3.0

        if (
            abs(
                lateral_velocity
            )
            <=
            0.40
        ):
            reward += 1.0

        # =============================================================
        # STRICT FAILURE ENVELOPE
        # =============================================================

        # Do not let training normalize the old 35-ft dive.
        if altitude < 285.0:
            reward -= 250.0
            terminated = True
            info["success"] = False
            info[
                "termination_reason"
            ] = "quality_altitude_drop"

        if altitude > 315.0:
            reward -= 200.0
            terminated = True
            info["success"] = False
            info[
                "termination_reason"
            ] = "quality_altitude_rise"

        if (
            abs(
                self.quality_cross_track_est
            )
            >
            15.0
        ):
            reward -= 200.0
            terminated = True
            info["success"] = False
            info[
                "termination_reason"
            ] = "quality_cross_track"

        # =============================================================
        # STRICT 300-FT ARRIVAL
        # =============================================================

        if (
            forward_distance
            >=
            TARGET_FORWARD_DISTANCE
        ):
            strict_success = bool(
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
                2.0
                and
                abs(
                    lateral_velocity
                )
                <=
                0.75
                and
                abs(
                    self.quality_cross_track_est
                )
                <=
                5.0
            )

            terminated = True
            info["success"] = (
                strict_success
            )

            if strict_success:
                reward += 1200.0

                info[
                    "termination_reason"
                ] = (
                    "quality_success_300ft"
                )

            else:
                reward -= 300.0

                info[
                    "termination_reason"
                ] = (
                    "quality_bad_300ft_arrival"
                )

        info[
            "quality_cross_track_est"
        ] = float(
            self.quality_cross_track_est
        )

        info[
            "quality_reward"
        ] = float(
            reward
        )

        return (
            obs,
            float(
                reward
            ),
            bool(
                terminated
            ),
            bool(
                truncated
            ),
            info,
        )


# =====================================================================
# CONTINUOUS-HANDOFF EVALUATOR
# =====================================================================

EARTH_RADIUS_FT = 20_902_231.0


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


def get_active_fdm(
    env,
):
    direct = getattr(
        env,
        "fdm",
        None,
    )

    if direct is not None:
        return direct

    base_env = getattr(
        env,
        "base_env",
        None,
    )

    if base_env is not None:
        nested = getattr(
            base_env,
            "fdm",
            None,
        )

        if nested is not None:
            return nested

    raise RuntimeError(
        "Active JSBSim FDM not found."
    )


def lat_deg(
    fdm,
):
    for key in [
        "position/lat-gc-deg",
        "position/lat-geod-deg",
    ]:
        value = fdm_float(
            fdm,
            key,
        )

        if np.isfinite(
            value
        ):
            return value

    return float(
        "nan"
    )


def lon_deg(
    fdm,
):
    return fdm_float(
        fdm,
        "position/long-gc-deg",
    )


def local_ne(
    lat,
    lon,
    lat0,
    lon0,
):
    dlat = math.radians(
        lat
        -
        lat0
    )

    dlon = math.radians(
        lon
        -
        lon0
    )

    lat0_rad = math.radians(
        lat0
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
            lat0_rad
        )
        *
        dlon
    )

    return (
        north,
        east,
    )


def mission_projection(
    north,
    east,
    heading,
):
    c = math.cos(
        heading
    )

    s = math.sin(
        heading
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
        forward,
        cross,
    )


def horizontal_speed_stage1(
    info,
):
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

    return float(
        np.hypot(
            vn,
            ve,
        )
    )


def run_continuous_handoff_eval(
    stage2_model,
    detailed=False,
):
    """
    Real mission evaluation:

      Stage1 locked PPO
          ->
      SAME JSBSim FDM
          ->
      candidate Stage2 PPO

    No active-aircraft reset at handoff.
    """

    stage1_model = PPO.load(
        STAGE1_MODEL_PATH
    )

    # -------------------------------------------------------------
    # STAGE 1
    # -------------------------------------------------------------

    env1 = (
        HelicopterEnvStage1Distill(
            teacher_model_path=None,
            training_mode=False,
        )
    )

    obs1, info1 = env1.reset()

    fdm = get_active_fdm(
        env1
    )

    fdm_id = id(
        fdm
    )

    heading0 = fdm_float(
        fdm,
        "attitude/psi-rad",
        default=fdm_float(
            fdm,
            "attitude/heading-true-rad",
            default=math.pi,
        ),
    )

    dt1 = float(
        getattr(
            env1,
            "dt",
            0.075,
        )
    )

    stable_time = 0.0

    handoff = None

    for step in range(
        int(
            120.0
            /
            dt1
        )
    ):
        a1, _ = (
            stage1_model.predict(
                obs1,
                deterministic=True,
            )
        )

        (
            obs1,
            r1,
            term1,
            trunc1,
            info1,
        ) = env1.step(
            a1
        )

        alt = float(
            info1[
                "altitude"
            ]
        )

        vs = float(
            info1[
                "vertical_speed"
            ]
        )

        drift = float(
            info1[
                "drift"
            ]
        )

        hs = (
            horizontal_speed_stage1(
                info1
            )
        )

        stable = (
            295.0
            <=
            alt
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

        if stable:
            stable_time += (
                dt1
            )
        else:
            stable_time = 0.0

        if stable_time >= 5.0:
            handoff = {
                "altitude":
                    alt,

                "vs":
                    vs,

                "hs":
                    hs,

                "drift":
                    drift,
            }

            break

        if (
            term1
            and
            not bool(
                info1.get(
                    "success",
                    False,
                )
            )
        ):
            break

    if handoff is None:
        env1.close()

        return {
            "pass":
                False,

            "reason":
                "stage1_handoff_not_found",
        }

    handoff_lat = lat_deg(
        fdm
    )

    handoff_lon = lon_deg(
        fdm
    )

    # -------------------------------------------------------------
    # STAGE 2 QUALITY ENV
    # -------------------------------------------------------------

    env2 = (
        HelicopterEnvStage2Quality()
    )

    # Disposable reset initializes Python bookkeeping.
    env2.reset()

    # Attach the ACTIVE Stage-1 aircraft.
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
        env2.target_heading = (
            heading0
        )

    for attr in [
        "steps",
        "target_hold_steps",
        "hold_steps",
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

    env2.quality_cross_track_est = (
        0.0
    )

    env2.quality_prev_forward_distance = (
        0.0
    )

    if id(
        get_active_fdm(
            env2
        )
    ) != fdm_id:
        raise RuntimeError(
            "Continuous handoff FDM identity failed."
        )

    obs2 = np.asarray(
        env2._get_obs(),
        dtype=np.float32,
    )

    min_alt = float(
        handoff[
            "altitude"
        ]
    )

    max_alt = float(
        handoff[
            "altitude"
        ]
    )

    max_cross = 0.0

    crossing = None

    physical_failure = False

    dt2 = float(
        getattr(
            env2,
            "dt",
            0.075,
        )
    )

    for step2 in range(
        int(
            65.0
            /
            dt2
        )
    ):
        a2, _ = (
            stage2_model.predict(
                obs2,
                deterministic=True,
            )
        )

        (
            obs2,
            r2,
            term2,
            trunc2,
            info2,
        ) = env2.step(
            a2
        )

        altitude = float(
            info2.get(
                "altitude",
                fdm_float(
                    fdm,
                    "position/h-agl-ft",
                ),
            )
        )

        vs = float(
            info2.get(
                "vertical_speed",
                0.0,
            )
        )

        distance = float(
            info2.get(
                "forward_distance",
                getattr(
                    env2,
                    "forward_distance",
                    0.0,
                ),
            )
        )

        lat = lat_deg(
            fdm
        )

        lon = lon_deg(
            fdm
        )

        north, east = (
            local_ne(
                lat,
                lon,
                handoff_lat,
                handoff_lon,
            )
        )

        ground_forward, cross = (
            mission_projection(
                north,
                east,
                heading0,
            )
        )

        min_alt = min(
            min_alt,
            altitude,
        )

        max_alt = max(
            max_alt,
            altitude,
        )

        max_cross = max(
            max_cross,
            abs(
                cross
            ),
        )

        if detailed and (
            step2 % 35 == 0
        ):
            print(
                f"EVAL | "
                f"D={distance:7.2f} | "
                f"GND={ground_forward:7.2f} | "
                f"X={cross:+6.2f} | "
                f"ALT={altitude:7.2f} | "
                f"VS={vs:+6.2f}"
            )

        if (
            distance
            >=
            TARGET_FORWARD_DISTANCE
        ):
            crossing = {
                "distance":
                    distance,

                "ground_forward":
                    ground_forward,

                "cross":
                    cross,

                "altitude":
                    altitude,

                "vs":
                    vs,

                "time":
                    (
                        (step2 + 1)
                        *
                        dt2
                    ),
            }

            break

        if term2:
            if not bool(
                info2.get(
                    "success",
                    False,
                )
            ):
                physical_failure = True

            break

        if trunc2:
            break

    # Do not let closing env2 null the shared FDM before env1 cleanup.
    env2.fdm = None
    env1.close()

    reached = (
        crossing
        is not None
    )

    if not reached:
        return {
            "pass":
                False,

            "reason":
                "did_not_reach_300",

            "handoff_alt":
                handoff[
                    "altitude"
                ],

            "min_alt":
                min_alt,

            "max_alt":
                max_alt,

            "max_cross":
                max_cross,

            "physical_failure":
                physical_failure,
        }

    passed = bool(
        not physical_failure
        and
        min_alt
        >=
        290.0
        and
        max_alt
        <=
        310.0
        and
        max_cross
        <=
        5.0
        and
        295.0
        <=
        crossing[
            "altitude"
        ]
        <=
        305.0
        and
        abs(
            crossing[
                "vs"
            ]
        )
        <=
        2.0
    )

    return {
        "pass":
            passed,

        "reason":
            (
                "success"
                if passed
                else
                "quality_limits"
            ),

        "handoff_alt":
            handoff[
                "altitude"
            ],

        "handoff_vs":
            handoff[
                "vs"
            ],

        "min_alt":
            min_alt,

        "max_alt":
            max_alt,

        "max_drop":
            (
                handoff[
                    "altitude"
                ]
                -
                min_alt
            ),

        "max_cross":
            max_cross,

        "crossing_distance":
            crossing[
                "distance"
            ],

        "crossing_ground_forward":
            crossing[
                "ground_forward"
            ],

        "crossing_alt":
            crossing[
                "altitude"
            ],

        "crossing_vs":
            crossing[
                "vs"
            ],

        "crossing_cross":
            crossing[
                "cross"
            ],

        "crossing_time":
            crossing[
                "time"
            ],

        "physical_failure":
            physical_failure,
    }


# =====================================================================
# CALLBACK
# =====================================================================

class ContinuousHandoffEvalCallback(
    BaseCallback
):
    def __init__(
        self,
        eval_every,
        verbose=1,
    ):
        super().__init__(
            verbose=verbose
        )

        self.eval_every = int(
            eval_every
        )

        self.next_eval = int(
            eval_every
        )

        self.success_found = False


    def _on_step(
        self,
    ):
        if (
            self.num_timesteps
            <
            self.next_eval
        ):
            return True

        print(
            "\n"
            +
            "=" * 120
        )

        print(
            f"TRUE CONTINUOUS HANDOFF EVAL "
            f"@ {self.num_timesteps} STEPS"
        )

        print(
            "=" * 120
        )

        result = (
            run_continuous_handoff_eval(
                self.model,
                detailed=False,
            )
        )

        if (
            "handoff_alt"
            in result
        ):
            print(
                f"HANDOFF ALT : "
                f"{result.get('handoff_alt', float('nan')):.2f}"
            )

        print(
            f"REASON      : "
            f"{result.get('reason')}"
        )

        print(
            f"MIN ALT     : "
            f"{result.get('min_alt', float('nan')):.2f}"
        )

        print(
            f"MAX ALT     : "
            f"{result.get('max_alt', float('nan')):.2f}"
        )

        print(
            f"MAX DROP    : "
            f"{result.get('max_drop', float('nan')):.2f}"
        )

        print(
            f"MAX XTRACK  : "
            f"{result.get('max_cross', float('nan')):.2f}"
        )

        print(
            f"CROSS ALT   : "
            f"{result.get('crossing_alt', float('nan')):.2f}"
        )

        print(
            f"CROSS VS    : "
            f"{result.get('crossing_vs', float('nan')):+.2f}"
        )

        print(
            f"CROSS X     : "
            f"{result.get('crossing_cross', float('nan')):+.2f}"
        )

        print(
            f"PASS        : "
            f"{result.get('pass', False)}"
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            /
            (
                "AH1S_STAGE2_QUALITY_"
                f"{self.num_timesteps}"
            )
        )

        self.model.save(
            checkpoint_path
        )

        print(
            "Checkpoint:",
            str(
                checkpoint_path
            )
            +
            ".zip"
        )

        self.next_eval += (
            self.eval_every
        )

        if bool(
            result.get(
                "pass",
                False,
            )
        ):
            self.model.save(
                SUCCESS_MODEL_PATH
            )

            self.success_found = True

            print(
                "\n"
                +
                "STAGE 2 QUALITY SUCCESS FOUND"
            )

            print(
                "Saved:",
                str(
                    SUCCESS_MODEL_PATH
                )
                +
                ".zip"
            )

            print(
                "Training auto-stopped."
            )

            return False

        return True


# =====================================================================
# START
# =====================================================================

print(
    "=" * 120
)

print(
    "AH-1S STAGE 2 QUALITY REFINEMENT"
)

print(
    "=" * 120
)

print(
    "\nSource Stage-2 model:"
)

print(
    SOURCE_STAGE2_MODEL
)

print(
    "\nGoal:"
)

print(
    "300 ft forward while maintaining altitude and straight track."
)

print(
    "\nLocked Stage-1 model:"
)

print(
    STAGE1_MODEL_PATH
)

print(
    "\nNo Stage-1 retraining."
)

print(
    "No architecture change."
)

print(
    "Stage-2 observation/action spaces remain unchanged."
)


# =====================================================================
# BEFORE TRAINING — TRUE CONTINUOUS BASELINE
# =====================================================================

print(
    "\n"
    +
    "=" * 120
)

print(
    "BASELINE TRUE CONTINUOUS HANDOFF"
)

print(
    "=" * 120
)

source_model = PPO.load(
    SOURCE_STAGE2_MODEL
)

baseline = (
    run_continuous_handoff_eval(
        source_model,
        detailed=False,
    )
)

for key, value in baseline.items():
    print(
        f"{key:26s}: {value}"
    )


# =====================================================================
# TRAIN ENV
# =====================================================================

train_env = Monitor(
    HelicopterEnvStage2Quality()
)

model = PPO.load(
    SOURCE_STAGE2_MODEL,
    env=train_env,
    learning_rate=LEARNING_RATE,
    target_kl=0.015,
    ent_coef=0.0,
    device="auto",
)

print(
    "\nFine-tune learning rate:",
    LEARNING_RATE
)

print(
    "Maximum timesteps:",
    TOTAL_TIMESTEPS
)

print(
    "True handoff evaluation every:",
    EVAL_EVERY
)


callback = (
    ContinuousHandoffEvalCallback(
        eval_every=EVAL_EVERY,
        verbose=1,
    )
)


# =====================================================================
# TRAIN
# =====================================================================

print(
    "\n"
    +
    "=" * 120
)

print(
    "TRAINING"
)

print(
    "=" * 120
)

model.learn(
    total_timesteps=
        TOTAL_TIMESTEPS,

    callback=
        callback,

    reset_num_timesteps=
        True,

    progress_bar=
        True,
)


# =====================================================================
# SAVE FINAL
# =====================================================================

model.save(
    FINAL_MODEL_PATH
)

print(
    "\nFinal model saved:"
)

print(
    str(
        FINAL_MODEL_PATH
    )
    +
    ".zip"
)


# =====================================================================
# FINAL TRUE CONTINUOUS VALIDATION
# =====================================================================

print(
    "\n"
    +
    "=" * 120
)

print(
    "FINAL TRUE CONTINUOUS HANDOFF VALIDATION"
)

print(
    "=" * 120
)

if callback.success_found:
    final_model = PPO.load(
        str(
            SUCCESS_MODEL_PATH
        )
        +
        ".zip"
    )
else:
    final_model = model


final_result = (
    run_continuous_handoff_eval(
        final_model,
        detailed=True,
    )
)

print(
    "\n"
    +
    "=" * 120
)

print(
    "STAGE 2 QUALITY FINAL RESULT"
)

print(
    "=" * 120
)

for key, value in (
    final_result.items()
):
    print(
        f"{key:26s}: {value}"
    )


if bool(
    final_result.get(
        "pass",
        False,
    )
):
    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "STAGE 1 + STAGE 2 LOCKED"
    )

    print(
        "=" * 120
    )

    print(
        "Takeoff -> 300 ft hover -> 300 ft forward: PASS"
    )

    print(
        "\nNEXT: forward-flight stop / transition, then vertical descent."
    )

else:
    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "STAGE 2 NOT LOCKED YET"
    )

    print(
        "=" * 120
    )

    print(
        "Do not alter Stage 1."
    )

    print(
        "Use the final MIN ALT / MAX XTRACK values to decide "
        "whether Stage 2 needs more altitude or lateral refinement."
    )
