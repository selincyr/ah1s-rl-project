import numpy as np

from helicopter_env_stage1_4action_v2 import (
    HelicopterEnvStage1FourActionV2
)


class HelicopterEnvStage1CurriculumV3(
    HelicopterEnvStage1FourActionV2
):

    # ========================================================
    # PHASE NAMES
    # ========================================================

    PHASE_XY_EARLY = "xy_early"
    PHASE_XY_FULL = "xy_full"
    PHASE_JOINT = "joint"
    PHASE_FINAL = "final"


    def __init__(
        self,
        phase="xy_early",
        teacher_model_path=(
            "models_v2/"
            "AH1S_STAGE1_SUCCESS.zip"
        )
    ):

        self.phase = phase

        training_mode = (
            phase
            !=
            self.PHASE_FINAL
        )

        if phase == self.PHASE_FINAL:

            teacher_model_path = None

        super().__init__(
            teacher_model_path=
                teacher_model_path,

            training_mode=
                training_mode
        )

        # --------------------------------------------
        # More cyclic authority.
        #
        # Still limited / safe.
        # --------------------------------------------

        self.ELEVATOR_AUTHORITY = 0.025
        self.AILERON_AUTHORITY = 0.025
        self.RUDDER_AUTHORITY = 0.025

        self.previous_abs_north = 0.0
        self.previous_abs_east = 0.0

        # Phase C starts its own teacher fade counter
        self.phase_training_steps = 0


    # ========================================================
    # TEACHER RATIO
    # ========================================================

    def _teacher_ratio(self):

        # ----------------------------------------------------
        # PHASE A/B
        #
        # Teacher physically controls ONLY collective.
        #
        # PPO still outputs four actions.
        # Cyclic/pedal completely PPO.
        # ----------------------------------------------------

        if self.phase in (
            self.PHASE_XY_EARLY,
            self.PHASE_XY_FULL
        ):

            return 1.0


        # ----------------------------------------------------
        # PHASE C
        #
        # Teacher gradually disappears.
        # ----------------------------------------------------

        if self.phase == self.PHASE_JOINT:

            if self.teacher_model is None:

                return 0.0

            # First 10K fully teacher collective
            if self.phase_training_steps < 10_000:

                return 1.0

            # 10K -> 70K fade
            if self.phase_training_steps < 70_000:

                progress = (
                    self.phase_training_steps
                    -
                    10_000
                ) / 60_000.0

                return float(
                    1.0
                    -
                    progress
                )

            return 0.0


        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        return 0.0


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        self.previous_abs_north = 0.0
        self.previous_abs_east = 0.0

        return obs, info


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action
    ):

        if self.phase == self.PHASE_JOINT:

            self.phase_training_steps += 1


        (
            obs,
            _old_reward,
            terminated,
            truncated,
            info
        ) = super().step(
            action
        )


        # ====================================================
        # CURRENT STATE
        # ====================================================

        north = float(
            info["north"]
        )

        east = float(
            info["east"]
        )

        vn = float(
            info["vn"]
        )

        ve = float(
            info["ve"]
        )

        drift = float(
            info["drift"]
        )

        max_drift = float(
            info["max_drift"]
        )

        horizontal_speed = float(
            info["horizontal_speed"]
        )

        altitude = float(
            info["altitude"]
        )

        altitude_error = float(
            info["altitude_error"]
        )

        vertical_speed = float(
            info["vertical_speed"]
        )

        heading_error = float(
            info["heading_error"]
        )

        pitch = float(
            info["pitch"]
        )

        roll = float(
            info["roll"]
        )


        # ====================================================
        # DIRECT AXIS PROGRESS
        #
        # This is the key V3 change.
        #
        # Instead of only saying "drift bad",
        # PPO gets immediate feedback:
        #
        # did |N| shrink?
        # did |E| shrink?
        # ====================================================

        abs_north = abs(
            north
        )

        abs_east = abs(
            east
        )


        north_progress = (
            self.previous_abs_north
            -
            abs_north
        )

        east_progress = (
            self.previous_abs_east
            -
            abs_east
        )


        self.previous_abs_north = (
            abs_north
        )

        self.previous_abs_east = (
            abs_east
        )


        # ====================================================
        # OUTWARD VELOCITY PER AXIS
        #
        # N > 0 and VN > 0 -> moving farther out
        #
        # N < 0 and VN < 0 -> moving farther out
        #
        # Same for East.
        # ====================================================

        if abs_north > 0.25:

            north_outward = max(
                0.0,
                np.sign(north)
                *
                vn
            )

        else:

            north_outward = abs(
                vn
            )


        if abs_east > 0.25:

            east_outward = max(
                0.0,
                np.sign(east)
                *
                ve
            )

        else:

            east_outward = abs(
                ve
            )


        # ====================================================
        # COMMON XY REWARD
        # ====================================================

        reward = 0.0


        # ----------------------------------------------------
        # 1. CENTER BONUS
        # ----------------------------------------------------

        reward += (
            4.0
            *
            float(
                np.exp(
                    -0.5
                    *
                    (
                        drift
                        /
                        3.0
                    ) ** 2
                )
            )
        )


        # ----------------------------------------------------
        # 2. ABSOLUTE POSITION
        # ----------------------------------------------------

        reward -= (
            0.15
            *
            min(
                abs_north,
                25.0
            )
        )

        reward -= (
            0.15
            *
            min(
                abs_east,
                25.0
            )
        )


        # ----------------------------------------------------
        # 3. DIRECT POSITION PROGRESS
        # ----------------------------------------------------

        reward += (
            6.0
            *
            float(
                np.clip(
                    north_progress,
                    -0.20,
                    0.20
                )
            )
        )

        reward += (
            6.0
            *
            float(
                np.clip(
                    east_progress,
                    -0.20,
                    0.20
                )
            )
        )


        # ----------------------------------------------------
        # 4. VELOCITY
        # ----------------------------------------------------

        reward -= (
            0.30
            *
            min(
                abs(vn),
                6.0
            )
        )

        reward -= (
            0.30
            *
            min(
                abs(ve),
                6.0
            )
        )


        # ----------------------------------------------------
        # 5. MOVING AWAY FROM CENTER
        # ----------------------------------------------------

        reward -= (
            1.20
            *
            min(
                north_outward,
                5.0
            )
        )

        reward -= (
            1.20
            *
            min(
                east_outward,
                5.0
            )
        )


        # ----------------------------------------------------
        # 6. PATH / S-SHAPE PENALTY
        # ----------------------------------------------------

        reward -= (
            0.20
            *
            min(
                horizontal_speed,
                8.0
            )
        )


        # ----------------------------------------------------
        # 7. HEADING
        # ----------------------------------------------------

        reward -= (
            0.40
            *
            min(
                abs(
                    heading_error
                )
                /
                0.15,
                2.0
            )
        )


        # ----------------------------------------------------
        # 8. ATTITUDE
        # ----------------------------------------------------

        reward -= (
            0.20
            *
            min(
                abs(pitch)
                /
                0.20,
                2.0
            )
        )

        reward -= (
            0.20
            *
            min(
                abs(roll)
                /
                0.20,
                2.0
            )
        )


        # ====================================================
        # COLLECTIVE IMITATION
        #
        # Student learns teacher's action[0] even while
        # teacher is physically flying collective.
        # ====================================================

        student_a0 = float(
            info["student_a0"]
        )

        teacher_a0 = (
            info["teacher_a0"]
        )


        if np.isfinite(
            teacher_a0
        ):

            reward -= (
                4.0
                *
                abs(
                    student_a0
                    -
                    teacher_a0
                )
            )


        # ====================================================
        # PHASE-SPECIFIC REWARD
        # ====================================================

        if self.phase == self.PHASE_XY_EARLY:

            # ------------------------------------------------
            # PHASE A
            #
            # Ignore mission completion.
            # Learn ONLY:
            #
            # "while climbing, do not leave X=0,Y=0"
            # ------------------------------------------------

            # Tiny survival bonus
            reward += 0.5


            # Episode only lasts 35 seconds.
            #
            # Short horizon =
            # much easier credit assignment.
            if (
                self.steps
                *
                self.dt
                >=
                35.0
            ):

                truncated = True


            # Extra strong penalty outside 8 ft
            if drift > 8.0:

                reward -= (
                    0.80
                    *
                    (
                        drift
                        -
                        8.0
                    )
                )


        elif self.phase == self.PHASE_XY_FULL:

            # ------------------------------------------------
            # PHASE B
            #
            # Teacher handles collective.
            #
            # PPO must maintain XY during ENTIRE climb/hover.
            # ------------------------------------------------

            reward += 0.5


            # Target altitude reached
            if altitude_error < 15.0:

                reward += 1.0


            if (
                altitude_error < 10.0
                and
                abs(
                    vertical_speed
                ) < 1.0
                and
                drift < 5.0
            ):

                reward += 3.0


            if drift > 8.0:

                reward -= (
                    1.0
                    *
                    (
                        drift
                        -
                        8.0
                    )
                )


        elif self.phase == self.PHASE_JOINT:

            # ------------------------------------------------
            # PHASE C
            #
            # Now altitude + XY matter together.
            # Teacher collective gradually disappears.
            # ------------------------------------------------

            signed_altitude_error = (
                300.0
                -
                altitude
            )


            desired_vs = float(
                np.clip(
                    0.05
                    *
                    signed_altitude_error,
                    -3.0,
                    6.0
                )
            )


            vs_error = abs(
                vertical_speed
                -
                desired_vs
            )


            # Altitude center bonus
            reward += (
                2.0
                *
                float(
                    np.exp(
                        -0.5
                        *
                        (
                            altitude_error
                            /
                            10.0
                        ) ** 2
                    )
                )
            )


            # VS tracking
            reward += (
                1.0
                *
                (
                    1.0
                    -
                    min(
                        vs_error
                        /
                        6.0,
                        1.0
                    )
                )
            )


            # Away from 300 is expensive
            if altitude_error > 10.0:

                reward -= (
                    0.06
                    *
                    min(
                        altitude_error
                        -
                        10.0,
                        100.0
                    )
                )


            # Overshoot especially expensive
            if altitude > 315.0:

                reward -= (
                    0.20
                    *
                    (
                        altitude
                        -
                        315.0
                    )
                )


            if altitude < 285.0:

                # Only once aircraft should already be near
                # target; lower-climb altitudes are handled
                # naturally by desired VS.
                if self.steps * self.dt > 45.0:

                    reward -= (
                        0.10
                        *
                        (
                            285.0
                            -
                            altitude
                        )
                    )


            # Strict center band
            if drift > 8.0:

                reward -= (
                    1.0
                    *
                    (
                        drift
                        -
                        8.0
                    )
                )


            if (
                altitude_error < 10.0
                and
                abs(
                    vertical_speed
                ) < 1.0
                and
                drift < 5.0
                and
                horizontal_speed < 1.5
            ):

                reward += 5.0


        # ====================================================
        # FINAL SUCCESS BONUS
        # ====================================================

        if info["success"]:

            reward += 1500.0


        # ====================================================
        # FAILURE
        # ====================================================

        if terminated and (
            not info["success"]
        ):

            reward -= 200.0


        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )
