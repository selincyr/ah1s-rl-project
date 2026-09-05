import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO

from helicopter_env_v2 import HelicopterEnvV2


class HelicopterEnvStage1FourActionV2(gym.Env):

    metadata = {
        "render_modes": []
    }

    # ========================================================
    # MISSION
    # ========================================================

    TARGET_ALTITUDE = 300.0
    MAX_TIME = 120.0

    # STRICT GEOMETRY
    SUCCESS_MAX_DRIFT = 8.0
    SUCCESS_FINAL_DRIFT = 5.0
    SUCCESS_MAX_PATH = 25.0

    # Sadece safety.
    # Toparlanmayı öğrenebilmesi için çok erken öldürmüyoruz.
    FAILURE_DRIFT = 40.0

    # ========================================================
    # TRIMS
    # ========================================================

    BASE_ELEVATOR = -0.15390
    BASE_AILERON = 0.19100
    BASE_RUDDER = 0.39000

    # ========================================================
    # ACTION AUTHORITY
    # ========================================================

    ELEVATOR_AUTHORITY = 0.020
    AILERON_AUTHORITY = 0.020
    RUDDER_AUTHORITY = 0.025

    # ========================================================
    # COLLECTIVE CURRICULUM
    # ========================================================

    TEACHER_FULL_STEPS = 30_000
    TEACHER_FADE_END = 60_000


    def __init__(
        self,
        teacher_model_path=(
            "models_v2/"
            "AH1S_STAGE1_SUCCESS.zip"
        ),
        training_mode=True
    ):

        super().__init__()

        self.base_env = HelicopterEnvV2()

        self.fdm = self.base_env.fdm

        self.training_mode = bool(
            training_mode
        )

        self.teacher_model = None

        if (
            teacher_model_path is not None
            and self.training_mode
        ):

            self.teacher_model = PPO.load(
                teacher_model_path
            )

        # ====================================================
        # 4 REAL ACTIONS
        # ====================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # ====================================================
        # OBSERVATION
        #
        # Original 12 +
        # N / E / VN / VE / heading error / yaw rate
        # ====================================================

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(18,),
            dtype=np.float32
        )

        self.dt = (
            self.base_env.JSBSIM_DT
            *
            self.base_env.PHYSICS_STEPS
        )

        self.max_steps = int(
            self.MAX_TIME / self.dt
        )

        # IMPORTANT:
        # reset() bunu sıfırlamayacak.
        # Curriculum bütün eğitim boyunca ilerler.
        self.global_training_steps = 0

        self.steps = 0

        self.north = 0.0
        self.east = 0.0

        self.max_drift = 0.0
        self.horizontal_path = 0.0

        self.initial_heading = 0.0

        self.previous_altitude_error = 0.0
        self.previous_drift = 0.0

        self.previous_action = np.zeros(
            4,
            dtype=np.float32
        )

        self.previous_elevator = (
            self.BASE_ELEVATOR
        )

        self.previous_aileron = (
            self.BASE_AILERON
        )

        self.previous_rudder = (
            self.BASE_RUDDER
        )

        self.stable_steps = 0


    # ========================================================
    # ANGLE
    # ========================================================

    @staticmethod
    def _wrap_angle(angle):

        return float(
            np.arctan2(
                np.sin(angle),
                np.cos(angle)
            )
        )


    # ========================================================
    # STATE
    # ========================================================

    def _state(self):

        raw = self.base_env._raw_state()

        fdm = self.base_env.fdm

        vn = float(
            fdm["velocities/v-north-fps"]
        )

        ve = float(
            fdm["velocities/v-east-fps"]
        )

        heading = float(
            fdm["attitude/psi-rad"]
        )

        yaw_rate = float(
            fdm["velocities/r-rad_sec"]
        )

        heading_error = self._wrap_angle(
            heading - self.initial_heading
        )

        return {
            "altitude":
                float(raw["altitude"]),

            "vertical_speed":
                float(raw["vertical_speed"]),

            "forward_velocity":
                float(
                    raw["forward_velocity"]
                ),

            "lateral_velocity":
                float(
                    raw["lateral_velocity"]
                ),

            "pitch":
                float(raw["pitch"]),

            "roll":
                float(raw["roll"]),

            "vn":
                vn,

            "ve":
                ve,

            "heading":
                heading,

            "heading_error":
                heading_error,

            "yaw_rate":
                yaw_rate,
        }


    # ========================================================
    # OBS
    # ========================================================

    def _get_obs(self):

        base_obs = (
            self.base_env._get_obs()
        ).astype(
            np.float32
        )

        s = self._state()

        extra = np.array(
            [
                np.clip(
                    self.north / 10.0,
                    -10.0,
                    10.0
                ),

                np.clip(
                    self.east / 10.0,
                    -10.0,
                    10.0
                ),

                np.clip(
                    s["vn"] / 5.0,
                    -10.0,
                    10.0
                ),

                np.clip(
                    s["ve"] / 5.0,
                    -10.0,
                    10.0
                ),

                np.clip(
                    s["heading_error"] / 0.20,
                    -10.0,
                    10.0
                ),

                np.clip(
                    s["yaw_rate"] / 0.50,
                    -10.0,
                    10.0
                ),
            ],
            dtype=np.float32
        )

        return np.concatenate(
            [
                base_obs,
                extra
            ]
        ).astype(
            np.float32
        )


    # ========================================================
    # TEACHER RATIO
    # ========================================================

    def _teacher_ratio(self):

        if (
            not self.training_mode
            or self.teacher_model is None
        ):

            return 0.0

        step = self.global_training_steps

        if step <= self.TEACHER_FULL_STEPS:

            return 1.0

        if step >= self.TEACHER_FADE_END:

            return 0.0

        progress = (
            step
            -
            self.TEACHER_FULL_STEPS
        ) / (
            self.TEACHER_FADE_END
            -
            self.TEACHER_FULL_STEPS
        )

        return float(
            1.0 - progress
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.steps = 0

        self.north = 0.0
        self.east = 0.0

        self.max_drift = 0.0
        self.horizontal_path = 0.0

        self.stable_steps = 0

        self.previous_action = np.zeros(
            4,
            dtype=np.float32
        )

        self.previous_elevator = (
            self.BASE_ELEVATOR
        )

        self.previous_aileron = (
            self.BASE_AILERON
        )

        self.previous_rudder = (
            self.BASE_RUDDER
        )

        self.base_env.reset(
            seed=seed,
            options=options
        )

        self.fdm = self.base_env.fdm

        self.initial_heading = float(
            self.fdm[
                "attitude/psi-rad"
            ]
        )

        s = self._state()

        self.previous_altitude_error = abs(
            self.TARGET_ALTITUDE
            -
            s["altitude"]
        )

        self.previous_drift = 0.0

        return (
            self._get_obs(),
            {
                "success": False,
                "north": 0.0,
                "east": 0.0,
                "drift": 0.0,
                "max_drift": 0.0,
                "path": 0.0,
            }
        )


    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        self.steps += 1

        if self.training_mode:

            self.global_training_steps += 1


        # ====================================================
        # TEACHER COLLECTIVE ACTION
        # ====================================================

        teacher_a0 = None

        if self.teacher_model is not None:

            teacher_obs = (
                self.base_env._get_obs()
            )

            teacher_action, _ = (
                self.teacher_model.predict(
                    teacher_obs,
                    deterministic=True
                )
            )

            teacher_action = np.asarray(
                teacher_action,
                dtype=np.float32
            )

            teacher_a0 = float(
                np.clip(
                    teacher_action[0],
                    -1.0,
                    1.0
                )
            )


        # ====================================================
        # STUDENT 4 ACTIONS
        # ====================================================

        student_a0 = float(action[0])

        teacher_ratio = (
            self._teacher_ratio()
        )


        # ====================================================
        # COLLECTIVE CURRICULUM
        #
        # Training:
        # teacher -> gradually student
        #
        # Evaluation:
        # 100% student
        # ====================================================

        if teacher_a0 is not None:

            used_a0 = (
                teacher_ratio
                *
                teacher_a0
                +
                (
                    1.0
                    -
                    teacher_ratio
                )
                *
                student_a0
            )

        else:

            used_a0 = student_a0


        used_a0 = float(
            np.clip(
                used_a0,
                -1.0,
                1.0
            )
        )


        collective = (
            0.620
            +
            0.030
            *
            used_a0
        )

        collective = float(
            np.clip(
                collective,
                0.590,
                0.650
            )
        )


        # ====================================================
        # ELEVATOR
        # ====================================================

        elevator_target = (
            self.BASE_ELEVATOR
            +
            self.ELEVATOR_AUTHORITY
            *
            float(action[1])
        )

        elevator_target = float(
            np.clip(
                elevator_target,
                -0.1760,
                -0.1320
            )
        )


        # ====================================================
        # AILERON
        # ====================================================

        aileron_target = (
            self.BASE_AILERON
            +
            self.AILERON_AUTHORITY
            *
            float(action[2])
        )

        aileron_target = float(
            np.clip(
                aileron_target,
                0.1680,
                0.2140
            )
        )


        # ====================================================
        # RUDDER
        # ====================================================

        rudder_target = (
            self.BASE_RUDDER
            +
            self.RUDDER_AUTHORITY
            *
            float(action[3])
        )

        rudder_target = float(
            np.clip(
                rudder_target,
                0.3500,
                0.4300
            )
        )


        # ====================================================
        # SMOOTH CYCLIC / PEDAL
        # ====================================================

        alpha = 0.25

        elevator = (
            self.previous_elevator
            +
            alpha
            *
            (
                elevator_target
                -
                self.previous_elevator
            )
        )

        aileron = (
            self.previous_aileron
            +
            alpha
            *
            (
                aileron_target
                -
                self.previous_aileron
            )
        )

        rudder = (
            self.previous_rudder
            +
            alpha
            *
            (
                rudder_target
                -
                self.previous_rudder
            )
        )

        self.previous_elevator = elevator
        self.previous_aileron = aileron
        self.previous_rudder = rudder


        # ====================================================
        # APPLY ALL FOUR
        # ====================================================

        fdm = self.base_env.fdm

        fdm[
            "fcs/collective-cmd-norm"
        ] = collective

        fdm[
            "fcs/elevator-cmd-norm"
        ] = elevator

        fdm[
            "fcs/aileron-cmd-norm"
        ] = aileron

        fdm[
            "fcs/rudder-cmd-norm"
        ] = rudder


        # ====================================================
        # PHYSICS
        # ====================================================

        physics_ok = True

        for _ in range(
            self.base_env.PHYSICS_STEPS
        ):

            if not fdm.run():

                physics_ok = False
                break


        s = self._state()


        # ====================================================
        # POSITION
        # ====================================================

        self.north += (
            s["vn"]
            *
            self.dt
        )

        self.east += (
            s["ve"]
            *
            self.dt
        )

        horizontal_speed = float(
            np.hypot(
                s["vn"],
                s["ve"]
            )
        )

        path_increment = (
            horizontal_speed
            *
            self.dt
        )

        self.horizontal_path += (
            path_increment
        )

        drift = float(
            np.hypot(
                self.north,
                self.east
            )
        )

        self.max_drift = max(
            self.max_drift,
            drift
        )


        # ====================================================
        # ALTITUDE PROFILE
        # ====================================================

        signed_alt_error = (
            self.TARGET_ALTITUDE
            -
            s["altitude"]
        )

        altitude_error = abs(
            signed_alt_error
        )

        # CRITICAL FIX:
        # Negative desired VS is allowed above target.
        desired_vs = float(
            np.clip(
                0.05
                *
                signed_alt_error,
                -3.0,
                6.0
            )
        )

        vs_error = abs(
            desired_vs
            -
            s["vertical_speed"]
        )

        # Progress TOWARD 300.
        # Going above 300 and continuing upward becomes bad.
        altitude_progress = (
            self.previous_altitude_error
            -
            altitude_error
        )

        self.previous_altitude_error = (
            altitude_error
        )


        # ====================================================
        # HORIZONTAL PROGRESS
        # ====================================================

        drift_progress = (
            self.previous_drift
            -
            drift
        )

        self.previous_drift = drift


        # ====================================================
        # OUTWARD VELOCITY
        # ====================================================

        if drift > 0.25:

            radial_velocity = (
                self.north
                *
                s["vn"]
                +
                self.east
                *
                s["ve"]
            ) / drift

        else:

            radial_velocity = 0.0

        outward_velocity = max(
            0.0,
            radial_velocity
        )


        # ====================================================
        # REWARD
        # ====================================================

        reward = 0.0


        # ----------------------------------------------------
        # A. ALTITUDE - progress toward 300
        # ----------------------------------------------------

        reward += (
            1.50
            *
            float(
                np.clip(
                    altitude_progress,
                    -1.0,
                    1.0
                )
            )
        )


        # ----------------------------------------------------
        # B. VERTICAL SPEED TRACKING
        # ----------------------------------------------------

        reward += (
            1.00
            *
            (
                1.0
                -
                min(
                    vs_error / 6.0,
                    1.0
                )
            )
        )


        # ----------------------------------------------------
        # C. TARGET ALTITUDE BONUS
        # ----------------------------------------------------

        reward += (
            1.00
            *
            float(
                np.exp(
                    -0.5
                    *
                    (
                        altitude_error
                        /
                        12.0
                    ) ** 2
                )
            )
        )


        # Strong overshoot/undershoot penalty
        if altitude_error > 20.0:

            reward -= (
                0.02
                *
                min(
                    altitude_error - 20.0,
                    100.0
                )
            )


        # Extra punishment above 320 ft.
        if s["altitude"] > 320.0:

            reward -= (
                0.10
                *
                min(
                    s["altitude"]
                    -
                    320.0,
                    70.0
                )
            )


        # ----------------------------------------------------
        # D. CENTERLINE
        # ----------------------------------------------------

        center_bonus = float(
            np.exp(
                -0.5
                *
                (
                    drift / 3.0
                ) ** 2
            )
        )

        reward += (
            2.0
            *
            center_bonus
        )

        reward -= (
            0.12
            *
            min(
                drift,
                25.0
            )
        )


        # Immediate signal:
        # did we move toward center this step?
        reward += (
            2.0
            *
            float(
                np.clip(
                    drift_progress,
                    -0.30,
                    0.30
                )
            )
        )


        # ----------------------------------------------------
        # E. NO S / ARC
        # ----------------------------------------------------

        reward -= (
            0.20
            *
            min(
                horizontal_speed,
                8.0
            )
        )

        reward -= (
            0.40
            *
            min(
                outward_velocity,
                5.0
            )
        )

        reward -= (
            0.10
            *
            min(
                path_increment,
                1.0
            )
        )


        # ----------------------------------------------------
        # F. HEADING
        # ----------------------------------------------------

        reward -= (
            0.50
            *
            min(
                abs(
                    s["heading_error"]
                )
                /
                0.15,
                2.0
            )
        )


        # ----------------------------------------------------
        # G. ATTITUDE
        # ----------------------------------------------------

        reward -= (
            0.25
            *
            min(
                abs(
                    s["pitch"]
                )
                /
                0.20,
                2.0
            )
        )

        reward -= (
            0.25
            *
            min(
                abs(
                    s["roll"]
                )
                /
                0.20,
                2.0
            )
        )


        # ----------------------------------------------------
        # H. STUDENT COLLECTIVE IMITATION
        #
        # Even while teacher is physically used,
        # student learns what teacher would do.
        # ----------------------------------------------------

        if teacher_a0 is not None:

            collective_imitation_error = abs(
                student_a0
                -
                teacher_a0
            )

            reward -= (
                2.0
                *
                collective_imitation_error
            )


        # ----------------------------------------------------
        # I. SMOOTH ACTIONS
        # ----------------------------------------------------

        action_change = float(
            np.mean(
                np.abs(
                    action
                    -
                    self.previous_action
                )
            )
        )

        reward -= (
            0.05
            *
            action_change
        )

        self.previous_action = (
            action.copy()
        )


        # ====================================================
        # STABLE HOVER
        # ====================================================

        stable_now = (

            altitude_error < 10.0

            and

            abs(
                s["vertical_speed"]
            ) < 1.0

            and

            horizontal_speed < 1.5

            and

            drift < 5.0

            and

            abs(
                s["heading_error"]
            ) < 0.08

            and

            abs(
                s["pitch"]
            ) < 0.12

            and

            abs(
                s["roll"]
            ) < 0.12
        )

        if stable_now:

            self.stable_steps += 1

            reward += 3.0

        else:

            self.stable_steps = 0


        required_stable_steps = int(
            10.0 / self.dt
        )


        # ====================================================
        # STRICT SUCCESS
        # ====================================================

        success = bool(

            self.stable_steps
            >=
            required_stable_steps

            and

            self.max_drift
            <=
            self.SUCCESS_MAX_DRIFT

            and

            drift
            <=
            self.SUCCESS_FINAL_DRIFT

            and

            self.horizontal_path
            <=
            self.SUCCESS_MAX_PATH
        )

        if success:

            reward += 1000.0


        # ====================================================
        # FAILURE
        # ====================================================

        failure = False

        if not physics_ok:

            failure = True

        if drift > self.FAILURE_DRIFT:

            failure = True

        if s["altitude"] > 380.0:

            failure = True

        if abs(
            s["pitch"]
        ) > 0.60:

            failure = True

        if abs(
            s["roll"]
        ) > 0.60:

            failure = True

        if failure:

            reward -= 150.0


        terminated = bool(
            success
            or
            failure
        )

        truncated = bool(
            self.steps >= self.max_steps
        )


        return (
            self._get_obs(),

            float(reward),

            terminated,

            truncated,

            {
                "success":
                    success,

                "altitude":
                    s["altitude"],

                "altitude_error":
                    altitude_error,

                "desired_vs":
                    desired_vs,

                "vertical_speed":
                    s["vertical_speed"],

                "north":
                    self.north,

                "east":
                    self.east,

                "vn":
                    s["vn"],

                "ve":
                    s["ve"],

                "horizontal_speed":
                    horizontal_speed,

                "drift":
                    drift,

                "max_drift":
                    self.max_drift,

                "path":
                    self.horizontal_path,

                "heading_error":
                    s["heading_error"],

                "pitch":
                    s["pitch"],

                "roll":
                    s["roll"],

                "student_a0":
                    student_a0,

                "teacher_a0":
                    (
                        teacher_a0
                        if teacher_a0 is not None
                        else np.nan
                    ),

                "teacher_ratio":
                    teacher_ratio,

                "collective":
                    collective,

                "elevator":
                    elevator,

                "aileron":
                    aileron,

                "rudder":
                    rudder,
            }
        )


    def close(self):

        if self.base_env is not None:

            self.base_env.close()
