import gymnasium as gym
from gymnasium import spaces

import numpy as np

from stable_baselines3 import PPO

from helicopter_env_v2 import HelicopterEnvV2


class HelicopterEnvStage1Distill(gym.Env):

    metadata = {
        "render_modes": []
    }

    # ========================================================
    # MISSION
    # ========================================================

    TARGET_ALTITUDE = 300.0

    MAX_TIME = 110.0

    SUCCESS_MAX_DRIFT = 8.0
    SUCCESS_FINAL_DRIFT = 5.0
    SUCCESS_MAX_PATH = 25.0


    # ========================================================
    # CONTROL TRIMS
    # ========================================================

    BASE_ELEVATOR = -0.15390
    BASE_AILERON = 0.19100
    BASE_RUDDER = 0.39000


    # ========================================================
    # STUDENT ACTION AUTHORITY
    # ========================================================

    ELEVATOR_AUTHORITY = 0.026
    AILERON_AUTHORITY = 0.026
    RUDDER_AUTHORITY = 0.020


    # ========================================================
    # IDENTIFIED XY CONTROLLER
    # ========================================================

    KP = 0.016
    KD = 0.200

    A_MAX = 0.12


    # This is inverse of B = G / 1.5
    B_INV = np.array(
        [
            [
                -0.20338265,
                -0.00859620
            ],

            [
                 0.01952073,
                -0.14631468
            ],
        ],
        dtype=np.float64
    )


    # ========================================================
    # INIT
    # ========================================================

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


        # ====================================================
        # TEACHER VERTICAL PPO
        # ====================================================

        self.teacher_model = None

        if teacher_model_path is not None:

            self.teacher_model = PPO.load(
                teacher_model_path
            )


        # ====================================================
        # TEACHER BLENDING
        #
        # 1.0 = teacher physically flies
        # 0.0 = student physically flies
        #
        # Training script changes this.
        # ====================================================

        self.teacher_blend = (
            1.0
            if training_mode
            else 0.0
        )


        # ====================================================
        # ACTIONS
        #
        # 0 collective
        # 1 elevator
        # 2 aileron
        # 3 rudder
        # ====================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )


        # ====================================================
        # OBS
        #
        # original Stage1 = 12
        #
        # +
        #
        # north
        # east
        # vn
        # ve
        # heading error
        # yaw rate
        #
        # = 18
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
            self.MAX_TIME
            /
            self.dt
        )


        # ====================================================
        # STATE
        # ====================================================

        self.steps = 0

        self.north = 0.0
        self.east = 0.0

        self.max_drift = 0.0
        self.horizontal_path = 0.0

        self.initial_heading = 0.0

        self.previous_elevator = (
            self.BASE_ELEVATOR
        )

        self.previous_aileron = (
            self.BASE_AILERON
        )

        self.previous_rudder = (
            self.BASE_RUDDER
        )

        self.previous_student_action = np.zeros(
            4,
            dtype=np.float32
        )

        self.stable_steps = 0


    # ========================================================
    # ANGLE WRAP
    # ========================================================

    @staticmethod
    def _wrap_angle(x):

        return float(
            np.arctan2(
                np.sin(x),
                np.cos(x)
            )
        )


    # ========================================================
    # STATE
    # ========================================================

    def _state(self):

        raw = (
            self.base_env._raw_state()
        )

        vn = float(
            self.fdm[
                "velocities/v-north-fps"
            ]
        )

        ve = float(
            self.fdm[
                "velocities/v-east-fps"
            ]
        )

        heading = float(
            self.fdm[
                "attitude/psi-rad"
            ]
        )

        yaw_rate = float(
            self.fdm[
                "velocities/r-rad_sec"
            ]
        )

        heading_error = (
            self._wrap_angle(
                heading
                -
                self.initial_heading
            )
        )


        return {
            "altitude":
                float(
                    raw["altitude"]
                ),

            "vertical_speed":
                float(
                    raw["vertical_speed"]
                ),

            "forward_velocity":
                float(
                    raw["forward_velocity"]
                ),

            "lateral_velocity":
                float(
                    raw["lateral_velocity"]
                ),

            "pitch":
                float(
                    raw["pitch"]
                ),

            "roll":
                float(
                    raw["roll"]
                ),

            "vn":
                vn,

            "ve":
                ve,

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
    # TEACHER ACTION
    #
    # Returns normalized four-action command.
    # ========================================================

    def get_teacher_action(self):

        if self.teacher_model is None:

            return np.zeros(
                4,
                dtype=np.float32
            )


        # ====================================================
        # COLLECTIVE TEACHER
        # ====================================================

        base_obs = (
            self.base_env._get_obs()
        )


        vertical_action, _ = (
            self.teacher_model.predict(
                base_obs,
                deterministic=True
            )
        )


        vertical_action = np.asarray(
            vertical_action,
            dtype=np.float32
        )


        teacher_a0 = float(
            np.clip(
                vertical_action[0],
                -1.0,
                1.0
            )
        )


        # ====================================================
        # XY TEACHER
        # ====================================================

        s = self._state()


        p = np.array(
            [
                self.north,
                self.east
            ],
            dtype=np.float64
        )


        v = np.array(
            [
                s["vn"],
                s["ve"]
            ],
            dtype=np.float64
        )


        desired_acceleration = (

            -self.KP
            *
            p

            -

            self.KD
            *
            v
        )


        desired_acceleration = np.clip(
            desired_acceleration,
            -self.A_MAX,
            self.A_MAX
        )


        cyclic_delta = (
            self.B_INV
            @
            desired_acceleration
        )


        elevator_delta = float(
            np.clip(
                cyclic_delta[0],
                -self.ELEVATOR_AUTHORITY,
                self.ELEVATOR_AUTHORITY
            )
        )


        aileron_delta = float(
            np.clip(
                cyclic_delta[1],
                -self.AILERON_AUTHORITY,
                self.AILERON_AUTHORITY
            )
        )


        teacher_a1 = (
            elevator_delta
            /
            self.ELEVATOR_AUTHORITY
        )


        teacher_a2 = (
            aileron_delta
            /
            self.AILERON_AUTHORITY
        )


        # Heading is already extremely stable.
        teacher_a3 = 0.0


        return np.array(
            [
                teacher_a0,
                teacher_a1,
                teacher_a2,
                teacher_a3
            ],
            dtype=np.float32
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


        self.previous_elevator = (
            self.BASE_ELEVATOR
        )

        self.previous_aileron = (
            self.BASE_AILERON
        )

        self.previous_rudder = (
            self.BASE_RUDDER
        )


        self.previous_student_action = (
            np.zeros(
                4,
                dtype=np.float32
            )
        )


        self.base_env.reset(
            seed=seed,
            options=options
        )


        self.fdm = (
            self.base_env.fdm
        )


        self.initial_heading = float(
            self.fdm[
                "attitude/psi-rad"
            ]
        )


        return (
            self._get_obs(),
            {
                "success": False
            }
        )


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action
    ):

        student_action = np.asarray(
            action,
            dtype=np.float32
        )


        student_action = np.clip(
            student_action,
            -1.0,
            1.0
        )


        self.steps += 1


        # ====================================================
        # TEACHER
        # ====================================================

        teacher_action = (
            self.get_teacher_action()
        )


        # ====================================================
        # BLEND
        #
        # IMPORTANT:
        #
        # student always outputs all 4 actions.
        #
        # Teacher is only training scaffold.
        # ====================================================

        blend = float(
            np.clip(
                self.teacher_blend,
                0.0,
                1.0
            )
        )


        if (
            self.training_mode
            and
            self.teacher_model
            is not None
        ):

            executed_action = (

                blend
                *
                teacher_action

                +

                (
                    1.0
                    -
                    blend
                )
                *
                student_action
            )

        else:

            executed_action = (
                student_action.copy()
            )


        executed_action = np.clip(
            executed_action,
            -1.0,
            1.0
        )


        # ====================================================
        # COLLECTIVE
        # ====================================================

        collective = (
            0.620
            +
            0.030
            *
            float(
                executed_action[0]
            )
        )


        collective = float(
            np.clip(
                collective,
                0.590,
                0.650
            )
        )


        # ====================================================
        # CYCLIC/PEDAL TARGETS
        # ====================================================

        elevator_target = (
            self.BASE_ELEVATOR
            +
            self.ELEVATOR_AUTHORITY
            *
            float(
                executed_action[1]
            )
        )


        aileron_target = (
            self.BASE_AILERON
            +
            self.AILERON_AUTHORITY
            *
            float(
                executed_action[2]
            )
        )


        rudder_target = (
            self.BASE_RUDDER
            +
            self.RUDDER_AUTHORITY
            *
            float(
                executed_action[3]
            )
        )


        elevator_target = float(
            np.clip(
                elevator_target,
                -0.1800,
                -0.1280
            )
        )


        aileron_target = float(
            np.clip(
                aileron_target,
                0.1650,
                0.2170
            )
        )


        rudder_target = float(
            np.clip(
                rudder_target,
                0.3500,
                0.4300
            )
        )


        # ====================================================
        # SAME SMOOTHING AS SUCCESSFUL TEACHER
        # ====================================================

        alpha = 0.18


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
        # APPLY
        # ====================================================

        self.fdm[
            "fcs/collective-cmd-norm"
        ] = collective


        self.fdm[
            "fcs/elevator-cmd-norm"
        ] = elevator


        self.fdm[
            "fcs/aileron-cmd-norm"
        ] = aileron


        self.fdm[
            "fcs/rudder-cmd-norm"
        ] = rudder


        # ====================================================
        # PHYSICS
        # ====================================================

        physics_ok = True


        for _ in range(
            self.base_env.PHYSICS_STEPS
        ):

            if not self.fdm.run():

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


        self.horizontal_path += (
            horizontal_speed
            *
            self.dt
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
        # ALTITUDE
        # ====================================================

        altitude = (
            s["altitude"]
        )


        signed_alt_error = (
            self.TARGET_ALTITUDE
            -
            altitude
        )


        altitude_error = abs(
            signed_alt_error
        )


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
            s["vertical_speed"]
            -
            desired_vs
        )


        # ====================================================
        # REWARD
        # ====================================================

        reward = 0.0


        # XY center
        reward += (
            3.0
            *
            float(
                np.exp(
                    -0.5
                    *
                    (
                        drift / 3.0
                    ) ** 2
                )
            )
        )


        reward -= (
            0.08
            *
            min(
                drift,
                25.0
            )
        )


        reward -= (
            0.15
            *
            min(
                horizontal_speed,
                8.0
            )
        )


        # Vertical profile
        reward += (
            1.0
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


        reward += (
            1.0
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


        # Overshoot
        if altitude > 315.0:

            reward -= (
                0.10
                *
                (
                    altitude
                    -
                    315.0
                )
            )


        # Heading
        reward -= (
            0.30
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


        # Attitude
        reward -= (
            0.15
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
            0.15
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


        # ====================================================
        # IMITATION REWARD
        #
        # Direct dense target for all four PPO outputs.
        # ====================================================

        if (
            self.training_mode
            and
            self.teacher_model
            is not None
        ):

            imitation_mse = float(
                np.mean(
                    (
                        student_action
                        -
                        teacher_action
                    ) ** 2
                )
            )


            reward += (
                4.0
                -
                8.0
                *
                imitation_mse
            )

        else:

            imitation_mse = np.nan


        # ====================================================
        # SMOOTH STUDENT ACTION
        # ====================================================

        action_change = float(
            np.mean(
                np.abs(
                    student_action
                    -
                    self.previous_student_action
                )
            )
        )


        reward -= (
            0.05
            *
            action_change
        )


        self.previous_student_action = (
            student_action.copy()
        )


        # ====================================================
        # STABILITY
        # ====================================================

        stable = (

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
                s["pitch"]
            ) < 0.12

            and

            abs(
                s["roll"]
            ) < 0.12
        )


        if stable:

            self.stable_steps += 1

            reward += 2.0

        else:

            self.stable_steps = 0


        required_stable_steps = int(
            10.0
            /
            self.dt
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


        if drift > 60.0:

            failure = True


        if altitude > 380.0:

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
            self.steps
            >=
            self.max_steps
        )


        # ====================================================
        # INFO
        # ====================================================

        info = {
            "success":
                success,

            "altitude":
                altitude,

            "altitude_error":
                altitude_error,

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

            "drift":
                drift,

            "max_drift":
                self.max_drift,

            "path":
                self.horizontal_path,

            "collective":
                collective,

            "elevator":
                elevator,

            "aileron":
                aileron,

            "rudder":
                rudder,

            "teacher_action":
                teacher_action.copy(),

            "student_action":
                student_action.copy(),

            "executed_action":
                executed_action.copy(),

            "teacher_blend":
                blend,

            "imitation_mse":
                imitation_mse,
        }


        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info
        )


    def close(self):

        if self.base_env is not None:

            self.base_env.close()
