import gymnasium as gym
from gymnasium import spaces

import numpy as np

from stable_baselines3 import PPO

from helicopter_env_v2 import HelicopterEnvV2


class HelicopterEnvStage1FourAction(gym.Env):

    metadata = {
        "render_modes": []
    }

    # ========================================================
    # MISSION
    # ========================================================

    TARGET_ALTITUDE = 300.0

    MAX_TIME = 120.0

    # --------------------------------------------------------
    # STRICT "STICK-LIKE" TAKEOFF
    # --------------------------------------------------------

    SUCCESS_MAX_DRIFT = 8.0
    SUCCESS_FINAL_DRIFT = 5.0
    SUCCESS_MAX_PATH = 25.0

    FAILURE_DRIFT = 40.0

    # --------------------------------------------------------
    # BASE TRIMS
    # --------------------------------------------------------

    BASE_ELEVATOR = -0.15390
    BASE_AILERON = 0.19100
    BASE_RUDDER = 0.39000

    # --------------------------------------------------------
    # PPO CONTROL AUTHORITY
    # --------------------------------------------------------

    ELEVATOR_AUTHORITY = 0.012
    AILERON_AUTHORITY = 0.012
    RUDDER_AUTHORITY = 0.020


    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        teacher_model_path=(
            "models_v2/"
            "AH1S_STAGE1_SUCCESS.zip"
        ),
        use_teacher_reward=True
    ):

        super().__init__()

        # ====================================================
        # JSBSIM BASE ENVIRONMENT
        # ====================================================

        self.base_env = HelicopterEnvV2()

        self.fdm = self.base_env.fdm


        # ====================================================
        # TEACHER
        #
        # IMPORTANT:
        #
        # Teacher controls NOTHING.
        #
        # It is only used during training to prevent the
        # already-good collective behavior from being lost.
        #
        # Final PPO controls all 4 actions itself.
        # ====================================================

        self.use_teacher_reward = bool(
            use_teacher_reward
        )

        self.teacher_model = None

        if (
            teacher_model_path is not None
            and
            self.use_teacher_reward
        ):

            self.teacher_model = PPO.load(
                teacher_model_path
            )


        # ====================================================
        # ACTION SPACE
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
        # OBSERVATION
        #
        # First 12 observations are EXACTLY the original
        # Stage 1 observations.
        #
        # Then:
        #
        # 12 north position
        # 13 east position
        # 14 north velocity
        # 15 east velocity
        # 16 heading error
        # 17 yaw rate
        #
        # Total = 18
        # ====================================================

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(18,),
            dtype=np.float32
        )


        # ====================================================
        # TIMING
        # ====================================================

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
        # INTERNAL STATE
        # ====================================================

        self.steps = 0

        self.north = 0.0
        self.east = 0.0

        self.max_drift = 0.0
        self.horizontal_path = 0.0

        self.initial_heading = 0.0

        self.previous_altitude = 0.0

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
    # ANGLE WRAP
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
    # RAW STATE
    # ========================================================

    def _state(self):

        raw = (
            self.base_env._raw_state()
        )

        fdm = self.base_env.fdm

        vn = float(
            fdm[
                "velocities/v-north-fps"
            ]
        )

        ve = float(
            fdm[
                "velocities/v-east-fps"
            ]
        )

        heading = float(
            fdm[
                "attitude/psi-rad"
            ]
        )

        yaw_rate = float(
            fdm[
                "velocities/r-rad_sec"
            ]
        )

        heading_error = self._wrap_angle(
            heading
            -
            self.initial_heading
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

            "heading":
                heading,

            "heading_error":
                heading_error,

            "yaw_rate":
                yaw_rate,
        }


    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_obs(self):

        # Original Stage 1 observation.
        #
        # This is critical for weight transfer.
        base_obs = (
            self.base_env._get_obs()
        ).astype(
            np.float32
        )

        s = self._state()

        extra_obs = np.array(
            [
                # North position
                np.clip(
                    self.north / 10.0,
                    -10.0,
                    10.0
                ),

                # East position
                np.clip(
                    self.east / 10.0,
                    -10.0,
                    10.0
                ),

                # North velocity
                np.clip(
                    s["vn"] / 5.0,
                    -10.0,
                    10.0
                ),

                # East velocity
                np.clip(
                    s["ve"] / 5.0,
                    -10.0,
                    10.0
                ),

                # Heading error
                np.clip(
                    s["heading_error"] / 0.20,
                    -10.0,
                    10.0
                ),

                # Yaw rate
                np.clip(
                    s["yaw_rate"] / 0.50,
                    -10.0,
                    10.0
                ),
            ],
            dtype=np.float32
        )

        obs = np.concatenate(
            [
                base_obs,
                extra_obs
            ]
        )

        return obs.astype(
            np.float32
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


        # Parent reset:
        # rotor/governor becomes ready.
        base_obs, base_info = (
            self.base_env.reset(
                seed=seed,
                options=options
            )
        )

        self.fdm = (
            self.base_env.fdm
        )


        self.initial_heading = float(
            self.fdm[
                "attitude/psi-rad"
            ]
        )


        s = self._state()

        self.previous_altitude = (
            s["altitude"]
        )


        obs = self._get_obs()


        info = {
            "success": False,

            "altitude":
                s["altitude"],

            "north":
                0.0,

            "east":
                0.0,

            "drift":
                0.0,

            "max_drift":
                0.0,

            "path":
                0.0,

            "heading_error":
                0.0,
        }


        return obs, info


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action
    ):

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


        # ====================================================
        # TEACHER COLLECTIVE REFERENCE
        #
        # TEACHER DOES NOT CONTROL AIRCRAFT.
        # ====================================================

        teacher_collective_action = None

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

            teacher_collective_action = float(
                teacher_action[0]
            )


        # ====================================================
        # ACTION 0 -> COLLECTIVE
        # ====================================================

        collective = (
            0.620
            +
            0.030
            *
            float(action[0])
        )

        collective = float(
            np.clip(
                collective,
                0.590,
                0.650
            )
        )


        # ====================================================
        # ACTION 1 -> ELEVATOR
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
                -0.1680,
                -0.1400
            )
        )


        # ====================================================
        # ACTION 2 -> AILERON
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
                0.1770,
                0.2050
            )
        )


        # ====================================================
        # ACTION 3 -> RUDDER
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
                0.3550,
                0.4250
            )
        )


        # ====================================================
        # SMOOTH CYCLIC / PEDAL
        #
        # Collective is NOT smoothed because the old policy
        # already has a good collective trajectory.
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


        self.previous_elevator = (
            elevator
        )

        self.previous_aileron = (
            aileron
        )

        self.previous_rudder = (
            rudder
        )


        # ====================================================
        # APPLY ALL 4 PPO ACTIONS
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
        # RUN PHYSICS
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
        # POSITION INTEGRATION
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
            np.sqrt(
                s["vn"] ** 2
                +
                s["ve"] ** 2
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
            np.sqrt(
                self.north ** 2
                +
                self.east ** 2
            )
        )


        self.max_drift = max(
            self.max_drift,
            drift
        )


        # ====================================================
        # ALTITUDE PROFILE
        # ====================================================

        altitude = s["altitude"]

        altitude_error = abs(
            self.TARGET_ALTITUDE
            -
            altitude
        )


        altitude_gain = (
            altitude
            -
            self.previous_altitude
        )


        self.previous_altitude = (
            altitude
        )


        # We already know this climb profile works well.
        desired_vs = float(
            np.clip(
                0.05
                *
                (
                    self.TARGET_ALTITUDE
                    -
                    altitude
                ),
                0.0,
                6.0
            )
        )


        vertical_tracking_error = abs(
            s["vertical_speed"]
            -
            desired_vs
        )


        # ====================================================
        # REWARD
        #
        # IMPORTANT:
        #
        # Keep values moderate.
        #
        # Previous reward became ~ -9000/episode and critic
        # effectively collapsed.
        # ====================================================

        reward = 0.0


        # ----------------------------------------------------
        # 1. KEEP CLIMBING
        # ----------------------------------------------------

        reward += (
            0.60
            *
            float(
                np.clip(
                    altitude_gain,
                    -1.0,
                    1.0
                )
            )
        )


        # ----------------------------------------------------
        # 2. FOLLOW GOOD VERTICAL PROFILE
        # ----------------------------------------------------

        reward += (
            0.50
            *
            (
                1.0
                -
                min(
                    vertical_tracking_error
                    /
                    8.0,
                    1.0
                )
            )
        )


        # ----------------------------------------------------
        # 3. POSITION HOLD
        #
        # Main "stick-like takeoff" reward.
        # ----------------------------------------------------

        center_reward = float(
            np.exp(
                -0.5
                *
                (
                    drift / 3.0
                ) ** 2
            )
        )

        reward += (
            1.50
            *
            center_reward
        )


        # Gradual position penalty
        reward -= (
            0.08
            *
            min(
                drift,
                25.0
            )
        )


        # ----------------------------------------------------
        # 4. DO NOT DRAW S / ARC
        # ----------------------------------------------------

        reward -= (
            0.10
            *
            min(
                horizontal_speed,
                10.0
            )
        )


        # ----------------------------------------------------
        # 5. HEADING HOLD
        # ----------------------------------------------------

        reward -= (
            0.40
            *
            min(
                abs(
                    s["heading_error"]
                )
                /
                0.20,
                2.0
            )
        )


        # ----------------------------------------------------
        # 6. ATTITUDE
        # ----------------------------------------------------

        reward -= (
            0.20
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
            0.20
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
        # 7. PRESERVE LEARNED COLLECTIVE BEHAVIOR
        #
        # Teacher is only a TRAINING REGULARIZER.
        #
        # Aircraft is still controlled by action[0].
        # ----------------------------------------------------

        if teacher_collective_action is not None:

            reward -= (
                0.50
                *
                abs(
                    float(action[0])
                    -
                    teacher_collective_action
                )
            )


        # ----------------------------------------------------
        # 8. SMOOTH 4-ACTION CONTROL
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


        # ----------------------------------------------------
        # 9. NEAR TARGET ALTITUDE
        # ----------------------------------------------------

        if altitude_error < 20.0:

            reward += 0.50


        if altitude_error < 10.0:

            reward += 0.50


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

            reward += 500.0


        # ====================================================
        # FAILURE
        # ====================================================

        failure = False


        if not physics_ok:

            failure = True


        if drift > self.FAILURE_DRIFT:

            failure = True


        if altitude > 390.0:

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

            reward -= 100.0


        # ====================================================
        # TERMINATION
        # ====================================================

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
        # OBSERVATION
        # ====================================================

        obs = self._get_obs()


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

            "forward_velocity":
                s["forward_velocity"],

            "lateral_velocity":
                s["lateral_velocity"],

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

            "yaw_rate":
                s["yaw_rate"],

            "pitch":
                s["pitch"],

            "roll":
                s["roll"],

            "collective":
                collective,

            "elevator":
                elevator,

            "aileron":
                aileron,

            "rudder":
                rudder,
        }


        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.base_env is not None:

            self.base_env.close()
