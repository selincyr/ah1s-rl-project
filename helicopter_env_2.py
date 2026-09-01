import os
import numpy as np
import jsbsim
import gymnasium as gym

from gymnasium import spaces


class HelicopterEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):

        super().__init__()

        # =====================================================
        # JSBSIM
        # =====================================================

        self.root_dir = os.path.dirname(
            jsbsim.__file__
        )

        project_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.script_path = os.path.join(
            project_dir,
            "scripts",
            "ah1s_rl_start.xml"
        )

        # =====================================================
        # ACTION SPACE
        # =====================================================
        #
        # action[0] = collective residual
        # action[1] = elevator residual
        # action[2] = aileron residual
        # action[3] = rudder residual
        #

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # =====================================================
        # OBSERVATION SPACE
        # =====================================================
        #
        # 0  altitude AGL
        # 1  forward velocity
        # 2  lateral velocity
        # 3  vertical speed
        # 4  heading
        # 5  pitch
        # 6  roll
        # 7  roll rate
        # 8  pitch rate
        # 9  yaw rate
        # 10 rotor RPM
        #

        self.observation_space = spaces.Box(

            low=np.array(
                [
                    -1000,
                    -500,
                    -500,
                    -500,
                    0.0,
                    -np.pi / 2,
                    -np.pi,
                    -20,
                    -20,
                    -20,
                    0
                ],
                dtype=np.float32
            ),

            high=np.array(
                [
                    10000,
                    500,
                    500,
                    500,
                    2 * np.pi,
                    np.pi / 2,
                    np.pi,
                    20,
                    20,
                    20,
                    700
                ],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        # =====================================================
        # MISSION
        # =====================================================

        self.target_altitude = 1000.0

        self.phase = "TAKEOFF"

        self.target_heading = None

        # 50 RL step:
        # 50 * 0.075 = 3.75 saniye
        self.required_hold_steps = 50

        self.target_hold_steps = 0

        # 5000 * 0.075 ≈ 375 saniye
        self.max_steps = 5000

        self.steps = 0

        self.previous_altitude = None

        self.previous_altitude_error = None

        self.fdm = None

        # =====================================================
        # BASE CONTROL
        # =====================================================
        #
        # AH-1S 0-knot / hover trim civarı.
        # PPO bunların çevresinde residual kontrol yapar.
        #

        self.base_collective = 0.560

        self.base_elevator = -0.223

        self.base_aileron = 0.240

        self.base_rudder = 0.386

        # =====================================================
        # RESIDUAL CONTROL SCALE
        # =====================================================

        self.collective_scale = 0.14

        self.elevator_scale = 0.04

        self.aileron_scale = 0.04

        self.rudder_scale = 0.07

    # =========================================================
    # CREATE JSBSIM
    # =========================================================

    def _create_fdm(self):

        self.fdm = jsbsim.FGFDMExec(
            root_dir=self.root_dir
        )

        if not self.fdm.load_script(
            self.script_path
        ):

            raise RuntimeError(
                "AH-1S RL baslangic scripti yuklenemedi"
            )

        self.fdm.run_ic()

    # =========================================================
    # ROTOR WARM-UP
    # =========================================================

    def _warmup_rotor(self):

        while (
            self.fdm[
                "propulsion/engine/rotor-rpm"
            ] < 320.0
        ):

            if not self.fdm.run():

                raise RuntimeError(
                    "Rotor warm-up sirasinda JSBSim durdu"
                )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_obs(self):

        return np.array(
            [
                self.fdm[
                    "position/h-agl-ft"
                ],

                self.fdm[
                    "velocities/u-aero-fps"
                ],

                self.fdm[
                    "velocities/v-aero-fps"
                ],

                self.fdm[
                    "velocities/h-dot-fps"
                ],

                self.fdm[
                    "attitude/heading-true-rad"
                ],

                self.fdm[
                    "attitude/pitch-rad"
                ],

                self.fdm[
                    "attitude/roll-rad"
                ],

                self.fdm[
                    "velocities/p-rad_sec"
                ],

                self.fdm[
                    "velocities/q-rad_sec"
                ],

                self.fdm[
                    "velocities/r-rad_sec"
                ],

                self.fdm[
                    "propulsion/engine/rotor-rpm"
                ]
            ],

            dtype=np.float32
        )

    # =========================================================
    # FLIGHT PHASE
    # =========================================================

    def _update_phase(
        self,
        altitude
    ):

        if altitude < 30.0:

            self.phase = "TAKEOFF"

        elif altitude < 850.0:

            self.phase = "CLIMB"

        elif altitude < 970.0:

            self.phase = "APPROACH"

        else:

            self.phase = "HOVER"

    # =========================================================
    # TARGET VERTICAL SPEED
    # =========================================================

    def _target_vertical_speed(
        self,
        altitude
    ):

        # -----------------------------------------------------
        # TAKEOFF
        # -----------------------------------------------------
        #
        # Yerden kontrollü ayrıl.
        #

        if self.phase == "TAKEOFF":

            return 6.0

        # -----------------------------------------------------
        # CLIMB
        # -----------------------------------------------------
        #
        # 15 ft/s ≈ 900 ft/min
        #

        if self.phase == "CLIMB":

            return 12.0

        # -----------------------------------------------------
        # APPROACH
        # -----------------------------------------------------
        #
        # 850 ft -> yaklaşık 15 ft/s
        # 970 ft -> yaklaşık 3 ft/s
        #
        # 1000 ft'e yaklaşırken collective'i
        # yavaş yavaş azaltmayı öğrenmesi için.
        #

        if self.phase == "APPROACH":

            remaining = (
                self.target_altitude
                - altitude
            )

            target_rate = (
                remaining / 20.0
            )

            return float(
                np.clip(
                    target_rate,
                    1.5,
                    8.0
                )
            )

        # -----------------------------------------------------
        # HOVER
        # -----------------------------------------------------

        return 0.0

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.steps = 0

        self.phase = "TAKEOFF"

        self.target_hold_steps = 0

        self._create_fdm()

        self._warmup_rotor()

        self.target_heading = float(
            self.fdm[
                "attitude/heading-true-rad"
            ]
        )

        obs = self._get_obs()

        self.previous_altitude = float(
            obs[0]
        )

        self.previous_altitude_error = abs(
            self.target_altitude
            - float(obs[0])
        )

        info = self._create_info(
            obs=obs,
            collective=self.base_collective,
            elevator=self.base_elevator,
            aileron=self.base_aileron,
            rudder=self.base_rudder,
            heading_error=0.0,
            success=False
        )

        return obs, info

    # =========================================================
    # INFO
    # =========================================================

    def _create_info(
        self,
        obs,
        collective,
        elevator,
        aileron,
        rudder,
        heading_error,
        success
    ):

        altitude = float(obs[0])

        self._update_phase(
            altitude
        )

        target_vertical_speed = (
            self._target_vertical_speed(
                altitude
            )
        )

        return {

            "phase":
                self.phase,

            "altitude":
                float(obs[0]),

            "target_altitude":
                self.target_altitude,

            "forward_velocity":
                float(obs[1]),

            "lateral_velocity":
                float(obs[2]),

            "vertical_speed":
                float(obs[3]),

            "target_vertical_speed":
                target_vertical_speed,

            "heading":
                float(obs[4]),

            "target_heading":
                self.target_heading,

            "heading_error":
                float(heading_error),

            "pitch":
                float(obs[5]),

            "roll":
                float(obs[6]),

            "roll_rate":
                float(obs[7]),

            "pitch_rate":
                float(obs[8]),

            "yaw_rate":
                float(obs[9]),

            "rotor_rpm":
                float(obs[10]),

            "collective":
                float(collective),

            "elevator":
                float(elevator),

            "aileron":
                float(aileron),

            "rudder":
                float(rudder),

            "target_hold_steps":
                self.target_hold_steps,

            "success":
                success
        }

    # =========================================================
    # STEP
    # =========================================================

    def step(
        self,
        action
    ):

        self.steps += 1

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # =====================================================
        # RESIDUAL CONTROL
        # =====================================================

        collective = (
            self.base_collective
            + self.collective_scale
            * float(action[0])
        )

        elevator = (
            self.base_elevator
            + self.elevator_scale
            * float(action[1])
        )

        aileron = (
            self.base_aileron
            + self.aileron_scale
            * float(action[2])
        )

        rudder = (
            self.base_rudder
            + self.rudder_scale
            * float(action[3])
        )

        # =====================================================
        # CONTROL LIMITS
        # =====================================================

        collective = float(
            np.clip(
                collective,
                0.0,
                1.0
            )
        )

        elevator = float(
            np.clip(
                elevator,
                -1.0,
                1.0
            )
        )

        aileron = float(
            np.clip(
                aileron,
                -1.0,
                1.0
            )
        )

        rudder = float(
            np.clip(
                rudder,
                -1.0,
                1.0
            )
        )

        # =====================================================
        # APPLY CONTROLS
        # =====================================================

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

        # =====================================================
        # PHYSICS
        # =====================================================
        #
        # JSBSim dt ≈ 0.0075
        #
        # 10 physics step
        # = yaklaşık 0.075 saniye / RL step
        #

        for _ in range(10):

            if not self.fdm.run():

                break

        obs = self._get_obs()

        # =====================================================
        # STATE VALUES
        # =====================================================

        altitude = float(
            obs[0]
        )

        forward_velocity = float(
            obs[1]
        )

        lateral_velocity = float(
            obs[2]
        )

        vertical_speed = float(
            obs[3]
        )

        heading = float(
            obs[4]
        )

        pitch = float(
            obs[5]
        )

        roll = float(
            obs[6]
        )

        roll_rate = float(
            obs[7]
        )

        pitch_rate = float(
            obs[8]
        )

        yaw_rate = float(
            obs[9]
        )

        rotor_rpm = float(
            obs[10]
        )

        # =====================================================
        # UPDATE PHASE
        # =====================================================

        self._update_phase(
            altitude
        )

        target_vertical_speed = (
            self._target_vertical_speed(
                altitude
            )
        )

        # =====================================================
        # HEADING ERROR
        # =====================================================

        heading_error = np.arctan2(

            np.sin(
                heading
                - self.target_heading
            ),

            np.cos(
                heading
                - self.target_heading
            )
        )

        # =====================================================
        # ALTITUDE
        # =====================================================

        altitude_progress = (
            altitude
            - self.previous_altitude
        )

        self.previous_altitude = altitude

        altitude_error = (
            self.target_altitude
            - altitude
        )

        current_altitude_error = abs(
            altitude_error
        )

        error_improvement = (
            self.previous_altitude_error
            - current_altitude_error
        )

        self.previous_altitude_error = (
            current_altitude_error
        )

        # =====================================================
        # REWARD
        # =====================================================

        reward = 0.0

        terminated = False

        success = False

        # =====================================================
        # 1) ALTITUDE PROGRESS
        # =====================================================
        #
        # 1000 ft hedefine yaklaştıkça ödül.
        #

        reward += (
            10.0
            * error_improvement
        )

        # =====================================================
        # 2) YERDE KALMA
        # =====================================================

        if altitude < 8.0:

            reward -= 5.0

        # =====================================================
        # 3) TARGET CLIMB RATE
        # =====================================================

        climb_rate_error = (
            vertical_speed
            - target_vertical_speed
        )

        if self.phase == "TAKEOFF":

            reward -= (
                0.40
                * abs(
                    climb_rate_error
                )
            )

        elif self.phase == "CLIMB":

            reward -= (
                0.30
                * abs(
                    climb_rate_error
                )
            )

        elif self.phase == "APPROACH":

            reward -= (
                0.50
                * abs(
                    climb_rate_error
                )
            )

        else:

            reward -= (
                1.50
                * abs(
                    vertical_speed
                )
            )

        # =====================================================
        # 4) EXCESSIVE VERTICAL SPEED
        # =====================================================

        if vertical_speed > 25.0:

            reward -= (
                3.0
                * (
                    vertical_speed
                    - 25.0
                )
            )

        if vertical_speed < -10.0:

            reward -= (
                3.0
                * abs(
                    vertical_speed
                    + 10.0
                )
            )

        # =====================================================
        # 5) FORWARD / BACKWARD DRIFT
        # =====================================================

        reward -= (
            0.60
            * abs(
                forward_velocity
            )
        )

        # =====================================================
        # 6) LATERAL DRIFT
        # =====================================================

        reward -= (
            0.75
            * abs(
                lateral_velocity
            )
        )

        # =====================================================
        # 7) HEADING
        # =====================================================

        reward -= (
            5.0
            * abs(
                heading_error
            )
        )

        # =====================================================
        # 8) PITCH / ROLL
        # =====================================================

        reward -= (
            8.0
            * abs(
                pitch
            )
        )

        reward -= (
            10.0
            * abs(
                roll
            )
        )

        # =====================================================
        # 9) ANGULAR RATES
        # =====================================================

        reward -= (
            1.5
            * abs(
                roll_rate
            )
        )

        reward -= (
            1.5
            * abs(
                pitch_rate
            )
        )

        reward -= (
            1.0
            * abs(
                yaw_rate
            )
        )

        # =====================================================
        # 10) LARGE ATTITUDE
        # =====================================================

        if abs(pitch) > 0.5:

            reward -= 30.0

        if abs(roll) > 0.5:

            reward -= 40.0

        # =====================================================
        # 11) ACTION ENERGY
        # =====================================================

        reward -= (
            0.10
            * float(
                np.sum(
                    np.square(
                        action
                    )
                )
            )
        )

        # =====================================================
        # 12) APPROACH BRAKING
        # =====================================================
        #
        # 950 ft'in üzerinde hâlâ çok hızlı yükseliyorsa
        # ciddi ceza.
        #

        if (
            altitude > 950.0
            and vertical_speed > 5.0
        ):

            reward -= (
                2.0
                * (
                    vertical_speed
                    - 5.0
                )
            )

        # =====================================================
        # 13) OVERSHOOT
        # =====================================================

        if altitude > 1030.0:

            reward -= (
                5.0
                * (
                    altitude
                    - 1030.0
                )
            )

        if altitude > 1080.0:
            
            reward -= 500.0
            
            terminated = True

        # =====================================================
        # 14) TARGET ALTITUDE ZONE
        # =====================================================
        #
        # 970 - 1030 ft
        #

        if abs(
            altitude_error
        ) < 30.0:

            reward += 20.0

            reward -= (
                0.20
                * abs(
                    altitude_error
                )
            )

            reward -= (
                2.0
                * abs(
                    vertical_speed
                )
            )

            # =================================================
            # HOVER HOLD
            # =================================================

            if (
                abs(vertical_speed) < 2.0
                and abs(pitch) < 0.20
                and abs(roll) < 0.20
                and abs(roll_rate) < 0.15
                and abs(pitch_rate) < 0.15
                and abs(forward_velocity) < 8.0
                and abs(lateral_velocity) < 8.0
                and abs(heading_error) < 0.30
            ):

                self.target_hold_steps += 1

                reward += 15.0

            else:

                self.target_hold_steps = 0

        else:

            self.target_hold_steps = 0

        # =====================================================
        # SUCCESS
        # =====================================================

        if (
            self.target_hold_steps
            >= self.required_hold_steps
        ):

            reward += 1000.0

            success = True

            terminated = True

        # =====================================================
        # SAFETY - PITCH
        # =====================================================

        if abs(
            pitch
        ) >0.70:

            reward -= 300.0

            terminated = True

        # =====================================================
        # SAFETY - ROLL
        # =====================================================

        if abs(
            roll
        ) > 0.70:

            reward -= 300.0

            terminated = True

        # =====================================================
        # SAFETY - EXCESSIVE HORIZONTAL SPEED
        # =====================================================

        if (
            abs(forward_velocity) > 55.0
            or abs(lateral_velocity) > 45.0
        ):

            reward -= 300.0

            terminated = True

        # =====================================================
        # SAFETY - ROTOR
        # =====================================================

        if rotor_rpm < 250.0:

            reward -= 250.0

            terminated = True

        # =====================================================
        # SAFETY - TOO HIGH
        # =====================================================

        if altitude > 1150.0:

            reward -= 300.0

            terminated = True

        # =====================================================
        # HARD RETURN TO GROUND
        # =====================================================

        if (
            self.steps > 150
            and altitude < 7.0
            and vertical_speed < -5.0
        ):

            reward -= 250.0

            terminated = True

        # =====================================================
        # FAILED TAKEOFF
        # =====================================================

        if (
            self.steps > 200
            and altitude < 15.0
        ):

            reward -= 200.0

            terminated = True

        # =====================================================
        # TIME LIMIT
        # =====================================================

        truncated = (
            self.steps
            >= self.max_steps
        )

        # =====================================================
        # INFO
        # =====================================================

        info = {

            "phase":
                self.phase,

            "altitude":
                altitude,

            "target_altitude":
                self.target_altitude,

            "altitude_error":
                altitude_error,

            "altitude_progress":
                altitude_progress,

            "forward_velocity":
                forward_velocity,

            "lateral_velocity":
                lateral_velocity,

            "vertical_speed":
                vertical_speed,

            "target_vertical_speed":
                target_vertical_speed,

            "heading":
                heading,

            "target_heading":
                self.target_heading,

            "heading_error":
                float(
                    heading_error
                ),

            "pitch":
                pitch,

            "roll":
                roll,

            "roll_rate":
                roll_rate,

            "pitch_rate":
                pitch_rate,

            "yaw_rate":
                yaw_rate,

            "rotor_rpm":
                rotor_rpm,

            "collective":
                collective,

            "elevator":
                elevator,

            "aileron":
                aileron,

            "rudder":
                rudder,

            "target_hold_steps":
                self.target_hold_steps,

            "success":
                success
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self.fdm = None
