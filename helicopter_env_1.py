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
        # 0  = altitude AGL
        # 1  = forward velocity
        # 2  = lateral velocity
        # 3  = vertical speed
        # 4  = heading
        # 5  = pitch
        # 6  = roll
        # 7  = roll rate
        # 8  = pitch rate
        # 9  = yaw rate
        # 10 = rotor RPM
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
        # TAKEOFF-ONLY TASK
        # =====================================================

        # Helikopter yerde yaklaşık 6.3 ft AGL'de başlıyor.
        # İlk hedef:
        #
        # 20 ft AGL'e dikeye yakın kalk
        # ve orada stabil kal.

        self.target_altitude = 20.0

        self.target_climb_rate = 5.0

        self.target_heading = None

        self.phase = "TAKEOFF"

        self.target_hold_steps = 0

        self.required_hold_steps = 50

        self.max_steps = 5000

        self.steps = 0

        self.previous_altitude = None

        self.previous_altitude_error = None

        self.fdm = None

        # =====================================================
        # BASE CONTROL
        # =====================================================
        #
        # AH-1S'in 0 knot / 20 ft civarındaki
        # steady-flight trim değerlerine yakın değerler.
        #
        # PPO bu değerlerin çevresinde residual kontrol yapacak.
        #

        self.base_collective = 0.560

        self.base_elevator = -0.223

        self.base_aileron = 0.240

        self.base_rudder = 0.386

        # =====================================================
        # RESIDUAL CONTROL SCALE
        # =====================================================
        #
        # Collective kalkış için daha geniş tutuluyor.
        #
        # Elevator / aileron / rudder ise daha dar;
        # çünkü gereğinden fazla açı üretmesini istemiyoruz.
        #

        self.collective_scale = 0.14

        self.elevator_scale = 0.04

        self.aileron_scale = 0.04

        self.rudder_scale = 0.04

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
                "AH-1S RL başlangıç scripti yüklenemedi"
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
                    "Rotor warm-up sırasında JSBSim durdu"
                )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_obs(self):

        return np.array(
            [
                # 0 - altitude
                self.fdm[
                    "position/h-agl-ft"
                ],

                # 1 - forward velocity
                self.fdm[
                    "velocities/u-aero-fps"
                ],

                # 2 - lateral velocity
                self.fdm[
                    "velocities/v-aero-fps"
                ],

                # 3 - vertical speed
                self.fdm[
                    "velocities/h-dot-fps"
                ],

                # 4 - heading
                self.fdm[
                    "attitude/heading-true-rad"
                ],

                # 5 - pitch
                self.fdm[
                    "attitude/pitch-rad"
                ],

                # 6 - roll
                self.fdm[
                    "attitude/roll-rad"
                ],

                # 7 - roll rate
                self.fdm[
                    "velocities/p-rad_sec"
                ],

                # 8 - pitch rate
                self.fdm[
                    "velocities/q-rad_sec"
                ],

                # 9 - yaw rate
                self.fdm[
                    "velocities/r-rad_sec"
                ],

                # 10 - rotor RPM
                self.fdm[
                    "propulsion/engine/rotor-rpm"
                ]
            ],

            dtype=np.float32
        )

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

        info = {

            "phase":
                self.phase,

            "altitude":
                float(obs[0]),

            "forward_velocity":
                float(obs[1]),

            "lateral_velocity":
                float(obs[2]),

            "vertical_speed":
                float(obs[3]),

            "heading":
                float(obs[4]),

            "target_heading":
                self.target_heading,

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
                self.base_collective,

            "elevator":
                self.base_elevator,

            "aileron":
                self.base_aileron,

            "rudder":
                self.base_rudder,

            "target_hold_steps":
                0,

            "success":
                False
        }

        return obs, info

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

        collective = np.clip(
            collective,
            0.0,
            1.0
        )

        elevator = np.clip(
            elevator,
            -1.0,
            1.0
        )

        aileron = np.clip(
            aileron,
            -1.0,
            1.0
        )

        rudder = np.clip(
            rudder,
            -1.0,
            1.0
        )

        # =====================================================
        # APPLY CONTROLS
        # =====================================================

        self.fdm[
            "fcs/collective-cmd-norm"
        ] = float(
            collective
        )

        self.fdm[
            "fcs/elevator-cmd-norm"
        ] = float(
            elevator
        )

        self.fdm[
            "fcs/aileron-cmd-norm"
        ] = float(
            aileron
        )

        self.fdm[
            "fcs/rudder-cmd-norm"
        ] = float(
            rudder
        )

        # =====================================================
        # PHYSICS
        # =====================================================

        for _ in range(10):

            if not self.fdm.run():
                break

        obs = self._get_obs()

        # =====================================================
        # OBSERVATION VALUES
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
        # ALTITUDE PROGRESS
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

        reward = 0.0

        terminated = False

        success = False

        # =====================================================
        # 1) HEDEFE YAKLASMA REWARD
        # =====================================================
        #
        # 20 ft hedefine yaklaşmak ödüllendirilir,
        # hedeften uzaklaşmak cezalandırılır.
        #

        current_altitude_error = abs(
            self.target_altitude
            - altitude
        )

        error_improvement = (
            self.previous_altitude_error
            - current_altitude_error
        )

        reward += (
            10.0
            * error_improvement
        )

        self.previous_altitude_error = (
            current_altitude_error
        )

        # =====================================================
        # 2) HEDEFTEN UZAKLIK CEZASI
        # =====================================================

        reward -= (
            0.10
            * current_altitude_error
        )

        # =====================================================
        # 3) YERDE KALMA CEZASI
        # =====================================================
        #
        # PPO'nun "hiç kalkmamak" çözümünü seçmesini
        # önlemek için yerde kalma cezasını güçlendiriyoruz.
        #

        if altitude < 8.0:

            reward -= 4.0

        # =====================================================
        # 4) CLIMB RATE
        # =====================================================

        if (
            altitude > 7.0
            and altitude < 17.0
        ):

            climb_rate_error = (
                vertical_speed
                - self.target_climb_rate
            )

            reward -= (
                0.15
                * abs(
                    climb_rate_error
                )
            )

        # =====================================================
        # 5) AŞIRI DİKEY HIZ
        # =====================================================

        if vertical_speed > 10.0:

            reward -= (
                2.0
                * (
                    vertical_speed
                    - 10.0
                )
            )

        if vertical_speed < -5.0:

            reward -= (
                2.0
                * abs(
                    vertical_speed
                    + 5.0
                )
            )

        # =====================================================
        # 6) FORWARD / BACKWARD MOVEMENT
        # =====================================================
        #
        # Son testte forward velocity yaklaşık
        # 65 ft/s değerine kadar çıktığı için
        # bu cezayı güçlendiriyoruz.
        #

        reward -= (
            0.60
            * abs(
                forward_velocity
            )
        )

        # =====================================================
        # 7) LATERAL MOVEMENT
        # =====================================================

        reward -= (
            0.30
            * abs(
                lateral_velocity
            )
        )

        # =====================================================
        # 8) HEADING
        # =====================================================

        reward -= (
            2.0
            * abs(
                heading_error
            )
        )

        # =====================================================
        # 9) PITCH
        # =====================================================

        reward -= (
            5.0
            * abs(
                pitch
            )
        )

        # =====================================================
        # 10) ROLL
        # =====================================================

        reward -= (
            5.0
            * abs(
                roll
            )
        )

        # =====================================================
        # 11) ANGULAR RATES
        # =====================================================

        reward -= (
            2.0
            * abs(
                roll_rate
            )
        )

        reward -= (
            2.0
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
        # 12) LARGE ATTITUDE PENALTY
        # =====================================================

        if abs(
            pitch
        ) > 0.5:

            reward -= 20.0

        if abs(
            roll
        ) > 0.5:

            reward -= 30.0

        # =====================================================
        # 13) ACTION ENERGY
        # =====================================================

        reward -= (
            0.2
            * float(
                np.sum(
                    np.square(
                        action
                    )
                )
            )
        )

        # =====================================================
        # 14) 20 FT TARGET ZONE
        # =====================================================
        #
        # 17 - 23 ft aralığı
        #

        if abs(
            altitude_error
        ) < 3.0:

            reward += 10.0

            # Hedefte dikey hızın sıfıra yaklaşması
            # gerekiyor.

            reward -= (
                1.0
                * abs(
                    vertical_speed
                )
            )

            # =================================================
            # HOLD
            # =================================================

            if (
                abs(vertical_speed) < 2.0
                and abs(pitch) < 0.20
                and abs(roll) < 0.20
                and abs(roll_rate) < 0.10
                and abs(pitch_rate) < 0.10
                and abs(forward_velocity) < 5.0
                and abs(lateral_velocity) < 5.0
            ):

                self.target_hold_steps += 1

                reward += 10.0

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

            reward += 500.0

            success = True

            terminated = True

        # =====================================================
        # SAFETY - PITCH
        # =====================================================

        if abs(
            pitch
        ) > 1.2:

            reward -= 150.0

            terminated = True

        # =====================================================
        # SAFETY - ROLL
        # =====================================================

        if abs(
            roll
        ) > 1.2:

            reward -= 150.0

            terminated = True

        # =====================================================
        # SAFETY - ROTOR RPM
        # =====================================================

        if rotor_rpm < 250.0:

            reward -= 150.0

            terminated = True

        # =====================================================
        # SAFETY - TOO HIGH
        # =====================================================

        if altitude > 40.0:

            reward -= 150.0

            terminated = True

        # =====================================================
        # HARD RETURN TO GROUND
        # =====================================================

        if (
            self.steps > 100
            and altitude < 7.0
            and vertical_speed < -5.0
        ):

            reward -= 150.0

            terminated = True

        # =====================================================
        # FAILED TAKEOFF
        # =====================================================
        #
        # Yaklaşık 11 saniye içinde 8 ft üzerine çıkamıyorsa
        # kalkış denemesi başarısız kabul edilir.
        #

        if (
            self.steps > 150
            and altitude < 8.0
        ):

            reward -= 100.0

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

            "forward_velocity":
                forward_velocity,

            "lateral_velocity":
                lateral_velocity,

            "vertical_speed":
                vertical_speed,

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
                float(
                    collective
                ),

            "elevator":
                float(
                    elevator
                ),

            "aileron":
                float(
                    aileron
                ),

            "rudder":
                float(
                    rudder
                ),

            "altitude_progress":
                altitude_progress,

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
