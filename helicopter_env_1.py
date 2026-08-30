import os
import numpy as np
import jsbsim
import gymnasium as gym
from gymnasium import spaces


class HelicopterEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # ==========================================
        # JSBSIM
        # ==========================================

        self.root_dir = os.path.dirname(jsbsim.__file__)

        project_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.script_path = os.path.join(
            project_dir,
            "scripts",
            "ah1s_rl_start.xml"
        )

        # ==========================================
        # ACTION SPACE
        # ==========================================

        # action[0] = collective residual
        # action[1] = elevator residual
        # action[2] = aileron residual
        # action[3] = rudder residual

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # ==========================================
        # OBSERVATION SPACE
        # ==========================================

        # 0 = altitude AGL
        # 1 = forward velocity
        # 2 = vertical speed
        # 3 = heading
        # 4 = pitch
        # 5 = roll
        # 6 = roll rate
        # 7 = pitch rate
        # 8 = yaw rate
        # 9 = rotor RPM

        self.observation_space = spaces.Box(

            low=np.array(
                [
                    -1000,
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

        # ==========================================
        # TASK 1
        # ==========================================

        self.target_altitude = 1000.0

        self.target_climb_rate = 15.0

        self.target_heading = None

        self.phase = "TAKEOFF"

        self.target_hold_steps = 0

        self.required_hold_steps = 50

        self.max_steps = 5000

        self.steps = 0

        self.previous_altitude = None

        self.fdm = None

        # ==========================================
        # BASE CONTROL
        # ==========================================

        self.base_collective = 0.615

        self.base_elevator = -0.18

        self.base_aileron = 0.22

        self.base_rudder = 0.39

        # ==========================================
        # RESIDUAL SCALE
        # ==========================================

        # Collective biraz daha geniş kalabilir.
        #
        # Cyclic değerlerini küçültüyoruz çünkü
        # önceki testte roll giderek büyüyordu.

        self.collective_scale = 0.10

        self.elevator_scale = 0.06

        self.aileron_scale = 0.05

        self.rudder_scale = 0.06

    # ==========================================
    # JSBSIM CREATE
    # ==========================================

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

    # ==========================================
    # ROTOR WARM-UP
    # ==========================================

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

    # ==========================================
    # OBSERVATION
    # ==========================================

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

                # Roll rate
                self.fdm[
                    "velocities/p-rad_sec"
                ],

                # Pitch rate
                self.fdm[
                    "velocities/q-rad_sec"
                ],

                # Yaw rate
                self.fdm[
                    "velocities/r-rad_sec"
                ],

                self.fdm[
                    "propulsion/engine/rotor-rpm"
                ]
            ],

            dtype=np.float32
        )

    # ==========================================
    # RESET
    # ==========================================

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

        info = {

            "phase":
                self.phase,

            "altitude":
                float(obs[0]),

            "forward_velocity":
                float(obs[1]),

            "vertical_speed":
                float(obs[2]),

            "heading":
                float(obs[3]),

            "target_heading":
                self.target_heading,

            "pitch":
                float(obs[4]),

            "roll":
                float(obs[5]),

            "roll_rate":
                float(obs[6]),

            "pitch_rate":
                float(obs[7]),

            "yaw_rate":
                float(obs[8]),

            "rotor_rpm":
                float(obs[9]),

            "collective":
                self.base_collective,

            "elevator":
                self.base_elevator,

            "aileron":
                self.base_aileron,

            "rudder":
                self.base_rudder,

            "success":
                False
        }

        return obs, info

    # ==========================================
    # STEP
    # ==========================================

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

        # ==========================================
        # RESIDUAL CONTROL
        # ==========================================

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

        # ==========================================
        # CONTROL LIMITS
        # ==========================================

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

        # ==========================================
        # APPLY CONTROLS
        # ==========================================

        self.fdm[
            "fcs/collective-cmd-norm"
        ] = float(collective)

        self.fdm[
            "fcs/elevator-cmd-norm"
        ] = float(elevator)

        self.fdm[
            "fcs/aileron-cmd-norm"
        ] = float(aileron)

        self.fdm[
            "fcs/rudder-cmd-norm"
        ] = float(rudder)

        # ==========================================
        # PHYSICS
        # ==========================================

        for _ in range(10):

            if not self.fdm.run():
                break

        obs = self._get_obs()

        altitude = float(obs[0])

        forward_velocity = float(obs[1])

        vertical_speed = float(obs[2])

        heading = float(obs[3])

        pitch = float(obs[4])

        roll = float(obs[5])

        roll_rate = float(obs[6])

        pitch_rate = float(obs[7])

        yaw_rate = float(obs[8])

        rotor_rpm = float(obs[9])

        # ==========================================
        # HEADING ERROR
        # ==========================================

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

        # ==========================================
        # ALTITUDE PROGRESS
        # ==========================================

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

        # ==========================================
        # 1) YÜKSELME ÖDÜLÜ
        # ==========================================

        reward += (
            4.0
            * altitude_progress
        )

        # ==========================================
        # 2) YERDE BEKLEME
        # ==========================================

        if altitude < 10.0:

            reward -= 1.0

        # ==========================================
        # 3) ALTITUDE TARGET
        # ==========================================

        reward -= (
            0.001
            * abs(
                altitude_error
            )
        )

        # ==========================================
        # 4) CLIMB RATE
        # ==========================================

        if (
            altitude > 15.0
            and altitude < 950.0
        ):

            climb_rate_error = (
                vertical_speed
                - self.target_climb_rate
            )

            reward -= (
                0.10
                * abs(
                    climb_rate_error
                )
            )

        # ==========================================
        # 5) AŞIRI DİKEY HIZ
        # ==========================================

        if vertical_speed > 25.0:

            reward -= (
                2.0
                * (
                    vertical_speed
                    - 25.0
                )
            )

        if vertical_speed < -15.0:

            reward -= (
                2.0
                * abs(
                    vertical_speed
                    + 15.0
                )
            )

        # ==========================================
        # 6) İLERİ / GERİ KAÇIŞ
        # ==========================================

        reward -= (
            0.05
            * abs(
                forward_velocity
            )
        )

        # 70 ft/s üzerinde ek ceza

        if abs(
            forward_velocity
        ) > 70.0:

            reward -= (
                0.20
                * (
                    abs(
                        forward_velocity
                    )
                    - 70.0
                )
            )

        # ==========================================
        # 7) HEADING
        # ==========================================

        reward -= (
            1.5
            * abs(
                heading_error
            )
        )

        # ==========================================
        # 8) PITCH / ROLL ANGLE
        # ==========================================

        reward -= (
            6.0
            * abs(
                pitch
            )
        )

        reward -= (
            8.0
            * abs(
                roll
            )
        )

        # ==========================================
        # 9) ANGULAR RATE
        # ==========================================

        # Buradaki asıl yeni bilgi:
        #
        # PPO artık açının sadece ne kadar olduğunu
        # değil, ne hızla değiştiğini de görüyor.

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

        # ==========================================
        # 10) BÜYÜK AÇILARA EK CEZA
        # ==========================================

        if abs(pitch) > 0.5:

            reward -= 20.0

        if abs(roll) > 0.5:

            reward -= 30.0

        # ==========================================
        # 11) ACTION ENERGY
        # ==========================================

        reward -= (
            0.2
            * float(
                np.sum(
                    np.square(action)
                )
            )
        )

        # ==========================================
        # 1000 FT TARGET ZONE
        # ==========================================

        if abs(
            altitude_error
        ) < 30.0:

            reward += 10.0

            reward -= (
                0.2
                * abs(
                    vertical_speed
                )
            )

            if (
                abs(vertical_speed) < 8.0
                and abs(pitch) < 0.25
                and abs(roll) < 0.25
                and abs(roll_rate) < 0.15
                and abs(pitch_rate) < 0.15
            ):

                self.target_hold_steps += 1

                reward += 10.0

            else:

                self.target_hold_steps = 0

        else:

            self.target_hold_steps = 0

        # ==========================================
        # SUCCESS
        # ==========================================

        if (
            self.target_hold_steps
            >= self.required_hold_steps
        ):

            reward += 500.0

            success = True

            terminated = True

        # ==========================================
        # SAFETY
        # ==========================================

        if abs(pitch) > 1.2:

            reward -= 150.0

            terminated = True

        if abs(roll) > 1.2:

            reward -= 150.0

            terminated = True

        if rotor_rpm < 250.0:

            reward -= 150.0

            terminated = True

        if altitude > 1500.0:

            reward -= 150.0

            terminated = True

        # Yere sert dönüş

        if (
            self.steps > 100
            and altitude < 7.0
            and vertical_speed < -5.0
        ):

            reward -= 150.0

            terminated = True

        # ==========================================
        # TIME LIMIT
        # ==========================================

        truncated = (
            self.steps
            >= self.max_steps
        )

        # ==========================================
        # INFO
        # ==========================================

        info = {

            "phase":
                self.phase,

            "altitude":
                altitude,

            "forward_velocity":
                forward_velocity,

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

    # ==========================================
    # CLOSE
    # ==========================================

    def close(self):

        self.fdm = None
