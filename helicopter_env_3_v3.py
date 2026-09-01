import os
import numpy as np
import jsbsim
import gymnasium as gym

from gymnasium import spaces


class HelicopterEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.root_dir = os.path.dirname(jsbsim.__file__)
        project_dir = os.path.dirname(os.path.abspath(__file__))

        self.script_path = os.path.join(
            project_dir,
            "scripts",
            "ah1s_rl_start.xml"
        )

        self.target_altitude = 1000.0
        self.target_forward_speed = 35.0

        self.phase = "TAKEOFF"
        self.target_heading = None

        self.required_hold_steps = 100
        self.target_hold_steps = 0

        self.max_steps = 5000
        self.steps = 0

        self.fdm = None

        self.base_collective = 0.5601
        self.base_elevator = -0.2227
        self.base_aileron = 0.2399
        self.base_rudder = 0.3855

        self.trim_altitudes = np.array(
            [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
            dtype=np.float32
        )

        self.trim_collective = np.array(
            [0.5601, 0.5699, 0.5706, 0.5719, 0.5759, 0.5827],
            dtype=np.float32
        )

        self.trim_elevator = np.array(
            [-0.2227, -0.2228, -0.2228, -0.2228, -0.2228, -0.2228],
            dtype=np.float32
        )

        self.trim_aileron = np.array(
            [0.2399, 0.2423, 0.2426, 0.2432, 0.2448, 0.2477],
            dtype=np.float32
        )

        self.trim_rudder = np.array(
            [0.3855, 0.3901, 0.3905, 0.3913, 0.3937, 0.3978],
            dtype=np.float32
        )

        self.collective_scale = 0.14
        self.elevator_scale = 0.06
        self.aileron_scale = 0.05
        self.rudder_scale = 0.07

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, -1.5, -5.0, -5.0, -10.0, -1.0,
                 -3.0, -5.0, -10.0, -10.0, -10.0, 0.0],
                dtype=np.float32
            ),
            high=np.array(
                [10.0, 2.0, 5.0, 5.0, 10.0, 1.0,
                 3.0, 5.0, 10.0, 10.0, 10.0, 2.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

    def _create_fdm(self):
        self.fdm = jsbsim.FGFDMExec(root_dir=self.root_dir)

        if not self.fdm.load_script(self.script_path):
            raise RuntimeError(
                "AH-1S RL baslangic scripti yuklenemedi"
            )

        self.fdm.run_ic()

    def _warmup_rotor(self):
        while self.fdm["propulsion/engine/rotor-rpm"] < 320.0:
            if not self.fdm.run():
                raise RuntimeError(
                    "Rotor warm-up sirasinda JSBSim durdu"
                )

    @staticmethod
    def _wrap_angle(angle):
        return float(
            np.arctan2(
                np.sin(angle),
                np.cos(angle)
            )
        )

    def _get_state(self):
        altitude = float(self.fdm["position/h-agl-ft"])
        heading = float(self.fdm["attitude/heading-true-rad"])

        if self.target_heading is None:
            heading_error = 0.0
        else:
            heading_error = self._wrap_angle(
                heading - self.target_heading
            )

        return {
            "altitude": altitude,
            "altitude_error": self.target_altitude - altitude,
            "forward_velocity": float(
                self.fdm["velocities/u-aero-fps"]
            ),
            "lateral_velocity": float(
                self.fdm["velocities/v-aero-fps"]
            ),
            "vertical_speed": float(
                self.fdm["velocities/h-dot-fps"]
            ),
            "heading": heading,
            "heading_error": heading_error,
            "pitch": float(
                self.fdm["attitude/pitch-rad"]
            ),
            "roll": float(
                self.fdm["attitude/roll-rad"]
            ),
            "roll_rate": float(
                self.fdm["velocities/p-rad_sec"]
            ),
            "pitch_rate": float(
                self.fdm["velocities/q-rad_sec"]
            ),
            "yaw_rate": float(
                self.fdm["velocities/r-rad_sec"]
            ),
            "rotor_rpm": float(
                self.fdm["propulsion/engine/rotor-rpm"]
            )
        }

    def _get_obs_from_state(self, state):
        obs = np.array(
            [
                state["altitude"] / 1200.0,
                state["altitude_error"] / 1000.0,
                state["forward_velocity"] / 100.0,
                state["lateral_velocity"] / 100.0,
                state["vertical_speed"] / 30.0,
                state["heading_error"] / np.pi,
                state["pitch"] / 0.70,
                state["roll"] / 0.70,
                state["roll_rate"] / 2.0,
                state["pitch_rate"] / 2.0,
                state["yaw_rate"] / 2.0,
                state["rotor_rpm"] / 400.0
            ],
            dtype=np.float32
        )

        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high
        ).astype(np.float32)

    def _get_trim_controls(self, altitude):
        collective = float(
            np.interp(
                altitude,
                self.trim_altitudes,
                self.trim_collective
            )
        )

        elevator = float(
            np.interp(
                altitude,
                self.trim_altitudes,
                self.trim_elevator
            )
        )

        aileron = float(
            np.interp(
                altitude,
                self.trim_altitudes,
                self.trim_aileron
            )
        )

        rudder = float(
            np.interp(
                altitude,
                self.trim_altitudes,
                self.trim_rudder
            )
        )

        return collective, elevator, aileron, rudder

    def _update_phase(self, altitude):
        # Start slowing the climb earlier. In v2 the helicopter was still
        # in CLIMB near 875 ft and reached excessive vertical speed/attitude.
        if altitude < 30.0:
            self.phase = "TAKEOFF"
        elif altitude < 800.0:
            self.phase = "CLIMB"
        elif altitude < 970.0:
            self.phase = "APPROACH"
        else:
            self.phase = "FORWARD"

    def _target_vertical_speed(self, altitude):
        if self.phase == "TAKEOFF":
            return 6.0

        if self.phase == "CLIMB":
            # 30-650 ft: nominal 12 ft/s climb.
            # 650-800 ft: smoothly reduce 12 -> 8 ft/s.
            return float(
                np.interp(
                    altitude,
                    [30.0, 650.0, 800.0],
                    [12.0, 12.0, 8.0]
                )
            )

        if self.phase == "APPROACH":
            # Progressive level-off before 1000 ft.
            # This avoids asking the policy to brake abruptly at 970 ft.
            return float(
                np.interp(
                    altitude,
                    [800.0, 850.0, 900.0, 940.0, 970.0],
                    [8.0, 6.0, 4.0, 2.5, 1.5]
                )
            )

        return 0.0

    def _target_forward_velocity(self):
        if self.phase == "FORWARD":
            return self.target_forward_speed

        return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.phase = "TAKEOFF"
        self.target_hold_steps = 0

        self._create_fdm()
        self._warmup_rotor()

        self.target_heading = float(
            self.fdm["attitude/heading-true-rad"]
        )

        state = self._get_state()
        self._update_phase(state["altitude"])

        obs = self._get_obs_from_state(state)

        (
            trim_collective,
            trim_elevator,
            trim_aileron,
            trim_rudder
        ) = self._get_trim_controls(
            state["altitude"]
        )

        info = self._create_info(
            state,
            self._target_vertical_speed(state["altitude"]),
            self._target_forward_velocity(),
            trim_collective,
            trim_elevator,
            trim_aileron,
            trim_rudder,
            False
        )

        return obs, info

    def step(self, action):
        self.steps += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        current_altitude = float(
            self.fdm["position/h-agl-ft"]
        )

        (
            trim_collective,
            trim_elevator,
            trim_aileron,
            trim_rudder
        ) = self._get_trim_controls(
            current_altitude
        )

        collective = float(
            np.clip(
                trim_collective
                + self.collective_scale * float(action[0]),
                0.0,
                1.0
            )
        )

        elevator = float(
            np.clip(
                trim_elevator
                + self.elevator_scale * float(action[1]),
                -1.0,
                1.0
            )
        )

        aileron = float(
            np.clip(
                trim_aileron
                + self.aileron_scale * float(action[2]),
                -1.0,
                1.0
            )
        )

        rudder = float(
            np.clip(
                trim_rudder
                + self.rudder_scale * float(action[3]),
                -1.0,
                1.0
            )
        )

        self.fdm["fcs/collective-cmd-norm"] = collective
        self.fdm["fcs/elevator-cmd-norm"] = elevator
        self.fdm["fcs/aileron-cmd-norm"] = aileron
        self.fdm["fcs/rudder-cmd-norm"] = rudder

        for _ in range(10):
            if not self.fdm.run():
                break

        state = self._get_state()
        self._update_phase(state["altitude"])

        target_vs = self._target_vertical_speed(
            state["altitude"]
        )
        target_fwd = self._target_forward_velocity()

        reward = 0.0
        terminated = False
        success = False

        altitude = state["altitude"]
        altitude_error = state["altitude_error"]
        forward_velocity = state["forward_velocity"]
        lateral_velocity = state["lateral_velocity"]
        vertical_speed = state["vertical_speed"]
        heading_error = state["heading_error"]
        pitch = state["pitch"]
        roll = state["roll"]
        roll_rate = state["roll_rate"]
        pitch_rate = state["pitch_rate"]
        yaw_rate = state["yaw_rate"]
        rotor_rpm = state["rotor_rpm"]

        # Common stability reward terms
        reward -= 1.50 * abs(heading_error)
        reward -= 0.08 * abs(lateral_velocity)

        reward -= 0.80 * abs(pitch)
        reward -= 1.00 * abs(roll)

        reward -= 0.15 * abs(roll_rate)
        reward -= 0.15 * abs(pitch_rate)
        reward -= 0.10 * abs(yaw_rate)

        reward -= 0.02 * float(
            np.sum(np.square(action))
        )

        # TAKEOFF
        if self.phase == "TAKEOFF":
            reward += 0.35 * np.clip(
                vertical_speed,
                -5.0,
                8.0
            )

            reward -= 0.20 * abs(
                vertical_speed - target_vs
            )

            reward -= 0.05 * abs(
                forward_velocity
            )

            if altitude < 8.0:
                reward -= 1.0

        # CLIMB
        elif self.phase == "CLIMB":
            # Reward useful upward motion, but do not reward climbing faster
            # and faster. v2 could still receive a strong positive term at
            # excessive vertical speed.
            reward += 0.30 * np.clip(
                vertical_speed,
                -5.0,
                target_vs + 2.0
            )

            reward -= 0.35 * abs(
                vertical_speed - target_vs
            )

            # Explicit overspeed braking pressure.
            if vertical_speed > target_vs + 3.0:
                reward -= 0.80 * (
                    vertical_speed
                    - (target_vs + 3.0)
                )

            reward -= 0.06 * abs(
                forward_velocity
            )

        # APPROACH / LEVEL-OFF
        elif self.phase == "APPROACH":
            reward -= 0.010 * abs(
                altitude_error
            )

            reward -= 0.55 * abs(
                vertical_speed - target_vs
            )

            reward -= 0.06 * abs(
                forward_velocity
            )

            # During approach the main objective is controlled deceleration
            # of the climb before entering the forward-flight phase.
            if vertical_speed > target_vs + 2.0:
                reward -= 1.20 * (
                    vertical_speed
                    - (target_vs + 2.0)
                )

        # FORWARD FLIGHT
        else:
            reward -= 0.05 * abs(
                altitude_error
            )

            reward -= 0.60 * abs(
                vertical_speed
            )

            forward_error = (
                target_fwd - forward_velocity
            )

            reward -= 0.12 * abs(
                forward_error
            )

            reward -= 0.12 * abs(
                lateral_velocity
            )

            reward -= 1.50 * abs(
                heading_error
            )

            if abs(altitude_error) < 30.0:
                reward += 3.0

            if abs(forward_error) < 10.0:
                reward += 2.0

            if (
                abs(altitude_error) < 30.0
                and abs(vertical_speed) < 3.0
                and 25.0 < forward_velocity < 45.0
                and abs(lateral_velocity) < 10.0
                and abs(heading_error) < 0.30
                and abs(pitch) < 0.30
                and abs(roll) < 0.30
                and abs(roll_rate) < 0.20
                and abs(pitch_rate) < 0.20
            ):
                self.target_hold_steps += 1
                reward += 5.0
            else:
                self.target_hold_steps = 0

        if (
            self.target_hold_steps
            >= self.required_hold_steps
        ):
            reward += 500.0
            success = True
            terminated = True

        # Safety
        if abs(pitch) > 0.75:
            reward -= 100.0
            terminated = True

        if abs(roll) > 0.75:
            reward -= 100.0
            terminated = True

        if rotor_rpm < 250.0:
            reward -= 100.0
            terminated = True

        if altitude > 1100.0:
            reward -= 150.0
            terminated = True

        if abs(lateral_velocity) > 70.0:
            reward -= 100.0
            terminated = True

        if abs(forward_velocity) > 100.0:
            reward -= 100.0
            terminated = True

        if (
            self.steps > 250
            and altitude < 10.0
        ):
            reward -= 100.0
            terminated = True

        truncated = self.steps >= self.max_steps

        obs = self._get_obs_from_state(state)

        info = self._create_info(
            state,
            target_vs,
            target_fwd,
            collective,
            elevator,
            aileron,
            rudder,
            success
        )

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )

    def _create_info(
        self,
        state,
        target_vertical_speed,
        target_forward_velocity,
        collective,
        elevator,
        aileron,
        rudder,
        success
    ):
        return {
            "phase": self.phase,

            "altitude": state["altitude"],
            "target_altitude": self.target_altitude,
            "altitude_error": state["altitude_error"],

            "forward_velocity": state["forward_velocity"],
            "target_forward_velocity": target_forward_velocity,

            "lateral_velocity": state["lateral_velocity"],

            "vertical_speed": state["vertical_speed"],
            "target_vertical_speed": target_vertical_speed,

            "heading": state["heading"],
            "target_heading": self.target_heading,
            "heading_error": state["heading_error"],

            "pitch": state["pitch"],
            "roll": state["roll"],

            "roll_rate": state["roll_rate"],
            "pitch_rate": state["pitch_rate"],
            "yaw_rate": state["yaw_rate"],

            "rotor_rpm": state["rotor_rpm"],

            "collective": collective,
            "elevator": elevator,
            "aileron": aileron,
            "rudder": rudder,

            "target_hold_steps": self.target_hold_steps,
            "success": success
        }

    def close(self):
        self.fdm = None
