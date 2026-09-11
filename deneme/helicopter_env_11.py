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

        self.target_altitude = 300.0
        self.target_forward_speed = 0.0

        # Natural 0-knot hover attitude from AH-1S steady-flight data.
        self.target_hover_roll = -0.0493
        self.target_hover_pitch = -0.0064

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

        # Residual commands around the altitude-dependent trim.
        # V4 frequently saturated collective/aileron, so V5 uses
        # slightly smaller cyclic/collective authority while keeping
        # enough rudder authority for heading correction.
        self.collective_scale = 0.12
        self.elevator_scale = 0.035
        self.aileron_scale = 0.030
        self.rudder_scale = 0.070

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Observation vector (13):
        # 0 altitude
        # 1 altitude error
        # 2 forward velocity
        # 3 lateral velocity
        # 4 vertical speed
        # 5 sin(heading error)
        # 6 cos(heading error)
        # 7 pitch
        # 8 roll
        # 9 roll rate
        # 10 pitch rate
        # 11 yaw rate
        # 12 rotor RPM
        #
        # Using sin/cos removes the artificial discontinuity between
        # +pi and -pi that existed in the previous heading observation.
        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, -1.5, -5.0, -5.0, -10.0,
                 -1.0, -1.0,
                 -3.0, -5.0,
                 -10.0, -10.0, -10.0,
                 0.0],
                dtype=np.float32
            ),
            high=np.array(
                [10.0, 2.0, 5.0, 5.0, 10.0,
                 1.0, 1.0,
                 3.0, 5.0,
                 10.0, 10.0, 10.0,
                 2.0],
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
        heading_error = state["heading_error"]

        obs = np.array(
            [
                state["altitude"] / 400.0,
                state["altitude_error"] / 300.0,
                state["forward_velocity"] / 100.0,
                state["lateral_velocity"] / 100.0,
                state["vertical_speed"] / 30.0,

                # Continuous circular heading representation.
                np.sin(heading_error),
                np.cos(heading_error),

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
        # V11 keeps the successful V10 mission structure.
        # HOVER means level-off / hover-acquisition starts at 285 ft.
        if altitude < 30.0:
            self.phase = "TAKEOFF"
        elif altitude < 285.0:
            self.phase = "CLIMB"
        else:
            self.phase = "HOVER"

    def _target_vertical_speed(self, altitude):
        # V10 vertical-speed schedule retained in V11.
        if altitude < 30.0:
            return 6.0

        if altitude < 120.0:
            return 8.0

        if altitude < 180.0:
            return float(
                np.interp(
                    altitude,
                    [120.0, 180.0],
                    [8.0, 7.0]
                )
            )

        if altitude < 230.0:
            return float(
                np.interp(
                    altitude,
                    [180.0, 230.0],
                    [7.0, 5.0]
                )
            )

        if altitude < 260.0:
            return float(
                np.interp(
                    altitude,
                    [230.0, 260.0],
                    [5.0, 3.0]
                )
            )

        if altitude < 280.0:
            return float(
                np.interp(
                    altitude,
                    [260.0, 280.0],
                    [3.0, 1.5]
                )
            )

        if altitude < 295.0:
            return float(
                np.interp(
                    altitude,
                    [280.0, 295.0],
                    [1.5, 0.5]
                )
            )

        if altitude < 300.0:
            return float(
                np.interp(
                    altitude,
                    [295.0, 300.0],
                    [0.5, 0.0]
                )
            )

        return float(
            np.clip(
                (self.target_altitude - altitude) / 8.0,
                -2.5,
                0.0
            )
        )

    def _target_forward_velocity(self):
        # This curriculum explicitly learns a vertical climb / hover.
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
            False,
            None
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
        termination_reason = None

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

        vertical_error = vertical_speed - target_vs

        # Natural hover attitude errors.
        pitch_error = pitch - self.target_hover_pitch
        roll_error = roll - self.target_hover_roll

        # ==========================================================
        # V11 CURRICULUM REWARD: 300 FT STABLE HOVER
        # ==========================================================

        # Circular heading cost. 0 rad -> 0, pi rad -> 2.
        heading_cost = 1.0 - np.cos(heading_error)

        # V11 keeps the global heading term moderate so the lower climb
        # learned by V10 is not disrupted. Stronger heading guidance is
        # applied only in the target-acquisition / hover zone.
        reward -= 4.50 * heading_cost

        # Horizontal drift must remain small throughout the climb.
        reward -= 0.16 * abs(forward_velocity)
        reward -= 0.24 * abs(lateral_velocity)

        # Stabilize around the AH-1S natural 0-knot hover attitude.
        reward -= 3.50 * abs(pitch_error)
        reward -= 4.50 * abs(roll_error)

        reward -= 0.50 * abs(roll_rate)
        reward -= 0.50 * abs(pitch_rate)
        reward -= 0.45 * abs(yaw_rate)

        # ==========================================================
        # ACTION REGULARIZATION
        # ==========================================================

        reward -= (
            0.030 * float(action[0] ** 2)
            + 0.030 * float(action[1] ** 2)
            + 0.035 * float(action[2] ** 2)
            + 0.020 * float(action[3] ** 2)
        )

        # V5 held collective at +1 almost continuously.
        # Do not ban strong collective, but make persistent saturation costly.
        if abs(action[0]) > 0.85:
            collective_excess = abs(action[0]) - 0.85
            reward -= 2.0 * collective_excess
            reward -= 8.0 * (collective_excess ** 2)

        if abs(action[1]) > 0.85:
            elevator_excess = abs(action[1]) - 0.85
            reward -= 1.5 * elevator_excess
            reward -= 5.0 * (elevator_excess ** 2)

        if abs(action[2]) > 0.80:
            aileron_excess = abs(action[2]) - 0.80
            reward -= 1.5 * aileron_excess
            reward -= 5.0 * (aileron_excess ** 2)

        if abs(action[3]) > 0.90:
            rudder_excess = abs(action[3]) - 0.90
            reward -= 1.5 * rudder_excess

        # ==========================================================
        # SOFT SAFETY WALLS
        # ==========================================================

        if abs(roll_error) > 0.25:
            reward -= 18.0 * (
                abs(roll_error) - 0.25
            )

        if abs(pitch_error) > 0.25:
            reward -= 14.0 * (
                abs(pitch_error) - 0.25
            )

        if abs(forward_velocity) > 10.0:
            reward -= 0.70 * (
                abs(forward_velocity) - 10.0
            )

        if abs(lateral_velocity) > 8.0:
            reward -= 0.90 * (
                abs(lateral_velocity) - 8.0
            )

        if abs(yaw_rate) > 0.30:
            reward -= 2.5 * (
                abs(yaw_rate) - 0.30
            )

        # ==========================================================
        # V11 STRAIGHT-CLIMB STABILIZATION
        # ==========================================================
        # Preserve the successful V9 lower-climb behavior.
        if 30.0 < altitude < 260.0:
            reward -= 0.12 * abs(forward_velocity)
            reward -= 0.18 * abs(lateral_velocity)
            reward -= 0.80 * abs(roll_error)
            reward -= 0.35 * abs(pitch_error)
            reward -= 0.15 * abs(roll_rate)
            reward -= 0.15 * abs(pitch_rate)
            reward -= 0.20 * abs(yaw_rate)

            if (
                abs(forward_velocity) < 7.0
                and abs(lateral_velocity) < 7.0
                and abs(roll_error) < 0.15
                and abs(pitch_error) < 0.15
            ):
                reward += 1.5

        # ==========================================================
        # V11 TARGET APPROACH / LEVEL-OFF
        # ==========================================================
        # V10 learned the important vertical level-off behavior.
        # V11 preserves that behavior and shifts the extra reward pressure
        # toward horizontal drift and heading stabilization.
        if altitude >= 260.0:
            reward -= 0.36 * abs(forward_velocity)
            reward -= 0.52 * abs(lateral_velocity)
            reward -= 1.20 * abs(roll_error)
            reward -= 0.50 * abs(pitch_error)
            reward -= 0.70 * abs(vertical_speed)

            # Heading/yaw become more important while approaching hover.
            reward -= 1.60 * abs(heading_error)
            reward -= 0.80 * abs(yaw_rate)

        # ==========================================================
        # V11 HOVER-ZONE HORIZONTAL / YAW STABILIZATION
        # ==========================================================
        # V10 FINAL repeatedly stayed close to 300 ft with VS near zero,
        # but failed the hold because forward/lateral drift and heading
        # error remained too large. These dense terms guide those states
        # toward zero without changing the vertical-speed schedule.
        if altitude >= 285.0:
            reward -= 0.65 * abs(forward_velocity)
            reward -= 0.90 * abs(lateral_velocity)
            reward -= 2.50 * heading_cost
            reward -= 1.00 * abs(yaw_rate)

            # Smooth bonuses are useful even before the strict hover
            # condition is satisfied.
            reward += 2.0 * np.exp(
                -abs(forward_velocity) / 5.0
            )
            reward += 3.0 * np.exp(
                -abs(lateral_velocity) / 5.0
            )
            reward += 4.0 * np.exp(
                -abs(heading_error) / 0.40
            )

            # Entering the useful hover corridor should be clearly better
            # than merely circling near the correct altitude.
            if (
                abs(forward_velocity) < 10.0
                and abs(lateral_velocity) < 10.0
                and abs(heading_error) < 0.45
            ):
                reward += 5.0

            if (
                abs(forward_velocity) < 8.0
                and abs(lateral_velocity) < 8.0
                and abs(heading_error) < 0.30
                and abs(yaw_rate) < 0.20
            ):
                reward += 7.0

        # ==========================================================
        # 280 FT+ VERTICAL BRAKING
        # ==========================================================
        if altitude >= 280.0:
            reward -= 1.20 * abs(vertical_speed)

            if vertical_speed > 1.5:
                reward -= 3.00 * (
                    vertical_speed - 1.5
                )

        # ==========================================================
        # 290 FT+ COLLECTIVE BRAKING
        # ==========================================================
        # action[0] is the residual around altitude-dependent trim.
        if altitude >= 290.0 and action[0] > 0.15:
            collective_excess = action[0] - 0.15
            reward -= 5.0 * collective_excess
            reward -= 10.0 * (
                collective_excess ** 2
            )

        # ==========================================================
        # STATE-ACTION COUPLED BRAKING
        # ==========================================================
        # If the aircraft is already climbing near the target,
        # continued positive collective residual is explicitly wrong.
        if (
            altitude >= 285.0
            and vertical_speed > 1.0
            and action[0] > 0.0
        ):
            reward -= (
                7.0
                * float(action[0])
                * vertical_speed
            )

        # ==========================================================
        # ABOVE 300 FT: STOP THE CLIMB
        # ==========================================================
        if (
            altitude > 300.0
            and vertical_speed > 0.0
        ):
            altitude_excess = altitude - 300.0

            reward -= 5.0 * vertical_speed
            reward -= 0.40 * altitude_excess

            if action[0] > 0.0:
                reward -= 10.0 * float(action[0])

        # ==========================================================
        # GOOD LEVEL-OFF BONUS
        # ==========================================================
        if (
            290.0 <= altitude <= 310.0
            and abs(vertical_speed) < 1.5
        ):
            reward += 4.0

        if (
            295.0 <= altitude <= 305.0
            and abs(vertical_speed) < 1.0
            and abs(forward_velocity) < 8.0
            and abs(lateral_velocity) < 8.0
            and abs(heading_error) < 0.30
        ):
            reward += 10.0

        # ==========================================================
        # TARGET VERTICAL-SPEED TRACKING
        # ==========================================================

        tracking_bonus = 4.0 * np.exp(
            -abs(vertical_error) / 2.0
        )

        reward += tracking_bonus

        reward -= 0.60 * abs(
            vertical_error
        )

        # Strong overspeed penalty.
        if vertical_speed > target_vs + 1.5:
            overspeed = (
                vertical_speed
                - (target_vs + 1.5)
            )

            reward -= 1.40 * overspeed
            reward -= 0.10 * (
                overspeed ** 2
            )

        # Strong unwanted-descent penalty while still below target.
        if (
            altitude < self.target_altitude - 20.0
            and vertical_speed < -1.5
        ):
            reward -= 1.20 * (
                abs(vertical_speed) - 1.5
            )

        # ==========================================================
        # TAKEOFF
        # ==========================================================

        if self.phase == "TAKEOFF":
            if altitude < 8.0:
                reward -= 1.0

            reward += 0.75 * np.clip(
                altitude / 30.0,
                0.0,
                1.0
            )

        # ==========================================================
        # CLIMB
        # ==========================================================

        elif self.phase == "CLIMB":
            climb_progress = np.clip(
                (altitude - 30.0) / 255.0,
                0.0,
                1.0
            )

            reward += 1.50 * climb_progress

            stable_climb = (
                abs(vertical_error) < 1.75
                and abs(forward_velocity) < 8.0
                and abs(lateral_velocity) < 8.0
                and abs(heading_error) < 0.30
                and abs(pitch_error) < 0.22
                and abs(roll_error) < 0.22
                and abs(roll_rate) < 0.18
                and abs(pitch_rate) < 0.18
                and abs(yaw_rate) < 0.20
            )

            if stable_climb:
                reward += 3.0

        # ==========================================================
        # HOVER / LEVEL-OFF
        # ==========================================================

        else:
            reward -= 0.090 * abs(
                altitude_error
            )

            # Keep the V10 vertical-control pressure intact.
            reward -= 0.65 * abs(
                vertical_error
            )

            reward -= 2.00 * abs(roll_error)
            reward -= 1.50 * abs(pitch_error)

            # V11: once in hover acquisition, horizontal and heading
            # precision must improve together with altitude control.
            reward -= 0.35 * abs(forward_velocity)
            reward -= 0.50 * abs(lateral_velocity)
            reward -= 1.50 * heading_cost

            if abs(altitude_error) < 40.0:
                reward += 2.5

            if abs(altitude_error) < 20.0:
                reward += 5.0

            if abs(altitude_error) < 10.0:
                reward += 2.0

            hover_ok = (
                abs(altitude_error) < 20.0
                and abs(vertical_speed) < 2.5
                and abs(forward_velocity) < 8.0
                and abs(lateral_velocity) < 8.0
                and abs(heading_error) < 0.25
                and abs(pitch_error) < 0.20
                and abs(roll_error) < 0.20
                and abs(roll_rate) < 0.15
                and abs(pitch_rate) < 0.15
                and abs(yaw_rate) < 0.20
            )

            if hover_ok:
                self.target_hold_steps += 1

                # V11 gives progressively more value to maintaining the
                # condition instead of touching it for only one step.
                reward += 10.0
                reward += 0.05 * float(
                    self.target_hold_steps
                )
            else:
                self.target_hold_steps = 0

        # ==========================================================
        # SUCCESS
        # ==========================================================

        if (
            self.target_hold_steps
            >= self.required_hold_steps
        ):
            reward += 1000.0
            success = True
            terminated = True
            termination_reason = "success_hover_300ft"

        # ==========================================================
        # SAFETY TERMINATIONS
        # ==========================================================

        if not terminated and abs(pitch) > 0.75:
            reward -= 150.0
            terminated = True
            termination_reason = "pitch_limit"

        if not terminated and abs(roll) > 0.75:
            reward -= 150.0
            terminated = True
            termination_reason = "roll_limit"

        if not terminated and rotor_rpm < 250.0:
            reward -= 150.0
            terminated = True
            termination_reason = "low_rotor_rpm"

        # 300 ft curriculum should not wander far above the target.
        if not terminated and altitude > 330.0:
            reward -= 180.0
            terminated = True
            termination_reason = "altitude_overshoot"

        if not terminated and abs(lateral_velocity) > 50.0:
            reward -= 150.0
            terminated = True
            termination_reason = "lateral_velocity_limit"

        if not terminated and abs(forward_velocity) > 70.0:
            reward -= 150.0
            terminated = True
            termination_reason = "forward_velocity_limit"

        if (
            not terminated
            and self.steps > 250
            and altitude < 10.0
        ):
            reward -= 150.0
            terminated = True
            termination_reason = "failed_takeoff_or_crash"

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
            success,
            termination_reason
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
        success,
        termination_reason
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
            "heading_error_sin": float(
                np.sin(state["heading_error"])
            ),
            "heading_error_cos": float(
                np.cos(state["heading_error"])
            ),

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
            "success": success,
            "termination_reason": termination_reason
        }

    def close(self):
        self.fdm = None
