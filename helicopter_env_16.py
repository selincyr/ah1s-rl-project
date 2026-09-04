import os
import numpy as np
import jsbsim
import gymnasium as gym

from gymnasium import spaces


# ==========================================================
# V16-A PHYSICS-NORMALIZATION REFERENCES
# ==========================================================
# These are physical reference/tolerance values, not reward weights.
POSITION_TOLERANCE_FT = 20.0
HORIZONTAL_SPEED_TOLERANCE_FPS = 10.0
VS_ERROR_TOLERANCE_FPS = 5.0
HEADING_TOLERANCE_RAD = 0.30
ROLL_TOLERANCE_RAD = 0.15
PITCH_TOLERANCE_RAD = 0.15


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

        # V16: takeoff-origin geodetic reference.
        self.initial_latitude_deg = None
        self.initial_longitude_deg = None
        self.previous_altitude = None

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

        # Observation vector (15):
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
        # 13 relative east displacement from takeoff point
        # 14 relative north displacement from takeoff point
        #
        # V16 retains absolute position error in the observation. This prevents
        # the policy from drifting away, stopping, and still looking like
        # a good hover only because instantaneous velocities are near zero.
        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, -1.5, -5.0, -5.0, -10.0,
                 -1.0, -1.0,
                 -3.0, -5.0,
                 -10.0, -10.0, -10.0,
                 0.0,
                 -5.0, -5.0],
                dtype=np.float32
            ),
            high=np.array(
                [10.0, 2.0, 5.0, 5.0, 10.0,
                 1.0, 1.0,
                 3.0, 5.0,
                 10.0, 10.0, 10.0,
                 2.0,
                 5.0, 5.0],
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

    def _get_relative_position(self):
        """
        Return local east/north displacement in feet from the takeoff point.

        JSBSim supplies geodetic latitude/longitude. For the small distances
        used in this curriculum, a local tangent-plane approximation is more
        than accurate enough and is much better than integrating velocities.
        """
        latitude_deg = float(
            self.fdm["position/lat-geod-deg"]
        )
        longitude_deg = float(
            self.fdm["position/long-gc-deg"]
        )

        if (
            self.initial_latitude_deg is None
            or self.initial_longitude_deg is None
        ):
            return 0.0, 0.0

        earth_radius_ft = 20925524.9

        lat0_rad = np.deg2rad(
            self.initial_latitude_deg
        )

        delta_lat_rad = np.deg2rad(
            latitude_deg - self.initial_latitude_deg
        )

        delta_lon_rad = np.deg2rad(
            longitude_deg - self.initial_longitude_deg
        )

        north_ft = float(
            earth_radius_ft * delta_lat_rad
        )

        east_ft = float(
            earth_radius_ft
            * np.cos(lat0_rad)
            * delta_lon_rad
        )

        return east_ft, north_ft

    def _get_state(self):
        altitude = float(self.fdm["position/h-agl-ft"])

        relative_east, relative_north = (
            self._get_relative_position()
        )

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
            ),
            "relative_east": relative_east,
            "relative_north": relative_north,
            "horizontal_distance": float(
                np.hypot(relative_east, relative_north)
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
                state["rotor_rpm"] / 400.0,
                state["relative_east"] / 50.0,
                state["relative_north"] / 50.0
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
        # V14 keeps the 300-ft mission structure.
        # HOVER means level-off / hover-acquisition starts at 285 ft.
        if altitude < 30.0:
            self.phase = "TAKEOFF"
        elif altitude < 285.0:
            self.phase = "CLIMB"
        else:
            self.phase = "HOVER"

    def _target_vertical_speed(self, altitude):
        # The proven vertical-speed schedule is retained unchanged in V14.
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

        self.initial_latitude_deg = float(
            self.fdm["position/lat-geod-deg"]
        )
        self.initial_longitude_deg = float(
            self.fdm["position/long-gc-deg"]
        )

        self.previous_altitude = float(
            self.fdm["position/h-agl-ft"]
        )

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
        relative_east = state["relative_east"]
        relative_north = state["relative_north"]
        horizontal_distance = state["horizontal_distance"]

        vertical_error = vertical_speed - target_vs

        # Natural hover attitude errors.
        pitch_error = pitch - self.target_hover_pitch
        roll_error = roll - self.target_hover_roll

        if self.previous_altitude is None:
            altitude_delta = 0.0
        else:
            altitude_delta = altitude - self.previous_altitude

        # ==========================================================
        # V16-A PHYSICS-NORMALIZED REWARD
        # ==========================================================
        # Main idea:
        # 1) convert physically different errors into dimensionless values,
        # 2) pass them through a bounded smooth loss,
        # 3) keep task progress (takeoff/climb) separate from tracking quality.
        #
        # This avoids stacking many unrelated hand-tuned coefficients such as
        # "0.16 * distance" and "0.32 * lateral velocity".

        # Altitude-gated horizontal corridor.
        # Below 30 ft the priority is liftoff; position control is introduced
        # gradually after the helicopter is clearly airborne.
        if altitude < 30.0:
            corridor_weight = 0.0
        elif altitude < 100.0:
            corridor_weight = float(
                np.interp(
                    altitude,
                    [30.0, 100.0],
                    [0.10, 0.50]
                )
            )
        elif altitude < 180.0:
            corridor_weight = float(
                np.interp(
                    altitude,
                    [100.0, 180.0],
                    [0.50, 1.00]
                )
            )
        else:
            corridor_weight = 1.0

        # ----------------------------------------------------------
        # Dimensionless normalized errors
        # ----------------------------------------------------------
        position_error_norm = (
            horizontal_distance
            / POSITION_TOLERANCE_FT
        )

        horizontal_speed = float(
            np.hypot(
                forward_velocity,
                lateral_velocity
            )
        )

        horizontal_speed_norm = (
            horizontal_speed
            / HORIZONTAL_SPEED_TOLERANCE_FPS
        )

        vs_error_norm = (
            abs(vertical_error)
            / VS_ERROR_TOLERANCE_FPS
        )

        heading_error_norm = (
            abs(heading_error)
            / HEADING_TOLERANCE_RAD
        )

        roll_error_norm = (
            abs(roll_error)
            / ROLL_TOLERANCE_RAD
        )

        pitch_error_norm = (
            abs(pitch_error)
            / PITCH_TOLERANCE_RAD
        )

        # Smooth bounded loss:
        #   norm = 0   -> loss = 0
        #   norm = 1   -> tanh(1)^2 ~= 0.58
        #   very large -> approaches 1
        #
        # Therefore one extreme physical variable cannot dominate the whole
        # reward only because its raw numerical scale is large.
        position_loss = float(
            np.tanh(position_error_norm) ** 2
        )
        horizontal_speed_loss = float(
            np.tanh(horizontal_speed_norm) ** 2
        )
        vs_loss = float(
            np.tanh(vs_error_norm) ** 2
        )
        heading_loss = float(
            np.tanh(heading_error_norm) ** 2
        )
        roll_loss = float(
            np.tanh(roll_error_norm) ** 2
        )
        pitch_loss = float(
            np.tanh(pitch_error_norm) ** 2
        )

        # V16-A baseline: equal normalized weights.
        tracking_cost = (
            corridor_weight * position_loss
            + horizontal_speed_loss
            + vs_loss
            + heading_loss
            + roll_loss
            + pitch_loss
        )

        reward -= tracking_cost

        # ----------------------------------------------------------
        # Action regularization
        # ----------------------------------------------------------
        # Small smooth-control cost only; this is not a flight-state error.
        reward -= 0.02 * float(
            np.mean(np.square(action))
        )

        # ----------------------------------------------------------
        # Soft safety walls
        # ----------------------------------------------------------
        # These are guard rails rather than normal tracking terms.
        if abs(roll_error) > 0.30:
            reward -= 2.0 * float(
                abs(roll_error) - 0.30
            )

        if abs(pitch_error) > 0.30:
            reward -= 2.0 * float(
                abs(pitch_error) - 0.30
            )

        if abs(roll_rate) > 0.35:
            reward -= 1.0 * float(
                abs(roll_rate) - 0.35
            )

        if abs(pitch_rate) > 0.35:
            reward -= 1.0 * float(
                abs(pitch_rate) - 0.35
            )

        if abs(yaw_rate) > 0.35:
            reward -= 1.0 * float(
                abs(yaw_rate) - 0.35
            )

        # ----------------------------------------------------------
        # Task-progress shaping
        # ----------------------------------------------------------
        if self.phase == "TAKEOFF":
            # Staying on the skids is not a solution.
            if altitude < 8.0:
                reward -= 1.0

            # Step-to-step upward progress. 0.5 ft/step is the normalization
            # reference for this short takeoff phase.
            upward_progress = float(
                np.clip(
                    altitude_delta / 0.5,
                    -1.0,
                    1.0
                )
            )
            reward += 2.0 * upward_progress

            # Positive climb-rate signal helps the policy discover collective
            # without rewarding unbounded vertical speed.
            reward += float(
                np.clip(
                    vertical_speed / 6.0,
                    0.0,
                    1.0
                )
            )

        elif self.phase == "CLIMB":
            # Continue rewarding progress, but less than during liftoff.
            climb_progress_step = float(
                np.clip(
                    altitude_delta / 0.5,
                    -1.0,
                    1.0
                )
            )
            reward += 1.0 * climb_progress_step

            # Small mission-progress signal; bounded in [0, 1].
            mission_progress = float(
                np.clip(
                    (altitude - 30.0) / 255.0,
                    0.0,
                    1.0
                )
            )
            reward += 0.5 * mission_progress

        else:
            # ------------------------------------------------------
            # Hover / level-off
            # ------------------------------------------------------
            altitude_error_norm = (
                abs(altitude_error) / 20.0
            )
            altitude_loss = float(
                np.tanh(altitude_error_norm) ** 2
            )

            # Altitude becomes an explicit tracking objective in the hover
            # phase. The other state errors are already included above.
            reward -= altitude_loss

            hover_ok = (
                abs(altitude_error) < 20.0
                and abs(vertical_speed) < 2.5
                and horizontal_speed < 8.0
                and abs(heading_error) < 0.25
                and abs(pitch_error) < 0.20
                and abs(roll_error) < 0.20
                and abs(roll_rate) < 0.15
                and abs(pitch_rate) < 0.15
                and abs(yaw_rate) < 0.20
                and horizontal_distance < 8.0
            )

            if hover_ok:
                self.target_hold_steps += 1
                reward += 2.0
            else:
                self.target_hold_steps = 0

        # ----------------------------------------------------------
        # Target-region bonus
        # ----------------------------------------------------------
        # This is intentionally small compared with the terminal success reward.
        if (
            290.0 <= altitude <= 310.0
            and abs(vertical_error) < 1.5
            and horizontal_distance < 10.0
        ):
            reward += 1.0

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

        if not terminated and horizontal_distance > 100.0:
            reward -= 180.0
            terminated = True
            termination_reason = "horizontal_position_limit"

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

        self.previous_altitude = altitude

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

            "relative_east": state["relative_east"],
            "relative_north": state["relative_north"],
            "horizontal_distance": state["horizontal_distance"],

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
