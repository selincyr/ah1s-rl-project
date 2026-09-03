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

        # V13: takeoff-origin geodetic reference.
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
        # V13 adds absolute position error to the observation. This prevents
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
        # V14 CURRICULUM REWARD: 300 FT STABLE HOVER
        # ==========================================================

        # Circular heading cost. 0 rad -> 0, pi rad -> 2.
        heading_cost = 1.0 - np.cos(heading_error)

        # V14: continuous circular heading cost. Keep it firm but not dominant,
        # so the policy does not sacrifice roll/translation just to turn the nose.
        reward -= 4.00 * heading_cost

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
        # V14 ALTITUDE-GATED VERTICAL-CORRIDOR SHAPING
        # ==========================================================
        # V13 made position-holding attractive before the helicopter had
        # learned to leave the ground. V14 keeps X/Y in the observation,
        # but does NOT meaningfully punish position during the first 30 ft.
        # The corridor is introduced gradually as altitude increases.
        if altitude < 30.0:
            corridor_weight = 0.0
        elif altitude < 100.0:
            corridor_weight = float(
                np.interp(
                    altitude,
                    [30.0, 100.0],
                    [0.05, 0.25]
                )
            )
        elif altitude < 200.0:
            corridor_weight = float(
                np.interp(
                    altitude,
                    [100.0, 200.0],
                    [0.25, 0.60]
                )
            )
        else:
            corridor_weight = 1.0

        reward -= corridor_weight * 0.10 * horizontal_distance

        if horizontal_distance > 5.0:
            corridor_excess = horizontal_distance - 5.0
            reward -= (
                corridor_weight
                * 0.06
                * (corridor_excess ** 2)
            )

        if horizontal_distance > 12.0:
            corridor_excess = horizontal_distance - 12.0
            reward -= (
                corridor_weight
                * 0.18
                * (corridor_excess ** 2)
            )

        # Centerline bonus is intentionally disabled on the ground.
        # It becomes useful only after a real climb has started.
        if altitude >= 30.0 and horizontal_distance < 5.0:
            reward += 1.0 * corridor_weight

        # ==========================================================
        # ACTION REGULARIZATION
        # ==========================================================

        reward -= (
            0.030 * float(action[0] ** 2)
            + 0.030 * float(action[1] ** 2)
            + 0.045 * float(action[2] ** 2)
            + 0.035 * float(action[3] ** 2)
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
        # V14 STRAIGHT-CLIMB STABILIZATION
        # ==========================================================
        # Preserve the successful V10 lower-climb behavior.
        if 30.0 < altitude < 260.0:
            reward -= 0.12 * abs(forward_velocity)
            reward -= 0.18 * abs(lateral_velocity)
            reward -= 0.80 * abs(roll_error)
            reward -= 0.35 * abs(pitch_error)
            reward -= 0.15 * abs(roll_rate)
            reward -= 0.15 * abs(pitch_rate)
            reward -= 0.20 * abs(yaw_rate)
            reward -= 0.16 * corridor_weight * horizontal_distance

            if (
                abs(forward_velocity) < 7.0
                and abs(lateral_velocity) < 7.0
                and abs(roll_error) < 0.15
                and abs(pitch_error) < 0.15
                and horizontal_distance < 8.0
            ):
                reward += 1.5

        # ==========================================================
        # V14 TARGET APPROACH / LEVEL-OFF
        # ==========================================================
        # V10 FINAL learned useful level-off behavior near 300 ft.
        # V12 keeps that vertical-control logic and adds controlled
        # horizontal/yaw stabilization without increasing cyclic freedom.
        if altitude >= 260.0:
            reward -= 0.30 * abs(forward_velocity)
            reward -= 0.40 * abs(lateral_velocity)
            reward -= 1.20 * abs(roll_error)
            reward -= 0.50 * abs(pitch_error)
            reward -= 0.70 * abs(vertical_speed)

            # Preserve heading during level-off, but do not let yaw
            # correction dominate translational/attitude stability.
            reward -= 0.90 * abs(heading_error)
            reward -= 0.70 * abs(yaw_rate)
            reward -= 0.22 * horizontal_distance

        # ==========================================================
        # V14 HORIZONTAL-DRIFT SHAPING (285 FT+)
        # ==========================================================
        # Linear penalties are gentle near zero. Once drift grows, the
        # quadratic excess makes continued acceleration increasingly costly.
        if altitude >= 285.0:
            abs_fwd = abs(forward_velocity)
            abs_lat = abs(lateral_velocity)

            reward -= 0.18 * (abs_fwd ** 2) / 8.0
            reward -= 0.24 * (abs_lat ** 2) / 8.0

            if abs_fwd > 8.0:
                fwd_excess = abs_fwd - 8.0
                reward -= 0.18 * (fwd_excess ** 2)

            if abs_lat > 8.0:
                lat_excess = abs_lat - 8.0
                reward -= 0.24 * (lat_excess ** 2)

            # Earlier roll protection than V10/V11. V11 showed that
            # heading correction can turn into a large bank/translation.
            if abs(roll_error) > 0.18:
                roll_excess = abs(roll_error) - 0.18
                reward -= 18.0 * roll_excess
                reward -= 35.0 * (roll_excess ** 2)

            # Discourage yaw-rate growth before heading correction
            # becomes a large oscillation.
            if abs(yaw_rate) > 0.20:
                reward -= 3.0 * (abs(yaw_rate) - 0.20)

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
        # V14 COMBINED HOVER-ACQUISITION BONUS
        # ==========================================================
        # Keep V10's useful level-off signal, then reward simultaneous
        # altitude, VS, horizontal-drift and heading stability.
        if (
            290.0 <= altitude <= 310.0
            and abs(vertical_speed) < 1.5
        ):
            reward += 4.0

        if 290.0 <= altitude <= 310.0:
            hover_quality = (
                np.exp(-abs(altitude_error) / 8.0)
                * np.exp(-abs(vertical_speed) / 1.5)
                * np.exp(-abs(forward_velocity) / 7.0)
                * np.exp(-abs(lateral_velocity) / 7.0)
                * np.exp(-abs(heading_error) / 0.35)
                * np.exp(-horizontal_distance / 6.0)
            )
            reward += 8.0 * float(hover_quality)

        if (
            295.0 <= altitude <= 305.0
            and abs(vertical_speed) < 1.0
            and abs(forward_velocity) < 6.0
            and abs(lateral_velocity) < 6.0
            and abs(heading_error) < 0.25
            and abs(roll_error) < 0.15
            and abs(pitch_error) < 0.15
            and horizontal_distance < 6.0
        ):
            reward += 12.0

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
            # V14: leaving the ground must be more valuable than simply
            # remaining perfectly centered at the takeoff point.
            if altitude < 8.0:
                reward -= 1.5

            # Reward actual upward motion. At ~6 ft/s target climb rate this
            # gives a clear positive learning signal immediately after lift-off.
            reward += 2.0 * float(
                np.clip(
                    vertical_speed / 6.0,
                    0.0,
                    1.0
                )
            )

            # Reward step-to-step altitude progress, without encouraging
            # uncontrolled overspeed because the existing VS tracking and
            # overspeed penalties remain active.
            reward += 4.0 * float(
                np.clip(
                    altitude_delta,
                    -0.5,
                    0.75
                )
            )

            # Smooth altitude progress through the first 30 ft.
            reward += 2.5 * float(
                np.clip(
                    (altitude - 6.0) / 24.0,
                    0.0,
                    1.0
                )
            )

            # Small exploration encouragement for positive residual
            # collective while still very close to the ground.
            if altitude < 12.0 and action[0] > 0.0:
                reward += 0.75 * float(action[0])

        # ==========================================================
        # CLIMB
        # ==========================================================

        elif self.phase == "CLIMB":
            # Continue rewarding upward progress, but less strongly than
            # during the actual liftoff phase.
            reward += 1.5 * float(
                np.clip(
                    altitude_delta,
                    -0.5,
                    0.75
                )
            )

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
                and horizontal_distance < 10.0
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

            # At the target, vertical-speed precision remains critical.
            reward -= 0.65 * abs(
                vertical_error
            )

            # V14: once in hover acquisition, translation and yaw must
            # settle together instead of trading one error for another.
            reward -= 0.45 * abs(forward_velocity)
            reward -= 0.60 * abs(lateral_velocity)
            reward -= 1.00 * abs(heading_error)
            reward -= 0.45 * horizontal_distance

            reward -= 2.00 * abs(roll_error)
            reward -= 1.50 * abs(pitch_error)

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
                and horizontal_distance < 8.0
            )

            if hover_ok:
                self.target_hold_steps += 1
                reward += 8.0
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
