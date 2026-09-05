import gymnasium as gym
from gymnasium import spaces
import numpy as np

from helicopter_env_v2 import HelicopterEnvV2


class HelicopterEnvStage1Straight(HelicopterEnvV2):

    TARGET_ALTITUDE = 300.0

    # Geometrik kalite
    SUCCESS_MAX_DRIFT = 8.0
    SUCCESS_FINAL_DRIFT = 5.0

    # Eğitim sırasında tamamen kaçmasına izin vermiyoruz
    FAILURE_DRIFT = 25.0

    MAX_TIME = 120.0

    BASE_ELEVATOR = -0.15390
    BASE_AILERON = 0.19100

    def __init__(self):

        # Parent reset sırasında _get_obs çağrılırsa hazır olsun
        self.north_position = 0.0
        self.east_position = 0.0

        self.max_horizontal_drift = 0.0
        self.horizontal_path = 0.0

        self.previous_altitude = 0.0
        self.previous_action_straight = np.zeros(
            4,
            dtype=np.float32
        )

        self.stable_steps_straight = 0
        self.steps_straight = 0

        super().__init__()

        # Eski 12 observation
        # +
        # north position
        # east position
        #
        # = 14
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(14,),
            dtype=np.float32
        )

        self.control_dt = (
            self.JSBSIM_DT
            *
            self.PHYSICS_STEPS
        )

        self.max_steps_straight = int(
            self.MAX_TIME
            /
            self.control_dt
        )


    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_obs(self):

        base_obs = super()._get_obs()

        position_obs = np.array(
            [
                np.clip(
                    self.north_position / 25.0,
                    -10.0,
                    10.0
                ),

                np.clip(
                    self.east_position / 25.0,
                    -10.0,
                    10.0
                ),
            ],
            dtype=np.float32
        )

        return np.concatenate(
            [
                base_obs.astype(
                    np.float32
                ),
                position_obs
            ]
        ).astype(
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

        self.north_position = 0.0
        self.east_position = 0.0

        self.max_horizontal_drift = 0.0
        self.horizontal_path = 0.0

        self.previous_action_straight = np.zeros(
            4,
            dtype=np.float32
        )

        self.stable_steps_straight = 0
        self.steps_straight = 0

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        s = self._raw_state()

        self.previous_altitude = (
            s["altitude"]
        )

        obs = self._get_obs()

        info = dict(info)

        info.update(
            {
                "north_position":
                    0.0,

                "east_position":
                    0.0,

                "horizontal_drift":
                    0.0,

                "max_horizontal_drift":
                    0.0,

                "horizontal_path":
                    0.0,

                "success":
                    False,
            }
        )

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

        self.steps_straight += 1


        # ====================================================
        # PPO CONTROLS
        # ====================================================

        # Collective:
        # çalışan Stage 1 mapping aynen korunuyor
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


        # PPO'ya yalnızca KÜÇÜK cyclic authority
        elevator = (
            self.BASE_ELEVATOR
            +
            0.006
            *
            float(action[1])
        )

        aileron = (
            self.BASE_AILERON
            +
            0.006
            *
            float(action[2])
        )

        elevator = float(
            np.clip(
                elevator,
                -0.1600,
                -0.1480
            )
        )

        aileron = float(
            np.clip(
                aileron,
                0.1850,
                0.1970
            )
        )


        # Pedal sabit
        rudder = 0.39


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
            self.PHYSICS_STEPS
        ):

            if not self.fdm.run():

                physics_ok = False
                break


        s = self._raw_state()


        # ====================================================
        # EARTH-FRAME POSITION
        # ====================================================

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


        self.north_position += (
            vn
            *
            self.control_dt
        )

        self.east_position += (
            ve
            *
            self.control_dt
        )


        horizontal_speed = float(
            np.sqrt(
                vn**2
                +
                ve**2
            )
        )


        self.horizontal_path += (
            horizontal_speed
            *
            self.control_dt
        )


        drift = float(
            np.sqrt(
                self.north_position**2
                +
                self.east_position**2
            )
        )


        self.max_horizontal_drift = max(
            self.max_horizontal_drift,
            drift
        )


        # ====================================================
        # REWARD
        # ====================================================

        altitude = s["altitude"]
        vertical_speed = (
            s["vertical_speed"]
        )

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

        self.previous_altitude = altitude


        # İstenen climb profile
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


        reward = 0.0


        # --------------------------------------------
        # 1) Yukarı çık
        # --------------------------------------------

        reward += (
            2.0
            *
            altitude_gain
        )


        # --------------------------------------------
        # 2) Doğru vertical speed
        # --------------------------------------------

        reward -= (
            0.10
            *
            abs(
                vertical_speed
                -
                desired_vs
            )
        )


        # --------------------------------------------
        # 3) ANA HEDEF:
        # X/Y'de başlangıç noktasında kal
        # --------------------------------------------

        reward -= (
            0.30
            *
            drift
        )


        # --------------------------------------------
        # 4) S/yay çizmesini engelle
        # --------------------------------------------

        reward -= (
            0.20
            *
            horizontal_speed
        )


        # 5 ft dışına çıkınca ekstra ağır ceza
        if drift > 5.0:

            reward -= (
                1.0
                *
                (
                    drift
                    -
                    5.0
                )
            )


        # 10 ft dışına çıkınca daha da ağır
        if drift > 10.0:

            reward -= (
                2.0
                *
                (
                    drift
                    -
                    10.0
                )
            )


        # --------------------------------------------
        # 5) Attitude
        # --------------------------------------------

        reward -= (
            2.0
            *
            abs(
                s["pitch"]
            )
        )

        reward -= (
            2.0
            *
            abs(
                s["roll"]
            )
        )


        # --------------------------------------------
        # 6) Smooth actions
        # --------------------------------------------

        action_change = np.mean(
            np.abs(
                action
                -
                self.previous_action_straight
            )
        )

        reward -= (
            0.05
            *
            action_change
        )

        self.previous_action_straight = (
            action.copy()
        )


        # ====================================================
        # HOVER QUALITY
        # ====================================================

        hover_good = (

            altitude_error < 10.0

            and
            abs(
                vertical_speed
            ) < 1.0

            and
            horizontal_speed < 2.0

            and
            drift < (
                self.SUCCESS_MAX_DRIFT
            )

            and
            abs(
                s["pitch"]
            ) < 0.12

            and
            abs(
                s["roll"]
            ) < 0.12
        )


        if hover_good:

            self.stable_steps_straight += 1

            reward += 3.0

        else:

            self.stable_steps_straight = 0


        required_stable_steps = int(
            10.0
            /
            self.control_dt
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        success = (

            self.stable_steps_straight
            >=
            required_stable_steps

            and
            self.max_horizontal_drift
            <=
            self.SUCCESS_MAX_DRIFT

            and
            drift
            <=
            self.SUCCESS_FINAL_DRIFT
        )


        if success:

            reward += 1000.0


        # ====================================================
        # FAILURE
        # ====================================================

        failure = False


        if not physics_ok:
            failure = True


        if altitude > 380.0:
            failure = True


        if (
            abs(
                s["pitch"]
            ) > 0.60
        ):
            failure = True


        if (
            abs(
                s["roll"]
            ) > 0.60
        ):
            failure = True


        if (
            s["rpm"] < 260.0
        ):
            failure = True


        if (
            drift
            >
            self.FAILURE_DRIFT
        ):

            failure = True


        if failure:

            reward -= 500.0


        terminated = bool(
            success
            or
            failure
        )


        truncated = bool(
            self.steps_straight
            >=
            self.max_steps_straight
        )


        obs = self._get_obs()


        info = {

            "success":
                success,

            "altitude":
                altitude,

            "altitude_error":
                altitude_error,

            "vertical_speed":
                vertical_speed,

            "forward_velocity":
                s[
                    "forward_velocity"
                ],

            "lateral_velocity":
                s[
                    "lateral_velocity"
                ],

            "north_velocity":
                vn,

            "east_velocity":
                ve,

            "north_position":
                self.north_position,

            "east_position":
                self.east_position,

            "horizontal_drift":
                drift,

            "max_horizontal_drift":
                self.max_horizontal_drift,

            "horizontal_path":
                self.horizontal_path,

            "collective":
                collective,

            "elevator":
                elevator,

            "aileron":
                aileron,

            "rudder":
                rudder,

            "pitch":
                s["pitch"],

            "roll":
                s["roll"],

            "rpm":
                s["rpm"],
        }


        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )
