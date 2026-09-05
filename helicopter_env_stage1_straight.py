import gymnasium as gym
from gymnasium import spaces
import numpy as np

from helicopter_env_v2 import HelicopterEnvV2


class HelicopterEnvStage1Straight(HelicopterEnvV2):

    # ========================================================
    # TARGETS
    # ========================================================

    TARGET_ALTITUDE = 300.0

    # "Çubuk gibi" kalkış kriterleri
    SUCCESS_MAX_DRIFT = 8.0
    SUCCESS_FINAL_DRIFT = 5.0

    # Eğitim sırasında fazla kaçarsa episode bitsin
    FAILURE_DRIFT = 25.0

    MAX_TIME = 120.0

    # Fine grid search sonucunda seçtiğimiz merkez trim
    BASE_ELEVATOR = -0.15390
    BASE_AILERON = 0.19100

    BASE_RUDDER = 0.39


    # ========================================================
    # INIT
    # ========================================================

    def __init__(self):

        # Parent __init__ sırasında _get_obs çağrılabilme ihtimali
        # olduğu için önceden oluşturuyoruz.
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

        # Parent env = 12 observation
        # +
        # North position
        # East position
        # =
        # 14 observation
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

        obs = np.concatenate(
            [
                base_obs.astype(np.float32),
                position_obs
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

        self.previous_altitude = float(
            s["altitude"]
        )

        obs = self._get_obs()

        info = dict(info)

        info.update(
            {
                "north_position": 0.0,
                "east_position": 0.0,

                "horizontal_drift": 0.0,
                "max_horizontal_drift": 0.0,
                "horizontal_path": 0.0,

                "success": False,
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

        # ----------------------------------------------------
        # ACTION 0:
        # COLLECTIVE
        #
        # Çalışan Stage 1 mapping aynen korunuyor
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ACTION 1:
        # ELEVATOR
        #
        # PPO'ya yalnızca küçük authority
        # ----------------------------------------------------

        elevator = (
            self.BASE_ELEVATOR
            +
            0.006
            *
            float(action[1])
        )

        elevator = float(
            np.clip(
                elevator,
                -0.1600,
                -0.1480
            )
        )


        # ----------------------------------------------------
        # ACTION 2:
        # AILERON
        # ----------------------------------------------------

        aileron = (
            self.BASE_AILERON
            +
            0.006
            *
            float(action[2])
        )

        aileron = float(
            np.clip(
                aileron,
                0.1850,
                0.1970
            )
        )


        # ----------------------------------------------------
        # ACTION 3 şu an kullanılmıyor
        # Pedal sabit
        # ----------------------------------------------------

        rudder = self.BASE_RUDDER


        # ====================================================
        # APPLY CONTROLS
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
        # RUN JSBSIM
        # ====================================================

        physics_ok = True

        for _ in range(
            self.PHYSICS_STEPS
        ):

            result = self.fdm.run()

            if not result:

                physics_ok = False
                break


        # ====================================================
        # STATE
        # ====================================================

        s = self._raw_state()

        altitude = float(
            s["altitude"]
        )

        vertical_speed = float(
            s["vertical_speed"]
        )

        forward_velocity = float(
            s["forward_velocity"]
        )

        lateral_velocity = float(
            s["lateral_velocity"]
        )

        pitch = float(
            s["pitch"]
        )

        roll = float(
            s["roll"]
        )


        # ====================================================
        # EARTH-FRAME VELOCITY
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


        # ====================================================
        # POSITION INTEGRATION
        # ====================================================

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
                vn ** 2
                +
                ve ** 2
            )
        )


        self.horizontal_path += (
            horizontal_speed
            *
            self.control_dt
        )


        horizontal_drift = float(
            np.sqrt(
                self.north_position ** 2
                +
                self.east_position ** 2
            )
        )


        self.max_horizontal_drift = max(
            self.max_horizontal_drift,
            horizontal_drift
        )


        # ====================================================
        # ALTITUDE
        # ====================================================

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


        # ====================================================
        # DESIRED CLIMB PROFILE
        # ====================================================

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


        # ====================================================
        # REWARD
        # ====================================================

        reward = 0.0


        # ----------------------------------------------------
        # 1. CLIMB REWARD
        # ----------------------------------------------------

        reward += (
            2.0
            *
            altitude_gain
        )


        # ----------------------------------------------------
        # 2. VERTICAL SPEED TRACKING
        # ----------------------------------------------------

        reward -= (
            0.10
            *
            abs(
                vertical_speed
                -
                desired_vs
            )
        )


        # ----------------------------------------------------
        # 3. POSITION ERROR
        #
        # ANA HEDEF:
        #
        # başlangıç noktasının üstünden kaçma
        # ----------------------------------------------------

        reward -= (
            0.30
            *
            horizontal_drift
        )


        # ----------------------------------------------------
        # 4. HORIZONTAL MOTION PENALTY
        #
        # S / yay çizmesini pahalı hale getiriyor
        # ----------------------------------------------------

        reward -= (
            0.20
            *
            horizontal_speed
        )


        # ----------------------------------------------------
        # 5 FT DIŞINDA EKSTRA CEZA
        # ----------------------------------------------------

        if horizontal_drift > 5.0:

            reward -= (
                1.0
                *
                (
                    horizontal_drift
                    -
                    5.0
                )
            )


        # ----------------------------------------------------
        # 10 FT DIŞINDA ÇOK DAHA AĞIR CEZA
        # ----------------------------------------------------

        if horizontal_drift > 10.0:

            reward -= (
                2.0
                *
                (
                    horizontal_drift
                    -
                    10.0
                )
            )


        # ----------------------------------------------------
        # 6. ATTITUDE PENALTY
        # ----------------------------------------------------

        reward -= (
            2.0
            *
            abs(pitch)
        )

        reward -= (
            2.0
            *
            abs(roll)
        )


        # ----------------------------------------------------
        # 7. ACTION SMOOTHNESS
        # ----------------------------------------------------

        action_change = float(
            np.mean(
                np.abs(
                    action
                    -
                    self.previous_action_straight
                )
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

            horizontal_drift
            <
            self.SUCCESS_MAX_DRIFT

            and

            abs(
                pitch
            ) < 0.12

            and

            abs(
                roll
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
        #
        # Sadece 300 ft'e çıkmak YETMEZ.
        #
        # Bütün uçuş boyunca max drift <= 8 ft
        # final drift <= 5 ft
        # 10 saniye stabil hover
        # ====================================================

        success = bool(

            self.stable_steps_straight
            >=
            required_stable_steps

            and

            self.max_horizontal_drift
            <=
            self.SUCCESS_MAX_DRIFT

            and

            horizontal_drift
            <=
            self.SUCCESS_FINAL_DRIFT
        )


        if success:

            reward += 1000.0


        # ====================================================
        # FAILURE
        # ====================================================

        failure = False


        # JSBSim failure
        if not physics_ok:

            failure = True


        # Çok yükselme
        if altitude > 380.0:

            failure = True


        # Çok fazla yatay kaçış
        if (
            horizontal_drift
            >
            self.FAILURE_DRIFT
        ):

            failure = True


        # Attitude güvenliği
        if abs(pitch) > 0.60:

            failure = True


        if abs(roll) > 0.60:

            failure = True


        # NOT:
        #
        # Önceki sürümde burada:
        #
        # s["rpm"]
        #
        # kullanılmıştı.
        #
        # _raw_state() rpm anahtarı döndürmediği için
        # KeyError oluşuyordu.
        #
        # RPM termination şimdilik YOK.
        #
        # Rotor/governor hazırlığı zaten parent reset()
        # tarafından yapılıyor.


        if failure:

            reward -= 500.0


        # ====================================================
        # TERMINATION
        # ====================================================

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


        # ====================================================
        # OBS
        # ====================================================

        obs = self._get_obs()


        # ====================================================
        # RPM
        #
        # info için doğrudan JSBSim'den okumayı deniyoruz.
        # Bulunamazsa NaN.
        # ====================================================

        try:

            rpm = float(
                self.fdm[
                    "propulsion/engine[0]/propeller-rpm"
                ]
            )

        except Exception:

            rpm = np.nan


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
                vertical_speed,

            "forward_velocity":
                forward_velocity,

            "lateral_velocity":
                lateral_velocity,

            "north_velocity":
                vn,

            "east_velocity":
                ve,

            "north_position":
                self.north_position,

            "east_position":
                self.east_position,

            "horizontal_speed":
                horizontal_speed,

            "horizontal_drift":
                horizontal_drift,

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
                pitch,

            "roll":
                roll,

            "rpm":
                rpm,
        }


        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )
