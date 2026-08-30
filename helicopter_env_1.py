import os
import numpy as np
import jsbsim
import gymnasium as gym
from gymnasium import spaces


class HelicopterEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # =========================
        # JSBSIM DOSYA YOLLARI
        # =========================

        self.root_dir = os.path.dirname(jsbsim.__file__)

        project_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.script_path = os.path.join(
            project_dir,
            "scripts",
            "ah1s_rl_start.xml"
        )

        # =========================
        # ACTION SPACE
        # =========================

        # action[0] = collective
        # action[1] = elevator
        # action[2] = aileron
        # action[3] = rudder
        #
        # PPO açısından hepsi
        # -1 ile +1 arasında olacak.

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # =========================
        # OBSERVATION SPACE
        # =========================

        # 0 = altitude (ft AGL)
        # 1 = forward velocity (ft/s)
        # 2 = vertical speed (ft/s)
        # 3 = heading (rad)
        # 4 = pitch (rad)
        # 5 = roll (rad)
        # 6 = yaw rate (rad/s)
        # 7 = rotor RPM

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
                    700
                ],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        # =========================
        # TASK 1 HEDEFLERI
        # =========================

        # Hedef irtifa:
        # 1000 ft AGL
        self.target_altitude = 1000.0

        # Cruise sırasında hedef ileri hız
        # 35 ft/s yaklaşık 21 knot
        self.target_velocity = 35.0

        # Başlangıç heading'i
        # reset sırasında alınacak
        self.target_heading = None

        # Başlangıç fazı
        self.phase = "TAKEOFF"

        # Cruise sayacı
        self.cruise_steps = 0

        # Yaklaşık 15 saniye
        self.required_cruise_steps = 200

        # Maksimum episode uzunluğu
        self.max_steps = 5000

        self.steps = 0

        self.fdm = None

    # ==========================================
    # JSBSIM OLUŞTURMA
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

        # PPO kontrolü almadan önce
        # rotor çalışma RPM'ine geliyor.

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

        self.cruise_steps = 0

        self.phase = "TAKEOFF"

        # JSBSim yeniden başlat
        self._create_fdm()

        # Rotor hazır hale gelsin
        self._warmup_rotor()

        # Başlangıç yönünü
        # hedef heading yap
        self.target_heading = float(
            self.fdm[
                "attitude/heading-true-rad"
            ]
        )

        obs = self._get_obs()

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

            "yaw_rate":
                float(obs[6]),

            "rotor_rpm":
                float(obs[7]),

            "sim_time":
                float(
                    self.fdm[
                        "simulation/sim-time-sec"
                    ]
                )
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

        # Action numpy array'e çevrilir
        action = np.asarray(
            action,
            dtype=np.float32
        )

        # -1 / +1 sınırları
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        # ==================================
        # ACTION DÖNÜŞÜMÜ
        # ==================================

        # PPO:
        #
        # collective = -1 ... +1
        #
        # JSBSim:
        #
        # collective = 0 ... 1
        #
        # Dolayısıyla dönüştürüyoruz.

        collective = (
            float(action[0]) + 1.0
        ) / 2.0

        self.fdm[
            "fcs/collective-cmd-norm"
        ] = collective

        self.fdm[
            "fcs/elevator-cmd-norm"
        ] = float(
            action[1]
        )

        self.fdm[
            "fcs/aileron-cmd-norm"
        ] = float(
            action[2]
        )

        self.fdm[
            "fcs/rudder-cmd-norm"
        ] = float(
            action[3]
        )

        # ==================================
        # JSBSIM PHYSICS
        # ==================================

        # JSBSim dt = 0.0075
        #
        # 10 physics step:
        #
        # 0.0075 * 10
        # = 0.075 saniye

        for _ in range(10):

            if not self.fdm.run():
                break

        # Yeni gözlem
        obs = self._get_obs()

        altitude = float(
            obs[0]
        )

        forward_velocity = float(
            obs[1]
        )

        vertical_speed = float(
            obs[2]
        )

        heading = float(
            obs[3]
        )

        pitch = float(
            obs[4]
        )

        roll = float(
            obs[5]
        )

        yaw_rate = float(
            obs[6]
        )

        rotor_rpm = float(
            obs[7]
        )

        # ==================================
        # HEADING ERROR
        # ==================================

        # Açının 0 / 2pi geçişinde
        # hata oluşmasını engeller.

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

        reward = 0.0

        terminated = False

        # ==================================
        # 1. FAZ
        # TAKEOFF
        # ==================================

        if self.phase == "TAKEOFF":

            altitude_error = (
                self.target_altitude
                - altitude
            )

            # --------------------------------
            # YERDE BEKLEME CEZASI
            # --------------------------------

            # AH-1S yerde yaklaşık
            # 6.3 ft AGL gösteriyor.
            #
            # Model yerde kalırsa
            # sürekli ceza alacak.

            if altitude < 10.0:

                reward -= 2.0

            # --------------------------------
            # YÜKSELME ÖDÜLÜ
            # --------------------------------

            # Pozitif vertical speed:
            # helikopter yükseliyor.

            if vertical_speed > 0:

                reward += (
                    0.2
                    * vertical_speed
                )

            # --------------------------------
            # İRTİFA ÖDÜLÜ
            # --------------------------------

            # İrtifa arttıkça
            # reward yükselir.

            reward += (
                0.02
                * altitude
            )

            # --------------------------------
            # HEDEFE YAKLAŞMA
            # --------------------------------

            reward -= (
                0.001
                * abs(
                    altitude_error
                )
            )

            # --------------------------------
            # ÇOK HIZLI YÜKSELME CEZASI
            # --------------------------------

            if vertical_speed > 40:

                reward -= 5.0

            # --------------------------------
            # HEADING
            # --------------------------------

            reward -= (
                0.5
                * abs(
                    heading_error
                )
            )

            # --------------------------------
            # STABİLİTE
            # --------------------------------

            reward -= (
                0.3
                * abs(
                    pitch
                )
            )

            reward -= (
                0.3
                * abs(
                    roll
                )
            )

            # --------------------------------
            # GERÇEK HAVALANMA
            # --------------------------------

            # 15 ft'i geçtiyse
            # artık gerçekten yerden ayrıldı.

            if altitude > 15.0:

                reward += 2.0

            # --------------------------------
            # 1000 FT HEDEFİ
            # --------------------------------

            if abs(
                altitude_error
            ) < 30:

                self.phase = "CRUISE"

                reward += 200.0

        # ==================================
        # 2. FAZ
        # CRUISE
        # ==================================

        elif self.phase == "CRUISE":

            altitude_error = (
                altitude
                - self.target_altitude
            )

            velocity_error = (
                forward_velocity
                - self.target_velocity
            )

            # Cruise'da kalma ödülü
            reward += 2.0

            # 1000 ft civarında kal
            reward -= (
                0.01
                * abs(
                    altitude_error
                )
            )

            # 35 ft/s hedef hız
            reward -= (
                0.05
                * abs(
                    velocity_error
                )
            )

            # Heading'i koru
            reward -= (
                2.0
                * abs(
                    heading_error
                )
            )

            # Stabilite
            reward -= (
                0.5
                * abs(
                    pitch
                )
            )

            reward -= (
                0.5
                * abs(
                    roll
                )
            )

            # İstenen cruise şartları
            if (
                abs(
                    altitude_error
                ) < 50

                and abs(
                    velocity_error
                ) < 10

                and abs(
                    heading_error
                ) < 0.15
            ):

                self.cruise_steps += 1

                reward += 3.0

            else:

                self.cruise_steps = 0

            # Yaklaşık 15 saniye
            # düzgün uçuş yaptıysa
            # inişe geç

            if (
                self.cruise_steps
                >= self.required_cruise_steps
            ):

                self.phase = "LANDING"

                reward += 100.0

        # ==================================
        # 3. FAZ
        # LANDING
        # ==================================

        elif self.phase == "LANDING":

            # AH-1S yerde yaklaşık
            # 6.3 ft AGL

            ground_altitude = 6.3

            altitude_error = (
                altitude
                - ground_altitude
            )

            # Yere yaklaş
            reward -= (
                0.01
                * abs(
                    altitude_error
                )
            )

            # İleri hızını azalt
            reward -= (
                0.05
                * abs(
                    forward_velocity
                )
            )

            # Heading'i koru
            reward -= (
                1.0
                * abs(
                    heading_error
                )
            )

            # Stabilite
            reward -= (
                0.5
                * abs(
                    pitch
                )
            )

            reward -= (
                0.5
                * abs(
                    roll
                )
            )

            # Çok sert alçalma
            if vertical_speed < -15:

                reward -= 20.0

            # ==================================
            # BAŞARILI İNİŞ
            # ==================================

            if (
                altitude < 7.0

                and abs(
                    vertical_speed
                ) < 5.0

                and abs(
                    forward_velocity
                ) < 5.0
            ):

                reward += 500.0

                terminated = True

        # ==================================
        # GÜVENLİK
        # ==================================

        # Çok fazla pitch

        if abs(
            pitch
        ) > 1.2:

            reward -= 100.0

            terminated = True

        # Çok fazla roll

        if abs(
            roll
        ) > 1.2:

            reward -= 100.0

            terminated = True

        # Rotor RPM çok düşerse

        if rotor_rpm < 250:

            reward -= 100.0

            terminated = True

        # ==================================
        # TIME LIMIT
        # ==================================

        truncated = (
            self.steps
            >= self.max_steps
        )

        # ==================================
        # INFO
        # ==================================

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

            "pitch":
                pitch,

            "roll":
                roll,

            "yaw_rate":
                yaw_rate,

            "rotor_rpm":
                rotor_rpm,

            "collective":
                collective
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
