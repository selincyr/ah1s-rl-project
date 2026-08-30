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

        # action[0] = collective
        # action[1] = elevator
        # action[2] = aileron
        # action[3] = rudder
        #
        # PPO bütün action'ları -1 / +1 arasında üretir.

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # ==========================================
        # OBSERVATION SPACE
        # ==========================================

        # 0 = altitude AGL (ft)
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

        # ==========================================
        # TASK 1 - CURRICULUM 1
        # ==========================================

        # Şimdilik tek hedef:
        #
        # YERDEN KALK
        # ->
        # 1000 FT AGL'E ÇIK
        # ->
        # STABİL KAL

        self.target_altitude = 1000.0

        # İstenen yaklaşık yükselme hızı
        self.target_climb_rate = 15.0

        self.target_heading = None

        self.phase = "TAKEOFF"

        # Hedef irtifada kaç step stabil kaldı?
        self.target_hold_steps = 0

        # 50 step yaklaşık 3.75 saniye
        self.required_hold_steps = 50

        self.max_steps = 5000

        self.steps = 0

        # Bir önceki irtifayı tutacağız.
        # Böylece gerçekten yükselmeyi ödüllendireceğiz.
        self.previous_altitude = None

        self.fdm = None

    # ==========================================
    # JSBSIM OLUŞTUR
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

        # Başlangıç yönünü hedef heading kabul ediyoruz.
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

            "yaw_rate":
                float(obs[6]),

            "rotor_rpm":
                float(obs[7]),

            "collective":
                0.0,

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
            self.action_space.low,
            self.action_space.high
        )

        # ==================================
        # ACTION DÖNÜŞÜMÜ
        # ==================================

        # PPO:
        # -1 ... +1
        #
        # JSBSim collective:
        # 0 ... 1

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
        # PHYSICS
        # ==================================

        # JSBSim dt = 0.0075
        # 10 physics step = 0.075 saniye

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

        yaw_rate = float(obs[6])

        rotor_rpm = float(obs[7])

        # ==================================
        # HEADING ERROR
        # ==================================

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

        # ==================================
        # ALTITUDE PROGRESS
        # ==================================

        altitude_progress = (
            altitude
            - self.previous_altitude
        )

        self.previous_altitude = altitude

        reward = 0.0

        terminated = False

        success = False

        # ==================================
        # TAKEOFF REWARD
        # ==================================

        altitude_error = (
            self.target_altitude
            - altitude
        )

        # ----------------------------------
        # 1) GERÇEKTEN YÜKSELMESİNİ ÖDÜLLENDİR
        # ----------------------------------

        # Örneğin:
        #
        # önceki altitude = 100
        # yeni altitude     = 101
        #
        # progress = +1
        #
        # reward = +5

        reward += (
            5.0
            * altitude_progress
        )

        # Alçalırsa progress negatif olur
        # ve otomatik olarak ceza alır.

        # ----------------------------------
        # 2) YERDE BEKLEME CEZASI
        # ----------------------------------

        if altitude < 10.0:

            reward -= 1.0

        # ----------------------------------
        # 3) HEDEF İRTİFAYA YAKINLIK
        # ----------------------------------

        # 1000 ft'e yaklaştıkça ceza azalır.

        reward -= (
            0.001
            * abs(
                altitude_error
            )
        )

        # ----------------------------------
        # 4) YÜKSELME HIZI
        # ----------------------------------

        # Havalandıktan sonra yaklaşık
        # 15 ft/s climb rate istiyoruz.

        if (
            altitude > 15.0
            and altitude < 950.0
        ):

            climb_rate_error = (
                vertical_speed
                - self.target_climb_rate
            )

            reward -= (
                0.02
                * abs(
                    climb_rate_error
                )
            )

        # ----------------------------------
        # 5) ÇOK HIZLI YÜKSELME
        # ----------------------------------

        if vertical_speed > 40.0:

            reward -= 10.0

        # Çok hızlı aşağı düşerse
        if vertical_speed < -20.0:

            reward -= 10.0

        # ----------------------------------
        # 6) İLERİ / GERİ KAÇMASIN
        # ----------------------------------

        # Bu aşamada ileri uçmayı henüz
        # öğretmiyoruz.
        #
        # Önce dikey ve stabil kalkış.

        reward -= (
            0.01
            * abs(
                forward_velocity
            )
        )

        # ----------------------------------
        # 7) HEADING
        # ----------------------------------

        reward -= (
            1.0
            * abs(
                heading_error
            )
        )

        # ----------------------------------
        # 8) PITCH / ROLL STABİLİTESİ
        # ----------------------------------

        reward -= (
            2.0
            * abs(
                pitch
            )
        )

        reward -= (
            2.0
            * abs(
                roll
            )
        )

        # ==================================
        # 1000 FT BÖLGESİ
        # ==================================

        if abs(
            altitude_error
        ) < 30.0:

            # Hedefe ulaşma ödülü
            reward += 5.0

            # Burada artık dikey hızın
            # sıfıra yakın olmasını istiyoruz.

            reward -= (
                0.1
                * abs(
                    vertical_speed
                )
            )

            # Stabil durumda mı?
            if (
                abs(vertical_speed) < 8.0
                and abs(pitch) < 0.25
                and abs(roll) < 0.25
            ):

                self.target_hold_steps += 1

                reward += 5.0

            else:

                self.target_hold_steps = 0

        else:

            self.target_hold_steps = 0

        # ==================================
        # BAŞARI
        # ==================================

        if (
            self.target_hold_steps
            >= self.required_hold_steps
        ):

            reward += 500.0

            success = True

            terminated = True

        # ==================================
        # GÜVENLİK
        # ==================================

        # Yaklaşık ±69 derece
        if abs(
            pitch
        ) > 1.2:

            reward -= 100.0

            terminated = True

        if abs(
            roll
        ) > 1.2:

            reward -= 100.0

            terminated = True

        if rotor_rpm < 250.0:

            reward -= 100.0

            terminated = True

        # Aşırı irtifa
        if altitude > 1500.0:

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

            "heading_error":
                float(
                    heading_error
                ),

            "pitch":
                pitch,

            "roll":
                roll,

            "yaw_rate":
                yaw_rate,

            "rotor_rpm":
                rotor_rpm,

            "collective":
                collective,

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
