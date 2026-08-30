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

        # Action:
        # [collective, elevator, aileron, rudder]
        self.action_space = spaces.Box(
            low=np.array(
                [0.0, -1.0, -1.0, -1.0],
                dtype=np.float32
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # Observation:
        # 0 altitude
        # 1 forward velocity
        # 2 vertical speed
        # 3 heading
        # 4 pitch
        # 5 roll
        # 6 yaw rate
        # 7 rotor RPM
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

        # 1000 ft AGL
        self.target_altitude = 1000.0

        # Yaklaşık 21 knot
        self.target_velocity = 35.0

        # Başlangıç heading'i reset sırasında alınacak
        self.target_heading = None

        # Görev başlangıç fazı
        self.phase = "TAKEOFF"

        # Cruise sırasında kaç step düzgün uçtuğunu sayacağız
        self.cruise_steps = 0

        # 200 RL step
        # Her RL step yaklaşık 0.075 saniye
        # yaklaşık 15 saniyelik cruise
        self.required_cruise_steps = 200

        self.max_steps = 5000
        self.steps = 0

        self.fdm = None

    def _create_fdm(self):

        self.fdm = jsbsim.FGFDMExec(
            root_dir=self.root_dir
        )

        if not self.fdm.load_script(self.script_path):
            raise RuntimeError(
                "AH-1S RL başlangıç scripti yüklenemedi"
            )

        self.fdm.run_ic()

    def _warmup_rotor(self):

        # PPO kontrolü almadan önce
        # rotor yaklaşık çalışma RPM'ine gelsin
        while (
            self.fdm["propulsion/engine/rotor-rpm"] < 320.0
        ):

            if not self.fdm.run():
                raise RuntimeError(
                    "Rotor warm-up sırasında JSBSim durdu"
                )

    def _get_obs(self):

        return np.array(
            [
                self.fdm["position/h-agl-ft"],
                self.fdm["velocities/u-aero-fps"],
                self.fdm["velocities/h-dot-fps"],
                self.fdm["attitude/heading-true-rad"],
                self.fdm["attitude/pitch-rad"],
                self.fdm["attitude/roll-rad"],
                self.fdm["velocities/r-rad_sec"],
                self.fdm["propulsion/engine/rotor-rpm"]
            ],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.steps = 0
        self.cruise_steps = 0
        self.phase = "TAKEOFF"

        self._create_fdm()
        self._warmup_rotor()

        # Helikopter başlangıçta hangi yöne bakıyorsa
        # o yönü hedef heading kabul ediyoruz
        self.target_heading = float(
            self.fdm["attitude/heading-true-rad"]
        )

        obs = self._get_obs()

        info = {
            "phase": self.phase,
            "altitude": float(obs[0]),
            "forward_velocity": float(obs[1]),
            "vertical_speed": float(obs[2]),
            "heading": float(obs[3]),
            "target_heading": self.target_heading,
            "pitch": float(obs[4]),
            "roll": float(obs[5]),
            "yaw_rate": float(obs[6]),
            "rotor_rpm": float(obs[7]),
            "sim_time": float(
                self.fdm["simulation/sim-time-sec"]
            )
        }

        return obs, info

    def step(self, action):

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

        # PPO'nun ürettiği kontrol komutları
        self.fdm["fcs/collective-cmd-norm"] = float(action[0])
        self.fdm["fcs/elevator-cmd-norm"] = float(action[1])
        self.fdm["fcs/aileron-cmd-norm"] = float(action[2])
        self.fdm["fcs/rudder-cmd-norm"] = float(action[3])

        # 1 RL step içinde 10 physics step
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

        # Heading farkını -pi ile +pi arasında hesapla
        heading_error = np.arctan2(
            np.sin(
                heading - self.target_heading
            ),
            np.cos(
                heading - self.target_heading
            )
        )

        reward = 0.0
        terminated = False

        # ==================================
        # 1. FAZ - TAKEOFF
        # ==================================

        if self.phase == "TAKEOFF":

            altitude_error = (
                self.target_altitude - altitude
            )

            # Yükseldikçe ödül
            reward += 0.01 * altitude

            # 1000 ft hedefinden uzaksa ceza
            reward -= 0.002 * abs(
                altitude_error
            )

            # Heading'i koru
            reward -= 1.0 * abs(
                heading_error
            )

            # Fazla pitch ve roll istemiyoruz
            reward -= 0.5 * abs(pitch)
            reward -= 0.5 * abs(roll)

            # 1000 ft civarına ulaştıysa
            # CRUISE fazına geç
            if abs(altitude_error) < 30:

                self.phase = "CRUISE"

                reward += 100.0

        # ==================================
        # 2. FAZ - CRUISE
        # ==================================

        elif self.phase == "CRUISE":

            altitude_error = (
                altitude - self.target_altitude
            )

            velocity_error = (
                forward_velocity
                - self.target_velocity
            )

            reward += 2.0

            # 1000 ft civarında kal
            reward -= 0.01 * abs(
                altitude_error
            )

            # Yaklaşık 35 ft/s ileri git
            reward -= 0.05 * abs(
                velocity_error
            )

            # Heading'i koru
            reward -= 2.0 * abs(
                heading_error
            )

            # Dengeli kal
            reward -= 0.5 * abs(pitch)
            reward -= 0.5 * abs(roll)

            # İstenen uçuş şartlarını sağlıyorsa
            if (
                abs(altitude_error) < 50
                and abs(velocity_error) < 10
                and abs(heading_error) < 0.15
            ):

                self.cruise_steps += 1

                reward += 3.0

            else:

                self.cruise_steps = 0

            # Yaklaşık 15 saniye düzgün cruise
            # yaptıysa LANDING'e geç
            if (
                self.cruise_steps
                >= self.required_cruise_steps
            ):

                self.phase = "LANDING"

                reward += 100.0

        # ==================================
        # 3. FAZ - LANDING
        # ==================================

        elif self.phase == "LANDING":

            # AH-1S yerde yaklaşık 6.3 ft AGL
            ground_altitude = 6.3

            altitude_error = (
                altitude - ground_altitude
            )

            # Yere yaklaşmasını teşvik et
            reward -= 0.01 * abs(
                altitude_error
            )

            # İleri hızı azalt
            reward -= 0.05 * abs(
                forward_velocity
            )

            # Heading'i koru
            reward -= 1.0 * abs(
                heading_error
            )

            # Stabilite
            reward -= 0.5 * abs(pitch)
            reward -= 0.5 * abs(roll)

            # Çok sert alçalma cezası
            if vertical_speed < -15:

                reward -= 20.0

            # Başarılı iniş
            if (
                altitude < 7.0
                and abs(vertical_speed) < 5.0
                and abs(forward_velocity) < 5.0
            ):

                reward += 500.0

                terminated = True

        # ==================================
        # GÜVENLİK KONTROLLERİ
        # ==================================

        if abs(pitch) > 1.2:

            reward -= 100.0

            terminated = True

        if abs(roll) > 1.2:

            reward -= 100.0

            terminated = True

        if rotor_rpm < 250:

            reward -= 100.0

            terminated = True

        truncated = (
            self.steps >= self.max_steps
        )

        info = {
            "phase": self.phase,
            "altitude": altitude,
            "forward_velocity": forward_velocity,
            "vertical_speed": vertical_speed,
            "heading": heading,
            "target_heading": self.target_heading,
            "pitch": pitch,
            "roll": roll,
            "yaw_rate": yaw_rate,
            "rotor_rpm": rotor_rpm
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    def close(self):
        self.fdm = None
