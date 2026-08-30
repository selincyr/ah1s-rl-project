import os
import numpy as np
import jsbsim
import gymnasium as gym
from gymnasium import spaces


class HelicopterEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # JSBSim'in kurulu olduğu klasör
        self.root_dir = os.path.dirname(jsbsim.__file__)

        # Bu dosyanın bulunduğu proje klasörü
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Bizim temiz RL başlangıç scriptimiz
        self.script_path = os.path.join(
            project_dir,
            "scripts",
            "ah1s_rl_start.xml"
        )

        # ACTION
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

        # OBSERVATION
        #
        # 0 altitude
        # 1 forward velocity
        # 2 vertical speed
        # 3 heading
        # 4 pitch
        # 5 roll
        # 6 yaw rate
        # 7 rotor rpm
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

        # Şimdilik eski altitude hedefi
        # Bir sonraki aşamada Task 1'e göre değiştireceğiz.
        self.target_altitude = 75.0

        self.max_steps = 2000
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

        # Rotor yaklaşık çalışma devrine ulaşana kadar
        # simülasyonu kendi içinde ilerletiyoruz.
        #
        # Bu kısım RL timestep'lerine dahil değil.
        while (
            self.fdm["propulsion/engine/rotor-rpm"]
            < 320.0
        ):

            if not self.fdm.run():
                raise RuntimeError(
                    "Rotor warm-up sırasında JSBSim durdu"
                )

    def _get_obs(self):

        return np.array(
            [
                # 0 - Yerden yükseklik
                self.fdm["position/h-agl-ft"],

                # 1 - Gövdenin ileri yönündeki hız
                self.fdm["velocities/u-aero-fps"],

                # 2 - Dikey hız
                self.fdm["velocities/h-dot-fps"],

                # 3 - Heading
                self.fdm["attitude/heading-true-rad"],

                # 4 - Pitch
                self.fdm["attitude/pitch-rad"],

                # 5 - Roll
                self.fdm["attitude/roll-rad"],

                # 6 - Yaw rate
                self.fdm["velocities/r-rad_sec"],

                # 7 - Rotor RPM
                self.fdm[
                    "propulsion/engine/rotor-rpm"
                ]
            ],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.steps = 0

        # Her episode'da temiz bir JSBSim örneği
        self._create_fdm()

        # Helikopter yerde.
        # Sadece rotor çalışma devrine getiriliyor.
        self._warmup_rotor()

        obs = self._get_obs()

        info = {
            "altitude": float(obs[0]),
            "forward_velocity": float(obs[1]),
            "vertical_speed": float(obs[2]),
            "heading": float(obs[3]),
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

        # PPO'nun verdiği kontrol komutları

        # Collective:
        # rotor palalarının toplam açısını değiştirir.
        self.fdm["fcs/collective-cmd-norm"] = float(
            action[0]
        )

        # Elevator / longitudinal cyclic:
        # ileri-geri hareket üzerinde etkili
        self.fdm["fcs/elevator-cmd-norm"] = float(
            action[1]
        )

        # Aileron / lateral cyclic:
        # sağ-sol yatış üzerinde etkili
        self.fdm["fcs/aileron-cmd-norm"] = float(
            action[2]
        )

        # Rudder / pedal:
        # yaw üzerinde etkili
        self.fdm["fcs/rudder-cmd-norm"] = float(
            action[3]
        )

        # JSBSim dt = 0.0075 saniye.
        # Her RL step'inde 10 fizik adımı ilerliyoruz.
        physics_steps = 10

        for _ in range(physics_steps):

            if not self.fdm.run():
                break

        obs = self._get_obs()

        # Observation değerlerini açıyoruz.
        altitude = float(obs[0])
        forward_velocity = float(obs[1])
        vertical_speed = float(obs[2])
        heading = float(obs[3])
        pitch = float(obs[4])
        roll = float(obs[5])
        yaw_rate = float(obs[6])
        rotor_rpm = float(obs[7])

        # ----------------------------------
        # ŞİMDİLİK ESKİ REWARD
        # ----------------------------------
        #
        # Henüz Task 1 reward'una geçmedik.
        # Önce yeni observation'ların
        # doğru çalıştığını test edeceğiz.

        altitude_error = (
            altitude - self.target_altitude
        )

        reward = (
            1.0
            - 0.02 * abs(altitude_error)
            - 1.0 * abs(pitch)
            - 1.0 * abs(roll)
            - 0.05 * abs(yaw_rate)
            - 0.01 * abs(vertical_speed)
            - 0.01 * abs(rotor_rpm - 324.0)
        )

        if abs(altitude_error) < 5:
            reward += 5.0

        if (
            abs(pitch) < 0.05
            and abs(roll) < 0.05
        ):
            reward += 2.0

        terminated = False

        # altitude < 7 kontrolü yok.
        #
        # Çünkü helikopter yerdeyken
        # AGL yaklaşık 6.29 ft gösteriyor.

        if abs(pitch) > 1.2:
            terminated = True
            reward -= 50.0

        if abs(roll) > 1.2:
            terminated = True
            reward -= 50.0

        if rotor_rpm < 250:
            terminated = True
            reward -= 50.0

        truncated = (
            self.steps >= self.max_steps
        )

        info = {
            "altitude": altitude,
            "forward_velocity": forward_velocity,
            "vertical_speed": vertical_speed,
            "heading": heading,
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
