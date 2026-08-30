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

        # Bizim GitHub reposunun bulunduğu klasör
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Artık JSBSim'in otomatik flight-test scriptini değil,
        # bizim temiz RL başlangıç scriptimizi kullanıyoruz.
        self.script_path = os.path.join(
            project_dir,
            "scripts",
            "ah1s_rl_start.xml"
        )

        # action =
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

        # observation =
        # [altitude, pitch, roll, yaw_rate, vertical_speed, rotor_rpm]
        self.observation_space = spaces.Box(
            low=np.array(
                [-1000, -np.pi/2, -np.pi, -20, -500, 0],
                dtype=np.float32
            ),
            high=np.array(
                [10000, np.pi/2, np.pi, 20, 500, 700],
                dtype=np.float32
            ),
            dtype=np.float32
        )

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

        # Rotor motor çalışır duruma gelene kadar simülasyonu ilerlet.
        # Testimizde yaklaşık 2.3 saniyede 320 RPM'e ulaştığını gördük.
        while (
            self.fdm["propulsion/engine/rotor-rpm"]
            < 320.0
        ):

            if not self.fdm.run():
                raise RuntimeError(
                    "Rotor warm-up sırasında JSBSim durdu"
                )

    def _get_obs(self):

        return np.array([
            self.fdm["position/h-agl-ft"],
            self.fdm["attitude/pitch-rad"],
            self.fdm["attitude/roll-rad"],
            self.fdm["velocities/r-rad_sec"],
            self.fdm["velocities/h-dot-fps"],
            self.fdm["propulsion/engine/rotor-rpm"]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.steps = 0

        # Her episode'da temiz JSBSim instance
        self._create_fdm()

        # Artık 55 saniyelik hover beklemiyoruz.
        # Sadece rotor çalışır hale gelene kadar bekliyoruz.
        self._warmup_rotor()

        obs = self._get_obs()

        info = {
            "rotor_rpm": float(
                self.fdm["propulsion/engine/rotor-rpm"]
            ),
            "altitude": float(
                self.fdm["position/h-agl-ft"]
            ),
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

        # PPO'nun ürettiği pilot komutları
        self.fdm["fcs/collective-cmd-norm"] = float(
            action[0]
        )

        self.fdm["fcs/elevator-cmd-norm"] = float(
            action[1]
        )

        self.fdm["fcs/aileron-cmd-norm"] = float(
            action[2]
        )

        self.fdm["fcs/rudder-cmd-norm"] = float(
            action[3]
        )

        # Bir RL step içerisinde
        # JSBSim fiziğini birkaç kez ilerletiyoruz.
        physics_steps = 10

        for _ in range(physics_steps):

            if not self.fdm.run():
                break

        obs = self._get_obs()

        altitude = float(obs[0])
        pitch = float(obs[1])
        roll = float(obs[2])
        yaw_rate = float(obs[3])
        vertical_speed = float(obs[4])
        rotor_rpm = float(obs[5])

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

        # ŞİMDİLİK altitude < 7 termination YOK.
        # Çünkü helikopter yerde ~6.29 ft gösteriyor.

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
            "rotor_rpm": rotor_rpm,
            "pitch": pitch,
            "roll": roll,
            "vertical_speed": vertical_speed
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
