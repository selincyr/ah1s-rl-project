
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
        self.script_path = os.path.join(
            self.root_dir,
            "scripts",
            "ah1s_flight_test.xml"
        )

        # action =
        # [collective, elevator, aileron, rudder]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
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
        self.fdm = jsbsim.FGFDMExec(root_dir=self.root_dir)

        if not self.fdm.load_script(self.script_path):
            raise RuntimeError("AH-1S flight script yüklenemedi")

        self.fdm.run_ic()

    def _warmup_to_hover(self):
        # Flight-test script dt = 0.0075 s
        # Yaklaşık 55 saniyeye kadar ilerletiyoruz.
        target_time = 55.0

        while self.fdm["simulation/sim-time-sec"] < target_time:
            if not self.fdm.run():
                raise RuntimeError("JSBSim warm-up sırasında durdu")

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

        # Her episode'da temiz JSBSim örneği
        self._create_fdm()

        # Rotor governor + liftoff + hover başlangıcına kadar ilerle
        self._warmup_to_hover()

        obs = self._get_obs()

        info = {
            "rotor_rpm": float(
                self.fdm["propulsion/engine/rotor-rpm"]
            ),
            "altitude": float(
                self.fdm["position/h-agl-ft"]
            )
        }

        return obs, info

    def step(self, action):
        self.steps += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        # Pilot command'leri
        self.fdm["fcs/collective-cmd-norm"] = float(action[0])
        self.fdm["fcs/elevator-cmd-norm"] = float(action[1])
        self.fdm["fcs/aileron-cmd-norm"] = float(action[2])
        self.fdm["fcs/rudder-cmd-norm"] = float(action[3])

        # Tek RL step'te birkaç fizik step'i ilerlet
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

        altitude_error = altitude - self.target_altitude

        reward = (
            1.0
            - 0.02 * abs(altitude_error)
            - 1.0 * abs(pitch)
            - 1.0 * abs(roll)
            - 0.05 * abs(yaw_rate)
            - 0.01 * abs(vertical_speed)
            - 0.01 * abs(rotor_rpm - 324.0)
        )

        terminated = False

        # Güvenlik / fiziksel limitler
        if altitude < 7:
            terminated = True
            reward -= 100.0
        
        if abs(altitude_error) < 5:
            reward +=5.0

        if abs(pitch) <0.05 and abs(roll) < 0.05:
            reward += 2.0

        if abs(pitch) > 1.2:
            terminated = True
            reward -= 50.0

        if abs(roll) > 1.2:
            terminated = True
            reward -= 50.0

        if rotor_rpm < 250:
            terminated = True
            reward -= 50.0

        truncated = self.steps >= self.max_steps

        info = {
            "altitude": altitude,
            "rotor_rpm": rotor_rpm,
            "pitch": pitch,
            "roll": roll
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self.fdm = None
