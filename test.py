from stable_baselines3 import PPO
from helicopter_env import HelicopterEnv

env = HelicopterEnv()

model = PPO.load("ppo_ah1s")

obs, info = env.reset()

for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

print("Model testi tamamlandı")
