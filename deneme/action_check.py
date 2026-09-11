import numpy as np
from helicopter_env_2 import HelicopterEnv

def run_test(name,collective_aciton):
  env = HelicopterEnv()
  obs, info = env.reset()

  max_altitude = info["altitude"]

  for step in range(300):
    action = np.array([collective_aciton,0.0,0.0,0.0],dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    max_altitude = max(max_altitude,info["altitude"])

    if terminated or truncated:
      break
  print(name,"MaxAlt:",round(max_altitude, 2),"finalAlt:",round(info["altitude"],2),"Collective:",round(info["collective"],3))
  env.close()

def main():
  print("TEST 1")
  run_test("Action 0.0",0.0)
  print("TEST 2")
  run_test("Action 0.5",0.5)
  print("TEST 3")
  run_test("Action 1.0",1.0)
  

if __name__ == "__main__":
    main()
