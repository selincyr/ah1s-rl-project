import os
import jsbsim



def create_fdm():
  root_dir = os.path.dirname(jsbsim.__file__)
  project_dir = os.path.dirname(os.path.abspath(__file__))
  script_path = os.path.join(project_dir,"scripts","ah1s_rl_start.xml")
  fdm = jsbsim.FGFDMExec(root_dir=root_dir)
  if not fdm.load_script(script_path):
    raise RuntimeError("Script yuklenmedi")
  fdm.run_ic()
  while(
    fdm["propulsion/engine/rotor-rpm"] < 320.0):
    fdm.run()
  return fdm

def run_test(collective_value):
  fdm = create_fdm()
  for step in range(500):
  #temel trim kontrolleri
      fdm["fcs/elevator-cmd-norm"] = -0.223
      fdm["fcs/aileron-cmd-norm"] = 0.240
      fdm["fsc/rudder-cmd-norm"] = 0.386
  for _ in range(10):
      fdm.run()
altitude = fdm["position/h-agl-ft"]
heading = fdm["attitude/heading-true-rad"]
yaw_rate = fdm["velocities/r-rad_sec"]
collective_seen = fdm["fcs/collective-cmd-norm"]
return(altitude,heading,yaw_rate,collevtive_seen)
 # fdm["fcs/collective-cmd-run"] = collective_value
#  print(
   # ["commanded collective:",collective_value,"jsbsim sees:",fdm["fcs/collective-cmd-norm"]
 # )
 # start_heading = fdm["attitude/heading-true-rad"]
 # for _ in range(2000):
  #    fdm.run()

 # print("After simulation:",fdm["fcs/collective-cmd-norm"])
 # final_heading = fdm["attitude/heading-true-rad"]
 # final_yaw_rate = fdm["velocities/r-rad_sec"]
 # final_altitude = fdm["position/h-agl-ft"]
  #return(start_heading,final_heading,final_yaw_rate,final_altitude)

def main():
    #run_test(0.700)
    c1 = run_test(0.560)
    c2 = run_test(0.620)
    c3 = run_test(0.660)
    c4 = run_test(0.700)
    print()
    print("========COLLECTIVE-YAW CHECK=============")
    print("0.560:",c1)
    print("0.620:",c2)
    print("0.660:",c3)
    print("0.700:",c4)

if __name__ == "__main__":
    main()
