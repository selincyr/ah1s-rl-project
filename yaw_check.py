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

def run_test(rudder_value):
  #print()

 # print("===============================")
 # print(name)
 # print("Rudder:", rudder_value)
 # print("===============================")
  fdm = create_fdm()

#hover civarı temel kontroller
  
  fdm ["fcs/elevator-cmd-norm"] = -0.223
  fdm ["fcs/aileron-cmd-norm"] = 0.240
  fdm ["fcs/rudder-cmd-norm"] = 0.386
  fdm ["fcs/collective-cmd-norm"] = 0.650
  #start_heading = fdm["attitude/heading-true-rad"]
  for _ in range(2000):
    fdm.run()
  start_altitude = fdm["position/h-agl-ft"]
  start_heading = fdm["attitude/heading-true-rad"]

  #RUDDER TESTİ
  fdm["fcs/rudder-cmd-norm"] = rudder_value
  for _ in range(1000):
    fdm.run()
  final_altitude = fdm["position/h-agl-ft"]
  final_yaw_rate = fdm["velocities/r-rad_sec"]
  return(start_altitude, final_altitude,start_heading,final_heading,final_yaw_rate)
                       

  
   
def main():
  
    base = run_test(0.386)
    low =run_test(0.316)
    high= run_test(0.456)
    print()
    print("=============YAW CHECK OZET===============")
    print("BASE:","Alt",round(base[0],1),"->",round(base[1],1),"Heading",round(base[2],3),"->",round(base[3],3),"YawRate",round(base[4],3))
    print("LOW:","Alt",round(low[0],1),"->",round(low[1],1),"Heading",round(low[2],3),"->",round(low[3],3),"YawRate",round(low[4],3))
    print("LOW:","Alt",round(high[0],1),"->",round(high[1],1),"Heading",round(high[2],3),"->",round(high[3],3),"YawRate",round(high[4],3))
    


if __name__ == "__main__":
  main()



    
