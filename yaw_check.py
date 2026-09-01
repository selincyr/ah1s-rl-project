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

def run_test(name,rudder_value):
  print()

  print("===============================")
  print(name)
  print("Rudder:", rudder_value)
  print("===============================")
  fdm = create_fdm()

#hover civarı temel kontroller
  fdm ["fcs/collective-cmd-norm"] = 0.560
  fdm ["fcs/elevator-cmd-norm"] = -0.223
  fdm ["fcs/aileron-cmd-norm"] = 0.240
  fdm ["fcs/rudder-cmd-norm"] = rudder_value
  start_heading = fdm["attitude/heading-true-rad"]

  for step in range(500):
    for _ in range(10):
      fdm.run()
    if step % 50 == 0:
      heading = fdm["attitude/heading-true-rad"]
      yaw_rate = fdm["velocities/r-rad_sec"]
      altitude = fdm["position/h-agl-ft"]
      print(
        f"Step {step:3d} |" f"Heading {heading:7.3f} |" f"Yaw rate {yaw_rate:7.3f} |" f"Altitude {altitude:7.2f}")
    final_heading = fdm["attitude/heading-true-rad"]
    print()
    print("Start heading:",round(start_heading,3))
    print("Final heading:",round(final_heading,3))
def main(): 
  #base trim rudder
    run_test("BASE RUDDER",0.386)
  #RUDDER biraz azaltilmiş
    run_test("LOW RUDDER",0.316)
  #RUDER BİRAZ ARTİRİLMİŞ
    run_test("HIGH RUDDER",0.456)


if __name__ == "__main__":
  main()



    
