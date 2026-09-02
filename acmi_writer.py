from datetime import datetime, timezone

class ACMIWriter:
  def __init__(self,filename,aircraft_name="AH-1S",callsign="RL-AH1S"):
    self.filename=filename
    self.object_id =1
    self.file = open(filename,"w",encoding="utf-8")

    ###ACMI HEADER##
    self.file.write("FileType=text/acmi/tacview\n")
    self.file.write("FileVersion=2.2\n")
    reference_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    self.file.write(f"0,ReferenceTime={reference_time}\n")
    
    ########HELİCOPTER OBJECT########
    self.file.writef"{self.object_id},"f"Type=Air+Rotorcraft,"f"Name={aircraft_name},f"CallSign={callsign}\n")
  def write(self,
            time,
            longitude,
            latitude,
            altitude_ft,
            roll_deg,
            pitch_deg,
            yaw_deg):
    #JSBSim altitude is in feet
    #ACMI POSİTİON ALTİTUDE SHOULD BE İN METERS.
    altitude_m=altitude_ft*0.3048
    ##TIME FRAME
    self.file.write(f"#{time:.3f}\n")

    ####OBJECT POSITION
    #T=
    #longitude |
    #latitude|
    #altitude|
    #roll|
    # pitch|
    # yaw

    self.file.write(f"{self.object_id},"f"T="f"{longitude:.7f}|"f"{latitude:.7f}|"f"{altitude_m:.2f}|"f"{roll_deg:.3f}|"f"{pitch_deg:.3f}|"f"{yaw_deg:.3f}\n")


  def close(self):
    if not self.file.closed:
      self.file.flush()
      self.file.close()

  def  __enter__(self):
    return self

  def __exit__(self,exc_type,exc_value,traceback):
    self.close()
    
            
