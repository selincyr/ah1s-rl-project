from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from helicopter_env_stage1_distill import HelicopterEnvStage1Distill
from helicopter_env_stage2_refine_mapped import HelicopterEnvStage2RefineMapped

STAGE1_MODEL_PATH="models_stage1_early_distilled/AH1S_STAGE1_EARLY_DISTILLED.zip"
STAGE2_MODEL_PATH="models_stage2_distilled/AH1S_STAGE2_DISTILLED_SUCCESS.zip"
OUT_DIR=Path("results_stage3_endpoint"); OUT_DIR.mkdir(parents=True,exist_ok=True)
OUT_JSON=OUT_DIR/"stage2_endpoint_state.json"

EARTH_RADIUS_FT=20_902_231.0
HANDOFF_STABLE_TIME=5.0
TARGET_DISTANCE=300.0
AILERON_SCALE=0.026
RUDDER_SCALE=0.040

def ff(fdm,key,default=float("nan")):
    try:return float(fdm[key])
    except Exception:return float(default)

def inf(info,key,default=float("nan")):
    try:return float(info.get(key,default))
    except Exception:return float(default)

def get_fdm(env):
    if getattr(env,"fdm",None) is not None:return env.fdm
    base=getattr(env,"base_env",None)
    if base is not None and getattr(base,"fdm",None) is not None:return base.fdm
    raise RuntimeError("Active FDM not found")

def lat_deg(fdm):
    for k in ["position/lat-gc-deg","position/lat-geod-deg"]:
        x=ff(fdm,k)
        if np.isfinite(x):return x
    return float("nan")

def lon_deg(fdm): return ff(fdm,"position/long-gc-deg")

def heading(fdm):
    for k in ["attitude/heading-true-rad","attitude/psi-rad"]:
        x=ff(fdm,k)
        if np.isfinite(x):return x
    return float("nan")

def wrap(x): return math.atan2(math.sin(x),math.cos(x))

def local_ne(lat,lon,lat0,lon0):
    dlat=math.radians(lat-lat0); dlon=math.radians(lon-lon0)
    return (EARTH_RADIUS_FT*dlat,
            EARTH_RADIUS_FT*math.cos(math.radians(lat0))*dlon)

def mission_axes(n,e,h):
    c=math.cos(h); s=math.sin(h)
    return n*c+e*s, -n*s+e*c

def raw_snapshot(fdm):
    keys=[
        "simulation/sim-time-sec","position/h-agl-ft",
        "position/lat-gc-deg","position/lat-geod-deg","position/long-gc-deg",
        "attitude/heading-true-rad","attitude/psi-rad","attitude/pitch-rad","attitude/roll-rad",
        "velocities/u-aero-fps","velocities/v-aero-fps","velocities/w-aero-fps",
        "velocities/h-dot-fps","velocities/p-rad_sec","velocities/q-rad_sec","velocities/r-rad_sec",
        "propulsion/engine/rotor-rpm","fcs/collective-cmd-norm","fcs/elevator-cmd-norm",
        "fcs/aileron-cmd-norm","fcs/rudder-cmd-norm","fcs/lateral-ctrl-rad","fcs/pedal-ctrl-rad"
    ]
    out={}
    for k in keys:
        x=ff(fdm,k)
        if np.isfinite(x): out[k]=x
    return out

stage1=PPO.load(STAGE1_MODEL_PATH)
stage2=PPO.load(STAGE2_MODEL_PATH)

print("="*110)
print("STAGE 3 PREPARATION — RECORD FINAL STAGE-2 ENDPOINT STATE")
print("="*110)
print("NO TRAINING. Stage 1 locked. Stage 2 locked. Runtime teachers OFF.\n")

env1=HelicopterEnvStage1Distill(teacher_model_path=None,training_mode=False)
obs,info=env1.reset()
fdm=get_fdm(env1); fdm_id=id(fdm)
origin_lat,origin_lon=lat_deg(fdm),lon_deg(fdm)
mission_heading=heading(fdm)
dt=float(getattr(env1,"dt",0.075) or 0.075)
stable_time=0.0; handoff=None; last_a1=None

for step in range(int(120/dt)):
    a,_=stage1.predict(obs,deterministic=True)
    a=np.asarray(a,dtype=np.float32).reshape(-1)
    obs,_,term,trunc,info=env1.step(a); last_a1=a.copy()
    alt=inf(info,"altitude"); vs=inf(info,"vertical_speed")
    vn=inf(info,"vn",0.0); ve=inf(info,"ve",0.0)
    hs=float(np.hypot(vn,ve)); drift=inf(info,"drift",999.0)
    stable=(295<=alt<=305 and abs(vs)<=0.5 and hs<=1 and drift<=3)
    stable_time=stable_time+dt if stable else 0.0
    if stable_time>=HANDOFF_STABLE_TIME:
        handoff={"time_s":(step+1)*dt,"altitude_ft":alt,"vertical_speed_fps":vs,
                 "horizontal_speed_fps":hs,"drift_ft":drift,
                 "last_stage1_action":[float(x) for x in last_a1]}
        break
    if term and not bool(info.get("success",False)): raise RuntimeError("Stage 1 failed")
    if trunc: raise RuntimeError("Stage 1 truncated")
if handoff is None: raise RuntimeError("No stable Stage-1 handoff")

env2=HelicopterEnvStage2RefineMapped(aileron_scale=AILERON_SCALE,rudder_scale=RUDDER_SCALE)
env2.reset(); env2.fdm=fdm
if hasattr(env2,"forward_distance"): env2.forward_distance=0.0
if hasattr(env2,"target_heading"): env2.target_heading=float(mission_heading)
for attr in ["steps","target_hold_steps","hold_steps","success_hold_steps"]:
    if hasattr(env2,attr): setattr(env2,attr,0)
if id(get_fdm(env2))!=fdm_id: raise RuntimeError("FDM continuity failed")
obs2=np.asarray(env2._get_obs(),dtype=np.float32)
dt2=float(getattr(env2,"dt",0.075) or 0.075)
endpoint=None

for step in range(int(55/dt2)):
    a,_=stage2.predict(obs2,deterministic=True)
    a=np.asarray(a,dtype=np.float32).reshape(-1)
    obs2,_,term,trunc,info2=env2.step(a)
    obs2=np.asarray(obs2,dtype=np.float32)
    dist=inf(info2,"forward_distance",getattr(env2,"forward_distance",0.0))
    if dist>=TARGET_DISTANCE:
        n,e=local_ne(lat_deg(fdm),lon_deg(fdm),origin_lat,origin_lon)
        ground_fwd,cross=mission_axes(n,e,mission_heading)
        h_err=wrap(heading(fdm)-mission_heading)
        endpoint={
            "stage2_time_s":(step+1)*dt2,
            "forward_distance_env_ft":dist,
            "ground_forward_from_takeoff_ft":float(ground_fwd),
            "cross_track_from_takeoff_ft":float(cross),
            "altitude_ft":inf(info2,"altitude",ff(fdm,"position/h-agl-ft")),
            "forward_velocity_fps":inf(info2,"forward_velocity",ff(fdm,"velocities/u-aero-fps",0.0)),
            "lateral_velocity_fps":inf(info2,"lateral_velocity",ff(fdm,"velocities/v-aero-fps",0.0)),
            "vertical_speed_fps":inf(info2,"vertical_speed",ff(fdm,"velocities/h-dot-fps",0.0)),
            "pitch_deg":math.degrees(inf(info2,"pitch",ff(fdm,"attitude/pitch-rad",0.0))),
            "roll_deg":math.degrees(inf(info2,"roll",ff(fdm,"attitude/roll-rad",0.0))),
            "heading_error_deg":math.degrees(h_err),
            "roll_rate_rad_s":ff(fdm,"velocities/p-rad_sec",0.0),
            "pitch_rate_rad_s":ff(fdm,"velocities/q-rad_sec",0.0),
            "yaw_rate_rad_s":ff(fdm,"velocities/r-rad_sec",0.0),
            "rotor_rpm":ff(fdm,"propulsion/engine/rotor-rpm",0.0),
            "stage2_observation":[float(x) for x in obs2],
            "last_stage2_action":[float(x) for x in a],
            "raw_fdm":raw_snapshot(fdm),
        }
        break
    if term and not bool(info2.get("success",False)): raise RuntimeError("Stage 2 failed")
    if trunc: raise RuntimeError("Stage 2 truncated")
if endpoint is None: raise RuntimeError("Stage 2 did not reach endpoint")

payload={
    "description":"Locked Stage1->Stage2 continuous endpoint snapshot for Stage3 braking identification.",
    "stage1_model":STAGE1_MODEL_PATH,
    "stage2_model":STAGE2_MODEL_PATH,
    "same_fdm_object":True,
    "stage1_handoff":handoff,
    "stage2_endpoint":endpoint,
}
OUT_JSON.write_text(json.dumps(payload,indent=2),encoding="utf-8")

print("\n"+"="*110)
print("STAGE-2 ENDPOINT STATE")
print("="*110)
for k in ["stage2_time_s","forward_distance_env_ft","ground_forward_from_takeoff_ft",
          "cross_track_from_takeoff_ft","altitude_ft","forward_velocity_fps",
          "lateral_velocity_fps","vertical_speed_fps","pitch_deg","roll_deg",
          "heading_error_deg","roll_rate_rad_s","pitch_rate_rad_s","yaw_rate_rad_s","rotor_rpm"]:
    print(f"{k:34s}: {endpoint[k]}")
print("\nLast Stage-2 action:")
print(np.array2string(np.asarray(endpoint["last_stage2_action"]),precision=6,floatmode="fixed"))
print("\nSaved:",OUT_JSON)
print("\nNEXT: elevator braking-authority identification from this endpoint.")

env2.fdm=None
env1.close()
