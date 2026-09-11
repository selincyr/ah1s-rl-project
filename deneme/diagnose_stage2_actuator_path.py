from __future__ import annotations

import inspect
import math
import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage1_distill import HelicopterEnvStage1Distill
from helicopter_env_stage2_refine import HelicopterEnvStage2Refine

STAGE1_MODEL_PATH = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)
STAGE2_MODEL_PATH = (
    "models_stage2_refine/"
    "AH1S_STAGE2_REFINE_SUCCESS.zip"
)

HANDOFF_ALT_MIN = 295.0
HANDOFF_ALT_MAX = 305.0
HANDOFF_MAX_VS = 0.50
HANDOFF_MAX_HS = 1.0
HANDOFF_MAX_DRIFT = 3.0
HANDOFF_STABLE_TIME = 5.0
STAGE1_MAX_TIME = 120.0


def rule(title: str):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)


def fdm_float(fdm, key, default=float("nan")):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def info_float(info, key, default=float("nan")):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def get_fdm(env):
    direct = getattr(env, "fdm", None)
    if direct is not None:
        return direct
    base = getattr(env, "base_env", None)
    if base is not None:
        nested = getattr(base, "fdm", None)
        if nested is not None:
            return nested
    raise RuntimeError("Active JSBSim FDM not found")


def horizontal_speed_stage1(info):
    vn = info_float(info, "vn", 0.0)
    ve = info_float(info, "ve", 0.0)
    return float(np.hypot(vn, ve))


def heading_rad(fdm):
    for key in ["attitude/heading-true-rad", "attitude/psi-rad"]:
        x = fdm_float(fdm, key)
        if np.isfinite(x):
            return x
    return float("nan")


stage1_model = PPO.load(STAGE1_MODEL_PATH)
stage2_model = PPO.load(STAGE2_MODEL_PATH)


def build_handoff():
    env1 = HelicopterEnvStage1Distill(
        teacher_model_path=None,
        training_mode=False,
    )
    obs, info = env1.reset()
    fdm = get_fdm(env1)
    mission_heading = heading_rad(fdm)

    dt = float(getattr(env1, "dt", 0.075))
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.075

    stable_time = 0.0
    last = None

    for step in range(int(STAGE1_MAX_TIME / dt)):
        action, _ = stage1_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env1.step(action)

        alt = info_float(info, "altitude")
        vs = info_float(info, "vertical_speed")
        hs = horizontal_speed_stage1(info)
        drift = info_float(info, "drift", 999.0)

        stable = (
            HANDOFF_ALT_MIN <= alt <= HANDOFF_ALT_MAX
            and abs(vs) <= HANDOFF_MAX_VS
            and hs <= HANDOFF_MAX_HS
            and drift <= HANDOFF_MAX_DRIFT
        )

        stable_time = stable_time + dt if stable else 0.0
        last = (alt, vs, hs, drift, (step + 1) * dt)

        if stable_time >= HANDOFF_STABLE_TIME:
            return env1, fdm, mission_heading, last

        if terminated and not bool(info.get("success", False)):
            break
        if truncated:
            break

    env1.close()
    raise RuntimeError(f"Stage-1 stable handoff not reached. last={last}")


def attach_stage2(active_fdm, mission_heading):
    env2 = HelicopterEnvStage2Refine()
    env2.reset()  # disposable bookkeeping FDM
    env2.fdm = active_fdm

    if hasattr(env2, "forward_distance"):
        env2.forward_distance = 0.0
    if hasattr(env2, "target_heading") and np.isfinite(mission_heading):
        env2.target_heading = float(mission_heading)

    for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
        if hasattr(env2, attr):
            setattr(env2, attr, 0)

    obs = np.asarray(env2._get_obs(), dtype=np.float32)
    return env2, obs


# Properties that reveal the control path at progressively deeper levels.
PROP_KEYS = [
    "fcs/aileron-cmd-norm",
    "fcs/aileron-cmd-norm-exmod",
    "fcs/lateral-cmd-trim-sum",
    "fcs/lateral-gain",
    "fcs/lateral-sum",
    "fcs/lateral-ctrl-rad",
    "fcs/lateral-ctrl-rad-lag",
    "ap/aileron-cmd",
    "fcs/rudder-cmd-norm",
    "fcs/rudder-cmd-norm-exmod",
    "fcs/pedal-cmd-trim-sum",
    "fcs/pedal-gain",
    "fcs/pedal-sum",
    "fcs/pedal-ctrl-rad",
    "fcs/pedal-ctrl-rad-lag",
    "fcs/antitorque-ctrl-rad",
    "ap/rudder-cmd",
    "attitude/roll-rad",
    "velocities/p-rad_sec",
    "velocities/r-rad_sec",
]


def snapshot(fdm):
    return {k: fdm_float(fdm, k) for k in PROP_KEYS}


def print_snapshot(label, snap):
    print(f"\n[{label}]")
    for key in PROP_KEYS:
        value = snap[key]
        if np.isfinite(value):
            print(f"  {key:34s} = {value:+.8f}")
        else:
            print(f"  {key:34s} = N/A")


def source_summary():
    rule("A — EXACT STAGE-2 CLASS / STEP SOURCE INSPECTION")

    print("Concrete class:", HelicopterEnvStage2Refine)
    print("Module        :", HelicopterEnvStage2Refine.__module__)
    print("Source file   :", inspect.getsourcefile(HelicopterEnvStage2Refine))
    print("MRO:")
    for cls in HelicopterEnvStage2Refine.__mro__:
        print("  -", cls.__module__ + "." + cls.__name__)

    for cls in HelicopterEnvStage2Refine.__mro__:
        if "step" not in getattr(cls, "__dict__", {}):
            continue
        try:
            src = inspect.getsource(cls.__dict__["step"])
        except Exception as exc:
            print(f"\nCould not inspect {cls.__name__}.step: {exc}")
            continue

        rule(f"STEP DEFINED BY {cls.__module__}.{cls.__name__}")
        relevant = []
        for i, line in enumerate(src.splitlines(), start=1):
            low = line.lower()
            if (
                "action" in low
                or "aileron" in low
                or "rudder" in low
                or "collective" in low
                or "elevator" in low
                or "fcs/" in low
                or "super().step" in low
            ):
                relevant.append((i, line))

        if relevant:
            for i, line in relevant:
                print(f"{i:4d}: {line}")
        else:
            print("No action/control-related lines found in this step source.")


def env_step_probe(a2_value, a3_value):
    env1, fdm, mission_heading, handoff = build_handoff()
    active_id = id(fdm)
    env2, obs = attach_stage2(fdm, mission_heading)

    if id(get_fdm(env2)) != active_id:
        raise RuntimeError("FDM continuity failed")

    base_action, _ = stage2_model.predict(obs, deterministic=True)
    base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
    test_action = base_action.copy()
    test_action[2] = float(a2_value)
    test_action[3] = float(a3_value)

    before = snapshot(fdm)
    obs2, _, terminated, truncated, info2 = env2.step(test_action)
    after = snapshot(fdm)

    result = {
        "handoff": handoff,
        "base_action": base_action,
        "test_action": test_action,
        "info_aileron": info_float(info2, "aileron"),
        "info_rudder": info_float(info2, "rudder"),
        "info_roll": info_float(info2, "roll"),
        "info_lat": info_float(info2, "lateral_velocity"),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "before": before,
        "after": after,
    }

    # Detach before closing env1 so env2 cannot own/clear the shared FDM.
    env2.fdm = None
    env1.close()
    return result


def direct_raw_write_probe(axis, value):
    env1, fdm, mission_heading, handoff = build_handoff()
    env2, obs = attach_stage2(fdm, mission_heading)

    if axis == "aileron":
        key = "fcs/aileron-cmd-norm"
    elif axis == "rudder":
        key = "fcs/rudder-cmd-norm"
    else:
        raise ValueError(axis)

    before = snapshot(fdm)
    fdm[key] = float(value)
    immediate = snapshot(fdm)

    # One raw JSBSim integration step. This tells us whether the script/FCS
    # immediately overwrites the external command before env.step is involved.
    fdm.run()
    after_one_run = snapshot(fdm)

    result = {
        "axis": axis,
        "requested": float(value),
        "handoff": handoff,
        "before": before,
        "immediate": immediate,
        "after_one_run": after_one_run,
    }

    env2.fdm = None
    env1.close()
    return result


def delta(a, b, key):
    va = a.get(key, float("nan"))
    vb = b.get(key, float("nan"))
    if np.isfinite(va) and np.isfinite(vb):
        return vb - va
    return float("nan")


source_summary()

rule("B — ENV.STEP INPUT AUTHORITY TEST")
print("Each case starts from a fresh, real Stage-1 handoff.")
print("Only action[2]/action[3] are changed for ONE Stage-2 env.step().")

cases = [
    (0.0, 0.0),
    (-1.0, 0.0),
    (+1.0, 0.0),
    (0.0, -1.0),
    (0.0, +1.0),
]

env_results = []
for a2, a3 in cases:
    r = env_step_probe(a2, a3)
    env_results.append(r)
    b = r["before"]
    a = r["after"]

    print("\n" + "-" * 132)
    print(f"CASE action[2]={a2:+.1f}, action[3]={a3:+.1f}")
    print("Base PPO action:", np.array2string(r["base_action"], precision=5, floatmode="fixed"))
    print("Used action    :", np.array2string(r["test_action"], precision=5, floatmode="fixed"))
    print(f"info aileron/rudder = {r['info_aileron']:+.8f} / {r['info_rudder']:+.8f}")
    print(f"FDM aileron cmd      = {b['fcs/aileron-cmd-norm']:+.8f} -> {a['fcs/aileron-cmd-norm']:+.8f}")
    print(f"FDM rudder cmd       = {b['fcs/rudder-cmd-norm']:+.8f} -> {a['fcs/rudder-cmd-norm']:+.8f}")
    print(f"lateral ctrl rad     = {b['fcs/lateral-ctrl-rad']:+.8f} -> {a['fcs/lateral-ctrl-rad']:+.8f}")
    print(f"pedal ctrl rad       = {b['fcs/pedal-ctrl-rad']:+.8f} -> {a['fcs/pedal-ctrl-rad']:+.8f}")
    print(f"roll / lat velocity  = {math.degrees(r['info_roll']):+.5f} deg / {r['info_lat']:+.5f} ft/s")


rule("C — DIRECT RAW JSBSIM COMMAND TEST")
print("This bypasses HelicopterEnvStage2Refine.step entirely.")
print("We write the FDM command property, read it immediately, run ONE raw JSBSim step, then read it again.")

raw_cases = [
    ("aileron", 0.05),
    ("aileron", 0.35),
    ("rudder", 0.20),
    ("rudder", 0.55),
]

raw_results = []
for axis, value in raw_cases:
    r = direct_raw_write_probe(axis, value)
    raw_results.append(r)
    key = "fcs/aileron-cmd-norm" if axis == "aileron" else "fcs/rudder-cmd-norm"

    print("\n" + "-" * 132)
    print(f"RAW {axis.upper()} request = {value:+.5f}")
    print(f"before       : {r['before'][key]:+.8f}")
    print(f"immediate    : {r['immediate'][key]:+.8f}")
    print(f"after 1 run  : {r['after_one_run'][key]:+.8f}")

    if axis == "aileron":
        print(f"exmod        : {r['immediate']['fcs/aileron-cmd-norm-exmod']:+.8f} -> {r['after_one_run']['fcs/aileron-cmd-norm-exmod']:+.8f}")
        print(f"lateral ctrl : {r['immediate']['fcs/lateral-ctrl-rad']:+.8f} -> {r['after_one_run']['fcs/lateral-ctrl-rad']:+.8f}")
    else:
        print(f"exmod        : {r['immediate']['fcs/rudder-cmd-norm-exmod']:+.8f} -> {r['after_one_run']['fcs/rudder-cmd-norm-exmod']:+.8f}")
        print(f"pedal ctrl   : {r['immediate']['fcs/pedal-ctrl-rad']:+.8f} -> {r['after_one_run']['fcs/pedal-ctrl-rad']:+.8f}")


rule("D — AUTOMATIC DIAGNOSIS")

# Compare env.step results for ±1 commands.
minus_ail = env_results[1]["after"]["fcs/aileron-cmd-norm"]
plus_ail = env_results[2]["after"]["fcs/aileron-cmd-norm"]
minus_rud = env_results[3]["after"]["fcs/rudder-cmd-norm"]
plus_rud = env_results[4]["after"]["fcs/rudder-cmd-norm"]

env_ail_changes = (
    np.isfinite(minus_ail) and np.isfinite(plus_ail)
    and abs(plus_ail - minus_ail) > 1e-5
)
env_rud_changes = (
    np.isfinite(minus_rud) and np.isfinite(plus_rud)
    and abs(plus_rud - minus_rud) > 1e-5
)

# Direct property survives one raw FDM run?
def survives_raw(axis):
    subset = [r for r in raw_results if r["axis"] == axis]
    if len(subset) < 2:
        return False
    key = "fcs/aileron-cmd-norm" if axis == "aileron" else "fcs/rudder-cmd-norm"
    vals = [r["after_one_run"][key] for r in subset]
    return all(np.isfinite(v) for v in vals) and abs(vals[1] - vals[0]) > 1e-5

raw_ail_survives = survives_raw("aileron")
raw_rud_survives = survives_raw("rudder")

print(f"env.step changes FDM aileron command across -1/+1 : {env_ail_changes}")
print(f"env.step changes FDM rudder command across -1/+1  : {env_rud_changes}")
print(f"direct raw aileron write survives one FDM run      : {raw_ail_survives}")
print(f"direct raw rudder write survives one FDM run       : {raw_rud_survives}")
print()

if (not env_ail_changes and raw_ail_survives) or (not env_rud_changes and raw_rud_survives):
    print("DIAGNOSIS: STAGE-2 ENVIRONMENT CONTROL-MAPPING PROBLEM")
    print("The raw JSBSim command path is alive, but Stage-2 env.step is not forwarding PPO lateral/yaw actions to it.")
    print("NEXT: repair Stage-2 action mapping before any more training or gain search.")
elif (not raw_ail_survives) or (not raw_rud_survives):
    print("DIAGNOSIS: JSBSIM SCRIPT / AFCS / COMMAND-OVERRIDE PROBLEM")
    print("Direct writes are being overwritten inside the FDM path.")
    print("NEXT: identify the writable upstream command property / AFCS configuration before retraining.")
elif env_ail_changes and env_rud_changes:
    print("DIAGNOSIS: COMMAND PATH IS LIVE")
    print("Then the previous identical full-flight sweep was caused by a higher-level test/controller issue, not dead actuator wiring.")
    print("NEXT: inspect command timing/gating and use short control-effectiveness pulses.")
else:
    print("DIAGNOSIS: MIXED AXIS RESULT")
    print("One control axis is live and the other is not. Use the detailed source/property output above to repair only the dead axis.")

print("\nNO TRAINING WAS PERFORMED. NO MODEL WAS MODIFIED.")
