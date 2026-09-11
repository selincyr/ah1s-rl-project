
from pathlib import Path
import csv, json, shutil
import numpy as np
import torch
from stable_baselines3 import PPO

# Stage-4 V4 localized corrective distillation.
# NOT RL. V2 PPO weights stay unchanged.
# Only an additive collective residual is learned.
# Residual gate: descent_300_to_30 only; 0 above 45 ft, full below 40 ft.

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path("models_stage4_corrective_distilled_v2")
BASE_MODEL = BASE_DIR / "AH1S_STAGE4_CORRECTIVE_DISTILLED_V2.zip"
OBS_NORM = BASE_DIR / "stage4_obs_normalization.npz"
ACTION_MAP = BASE_DIR / "stage4_action_mapping.npz"

NOM_DIR = Path("results_stage4_teacher_rollouts_v4")
HIGH_DIR = Path("results_stage4_dagger_shadow_v1")
LOW_DIR = Path("results_stage4_dagger_lowalt_v2")

OUT_M = Path("models_stage4_localized_v4")
OUT_R = Path("results_stage4_localized_v4")
OUT_M.mkdir(parents=True, exist_ok=True)
OUT_R.mkdir(parents=True, exist_ok=True)

VAL_ID = 5
ALT_I = 0
DESCENT_PHASE_I = 21
GATE_FULL = 40.0
GATE_ZERO = 45.0

MAX_NOM = 0.0065
MAX_HIGH_RATIO = 1.15
MIN_LOW_IMPROVE = 0.35

HIGH_WEIGHTS = [0.5, 1.0, 2.0, 4.0]
LOW_WEIGHTS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
RIDGES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
          1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]


def rule(s):
    print("\n" + "=" * 120)
    print(s)
    print("=" * 120)


def rmse(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def norm_phys(a, lo, hi):
    return (2.0 * (a - lo) / (hi - lo) - 1.0).astype(np.float32)


def denorm(z, lo, hi):
    return (lo + 0.5 * (z + 1.0) * (hi - lo)).astype(np.float32)


def gate(raw):
    alt = raw[:, ALT_I]
    phase = (raw[:, DESCENT_PHASE_I] > 0.5).astype(np.float64)
    x = np.clip((GATE_ZERO - alt) / (GATE_ZERO - GATE_FULL), 0.0, 1.0)
    smooth = x * x * (3.0 - 2.0 * x)
    return phase * smooth


@torch.no_grad()
def latent_action(policy, obs, device, bs=4096):
    L, A = [], []
    for i in range(0, len(obs), bs):
        x = torch.as_tensor(obs[i:i+bs], dtype=torch.float32, device=device)
        f = policy.extract_features(x)
        l = policy.mlp_extractor.forward_actor(f)
        a = torch.clamp(policy.action_net(l), -1.0, 1.0)
        L.append(l.cpu().numpy().astype(np.float64))
        A.append(a.cpu().numpy().astype(np.float32))
    return np.concatenate(L), np.concatenate(A)


def phi(latent):
    return np.c_[latent, np.ones(len(latent), dtype=np.float64)]


def wnorm(w):
    w = np.asarray(w, dtype=np.float64)
    return w / max(1e-12, float(np.mean(w)))


def fit_ridge(Z, y, w, rf):
    w = wnorm(w)
    sw = np.sqrt(w)[:, None]
    Zw = Z * sw
    yw = y * sw[:, 0]
    M = Zw.T @ Zw
    b = Zw.T @ yw
    lam = max(1e-12, float(rf) * float(np.mean(np.diag(M))))
    beta = np.linalg.solve(M + lam * np.eye(Z.shape[1]), b)
    return beta, lam


def apply(base, P, g, beta):
    out = base.copy()
    corr = g * (P @ beta)
    out[:, 0] = np.clip(out[:, 0] + corr, -1.0, 1.0)
    return out, corr


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


rule("A — LOAD LOCKED V2 + THREE SOURCES")

required = [
    BASE_MODEL, OBS_NORM, ACTION_MAP,
    NOM_DIR/"stage4_obs_raw.npy",
    NOM_DIR/"stage4_actions_physical.npy",
    NOM_DIR/"stage4_rollout_ids.npy",
    NOM_DIR/"stage4_sample_weights.npy",
    HIGH_DIR/"dagger_obs_raw.npy",
    HIGH_DIR/"dagger_teacher_actions_physical.npy",
    HIGH_DIR/"dagger_rollout_ids.npy",
    HIGH_DIR/"dagger_priority_weights.npy",
    LOW_DIR/"dagger_obs_raw.npy",
    LOW_DIR/"dagger_teacher_actions_physical.npy",
    LOW_DIR/"dagger_student_actions_physical.npy",
    LOW_DIR/"dagger_rollout_ids.npy",
    LOW_DIR/"dagger_priority_weights.npy",
]
for p in required:
    if not p.exists():
        raise FileNotFoundError(p)

npz = np.load(OBS_NORM)
mean = npz["mean"].astype(np.float32)
std = npz["std"].astype(np.float32)
am = np.load(ACTION_MAP)
lo = am["physical_low"].astype(np.float32)
hi = am["physical_high"].astype(np.float32)

S = {
    "nom": {
        "raw": np.load(NOM_DIR/"stage4_obs_raw.npy").astype(np.float32),
        "act": np.load(NOM_DIR/"stage4_actions_physical.npy").astype(np.float32),
        "rid": np.load(NOM_DIR/"stage4_rollout_ids.npy").astype(np.int64),
        "w": np.load(NOM_DIR/"stage4_sample_weights.npy").astype(np.float32),
    },
    "high": {
        "raw": np.load(HIGH_DIR/"dagger_obs_raw.npy").astype(np.float32),
        "act": np.load(HIGH_DIR/"dagger_teacher_actions_physical.npy").astype(np.float32),
        "rid": np.load(HIGH_DIR/"dagger_rollout_ids.npy").astype(np.int64),
        "w": np.load(HIGH_DIR/"dagger_priority_weights.npy").astype(np.float32),
    },
    "low": {
        "raw": np.load(LOW_DIR/"dagger_obs_raw.npy").astype(np.float32),
        "act": np.load(LOW_DIR/"dagger_teacher_actions_physical.npy").astype(np.float32),
        "rid": np.load(LOW_DIR/"dagger_rollout_ids.npy").astype(np.int64),
        "w": np.load(LOW_DIR/"dagger_priority_weights.npy").astype(np.float32),
    },
}

if S["nom"]["raw"].shape[1] != 26:
    raise RuntimeError("Stage-4 observation must be 26-D.")

if not np.all(S["nom"]["act"] >= lo - 1e-7) or not np.all(S["nom"]["act"] <= hi + 1e-7):
    raise RuntimeError("Nominal teacher targets violate locked action mapping.")

for k in ["high", "low"]:
    S[k]["act_fit"] = S[k]["act"].copy()
    S[k]["act_fit"][:, 0] = np.clip(S[k]["act_fit"][:, 0], lo[0], hi[0])
S["nom"]["act_fit"] = S["nom"]["act"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PPO.load(str(BASE_MODEL), device=device)
model.policy.eval()

for k in S:
    d = S[k]
    d["obs"] = ((d["raw"] - mean) / std).astype(np.float32)
    d["target"] = norm_phys(d["act_fit"], lo, hi)
    d["latent"], d["base"] = latent_action(model.policy, d["obs"], device)
    d["phi"] = phi(d["latent"])
    d["gate"] = gate(d["raw"])
    d["train"] = d["rid"] != VAL_ID
    d["val"] = d["rid"] == VAL_ID
    if not np.any(d["val"]):
        raise RuntimeError(f"{k}: validation rollout {VAL_ID} missing")

print("device:", device)
for k in S:
    d = S[k]
    print(
        f"{k:5s} n={len(d['raw']):5d} | "
        f"gate>0={np.sum(d['gate']>0):5d} | "
        f"gate=1={np.sum(d['gate']>=1-1e-12):5d} | "
        f"train/val={np.sum(d['train'])}/{np.sum(d['val'])}"
    )

# Verify low DAgger was generated by this V2 policy/interface.
stored_low = np.load(LOW_DIR/"dagger_student_actions_physical.npy").astype(np.float32)
recomputed_low = denorm(S["low"]["base"], lo, hi)
stored_delta = float(np.max(np.abs(stored_low - recomputed_low)))
print(f"V2 vs stored low-DAgger student max delta: {stored_delta:.8f}")
if stored_delta > 2e-4:
    raise RuntimeError("Low DAgger and current V2 policy/interface do not match.")


rule("B — V2 BASELINE")

for k in S:
    d = S[k]
    d["base_rmse"] = rmse(d["base"][d["val"], 0] - d["target"][d["val"], 0])

base_nom = S["nom"]["base_rmse"]
base_high = S["high"]["base_rmse"]
base_low = S["low"]["base_rmse"]

base_low_phys = denorm(S["low"]["base"][S["low"]["val"]], lo, hi)[:, 0]
low_t_phys = S["low"]["act_fit"][S["low"]["val"], 0]
base_low_phys_rmse = rmse(base_low_phys - low_t_phys)

print(f"V2 nominal RMSE : {base_nom:.6f}")
print(f"V2 high RMSE    : {base_high:.6f}")
print(f"V2 low RMSE     : {base_low:.6f}")
print(f"V2 low phys RMSE: {base_low_phys_rmse:.8f}")
if base_nom > MAX_NOM:
    raise RuntimeError("Loaded V2 fails nominal baseline gate.")


rule("C — FIT LOCALIZED RESIDUAL ADAPTER")

# normalized residual target = projected teacher - frozen V2 action
for k in S:
    S[k]["y"] = (S[k]["target"][:, 0] - S[k]["base"][:, 0]).astype(np.float64)
    S[k]["fit"] = S[k]["train"] & (S[k]["gate"] > 1e-12)

if not np.any(S["nom"]["fit"]) or not np.any(S["low"]["fit"]):
    raise RuntimeError("Localized nominal/low training samples missing.")

def source_design(k):
    d = S[k]
    m = d["fit"]
    return (
        d["gate"][m, None] * d["phi"][m],
        d["y"][m],
        wnorm(d["w"][m]),
    )

Zn, yn, wn = source_design("nom")
Zl, yl, wl = source_design("low")
if np.any(S["high"]["fit"]):
    Zh, yh, wh = source_design("high")
else:
    Zh = yh = wh = None

rows, best = [], None

for hw in HIGH_WEIGHTS:
    for lw in LOW_WEIGHTS:
        Zs, ys, ws = [Zn], [yn], [wn]
        if Zh is not None:
            Zs.append(Zh); ys.append(yh); ws.append(wh * hw)
        Zs.append(Zl); ys.append(yl); ws.append(wl * lw)

        Z = np.concatenate(Zs)
        y = np.concatenate(ys)
        w = np.concatenate(ws)

        for rf in RIDGES:
            beta, lam = fit_ridge(Z, y, w, rf)

            metrics = {}
            for k in S:
                d = S[k]
                pred, _ = apply(d["base"][d["val"]], d["phi"][d["val"]],
                                d["gate"][d["val"]], beta)
                metrics[k] = rmse(pred[:, 0] - d["target"][d["val"], 0])

            high_ratio = metrics["high"] / max(1e-12, base_high)
            low_improve = 1.0 - metrics["low"] / max(1e-12, base_low)
            ok = (
                metrics["nom"] <= MAX_NOM
                and high_ratio <= MAX_HIGH_RATIO
                and low_improve >= MIN_LOW_IMPROVE
            )
            score = metrics["low"] + 0.5*metrics["high"] + 0.35*metrics["nom"]

            row = {
                "high_weight": hw,
                "low_weight": lw,
                "ridge_factor": rf,
                "lambda": lam,
                "nominal_rmse": metrics["nom"],
                "high_rmse": metrics["high"],
                "high_ratio_vs_v2": high_ratio,
                "low_rmse": metrics["low"],
                "low_improvement_fraction": low_improve,
                "score": score,
                "feasible": ok,
            }
            rows.append(row)
            if ok and (best is None or score < best["score"]):
                best = {**row, "beta": beta.copy()}

save_csv(OUT_R/"candidate_grid.csv", rows)
print("candidate count:", len(rows))
print("feasible count :", sum(bool(r["feasible"]) for r in rows))

safe = [r for r in rows if r["nominal_rmse"] <= MAX_NOM
        and r["high_ratio_vs_v2"] <= MAX_HIGH_RATIO]
safe.sort(key=lambda r: (-r["low_improvement_fraction"], r["score"]))

print("\nBest preservation-compatible candidates:")
for r in safe[:12]:
    print(
        f"H={r['high_weight']:3.1f} L={r['low_weight']:3.1f} "
        f"ridge={r['ridge_factor']:.1e} | "
        f"nom={r['nominal_rmse']:.6f} | "
        f"high={r['high_rmse']:.6f} ({r['high_ratio_vs_v2']:.3f}x) | "
        f"low={r['low_rmse']:.6f} "
        f"improve={100*r['low_improvement_fraction']:.2f}% | "
        f"ok={r['feasible']}"
    )

if best is None:
    raise RuntimeError(
        "No localized V4 adapter satisfied nominal + high preservation + low improvement gates."
    )


rule("D — FINAL V4 AUDIT + SAVE")

beta = best["beta"]
V4 = {}
outside_max = 0.0
noncoll_max = 0.0

for k in S:
    d = S[k]
    pred, corr = apply(d["base"], d["phi"], d["gate"], beta)
    V4[k] = (pred, corr)

    off = d["gate"] <= 0.0
    if np.any(off):
        outside_max = max(outside_max, float(np.max(np.abs(corr[off]))))
    noncoll_max = max(
        noncoll_max,
        float(np.max(np.abs(pred[:, 1:] - d["base"][:, 1:]))),
    )

nom_rmse = rmse(V4["nom"][0][S["nom"]["val"], 0] - S["nom"]["target"][S["nom"]["val"], 0])
high_rmse = rmse(V4["high"][0][S["high"]["val"], 0] - S["high"]["target"][S["high"]["val"], 0])
low_rmse = rmse(V4["low"][0][S["low"]["val"], 0] - S["low"]["target"][S["low"]["val"], 0])

high_ratio = high_rmse / max(1e-12, base_high)
low_improve = 1.0 - low_rmse / max(1e-12, base_low)

v4_low_phys = denorm(V4["low"][0][S["low"]["val"]], lo, hi)[:, 0]
v4_low_phys_rmse = rmse(v4_low_phys - low_t_phys)

max_corr_norm = max(float(np.max(np.abs(V4[k][1]))) for k in V4)
max_corr_phys = 0.5 * float(hi[0] - lo[0]) * max_corr_norm

ready = (
    nom_rmse <= MAX_NOM
    and high_ratio <= MAX_HIGH_RATIO
    and low_improve >= MIN_LOW_IMPROVE
    and outside_max <= 1e-12
    and noncoll_max <= 1e-12
)

print("selected H weight :", best["high_weight"])
print("selected L weight :", best["low_weight"])
print("selected ridge    :", f"{best['ridge_factor']:.1e}")
print(f"nominal RMSE      : {base_nom:.6f} -> {nom_rmse:.6f}")
print(f"high RMSE         : {base_high:.6f} -> {high_rmse:.6f} ({high_ratio:.3f}x)")
print(f"low RMSE          : {base_low:.6f} -> {low_rmse:.6f} "
      f"(improve={100*low_improve:.2f}%)")
print(f"low physical RMSE : {base_low_phys_rmse:.8f} -> {v4_low_phys_rmse:.8f}")
print(f"outside-gate correction : {outside_max:.12g}")
print(f"non-collective delta    : {noncoll_max:.12g}")
print(f"max correction norm     : {max_corr_norm:.8f}")
print(f"max correction phys     : {max_corr_phys:.8f}")

out_base = OUT_M/"AH1S_STAGE4_LOCALIZED_V4_BASE.zip"
shutil.copy2(BASE_MODEL, out_base)
shutil.copy2(OBS_NORM, OUT_M/"stage4_obs_normalization.npz")
shutil.copy2(ACTION_MAP, OUT_M/"stage4_action_mapping.npz")

np.savez(
    OUT_M/"stage4_localized_collective_adapter.npz",
    beta=beta.astype(np.float64),
    gate_full_below_ft=np.array([GATE_FULL], dtype=np.float64),
    gate_zero_above_ft=np.array([GATE_ZERO], dtype=np.float64),
    altitude_index=np.array([ALT_I], dtype=np.int64),
    phase_descent_index=np.array([DESCENT_PHASE_I], dtype=np.int64),
)

interface = {
    "architecture": "locked V2 PPO + localized learned collective residual adapter",
    "training_type": "corrective policy distillation; NOT reinforcement learning",
    "base_model_parameters_changed": False,
    "observation_dim": 26,
    "action_dim": 4,
    "normalization": "unchanged from V2",
    "action_mapping": "unchanged from V2",
    "adapter": {
        "input": "frozen V2 actor latent 128-D + bias",
        "output": "additive normalized collective residual only",
        "phase": "descent_300_to_30",
        "phase_obs_index": DESCENT_PHASE_I,
        "altitude_obs_index": ALT_I,
        "gate_full_below_ft": GATE_FULL,
        "gate_zero_above_ft": GATE_ZERO,
        "fade": "smoothstep",
    },
    "runtime": (
        "z0=clip(base_z0 + gate(raw_obs)*(phi@beta),-1,+1); "
        "z1..z3=base_z1..z3"
    ),
    "teacher_runtime": False,
    "stage4_classical_controller_runtime": False,
    "reward_based_ppo_fine_tuning_used": False,
}
with open(OUT_M/"stage4_policy_interface.json", "w") as f:
    json.dump(interface, f, indent=2)

summary = {
    "reinforcement_learning_used": False,
    "base_model_parameters_changed": False,
    "gate": {
        "full_below_ft": GATE_FULL,
        "zero_above_ft": GATE_ZERO,
        "phase": "descent_300_to_30",
    },
    "baseline_v2": {
        "nominal_rmse": base_nom,
        "high_rmse": base_high,
        "low_rmse": base_low,
        "low_physical_rmse": base_low_phys_rmse,
    },
    "selected": {k: v for k, v in best.items() if k != "beta"},
    "v4": {
        "nominal_rmse": nom_rmse,
        "high_rmse": high_rmse,
        "high_ratio_vs_v2": high_ratio,
        "low_rmse": low_rmse,
        "low_improvement_fraction": low_improve,
        "low_physical_rmse": v4_low_phys_rmse,
        "outside_gate_max_correction": outside_max,
        "non_collective_max_delta": noncoll_max,
        "max_adapter_normalized_correction": max_corr_norm,
        "max_adapter_physical_equivalent": max_corr_phys,
    },
    "ready_for_fresh_teacher_off_validation": bool(ready),
}
with open(OUT_R/"final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nLOCALIZED CORRECTIVE DISTILLATION V4 COMPLETE:", True)
print("REWARD-BASED PPO TRAINING USED:", False)
print("BASE V2 PPO PARAMETERS CHANGED:", False)
print("OUTSIDE-GATE V2 BEHAVIOR PRESERVED:", outside_max <= 1e-12)
print("NON-COLLECTIVE OUTPUTS PRESERVED:", noncoll_max <= 1e-12)
print("READY FOR FRESH TEACHER-OFF VALIDATION:", ready)
print("Saved:")
print(" ", out_base)
print(" ", OUT_M/"stage4_localized_collective_adapter.npz")
print(" ", OUT_M/"stage4_policy_interface.json")
print(" ", OUT_R/"candidate_grid.csv")
print(" ", OUT_R/"final_summary.json")
print("\nNext: fresh teacher-OFF JSBSim validation. Do NOT call this RL.")
