
from pathlib import Path
import json
import hashlib
import numpy as np

RESULT_DIR = Path("results_stage4_teacher_rollouts_v1")

OBS_PATH = RESULT_DIR / "stage4_obs_raw.npy"
ACT_PATH = RESULT_DIR / "stage4_actions_physical.npy"
PHASE_PATH = RESULT_DIR / "stage4_phase_ids.npy"
ROLLOUT_PATH = RESULT_DIR / "stage4_rollout_ids.npy"
META_PATH = RESULT_DIR / "dataset_metadata.json"

for p in [OBS_PATH, ACT_PATH, PHASE_PATH, ROLLOUT_PATH, META_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

obs = np.load(OBS_PATH)
act = np.load(ACT_PATH)
phase = np.load(PHASE_PATH)
rid = np.load(ROLLOUT_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

rollout_ids = sorted(int(x) for x in np.unique(rid))

print("=" * 120)
print("A — DATASET / ROLLOUT SHAPES")
print("=" * 120)
print(f"obs shape     : {obs.shape}")
print(f"actions shape : {act.shape}")
print(f"phase shape   : {phase.shape}")
print(f"rollout shape : {rid.shape}")
print(f"rollout ids   : {rollout_ids}")
print()

def block_for_rollout(r):
    m = rid == r
    return obs[m], act[m], phase[m]

def block_hash(x):
    arr = np.ascontiguousarray(x)
    return hashlib.sha256(arr.tobytes()).hexdigest()

rows = []

for r in rollout_ids:
    o, a, p = block_for_rollout(r)
    rows.append({
        "rollout_id": r,
        "samples": len(o),
        "obs_hash": block_hash(o),
        "act_hash": block_hash(a),
        "phase_hash": block_hash(p),
    })

print("=" * 120)
print("B — PER-ROLLOUT HASHES")
print("=" * 120)
for row in rows:
    print(
        f"R{row['rollout_id']:02d} | n={row['samples']:5d} | "
        f"obs={row['obs_hash'][:16]} | "
        f"act={row['act_hash'][:16]} | "
        f"phase={row['phase_hash'][:16]}"
    )

print()
print("=" * 120)
print("C — PAIRWISE NUMERICAL DIFFERENCES")
print("=" * 120)

all_pairwise_identical = True

for i, r1 in enumerate(rollout_ids):
    o1, a1, p1 = block_for_rollout(r1)

    for r2 in rollout_ids[i + 1:]:
        o2, a2, p2 = block_for_rollout(r2)

        same_len = len(o1) == len(o2)

        if same_len:
            obs_max = float(np.max(np.abs(o1 - o2)))
            act_max = float(np.max(np.abs(a1 - a2)))
            phase_equal = bool(np.array_equal(p1, p2))
            obs_equal = bool(np.array_equal(o1, o2))
            act_equal = bool(np.array_equal(a1, a2))
        else:
            obs_max = float("nan")
            act_max = float("nan")
            phase_equal = False
            obs_equal = False
            act_equal = False

        identical = bool(
            same_len and obs_equal and act_equal and phase_equal
        )
        all_pairwise_identical = (
            all_pairwise_identical and identical
        )

        print(
            f"R{r1:02d} vs R{r2:02d} | "
            f"same_len={same_len} | "
            f"obs_equal={obs_equal} maxΔobs={obs_max:.9g} | "
            f"act_equal={act_equal} maxΔact={act_max:.9g} | "
            f"phase_equal={phase_equal}"
        )

print()
print("=" * 120)
print("D — UNIQUE STATE / TARGET COVERAGE")
print("=" * 120)

# Rounded uniqueness is more meaningful than raw float-byte uniqueness.
for decimals in [6, 5, 4, 3]:
    obs_unique = np.unique(np.round(obs, decimals), axis=0).shape[0]
    act_unique = np.unique(np.round(act, decimals), axis=0).shape[0]

    print(
        f"round={decimals} | "
        f"unique obs={obs_unique}/{len(obs)} "
        f"({100.0 * obs_unique / len(obs):.2f}%) | "
        f"unique actions={act_unique}/{len(act)} "
        f"({100.0 * act_unique / len(act):.2f}%)"
    )

print()
print("=" * 120)
print("E — CONCLUSION")
print("=" * 120)

if all_pairwise_identical:
    print("ROLLOUT_DIVERSITY_OK=False")
    print("All rollout blocks are byte-for-byte identical.")
    print(
        "Do NOT perform Stage-4 behavioral cloning from this 6-rollout dataset "
        "as if it contained six independent trajectories."
    )
    print(
        "The locked teacher itself is valid; only the rollout-diversity "
        "collection strategy must be changed."
    )
else:
    # Also count distinct hashes.
    distinct_obs_hashes = len({r["obs_hash"] for r in rows})
    distinct_act_hashes = len({r["act_hash"] for r in rows})

    print("ROLLOUT_DIVERSITY_OK=True")
    print(
        f"distinct observation rollout hashes: "
        f"{distinct_obs_hashes}/{len(rows)}"
    )
    print(
        f"distinct action rollout hashes: "
        f"{distinct_act_hashes}/{len(rows)}"
    )
    print(
        "If numerical differences are non-trivial and each rollout remains "
        "teacher-PASS, the dataset can proceed to BC design."
    )
