
from pathlib import Path
import csv
import json
import numpy as np

from sklearn.neighbors import NearestNeighbors


# =====================================================================
# STAGE-4 BC CLOSED-LOOP COVARIATE-SHIFT DIAGNOSTIC V1
# =====================================================================
#
# No simulation.
# No training.
# No teacher control.
#
# This script compares the teacher-OFF student's actually visited descent
# states against the locked teacher dataset and asks:
#
#   "At this student-visited state, what physical actuator command did the
#    nearest teacher states use?"
#
# The purpose is to diagnose whether the teacher-OFF failure is mainly:
#   - collective/action bias,
#   - out-of-distribution state drift,
#   - or both.
#
# Inputs must already exist in the Colab repo from previous steps.

TEACHER_DIR = Path("results_stage4_teacher_rollouts_v4")
STUDENT_DIR = Path("results_stage4_teacher_off_bc_v3")
MODEL_DIR = Path("models_stage4_distilled_bc_v1")

TEACHER_OBS = TEACHER_DIR / "stage4_obs_raw.npy"
TEACHER_ACT = TEACHER_DIR / "stage4_actions_physical.npy"
TEACHER_PHASE = TEACHER_DIR / "stage4_phase_ids.npy"
TEACHER_ROLLOUT = TEACHER_DIR / "stage4_rollout_ids.npy"

STUDENT_TRACE = STUDENT_DIR / "teacher_off_trace.csv"

OBS_NORM = MODEL_DIR / "stage4_obs_normalization.npz"

OUT_DIR = Path("results_stage4_bc_covariate_shift_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DESCENT_PHASE_ID = 1
DESCENT_PHASE_NAME = "descent_300_to_30"

K_NEIGHBORS = 8

ACTION_NAMES = [
    "collective",
    "elevator",
    "aileron",
    "rudder",
]


def read_student_trace(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("phase") != DESCENT_PHASE_NAME:
                continue

            try:
                obs = np.asarray(
                    [float(r[f"obs_raw_{i}"]) for i in range(26)],
                    dtype=np.float32,
                )
                student_action = np.asarray(
                    [float(r[f"student_action_phys_{i}"]) for i in range(4)],
                    dtype=np.float32,
                )
            except Exception:
                continue

            rows.append({
                "mission_time_s": float(r["mission_time_s"]),
                "phase_elapsed_s": float(r["phase_elapsed_s"]),
                "altitude_ft": float(r["altitude_ft"]),
                "vertical_speed_fps": float(r["vertical_speed_fps"]),
                "position_error_ft": float(r["position_error_ft"]),
                "forward_speed_fps": float(r["forward_speed_fps"]),
                "cross_track_ft": float(r["cross_track_ft"]),
                "lateral_speed_fps": float(r["lateral_speed_fps"]),
                "obs": obs,
                "student_action": student_action,
            })

    return rows


def save_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


for p in [
    TEACHER_OBS,
    TEACHER_ACT,
    TEACHER_PHASE,
    TEACHER_ROLLOUT,
    STUDENT_TRACE,
    OBS_NORM,
]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")


print("=" * 120)
print("A — LOAD TEACHER MANIFOLD + STUDENT TEACHER-OFF TRACE")
print("=" * 120)

teacher_obs = np.load(TEACHER_OBS).astype(np.float32)
teacher_act = np.load(TEACHER_ACT).astype(np.float32)
teacher_phase = np.load(TEACHER_PHASE).astype(np.int64)
teacher_rollout = np.load(TEACHER_ROLLOUT).astype(np.int64)

norm = np.load(OBS_NORM)
obs_mean = norm["mean"].astype(np.float32)
obs_std = norm["std"].astype(np.float32)

mask = teacher_phase == DESCENT_PHASE_ID
tobs_raw = teacher_obs[mask]
tact = teacher_act[mask]
troll = teacher_rollout[mask]

student_rows = read_student_trace(STUDENT_TRACE)

if not student_rows:
    raise RuntimeError("No descent_300_to_30 rows found in student trace.")

sobs_raw = np.stack([r["obs"] for r in student_rows]).astype(np.float32)
sact = np.stack([r["student_action"] for r in student_rows]).astype(np.float32)

tobs = ((tobs_raw - obs_mean) / obs_std).astype(np.float32)
sobs = ((sobs_raw - obs_mean) / obs_std).astype(np.float32)

print(f"teacher descent states : {len(tobs)}")
print(f"student descent states : {len(sobs)}")
print(f"student ALT range      : {student_rows[-1]['altitude_ft']:.3f} .. {student_rows[0]['altitude_ft']:.3f} ft")
print(f"student final VS       : {student_rows[-1]['vertical_speed_fps']:+.3f} ft/s")


print()
print("=" * 120)
print("B — FIT TEACHER-STATE NEAREST-NEIGHBOR INDEX")
print("=" * 120)

nn = NearestNeighbors(
    n_neighbors=K_NEIGHBORS,
    algorithm="auto",
    metric="euclidean",
    n_jobs=-1,
)
nn.fit(tobs)

dist, idx = nn.kneighbors(sobs)

teacher_neighbor_actions = tact[idx]
teacher_action_mean = np.mean(teacher_neighbor_actions, axis=1)
teacher_action_median = np.median(teacher_neighbor_actions, axis=1)

mean_nn_distance = np.mean(dist, axis=1)
min_nn_distance = dist[:, 0]

delta = sact - teacher_action_mean

print(f"K neighbors: {K_NEIGHBORS}")
print(
    "NN distance student->teacher | "
    f"min={np.min(min_nn_distance):.4f} "
    f"median={np.median(min_nn_distance):.4f} "
    f"p95={np.percentile(min_nn_distance,95):.4f} "
    f"max={np.max(min_nn_distance):.4f}"
)


print()
print("=" * 120)
print("C — CLOSED-LOOP ACTION MISMATCH SUMMARY")
print("=" * 120)

for j, name in enumerate(ACTION_NAMES):
    e = delta[:, j]
    print(
        f"{name:10s} | "
        f"mean(student-teacherNN)={np.mean(e):+.8f} | "
        f"MAE={np.mean(np.abs(e)):.8f} | "
        f"P95abs={np.percentile(np.abs(e),95):.8f} | "
        f"MAXabs={np.max(np.abs(e)):.8f}"
    )


print()
print("=" * 120)
print("D — 10-SECOND DESCENT SNAPSHOTS")
print("=" * 120)

report_rows = []

next_report_t = 0.0
for i, r in enumerate(student_rows):
    t = r["phase_elapsed_s"]
    if t + 1e-9 < next_report_t and i != len(student_rows) - 1:
        continue

    row = {
        "phase_elapsed_s": float(t),
        "altitude_ft": float(r["altitude_ft"]),
        "vertical_speed_fps": float(r["vertical_speed_fps"]),
        "position_error_ft": float(r["position_error_ft"]),
        "cross_track_ft": float(r["cross_track_ft"]),
        "nn_min_distance": float(min_nn_distance[i]),
        "nn_mean_distance": float(mean_nn_distance[i]),
        "student_collective": float(sact[i,0]),
        "teacher_nn_collective": float(teacher_action_mean[i,0]),
        "delta_collective": float(delta[i,0]),
        "student_elevator": float(sact[i,1]),
        "teacher_nn_elevator": float(teacher_action_mean[i,1]),
        "delta_elevator": float(delta[i,1]),
        "student_aileron": float(sact[i,2]),
        "teacher_nn_aileron": float(teacher_action_mean[i,2]),
        "delta_aileron": float(delta[i,2]),
        "student_rudder": float(sact[i,3]),
        "teacher_nn_rudder": float(teacher_action_mean[i,3]),
        "delta_rudder": float(delta[i,3]),
    }
    report_rows.append(row)

    print(
        f"t={t:7.2f}s | ALT={r['altitude_ft']:7.2f} "
        f"VS={r['vertical_speed_fps']:+6.3f} | "
        f"COLL student={sact[i,0]:.6f} "
        f"teacherNN={teacher_action_mean[i,0]:.6f} "
        f"Δ={delta[i,0]:+.7f} | "
        f"NNdist={min_nn_distance[i]:.4f}"
    )

    next_report_t += 10.0


print()
print("=" * 120)
print("E — FIRST VERTICAL-DIVERGENCE THRESHOLDS")
print("=" * 120)

threshold_rows = []

for threshold in [-1.20, -1.40, -1.60, -2.00, -2.40]:
    found = None
    for i, r in enumerate(student_rows):
        if r["vertical_speed_fps"] <= threshold:
            found = i
            break

    if found is None:
        print(f"VS <= {threshold:+.2f}: not reached")
        continue

    i = found
    r = student_rows[i]

    tr = {
        "vs_threshold_fps": float(threshold),
        "phase_elapsed_s": float(r["phase_elapsed_s"]),
        "altitude_ft": float(r["altitude_ft"]),
        "vertical_speed_fps": float(r["vertical_speed_fps"]),
        "nn_min_distance": float(min_nn_distance[i]),
        "student_collective": float(sact[i,0]),
        "teacher_nn_collective": float(teacher_action_mean[i,0]),
        "delta_collective": float(delta[i,0]),
    }
    threshold_rows.append(tr)

    print(
        f"VS <= {threshold:+.2f} | "
        f"t={r['phase_elapsed_s']:.2f}s "
        f"ALT={r['altitude_ft']:.2f} | "
        f"COLL student={sact[i,0]:.6f} "
        f"teacherNN={teacher_action_mean[i,0]:.6f} "
        f"Δ={delta[i,0]:+.7f} | "
        f"NNdist={min_nn_distance[i]:.4f}"
    )


print()
print("=" * 120)
print("F — DIAGNOSTIC CONCLUSION")
print("=" * 120)

# Evidence-based diagnostic flags.
collective_bias_last25 = float(np.mean(delta[-max(1, len(delta)//4):, 0]))
nn_growth = float(
    np.median(min_nn_distance[-max(1, len(min_nn_distance)//4):])
    - np.median(min_nn_distance[:max(1, len(min_nn_distance)//4)])
)

covariate_shift_detected = bool(
    nn_growth > 0.25
    or np.percentile(min_nn_distance, 95) > 1.0
)

collective_undercommand_detected = bool(
    collective_bias_last25 < -2.0e-5
)

print(
    f"late collective mean student-teacherNN: "
    f"{collective_bias_last25:+.8f}"
)
print(f"NN-distance median growth: {nn_growth:+.4f}")
print(f"COVARIATE_SHIFT_DETECTED={covariate_shift_detected}")
print(
    f"COLLECTIVE_UNDERCOMMAND_DETECTED="
    f"{collective_undercommand_detected}"
)

if covariate_shift_detected:
    print(
        "The student is leaving the teacher-state manifold during closed-loop "
        "descent. Open-loop BC validation alone is therefore insufficient."
    )

if collective_undercommand_detected:
    print(
        "The student collective is systematically below the nearest teacher "
        "collective late in the descent, consistent with the accelerating "
        "downward vertical speed."
    )

print(
    "NEXT_RECOMMENDATION="
    "collect corrective teacher labels on student-visited descent states "
    "(DAgger-style shadow labeling) before reward-based PPO fine-tuning."
)

save_csv(
    OUT_DIR / "descent_10s_snapshots.csv",
    report_rows,
)
save_csv(
    OUT_DIR / "vertical_divergence_thresholds.csv",
    threshold_rows,
)

summary = {
    "teacher_descent_states": int(len(tobs)),
    "student_descent_states": int(len(sobs)),
    "k_neighbors": int(K_NEIGHBORS),
    "nn_min_distance": {
        "min": float(np.min(min_nn_distance)),
        "median": float(np.median(min_nn_distance)),
        "p95": float(np.percentile(min_nn_distance,95)),
        "max": float(np.max(min_nn_distance)),
    },
    "action_delta_student_minus_teacher_nn": {
        ACTION_NAMES[j]: {
            "mean": float(np.mean(delta[:,j])),
            "mae": float(np.mean(np.abs(delta[:,j]))),
            "p95_abs": float(np.percentile(np.abs(delta[:,j]),95)),
            "max_abs": float(np.max(np.abs(delta[:,j]))),
        }
        for j in range(4)
    },
    "late_collective_mean_delta": float(collective_bias_last25),
    "nn_distance_median_growth": float(nn_growth),
    "covariate_shift_detected": bool(covariate_shift_detected),
    "collective_undercommand_detected": bool(
        collective_undercommand_detected
    ),
    "next_recommendation": (
        "DAgger-style shadow teacher labeling on student-visited descent states "
        "before reward-based PPO fine-tuning"
    ),
}

with open(
    OUT_DIR / "final_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

print("Saved:")
print(f"  {OUT_DIR / 'descent_10s_snapshots.csv'}")
print(f"  {OUT_DIR / 'vertical_divergence_thresholds.csv'}")
print(f"  {OUT_DIR / 'final_summary.json'}")
