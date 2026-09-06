# AH-1S RL Project — Progress Checkpoint
Date: 2026-09-06

## Locked / completed

### Stage 1
Model:
- models_stage1_early_distilled/AH1S_STAGE1_EARLY_DISTILLED.zip

Status:
- LOCKED
- Teacher-off validated
- Early max drift: 2.776 ft
- Do not retune.

### Stage 2
Model:
- models_stage2_hybrid_final/AH1S_STAGE2_HYBRID_FINAL.zip

Evidence:
- results_stage2_hybrid_final/final_summary.json
- results_stage2_hybrid_final/final_trace.csv

Status:
- LOCKED
- Teacher-off PASS
- Genuine PPO fine-tune already completed.

### Stage 3
Model:
- models_stage3_hybrid_final/AH1S_STAGE3_HYBRID_FINAL.zip

Evidence:
- results_stage3_hybrid_final/final_summary.json
- results_stage3_hybrid_final/final_trace.csv

Status:
- LOCKED
- Teacher-off PASS
- Genuine PPO fine-tune already completed.

## Stage 4 — completed work

### Physics / teacher
- System identification complete.
- 300->30 ft teacher qualified.
- Ground-effect curve qualified.
- 17->9 ft capture qualified.
- Touchdown/contact identification complete.
- Landed hold qualified.
- Full continuous same-FDM Stage-4 teacher LOCKED.

Locked teacher evidence:
- results_stage4_teacher_final_v1/final_trace.csv
- results_stage4_teacher_final_v1/final_summary.json

### Teacher rollout dataset
Correct dataset:
- results_stage4_teacher_rollouts_v4/

Result:
- 6/6 PASS
- obs=(47651,26)
- actions=(47651,4)
- distinct observation rollouts=6/6
- distinct action rollouts=6/6
- ROLLOUT_DIVERSITY_OK=True

### BC V1
Model:
- models_stage4_distilled_bc_v1/AH1S_STAGE4_BC_DISTILLED.zip

Result:
- Open-loop/held-out BC metrics excellent.
- Teacher-off closed-loop FAILED during high-altitude descent.
- Root cause: collective under-command / closed-loop covariate shift.

### High-altitude DAgger shadow data
Directory:
- results_stage4_dagger_shadow_v1/

Result:
- 6406 corrective samples
- Student controlled live FDM.
- Teacher produced shadow labels only.
- Not RL.

### Corrective distillation V2
Model:
- models_stage4_corrective_distilled_v2/AH1S_STAGE4_CORRECTIVE_DISTILLED_V2.zip

Interface:
- models_stage4_corrective_distilled_v2/stage4_obs_normalization.npz
- models_stage4_corrective_distilled_v2/stage4_action_mapping.npz
- models_stage4_corrective_distilled_v2/stage4_policy_interface.json

Result:
- Only collective actor-head row 0 changed.
- Elevator/aileron/rudder outputs preserved exactly.
- High-altitude DAgger collective error greatly reduced.
- Teacher-off descent became stable from ~300 ft down to ~40 ft.

### Corrective V2 teacher-off result
Evidence:
- results_stage4_teacher_off_corrective_v2/

Result:
- High-altitude divergence FIXED.
- Failed because 30-ft settle transition was missed.
- At ~30 ft vertical speed was still too negative.
- Mission manager remained in descent_300_to_30 phase.
- Aircraft then continued below the intended transition and eventually touched ground in the wrong phase.
- Near-ground/9-ft/touchdown student phases are therefore NOT yet qualified.

### Low-altitude DAgger shadow data
Directory:
- results_stage4_dagger_lowalt_v2/

Result:
- 4868 corrective samples
- targeted roughly 70->24 ft
- teacher begins requesting noticeably more collective below ~44 ft
- at ~32 ft student was ~0.0068 physical collective below teacher
- READY FOR CORRECTIVE DISTILLATION=True

## Current next script

- train_stage4_corrective_distillation_v3.py

It has NOT been run yet.

V3 design:
- Base = corrective V2
- Only collective actor-head row 0 may change
- Nominal teacher replay + high-alt DAgger + low-alt DAgger
- Elevator/aileron/rudder remain frozen
- Held-out gates protect nominal and high-alt performance while improving low-alt behavior

## Tomorrow — exact next order

1. Run:
   python -m py_compile train_stage4_corrective_distillation_v3.py
   python train_stage4_corrective_distillation_v3.py

2. Inspect:
   - nominal RMSE
   - high DAgger RMSE
   - low DAgger RMSE
   - non-collective output delta = 0
   - READY FOR FRESH TEACHER-OFF VALIDATION=True

3. If V3 passes offline gates:
   - build/run fresh teacher-off V3 JSBSim validation
   - verify 30-ft settle transition actually occurs

4. If 30-ft transition passes:
   - test near_ground_30_to_native_eq
   - test capture_to_9ft
   - test continuous_touchdown
   - test landed_hold

5. Only after full Stage-4 teacher-off PASS:
   - genuine reward-based PPO fine-tune with teacher/controllers OFF

6. Then:
   - full Stage1->2->3->4 same-FDM validation
   - top/side trajectory plots
   - final evidence and presentation wording

## Methodology wording

Do not claim end-to-end PPO for the whole mission.

Accurate wording:
"Hybrid RL + control + policy distillation architecture."

Stage-4 corrective V1/V2/V3 steps are imitation/distillation, not RL.
DAgger shadow collection is corrective teacher labeling while the student controls the live FDM.
Reward-based PPO fine-tuning has not yet been done for Stage 4.
