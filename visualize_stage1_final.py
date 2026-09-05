import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OUTPUT FOLDER
# ============================================================

OUT_DIR = "results_stage1_final"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# STAGE 1 FINAL VALIDATION DATA
# Source: final 120-second sustained validation log
# ============================================================

time_s = np.array([
    0.1, 5.0, 10.0, 15.0, 20.0, 25.1, 30.0, 35.0, 40.0, 45.0,
    50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
    100.0, 105.0, 110.0, 115.0, 120.0
])

altitude_ft = np.array([
    6.30, 37.24, 70.90, 103.07, 136.44, 170.65, 204.04, 235.20, 261.05,
    279.53, 290.91, 296.19, 297.64, 297.68, 297.65, 297.60, 297.66,
    298.02, 298.67, 299.52, 300.43, 301.28, 302.05, 302.72, 303.29
])

vertical_speed_fps = np.array([
    0.483, 7.141, 6.483, 6.553, 6.730, 6.853, 6.559, 5.755, 4.469,
    2.989, 1.588, 0.594, 0.059, 0.006, -0.026, -0.004, 0.037, 0.105,
    0.154, 0.179, 0.179, 0.163, 0.144, 0.123, 0.107
])

drift_ft = np.array([
    0.00, 4.44, 2.55, 0.77, 0.56, 0.85, 1.04, 1.30, 1.29, 1.25,
    1.24, 1.18, 1.22, 0.91, 0.79, 1.10, 1.13, 1.02, 1.03, 0.95,
    0.92, 0.88, 0.84, 0.81, 0.77
])

path_ft = np.array([
    0.00, 4.51, 7.11, 9.36, 10.64, 11.84, 12.39, 12.90, 13.12, 13.30,
    13.47, 13.55, 13.62, 14.00, 14.39, 14.87, 15.01, 15.14, 15.21, 15.29,
    15.36, 15.42, 15.47, 15.53, 15.57
])

north_ft = np.array([
    0.00, -2.36, -0.67, -0.07, -0.56, 0.42, -0.02, 0.44, 0.39, 0.46,
    0.58, 0.52, 0.59, 0.59, 0.70, 0.73, 0.71, 0.59, 0.64, 0.58,
    0.60, 0.59, 0.59, 0.60, 0.60
])

east_ft = np.array([
    -0.00, -3.76, -2.46, -0.77, 0.10, 0.74, 1.04, 1.22, 1.23, 1.16,
    1.10, 1.06, 1.06, 0.69, 0.36, 0.83, 0.87, 0.84, 0.81, 0.76,
    0.70, 0.65, 0.59, 0.54, 0.49
])

collective = np.array([
    0.64932, 0.64994, 0.64998, 0.65000, 0.65000, 0.64911, 0.64494,
    0.63841, 0.62980, 0.62135, 0.61473, 0.61079, 0.60900, 0.60892,
    0.60860, 0.60844, 0.60844, 0.60844, 0.60828, 0.60804, 0.60772,
    0.60740, 0.60708, 0.60680, 0.60657
])

# ============================================================
# SUMMARY METRICS
# ============================================================

TARGET_ALT = 300.0
HOVER_BAND_LOW = 295.0
HOVER_BAND_HIGH = 305.0
HOVER_START = 60.0

mean_alt = 299.473
std_alt = 1.958
min_alt = 297.601
max_alt = 303.285
max_alt_error = 3.285
max_abs_vs = 0.182
alt_violations = 0
vs_violations = 0
max_drift = 4.648
final_drift = 0.774
final_path = 15.574
final_alt = 303.285
final_vs = 0.107

# ============================================================
# STYLE
# ============================================================

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

# ============================================================
# 1) ALTITUDE PROFILE
# ============================================================

plt.figure()
plt.plot(time_s, altitude_ft, marker="o", linewidth=2, label="Altitude")
plt.axhline(TARGET_ALT, linestyle="--", linewidth=1.5, label="300 ft target")
plt.axhspan(HOVER_BAND_LOW, HOVER_BAND_HIGH, alpha=0.15, label="295-305 ft hover band")
plt.axvline(HOVER_START, linestyle=":", linewidth=1.5, label="Hover validation start (60 s)")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (ft)")
plt.title("AH-1S Stage 1 - Altitude Profile")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stage1_altitude_profile.png", dpi=200)
plt.close()

# ============================================================
# 2) XY PATH
# ============================================================

plt.figure()
plt.plot(east_ft, north_ft, marker="o", linewidth=2, label="Flight path")
plt.scatter(east_ft[0], north_ft[0], s=80, label="Start")
plt.scatter(east_ft[-1], north_ft[-1], s=80, label="End")
plt.xlabel("East displacement (ft)")
plt.ylabel("North displacement (ft)")
plt.title("AH-1S Stage 1 - Top View XY Path")
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stage1_xy_path.png", dpi=200)
plt.close()

# ============================================================
# 3) DRIFT
# ============================================================

plt.figure()
plt.plot(time_s, drift_ft, marker="o", linewidth=2, label="Drift")
plt.axhline(max_drift, linestyle="--", linewidth=1.5, label=f"Max drift = {max_drift:.3f} ft")
plt.xlabel("Time (s)")
plt.ylabel("Drift from start (ft)")
plt.title("AH-1S Stage 1 - Drift Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stage1_drift_profile.png", dpi=200)
plt.close()

# ============================================================
# 4) COLLECTIVE COMMAND
# ============================================================

plt.figure()
plt.plot(time_s, collective, marker="o", linewidth=2, label="Collective")
plt.xlabel("Time (s)")
plt.ylabel("Collective")
plt.title("AH-1S Stage 1 - Collective Command")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stage1_collective_profile.png", dpi=200)
plt.close()

# ============================================================
# 5) VERTICAL SPEED
# ============================================================

plt.figure()
plt.plot(time_s, vertical_speed_fps, marker="o", linewidth=2, label="Vertical speed")
plt.axhline(0.0, linestyle="--", linewidth=1.2)
plt.axvline(HOVER_START, linestyle=":", linewidth=1.5, label="Hover validation start (60 s)")
plt.xlabel("Time (s)")
plt.ylabel("Vertical speed (ft/s)")
plt.title("AH-1S Stage 1 - Vertical Speed")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stage1_vertical_speed.png", dpi=200)
plt.close()

# ============================================================
# 6) SUMMARY IMAGE
# ============================================================

fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor("white")

summary_text = f"""
AH-1S STAGE 1 FINAL RESULT

Mission:
Takeoff -> climb to 300 ft -> sustained hover

Controller:
Single PPO, 4 actions
Teacher OFF
PD controller OFF
Classical XY controller OFF
Runtime bias OFF

120 s Sustained Validation:
Mean altitude   : {mean_alt:.3f} ft
Std altitude    : {std_alt:.3f} ft
Min altitude    : {min_alt:.3f} ft
Max altitude    : {max_alt:.3f} ft
Max alt error   : {max_alt_error:.3f} ft
Max |VS|        : {max_abs_vs:.3f} ft/s
Alt violations  : {alt_violations}
VS violations   : {vs_violations}

XY Performance:
Max drift       : {max_drift:.3f} ft
Final drift     : {final_drift:.3f} ft
Path length     : {final_path:.3f} ft

Final state:
Final altitude  : {final_alt:.3f} ft
Final VS        : {final_vs:.3f} ft/s

Result:
STAGE 1 COMPLETE
"""

plt.axis("off")
plt.text(
    0.02, 0.98, summary_text,
    va="top", ha="left",
    fontsize=12,
    family="monospace"
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stage1_summary.png", dpi=200)
plt.close()

# ============================================================
# 7) MARKDOWN SUMMARY FILE
# ============================================================

md = f"""# AH-1S Stage 1 Final Result

## Mission
Takeoff -> climb to **300 ft** -> sustained hover

## Controller
- Single PPO
- 4 actions
- Teacher OFF
- PD controller OFF
- Classical XY controller OFF
- Runtime bias OFF

## Final validation result
- **Sustained success:** True
- **Validation duration:** 120 s
- **Hover window:** 60-120 s

## Hover performance
- Mean altitude: **{mean_alt:.3f} ft**
- Std altitude: **{std_alt:.3f} ft**
- Min altitude: **{min_alt:.3f} ft**
- Max altitude: **{max_alt:.3f} ft**
- Max altitude error: **{max_alt_error:.3f} ft**
- Max |vertical speed|: **{max_abs_vs:.3f} ft/s**
- Altitude violations: **{alt_violations}**
- Vertical speed violations: **{vs_violations}**

## XY performance
- Max drift: **{max_drift:.3f} ft**
- Final drift: **{final_drift:.3f} ft**
- Total XY path length: **{final_path:.3f} ft**

## Final state
- Final altitude: **{final_alt:.3f} ft**
- Final vertical speed: **{final_vs:.3f} ft/s**

## Figures

### Altitude Profile
![Altitude](stage1_altitude_profile.png)

### XY Path
![XY Path](stage1_xy_path.png)

### Drift Profile
![Drift](stage1_drift_profile.png)

### Collective Profile
![Collective](stage1_collective_profile.png)

### Vertical Speed
![Vertical Speed](stage1_vertical_speed.png)

### Summary
![Summary](stage1_summary.png)
"""

with open(f"{OUT_DIR}/README_STAGE1.md", "w", encoding="utf-8") as f:
    f.write(md)

print("=" * 70)
print("STAGE 1 VISUALIZATION FILES CREATED")
print("=" * 70)
print(f"Folder: {OUT_DIR}")
print()
print("Created files:")
for name in sorted(os.listdir(OUT_DIR)):
    print("-", name)
