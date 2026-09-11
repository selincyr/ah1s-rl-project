from pathlib import Path
import re

from stable_baselines3 import PPO


ROOT = Path(".")

STAGE1_MODEL = (
    "models_stage1_early_distilled/"
    "AH1S_STAGE1_EARLY_DISTILLED.zip"
)

STAGE2_MODEL = (
    "models_stage2_refine/"
    "AH1S_STAGE2_REFINE_SUCCESS.zip"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 120)
print("STAGE 2 CONTINUOUS-HANDOFF INTERFACE INSPECTION")
print("=" * 120)

print(
    "\nThis script DOES NOT train, modify, or overwrite anything."
)
print(
    "It only identifies the exact Stage-2 environment/interface "
    "needed for the continuous handoff."
)


# ============================================================
# MODEL SHAPES
# ============================================================

print("\n" + "=" * 120)
print("MODEL INTERFACES")
print("=" * 120)

for label, model_path in [
    ("STAGE 1 LOCKED", STAGE1_MODEL),
    ("STAGE 2 REFINE", STAGE2_MODEL),
]:
    print(f"\n[{label}]")
    print("Path:", model_path)

    if not Path(model_path).exists():
        print("FILE EXISTS: False")
        continue

    print("FILE EXISTS: True")

    model = PPO.load(model_path)

    print(
        "Observation space:",
        model.observation_space,
    )

    print(
        "Action space     :",
        model.action_space,
    )

    print(
        "Action head shape:",
        tuple(
            model.policy.action_net.weight.shape
        ),
    )

    try:
        print(
            "Policy features :",
            model.policy.features_dim,
        )
    except Exception:
        pass


# ============================================================
# FIND STAGE-2 RELATED PYTHON FILES
# ============================================================

print("\n" + "=" * 120)
print("SCANNING REPOSITORY FOR STAGE-2 SOURCE")
print("=" * 120)

IGNORE_PARTS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "venv",
    ".venv",
    "env",
    "site-packages",
}

KEYWORDS = {
    "AH1S_STAGE2_REFINE_SUCCESS": 25,
    "models_stage2_refine": 20,
    "STAGE 2 ALTITUDE-HOLD REFINEMENT": 20,
    "STAGE 2 REFINE": 15,
    "STAGE2_REFINE": 15,
    "forward_distance": 10,
    "target_distance": 10,
    "distance_travelled": 8,
    "distance_traveled": 8,
    "starting state": 5,
    "stable hover": 5,
    "300.0": 1,
}

candidates = []

for path in ROOT.rglob("*.py"):
    if any(
        part in IGNORE_PARTS
        for part in path.parts
    ):
        continue

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        continue

    lower = text.lower()

    score = 0
    matched = []

    for keyword, weight in KEYWORDS.items():
        if keyword.lower() in lower:
            score += weight
            matched.append(keyword)

    # Strong extra signal for SB3 training/evaluation files.
    if "ppo" in lower:
        score += 1

    if "gymnasium" in lower or "gym" in lower:
        score += 1

    if score > 0:
        candidates.append(
            (
                score,
                path,
                text,
                matched,
            )
        )

candidates.sort(
    key=lambda item: (
        -item[0],
        str(item[1]),
    )
)

print(
    f"\nFound {len(candidates)} candidate Python files."
)

print("\nTOP CANDIDATES:")

for rank, (
    score,
    path,
    text,
    matched,
) in enumerate(
    candidates[:15],
    start=1,
):
    print(
        f"{rank:2d}. score={score:3d} | {path}"
    )

    if matched:
        print(
            "    matches:",
            ", ".join(matched[:6]),
        )


# ============================================================
# PRINT USEFUL SOURCE STRUCTURE
# ============================================================

print("\n" + "=" * 120)
print("TOP CANDIDATE SOURCE STRUCTURE")
print("=" * 120)

PATTERNS = [
    re.compile(
        r"^\s*(?:from\s+[\w\.]+\s+import\s+.+|import\s+.+)$"
    ),
    re.compile(
        r"^\s*class\s+\w+.*:"
    ),
    re.compile(
        r"^\s*def\s+(?:__init__|reset|step|_get_obs|_state|"
        r"_create_info|evaluate|eval_model)\s*\("
    ),
]

IMPORTANT_TERMS = [
    "target_distance",
    "forward_distance",
    "distance",
    "initial_",
    "reset(",
    "run_ic",
    "_get_obs",
    "_create_info",
    "target_altitude",
    "target_forward",
    "collective_scale",
    "elevator_scale",
    "aileron_scale",
    "rudder_scale",
    "observation_space",
    "action_space",
    "fdm",
]

for rank, (
    score,
    path,
    text,
    matched,
) in enumerate(
    candidates[:6],
    start=1,
):
    print("\n" + "-" * 120)
    print(
        f"CANDIDATE {rank}: {path} | score={score}"
    )
    print("-" * 120)

    lines = text.splitlines()

    selected = []

    for lineno, line in enumerate(
        lines,
        start=1,
    ):
        stripped = line.strip()

        structural = any(
            pattern.search(line)
            for pattern in PATTERNS
        )

        important = any(
            term.lower() in stripped.lower()
            for term in IMPORTANT_TERMS
        )

        if structural or important:
            selected.append(
                (
                    lineno,
                    line.rstrip(),
                )
            )

    # Keep output useful instead of dumping entire files.
    for lineno, line in selected[:90]:
        print(
            f"{lineno:5d}: {line}"
        )

    if len(selected) > 90:
        print(
            f"... {len(selected) - 90} more matching lines omitted"
        )


# ============================================================
# FIND EXACT MODEL REFERENCES
# ============================================================

print("\n" + "=" * 120)
print("EXACT STAGE-2 MODEL REFERENCES")
print("=" * 120)

exact_hits = []

for score, path, text, matched in candidates:
    lines = text.splitlines()

    for lineno, line in enumerate(
        lines,
        start=1,
    ):
        if (
            "AH1S_STAGE2_REFINE_SUCCESS" in line
            or
            "models_stage2_refine" in line
        ):
            exact_hits.append(
                (
                    path,
                    lineno,
                    line.rstrip(),
                )
            )

for path, lineno, line in exact_hits[:80]:
    print(
        f"{path}:{lineno}: {line}"
    )

if not exact_hits:
    print(
        "No exact Stage-2-refine model reference found in Python files."
    )


# ============================================================
# ENVIRONMENT CLASS CANDIDATES
# ============================================================

print("\n" + "=" * 120)
print("ENVIRONMENT CLASS CANDIDATES")
print("=" * 120)

for score, path, text, matched in candidates[:10]:
    class_matches = re.findall(
        r"^\s*class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:",
        text,
        flags=re.MULTILINE,
    )

    for class_name, bases in class_matches:
        if (
            "env" in class_name.lower()
            or
            "gym" in bases.lower()
        ):
            print(
                f"{path} -> class {class_name}({bases})"
            )


# ============================================================
# IMPORTANT COMPATIBILITY CHECK
# ============================================================

print("\n" + "=" * 120)
print("HANDOFF COMPATIBILITY SUMMARY")
print("=" * 120)

try:
    stage1 = PPO.load(
        STAGE1_MODEL
    )
    stage2 = PPO.load(
        STAGE2_MODEL
    )

    s1_obs = stage1.observation_space.shape
    s2_obs = stage2.observation_space.shape

    s1_act = stage1.action_space.shape
    s2_act = stage2.action_space.shape

    print(
        "Stage1 obs shape:",
        s1_obs,
    )

    print(
        "Stage2 obs shape:",
        s2_obs,
    )

    print(
        "Stage1 action shape:",
        s1_act,
    )

    print(
        "Stage2 action shape:",
        s2_act,
    )

    print(
        "\nObservation shapes equal:",
        s1_obs == s2_obs,
    )

    print(
        "Action shapes equal     :",
        s1_act == s2_act,
    )

    if s1_obs != s2_obs:
        print(
            "\nIMPORTANT:"
        )
        print(
            "The two PPOs do NOT consume the same observation shape."
        )
        print(
            "That is okay for continuous FDM handoff, but Stage 2 "
            "must build its OWN observation from the SAME JSBSim state."
        )

except Exception as exc:
    print(
        "Compatibility check failed:",
        repr(exc),
    )


print("\n" + "=" * 120)
print("INSPECTION COMPLETE")
print("=" * 120)

print(
    "\nSend this output back. The next script will be the actual:"
)
print(
    "Stage1 locked PPO -> SAME JSBSim FDM -> Stage2 refine PPO"
)
print(
    "continuous-handoff validation, with NO reset and NO training."
)
