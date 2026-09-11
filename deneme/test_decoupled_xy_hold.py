import numpy as np

from stable_baselines3 import PPO

from helicopter_env_v2 import HelicopterEnvV2


# ============================================================
# EXISTING SUCCESSFUL VERTICAL PPO
# ============================================================

MODEL_PATH = (
    "models_v2/"
    "AH1S_STAGE1_SUCCESS.zip"
)

vertical_model = PPO.load(
    MODEL_PATH
)


# ============================================================
# BASE TRIMS
# ============================================================

BASE_ELEVATOR = -0.15390
BASE_AILERON = 0.19100
BASE_RUDDER = 0.39000


# ============================================================
# IDENTIFIED AH-1S CONTROL EFFECTIVENESS MATRIX
#
# Average of:
#
# t=10
# t=20
# t=30
# t=40
#
# G maps cyclic command change to
# velocity change after a 1.5 second pulse:
#
# [dVN]   [ G11 G12 ] [dELE]
# [dVE] = [ G21 G22 ] [dAIL]
# ============================================================

G = np.array(
    [
        [
            -7.33390460,
             0.43087764
        ],

        [
            -0.97846083,
            -10.19439028
        ],
    ],
    dtype=np.float64
)


PULSE_DURATION = 1.5


# ============================================================
# Convert velocity-response matrix into approximate
# acceleration effectiveness matrix:
#
# dv = G du
#
# dv / dt = B du
#
# B = G / 1.5
# ============================================================

B = (
    G
    /
    PULSE_DURATION
)


B_INV = np.linalg.inv(
    B
)


print("=" * 120)
print("AH-1S DECOUPLED XY POSITION HOLD")
print("=" * 120)

print("\nG:")
print(G)

print("\nB:")
print(B)

print("\nB inverse:")
print(B_INV)

print(
    "\ncond(G):",
    np.linalg.cond(G)
)

print(
    "det(G):",
    np.linalg.det(G)
)


# ============================================================
# CONTROLLER SETTINGS
#
# kp = position feedback
# kd = velocity damping
#
# We sweep several sensible values.
# ============================================================

GAIN_SETS = [

    # gentle
    (0.004, 0.080),

    (0.006, 0.100),

    (0.008, 0.120),

    (0.010, 0.140),

    (0.012, 0.160),

    (0.014, 0.180),

    (0.016, 0.200),
]


# Maximum horizontal acceleration request
A_MAX = 0.12


# Maximum cyclic correction around trim
MAX_ELEVATOR_DELTA = 0.026
MAX_AILERON_DELTA = 0.026


# Command smoothing
SMOOTH_ALPHA = 0.18


# ============================================================
# RUN ONE CONTROLLER
# ============================================================

def run_controller(
    kp,
    kd,
    detailed=False
):

    env = HelicopterEnvV2()

    obs, info = env.reset()


    dt = (
        env.JSBSIM_DT
        *
        env.PHYSICS_STEPS
    )


    north = 0.0
    east = 0.0

    horizontal_path = 0.0
    max_drift = 0.0


    previous_elevator = (
        BASE_ELEVATOR
    )

    previous_aileron = (
        BASE_AILERON
    )


    stable_steps = 0

    required_stable_steps = int(
        10.0
        /
        dt
    )


    success = False
    failed = False


    max_time = 110.0

    max_steps = int(
        max_time
        /
        dt
    )


    next_print = 0.0


    trajectory = []


    for step in range(
        max_steps
    ):

        t = (
            step
            *
            dt
        )


        # ====================================================
        # EXISTING STAGE 1 PPO
        #
        # COLLECTIVE ONLY
        # ====================================================

        vertical_obs = (
            env._get_obs()
        )


        action, _ = (
            vertical_model.predict(
                vertical_obs,
                deterministic=True
            )
        )


        action = np.asarray(
            action,
            dtype=np.float32
        )


        collective = (
            0.620
            +
            0.030
            *
            float(
                action[0]
            )
        )


        collective = float(
            np.clip(
                collective,
                0.590,
                0.650
            )
        )


        # ====================================================
        # CURRENT EARTH-FRAME VELOCITY
        # ====================================================

        vn = float(
            env.fdm[
                "velocities/v-north-fps"
            ]
        )

        ve = float(
            env.fdm[
                "velocities/v-east-fps"
            ]
        )


        # ====================================================
        # POSITION + VELOCITY CONTROLLER
        #
        # Desired acceleration:
        #
        # a = -Kp*p - Kd*v
        #
        # This is exactly what the PPO was failing to learn:
        #
        # position correction
        # +
        # velocity braking
        # ====================================================

        position_vector = np.array(
            [
                north,
                east
            ],
            dtype=np.float64
        )


        velocity_vector = np.array(
            [
                vn,
                ve
            ],
            dtype=np.float64
        )


        desired_acceleration = (
            -kp
            *
            position_vector

            -

            kd
            *
            velocity_vector
        )


        # ====================================================
        # LIMIT ACCELERATION DEMAND
        # ====================================================

        desired_acceleration = np.clip(
            desired_acceleration,
            -A_MAX,
            A_MAX
        )


        # ====================================================
        # DECOUPLING
        #
        # a = B * du
        #
        # therefore:
        #
        # du = B^-1 * a
        # ====================================================

        cyclic_delta = (
            B_INV
            @
            desired_acceleration
        )


        elevator_delta = float(
            cyclic_delta[0]
        )

        aileron_delta = float(
            cyclic_delta[1]
        )


        # ====================================================
        # AUTHORITY LIMIT
        # ====================================================

        elevator_delta = float(
            np.clip(
                elevator_delta,
                -MAX_ELEVATOR_DELTA,
                MAX_ELEVATOR_DELTA
            )
        )


        aileron_delta = float(
            np.clip(
                aileron_delta,
                -MAX_AILERON_DELTA,
                MAX_AILERON_DELTA
            )
        )


        # ====================================================
        # TARGET CONTROLS
        # ====================================================

        elevator_target = (
            BASE_ELEVATOR
            +
            elevator_delta
        )


        aileron_target = (
            BASE_AILERON
            +
            aileron_delta
        )


        elevator_target = float(
            np.clip(
                elevator_target,
                -0.1800,
                -0.1280
            )
        )


        aileron_target = float(
            np.clip(
                aileron_target,
                0.1650,
                0.2170
            )
        )


        # ====================================================
        # SMOOTH COMMANDS
        # ====================================================

        elevator = (
            previous_elevator
            +
            SMOOTH_ALPHA
            *
            (
                elevator_target
                -
                previous_elevator
            )
        )


        aileron = (
            previous_aileron
            +
            SMOOTH_ALPHA
            *
            (
                aileron_target
                -
                previous_aileron
            )
        )


        previous_elevator = (
            elevator
        )

        previous_aileron = (
            aileron
        )


        # ====================================================
        # APPLY
        # ====================================================

        env.fdm[
            "fcs/collective-cmd-norm"
        ] = collective


        env.fdm[
            "fcs/elevator-cmd-norm"
        ] = elevator


        env.fdm[
            "fcs/aileron-cmd-norm"
        ] = aileron


        env.fdm[
            "fcs/rudder-cmd-norm"
        ] = BASE_RUDDER


        # ====================================================
        # PHYSICS
        # ====================================================

        physics_ok = True


        for _ in range(
            env.PHYSICS_STEPS
        ):

            if not env.fdm.run():

                physics_ok = False
                break


        if not physics_ok:

            failed = True
            break


        # ====================================================
        # NEW STATE
        # ====================================================

        raw = (
            env._raw_state()
        )


        vn = float(
            env.fdm[
                "velocities/v-north-fps"
            ]
        )

        ve = float(
            env.fdm[
                "velocities/v-east-fps"
            ]
        )


        # ====================================================
        # INTEGRATE POSITION
        # ====================================================

        north += (
            vn
            *
            dt
        )

        east += (
            ve
            *
            dt
        )


        horizontal_speed = float(
            np.hypot(
                vn,
                ve
            )
        )


        horizontal_path += (
            horizontal_speed
            *
            dt
        )


        drift = float(
            np.hypot(
                north,
                east
            )
        )


        max_drift = max(
            max_drift,
            drift
        )


        altitude = float(
            raw["altitude"]
        )


        vertical_speed = float(
            raw["vertical_speed"]
        )


        pitch = float(
            raw["pitch"]
        )


        roll = float(
            raw["roll"]
        )


        trajectory.append(
            (
                t + dt,
                altitude,
                north,
                east,
                drift,
                vn,
                ve,
                collective,
                elevator,
                aileron
            )
        )


        # ====================================================
        # PRINT
        # ====================================================

        if (
            detailed
            and
            t >= next_print
        ):

            print(
                f"t={t:6.1f}s | "
                f"ALT={altitude:7.2f} | "
                f"DRIFT={drift:6.2f} | "
                f"MAX={max_drift:6.2f} | "
                f"PATH={horizontal_path:6.2f} | "
                f"N={north:7.2f} | "
                f"E={east:7.2f} | "
                f"VN={vn:6.2f} | "
                f"VE={ve:6.2f} | "
                f"ELE={elevator:.5f} | "
                f"AIL={aileron:.5f}"
            )


            next_print += 5.0


        # ====================================================
        # STRICT STABILITY
        # ====================================================

        altitude_error = abs(
            300.0
            -
            altitude
        )


        stable = (

            altitude_error < 10.0

            and

            abs(
                vertical_speed
            ) < 1.0

            and

            horizontal_speed < 1.5

            and

            drift < 5.0

            and

            abs(
                pitch
            ) < 0.12

            and

            abs(
                roll
            ) < 0.12
        )


        if stable:

            stable_steps += 1

        else:

            stable_steps = 0


        # ====================================================
        # STRICT SUCCESS
        # ====================================================

        if (

            stable_steps
            >=
            required_stable_steps

            and

            max_drift
            <=
            8.0

            and

            drift
            <=
            5.0

            and

            horizontal_path
            <=
            25.0
        ):

            success = True
            break


        # ====================================================
        # SAFETY ONLY
        # ====================================================

        if altitude > 380.0:

            failed = True
            break


        if abs(
            pitch
        ) > 0.60:

            failed = True
            break


        if abs(
            roll
        ) > 0.60:

            failed = True
            break


        if drift > 60.0:

            failed = True
            break


    # ========================================================
    # FINAL
    # ========================================================

    final_drift = float(
        np.hypot(
            north,
            east
        )
    )


    result = {

        "kp":
            kp,

        "kd":
            kd,

        "success":
            success,

        "failed":
            failed,

        "time":
            (step + 1)
            *
            dt,

        "altitude":
            altitude,

        "vertical_speed":
            vertical_speed,

        "max_drift":
            max_drift,

        "final_drift":
            final_drift,

        "path":
            horizontal_path,

        "north":
            north,

        "east":
            east,

        "vn":
            vn,

        "ve":
            ve,

        "trajectory":
            trajectory,
    }


    env.close()

    return result


# ============================================================
# GAIN SWEEP
# ============================================================

print("\n")
print("=" * 120)
print("GAIN SWEEP")
print("=" * 120)


results = []


for kp, kd in GAIN_SETS:

    r = run_controller(
        kp,
        kd,
        detailed=False
    )


    results.append(
        r
    )


    status = (
        "🏆"
        if r["success"]
        else
        "✅"
        if not r["failed"]
        else
        "❌"
    )


    print(
        f"{status} "
        f"KP={kp:.4f} | "
        f"KD={kd:.3f} | "
        f"MAX={r['max_drift']:6.2f} | "
        f"FINAL={r['final_drift']:6.2f} | "
        f"PATH={r['path']:6.2f} | "
        f"ALT={r['altitude']:7.2f} | "
        f"VS={r['vertical_speed']:6.2f} | "
        f"N={r['north']:7.2f} | "
        f"E={r['east']:7.2f}"
    )


# ============================================================
# RANK
#
# MAX drift is first priority.
# Then path.
# Then final drift.
# ============================================================

results.sort(
    key=lambda r: (
        r["max_drift"],
        r["path"],
        r["final_drift"]
    )
)


print("\n")
print("=" * 120)
print("BEST DECOUPLED CONTROLLERS")
print("=" * 120)


for i, r in enumerate(
    results,
    start=1
):

    print(
        f"{i:2d}. "
        f"KP={r['kp']:.4f} | "
        f"KD={r['kd']:.3f} | "
        f"MAX={r['max_drift']:6.2f} ft | "
        f"PATH={r['path']:6.2f} ft | "
        f"FINAL={r['final_drift']:6.2f} ft | "
        f"ALT={r['altitude']:7.2f}"
    )


# ============================================================
# DETAILED RE-RUN OF BEST
# ============================================================

best = results[0]


print("\n")
print("=" * 120)
print("DETAILED BEST CONTROLLER RUN")
print("=" * 120)

print(
    "KP =",
    best["kp"]
)

print(
    "KD =",
    best["kd"]
)

print()


best_detailed = run_controller(
    best["kp"],
    best["kd"],
    detailed=True
)


print("\n")
print("=" * 120)
print("FINAL BEST RESULT")
print("=" * 120)

print(
    "SUCCESS   :",
    best_detailed[
        "success"
    ]
)

print(
    "TIME      :",
    round(
        best_detailed["time"],
        2
    ),
    "s"
)

print(
    "ALT       :",
    round(
        best_detailed["altitude"],
        2
    ),
    "ft"
)

print(
    "VS        :",
    round(
        best_detailed[
            "vertical_speed"
        ],
        2
    ),
    "ft/s"
)

print(
    "MAX DRIFT :",
    round(
        best_detailed[
            "max_drift"
        ],
        2
    ),
    "ft"
)

print(
    "FINAL     :",
    round(
        best_detailed[
            "final_drift"
        ],
        2
    ),
    "ft"
)

print(
    "PATH      :",
    round(
        best_detailed[
            "path"
        ],
        2
    ),
    "ft"
)

print(
    "NORTH     :",
    round(
        best_detailed[
            "north"
        ],
        2
    ),
    "ft"
)

print(
    "EAST      :",
    round(
        best_detailed[
            "east"
        ],
        2
    ),
    "ft"
)

print(
    "VN        :",
    round(
        best_detailed[
            "vn"
        ],
        3
    ),
    "ft/s"
)

print(
    "VE        :",
    round(
        best_detailed[
            "ve"
        ],
        3
    ),
    "ft/s"
)

print("=" * 120)
