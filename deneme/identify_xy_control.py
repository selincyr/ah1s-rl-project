import numpy as np

from stable_baselines3 import PPO

from helicopter_env_v2 import HelicopterEnvV2


# ============================================================
# MODEL
#
# Sadece güvenilir collective trajectory üretmek için.
# ============================================================

VERTICAL_MODEL = (
    "models_v2/"
    "AH1S_STAGE1_SUCCESS.zip"
)

model = PPO.load(
    VERTICAL_MODEL
)


# ============================================================
# BASE TRIMS
# ============================================================

BASE_ELEVATOR = -0.15390
BASE_AILERON = 0.19100
BASE_RUDDER = 0.39000


# ============================================================
# PERTURBATIONS
# ============================================================

DELTA_ELEVATOR = 0.004
DELTA_AILERON = 0.004
DELTA_RUDDER = 0.010

PULSE_DURATION = 1.5


# Test at different points of climb
TEST_TIMES = [
    10.0,
    20.0,
    30.0,
    40.0,
]


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    pulse_start,
    channel=None,
    delta=0.0,
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

    pulse_end = (
        pulse_start
        +
        PULSE_DURATION
    )

    start_state = None
    end_state = None


    # Run a little past pulse
    total_time = (
        pulse_end
        +
        0.5
    )

    max_steps = int(
        total_time
        /
        dt
    )


    for step in range(
        max_steps
    ):

        t = (
            step
            *
            dt
        )


        # ====================================================
        # EXISTING PPO -> COLLECTIVE
        # ====================================================

        vertical_obs = (
            env._get_obs()
        )

        action, _ = model.predict(
            vertical_obs,
            deterministic=True
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
            float(action[0])
        )

        collective = float(
            np.clip(
                collective,
                0.590,
                0.650
            )
        )


        # ====================================================
        # BASE CONTROLS
        # ====================================================

        elevator = BASE_ELEVATOR
        aileron = BASE_AILERON
        rudder = BASE_RUDDER


        # ====================================================
        # APPLY PULSE
        # ====================================================

        pulse_active = (
            t >= pulse_start
            and
            t < pulse_end
        )


        if pulse_active:

            if channel == "elevator":

                elevator += delta

            elif channel == "aileron":

                aileron += delta

            elif channel == "rudder":

                rudder += delta


        # ====================================================
        # APPLY CONTROLS
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
        ] = rudder


        # ====================================================
        # JSBSIM
        # ====================================================

        for _ in range(
            env.PHYSICS_STEPS
        ):

            if not env.fdm.run():

                raise RuntimeError(
                    "JSBSim stopped"
                )


        # ====================================================
        # STATE
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


        raw = env._raw_state()


        state = {
            "t":
                t + dt,

            "alt":
                float(
                    raw["altitude"]
                ),

            "vs":
                float(
                    raw["vertical_speed"]
                ),

            "vn":
                vn,

            "ve":
                ve,

            "north":
                north,

            "east":
                east,

            "pitch":
                float(
                    raw["pitch"]
                ),

            "roll":
                float(
                    raw["roll"]
                ),
        }


        # State immediately before perturbation
        if (
            start_state is None
            and
            t >= pulse_start
        ):

            start_state = (
                state.copy()
            )


        # State at end of perturbation
        if (
            end_state is None
            and
            t >= pulse_end
        ):

            end_state = (
                state.copy()
            )


    env.close()


    return {
        "start":
            start_state,

        "end":
            end_state,

        "final":
            state,
    }


# ============================================================
# CENTRAL DIFFERENCE IDENTIFICATION
# ============================================================

print("=" * 120)
print("AH-1S XY CONTROL EFFECTIVENESS IDENTIFICATION")
print("=" * 120)

print()

print(
    "Pulse duration:",
    PULSE_DURATION,
    "s"
)

print(
    "Elevator delta:",
    DELTA_ELEVATOR
)

print(
    "Aileron delta :",
    DELTA_AILERON
)

print(
    "Rudder delta  :",
    DELTA_RUDDER
)

print()


all_results = []


for pulse_time in TEST_TIMES:

    print("\n")
    print("=" * 120)

    print(
        f"TEST POINT: t = "
        f"{pulse_time:.1f} s"
    )

    print("=" * 120)


    # ========================================================
    # BASELINE
    # ========================================================

    baseline = run_case(
        pulse_start=pulse_time,
        channel=None,
        delta=0.0
    )


    b = baseline["end"]


    print(
        "\nBASELINE END:"
    )

    print(
        f"ALT={b['alt']:7.2f} | "
        f"VN={b['vn']:7.3f} | "
        f"VE={b['ve']:7.3f} | "
        f"N={b['north']:7.3f} | "
        f"E={b['east']:7.3f}"
    )


    # ========================================================
    # ELEVATOR +/-
    # ========================================================

    ele_plus = run_case(
        pulse_start=pulse_time,
        channel="elevator",
        delta=+DELTA_ELEVATOR
    )

    ele_minus = run_case(
        pulse_start=pulse_time,
        channel="elevator",
        delta=-DELTA_ELEVATOR
    )


    # ========================================================
    # AILERON +/-
    # ========================================================

    ail_plus = run_case(
        pulse_start=pulse_time,
        channel="aileron",
        delta=+DELTA_AILERON
    )

    ail_minus = run_case(
        pulse_start=pulse_time,
        channel="aileron",
        delta=-DELTA_AILERON
    )


    # ========================================================
    # RUDDER +/-
    # ========================================================

    rud_plus = run_case(
        pulse_start=pulse_time,
        channel="rudder",
        delta=+DELTA_RUDDER
    )

    rud_minus = run_case(
        pulse_start=pulse_time,
        channel="rudder",
        delta=-DELTA_RUDDER
    )


    ep = ele_plus["end"]
    em = ele_minus["end"]

    ap = ail_plus["end"]
    am = ail_minus["end"]

    rp = rud_plus["end"]
    rm = rud_minus["end"]


    # ========================================================
    # CENTRAL DIFFERENCE
    #
    # Change in velocity per unit command.
    # ========================================================

    dVN_dELE = (
        ep["vn"]
        -
        em["vn"]
    ) / (
        2.0
        *
        DELTA_ELEVATOR
    )

    dVE_dELE = (
        ep["ve"]
        -
        em["ve"]
    ) / (
        2.0
        *
        DELTA_ELEVATOR
    )


    dVN_dAIL = (
        ap["vn"]
        -
        am["vn"]
    ) / (
        2.0
        *
        DELTA_AILERON
    )

    dVE_dAIL = (
        ap["ve"]
        -
        am["ve"]
    ) / (
        2.0
        *
        DELTA_AILERON
    )


    dVN_dRUD = (
        rp["vn"]
        -
        rm["vn"]
    ) / (
        2.0
        *
        DELTA_RUDDER
    )

    dVE_dRUD = (
        rp["ve"]
        -
        rm["ve"]
    ) / (
        2.0
        *
        DELTA_RUDDER
    )


    # ========================================================
    # POSITION EFFECT
    # ========================================================

    dN_dELE = (
        ep["north"]
        -
        em["north"]
    ) / (
        2.0
        *
        DELTA_ELEVATOR
    )

    dE_dELE = (
        ep["east"]
        -
        em["east"]
    ) / (
        2.0
        *
        DELTA_ELEVATOR
    )


    dN_dAIL = (
        ap["north"]
        -
        am["north"]
    ) / (
        2.0
        *
        DELTA_AILERON
    )

    dE_dAIL = (
        ap["east"]
        -
        am["east"]
    ) / (
        2.0
        *
        DELTA_AILERON
    )


    # ========================================================
    # 2x2 CYCLIC EFFECTIVENESS MATRIX
    #
    # [ VN ]   [ A B ] [ elevator ]
    # [ VE ] = [ C D ] [ aileron  ]
    # ========================================================

    G = np.array(
        [
            [
                dVN_dELE,
                dVN_dAIL
            ],

            [
                dVE_dELE,
                dVE_dAIL
            ],
        ],
        dtype=np.float64
    )


    determinant = float(
        np.linalg.det(
            G
        )
    )


    try:

        condition = float(
            np.linalg.cond(
                G
            )
        )

    except Exception:

        condition = np.inf


    print("\nCONTROL EFFECT ON VELOCITY")

    print(
        f"dVN/dELE = "
        f"{dVN_dELE:10.3f}"
    )

    print(
        f"dVE/dELE = "
        f"{dVE_dELE:10.3f}"
    )

    print(
        f"dVN/dAIL = "
        f"{dVN_dAIL:10.3f}"
    )

    print(
        f"dVE/dAIL = "
        f"{dVE_dAIL:10.3f}"
    )

    print(
        f"dVN/dRUD = "
        f"{dVN_dRUD:10.3f}"
    )

    print(
        f"dVE/dRUD = "
        f"{dVE_dRUD:10.3f}"
    )


    print(
        "\nCONTROL EFFECT ON POSITION"
    )

    print(
        f"dN/dELE  = "
        f"{dN_dELE:10.3f}"
    )

    print(
        f"dE/dELE  = "
        f"{dE_dELE:10.3f}"
    )

    print(
        f"dN/dAIL  = "
        f"{dN_dAIL:10.3f}"
    )

    print(
        f"dE/dAIL  = "
        f"{dE_dAIL:10.3f}"
    )


    print(
        "\nCYCLIC EFFECTIVENESS MATRIX G"
    )

    print(G)


    print(
        "\ndet(G)  =",
        round(
            determinant,
            6
        )
    )

    print(
        "cond(G) =",
        round(
            condition,
            3
        )
    )


    # ========================================================
    # INVERSE / DECOUPLING MATRIX
    # ========================================================

    if (
        abs(determinant)
        >
        1e-6
    ):

        G_inv = (
            np.linalg.inv(
                G
            )
        )

        print(
            "\nINVERSE EFFECTIVENESS MATRIX"
        )

        print(
            G_inv
        )

    else:

        G_inv = None

        print(
            "\n⚠️ Matrix nearly singular."
        )


    all_results.append(
        {
            "time":
                pulse_time,

            "G":
                G,

            "det":
                determinant,

            "condition":
                condition,

            "G_inv":
                G_inv,
        }
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 120)
print("SUMMARY")
print("=" * 120)


for r in all_results:

    print(
        f"\nt = "
        f"{r['time']:5.1f}s"
    )

    print(
        "G ="
    )

    print(
        r["G"]
    )

    print(
        f"det  = "
        f"{r['det']:.6f}"
    )

    print(
        f"cond = "
        f"{r['condition']:.3f}"
    )


print("\n")
print("=" * 120)
print("IDENTIFICATION COMPLETE")
print("=" * 120)
