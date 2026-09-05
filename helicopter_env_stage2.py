import numpy as np
import jsbsim
import gymnasium as gym
from gymnasium import spaces


class HelicopterEnvStage2(gym.Env):
    """
    AH-1S / JSBSim - Stage 2

    Görev:
        Motor/rotor hazır
        -> kalkış
        -> 300 ft
        -> 10 saniye stabil hover
        -> kontrollü ileri uçuş
        -> 300 ft ileri mesafe

    Stage 1:
        PPO collective kontrol eder.
        Elevator hover triminde sabit.

    Stage 2:
        PPO collective + longitudinal cyclic kontrol eder.
        Roll / yaw düşük seviye AFCS ile stabilize edilir.
    """

    metadata = {"render_modes": []}

    JSBSIM_DT = 0.0075
    PHYSICS_STEPS = 10

    CONTROL_DT = JSBSIM_DT * PHYSICS_STEPS

    TARGET_ALTITUDE = 300.0
    TARGET_FORWARD_DISTANCE = 300.0

    MAX_EPISODE_SECONDS = 150.0

    def __init__(self):
        super().__init__()

        # action[0] = collective
        # action[1] = longitudinal cyclic / elevator
        # action[2] = ileride lateral cyclic
        # action[3] = ileride pedal
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation shape'i Stage 1 ile aynı tutuyoruz.
        # Böylece Stage 1 PPO modelini Stage 2'ye aktarabiliriz.
        #
        # 0 altitude error
        # 1 altitude
        # 2 vertical speed
        # 3 forward velocity
        # 4 lateral velocity
        # 5 pitch
        # 6 roll
        # 7 roll rate
        # 8 pitch rate
        # 9 yaw rate
        # 10 forward distance progress
        # 11 rotor RPM error

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.fdm = None

        self.steps = 0
        self.airborne = False

        # 0 = climb + hover
        # 1 = forward flight
        self.phase = 0

        self.hover_steps = 0

        self.forward_distance = 0.0

        self.previous_altitude_error = (
            self.TARGET_ALTITUDE
        )

        self.previous_action = np.zeros(
            4,
            dtype=np.float32
        )

        self.max_steps = int(
            self.MAX_EPISODE_SECONDS
            /
            self.CONTROL_DT
        )

    # =========================================================
    # JSBSIM
    # =========================================================

    def _create_fdm(self):

        self.fdm = jsbsim.FGFDMExec(None)

        self.fdm.set_debug_level(0)

        if not self.fdm.load_model("ah1s"):
            raise RuntimeError(
                "AH-1S modeli yüklenemedi."
            )

        if not self.fdm.load_ic(
            "reset00.xml",
            True
        ):
            raise RuntimeError(
                "reset00.xml yüklenemedi."
            )

        self.fdm.set_dt(
            self.JSBSIM_DT
        )

        # Official AH-1S setup
        self.fdm[
            "ap/afcs/psi-trim-rad"
        ] = np.pi

        self.fdm[
            "propulsion/tank[0]/contents-lbs"
        ] = 0.0

        self.fdm[
            "propulsion/tank[1]/contents-lbs"
        ] = 0.0

        self.fdm[
            "aero/setup/downwash-enable"
        ] = 1.0

        self.fdm[
            "aero/setup/Nr_limiter"
        ] = 0.05

        self.fdm[
            "fcs/adj/collective-profile"
        ] = 0.0

        self.fdm[
            "fcs/adj/center-sensitivity"
        ] = 1.0

        self.fdm[
            "fcs/collective-cmd-norm"
        ] = 0.0

        self.fdm[
            "fcs/elevator-cmd-norm"
        ] = 0.0

        self.fdm[
            "fcs/aileron-cmd-norm"
        ] = 0.0

        self.fdm[
            "fcs/rudder-cmd-norm"
        ] = 0.0

        self.fdm[
            "fcs/rpm-governor-active-norm"
        ] = 0.0

        self.fdm[
            "ap/afcs/yaw-channel-active-norm"
        ] = 0.0

        self.fdm[
            "ap/afcs/pitch-channel-active-norm"
        ] = 0.0

        self.fdm[
            "ap/afcs/roll-channel-active-norm"
        ] = 0.0

        self.fdm.run_ic()

    # =========================================================
    # ROTOR WARMUP
    # =========================================================

    def _warmup_rotor(self):

        start_time = float(
            self.fdm[
                "simulation/sim-time-sec"
            ]
        )

        while True:

            t = float(
                self.fdm[
                    "simulation/sim-time-sec"
                ]
            )

            elapsed = (
                t - start_time
            )

            governor = np.clip(
                elapsed / 5.0,
                0.0,
                1.0
            )

            self.fdm[
                "fcs/rpm-governor-active-norm"
            ] = float(governor)

            self.fdm.run()

            rpm = float(
                self.fdm[
                    "propulsion/engine/rotor-rpm"
                ]
            )

            if (
                elapsed >= 7.0
                and rpm >= 320.0
            ):
                break

            if elapsed > 12.0:

                raise RuntimeError(
                    f"Rotor warmup başarısız. RPM={rpm}"
                )

        self.fdm[
            "fcs/rpm-governor-active-norm"
        ] = 1.0

        # Initial AFCS
        self.fdm[
            "ap/afcs/yaw-channel-active-norm"
        ] = 1.0

        self.fdm[
            "ap/afcs/pitch-channel-active-norm"
        ] = 1.0

        self.fdm[
            "ap/afcs/roll-channel-active-norm"
        ] = 1.0

        self.fdm[
            "ap/afcs/manual/phi-trim-rad"
        ] = -0.049254

        self.fdm[
            "ap/afcs/manual/theta-trim-rad"
        ] = -0.006428

    # =========================================================
    # RAW STATE
    # =========================================================

    def _raw_state(self):

        return {

            "altitude":
                float(
                    self.fdm[
                        "position/h-agl-ft"
                    ]
                ),

            "vertical_speed":
                float(
                    self.fdm[
                        "velocities/h-dot-fps"
                    ]
                ),

            "forward_velocity":
                float(
                    self.fdm[
                        "velocities/u-aero-fps"
                    ]
                ),

            "lateral_velocity":
                float(
                    self.fdm[
                        "velocities/v-aero-fps"
                    ]
                ),

            "pitch":
                float(
                    self.fdm[
                        "attitude/pitch-rad"
                    ]
                ),

            "roll":
                float(
                    self.fdm[
                        "attitude/roll-rad"
                    ]
                ),

            "p_rate":
                float(
                    self.fdm[
                        "velocities/p-rad_sec"
                    ]
                ),

            "q_rate":
                float(
                    self.fdm[
                        "velocities/q-rad_sec"
                    ]
                ),

            "r_rate":
                float(
                    self.fdm[
                        "velocities/r-rad_sec"
                    ]
                ),

            "rotor_rpm":
                float(
                    self.fdm[
                        "propulsion/engine/rotor-rpm"
                    ]
                ),
        }

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_obs(self):

        s = self._raw_state()

        altitude_error = (
            self.TARGET_ALTITUDE
            -
            s["altitude"]
        )

        obs = np.array(
            [
                altitude_error / 300.0,

                s["altitude"] / 300.0,

                s["vertical_speed"] / 10.0,

                s["forward_velocity"] / 30.0,

                s["lateral_velocity"] / 30.0,

                s["pitch"] / 0.35,

                s["roll"] / 0.35,

                s["p_rate"],

                s["q_rate"],

                s["r_rate"],

                (
                    self.forward_distance
                    /
                    self.TARGET_FORWARD_DISTANCE
                ),

                (
                    s["rotor_rpm"] - 320.0
                ) / 30.0,
            ],
            dtype=np.float32
        )

        return np.clip(
            obs,
            -5.0,
            5.0
        ).astype(
            np.float32
        )

    # =========================================================
    # ACTION MAPPING
    # =========================================================

    def _apply_action(
        self,
        action
    ):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # -----------------------------------------------------
        # COLLECTIVE
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # PHASE 0
        # Takeoff + hover
        # -----------------------------------------------------

        if self.phase == 0:

            elevator = -0.1558

            # Full pitch stabilization
            self.fdm[
                "ap/afcs/pitch-channel-active-norm"
            ] = 1.0

        # -----------------------------------------------------
        # PHASE 1
        # Forward flight
        # -----------------------------------------------------

        else:

            # Calibration:
            #
            # -0.18 -> ~0 ft/s
            # -0.1558 -> ~8 ft/s
            # -0.13 -> ~16 ft/s
            # -0.11 -> ~22 ft/s
            #
            # PPO action:
            #
            # -1 -> -0.18
            #  0 -> -0.145
            # +1 -> -0.11

            elevator = (
                -0.145
                +
                0.035
                *
                float(action[1])
            )

            elevator = float(
                np.clip(
                    elevator,
                    -0.180,
                    -0.110
                )
            )

            # Forward flight'ta PPO'ya
            # daha fazla pitch authority ver.
            self.fdm[
                "ap/afcs/pitch-channel-active-norm"
            ] = 0.5

        # Roll/yaw trim sabit
        aileron = 0.19095
        rudder = 0.39

        self.fdm[
            "fcs/collective-cmd-norm"
        ] = collective

        self.fdm[
            "fcs/elevator-cmd-norm"
        ] = elevator

        self.fdm[
            "fcs/aileron-cmd-norm"
        ] = aileron

        self.fdm[
            "fcs/rudder-cmd-norm"
        ] = rudder

        return (
            collective,
            elevator,
            aileron,
            rudder
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.steps = 0

        self.airborne = False

        self.phase = 0

        self.hover_steps = 0

        self.forward_distance = 0.0

        self.previous_action = np.zeros(
            4,
            dtype=np.float32
        )

        self._create_fdm()

        self._warmup_rotor()

        state = self._raw_state()

        self.previous_altitude_error = abs(
            self.TARGET_ALTITUDE
            -
            state["altitude"]
        )

        obs = self._get_obs()

        info = {
            **state,

            "phase":
                self.phase,

            "forward_distance":
                self.forward_distance,

            "success":
                False,
        }

        return (
            obs,
            info
        )

    # =========================================================
    # STEP
    # =========================================================

    def step(
        self,
        action
    ):

        self.steps += 1

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        controls = (
            self._apply_action(
                action
            )
        )

        jsbsim_ok = True

        for _ in range(
            self.PHYSICS_STEPS
        ):

            if not self.fdm.run():

                jsbsim_ok = False
                break

        s = self._raw_state()

        altitude = s["altitude"]
        vertical_speed = s["vertical_speed"]

        forward_velocity = (
            s["forward_velocity"]
        )

        lateral_velocity = (
            s["lateral_velocity"]
        )

        pitch = s["pitch"]
        roll = s["roll"]

        rotor_rpm = s["rotor_rpm"]

        if altitude > 10.0:
            self.airborne = True

        altitude_error = abs(
            self.TARGET_ALTITUDE
            -
            altitude
        )

        # Small time penalty:
        # ajan episode'u uzatıp reward toplayamasın.
        reward = -0.02

        success = False

        # =====================================================
        # PHASE 0
        # TAKEOFF + 300 FT + HOVER
        # =====================================================

        if self.phase == 0:

            signed_error = (
                self.TARGET_ALTITUDE
                -
                altitude
            )

            desired_vs = np.clip(
                0.05 * signed_error,
                -3.0,
                6.0
            )

            progress = (
                self.previous_altitude_error
                -
                altitude_error
            )

            self.previous_altitude_error = (
                altitude_error
            )

            # Gerçek progress ana reward
            reward += (
                3.0 * progress
            )

            reward -= (
                0.05
                *
                abs(
                    vertical_speed
                    -
                    desired_vs
                )
            )

            reward -= (
                0.01
                *
                abs(
                    forward_velocity
                )
            )

            reward -= (
                0.01
                *
                abs(
                    lateral_velocity
                )
            )

            stable_hover = (
                altitude_error < 10.0
                and
                abs(vertical_speed) < 1.0
                and
                abs(forward_velocity) < 3.0
                and
                abs(lateral_velocity) < 3.0
            )

            if stable_hover:

                self.hover_steps += 1

                reward += 0.10

            else:

                self.hover_steps = 0

            required_hover_steps = int(
                10.0
                /
                self.CONTROL_DT
            )

            # ---------------------------------------------
            # PHASE TRANSITION
            # ---------------------------------------------

            if (
                self.hover_steps
                >=
                required_hover_steps
            ):

                self.phase = 1

                self.forward_distance = 0.0

                self.hover_steps = 0

                reward += 100.0

        # =====================================================
        # PHASE 1
        # FORWARD FLIGHT
        # =====================================================

        else:

            # Signed forward progress
            delta_forward = (
                forward_velocity
                *
                self.CONTROL_DT
            )

            self.forward_distance += (
                delta_forward
            )

            # Ana ödül:
            # gerçekten ileri gidilen mesafe
            reward += (
                0.50
                *
                delta_forward
            )

            # 300 ft irtifayı koru
            reward -= (
                0.020
                *
                altitude_error
            )

            # Dikey hız küçük olsun
            reward -= (
                0.050
                *
                abs(
                    vertical_speed
                )
            )

            # Yaklaşık 15 ft/s hedef forward speed
            reward -= (
                0.010
                *
                abs(
                    forward_velocity
                    -
                    15.0
                )
            )

            # Lateral drift istemiyoruz
            reward -= (
                0.020
                *
                abs(
                    lateral_velocity
                )
            )

            # Aşırı attitude istemiyoruz
            reward -= (
                0.20
                *
                abs(
                    pitch
                )
            )

            reward -= (
                0.20
                *
                abs(
                    roll
                )
            )

            # ---------------------------------------------
            # STAGE 2 SUCCESS
            # ---------------------------------------------

            if (
                self.forward_distance
                >=
                self.TARGET_FORWARD_DISTANCE

                and altitude_error < 20.0

                and abs(
                    vertical_speed
                ) < 3.0

                and abs(
                    lateral_velocity
                ) < 5.0

                and abs(
                    pitch
                ) < 0.25

                and abs(
                    roll
                ) < 0.25
            ):

                success = True

                reward += 250.0

        # =====================================================
        # ACTION SMOOTHNESS
        # =====================================================

        action_change = np.mean(
            np.square(
                action
                -
                self.previous_action
            )
        )

        reward -= (
            0.02
            *
            action_change
        )

        self.previous_action = (
            action.copy()
        )

        # =====================================================
        # TERMINATION
        # =====================================================

        terminated = False

        if success:
            terminated = True

        if (
            self.airborne
            and
            altitude < 5.5
        ):

            reward -= 200.0
            terminated = True

        if (
            altitude < 0.0
            or
            altitude > 500.0
        ):

            reward -= 200.0
            terminated = True

        if (
            abs(pitch) > 1.05
            or
            abs(roll) > 1.05
        ):

            reward -= 200.0
            terminated = True

        if rotor_rpm < 260.0:

            reward -= 200.0
            terminated = True

        if self.forward_distance < -100.0:

            reward -= 100.0
            terminated = True

        if not jsbsim_ok:

            reward -= 200.0
            terminated = True

        truncated = (
            self.steps
            >=
            self.max_steps
        )

        obs = self._get_obs()

        info = {
            **s,

            "phase":
                self.phase,

            "target_altitude":
                self.TARGET_ALTITUDE,

            "altitude_error":
                altitude_error,

            "forward_distance":
                self.forward_distance,

            "target_forward_distance":
                self.TARGET_FORWARD_DISTANCE,

            "collective":
                controls[0],

            "elevator":
                controls[1],

            "aileron":
                controls[2],

            "rudder":
                controls[3],

            "success":
                success,
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )

    def close(self):

        self.fdm = None
