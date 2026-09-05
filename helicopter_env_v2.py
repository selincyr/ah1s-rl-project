import numpy as np
import jsbsim
import gymnasium as gym
from gymnasium import spaces


class HelicopterEnvV2(gym.Env):
    """
    AH-1S / JSBSim - Stage 1

    Görev:
        Motor/rotor çalışır durumda
        -> takeoff
        -> 300 ft'e çık
        -> 300 ft civarında stabil hover yap

    Stage 1'de:
        PPO sadece collective kontrol ediyor.

    Elevator / aileron / rudder:
        Resmi JSBSim AH-1S hover trim değerlerinde tutuluyor.

    Daha sonraki stage'lerde:
        Forward flight
        Descent
        Landing
    için cyclic ve pedal kontrollerini PPO'ya açacağız.
    """

    metadata = {"render_modes": []}

    # ---------------------------------------------------------
    # SIMULATION SETTINGS
    # ---------------------------------------------------------

    JSBSIM_DT = 0.0075

    # Her PPO action için 10 fizik step'i
    # 0.0075 * 10 = 0.075 saniye
    # PPO control frequency ~= 13.33 Hz
    PHYSICS_STEPS = 10

    TARGET_ALTITUDE = 300.0

    MAX_EPISODE_SECONDS = 120.0

    # ---------------------------------------------------------
    # INIT
    # ---------------------------------------------------------

    def __init__(self):
        super().__init__()

        # -----------------------------------------------------
        # ACTION SPACE
        # -----------------------------------------------------
        #
        # PPO için dört boyut bırakıyoruz:
        #
        # action[0] -> collective
        # action[1] -> ileride elevator
        # action[2] -> ileride aileron
        # action[3] -> ileride rudder
        #
        # Stage 1'de yalnızca action[0] aktif.
        #

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # OBSERVATION SPACE
        # -----------------------------------------------------
        #
        # 0 altitude error
        # 1 altitude
        # 2 vertical speed
        # 3 forward velocity
        # 4 lateral velocity
        # 5 pitch
        # 6 roll
        # 7 roll rate p
        # 8 pitch rate q
        # 9 yaw rate r
        # 10 heading error
        # 11 rotor RPM error
        #

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.fdm = None

        self.steps = 0

        self.airborne = False

        self.hover_steps = 0

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
            (
                self.JSBSIM_DT
                *
                self.PHYSICS_STEPS
            )
        )

    # =========================================================
    # JSBSIM INITIALIZATION
    # =========================================================

    def _create_fdm(self):

        self.fdm = jsbsim.FGFDMExec(None)

        self.fdm.set_debug_level(0)

        # AH-1S aircraft model
        if not self.fdm.load_model("ah1s"):
            raise RuntimeError(
                "AH-1S modeli yüklenemedi."
            )

        # Official initial conditions
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

        # -----------------------------------------------------
        # OFFICIAL AH-1S SETUP
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # INITIAL CONTROLS
        # -----------------------------------------------------

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

        # Governor initially off
        self.fdm[
            "fcs/rpm-governor-active-norm"
        ] = 0.0

        # AFCS initially off
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

        duration = 5.0

        while True:

            t = float(
                self.fdm[
                    "simulation/sim-time-sec"
                ]
            )

            elapsed = (
                t - start_time
            )

            # Governor 5 saniyede 0 -> 1
            governor = np.clip(
                elapsed / duration,
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

            # Resmi testte gördüğümüz çalışma bölgesi
            if (
                elapsed >= 7.0
                and rpm >= 320.0
            ):
                break

            if elapsed > 12.0:
                raise RuntimeError(
                    f"Rotor warmup başarısız. RPM={rpm}"
                )

        # Governor sürekli aktif
        self.fdm[
            "fcs/rpm-governor-active-norm"
        ] = 1.0

        # -----------------------------------------------------
        # AFCS
        # -----------------------------------------------------
        #
        # Altitude AFCS kullanmıyoruz.
        #
        # PPO altitude kontrolünü kendisi öğreniyor.
        #
        # Roll / pitch / yaw damping ise AH-1S'in
        # kendi low-level stabilizasyonuna bırakılıyor.
        #

        self.fdm[
            "ap/afcs/yaw-channel-active-norm"
        ] = 1.0

        self.fdm[
            "ap/afcs/pitch-channel-active-norm"
        ] = 1.0

        self.fdm[
            "ap/afcs/roll-channel-active-norm"
        ] = 1.0

        # Resmi AH-1S trim değerleri
        self.fdm[
            "ap/afcs/manual/phi-trim-rad"
        ] = -0.049254

        self.fdm[
            "ap/afcs/manual/theta-trim-rad"
        ] = -0.006428

    # =========================================================
    # ANGLE HELPER
    # =========================================================

    @staticmethod
    def _wrap_angle(angle):

        return (
            (angle + np.pi)
            %
            (2.0 * np.pi)
        ) - np.pi

    # =========================================================
    # RAW STATE
    # =========================================================

    def _raw_state(self):

        altitude = float(
            self.fdm[
                "position/h-agl-ft"
            ]
        )

        vertical_speed = float(
            self.fdm[
                "velocities/h-dot-fps"
            ]
        )

        forward_velocity = float(
            self.fdm[
                "velocities/u-aero-fps"
            ]
        )

        lateral_velocity = float(
            self.fdm[
                "velocities/v-aero-fps"
            ]
        )

        pitch = float(
            self.fdm[
                "attitude/pitch-rad"
            ]
        )

        roll = float(
            self.fdm[
                "attitude/roll-rad"
            ]
        )

        heading = float(
            self.fdm[
                "attitude/heading-true-rad"
            ]
        )

        p_rate = float(
            self.fdm[
                "velocities/p-rad_sec"
            ]
        )

        q_rate = float(
            self.fdm[
                "velocities/q-rad_sec"
            ]
        )

        r_rate = float(
            self.fdm[
                "velocities/r-rad_sec"
            ]
        )

        rotor_rpm = float(
            self.fdm[
                "propulsion/engine/rotor-rpm"
            ]
        )

        return {
            "altitude":
                altitude,

            "vertical_speed":
                vertical_speed,

            "forward_velocity":
                forward_velocity,

            "lateral_velocity":
                lateral_velocity,

            "pitch":
                pitch,

            "roll":
                roll,

            "heading":
                heading,

            "p_rate":
                p_rate,

            "q_rate":
                q_rate,

            "r_rate":
                r_rate,

            "rotor_rpm":
                rotor_rpm,
        }

    # =========================================================
    # NORMALIZED OBSERVATION
    # =========================================================

    def _get_obs(self):

        s = self._raw_state()

        altitude_error = (
            self.TARGET_ALTITUDE
            -
            s["altitude"]
        )

        heading_error = (
            self._wrap_angle(
                s["heading"]
                -
                np.pi
            )
        )

        obs = np.array(
            [
                # Altitude error
                altitude_error / 300.0,

                # Altitude
                s["altitude"] / 300.0,

                # Vertical speed
                s["vertical_speed"] / 10.0,

                # Forward / lateral velocity
                s["forward_velocity"] / 30.0,
                s["lateral_velocity"] / 30.0,

                # Attitude
                s["pitch"] / 0.35,
                s["roll"] / 0.35,

                # Angular rates
                s["p_rate"] / 1.0,
                s["q_rate"] / 1.0,
                s["r_rate"] / 1.0,

                # Heading
                heading_error / np.pi,

                # Rotor RPM error
                (
                    s["rotor_rpm"]
                    -
                    320.0
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
        # STAGE 1
        # -----------------------------------------------------
        #
        # PPO yalnızca collective öğreniyor.
        #
        # Kalibrasyon testlerimiz:
        #
        # 0.590 -> düşük collective
        # 0.609 -> hover civarı
        # 0.635 -> orta climb
        # 0.640 -> güçlü climb
        # 0.645 -> ~6 ft/s climb
        #
        # PPO:
        #
        # action[0] = -1 -> 0.590
        # action[0] =  0 -> 0.620
        # action[0] = +1 -> 0.650
        #

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

        # -----------------------------------------------------
        # OFFICIAL AH-1S HOVER TRIM
        # -----------------------------------------------------

        elevator = -0.1558

        aileron = 0.19095

        rudder = 0.39

        # Apply to JSBSim
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

        self.hover_steps = 0

        self.previous_action = (
            np.zeros(
                4,
                dtype=np.float32
            )
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

            "target_altitude":
                self.TARGET_ALTITUDE,

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

        # -----------------------------------------------------
        # PHYSICS STEPS
        # -----------------------------------------------------

        for _ in range(
            self.PHYSICS_STEPS
        ):

            if not self.fdm.run():

                jsbsim_ok = False

                break

        state = self._raw_state()

        altitude = (
            state["altitude"]
        )

        vertical_speed = (
            state["vertical_speed"]
        )

        pitch = (
            state["pitch"]
        )

        roll = (
            state["roll"]
        )

        p_rate = (
            state["p_rate"]
        )

        q_rate = (
            state["q_rate"]
        )

        r_rate = (
            state["r_rate"]
        )

        forward_velocity = (
            state["forward_velocity"]
        )

        lateral_velocity = (
            state["lateral_velocity"]
        )

        rotor_rpm = (
            state["rotor_rpm"]
        )

        # -----------------------------------------------------
        # AIRBORNE DETECTION
        # -----------------------------------------------------

        if altitude > 10.0:

            self.airborne = True

        # =====================================================
        # REWARD
        # =====================================================

        altitude_error = abs(
            self.TARGET_ALTITUDE
            -
            altitude
        )

        # -----------------------------------------------------
        # 1. ALTITUDE PROGRESS
        # -----------------------------------------------------

        progress = (
            self.previous_altitude_error
            -
            altitude_error
        )

        progress_reward = (
            2.0
            *
            progress
        )

        self.previous_altitude_error = (
            altitude_error
        )

        # -----------------------------------------------------
        # 2. ABSOLUTE ALTITUDE REWARD
        # -----------------------------------------------------

        altitude_reward = np.exp(
            -altitude_error
            /
            80.0
        )

        # -----------------------------------------------------
        # 3. DESIRED VERTICAL SPEED PROFILE
        # -----------------------------------------------------
        #
        # Çalışan reference controller'dan aldığımız profil:
        #
        # Hedeften uzakken:
        #     +6 ft/s
        #
        # Hedefe yaklaştıkça:
        #     yavaşlar
        #
        # 300 ft:
        #     desired VS = 0
        #

        signed_error = (
            self.TARGET_ALTITUDE
            -
            altitude
        )

        desired_vs = np.clip(
            0.05
            *
            signed_error,
            -3.0,
            6.0
        )

        vs_error = abs(
            vertical_speed
            -
            desired_vs
        )

        vertical_reward = np.exp(
            -vs_error
            /
            3.0
        )

        # -----------------------------------------------------
        # 4. ATTITUDE PENALTY
        # -----------------------------------------------------

        attitude_penalty = (
            1.5
            *
            abs(pitch)
            +
            1.5
            *
            abs(roll)
        )

        # -----------------------------------------------------
        # 5. ANGULAR RATE PENALTY
        # -----------------------------------------------------

        rate_penalty = (
            0.20
            *
            abs(p_rate)
            +
            0.20
            *
            abs(q_rate)
            +
            0.10
            *
            abs(r_rate)
        )

        # -----------------------------------------------------
        # 6. HORIZONTAL DRIFT PENALTY
        # -----------------------------------------------------

        drift_penalty = (
            0.01
            *
            abs(
                forward_velocity
            )
            +
            0.01
            *
            abs(
                lateral_velocity
            )
        )

        # -----------------------------------------------------
        # 7. ROTOR RPM PENALTY
        # -----------------------------------------------------

        rpm_penalty = (
            0.005
            *
            abs(
                rotor_rpm
                -
                320.0
            )
        )

        # -----------------------------------------------------
        # 8. ACTION SMOOTHNESS
        # -----------------------------------------------------

        action_change = np.mean(
            np.square(
                action
                -
                self.previous_action
            )
        )

        smoothness_penalty = (
            0.05
            *
            action_change
        )

        self.previous_action = (
            action.copy()
        )

        # -----------------------------------------------------
        # BASE REWARD
        # -----------------------------------------------------

        reward = (
            progress_reward
            +
            altitude_reward
            +
            vertical_reward
            -
            attitude_penalty
            -
            rate_penalty
            -
            drift_penalty
            -
            rpm_penalty
            -
            smoothness_penalty
        )

        # =====================================================
        # TARGET BONUSES
        # =====================================================

        inside_20 = (
            altitude_error
            <
            20.0
        )

        inside_10 = (
            altitude_error
            <
            10.0
        )

        stable_hover = (
            altitude_error
            <
            10.0
            and
            abs(
                vertical_speed
            )
            <
            1.0
            and
            abs(
                forward_velocity
            )
            <
            3.0
            and
            abs(
                lateral_velocity
            )
            <
            3.0
            and
            abs(
                pitch
            )
            <
            0.12
            and
            abs(
                roll
            )
            <
            0.12
        )

        if inside_20:

            reward += 1.0

        if inside_10:

            reward += 2.0

        if stable_hover:

            reward += 3.0

            self.hover_steps += 1

        else:

            self.hover_steps = 0

        # -----------------------------------------------------
        # 10 SECOND HOVER SUCCESS
        # -----------------------------------------------------

        success_steps = int(
            10.0
            /
            (
                self.JSBSIM_DT
                *
                self.PHYSICS_STEPS
            )
        )

        success = (
            self.hover_steps
            >=
            success_steps
        )

        terminated = False

        # -----------------------------------------------------
        # SUCCESS
        # -----------------------------------------------------

        if success:

            reward += 100.0

            terminated = True

        # =====================================================
        # FAILURE CONDITIONS
        # =====================================================

        # Havalandıktan sonra yere düşme
        if (
            self.airborne
            and
            altitude < 5.5
        ):

            reward -= 100.0

            terminated = True

        # Çok tehlikeli attitude
        if (
            abs(pitch) > 1.05
            or
            abs(roll) > 1.05
        ):

            reward -= 100.0

            terminated = True

        # Çok yüksek / fiziksel olarak saçma altitude
        if (
            altitude < 0.0
            or
            altitude > 500.0
        ):

            reward -= 100.0

            terminated = True

        # Rotor RPM problemi
        if rotor_rpm < 260.0:

            reward -= 100.0

            terminated = True

        # JSBSim kendi kendine durdu
        if not jsbsim_ok:

            reward -= 100.0

            terminated = True

        # -----------------------------------------------------
        # TIME LIMIT
        # -----------------------------------------------------

        truncated = (
            self.steps
            >=
            self.max_steps
        )

        # -----------------------------------------------------
        # OBSERVATION
        # -----------------------------------------------------

        obs = self._get_obs()

        # -----------------------------------------------------
        # INFO
        # -----------------------------------------------------

        info = {
            **state,

            "target_altitude":
                self.TARGET_ALTITUDE,

            "altitude_error":
                altitude_error,

            "desired_vertical_speed":
                float(
                    desired_vs
                ),

            "collective":
                controls[0],

            "elevator":
                controls[1],

            "aileron":
                controls[2],

            "rudder":
                controls[3],

            "hover_steps":
                self.hover_steps,

            "success":
                success,

            "sim_time":
                float(
                    self.fdm[
                        "simulation/sim-time-sec"
                    ]
                ),
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self.fdm = None
