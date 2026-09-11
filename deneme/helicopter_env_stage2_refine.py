import numpy as np

from helicopter_env_stage2 import HelicopterEnvStage2


class HelicopterEnvStage2Refine(HelicopterEnvStage2):
    """
    AH-1S Stage 2 Refinement

    Bu environment kalkış eğitmez.

    Reset sırasında:
        rotor hazır
        -> referans controller ile ~300 ft
        -> stabil hover

    PPO görevi:
        300 ft civarında kal
        -> ileri uç
        -> 300 ft forward distance
    """

    TARGET_ALTITUDE = 300.0
    TARGET_FORWARD_DISTANCE = 300.0

    # Hover'dan başladığımız için 60 saniye yeterli.
    MAX_EPISODE_SECONDS = 60.0

    def __init__(self):

        super().__init__()

        self.max_steps = int(
            self.MAX_EPISODE_SECONDS
            /
            self.CONTROL_DT
        )

    # ============================================================
    # REFERENCE ALTITUDE CONTROLLER
    # Sadece reset sırasında kullanılır.
    # PPO reward'una / uçuşuna karışmaz.
    # ============================================================

    def _reference_collective(
        self,
        altitude,
        vertical_speed
    ):

        altitude_error = (
            self.TARGET_ALTITUDE
            -
            altitude
        )

        desired_vs = np.clip(
            0.05 * altitude_error,
            -3.0,
            6.0
        )

        vs_error = (
            desired_vs
            -
            vertical_speed
        )

        collective = (
            0.6085
            +
            0.0058 * desired_vs
            +
            0.0025 * vs_error
        )

        return float(
            np.clip(
                collective,
                0.590,
                0.650
            )
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        # Parent:
        # - JSBSim create
        # - reset00
        # - rotor warmup
        obs, info = super().reset(
            seed=seed,
            options=options
        )

        # --------------------------------------------------------
        # REFERANS CONTROLLER İLE 300 FT HOVER
        # --------------------------------------------------------

        self.phase = 0

        stable_time = 0.0
        max_setup_time = 120.0
        setup_time = 0.0

        # Hover sırasında full pitch stabilization
        self.fdm[
            "ap/afcs/pitch-channel-active-norm"
        ] = 1.0

        self.fdm[
            "ap/afcs/roll-channel-active-norm"
        ] = 1.0

        self.fdm[
            "ap/afcs/yaw-channel-active-norm"
        ] = 1.0

        while setup_time < max_setup_time:

            s = self._raw_state()

            collective = (
                self._reference_collective(
                    s["altitude"],
                    s["vertical_speed"]
                )
            )

            self.fdm[
                "fcs/collective-cmd-norm"
            ] = collective

            # Official hover trim
            self.fdm[
                "fcs/elevator-cmd-norm"
            ] = -0.1558

            self.fdm[
                "fcs/aileron-cmd-norm"
            ] = 0.19095

            self.fdm[
                "fcs/rudder-cmd-norm"
            ] = 0.39

            for _ in range(
                self.PHYSICS_STEPS
            ):

                if not self.fdm.run():

                    raise RuntimeError(
                        "JSBSim reset sırasında durdu."
                    )

            setup_time += (
                self.CONTROL_DT
            )

            s = self._raw_state()

            altitude_error = abs(
                self.TARGET_ALTITUDE
                -
                s["altitude"]
            )

            stable = (
                altitude_error < 5.0
                and
                abs(
                    s["vertical_speed"]
                ) < 0.7
                and
                abs(
                    s["forward_velocity"]
                ) < 3.0
                and
                abs(
                    s["lateral_velocity"]
                ) < 3.0
            )

            if stable:

                stable_time += (
                    self.CONTROL_DT
                )

            else:

                stable_time = 0.0

            # 3 saniye stabil hover
            if stable_time >= 3.0:
                break

        if stable_time < 3.0:

            raise RuntimeError(
                "300 ft başlangıç hover'ı oluşturulamadı."
            )

        # --------------------------------------------------------
        # ARTIK PPO FORWARD FLIGHT FAZINDA
        # --------------------------------------------------------

        self.phase = 1

        self.steps = 0

        self.forward_distance = 0.0

        self.previous_action = np.zeros(
            4,
            dtype=np.float32
        )

        # Daha fazla longitudinal cyclic authority
        self.fdm[
            "ap/afcs/pitch-channel-active-norm"
        ] = 0.5

        # Roll / yaw stabilization devam
        self.fdm[
            "ap/afcs/roll-channel-active-norm"
        ] = 1.0

        self.fdm[
            "ap/afcs/yaw-channel-active-norm"
        ] = 1.0

        s = self._raw_state()

        self.previous_altitude_error = abs(
            self.TARGET_ALTITUDE
            -
            s["altitude"]
        )

        obs = self._get_obs()

        info = {
            **s,

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

    # ============================================================
    # STEP
    # ============================================================

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

        # Parent mapping Phase 1 olduğu için:
        #
        # action[0] -> collective
        # action[1] -> elevator
        #
        # elevator:
        # -1 -> -0.180
        #  0 -> -0.145
        # +1 -> -0.110

        controls = self._apply_action(
            action
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

        vertical_speed = (
            s["vertical_speed"]
        )

        forward_velocity = (
            s["forward_velocity"]
        )

        lateral_velocity = (
            s["lateral_velocity"]
        )

        pitch = s["pitch"]
        roll = s["roll"]

        rotor_rpm = (
            s["rotor_rpm"]
        )

        altitude_error = abs(
            self.TARGET_ALTITUDE
            -
            altitude
        )

        # ========================================================
        # FORWARD DISTANCE
        # ========================================================

        delta_forward = (
            forward_velocity
            *
            self.CONTROL_DT
        )

        self.forward_distance += (
            delta_forward
        )

        # ========================================================
        # REWARD
        # ========================================================

        # Küçük zaman cezası
        reward = -0.01

        # --------------------------------------------------------
        # 1. GERÇEK FORWARD PROGRESS
        # --------------------------------------------------------

        reward += (
            1.00
            *
            delta_forward
        )

        # --------------------------------------------------------
        # 2. ALTITUDE HOLD
        #
        # Stage 2 40k modelinin ana problemi buydu:
        # 331 ft'e çıkıyordu.
        #
        # Bu nedenle altitude penalty daha güçlü.
        # --------------------------------------------------------

        reward -= (
            0.030
            *
            altitude_error
        )

        # 20 ft dışına çıkınca ek ceza
        if altitude_error > 20.0:

            reward -= (
                0.080
                *
                (
                    altitude_error
                    -
                    20.0
                )
            )

        # --------------------------------------------------------
        # 3. VERTICAL SPEED
        # --------------------------------------------------------

        reward -= (
            0.080
            *
            abs(
                vertical_speed
            )
        )

        # --------------------------------------------------------
        # 4. FORWARD SPEED
        #
        # Calibration'da ~15 ft/s çok iyi çıktı.
        # --------------------------------------------------------

        reward -= (
            0.015
            *
            abs(
                forward_velocity
                -
                15.0
            )
        )

        # --------------------------------------------------------
        # 5. LATERAL DRIFT
        # --------------------------------------------------------

        reward -= (
            0.030
            *
            abs(
                lateral_velocity
            )
        )

        # --------------------------------------------------------
        # 6. ATTITUDE
        # --------------------------------------------------------

        reward -= (
            0.15
            *
            abs(
                pitch
            )
        )

        reward -= (
            0.15
            *
            abs(
                roll
            )
        )

        # --------------------------------------------------------
        # 7. ACTION SMOOTHNESS
        # --------------------------------------------------------

        action_change = np.mean(
            np.square(
                action
                -
                self.previous_action
            )
        )

        reward -= (
            0.030
            *
            action_change
        )

        self.previous_action = (
            action.copy()
        )

        # ========================================================
        # SUCCESS
        # ========================================================

        success = (
            self.forward_distance
            >=
            self.TARGET_FORWARD_DISTANCE

            and
            altitude_error < 15.0

            and
            abs(
                vertical_speed
            ) < 2.0

            and
            abs(
                lateral_velocity
            ) < 5.0

            and
            abs(
                pitch
            ) < 0.25

            and
            abs(
                roll
            ) < 0.25
        )

        terminated = False

        if success:

            # Success mutlaka en değerli sonuç olsun.
            reward += 750.0

            terminated = True

        # ========================================================
        # SAFETY / FAILURE
        # ========================================================

        # Forward-flight training başladığı için
        # 300 ft çevresinde tutulmasını istiyoruz.

        if (
            altitude < 240.0
            or
            altitude > 360.0
        ):

            reward -= 300.0
            terminated = True

        if (
            abs(pitch) > 0.60
            or
            abs(roll) > 0.60
        ):

            reward -= 300.0
            terminated = True

        if rotor_rpm < 260.0:

            reward -= 300.0
            terminated = True

        if (
            self.forward_distance
            <
            -100.0
        ):

            reward -= 200.0
            terminated = True

        # 300 ft'i geçmiş ama altitude tutamayıp
        # çok fazla ileri kaçtıysa başarısız kabul et.
        if (
            self.forward_distance
            >
            450.0
            and
            not success
        ):

            reward -= 150.0
            terminated = True

        if not jsbsim_ok:

            reward -= 300.0
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
                1,

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
                bool(success),
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )
