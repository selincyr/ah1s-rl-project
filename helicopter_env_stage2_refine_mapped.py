from __future__ import annotations

import numpy as np

from helicopter_env_stage2_refine import (
    HelicopterEnvStage2Refine,
)


class HelicopterEnvStage2RefineMapped(
    HelicopterEnvStage2Refine
):
    """
    Stage-2 Refine with repaired lateral/yaw action mapping.

    IMPORTANT:
    - The original Stage-2 _apply_action() remains responsible for
      collective/elevator behavior.
    - After the original mapping runs, this class adds bounded physical
      residuals for PPO action[2] (aileron) and action[3] (rudder).
    - The residual is applied around whatever aileron/rudder trim the
      original Stage-2 environment selected on that step.

    This preserves the already-working Stage-2 collective/elevator
    semantics while making all four PPO outputs physically live.
    """

    DEFAULT_AILERON_SCALE = 0.026
    DEFAULT_RUDDER_SCALE = 0.040

    def __init__(
        self,
        aileron_scale: float = DEFAULT_AILERON_SCALE,
        rudder_scale: float = DEFAULT_RUDDER_SCALE,
    ):
        super().__init__()

        self.mapped_aileron_scale = float(
            aileron_scale
        )

        self.mapped_rudder_scale = float(
            rudder_scale
        )

        self.last_base_aileron = float("nan")
        self.last_base_rudder = float("nan")
        self.last_mapped_aileron = float("nan")
        self.last_mapped_rudder = float("nan")


    @staticmethod
    def _replace_return_controls(
        controls,
        aileron,
        rudder,
    ):
        """
        Keep the parent method's return type whenever possible, but
        replace the lateral/yaw values so step() info stays truthful.
        """

        if isinstance(
            controls,
            dict,
        ):
            updated = dict(
                controls
            )

            updated[
                "aileron"
            ] = float(
                aileron
            )

            updated[
                "rudder"
            ] = float(
                rudder
            )

            return updated

        if isinstance(
            controls,
            np.ndarray,
        ):
            updated = (
                controls.copy()
            )

            if (
                updated.ndim == 1
                and
                updated.shape[0] >= 4
            ):
                updated[2] = float(
                    aileron
                )

                updated[3] = float(
                    rudder
                )

            return updated

        if isinstance(
            controls,
            list,
        ):
            updated = list(
                controls
            )

            if len(
                updated
            ) >= 4:
                updated[2] = float(
                    aileron
                )

                updated[3] = float(
                    rudder
                )

            return updated

        if isinstance(
            controls,
            tuple,
        ):
            updated = list(
                controls
            )

            if len(
                updated
            ) >= 4:
                updated[2] = float(
                    aileron
                )

                updated[3] = float(
                    rudder
                )

            return tuple(
                updated
            )

        # If the parent returns None or a custom object, leave it alone.
        # The FDM commands themselves have still been repaired.
        return controls


    def _apply_action(
        self,
        action,
    ):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        action = np.clip(
            action,
            -1.0,
            +1.0,
        )

        if action.shape[0] < 4:
            raise ValueError(
                "Stage-2 mapped environment requires 4 actions."
            )

        # ------------------------------------------------------------
        # Preserve the exact original Stage-2 collective/elevator
        # mapping and its trim calculations.
        # ------------------------------------------------------------

        controls = super()._apply_action(
            action
        )

        # The diagnosis proved that the original environment leaves
        # these at fixed trim values independent of action[2]/[3].
        # We use those values as the dynamic base trim for this step.
        base_aileron = float(
            self.fdm[
                "fcs/aileron-cmd-norm"
            ]
        )

        base_rudder = float(
            self.fdm[
                "fcs/rudder-cmd-norm"
            ]
        )

        # ------------------------------------------------------------
        # Repaired mappings.
        #
        # action[2] = normalized lateral cyclic residual
        # action[3] = normalized pedal/yaw residual
        # ------------------------------------------------------------

        mapped_aileron = float(
            np.clip(
                base_aileron
                +
                self.mapped_aileron_scale
                *
                float(
                    action[2]
                ),
                -1.0,
                +1.0,
            )
        )

        mapped_rudder = float(
            np.clip(
                base_rudder
                +
                self.mapped_rudder_scale
                *
                float(
                    action[3]
                ),
                -1.0,
                +1.0,
            )
        )

        self.fdm[
            "fcs/aileron-cmd-norm"
        ] = mapped_aileron

        self.fdm[
            "fcs/rudder-cmd-norm"
        ] = mapped_rudder

        self.last_base_aileron = (
            base_aileron
        )

        self.last_base_rudder = (
            base_rudder
        )

        self.last_mapped_aileron = (
            mapped_aileron
        )

        self.last_mapped_rudder = (
            mapped_rudder
        )

        return (
            self._replace_return_controls(
                controls,
                mapped_aileron,
                mapped_rudder,
            )
        )
