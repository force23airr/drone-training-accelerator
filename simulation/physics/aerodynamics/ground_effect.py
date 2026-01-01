"""
Ground Effect Model for Fixed-Wing Aircraft.

Models the increase in lift and reduction in induced drag
when an aircraft flies close to the ground (within approximately
one wingspan of the surface).

The ground effect is caused by:
1. Reduction in wingtip vortex strength (reduced induced drag)
2. Increase in effective aspect ratio
3. Ram air cushion effect (at very low heights)

Reference: McCormick, B.W., "Aerodynamics, Aeronautics, and Flight Mechanics"
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class GroundEffectConfig:
    """
    Configuration for ground effect model.

    Attributes:
        wingspan: Aircraft wingspan [m]
        wing_height: Height of wing above ground when on gear [m]
        ground_effect_factor: Scaling factor for effect strength (0-1)
        enable_ram_air: Enable ram air cushion effect at very low heights
    """

    wingspan: float = 10.0
    wing_height: float = 1.5
    ground_effect_factor: float = 1.0
    enable_ram_air: bool = True


class GroundEffectModel:
    """
    Ground effect aerodynamic modification model.

    Computes corrections to lift and induced drag coefficients
    based on height above ground.

    The model uses empirical correlations validated against
    wind tunnel and flight test data.
    """

    def __init__(self, config: GroundEffectConfig = None):
        """
        Initialize ground effect model.

        Args:
            config: Ground effect configuration parameters
        """
        self.config = config or GroundEffectConfig()

    def compute_ground_effect_factors(
        self,
        height_agl: float,
        aspect_ratio: float = 8.0
    ) -> Tuple[float, float]:
        """
        Compute ground effect correction factors for lift and induced drag.

        Args:
            height_agl: Height above ground level [m]
            aspect_ratio: Wing aspect ratio

        Returns:
            Tuple of (lift_factor, induced_drag_factor)
            - lift_factor > 1: Lift is increased
            - induced_drag_factor < 1: Induced drag is reduced
        """
        b = self.config.wingspan

        # Height ratio (h/b)
        h_ratio = height_agl / b

        # No ground effect above one wingspan
        if h_ratio > 1.0:
            return 1.0, 1.0

        # Clamp minimum height to avoid singularities
        h_ratio = max(h_ratio, 0.05)

        # Induced drag reduction factor (Hoerner model)
        # sigma = ground effect factor on induced drag
        # sigma = 1 - 1.32 * (h/b) / sqrt(1 + (1.32 * h/b)^2)
        # Simplified form for practical use
        sigma = self._compute_induced_drag_factor(h_ratio)

        # Lift increase factor
        # Approximate: dCL/CL ≈ (1 - sigma) / AR
        # More accurate empirical model:
        lift_factor = self._compute_lift_factor(h_ratio, aspect_ratio)

        # Apply configuration scaling
        sigma_adj = 1.0 - (1.0 - sigma) * self.config.ground_effect_factor
        lift_adj = 1.0 + (lift_factor - 1.0) * self.config.ground_effect_factor

        # Ram air cushion effect at very low heights
        if self.config.enable_ram_air and h_ratio < 0.1:
            ram_factor = 1.0 + 0.1 * (0.1 - h_ratio) / 0.1
            lift_adj *= ram_factor

        return lift_adj, sigma_adj

    def _compute_induced_drag_factor(self, h_ratio: float) -> float:
        """
        Compute induced drag reduction factor using empirical model.

        Based on Hoerner's formula:
        sigma = (16*h/b)^2 / (1 + (16*h/b)^2)

        Where sigma = 1 means no ground effect, sigma < 1 means reduced drag.

        Args:
            h_ratio: Height to wingspan ratio (h/b)

        Returns:
            Induced drag factor (0 to 1)
        """
        # Coefficient calibrated from experimental data
        k = 16.0 * h_ratio

        sigma = k ** 2 / (1.0 + k ** 2)

        return sigma

    def _compute_lift_factor(
        self,
        h_ratio: float,
        aspect_ratio: float
    ) -> float:
        """
        Compute lift increase factor in ground effect.

        The lift increase is related to the effective increase in
        aspect ratio due to the ground plane acting as a mirror.

        Args:
            h_ratio: Height to wingspan ratio (h/b)
            aspect_ratio: Wing aspect ratio

        Returns:
            Lift multiplication factor (>= 1)
        """
        # Induced drag factor
        sigma = self._compute_induced_drag_factor(h_ratio)

        # Lift coefficient increase
        # Derived from: effective AR increase = AR / sigma
        # CL increase ≈ CL_alpha_increase / CL_alpha ≈ 1 + k*(1-sigma)
        # where k depends on aspect ratio

        # Empirical correlation
        k = 0.5 / (1.0 + aspect_ratio / 10.0)
        lift_factor = 1.0 + k * (1.0 - sigma)

        return lift_factor

    def modify_coefficients(
        self,
        CL: float,
        CD_induced: float,
        height_agl: float,
        aspect_ratio: float = 8.0
    ) -> Tuple[float, float]:
        """
        Apply ground effect modifications to aerodynamic coefficients.

        Args:
            CL: Lift coefficient (without ground effect)
            CD_induced: Induced drag coefficient
            height_agl: Height above ground [m]
            aspect_ratio: Wing aspect ratio

        Returns:
            Tuple of (modified_CL, modified_CD_induced)
        """
        lift_factor, drag_factor = self.compute_ground_effect_factors(
            height_agl, aspect_ratio
        )

        return CL * lift_factor, CD_induced * drag_factor

    def get_effective_aspect_ratio(
        self,
        aspect_ratio: float,
        height_agl: float
    ) -> float:
        """
        Compute effective aspect ratio in ground effect.

        The ground acts as a mirror, effectively increasing the
        apparent aspect ratio of the wing.

        Args:
            aspect_ratio: Actual wing aspect ratio
            height_agl: Height above ground [m]

        Returns:
            Effective aspect ratio
        """
        _, sigma = self.compute_ground_effect_factors(height_agl, aspect_ratio)

        # Effective AR = AR / sigma (when sigma < 1)
        if sigma > 0.1:
            return aspect_ratio / sigma
        else:
            return aspect_ratio * 10.0  # Cap at 10x increase

    def get_flare_height_recommendation(self) -> float:
        """
        Get recommended flare initiation height.

        The flare should begin where ground effect becomes significant
        enough to require pitch adjustment.

        Returns:
            Recommended flare height [m]
        """
        # Ground effect becomes noticeable at h/b ≈ 0.5
        # Strong effect at h/b ≈ 0.25
        # Flare typically initiated at h/b ≈ 0.1-0.15 wingspan
        return self.config.wingspan * 0.12

    def get_cushion_height(self) -> float:
        """
        Get height where maximum cushioning effect occurs.

        At this height the aircraft will "float" significantly
        and require reduced power for level flight.

        Returns:
            Maximum cushion height [m]
        """
        # Maximum effect at h/b ≈ 0.05-0.1
        return self.config.wingspan * 0.08


def compute_landing_ground_effect(
    wingspan: float,
    gear_height: float,
    sink_rate: float,
    airspeed: float,
    CL: float,
    aspect_ratio: float = 8.0
) -> dict:
    """
    Convenience function to compute ground effect for landing analysis.

    Args:
        wingspan: Wing span [m]
        gear_height: Height of gear above ground [m]
        sink_rate: Descent rate (positive down) [m/s]
        airspeed: True airspeed [m/s]
        CL: Current lift coefficient
        aspect_ratio: Wing aspect ratio

    Returns:
        Dictionary with ground effect analysis results
    """
    config = GroundEffectConfig(
        wingspan=wingspan,
        wing_height=gear_height
    )
    model = GroundEffectModel(config)

    # Height of wing (assuming wing is mounted above gear)
    height_agl = gear_height + 0.2 * wingspan  # Approximate

    lift_factor, drag_factor = model.compute_ground_effect_factors(
        height_agl, aspect_ratio
    )

    # Time to touchdown
    if sink_rate > 0:
        time_to_touchdown = gear_height / sink_rate
    else:
        time_to_touchdown = float('inf')

    return {
        'height_agl': height_agl,
        'lift_factor': lift_factor,
        'induced_drag_factor': drag_factor,
        'effective_CL': CL * lift_factor,
        'effective_AR': model.get_effective_aspect_ratio(aspect_ratio, height_agl),
        'time_to_touchdown': time_to_touchdown,
        'in_ground_effect': height_agl < wingspan,
        'flare_height': model.get_flare_height_recommendation(),
        'cushion_height': model.get_cushion_height(),
    }
