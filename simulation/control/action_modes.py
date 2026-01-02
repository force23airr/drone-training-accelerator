"""
Action mode definitions for policy/control separation.
"""

from enum import Enum


class ActionMode(Enum):
    """Supported action interfaces for policies."""
    MOTOR_THRUSTS = "motor_thrusts"      # Raw motor commands (normalized)
    ATTITUDE_RATES = "attitude_rates"    # [roll_rate, pitch_rate, yaw_rate, thrust]
    ATTITUDE = "attitude"                # [roll, pitch, yaw, thrust]
    VELOCITY = "velocity"                # [vx, vy, vz] (optionally yaw)
