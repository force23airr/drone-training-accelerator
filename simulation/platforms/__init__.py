"""
Platform configurations for different drone types.
"""

from simulation.platforms.platform_configs import (
    get_platform_config,
    list_platforms,
    register_platform,
    PlatformConfig,
)

__all__ = [
    "get_platform_config",
    "list_platforms",
    "register_platform",
    "PlatformConfig",
]
