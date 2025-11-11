"""Shared utilities for applying performance presets."""

from typing import Tuple, Optional
from app.config import get_preset, DEFAULT_MODEL
from app.exceptions import PartiToneError


def apply_preset(preset_name: Optional[str]) -> Tuple[str, bool, bool, str]:
    """Apply preset configuration or return defaults.
    
    Args:
        preset_name: Name of preset (fast, balanced, quality, ultra) or None
        
    Returns:
        Tuple of (model, raw, repair_glitch, repair_mode)
        
    Raises:
        ValueError: If preset name is invalid
    """
    if preset_name:
        preset_config = get_preset(preset_name)
        return (
            preset_config.model,
            preset_config.raw,
            preset_config.repair_glitch,
            preset_config.repair_mode
        )
    else:
        # Return defaults
        return (DEFAULT_MODEL, False, False, "standard")

