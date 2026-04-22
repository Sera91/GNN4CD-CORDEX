import numpy as np
from .registry import PREDICTAND_INVERSE_TRANSFORM_REGISTRY

def inverse_transform_predictand(values_norm, stats):
    """
    Apply inverse transformation to model outputs or normalized targets.
    Loads stats saved during training.
    """
    
    mode = stats["mode"]
    if isinstance(mode, np.ndarray):
        mode = mode.item()

    if mode not in PREDICTAND_INVERSE_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown inverse transformation mode: {mode}")

    return PREDICTAND_INVERSE_TRANSFORM_REGISTRY[mode](values_norm, stats)