# utils/transformations/predictand_transform.py

import numpy as np
from .registry import PREDICTAND_TRANSFORM_REGISTRY

def predictand_transform(values, mode, stats_path):
    """
    Apply forward transformation to the predictand (target or model output)
    Saves stats when needed.
    """
    if mode not in PREDICTAND_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transformation mode: {mode}")

    return PREDICTAND_TRANSFORM_REGISTRY[mode](values, stats_path)
