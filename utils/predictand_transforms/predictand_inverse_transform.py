# utils/transformations/inverse_transform_predictand.py

import numpy as np
from .registry import PREDICTAND_INVERSE_TRANSFORM_REGISTRY

def predictand_inverse_transform(values_norm, stats_path):
    """
    Apply inverse transformation to model outputs or normalized targets.
    Loads stats saved during training.
    """
    stats = np.load(stats_path, allow_pickle=True)

    mode = stats["mode"]
    if isinstance(mode, np.ndarray):
        mode = mode.item()

    if mode not in PREDICTAND_INVERSE_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown inverse transformation mode: {mode}")

    return PREDICTAND_INVERSE_TRANSFORM_REGISTRY[mode](values_norm, stats)