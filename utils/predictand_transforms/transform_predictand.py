import numpy as np
from .registry import PREDICTAND_TRANSFORM_REGISTRY

def transform_predictand(values, mode, train_idxs, stats_save_path):
    """
    Apply transformation to the predictand (target or model output)
    Saves stats when needed.
    """
    if mode not in PREDICTAND_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transformation mode: {mode}")

    # Computes stats and transform accordingly
    values_std, stats_values = PREDICTAND_TRANSFORM_REGISTRY[mode](values, train_idxs)

    stats = {**stats_values, "mode": mode}

    if stats_save_path is not None:
        np.savez(stats_path, **stats)

    return values_std
