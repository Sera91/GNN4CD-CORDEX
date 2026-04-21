# utils/transformations/minmax.py

import numpy as np
from .registry import register_predictand_transform, register_predictand_inverse_transform

@register_transform("minmax")
def transform_minmax(values, stats_path):
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    np.savez(stats_path, mode="minmax", min=min_val, max=max_val)
    return (values - min_val) / (max_val - min_val)

@register_inverse_transform("minmax")
def inverse_minmax(values_norm, stats):
    return values_norm * (stats["max"] - stats["min"]) + stats["min"]
