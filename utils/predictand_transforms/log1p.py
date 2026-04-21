# utils/transformations/log1p.py

import numpy as np
from .registry import register_predictand_transform, register_predictand_inverse_transform

@register_predictand_transform("log1p")
def transform_log1p(values, stats_path):
    # No stats needed
    np.savez(stats_path, mode="log1p")
    return np.log1p(values)

@register_predictand_inverse_transform("log1p")
def inverse_log1p(values_norm, stats):
    values = np.expm1(values_norm)
    return np.expm1(values_norm)
