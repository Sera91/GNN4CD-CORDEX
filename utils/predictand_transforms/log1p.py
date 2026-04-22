import numpy as np
from .registry import register_predictand_transform, register_predictand_inverse_transform

@register_predictand_transform("log1p")
def transform_log1p(values, stats_path):
    # No stats needed
    stats = {}
    return np.log1p(values), stats

@register_predictand_inverse_transform("log1p")
def inverse_transform_log1p(values_norm, stats):
    values = np.expm1(values_norm)
    return np.expm1(values_norm)
