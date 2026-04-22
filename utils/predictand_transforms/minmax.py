import numpy as np
from .registry import register_predictand_transform, register_predictand_inverse_transform

@register_predictand_transform("minmax")
def transform_minmax(values, stats_path):
    min_val = np.floor(np.nanmin(values))
    max_val = np.ceil(np.nanmax(values))
    stats = {"min": min_val, "max": max_val}
    return (values - min_val) / (max_val - min_val), stats

@register_predictand_inverse_transform("minmax")
def inverse_transform_minmax(values_norm, stats):
    return values_norm * (stats["max"] - stats["min"]) + stats["min"]
