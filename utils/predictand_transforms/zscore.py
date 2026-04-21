# utils/transformations/zscore.py

import numpy as np
from .registry import register_predictand_transform, register_predictand_inverse_transform

@register_predictand_transform("z_score")
def transform_zscore(values, stats_path):
    mean = np.nanmean(values)
    std = np.nanstd(values)
    np.savez(stats_path, mode="z_score", mean=mean, std=std)
    return (values - mean) / std

@register_predictand_inverse_transform("z_score")
def inverse_zscore(values_norm, stats):
    return values_norm * stats["std"] + stats["mean"]
