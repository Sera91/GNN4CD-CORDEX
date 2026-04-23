import numpy as np
from .registry import register_predictand_transform, register_predictand_inverse_transform

@register_predictand_transform("zscore")
def transform_zscore(values, train_idxs=None):
    if train_idxs is not None:
        values_train = values[:,train_idxs]
    mean = np.nanmean(values_train)
    std = np.nanstd(values_train)
    stats = {"mean": mean, "std": std}
    return (values - mean) / std, stats

@register_predictand_inverse_transform("zscore")
def inverse_transform_zscore(values_norm, stats):
    return values_norm * stats["std"] + stats["mean"]
