import numpy as np
from .registry import register_predictor_transform, register_predictor_inverse_transform

@register_predictor_transform("zscore_high_independent")
def transform_zscore_high_independent(x_high, stats=None):
    """
    Independent z-score per feature.
    """
    if stats is None:
        means = np.array([np.nanmean(x_high[:,i]) for i in range(x_high.shape[1])])
        stds  = np.array([np.nanstd(x_high[:,i])  for i in range(x_high.shape[1])])
        stats = {"means_high": means, "stds_high": stds}
    else:
        means = stats["means_high"]
        stds = stats["stds_high"]

    x = x_high.copy()
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - means[i]) / stds[i]

    return x, stats