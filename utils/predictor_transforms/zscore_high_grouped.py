# utils/predictor_transforms/highres_zscore_grouped.py

import numpy as np
from .registry import register_predictor_transform, register_predictor_inverse_transform

@register_predictor_transform("zscore_high_grouped")
def transform_highres_zscore_grouped(x_high, stats=None):
    """
    Grouped z-score:
    - channel 0 separate # orography
    - channels 1-N grouped together # land use
    """
    if stats is None:
        if x_high.shape[1] > 1:
            means = np.array([
                np.nanmean(x_high[:,0]),
                np.nanmean(x_high[:,1:])
            ])
            stds = np.array([
                np.nanstd(x_high[:,0]),
                np.nanstd(x_high[:,1:])
            ])
        else:
            means = np.nanmean(x_high)
            stds  = np.nanstd(x_high)
        stats = {"means_high": means, "stds_high": stds}
    else:
        means = stats["means_high"]
        stds = stats["stds_high"]

    x = x_high.copy()

    if x.shape[1] > 1:
        x[:, 0]  = (x[:, 0]  - means[0]) / stds[0]
        x[:, 1:] = (x[:, 1:] - means[1]) / stds[1]
    else:
        x = (x - means) / stds

    return x, stats
