import numpy as np
from .registry import PREDICTOR_TRANSFORM_REGISTRY

def transform_predictors(
    x_low,
    x_high,
    train_idxs=None,
    mode_low="zscore_lowres_var",
    mode_high="zscore_highres_independent",
    stats=None,
    stats_save_path=None,
    **kwargs
):
    # --- LOW-RES ---
    x_low_std, stats_low = PREDICTOR_TRANSFORM_REGISTRY[mode_low](
        x_low, x_low.shape[2], train_idxs, stats
    )

    # --- HIGH-RES ---
    x_high_std, stats_high = PREDICTOR_TRANSFORM_REGISTRY[mode_high](
        x_high, stats
    )

    # --- Save unified stats ---
    if stats is None and stats_save_path is not None:

        stats = {**stats_low, **stats_high,
                "mode_low": mode_low,
                "mode_high": mode_high}
        
        np.savez(stats_save_path, **stats)

    return x_low_std, x_high_std
