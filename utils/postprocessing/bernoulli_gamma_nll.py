from .registry import register_extractor
import torch.nn.functional as F

@register_extractor("bernoulli_gamma_nll")
def extract_bg_mean(y_out):
    p_raw = y_out[..., 0]
    shape_raw = y_out[..., 1]
    scale_raw = y_out[..., 2]

    p = torch.sigmoid(p_raw)
    shape = F.softplus(shape_raw)
    scale = F.softplus(scale_raw)

    # mean of mixture: (1 - p_zero) * mean_gamma
    return (1 - p) * (shape * scale)
