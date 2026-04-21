from .registry import register_extractor

@register_extractor("gaussian_nll")
def extract_gaussian_mean(y_out):
    mu = y_out[..., 0]
    return mu
