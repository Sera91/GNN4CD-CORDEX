from .registry import register_extractor

@register_extractor("mse_qmse_psd")
def extract_mse_qmse_psd(y_out):
    return y_out.squeeze(-1)