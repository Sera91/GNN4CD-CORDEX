from .registry import EXTRACTOR_REGISTRY

def extract_prediction(y_out, loss_name):
    extractor = EXTRACTOR_REGISTRY[loss_name]
    return extractor(y_out)