from .registry import get_extractor

def extract_prediction(y_out, loss_name):
    extractor = get_extractor(loss_name)
    return extractor(y_out)