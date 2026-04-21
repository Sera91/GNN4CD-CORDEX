from .registry import EXTRACTOR_REGISTRY
from utils.helpers import invert_normalization

def get_final_values(y, y_out, args):
    extractor = EXTRACTOR_REGISTRY[args.loss_fn]
    pred = extractor(y_out)

    if args.inverse_norm:
        y = invert_normalization(y, args)
        pred = invert_normalization(pred, args)

    return y, pred

