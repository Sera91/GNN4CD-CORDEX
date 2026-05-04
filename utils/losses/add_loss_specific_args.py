from .registry import LOSS_REGISTRY

def add_loss_specific_args(parser, loss_name):
    LossClass = LOSS_REGISTRY[loss_name]
    if hasattr(LossClass, "add_loss_specific_args"):
        parser = LossClass.add_loss_specific_args(parser)
    return parser