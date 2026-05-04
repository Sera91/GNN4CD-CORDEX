import inspect
from .registry import LOSS_REGISTRY

def build_loss(args):
    """ 
    To instantate the LossClass with only
    the args in its signature
    """
    LossClass = LOSS_REGISTRY[args.loss_name]

    sig = inspect.signature(LossClass.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    
    filtered = {
        k: v for k, v in vars(args).items()
        if k in allowed
    }

    loss = LossClass(**filtered)

    return loss, LossClass.output_dim
