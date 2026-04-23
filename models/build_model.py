import inspect
from .registry import MODEL_REGISTRY

OUTPUT_DIM = {
    "MSE_QMSE_PSD_Loss": 1,
    "BernoulliGammaNLLLoss": 3,
    "GaussianNLLLoss": 2
}

def build_model(args):
    output_dim = OUTPUT_DIM[args.loss_fn]

    ModelClass = MODEL_REGISTRY[args.model_name]

    sig = inspect.signature(ModelClass.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    
    filtered = {
        k: v for k, v in vars(args).items()
        if k in allowed
    }

    return ModelClass(output_dim=output_dim, **filtered)
