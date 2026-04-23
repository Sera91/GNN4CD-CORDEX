import inspect
from .registry import MODEL_REGISTRY

def build_model(
    x_low_var_dim,
    x_low_lev_dim,
    x_high_dim,
    output_dim,
    args):

    ModelClass = MODEL_REGISTRY[args.model_name]

    sig = inspect.signature(ModelClass.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    
    filtered = {
        k: v for k, v in vars(args).items()
        if k in allowed
    }

    model = ModelClass(
        x_low_var_dim=x_low_var_dim,
        x_low_lev_dim=x_low_lev_dim,
        x_high_dim=x_high_dim,
        output_dim=output_dim,
        **filtered
        )

    return model
