from .registry import MODEL_REGISTRY

def add_model_specific_args(parser, model_name):
    ModelClass = MODEL_REGISTRY[model_name]
    if hasattr(ModelClass, "add_model_specific_args"):
        parser = ModelClass.add_model_specific_args(parser)
    return parser