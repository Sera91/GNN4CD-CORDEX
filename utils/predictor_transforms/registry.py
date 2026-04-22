# utils/predictor_transforms/registry.py

TRANSFORM_PREDICTOR_REGISTRY = {}

def register_predictor_transform(mode):
    """Decorator for forward transforms."""
    def decorator(func):
        """
        func: the function being decorated
        """
        TRANSFORM_PREDICTOR_REGISTRY[mode] = func
        return func
    return decorator