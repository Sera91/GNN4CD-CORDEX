# utils/predictor_transforms/registry.py

PREDICTOR_TRANSFORM_REGISTRY = {}

def register_predictor_transform(mode):
    """Decorator for forward transforms."""
    def decorator(func):
        """
        func: the function being decorated
        """
        PREDICTOR_TRANSFORM_REGISTRY[mode] = func
        return func
    return decorator