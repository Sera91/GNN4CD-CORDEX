# utils/transformations/registry.py

PREDICTAND_TRANSFORM_REGISTRY = {}
PREDICTAND_INVERSE_TRANSFORM_REGISTRY = {}

def register_predictand_transform(mode):
    """Decorator for forward transforms."""
    def decorator(func):
        PREDICTAND_TRANSFORM_REGISTRY[mode] = func
        return func
    return decorator

def register_predictand_inverse_transform(mode):
    """Decorator for inverse transforms."""
    def decorator(func):
        PREDICTAND_INVERSE_TRANSFORM_REGISTRY[mode] = func
        return func
    return decorator
