# utils/predictand_transforms/registry.py

TRANSFORM_PREDICTAND_REGISTRY = {}
INVERSE_TRANSFORM_PREDICTAND_REGISTRY = {}

def register_predictand_transform(mode):
    """Decorator for forward transforms."""
    def decorator(func):
        """
        func: the function being decorated
        """
        PREDICTAND_TRANSFORM_REGISTRY[mode] = func
        return func
    return decorator

def register_predictand_inverse_transform(mode):
    """Decorator for inverse transforms."""
    def decorator(func):
        """
        func: the function being decorated
        """
        PREDICTAND_INVERSE_REGISTRY[mode] = func
        return func
    return decorator
