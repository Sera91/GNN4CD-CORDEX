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

def get_predictor_transform(mode):
    if mode not in PREDICTOR_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown predictor transform: {mode}")
    return PREDICTOR_TRANSFORM_REGISTRY[mode]