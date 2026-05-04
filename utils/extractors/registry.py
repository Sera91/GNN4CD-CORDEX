EXTRACTOR_REGISTRY = {}

def register_extractor(loss_name):
    def decorator(func):
        EXTRACTOR_REGISTRY[loss_name] = func
        return func
    return decorator

def get_extractor(name):
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor: {name}")
    return EXTRACTOR_REGISTRY[name]