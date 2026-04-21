EXTRACTOR_REGISTRY = {}

def register_extractor(loss_name):
    def decorator(func):
        EXTRACTOR_REGISTRY[loss_name] = func
        return func
    return decorator