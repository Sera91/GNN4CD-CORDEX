DATASET_REGISTRY = {}

def register_dataset(name):
    def decorator(func):
        DATASET_REGISTRY[name] = func
        return func
    return decorator