DATASET_LOADER_REGISTRY = {}

def register_dataset_loader(name):
    def decorator(func):
        """
        func: the function being decorated
        """
        DATASET_LOADER_REGISTRY[name] = func
        return func
    return decorator

def get_dataset_loader(name):
    if name not in DATASET_LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset loader: {name}")
    return DATASET_LOADER_REGISTRY[name]