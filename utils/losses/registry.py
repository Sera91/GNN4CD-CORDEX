# utils/losses/registry.py

LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(cls):
        """
        cls: the object being decorated
        """
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator

def get_loss(name):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_REGISTRY[name]
