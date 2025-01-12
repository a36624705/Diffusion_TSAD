MODEL_REGISTRY = {}

def register_model(name, model_class, train_fn, eval_fn):
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is already registered.")
    MODEL_REGISTRY[name] = {
        "model_class": model_class,
        "train_fn": train_fn,
        "eval_fn": eval_fn,
    }

def get_model_class(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered.")
    return MODEL_REGISTRY[name]["model_class"]

def get_train_fn(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered.")
    return MODEL_REGISTRY[name]["train_fn"]

def get_eval_fn(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered.")
    return MODEL_REGISTRY[name]["eval_fn"]