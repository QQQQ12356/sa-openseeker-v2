"""Lazy model exports keep configuration inspection independent of torch."""
from importlib import import_module
__all__ = ["BeaconConfig", "BeaconMemory", "load_model", "load_tokenizer"]
def __getattr__(name):
    modules = {"BeaconConfig": "beacon_config", "BeaconMemory": "beacon_common",
               "load_model": "model_loader", "load_tokenizer": "model_loader"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module("." + modules[name], __name__), name)
    globals()[name] = value
    return value
