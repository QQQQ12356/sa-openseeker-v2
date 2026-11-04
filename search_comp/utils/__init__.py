"""通用工具。"""
from importlib import import_module


def __getattr__(name):
    """Do not import the training runtime when only progress utilities are needed."""
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module('.runtime', __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "append_jsonl",
    "count_parameters",
    "load_yaml_config",
    "prepare_run_artifacts",
    "require_keys",
    "resolve_experiment_dir",
    "write_json",
]
