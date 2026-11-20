#!/usr/bin/env python3
"""Evaluate a prepared SFT split using the matching shell launcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SCHEMA = {
    "ENVIRONMENT__HF_ENDPOINT": {"env": "HF_ENDPOINT", "type": "str"},
    "ENVIRONMENT__HF_HUB_OFFLINE": {"env": "HF_HUB_OFFLINE", "type": "str"},
    "ENVIRONMENT__CUDA_VISIBLE_DEVICES": {"env": "CUDA_VISIBLE_DEVICES", "type": "str"},
    "METHOD": {"env": "SA_METHOD", "type": "str"},
    "TRAINING_RUN": {"env": "SA_TRAINING_RUN", "type": "str"},
    "CHECKPOINT": {"env": "SA_CHECKPOINT", "type": "optional"},
    "DATASET_DIR": {"env": "SA_DATASET_DIR", "type": "optional"},
    "SPLIT": {"env": "SA_SPLIT", "type": "str"},
    "INDEX_DIR": {"env": "SA_INDEX_DIR", "type": "str"},
    "RESUME_FROM": {"env": "SA_RESUME_FROM", "type": "optional"},
    "RUN_NAME": {"env": "SA_RUN_NAME", "type": "str"},
    "GENERATION__LIMIT": {"env": "SA_GENERATION__LIMIT", "type": "int"},
    "GENERATION__MAX_CONTEXT": {"env": "SA_GENERATION__MAX_CONTEXT", "type": "int"},
    "GENERATION__MAX_NEW_TOKENS": {"env": "SA_GENERATION__MAX_NEW_TOKENS", "type": "int"},
    "GENERATION__MAX_TURNS": {"env": "SA_GENERATION__MAX_TURNS", "type": "int"},
    "GENERATION__MAX_TOOL_TOKENS": {"env": "SA_GENERATION__MAX_TOOL_TOKENS", "type": "int"},
    "GENERATION__TOOL_COUNT": {"env": "SA_GENERATION__TOOL_COUNT", "type": "int"},
    "GENERATION__TOOL_CALL_RETRIES": {"env": "SA_GENERATION__TOOL_CALL_RETRIES", "type": "int"},
    "GENERATION__DECODING": {"env": "SA_GENERATION__DECODING", "type": "str"},
    "GENERATION__REPETITION_PENALTY": {"env": "SA_GENERATION__REPETITION_PENALTY", "type": "float"},
    "GENERATION__TEMPERATURE": {"env": "SA_GENERATION__TEMPERATURE", "type": "float"},
    "GENERATION__TOP_P": {"env": "SA_GENERATION__TOP_P", "type": "float"},
    "GENERATION__GENERATION_SEED": {"env": "SA_GENERATION__GENERATION_SEED", "type": "int"},
    "GENERATION__VERIFY_INDEX": {"env": "SA_GENERATION__VERIFY_INDEX", "type": "bool"},
    "GENERATION__TOPK": {"env": "SA_GENERATION__TOPK", "type": "int"},
    "GENERATION__SNIPPET_TOKENS": {"env": "SA_GENERATION__SNIPPET_TOKENS", "type": "int"},
    "GENERATION__READ_TOKENS": {"env": "SA_GENERATION__READ_TOKENS", "type": "int"},
    "GENERATION__ALLOW_VISIT": {"env": "SA_GENERATION__ALLOW_VISIT", "type": "bool"},
}

if __name__ == "__main__":
    from search_comp.experiments.shell_config import load_settings
    from search_comp.experiments.training_data_evaluation import evaluate_training_data

    evaluate_training_data(**{key.lower(): value for key, value in load_settings(__file__, SCHEMA).items()})
