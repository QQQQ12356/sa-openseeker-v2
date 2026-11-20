#!/usr/bin/env python3
"""Prepare BrowseComp-Plus vector assets using the matching shell launcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SCHEMA = {
    "ENVIRONMENT__HF_ENDPOINT": {"env": "HF_ENDPOINT", "type": "str"},
    "ENVIRONMENT__HF_HUB_OFFLINE": {"env": "HF_HUB_OFFLINE", "type": "str"},
    "ENVIRONMENT__CUDA_VISIBLE_DEVICES": {"env": "CUDA_VISIBLE_DEVICES", "type": "str"},
    "VERSION": {"env": "SA_VERSION", "type": "str"},
    "INDEX_PROFILE": {"env": "SA_INDEX_PROFILE", "type": "str"},
    "PHASE": {"env": "SA_PHASE", "type": "str"},
    "SOURCE": {"env": "SA_SOURCE", "type": "str"},
    "MODEL_REVISION": {"env": "SA_MODEL_REVISION", "type": "str"},
    "INDEX_REVISION": {"env": "SA_INDEX_REVISION", "type": "str"},
    "EMBEDDING_MODEL_PATH": {"env": "SA_EMBEDDING_MODEL_PATH", "type": "optional"},
    "OFFICIAL_INDEX_DIR": {"env": "SA_OFFICIAL_INDEX_DIR", "type": "optional"},
    "DEVICE": {"env": "SA_DEVICE", "type": "str"},
    "BATCH_SIZE": {"env": "SA_BATCH_SIZE", "type": "int"},
    "DOC_MAX_LENGTH": {"env": "SA_DOC_MAX_LENGTH", "type": "int"},
}

if __name__ == "__main__":
    from search_comp.experiments.browsecomp_plus_dense import prepare_browsecomp_plus_dense
    from search_comp.experiments.shell_config import load_settings

    settings = load_settings(__file__, SCHEMA)
    prepare_browsecomp_plus_dense(**{key.lower(): value for key, value in settings.items()})
