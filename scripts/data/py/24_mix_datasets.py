#!/usr/bin/env python3
"""Run dataset mixing using the settings in the matching shell launcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SCHEMA = {
    "OPENSEEKER_PREPARED": {"env": "SA_OPENSEEKER_PREPARED", "type": "str"},
    "RESEARCH_PREPARED": {"env": "SA_RESEARCH_PREPARED", "type": "str"},
    "BROWSECOMP_SFT_PREPARED": {"env": "SA_BROWSECOMP_SFT_PREPARED", "type": "optional"},
    "PROFILE": {"env": "SA_PROFILE", "type": "str"},
    "TOKENIZER": {"env": "SA_TOKENIZER", "type": "str"},
    "MAX_LENGTH": {"env": "SA_MAX_LENGTH", "type": "int"},
    "RATIO": {"env": "SA_RATIO", "type": "str"},
    "SEED": {"env": "SA_SEED", "type": "int"},
}

if __name__ == "__main__":
    from search_comp.experiments.mixed_data import mix_datasets
    from search_comp.experiments.shell_config import load_settings

    mix_datasets(**{key.lower(): value for key, value in load_settings(__file__, SCHEMA).items()})
