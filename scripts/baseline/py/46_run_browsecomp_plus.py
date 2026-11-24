#!/usr/bin/env python3
"""Run BrowseComp-Plus with the matching method launcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from search_comp.experiments.browsecomp_plus_entry import launch, launcher_schema

SCHEMA = launcher_schema("baseline")

if __name__ == "__main__":
    launch("baseline", __file__, SCHEMA)
