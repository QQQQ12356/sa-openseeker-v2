#!/usr/bin/env python3
"""Entry point; edit the matching scripts/sh launcher for one-click runs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from search_comp.workflows.filter_observation_dataset import main

if __name__ == '__main__':
    main()
