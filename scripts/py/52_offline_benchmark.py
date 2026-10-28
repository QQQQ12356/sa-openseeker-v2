#!/usr/bin/env python3
"""Offline fixed-corpus benchmark entry point; see docs/43_offline_benchmarks.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from search_comp.workflows.offline_benchmark import main

if __name__ == "__main__":
    main()
