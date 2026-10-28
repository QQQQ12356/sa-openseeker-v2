#!/usr/bin/env python3
"""Pinned official OpenSeeker benchmark entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from search_comp.workflows.benchmark_openseeker import main

if __name__ == "__main__":
    main()
