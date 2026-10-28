#!/usr/bin/env python3
"""Serve a trained Plain checkpoint for official OpenSeeker generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from search_comp.workflows.serve_openseeker import main

if __name__ == "__main__":
    main()
