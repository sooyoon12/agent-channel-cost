#!/usr/bin/env python3
"""Verify the frozen dataset manifest."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.export import verify_frozen_dataset


if __name__ == "__main__":
    print(verify_frozen_dataset("data"))
