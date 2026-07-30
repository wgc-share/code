from __future__ import annotations

import os
from pathlib import Path

import torch


def load_torch_file(path, map_location=None):
    """Load a trusted local PyTorch file through a Unicode-safe file handle."""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        return torch.load(handle, map_location=map_location, weights_only=False)


def save_torch_file(value, path):
    """Atomically save a PyTorch file, including under non-ASCII Windows paths."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_name(f"{file_path.name}.tmp")
    try:
        with temp_path.open("wb") as handle:
            torch.save(value, handle)
        os.replace(temp_path, file_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
