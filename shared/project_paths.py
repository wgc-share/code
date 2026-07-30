from __future__ import annotations

from pathlib import Path


ORG_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ORG_ROOT / "data"
RESULTS_ROOT = ORG_ROOT / "results"
CODE_ROOT = ORG_ROOT / "code"


def first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of the candidate paths exist: " + ", ".join(str(p) for p in candidates))


def organized_processed_segments_dir() -> Path:
    path = DATA_ROOT / "processed" / "soccc_segments" / "segments"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def organized_processed_metadata_dir() -> Path:
    path = DATA_ROOT / "processed" / "soccc_segments" / "metadata"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def organized_raw_soccc_dir() -> Path:
    path = DATA_ROOT / "raw" / "soccc"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def organized_split_file() -> Path:
    path = DATA_ROOT / "splits" / "split_by_cell.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def organized_results_dir() -> Path:
    return RESULTS_ROOT


def processed_segments_dir() -> Path:
    return organized_processed_segments_dir()


def split_file_path() -> Path:
    return organized_split_file()


def baseline_results_dir() -> Path:
    return organized_results_dir() / "baseline"


def comparison_results_dir() -> Path:
    return organized_results_dir() / "comparison"


def figures_dir() -> Path:
    return organized_results_dir() / "figures"
