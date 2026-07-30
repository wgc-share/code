from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_ROOT.parent
SHARED_DIR = CODE_ROOT / "shared"
BASELINE_DIR = CODE_ROOT / "baseline"

if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from adaln_model import BatteryTDGCMModel as AdaLNModel
from model import BatteryTDGCMModel as FiLMModel
from model_causal import BatteryTDGCMModel as CausalFiLMModel
from project_paths import baseline_results_dir, processed_segments_dir, split_file_path
from scaling import PITDScaler
from soccc_schemeB import SEGMENT_RE, BatteryTDGCMDataset, split_soccc_by_cells
from train_utils import PITDPhysicsLoss, evaluate_dataset


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _import_entrypoint(path):
    spec = importlib.util.spec_from_file_location(f"smoke_{path.stem}", path)
    _check(spec is not None and spec.loader is not None, f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _check_workspace_paths():
    paths = [processed_segments_dir(), split_file_path(), baseline_results_dir()]
    root = WORKSPACE_ROOT.resolve()
    for path in paths:
        resolved = path.resolve()
        _check(resolved == root or root in resolved.parents, f"Path escapes workspace: {resolved}")
    _check(paths[0].is_dir(), f"Missing processed data directory: {paths[0]}")
    _check(paths[1].is_file(), f"Missing cell split file: {paths[1]}")


def _check_splits_and_dataset():
    data_dir = processed_segments_dir()
    splits = split_soccc_by_cells(str(data_dir), str(split_file_path()))
    names = ("train", "val", "test_random", "test_fixed")
    for name, files in zip(names, splits):
        _check(files, f"Empty split: {name}")

    split_sets = [set(files) for files in splits]
    for i, left in enumerate(split_sets):
        for right in split_sets[i + 1 :]:
            _check(left.isdisjoint(right), "The data split contains duplicate files across groups")

    smoke_dir = baseline_results_dir()
    smoke_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    cache_files = []
    try:
        for name, files in zip(names, splits):
            cache_file = smoke_dir / f".smoke_{name}.pt"
            cache_files.append(cache_file)
            dataset = BatteryTDGCMDataset(
                str(data_dir),
                files[:1],
                window_size=10,
                stride=10,
                cache_file=str(cache_file),
            )
            _check(len(dataset) > 0, f"No windows built for representative {name} file")
            sample = dataset[0]
            _check(sample["x_dyn"].shape == (10, 5), f"Unexpected feature shape in {name}")
            _check(sample["soc"].shape == (10, 1), f"Unexpected SOC shape in {name}")
            reloaded = BatteryTDGCMDataset(
                str(data_dir),
                files[:1],
                window_size=10,
                stride=10,
                cache_file=str(cache_file),
            )
            _check(len(reloaded) == len(dataset), f"Cache round-trip failed for {name}")
            datasets.append(dataset)

        scaler = PITDScaler()
        scaler.fit(datasets[0])
        x = datasets[0][0]["x_dyn"].unsqueeze(0)
        t = datasets[0][0]["t_mean"].unsqueeze(0)
        x_norm, t_norm = scaler.transform(x, t)
        _check(torch.isfinite(x_norm).all().item(), "Scaler produced non-finite features")
        _check(torch.isfinite(t_norm).all().item(), "Scaler produced non-finite temperature")
    finally:
        for cache_file in cache_files:
            cache_file.unlink(missing_ok=True)
            cache_file.with_name(f"{cache_file.name}.tmp").unlink(missing_ok=True)

    return [len(files) for files in splits]


def _check_all_segment_files():
    data_dir = processed_segments_dir()
    splits = split_soccc_by_cells(str(data_dir), str(split_file_path()))
    split_files = [filename for files in splits for filename in files]
    disk_files = {path.name for path in data_dir.glob("*.csv")}
    _check(len(split_files) == len(set(split_files)), "Duplicate files exist across data splits")
    _check(set(split_files) == disk_files, "Some segment files are missing from the cell split")

    required_groups = (
        {"Current(mA)", "Current(A)"},
        {"Voltage(mV)", "Voltage(V)"},
        {"T(C)", "T"},
        {"SOC", "SOC/DOD(%)"},
    )
    for filename in split_files:
        _check(SEGMENT_RE.match(filename) is not None, f"Invalid segment filename: {filename}")
        with (data_dir / filename).open("r", encoding="utf-8-sig", newline="") as handle:
            header = set(next(csv.reader(handle)))
        _check(
            all(header.intersection(group) for group in required_groups),
            f"Missing required CSV columns: {filename}",
        )
    print(f"Full segment audit: {len(split_files)} files passed")


def _check_models_and_loss():
    x = torch.randn(2, 12, 5)
    t = torch.randn(2, 1)
    models = (
        FiLMModel(d_model=32, nhead=4, num_layers=1),
        CausalFiLMModel(d_model=32, nhead=4, num_layers=1),
        AdaLNModel(d_model=32, nhead=4, num_layers=1, use_causal=False),
        AdaLNModel(d_model=32, nhead=4, num_layers=1, use_causal=True),
    )
    for model in models:
        y, h = model(x, t)
        _check(y.shape == (2, 12, 1), f"Unexpected model output shape: {y.shape}")
        _check(h.shape == (1, 2, 32), f"Unexpected hidden-state shape: {h.shape}")
        _check(torch.isfinite(y).all().item(), "Model produced non-finite output")

    loss_fn = PITDPhysicsLoss()
    prediction = torch.sigmoid(torch.randn(2, 12, 1, requires_grad=True))
    target = torch.rand(2, 12, 1)
    current = torch.randn(2, 12)
    capacity = torch.full((2,), 5000.0)
    values = loss_fn(
        prediction,
        target,
        current,
        capacity,
        torch.tensor([True, False]),
        10.0,
        torch.ones(2),
    )
    _check(all(torch.isfinite(value).item() for value in values), "Loss produced non-finite values")
    values[0].backward()


def _check_parallel_evaluation():
    class MemoryDataset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    generator = torch.Generator().manual_seed(123)
    samples = []
    for file_index, filename in enumerate(("a.csv", "b.csv", "c.csv")):
        for window_index in range(file_index + 2):
            samples.append(
                {
                    "x_dyn": torch.randn(12, 5, generator=generator),
                    "t_mean": torch.randn(1, generator=generator),
                    "soc": torch.rand(12, 1, generator=generator),
                    "is_first": torch.tensor(window_index == 0),
                    "filenames": filename,
                }
            )

    dataset = MemoryDataset(samples)
    scaler = PITDScaler()
    scaler.stats = {
        "x_mean": torch.zeros(5),
        "x_std": torch.ones(5),
        "t_mean": torch.tensor(0.0),
        "t_std": torch.tensor(1.0),
    }
    model = AdaLNModel(d_model=32, nhead=4, num_layers=1, use_causal=True)
    config = {"DEVICE": "cpu", "D_MODEL": 32}
    mae_one, files_one = evaluate_dataset(
        model,
        DataLoader(dataset, batch_size=1, shuffle=False),
        scaler,
        config,
        label="Smoke serial",
    )
    mae_parallel, files_parallel = evaluate_dataset(
        model,
        DataLoader(dataset, batch_size=4, shuffle=False),
        scaler,
        config,
        label="Smoke parallel",
    )
    _check(files_one.keys() == files_parallel.keys(), "Parallel evaluation changed file coverage")
    _check(abs(mae_one - mae_parallel) < 1e-6, "Parallel evaluation changed aggregate MAE")
    for filename in files_one:
        _check(
            abs(files_one[filename] - files_parallel[filename]) < 1e-6,
            f"Parallel evaluation changed MAE for {filename}",
        )


def main():
    parser = argparse.ArgumentParser(description="Validate the self-contained SOC workspace")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="also validate every processed segment filename and CSV header",
    )
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    _check_workspace_paths()
    for path in sorted(SHARED_DIR.glob("*.py")):
        _import_entrypoint(path)
    for path in sorted(BASELINE_DIR.glob("*.py")):
        _import_entrypoint(path)
    split_counts = _check_splits_and_dataset()
    if args.full_data:
        _check_all_segment_files()
    _check_models_and_loss()
    _check_parallel_evaluation()
    print(f"Split files: train={split_counts[0]}, val={split_counts[1]}, random={split_counts[2]}, fixed={split_counts[3]}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
