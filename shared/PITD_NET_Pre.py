from __future__ import annotations

from scaling import PITDScaler
from soccc_schemeB import split_soccc_by_cells
from train_utils import PITDPhysicsLoss


def split_data_by_files(data_dir):
    """
    Backward-compatible wrapper for legacy code.

    Returns train, val, and a merged test list so old callers that expect
    three outputs can still work.
    """
    train_f, val_f, test_random_f, test_fixed_f = split_soccc_by_cells(data_dir)
    return train_f, val_f, test_random_f + test_fixed_f
