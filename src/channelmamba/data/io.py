"""MAT file IO helpers."""

from __future__ import annotations

from pathlib import Path

try:
    import hdf5storage
except ImportError:  # pragma: no cover - optional dependency
    hdf5storage = None

try:
    from scipy.io import loadmat as scipy_loadmat
    from scipy.io import savemat as scipy_savemat
except ImportError:  # pragma: no cover - optional dependency
    scipy_loadmat = None
    scipy_savemat = None


def _loadmat(path: str | Path) -> dict:
    if hdf5storage is not None:
        return hdf5storage.loadmat(path)
    if scipy_loadmat is not None:
        return scipy_loadmat(path)
    raise ImportError("Either hdf5storage or scipy.io is required to read .mat files.")


def load_mat_array(path: str | Path, key: str):
    payload = _loadmat(path)
    if key not in payload:
        raise KeyError(f"Key '{key}' not found in MAT file: {path}")
    return payload[key]


def save_mat_mapping(path: str | Path, mapping: dict) -> Path:
    if scipy_savemat is None:
        raise ImportError("scipy.io.savemat is required to create synthetic .mat files.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scipy_savemat(target, mapping)
    return target
