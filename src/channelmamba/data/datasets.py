"""Dataset and tensor preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data as data
from einops import rearrange

from .io import load_mat_array


def _randn_like(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        return np.random.randn(*shape)
    return rng.standard_normal(shape)


def noise(values: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    sigma = 10 ** (-snr_db / 10)
    add_noise = np.sqrt(sigma / 2) * (_randn_like(values.shape, rng) + 1j * _randn_like(values.shape, rng))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(values) ** 2))
    return values + add_noise


def load_batch_ofdm(values: np.ndarray, num: int = 32) -> torch.Tensor:
    batch, steps, features = values.shape
    reshaped = rearrange(values, "b t (k a) -> (b a) t k", a=num)
    output = np.zeros((batch * num, steps, features // num, 2), dtype=np.float32)
    output[:, :, :, 0] = reshaped.real
    output[:, :, :, 1] = reshaped.imag
    return torch.tensor(output.reshape(batch * num, steps, features // num * 2), dtype=torch.float32)


def load_batch_ofdm_1(values: np.ndarray) -> torch.Tensor:
    batch, steps, features = values.shape
    output = np.zeros((batch, steps, features, 2), dtype=np.float32)
    output[:, :, :, 0] = values.real
    output[:, :, :, 1] = values.imag
    return torch.tensor(output.reshape(batch, steps, features * 2), dtype=torch.float32)


def load_batch_ofdm_2(values: np.ndarray) -> torch.Tensor:
    batch, antennas, steps, carriers = values.shape
    output = np.zeros((batch, antennas, steps, carriers, 2), dtype=np.float32)
    output[:, :, :, :, 0] = values.real
    output[:, :, :, :, 1] = values.imag
    return torch.tensor(output.reshape(batch, antennas, steps, carriers * 2), dtype=torch.float32)


def transform_tdd_fdd(values: torch.Tensor, n_t: int = 4, n_r: int = 4) -> torch.Tensor:
    reshaped = values.reshape(-1, n_t, n_r, 2)
    return torch.complex(reshaped[..., 0], reshaped[..., 1])


@dataclass
class ChannelSequenceDataset(data.Dataset):
    inputs: torch.Tensor
    targets: torch.Tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.inputs[index], "targets": self.targets[index]}

    def __len__(self) -> int:
        return self.inputs.shape[0]


def _select_split(values: np.ndarray, split: str, train_ratio: float, val_ratio: float) -> np.ndarray:
    batch = values.shape[1]
    train_end = int(train_ratio * batch)
    val_end = int((train_ratio + val_ratio) * batch)
    if split == "train":
        return values[:, :train_end, ...]
    if split == "val":
        return values[:, train_end:val_end, ...]
    raise ValueError(f"Unsupported split: {split}")


def _prepare_train_val_tensors(
    input_raw: np.ndarray,
    target_raw: np.ndarray,
    split: str,
    train_ratio: float,
    val_ratio: float,
    group_size: int,
    noise_min_snr_db: float,
    noise_max_snr_db: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    history = _select_split(input_raw, split, train_ratio, val_ratio)
    target = _select_split(target_raw, split, train_ratio, val_ratio)

    history = rearrange(history, "v n l k a b c -> (v n) l (k a b c)")
    target = rearrange(target, "v n l k a b c -> (v n) l (k a b c)")
    history = history.astype(np.complex64)
    target = target.astype(np.complex64)

    merged = np.concatenate((history, target), axis=1)
    order = rng.permutation(merged.shape[0])
    merged = merged[order]

    prev_len = history.shape[1]
    pred_len = target.shape[1]
    history = merged[:, :prev_len, ...]
    target = merged[:, -pred_len:, ...]

    for index in range(history.shape[0]):
        history[index, ...] = noise(history[index, ...], rng.uniform(noise_min_snr_db, noise_max_snr_db), rng)
        target[index, ...] = noise(target[index, ...], rng.uniform(noise_min_snr_db, noise_max_snr_db), rng)

    std = np.sqrt(np.std(np.abs(history) ** 2))
    if std > 0:
        history = history / std
        target = target / std

    target_tensor = load_batch_ofdm(target, num=group_size)
    history_tensor = load_batch_ofdm(history, num=group_size)
    return history_tensor, target_tensor


def build_train_val_datasets(
    input_path: str,
    target_path: str,
    input_key: str,
    target_key: str,
    train_ratio: float,
    val_ratio: float,
    group_size: int,
    noise_min_snr_db: float,
    noise_max_snr_db: float,
    seed: int,
) -> tuple[ChannelSequenceDataset, ChannelSequenceDataset]:
    input_raw = load_mat_array(input_path, input_key)
    target_raw = load_mat_array(target_path, target_key)
    train_inputs, train_targets = _prepare_train_val_tensors(
        input_raw=input_raw,
        target_raw=target_raw,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        group_size=group_size,
        noise_min_snr_db=noise_min_snr_db,
        noise_max_snr_db=noise_max_snr_db,
        seed=seed,
    )
    val_inputs, val_targets = _prepare_train_val_tensors(
        input_raw=input_raw,
        target_raw=target_raw,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        group_size=group_size,
        noise_min_snr_db=noise_min_snr_db,
        noise_max_snr_db=noise_max_snr_db,
        seed=seed + 1,
    )
    return (
        ChannelSequenceDataset(inputs=train_inputs, targets=train_targets),
        ChannelSequenceDataset(inputs=val_inputs, targets=val_targets),
    )


def prepare_eval_tensors(
    input_path: str,
    target_path: str,
    input_key: str,
    target_key: str,
    speed_index: int,
    snr_db: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_raw = load_mat_array(input_path, input_key)
    target_raw = load_mat_array(target_path, target_key)

    if input_raw.ndim != 7 or target_raw.ndim != 7:
        raise ValueError("Evaluation data must have shape [speed, sample, step, carrier, n, m, c].")
    if not 0 <= speed_index < input_raw.shape[0]:
        raise IndexError(f"Speed index out of range: {speed_index}")

    input_slice = input_raw[[speed_index], ...]
    target_slice = target_raw[[speed_index], ...]

    input_view = rearrange(input_slice, "v b l k n m c -> (v b c) (n m) l (k)")
    target_view = rearrange(target_slice, "v b l k n m c -> (v b c) (n m) l (k)")

    input_view = noise(input_view, snr_db)
    target_view = noise(target_view, snr_db)

    std = np.sqrt(np.std(np.abs(input_view) ** 2))
    if std > 0:
        input_view = input_view / std
        target_view = target_view / std

    return load_batch_ofdm_2(input_view), load_batch_ofdm_2(target_view)
