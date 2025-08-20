import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split


def preprocess_run(array, seq_len_total, stride):
    """
    Preprocess a single run into normalized sliding windows.

    Steps:
    1. Transpose the input and scale it by 1e4.
    2. Normalize values to the range [-1, 1].
    3. Generate sliding windows with a given length and stride.
    """
    # Step 1: transpose (C, T) -> (T, C) and scale
    final_data = torch.from_numpy(array.T).float() * 10000

    # Step 2: normalize to [-1, 1]
    min_val = final_data.min()
    max_val = final_data.max()
    x_01 = (final_data - min_val) / (max_val - min_val + 1e-8)  # scale to [0,1]
    x_m1_1 = x_01 * 2.0 - 1.0                                  # shift to [-1,1]
    x_m1_1 = x_m1_1.clamp(-1, 1)                               # ensure bounds

    # Step 3: construct sliding windows
    datas = []
    for i in range(0, x_m1_1.size(0) - seq_len_total + 1, stride):
        window = x_m1_1[i : i + seq_len_total]
        datas.append(window)

    return torch.stack(datas)  # shape: (num_windows, seq_len_total, C)


def make_loader(ds, bs, shuffle=False):
    """Helper to create a DataLoader with consistent settings."""
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


def build_loaders(train_data, test_data, seq_len_total, stride, batch_size, seed=42):
    """
    Build train/val/test loaders from raw training and testing runs.

    Args:
        train_data: list of training runs, each of shape (C, T)
        test_data: list of testing runs, each of shape (C, T)
        seq_len_total: length of each sliding window
        stride: stride for sliding window
        batch_size: batch size for DataLoaders
        seed: random seed for reproducibility
    """
    # ---- Step 1: process all training runs into windows ----
    all_train_windows = []
    for run in train_data:  # run shape: (C, T)
        windows = preprocess_run(run, seq_len_total, stride)
        all_train_windows.append(windows)
    all_train_windows = torch.cat(all_train_windows, dim=0)

    # ---- Step 2: split into training/validation sets ----
    N = all_train_windows.size(0)
    train_len = int(0.8 * N)
    val_len = N - train_len
    full_train_ds = TensorDataset(all_train_windows)
    train_ds, val_ds = random_split(
        full_train_ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    # ---- Step 3: process all test runs ----
    all_test_windows = []
    for run in test_data:
        windows = preprocess_run(run, seq_len_total, stride)
        all_test_windows.append(windows)
    all_test_windows = torch.cat(all_test_windows, dim=0)
    test_ds = TensorDataset(all_test_windows)

    # ---- Step 4: build DataLoaders ----
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader   = make_loader(val_ds, batch_size, shuffle=False)
    test_loader  = make_loader(test_ds, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_loo_loaders(all_runs, seq_len_total, stride, batch_size, seed=42):
    """
    Build DataLoaders for leave-one-out (LOO) cross-validation.

    Each fold:
        - Leave one run out as the test set.
        - Use the remaining runs as the training set.
        - Split training set further into train/validation (80/20).
    """
    all_windows = []
    run_sizes = []  # record number of windows per run

    # Preprocess all runs and record sizes
    for run in all_runs:
        windows = preprocess_run(run, seq_len_total, stride)
        all_windows.append(windows)
        run_sizes.append(len(windows))

    total_runs = len(all_runs)
    fold_loaders = []

    for test_idx in range(total_runs):
        # ---- Test set: the held-out run ----
        test_ds = TensorDataset(all_windows[test_idx])
        test_loader = make_loader(test_ds, batch_size, shuffle=False)

        # ---- Training set: all other runs ----
        train_runs = [all_windows[i] for i in range(total_runs) if i != test_idx]
        if not train_runs:  # edge case: no training data
            continue

        # Concatenate all training runs
        full_train = torch.cat(train_runs, dim=0)
        full_train_ds = TensorDataset(full_train)

        # Split train/val (80/20)
        N = len(full_train_ds)
        train_len = int(0.8 * N)
        val_len = N - train_len
        train_ds, val_ds = random_split(
            full_train_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = make_loader(train_ds, batch_size, shuffle=True)
        val_loader   = make_loader(val_ds, batch_size, shuffle=False)

        fold_loaders.append((train_loader, val_loader, test_loader, test_idx))

    return fold_loaders
