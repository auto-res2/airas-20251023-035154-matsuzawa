"""src/preprocess.py
Data-loading utilities.  Currently supports the Fashion-MNIST dataset but
is written so that additional datasets can be added with minimal effort.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST

################################################################################
#                         transformation helpers                               #
################################################################################

def _parse_transforms(entries):
    """Convert YAML-style transform specifications into torchvision objects."""
    tfs = []
    for e in entries:
        if isinstance(e, str):
            e_lc = e.lower()
            if e_lc == "totensor":
                tfs.append(T.ToTensor())
            elif e_lc.startswith("normalize"):
                # Format: Normalize(mean=0.5, std=0.5)
                mean, std = 0.5, 0.5
                if "(" in e and ")" in e:
                    inner = e[e.find("(") + 1 : e.find(")")]
                    parts = {k.strip(): float(v) for k, v in (p.split("=") for p in inner.split(","))}
                    mean, std = parts.get("mean", 0.5), parts.get("std", 0.5)
                tfs.append(T.Normalize(mean=[mean], std=[std]))
            else:
                raise ValueError(f"Unknown transform: {e}")
        elif isinstance(e, dict):
            if "Normalize" in e:
                m = e["Normalize"].get("mean", 0.5)
                s = e["Normalize"].get("std", 0.5)
                tfs.append(T.Normalize(mean=[m], std=[s]))
            else:
                raise ValueError(f"Unsupported dict transform: {e}")
        else:
            raise TypeError(f"Transform must be str or dict, got: {type(e)}")
    return T.Compose(tfs)

################################################################################
#                           public interface                                   #
################################################################################

def get_dataloaders(ds_cfg: DictConfig, tr_cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test dataloaders given Hydra configs."""
    if ds_cfg.name.replace("_", "-").lower() != "fashion-mnist":
        raise NotImplementedError("Currently only Fashion-MNIST is supported.")

    tf = _parse_transforms(ds_cfg.transforms)
    root = ".cache/torchvision/fashion_mnist"

    train_full = FashionMNIST(root=root, train=True, download=True, transform=tf)
    test_set = FashionMNIST(root=root, train=False, download=True, transform=tf)

    val_sz = int(len(train_full) * ds_cfg.validation_split)
    train_sz = len(train_full) - val_sz
    train_set, val_set = random_split(train_full, [train_sz, val_sz])

    dl_kwargs = dict(
        batch_size=tr_cfg.batch_size,
        num_workers=tr_cfg.num_workers,
        pin_memory=tr_cfg.pin_memory,
    )
    train_loader = DataLoader(train_set, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **dl_kwargs)

    return train_loader, val_loader, test_loader
