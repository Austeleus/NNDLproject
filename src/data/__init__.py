from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import Config
from .dataset import HierarchicalDataset, TestDataset, load_class_mappings
from .splits import DataSplits, create_splits
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


@dataclass
class DataLoaders:
    train: DataLoader
    val_seen: DataLoader
    val_unseen: DataLoader
    oe: DataLoader


def create_weighted_sampler(df, num_samples: Optional[int] = None) -> WeightedRandomSampler:
    class_counts = df["subclass_index"].value_counts().to_dict()
    weights = [1.0 / class_counts[row["subclass_index"]] for _, row in df.iterrows()]

    if num_samples is None:
        num_samples = len(weights)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )


def create_dataloaders(config: Config, splits: DataSplits) -> DataLoaders:
    data_dir = config.data.data_dir

    train_transform = get_train_transforms(config.augmentation, config.data.image_size)
    val_transform = get_val_transforms(config.augmentation, config.data.image_size)

    train_dataset = HierarchicalDataset(
        data_dir=data_dir,
        df=splits.train_seen,
        transform=train_transform,
        is_oe=False,
    )

    val_seen_dataset = HierarchicalDataset(
        data_dir=data_dir,
        df=splits.val_seen,
        transform=val_transform,
        is_oe=False,
    )

    val_unseen_dataset = HierarchicalDataset(
        data_dir=data_dir,
        df=splits.unseen_val,
        transform=val_transform,
        is_oe=False,
    )

    oe_dataset = HierarchicalDataset(
        data_dir=data_dir,
        df=splits.oe,
        transform=train_transform,
        is_oe=True,
    )

    train_sampler = create_weighted_sampler(splits.train_seen)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )

    val_seen_loader = DataLoader(
        val_seen_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    val_unseen_loader = DataLoader(
        val_unseen_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    oe_loader = DataLoader(
        oe_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )

    return DataLoaders(
        train=train_loader,
        val_seen=val_seen_loader,
        val_unseen=val_unseen_loader,
        oe=oe_loader,
    )


def create_test_dataloader(config: Config) -> DataLoader:
    test_transform = get_test_transforms(config.augmentation, config.data.image_size)

    test_dataset = TestDataset(
        data_dir=config.data.data_dir,
        transform=test_transform,
    )

    return DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )


__all__ = [
    "DataLoaders",
    "DataSplits",
    "HierarchicalDataset",
    "TestDataset",
    "create_dataloaders",
    "create_splits",
    "create_test_dataloader",
    "create_weighted_sampler",
    "load_class_mappings",
]
