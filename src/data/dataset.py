from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HierarchicalDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        is_oe: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_oe = is_oe

        self.images_dir = self.data_dir / "train_images"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        img_path = self.images_dir / row["image"]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "superclass": torch.tensor(row["superclass_index"], dtype=torch.long),
            "subclass": torch.tensor(row["subclass_index"], dtype=torch.long),
            "is_oe": torch.tensor(self.is_oe, dtype=torch.bool),
        }


class TestDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images_dir = self.data_dir / "test_images"

        self.image_files = sorted(self.images_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "filename": img_path.name,
        }


def load_class_mappings(data_dir: Path) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, int]]:
    data_dir = Path(data_dir)

    super_df = pd.read_csv(data_dir / "superclass_mapping.csv")
    superclass_names = dict(zip(super_df["index"], super_df["class"]))

    sub_df = pd.read_csv(data_dir / "subclass_mapping.csv")
    subclass_names = dict(zip(sub_df["index"], sub_df["class"]))

    train_df = pd.read_csv(data_dir / "train_data.csv")
    subclass_to_superclass = {}
    for _, row in train_df.drop_duplicates("subclass_index").iterrows():
        subclass_to_superclass[row["subclass_index"]] = row["superclass_index"]

    return superclass_names, subclass_names, subclass_to_superclass


def get_subclasses_by_superclass(data_dir: Path) -> Dict[int, List[int]]:
    _, _, sub_to_super = load_class_mappings(data_dir)

    super_to_subs: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for sub_idx, super_idx in sub_to_super.items():
        if super_idx in super_to_subs:
            super_to_subs[super_idx].append(sub_idx)

    for super_idx in super_to_subs:
        super_to_subs[super_idx].sort()

    return super_to_subs
