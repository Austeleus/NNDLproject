import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .dataset import get_subclasses_by_superclass


@dataclass
class DataSplits:
    train_seen: pd.DataFrame
    val_seen: pd.DataFrame
    oe: pd.DataFrame
    unseen_val: pd.DataFrame

    oe_subclasses: List[int]
    unseen_subclasses: List[int]
    seen_subclasses: List[int]


def create_splits(
    data_dir: Path,
    oe_per_super: int = 1,
    unseen_per_super: int = 1,
    train_ratio: float = 0.75,
    seed: int = 42,
    min_samples_per_class: int = 50,
) -> DataSplits:
    random.seed(seed)

    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train_data.csv")

    super_to_subs = get_subclasses_by_superclass(data_dir)

    subclass_counts = train_df.groupby("subclass_index").size().to_dict()

    oe_subclasses = []
    unseen_subclasses = []
    seen_subclasses = []

    for super_idx in [0, 1, 2]:
        subs = super_to_subs[super_idx]
        eligible = [s for s in subs if subclass_counts.get(s, 0) >= min_samples_per_class]

        if len(eligible) < oe_per_super + unseen_per_super:
            eligible = subs

        random.shuffle(eligible)

        oe_picks = eligible[:oe_per_super]
        unseen_picks = eligible[oe_per_super : oe_per_super + unseen_per_super]
        seen_picks = [s for s in subs if s not in oe_picks and s not in unseen_picks]

        oe_subclasses.extend(oe_picks)
        unseen_subclasses.extend(unseen_picks)
        seen_subclasses.extend(seen_picks)

    oe_df = train_df[train_df["subclass_index"].isin(oe_subclasses)]
    unseen_df = train_df[train_df["subclass_index"].isin(unseen_subclasses)]
    seen_df = train_df[train_df["subclass_index"].isin(seen_subclasses)]

    train_dfs = []
    val_dfs = []

    for sub_idx in seen_subclasses:
        sub_data = seen_df[seen_df["subclass_index"] == sub_idx].copy()
        sub_data = sub_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_train = int(len(sub_data) * train_ratio)
        train_dfs.append(sub_data.iloc[:n_train])
        val_dfs.append(sub_data.iloc[n_train:])

    train_seen_df = pd.concat(train_dfs, ignore_index=True)
    val_seen_df = pd.concat(val_dfs, ignore_index=True)

    return DataSplits(
        train_seen=train_seen_df,
        val_seen=val_seen_df,
        oe=oe_df,
        unseen_val=unseen_df,
        oe_subclasses=oe_subclasses,
        unseen_subclasses=unseen_subclasses,
        seen_subclasses=seen_subclasses,
    )


def save_splits_info(splits: DataSplits, output_path: Path) -> None:
    info = {
        "train_seen_count": len(splits.train_seen),
        "val_seen_count": len(splits.val_seen),
        "oe_count": len(splits.oe),
        "unseen_val_count": len(splits.unseen_val),
        "oe_subclasses": splits.oe_subclasses,
        "unseen_subclasses": splits.unseen_subclasses,
        "seen_subclasses": splits.seen_subclasses,
        "num_seen_subclasses": len(splits.seen_subclasses),
    }

    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)


def load_splits_from_info(data_dir: Path, info_path: Path) -> DataSplits:
    with open(info_path) as f:
        info = json.load(f)

    train_df = pd.read_csv(Path(data_dir) / "train_data.csv")

    oe_subclasses = info["oe_subclasses"]
    unseen_subclasses = info["unseen_subclasses"]
    seen_subclasses = info["seen_subclasses"]

    oe_df = train_df[train_df["subclass_index"].isin(oe_subclasses)]
    unseen_df = train_df[train_df["subclass_index"].isin(unseen_subclasses)]
    seen_df = train_df[train_df["subclass_index"].isin(seen_subclasses)]

    train_dfs = []
    val_dfs = []
    seed = 42
    train_ratio = 0.75

    for sub_idx in seen_subclasses:
        sub_data = seen_df[seen_df["subclass_index"] == sub_idx].copy()
        sub_data = sub_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_train = int(len(sub_data) * train_ratio)
        train_dfs.append(sub_data.iloc[:n_train])
        val_dfs.append(sub_data.iloc[n_train:])

    train_seen_df = pd.concat(train_dfs, ignore_index=True)
    val_seen_df = pd.concat(val_dfs, ignore_index=True)

    return DataSplits(
        train_seen=train_seen_df,
        val_seen=val_seen_df,
        oe=oe_df,
        unseen_val=unseen_df,
        oe_subclasses=oe_subclasses,
        unseen_subclasses=unseen_subclasses,
        seen_subclasses=seen_subclasses,
    )
