import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def load_mapping(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    df = pd.read_csv(path)
    if not {"index", "class"}.issubset(df.columns):
        raise ValueError(f"Mapping file {path} must contain 'index' and 'class' columns.")
    return dict(zip(df["index"], df["class"]))


def ensure_single_superclass(df: pd.DataFrame) -> Dict[int, int]:
    """Verify each subclass maps to exactly one superclass and return the mapping."""
    mapping = {}
    for subclass_idx, group in df.groupby("subclass_index"):
        supers = group["superclass_index"].unique()
        if len(supers) != 1:
            raise ValueError(
                f"Subclass {subclass_idx} maps to multiple superclasses: {supers}."
                " Check the dataset integrity."
            )
        mapping[subclass_idx] = int(supers[0])
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dataset class distribution.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("project_data"),
        help="Path to directory containing train_data.csv and mapping files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write summary files.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=30,
        help="Threshold below which a subclass is flagged as rare.",
    )
    args = parser.parse_args()

    train_csv = args.data_dir / "train_data.csv"
    superclass_map_path = args.data_dir / "superclass_mapping.csv"
    subclass_map_path = args.data_dir / "subclass_mapping.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training metadata not found: {train_csv}")

    # Load metadata
    train_df = pd.read_csv(train_csv)
    if not {"image", "superclass_index", "subclass_index"}.issubset(train_df.columns):
        raise ValueError("train_data.csv must contain image, superclass_index, subclass_index columns.")

    superclass_names = load_mapping(superclass_map_path)
    subclass_names = load_mapping(subclass_map_path)
    subclass_to_super = ensure_single_superclass(train_df)

    # Counts
    subclass_counts = (
        train_df.groupby("subclass_index")
        .size()
        .reset_index(name="count")
        .assign(superclass_index=lambda df: df["subclass_index"].map(subclass_to_super))
    )
    subclass_counts["subclass_name"] = subclass_counts["subclass_index"].map(subclass_names)
    subclass_counts["superclass_name"] = subclass_counts["superclass_index"].map(superclass_names)
    subclass_counts = subclass_counts[["subclass_index", "subclass_name", "superclass_index", "superclass_name", "count"]]
    subclass_counts = subclass_counts.sort_values(["superclass_index", "count"], ascending=[True, False])

    superclass_counts = (
        train_df.groupby("superclass_index")
        .size()
        .reset_index(name="count")
        .assign(superclass_name=lambda df: df["superclass_index"].map(superclass_names))
        .sort_values("superclass_index")
    )

    rare_subclasses = subclass_counts[subclass_counts["count"] < args.rare_threshold]

    # Output files
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subclass_counts.to_csv(args.output_dir / "data_inventory_subclasses.csv", index=False)
    superclass_counts.to_csv(args.output_dir / "data_inventory_superclasses.csv", index=False)

    summary = {
        "total_images": int(len(train_df)),
        "num_superclasses": int(superclass_counts.shape[0]),
        "num_subclasses": int(subclass_counts.shape[0]),
        "min_subclass_count": int(subclass_counts["count"].min()),
        "max_subclass_count": int(subclass_counts["count"].max()),
        "rare_threshold": args.rare_threshold,
        "num_rare_subclasses": int(rare_subclasses.shape[0]),
        "rare_subclasses": rare_subclasses[["subclass_index", "subclass_name", "superclass_name", "count"]]
        .to_dict(orient="records"),
    }

    with open(args.output_dir / "data_inventory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Total images: {summary['total_images']}")
    print("Superclass counts:")
    for _, row in superclass_counts.iterrows():
        print(f"  {row['superclass_index']} ({row['superclass_name']}): {row['count']}")

    print("\nTop subclasses by count:")
    for _, row in subclass_counts.head(10).iterrows():
        print(f"  {row['subclass_index']} ({row['subclass_name']}): {row['count']}")

    if not rare_subclasses.empty:
        print(f"\nRare subclasses (< {args.rare_threshold} images):")
        for _, row in rare_subclasses.iterrows():
            print(f"  {row['subclass_index']} ({row['subclass_name']}): {row['count']}")
    else:
        print(f"\nNo subclasses below {args.rare_threshold} images.")


if __name__ == "__main__":
    main()
