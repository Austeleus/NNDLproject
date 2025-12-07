#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.config import Config
from src.data import create_test_dataloader, load_class_mappings
from src.data.dataset import get_subclasses_by_superclass
from src.models import create_model
from src.inference import HierarchicalPredictor, MSPPredictor, CalibrationResult


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for test set")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Path to calibration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--use-msp",
        action="store_true",
        help="Use MSP-based novel detection (recommended)",
    )
    parser.add_argument(
        "--msp-threshold",
        type=float,
        default=0.6,
        help="MSP threshold for subclass novel detection (default: 0.6)",
    )
    parser.add_argument(
        "--msp-threshold-super",
        type=float,
        default=None,
        help="MSP threshold for superclass novel detection (default: None = disabled)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.batch_size:
        config.data.batch_size = args.batch_size

    print("=" * 60)
    print("Test Set Prediction")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading model...")
    model = create_model(config.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    calibration = None
    if args.calibration and args.calibration.exists():
        print(f"\nLoading calibration from {args.calibration}...")
        cal_data = torch.load(args.calibration)
        calibration = CalibrationResult(
            temperature=cal_data["temperature"],
            threshold_super=cal_data["threshold_super"],
            threshold_sub=cal_data["threshold_sub"],
            val_ce_super=0.0,
            val_ce_sub=0.0,
            novel_f1_super=0.0,
            novel_f1_sub=0.0,
        )
        print(f"  Temperature: {calibration.temperature:.3f}")
        print(f"  Threshold super: {calibration.threshold_super:.3f}")
        print(f"  Threshold sub: {calibration.threshold_sub:.3f}")
    else:
        print("\nNo calibration file provided, using defaults from config")

    print("\nLoading class mappings...")
    superclass_names, subclass_names, sub_to_super = load_class_mappings(config.data.data_dir)

    superclass_names[config.model.num_superclasses] = "novel"
    subclass_names[config.model.num_subclasses] = "novel"

    print(f"  Superclasses: {len(superclass_names)}")
    print(f"  Subclasses: {len(subclass_names)}")

    print("\nCreating test dataloader...")
    test_loader = create_test_dataloader(config)
    print(f"  Test images: {len(test_loader.dataset)}")
    print(f"  Test batches: {len(test_loader)}")

    if args.use_msp:
        print(f"\nUsing MSP-based prediction")
        print(f"  Subclass threshold: {args.msp_threshold}")
        print(f"  Superclass threshold: {args.msp_threshold_super or 'disabled'}")
        predictor = MSPPredictor(
            model=model,
            config=config,
            msp_threshold_sub=args.msp_threshold,
            temperature=calibration.temperature if calibration else 1.0,
            subclass_to_superclass=sub_to_super,
        )
        # Store superclass threshold for predict_batch
        predictor.msp_threshold_super = args.msp_threshold_super
    else:
        predictor = HierarchicalPredictor(
            model=model,
            config=config,
            calibration=calibration,
            subclass_to_superclass=sub_to_super,
        )

    print("\nGenerating predictions...")
    all_filenames = []
    all_super_preds = []
    all_sub_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch["image"].to(device)
            filenames = batch["filename"]

            super_preds, sub_preds = predictor.predict_batch(images)

            all_filenames.extend(filenames)
            all_super_preds.extend(super_preds.cpu().tolist())
            all_sub_preds.extend(sub_preds.cpu().tolist())

    print("\nCreating submission file...")
    rows = []
    for fname, sp, subp in zip(all_filenames, all_super_preds, all_sub_preds):
        super_name = superclass_names.get(sp, "novel")
        sub_name = subclass_names.get(subp, "novel")

        img_id = fname.replace(".jpg", "").replace(".png", "")

        rows.append({
            "id": img_id,
            "superclass_index": sp,
            "subclass_index": subp,
            "superclass": super_name,
            "subclass": sub_name,
        })

    df = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nSubmission saved to {args.output}")
    print(f"  Total predictions: {len(df)}")

    print("\nPrediction distribution:")
    print("  Superclass:")
    super_counts = df["superclass"].value_counts()
    for name, count in super_counts.items():
        print(f"    {name}: {count} ({count/len(df)*100:.1f}%)")

    print("  Subclass (top 10):")
    sub_counts = df["subclass"].value_counts().head(10)
    for name, count in sub_counts.items():
        print(f"    {name}: {count} ({count/len(df)*100:.1f}%)")

    novel_super_pct = (df["superclass"] == "novel").mean() * 100
    novel_sub_pct = (df["subclass"] == "novel").mean() * 100
    print(f"\n  Novel predictions: super={novel_super_pct:.1f}%, sub={novel_sub_pct:.1f}%")

    print("\n" + "=" * 60)
    print("Prediction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
