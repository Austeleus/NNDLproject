from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models import HierarchicalClassifier
from ..models.mahalanobis import MahalanobisNoveltyDetector
from .calibration import CalibrationResult


@dataclass
class PredictionResult:
    super_pred: int
    sub_pred: int
    super_probs: torch.Tensor
    sub_probs: torch.Tensor
    super_novel_prob: float
    sub_novel_prob: float


class MSPPredictor:
    """
    Maximum Softmax Probability based predictor.
    Uses confidence on known classes to detect novel samples.
    If max probability of known classes < threshold, predict novel.
    """

    def __init__(
        self,
        model: HierarchicalClassifier,
        config: Config,
        msp_threshold_sub: float = 0.6,
        temperature: float = 1.0,
        subclass_to_superclass: Optional[Dict[int, int]] = None,
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.temperature = temperature
        self.msp_threshold_sub = msp_threshold_sub

        self.num_superclasses = config.model.num_superclasses
        self.num_subclasses = config.model.num_subclasses
        self.novel_super_idx = config.model.num_superclasses
        self.novel_sub_idx = config.model.num_subclasses

        self.subclass_to_superclass = subclass_to_superclass or {}

    def predict_batch(
        self,
        images: torch.Tensor,
        msp_threshold_super: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        images = images.to(self.device)
        batch_size = images.size(0)

        # Use instance threshold or passed threshold
        super_thresh = msp_threshold_super if msp_threshold_super is not None else getattr(self, 'msp_threshold_super', None)

        with torch.no_grad():
            output = self.model(images)

            # Superclass: MSP-based novel detection if threshold provided
            super_probs_known = F.softmax(
                output.super_logits[:, :self.num_superclasses] / self.temperature,
                dim=1,
            )
            super_msp = super_probs_known.max(dim=1).values
            super_preds = super_probs_known.argmax(dim=1)

            # If superclass MSP < threshold, predict novel
            if super_thresh is not None:
                super_novel_mask = super_msp < super_thresh
                super_preds[super_novel_mask] = self.novel_super_idx

            # Subclass: MSP-based novel detection
            sub_probs_known = F.softmax(
                output.sub_logits[:, :self.num_subclasses] / self.temperature,
                dim=1,
            )
            sub_msp = sub_probs_known.max(dim=1).values
            sub_preds = sub_probs_known.argmax(dim=1)

            # If MSP < threshold, predict novel
            novel_mask = sub_msp < self.msp_threshold_sub
            sub_preds[novel_mask] = self.novel_sub_idx

            # If superclass is novel, subclass must also be novel
            if super_thresh is not None:
                sub_preds[super_novel_mask] = self.novel_sub_idx

        return super_preds, sub_preds


class HierarchicalPredictor:
    def __init__(
        self,
        model: HierarchicalClassifier,
        config: Config,
        calibration: Optional[CalibrationResult] = None,
        subclass_to_superclass: Optional[Dict[int, int]] = None,
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        if calibration:
            self.temperature = calibration.temperature
            self.threshold_super = calibration.threshold_super
            self.threshold_sub = calibration.threshold_sub
        else:
            self.temperature = config.inference.temperature
            self.threshold_super = config.inference.threshold_super
            self.threshold_sub = config.inference.threshold_sub

        self.num_superclasses = config.model.num_superclasses
        self.num_subclasses = config.model.num_subclasses
        self.novel_super_idx = config.model.num_superclasses
        self.novel_sub_idx = config.model.num_subclasses

        self.subclass_to_superclass = subclass_to_superclass or {}
        self._build_superclass_masks()

    def _build_superclass_masks(self) -> None:
        self.super_to_sub_mask = {}

        for super_idx in range(self.num_superclasses):
            mask = torch.zeros(self.num_subclasses + 1, dtype=torch.bool)

            for sub_idx, s_idx in self.subclass_to_superclass.items():
                if s_idx == super_idx:
                    mask[sub_idx] = True

            mask[self.novel_sub_idx] = True

            self.super_to_sub_mask[super_idx] = mask

    def predict_single(self, image: torch.Tensor) -> PredictionResult:
        self.model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)

            super_logits = output.super_logits / self.temperature
            sub_logits = output.sub_logits / self.temperature

            super_probs = F.softmax(super_logits, dim=1)[0]
            sub_probs = F.softmax(sub_logits, dim=1)[0]

        super_novel_prob = super_probs[self.novel_super_idx].item()

        if super_novel_prob > self.threshold_super:
            return PredictionResult(
                super_pred=self.novel_super_idx,
                sub_pred=self.novel_sub_idx,
                super_probs=super_probs,
                sub_probs=sub_probs,
                super_novel_prob=super_novel_prob,
                sub_novel_prob=sub_probs[self.novel_sub_idx].item(),
            )

        super_pred = super_probs[:self.num_superclasses].argmax().item()

        if super_pred in self.super_to_sub_mask:
            mask = self.super_to_sub_mask[super_pred].to(self.device)
            masked_sub_logits = sub_logits.clone()
            masked_sub_logits[0, ~mask] = float("-inf")
            sub_probs_masked = F.softmax(masked_sub_logits, dim=1)[0]
        else:
            sub_probs_masked = sub_probs

        sub_novel_prob = sub_probs_masked[self.novel_sub_idx].item()

        if sub_novel_prob > self.threshold_sub:
            sub_pred = self.novel_sub_idx
        else:
            valid_sub_probs = sub_probs_masked[:self.num_subclasses]
            sub_pred = valid_sub_probs.argmax().item()

        return PredictionResult(
            super_pred=super_pred,
            sub_pred=sub_pred,
            super_probs=super_probs,
            sub_probs=sub_probs_masked,
            super_novel_prob=super_novel_prob,
            sub_novel_prob=sub_novel_prob,
        )

    def predict_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            output = self.model(images)

            super_logits = output.super_logits / self.temperature
            sub_logits = output.sub_logits / self.temperature

            super_probs = F.softmax(super_logits, dim=1)

        batch_size = images.size(0)
        super_preds = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        sub_preds = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        super_novel_mask = super_probs[:, self.novel_super_idx] > self.threshold_super
        super_preds[super_novel_mask] = self.novel_super_idx
        sub_preds[super_novel_mask] = self.novel_sub_idx

        non_novel_mask = ~super_novel_mask
        if non_novel_mask.any():
            super_preds[non_novel_mask] = super_probs[non_novel_mask, :self.num_superclasses].argmax(dim=1)

            # Apply hierarchical masking: for each sample, mask invalid subclasses
            for i in range(batch_size):
                if not non_novel_mask[i]:
                    continue

                sp = super_preds[i].item()
                masked_sub_logits = sub_logits[i].clone()

                # Mask out subclasses that don't belong to the predicted superclass
                if sp in self.super_to_sub_mask:
                    mask = self.super_to_sub_mask[sp].to(self.device)
                    masked_sub_logits[~mask] = float("-inf")

                sub_probs_masked = F.softmax(masked_sub_logits.unsqueeze(0), dim=1)[0]

                # Check if novel subclass probability exceeds threshold
                if sub_probs_masked[self.novel_sub_idx] > self.threshold_sub:
                    sub_preds[i] = self.novel_sub_idx
                else:
                    sub_preds[i] = sub_probs_masked[:self.num_subclasses].argmax()

        return super_preds, sub_preds

    def predict_dataloader(
        self,
        dataloader: DataLoader,
        return_probs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()

        all_super_preds = []
        all_sub_preds = []
        all_super_probs = [] if return_probs else None
        all_sub_probs = [] if return_probs else None
        all_filenames = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                images = batch["image"].to(self.device)

                super_preds, sub_preds = self.predict_batch(images)

                all_super_preds.append(super_preds.cpu())
                all_sub_preds.append(sub_preds.cpu())

                if "filename" in batch:
                    all_filenames.extend(batch["filename"])

                if return_probs:
                    output = self.model(images)
                    super_probs = F.softmax(output.super_logits / self.temperature, dim=1)
                    sub_probs = F.softmax(output.sub_logits / self.temperature, dim=1)
                    all_super_probs.append(super_probs.cpu())
                    all_sub_probs.append(sub_probs.cpu())

        result = {
            "super_preds": torch.cat(all_super_preds),
            "sub_preds": torch.cat(all_sub_preds),
        }

        if all_filenames:
            result["filenames"] = all_filenames

        if return_probs:
            result["super_probs"] = torch.cat(all_super_probs)
            result["sub_probs"] = torch.cat(all_sub_probs)

        return result

    def generate_submission(
        self,
        dataloader: DataLoader,
        superclass_names: Dict[int, str],
        subclass_names: Dict[int, str],
        output_path: Path,
    ) -> pd.DataFrame:
        predictions = self.predict_dataloader(dataloader)

        super_preds = predictions["super_preds"].numpy()
        sub_preds = predictions["sub_preds"].numpy()
        filenames = predictions.get("filenames", [f"{i}.jpg" for i in range(len(super_preds))])

        rows = []
        for fname, sp, subp in zip(filenames, super_preds, sub_preds):
            super_name = superclass_names.get(sp, "novel")
            sub_name = subclass_names.get(subp, "novel")

            rows.append({
                "image": fname,
                "superclass": super_name,
                "subclass": sub_name,
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        return df


class MahalanobisPredictor:
    """
    Predictor that uses Mahalanobis distance for novel detection.
    Computes distance to class centroids in feature space.
    """

    def __init__(
        self,
        model: HierarchicalClassifier,
        config: Config,
        subclass_to_superclass: Optional[Dict[int, int]] = None,
        temperature: float = 1.0,
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.temperature = temperature

        self.num_superclasses = config.model.num_superclasses
        self.num_subclasses = config.model.num_subclasses
        self.novel_super_idx = config.model.num_superclasses
        self.novel_sub_idx = config.model.num_subclasses

        self.subclass_to_superclass = subclass_to_superclass or {}
        self._build_superclass_masks()

        feature_dim = model.feature_dim

        self.mahal_super = MahalanobisNoveltyDetector(
            feature_dim=feature_dim,
            num_classes=self.num_superclasses,
        ).to(self.device)

        self.mahal_sub = MahalanobisNoveltyDetector(
            feature_dim=feature_dim,
            num_classes=self.num_subclasses,
        ).to(self.device)

        self.threshold_super = 10.0
        self.threshold_sub = 10.0

    def _build_superclass_masks(self) -> None:
        self.super_to_sub_mask = {}
        for super_idx in range(self.num_superclasses):
            mask = torch.zeros(self.num_subclasses + 1, dtype=torch.bool)
            for sub_idx, s_idx in self.subclass_to_superclass.items():
                if s_idx == super_idx:
                    mask[sub_idx] = True
            mask[self.novel_sub_idx] = True
            self.super_to_sub_mask[super_idx] = mask

    def fit(self, dataloader: DataLoader) -> None:
        """
        Fit Mahalanobis detectors on training data features.
        """
        self.model.eval()

        all_features = []
        all_super_labels = []
        all_sub_labels = []

        print("Extracting features for Mahalanobis fitting...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images = batch["image"].to(self.device)
                super_labels = batch["superclass"]
                sub_labels = batch["subclass"]

                output = self.model(images)
                features = output.features

                all_features.append(features.cpu())
                all_super_labels.append(super_labels)
                all_sub_labels.append(sub_labels)

        features = torch.cat(all_features, dim=0).to(self.device)
        super_labels = torch.cat(all_super_labels, dim=0).to(self.device)
        sub_labels = torch.cat(all_sub_labels, dim=0).to(self.device)

        print(f"Fitting Mahalanobis on {features.size(0)} samples...")

        self.mahal_super.fit(features, super_labels)
        print(f"  Superclass detector fitted")

        self.mahal_sub.fit(features, sub_labels)
        print(f"  Subclass detector fitted")

    def tune_thresholds(
        self,
        seen_loader: DataLoader,
        unseen_loader: DataLoader,
        threshold_range: Tuple[float, float] = (1.0, 30.0),
        num_steps: int = 60,
    ) -> Tuple[float, float]:
        """
        Tune thresholds on validation data to maximize F1 for novel detection.
        """
        self.model.eval()

        seen_super_dists = []
        seen_sub_dists = []
        unseen_super_dists = []
        unseen_sub_dists = []

        with torch.no_grad():
            for batch in tqdm(seen_loader, desc="Seen distances"):
                images = batch["image"].to(self.device)
                output = self.model(images)
                features = output.features

                _, min_super = self.mahal_super.mahalanobis_distance(features)
                _, min_sub = self.mahal_sub.mahalanobis_distance(features)

                seen_super_dists.append(min_super.cpu())
                seen_sub_dists.append(min_sub.cpu())

            for batch in tqdm(unseen_loader, desc="Unseen distances"):
                images = batch["image"].to(self.device)
                output = self.model(images)
                features = output.features

                _, min_super = self.mahal_super.mahalanobis_distance(features)
                _, min_sub = self.mahal_sub.mahalanobis_distance(features)

                unseen_super_dists.append(min_super.cpu())
                unseen_sub_dists.append(min_sub.cpu())

        seen_super = torch.cat(seen_super_dists)
        seen_sub = torch.cat(seen_sub_dists)
        unseen_super = torch.cat(unseen_super_dists)
        unseen_sub = torch.cat(unseen_sub_dists)

        print(f"\nDistance statistics:")
        print(f"  Seen super: mean={seen_super.mean():.2f}, std={seen_super.std():.2f}")
        print(f"  Unseen super: mean={unseen_super.mean():.2f}, std={unseen_super.std():.2f}")
        print(f"  Seen sub: mean={seen_sub.mean():.2f}, std={seen_sub.std():.2f}")
        print(f"  Unseen sub: mean={unseen_sub.mean():.2f}, std={unseen_sub.std():.2f}")

        thresholds = torch.linspace(threshold_range[0], threshold_range[1], num_steps)

        best_thresh_super = 10.0
        best_f1_super = 0.0
        for thresh in thresholds:
            tp = (unseen_super > thresh).sum().item()
            fp = (seen_super > thresh).sum().item()
            fn = (unseen_super <= thresh).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best_f1_super:
                best_f1_super = f1
                best_thresh_super = thresh.item()

        best_thresh_sub = 10.0
        best_f1_sub = 0.0
        for thresh in thresholds:
            tp = (unseen_sub > thresh).sum().item()
            fp = (seen_sub > thresh).sum().item()
            fn = (unseen_sub <= thresh).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best_f1_sub:
                best_f1_sub = f1
                best_thresh_sub = thresh.item()

        self.threshold_super = best_thresh_super
        self.threshold_sub = best_thresh_sub

        print(f"\nOptimal thresholds:")
        print(f"  Super: {best_thresh_super:.2f} (F1={best_f1_super:.3f})")
        print(f"  Sub: {best_thresh_sub:.2f} (F1={best_f1_sub:.3f})")

        return best_thresh_super, best_thresh_sub

    def predict_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        images = images.to(self.device)
        batch_size = images.size(0)

        with torch.no_grad():
            output = self.model(images)
            features = output.features

            super_logits = output.super_logits[:, :self.num_superclasses] / self.temperature
            sub_logits = output.sub_logits[:, :self.num_subclasses] / self.temperature

            super_probs = F.softmax(super_logits, dim=1)
            sub_probs = F.softmax(sub_logits, dim=1)

            _, super_dists = self.mahal_super.mahalanobis_distance(features)
            _, sub_dists = self.mahal_sub.mahalanobis_distance(features)

        super_preds = super_probs.argmax(dim=1)
        sub_preds = sub_probs.argmax(dim=1)

        super_novel_mask = super_dists > self.threshold_super
        super_preds[super_novel_mask] = self.novel_super_idx

        sub_novel_mask = sub_dists > self.threshold_sub
        sub_preds[sub_novel_mask] = self.novel_sub_idx

        sub_preds[super_novel_mask] = self.novel_sub_idx

        return super_preds, sub_preds

    def save(self, path: Path) -> None:
        """Save Mahalanobis detector state."""
        torch.save({
            "mahal_super_state": self.mahal_super.state_dict(),
            "mahal_sub_state": self.mahal_sub.state_dict(),
            "threshold_super": self.threshold_super,
            "threshold_sub": self.threshold_sub,
            "temperature": self.temperature,
        }, path)
        print(f"Mahalanobis predictor saved to {path}")

    def load(self, path: Path) -> None:
        """Load Mahalanobis detector state."""
        state = torch.load(path, map_location=self.device)
        self.mahal_super.load_state_dict(state["mahal_super_state"])
        self.mahal_sub.load_state_dict(state["mahal_sub_state"])
        self.threshold_super = state["threshold_super"]
        self.threshold_sub = state["threshold_sub"]
        self.temperature = state.get("temperature", 1.0)
        print(f"Mahalanobis predictor loaded from {path}")
