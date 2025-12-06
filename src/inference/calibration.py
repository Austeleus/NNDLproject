from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import InferenceConfig
from ..models import HierarchicalClassifier


@dataclass
class CalibrationResult:
    temperature: float
    temperature_super: float
    temperature_sub: float
    threshold_super: float
    threshold_sub: float
    val_ce_super: float
    val_ce_sub: float
    novel_f1_super: float
    novel_f1_sub: float
    use_hybrid: bool = True
    hybrid_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)


class TemperatureScaler:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.temperature = 1.0

    def find_temperature(
        self,
        model: HierarchicalClassifier,
        dataloader: DataLoader,
        device: torch.device,
    ) -> float:
        model.eval()

        all_super_logits = []
        all_sub_logits = []
        all_super_targets = []
        all_sub_targets = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                super_targets = batch["superclass"].to(device)
                sub_targets = batch["subclass"].to(device)

                output = model(images)

                all_super_logits.append(output.super_logits.cpu())
                all_sub_logits.append(output.sub_logits.cpu())
                all_super_targets.append(super_targets.cpu())
                all_sub_targets.append(sub_targets.cpu())

        super_logits = torch.cat(all_super_logits)
        sub_logits = torch.cat(all_sub_logits)
        super_targets = torch.cat(all_super_targets)
        sub_targets = torch.cat(all_sub_targets)

        t_min, t_max = self.config.temperature_search_range
        num_steps = self.config.temperature_search_steps
        temperatures = torch.linspace(t_min, t_max, num_steps)

        best_temp = 1.0
        best_ce = float("inf")

        for t in temperatures:
            super_ce = F.cross_entropy(super_logits / t, super_targets).item()
            sub_ce = F.cross_entropy(sub_logits / t, sub_targets).item()
            total_ce = super_ce + sub_ce

            if total_ce < best_ce:
                best_ce = total_ce
                best_temp = t.item()

        self.temperature = best_temp
        return best_temp


class ThresholdTuner:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.threshold_super = config.threshold_super
        self.threshold_sub = config.threshold_sub

    def find_msp_thresholds(
        self,
        model: HierarchicalClassifier,
        seen_loader: DataLoader,
        unseen_loader: DataLoader,
        device: torch.device,
        temperature: float = 1.0,
        num_superclasses: int = 3,
        num_subclasses: int = 87,
    ) -> Tuple[float, float]:
        """Find optimal thresholds using MSP (max softmax probability).
        Novel is predicted when MSP < threshold."""
        model.eval()

        seen_sub_msp = []
        unseen_sub_msp = []

        with torch.no_grad():
            for batch in seen_loader:
                images = batch["image"].to(device)
                output = model(images)
                sub_probs = F.softmax(output.sub_logits[:, :num_subclasses] / temperature, dim=1)
                sub_msp = sub_probs.max(dim=1).values
                seen_sub_msp.append(sub_msp.cpu())

            for batch in unseen_loader:
                images = batch["image"].to(device)
                output = model(images)
                sub_probs = F.softmax(output.sub_logits[:, :num_subclasses] / temperature, dim=1)
                sub_msp = sub_probs.max(dim=1).values
                unseen_sub_msp.append(sub_msp.cpu())

        seen_sub = torch.cat(seen_sub_msp)
        unseen_sub = torch.cat(unseen_sub_msp)

        t_min, t_max = 0.3, 0.9
        num_steps = 61
        thresholds = torch.linspace(t_min, t_max, num_steps)

        best_thresh_sub = 0.6
        best_f1_sub = 0.0

        for thresh in thresholds:
            # MSP < thresh means novel
            tp = (unseen_sub < thresh).sum().item()
            fp = (seen_sub < thresh).sum().item()
            fn = (unseen_sub >= thresh).sum().item()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            if f1 > best_f1_sub:
                best_f1_sub = f1
                best_thresh_sub = thresh.item()

        self.threshold_sub = best_thresh_sub
        return 0.5, best_thresh_sub  # super threshold unused for MSP

    def find_thresholds(
        self,
        model: HierarchicalClassifier,
        seen_loader: DataLoader,
        unseen_loader: DataLoader,
        device: torch.device,
        temperature: float = 1.0,
    ) -> Tuple[float, float]:
        model.eval()

        seen_super_probs = []
        seen_sub_probs = []
        unseen_super_probs = []
        unseen_sub_probs = []

        with torch.no_grad():
            for batch in seen_loader:
                images = batch["image"].to(device)
                output = model(images)

                super_probs = F.softmax(output.super_logits / temperature, dim=1)
                sub_probs = F.softmax(output.sub_logits / temperature, dim=1)

                seen_super_probs.append(super_probs[:, -1].cpu())
                seen_sub_probs.append(sub_probs[:, -1].cpu())

            for batch in unseen_loader:
                images = batch["image"].to(device)
                output = model(images)

                super_probs = F.softmax(output.super_logits / temperature, dim=1)
                sub_probs = F.softmax(output.sub_logits / temperature, dim=1)

                unseen_super_probs.append(super_probs[:, -1].cpu())
                unseen_sub_probs.append(sub_probs[:, -1].cpu())

        seen_super = torch.cat(seen_super_probs)
        seen_sub = torch.cat(seen_sub_probs)
        unseen_super = torch.cat(unseen_super_probs)
        unseen_sub = torch.cat(unseen_sub_probs)

        t_min, t_max = self.config.threshold_search_range
        num_steps = self.config.threshold_search_steps
        thresholds = torch.linspace(t_min, t_max, num_steps)

        best_thresh_super = 0.5
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

        best_thresh_sub = 0.5
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

        return best_thresh_super, best_thresh_sub


def calibrate_model(
    model: HierarchicalClassifier,
    seen_loader: DataLoader,
    unseen_loader: DataLoader,
    config: InferenceConfig,
    device: torch.device,
) -> CalibrationResult:
    temp_scaler = TemperatureScaler(config)
    temperature = temp_scaler.find_temperature(model, seen_loader, device)
    print(f"Optimal temperature: {temperature:.3f}")

    thresh_tuner = ThresholdTuner(config)
    thresh_super, thresh_sub = thresh_tuner.find_thresholds(
        model, seen_loader, unseen_loader, device, temperature
    )
    print(f"Optimal thresholds: super={thresh_super:.3f}, sub={thresh_sub:.3f}")

    model.eval()
    total_super_ce = 0.0
    total_sub_ce = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in seen_loader:
            images = batch["image"].to(device)
            super_targets = batch["superclass"].to(device)
            sub_targets = batch["subclass"].to(device)

            output = model(images)

            super_ce = F.cross_entropy(
                output.super_logits / temperature, super_targets, reduction="sum"
            ).item()
            sub_ce = F.cross_entropy(
                output.sub_logits / temperature, sub_targets, reduction="sum"
            ).item()

            total_super_ce += super_ce
            total_sub_ce += sub_ce
            total_count += images.size(0)

    return CalibrationResult(
        temperature=temperature,
        threshold_super=thresh_super,
        threshold_sub=thresh_sub,
        val_ce_super=total_super_ce / total_count,
        val_ce_sub=total_sub_ce / total_count,
        novel_f1_super=0.0,
        novel_f1_sub=0.0,
    )
