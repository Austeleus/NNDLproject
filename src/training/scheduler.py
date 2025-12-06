from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..config import TrainingConfig
from ..models.classifier import HierarchicalClassifier


@dataclass
class UnfreezePhase:
    start_epoch: int
    end_epoch: int
    unfreeze_groups: List[int]
    lr: float
    freeze_virtual_logit: bool = False


class ProgressiveUnfreezeScheduler:
    def __init__(
        self,
        model: HierarchicalClassifier,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.layer_groups = model.get_layer_groups()

        self.phases = self._create_phases()
        self.current_phase_idx = 0

    def _create_phases(self) -> List[UnfreezePhase]:
        c = self.config
        phases = []

        phase1_end = c.phase1_epochs
        phase2_end = phase1_end + c.phase2_epochs
        phase3_end = phase2_end + c.phase3_epochs

        # Phase 1: Train conv1 stem (group 0) + layer1/2 (group 1) + heads (group 3)
        # Layer1/2 must adapt to new conv1 output from the start
        phases.append(UnfreezePhase(
            start_epoch=0,
            end_epoch=phase1_end,
            unfreeze_groups=[0, 1, 3],
            lr=c.phase1_lr,
        ))

        # Phase 2: Add layer3-4 (group 2) for deeper adaptation
        phases.append(UnfreezePhase(
            start_epoch=phase1_end,
            end_epoch=phase2_end,
            unfreeze_groups=[0, 1, 2, 3],
            lr=c.phase2_lr,
        ))

        # Phase 3: Full fine-tuning with lower LR
        phases.append(UnfreezePhase(
            start_epoch=phase2_end,
            end_epoch=phase3_end,
            unfreeze_groups=[0, 1, 2, 3],
            lr=c.phase3_lr,
            freeze_virtual_logit=False,
        ))

        return phases

    def get_phase(self, epoch: int) -> UnfreezePhase:
        for phase in self.phases:
            if phase.start_epoch <= epoch < phase.end_epoch:
                return phase
        return self.phases[-1]

    def apply_phase(self, epoch: int) -> Optional[float]:
        phase = self.get_phase(epoch)
        new_phase_idx = self.phases.index(phase)

        if new_phase_idx != self.current_phase_idx or epoch == 0:
            self.current_phase_idx = new_phase_idx

            for param in self.model.parameters():
                param.requires_grad = False

            for group_idx in phase.unfreeze_groups:
                if group_idx < len(self.layer_groups):
                    for param in self.layer_groups[group_idx]:
                        param.requires_grad = True

            # Freeze virtual logit parameters (α, β) if specified
            if phase.freeze_virtual_logit:
                self.model.super_head.virtual_logit.alpha.requires_grad = False
                self.model.super_head.virtual_logit.raw_beta.requires_grad = False
                self.model.sub_head.virtual_logit.alpha.requires_grad = False
                self.model.sub_head.virtual_logit.raw_beta.requires_grad = False

            return phase.lr

        return None

    def get_current_lr(self, epoch: int) -> float:
        return self.get_phase(epoch).lr


def create_optimizer(
    model: HierarchicalClassifier,
    config: TrainingConfig,
) -> Optimizer:
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_lr_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    if config.scheduler.lower() == "cosine":
        if warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=config.min_lr,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            return CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=config.min_lr,
            )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
