from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from .backbone import create_backbone
from .heads import (
    PureClassificationHead,
    PerSuperclassPureHead,
    NoveltyHead,
)


@dataclass
class DecoupledOutput:
    super_logits: torch.Tensor
    sub_logits: torch.Tensor
    features: torch.Tensor
    per_head_sub_logits: Optional[Dict[int, torch.Tensor]] = None
    super_novelty_score: Optional[torch.Tensor] = None
    sub_novelty_scores: Optional[Dict[int, torch.Tensor]] = None


class DecoupledClassifier(nn.Module):
    """
    Decoupled hierarchical classifier for two-phase training.

    Phase A: Pure classification (no novel class)
        - Trains backbone + super_head + sub_heads
        - No novelty detection

    Phase B: Add novelty detection with fine-tuning
        - Freeze or use small LR for backbone + heads
        - Train novelty heads with OE data
    """

    def __init__(
        self,
        config: ModelConfig,
        subclasses_per_super: Dict[int, List[int]],
    ):
        super().__init__()
        self.config = config
        self.subclasses_per_super = subclasses_per_super

        self.backbone, self.feature_dim = create_backbone(
            name=config.backbone,
            pretrained=config.pretrained,
        )

        self.super_head = PureClassificationHead(
            in_features=self.feature_dim,
            num_classes=config.num_superclasses,
            scale=config.cosine_scale,
            dropout=config.dropout,
        )

        self.sub_head = PerSuperclassPureHead(
            in_features=self.feature_dim,
            subclasses_per_super=subclasses_per_super,
            scale=config.cosine_scale,
            dropout=config.dropout,
            total_subclasses=config.num_subclasses,
        )

        self.super_novelty_head: Optional[NoveltyHead] = None
        self.sub_novelty_heads: Optional[nn.ModuleDict] = None

        self._phase = "A"

    @property
    def phase(self) -> str:
        return self._phase

    def init_phase_b(
        self,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        energy_weight: float = 0.5,
    ):
        """Initialize novelty heads for Phase B training."""
        self.super_novelty_head = NoveltyHead(
            in_features=self.feature_dim,
            hidden_dim=hidden_dim,
            temperature=temperature,
            energy_weight=energy_weight,
        )

        self.sub_novelty_heads = nn.ModuleDict()
        for super_idx in self.subclasses_per_super.keys():
            self.sub_novelty_heads[str(super_idx)] = NoveltyHead(
                in_features=self.feature_dim,
                hidden_dim=hidden_dim,
                temperature=temperature,
                energy_weight=energy_weight,
            )

        self._phase = "B"

        if next(self.backbone.parameters()).is_cuda:
            device = next(self.backbone.parameters()).device
            self.super_novelty_head.to(device)
            self.sub_novelty_heads.to(device)

    def forward(
        self,
        x: torch.Tensor,
        super_targets: Optional[torch.Tensor] = None,
    ) -> DecoupledOutput:
        features = self.backbone(x)
        super_logits = self.super_head(features)

        if super_targets is not None:
            sub_logits, per_head_logits = self.sub_head(
                features, super_targets=super_targets
            )
        else:
            super_preds = super_logits.argmax(dim=1)
            sub_logits, per_head_logits = self.sub_head(
                features, super_preds=super_preds
            )

        output = DecoupledOutput(
            super_logits=super_logits,
            sub_logits=sub_logits,
            features=features,
            per_head_sub_logits=per_head_logits,
        )

        if self._phase == "B" and self.super_novelty_head is not None:
            output.super_novelty_score = self.super_novelty_head(
                features, super_logits
            )

            if self.sub_novelty_heads is not None:
                output.sub_novelty_scores = {}
                routing = super_targets if super_targets is not None else super_logits.argmax(dim=1)

                for super_idx in range(len(self.subclasses_per_super)):
                    mask = routing == super_idx
                    if mask.any():
                        head_key = str(super_idx)
                        local_logits = per_head_logits.get(super_idx)
                        if local_logits is not None:
                            score = self.sub_novelty_heads[head_key](
                                features[mask], local_logits
                            )
                            output.sub_novelty_scores[super_idx] = (mask, score)

        return output

    def get_phase_a_params(self) -> List[nn.Parameter]:
        """Get parameters for Phase A training (classification only)."""
        return list(self.backbone.parameters()) + \
               list(self.super_head.parameters()) + \
               list(self.sub_head.parameters())

    def get_phase_b_params(self, include_backbone: bool = True, backbone_lr_mult: float = 0.01) -> List[Dict]:
        """
        Get parameter groups for Phase B training.
        Novelty heads get full LR, backbone/classifiers get reduced LR.
        """
        param_groups = []

        if self.super_novelty_head is not None:
            param_groups.append({
                "params": list(self.super_novelty_head.parameters()),
                "lr_mult": 1.0,
                "name": "super_novelty",
            })

        if self.sub_novelty_heads is not None:
            param_groups.append({
                "params": list(self.sub_novelty_heads.parameters()),
                "lr_mult": 1.0,
                "name": "sub_novelty",
            })

        if include_backbone:
            param_groups.append({
                "params": list(self.backbone.parameters()),
                "lr_mult": backbone_lr_mult,
                "name": "backbone",
            })
            param_groups.append({
                "params": list(self.super_head.parameters()) + list(self.sub_head.parameters()),
                "lr_mult": backbone_lr_mult,
                "name": "classifiers",
            })

        return param_groups

    def freeze_for_phase_b(self):
        """Freeze backbone and classification heads for Phase B."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.super_head.parameters():
            param.requires_grad = False
        for param in self.sub_head.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        backbone_groups = self.backbone.get_layer_groups()
        head_params = (
            list(self.super_head.parameters()) +
            list(self.sub_head.parameters())
        )
        groups = backbone_groups + [head_params]

        if self.super_novelty_head is not None:
            groups.append(list(self.super_novelty_head.parameters()))
        if self.sub_novelty_heads is not None:
            groups.append(list(self.sub_novelty_heads.parameters()))

        return groups

    def count_parameters(self) -> Dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        super_params = sum(p.numel() for p in self.super_head.parameters())
        sub_params = sum(p.numel() for p in self.sub_head.parameters())

        novelty_params = 0
        if self.super_novelty_head is not None:
            novelty_params += sum(p.numel() for p in self.super_novelty_head.parameters())
        if self.sub_novelty_heads is not None:
            novelty_params += sum(p.numel() for p in self.sub_novelty_heads.parameters())

        total = backbone_params + super_params + sub_params + novelty_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "backbone": backbone_params,
            "super_head": super_params,
            "sub_head": sub_params,
            "novelty_heads": novelty_params,
            "total": total,
            "trainable": trainable,
        }


def create_decoupled_model(
    config: ModelConfig,
    subclasses_per_super: Dict[int, List[int]],
) -> DecoupledClassifier:
    return DecoupledClassifier(config, subclasses_per_super)
