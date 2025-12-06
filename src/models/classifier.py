from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from .backbone import create_backbone
from .heads import ClassificationHead, PerSuperclassSubHead


@dataclass
class ModelOutput:
    super_logits: torch.Tensor
    sub_logits: torch.Tensor
    super_base_logits: torch.Tensor
    sub_base_logits: torch.Tensor
    features: torch.Tensor


class HierarchicalClassifier(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        subclasses_per_super: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()
        self.config = config
        self.use_per_super_heads = (
            getattr(config, "use_per_super_heads", False) and
            subclasses_per_super is not None
        )

        self.backbone, self.feature_dim = create_backbone(
            name=config.backbone,
            pretrained=config.pretrained,
        )

        self.super_head = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=config.num_superclasses,
            scale=config.cosine_scale,
            alpha_init=config.virtual_logit_alpha_init,
            beta_init=config.virtual_logit_beta_init,
            clamp_value=config.virtual_logit_clamp,
            dropout=config.dropout,
        )

        if self.use_per_super_heads:
            self.sub_head = PerSuperclassSubHead(
                in_features=self.feature_dim,
                subclasses_per_super=subclasses_per_super,
                scale=config.cosine_scale,
                alpha_init=config.virtual_logit_alpha_init,
                beta_init=config.virtual_logit_beta_init,
                clamp_value=config.virtual_logit_clamp,
                dropout=config.dropout,
                total_subclasses=config.num_subclasses,
            )
        else:
            self.sub_head = ClassificationHead(
                in_features=self.feature_dim,
                num_classes=config.num_subclasses,
                scale=config.cosine_scale,
                alpha_init=config.virtual_logit_alpha_init,
                beta_init=config.virtual_logit_beta_init,
                clamp_value=config.virtual_logit_clamp,
                dropout=config.dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        super_targets: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        features = self.backbone(x)

        super_logits, super_base = self.super_head(features)

        if self.use_per_super_heads:
            sub_logits, sub_base = self.sub_head(features, super_targets)
        else:
            sub_logits, sub_base = self.sub_head(features)

        return ModelOutput(
            super_logits=super_logits,
            sub_logits=sub_logits,
            super_base_logits=super_base,
            sub_base_logits=sub_base,
            features=features,
        )

    def get_virtual_logit_betas(self) -> tuple:
        """Get beta values from virtual logit heads for regularization."""
        super_beta = self.super_head.virtual_logit.beta
        if self.use_per_super_heads:
            sub_betas = self.sub_head.get_beta_values()
            sub_beta = sum(sub_betas) / len(sub_betas)
        else:
            sub_beta = self.sub_head.virtual_logit.beta
        return super_beta, sub_beta

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        backbone_groups = self.backbone.get_layer_groups()

        head_params = (
            list(self.super_head.parameters()) +
            list(self.sub_head.parameters())
        )

        return backbone_groups + [head_params]

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_layers(self, layer_indices: List[int]) -> None:
        groups = self.get_layer_groups()
        for idx in layer_indices:
            if idx < len(groups):
                for param in groups[idx]:
                    param.requires_grad = False

    def unfreeze_layers(self, layer_indices: List[int]) -> None:
        groups = self.get_layer_groups()
        for idx in layer_indices:
            if idx < len(groups):
                for param in groups[idx]:
                    param.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        super_params = sum(p.numel() for p in self.super_head.parameters())
        sub_params = sum(p.numel() for p in self.sub_head.parameters())
        total = backbone_params + super_params + sub_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "backbone": backbone_params,
            "super_head": super_params,
            "sub_head": sub_params,
            "total": total,
            "trainable": trainable,
        }


def create_model(config: ModelConfig) -> HierarchicalClassifier:
    return HierarchicalClassifier(config)
