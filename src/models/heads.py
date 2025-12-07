import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CosineClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 16.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        logits = F.linear(x_norm, w_norm) * self.scale
        return logits


class VirtualLogitHead(nn.Module):
    def __init__(
        self,
        alpha_init: float = 0.0,
        beta_init: float = 1.0,
        clamp_value: float = 15.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        # Store log(beta) so that actual beta = softplus(raw_beta) is always positive
        # Initialize so softplus(raw_beta) ≈ beta_init
        raw_beta_init = math.log(math.exp(beta_init) - 1) if beta_init > 0 else 0.0
        self.raw_beta = nn.Parameter(torch.tensor(raw_beta_init))
        self.clamp_value = clamp_value
        self.temperature = temperature

    @property
    def beta(self) -> torch.Tensor:
        # Softplus ensures beta is always positive
        return F.softplus(self.raw_beta)

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        energy = self.compute_energy(logits)
        novel_logit = self.alpha + self.beta * energy
        novel_logit = torch.clamp(novel_logit, -self.clamp_value, self.clamp_value)

        full_logits = torch.cat([logits, novel_logit.unsqueeze(1)], dim=1)
        return full_logits


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 16.0,
        alpha_init: float = 0.0,
        beta_init: float = 1.0,
        clamp_value: float = 15.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = CosineClassifier(in_features, num_classes, scale)
        self.virtual_logit = VirtualLogitHead(alpha_init, beta_init, clamp_value)

        self.num_classes = num_classes
        self.num_outputs = num_classes + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(x)
        base_logits = self.classifier(x)
        full_logits = self.virtual_logit(base_logits)
        return full_logits, base_logits


class PerSuperclassSubHead(nn.Module):
    """
    Per-superclass subclass classification heads.
    Each superclass has its own classifier over its subclasses + novel.
    """

    def __init__(
        self,
        in_features: int,
        subclasses_per_super: dict,
        scale: float = 16.0,
        alpha_init: float = 0.0,
        beta_init: float = 1.0,
        clamp_value: float = 15.0,
        dropout: float = 0.0,
        total_subclasses: int = 87,
    ):
        super().__init__()
        self.in_features = in_features
        self.subclasses_per_super = subclasses_per_super
        self.num_superclasses = len(subclasses_per_super)
        self.total_subclasses = total_subclasses
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.heads = nn.ModuleDict()
        self.virtual_logits = nn.ModuleDict()
        self.local_to_global = {}
        self.global_to_local = {}

        for super_idx, sub_indices in subclasses_per_super.items():
            num_subs = len(sub_indices)
            self.heads[str(super_idx)] = CosineClassifier(in_features, num_subs, scale)
            self.virtual_logits[str(super_idx)] = VirtualLogitHead(
                alpha_init, beta_init, clamp_value
            )
            self.local_to_global[super_idx] = {
                local_idx: global_idx for local_idx, global_idx in enumerate(sub_indices)
            }
            self.global_to_local[super_idx] = {
                global_idx: local_idx for local_idx, global_idx in enumerate(sub_indices)
            }

    def forward(
        self,
        x: torch.Tensor,
        super_targets: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional superclass routing.

        If super_targets is provided (training mode), route each sample to its head.
        Otherwise, compute all heads and aggregate.

        Returns:
            full_logits: [B, total_subclasses + 1] with novel at the end
            base_logits: [B, total_subclasses] without novel
        """
        x = self.dropout(x)
        batch_size = x.size(0)
        device = x.device

        full_logits = torch.full(
            (batch_size, self.total_subclasses + 1),
            float("-inf"),
            device=device,
            dtype=x.dtype,
        )
        base_logits = torch.full(
            (batch_size, self.total_subclasses),
            float("-inf"),
            device=device,
            dtype=x.dtype,
        )

        if super_targets is not None:
            for super_idx in range(self.num_superclasses):
                mask = super_targets == super_idx
                if not mask.any():
                    continue

                head = self.heads[str(super_idx)]
                vl = self.virtual_logits[str(super_idx)]

                local_logits = head(x[mask])
                local_full = vl(local_logits)

                for local_idx, global_idx in self.local_to_global[super_idx].items():
                    base_logits[mask, global_idx] = local_logits[:, local_idx]
                    full_logits[mask, global_idx] = local_full[:, local_idx]

                full_logits[mask, -1] = local_full[:, -1]
        else:
            all_novel_logits = []
            for super_idx in range(self.num_superclasses):
                head = self.heads[str(super_idx)]
                vl = self.virtual_logits[str(super_idx)]

                local_logits = head(x)
                local_full = vl(local_logits)

                for local_idx, global_idx in self.local_to_global[super_idx].items():
                    base_logits[:, global_idx] = local_logits[:, local_idx]
                    full_logits[:, global_idx] = local_full[:, local_idx]

                all_novel_logits.append(local_full[:, -1:])

            # Average novel logits from all heads
            novel_logit = torch.cat(all_novel_logits, dim=1).mean(dim=1)
            full_logits[:, -1] = novel_logit

        return full_logits, base_logits

    def get_beta_values(self) -> list:
        return [self.virtual_logits[str(i)].beta for i in range(self.num_superclasses)]


class NoveltyHead(nn.Module):
    """
    Separate novelty detection head for Phase B training.
    Combines learned MLP with energy-based scoring.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        energy_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.energy_weight = energy_weight

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.energy_scale = nn.Parameter(torch.tensor(1.0))
        self.energy_bias = nn.Parameter(torch.tensor(0.0))

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        classifier_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns novelty score (higher = more likely novel).

        Args:
            features: [B, D] feature vectors
            classifier_logits: [B, C] logits from classifier (for energy computation)

        Returns:
            novelty_score: [B] raw novelty scores (before sigmoid)
        """
        mlp_score = self.mlp(features).squeeze(-1)

        if classifier_logits is not None and self.energy_weight > 0:
            energy = self.compute_energy(classifier_logits)
            energy_score = self.energy_scale * energy + self.energy_bias
            novelty_score = (1 - self.energy_weight) * mlp_score + self.energy_weight * energy_score
        else:
            novelty_score = mlp_score

        return novelty_score


class PureClassificationHead(nn.Module):
    """
    Classification head without virtual logit for Phase A training.
    Pure classification only - no novel class.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = CosineClassifier(in_features, num_classes, scale)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class PerSuperclassPureHead(nn.Module):
    """
    Per-superclass subclass classification heads WITHOUT virtual logit.
    For Phase A decoupled training.
    """

    def __init__(
        self,
        in_features: int,
        subclasses_per_super: Dict[int, List[int]],
        scale: float = 16.0,
        dropout: float = 0.0,
        total_subclasses: int = 87,
    ):
        super().__init__()
        self.in_features = in_features
        self.subclasses_per_super = subclasses_per_super
        self.num_superclasses = len(subclasses_per_super)
        self.total_subclasses = total_subclasses
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.heads = nn.ModuleDict()
        self.local_to_global = {}
        self.global_to_local = {}

        for super_idx, sub_indices in subclasses_per_super.items():
            num_subs = len(sub_indices)
            self.heads[str(super_idx)] = CosineClassifier(in_features, num_subs, scale)
            self.local_to_global[super_idx] = {
                local_idx: global_idx for local_idx, global_idx in enumerate(sub_indices)
            }
            self.global_to_local[super_idx] = {
                global_idx: local_idx for local_idx, global_idx in enumerate(sub_indices)
            }

    def forward(
        self,
        x: torch.Tensor,
        super_targets: Optional[torch.Tensor] = None,
        super_preds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass with superclass routing.

        Args:
            x: [B, D] features
            super_targets: [B] ground truth superclass (training)
            super_preds: [B] predicted superclass (inference)

        Returns:
            global_logits: [B, total_subclasses] with -inf for invalid subclasses
            per_head_logits: {super_idx: [B_super, num_subs]} local logits per head
        """
        x = self.dropout(x)
        batch_size = x.size(0)
        device = x.device

        global_logits = torch.full(
            (batch_size, self.total_subclasses),
            float("-inf"),
            device=device,
            dtype=x.dtype,
        )

        per_head_logits = {}
        routing = super_targets if super_targets is not None else super_preds

        if routing is not None:
            for super_idx in range(self.num_superclasses):
                mask = routing == super_idx
                if not mask.any():
                    continue

                head = self.heads[str(super_idx)]
                local_logits = head(x[mask])
                per_head_logits[super_idx] = local_logits

                for local_idx, global_idx in self.local_to_global[super_idx].items():
                    global_logits[mask, global_idx] = local_logits[:, local_idx]
        else:
            for super_idx in range(self.num_superclasses):
                head = self.heads[str(super_idx)]
                local_logits = head(x)
                per_head_logits[super_idx] = local_logits

                for local_idx, global_idx in self.local_to_global[super_idx].items():
                    global_logits[:, global_idx] = local_logits[:, local_idx]

        return global_logits, per_head_logits

    def get_subclass_count(self, super_idx: int) -> int:
        return len(self.subclasses_per_super[super_idx])
