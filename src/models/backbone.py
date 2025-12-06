from typing import List, Tuple

import torch
import torch.nn as nn
import timm
from torchvision import models


class ModifiedResNet(nn.Module):
    def __init__(self, base_model: nn.Module, out_features: int):
        super().__init__()
        self.out_features = out_features

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        return [
            list(self.conv1.parameters()) + list(self.bn1.parameters()),
            list(self.layer1.parameters()) + list(self.layer2.parameters()),
            list(self.layer3.parameters()) + list(self.layer4.parameters()),
        ]


class ModifiedEfficientNet(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.out_features = self.model.num_features

        if hasattr(self.model, "conv_stem"):
            old_stem = self.model.conv_stem
            self.model.conv_stem = nn.Conv2d(
                3,
                old_stem.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        stem_params = []
        block_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "conv_stem" in name or "bn1" in name:
                stem_params.append(param)
            elif "blocks" in name:
                block_params.append(param)
            else:
                head_params.append(param)

        return [stem_params, block_params, head_params]


def create_backbone(
    name: str = "resnet50",
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    name = name.lower()

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)
        model = ModifiedResNet(base, out_features=2048)
        return model, 2048

    elif name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet34(weights=weights)
        model = ModifiedResNet(base, out_features=512)
        return model, 512

    elif name == "efficientnet_b0":
        model = ModifiedEfficientNet("efficientnet_b0", pretrained=pretrained)
        return model, model.out_features

    else:
        raise ValueError(f"Unknown backbone: {name}. Choose from: resnet50, resnet34, efficientnet_b0")
