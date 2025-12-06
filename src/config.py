from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import yaml


@dataclass
class DataConfig:
    data_dir: Path = Path("project_data")
    train_csv: str = "train_data.csv"
    superclass_mapping: str = "superclass_mapping.csv"
    subclass_mapping: str = "subclass_mapping.csv"
    train_images_dir: str = "train_images"
    test_images_dir: str = "test_images"

    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    train_seen_ratio: float = 0.75
    oe_subclasses_per_super: int = 2
    unseen_val_subclasses_per_super: int = 2

    image_size: int = 128

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


@dataclass
class AugmentationConfig:
    randaugment_n: int = 2
    randaugment_m: int = 9
    randaugment_m_fallback: int = 7

    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.1

    random_crop_scale: Tuple[float, float] = (0.7, 1.0)
    random_crop_ratio: Tuple[float, float] = (0.8, 1.2)

    random_erasing_prob: float = 0.3
    random_erasing_scale: Tuple[float, float] = (0.02, 0.2)

    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5

    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    backbone: str = "resnet34"
    pretrained: bool = True

    num_superclasses: int = 3
    num_subclasses: int = 87

    cosine_scale: float = 16.0

    virtual_logit_alpha_init: float = 0.0
    virtual_logit_beta_init: float = 1.0
    virtual_logit_clamp: float = 15.0

    dropout: float = 0.5

    use_per_super_heads: bool = False

    @property
    def num_super_outputs(self) -> int:
        return self.num_superclasses + 1

    @property
    def num_sub_outputs(self) -> int:
        return self.num_subclasses + 1


@dataclass
class TrainingConfig:
    epochs: int = 25

    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.05

    scheduler: str = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6

    phase1_epochs: int = 5
    phase2_epochs: int = 10
    phase3_epochs: int = 10

    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-4
    phase3_lr: float = 1e-5

    grad_clip: float = 1.0

    early_stopping_patience: int = 10

    seed: int = 42


@dataclass
class LossConfig:
    label_smoothing: float = 0.1

    lambda_super: float = 1.0
    lambda_sub: float = 1.0
    lambda_oe: float = 1.0

    oe_batch_ratio: float = 0.25

    use_margin_loss: bool = True
    margin: float = 2.0
    lambda_margin: float = 0.5


@dataclass
class InferenceConfig:
    temperature: float = 1.0
    temperature_search_range: Tuple[float, float] = (0.5, 2.5)
    temperature_search_steps: int = 21

    threshold_super: float = 0.5
    threshold_sub: float = 0.6
    threshold_search_range: Tuple[float, float] = (0.3, 0.9)
    threshold_search_steps: int = 21

    use_msp: bool = True
    msp_threshold_sub: float = 0.6


@dataclass
class LoggingConfig:
    project_name: str = "nndl-hierarchical-classification"
    experiment_name: Optional[str] = None

    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1

    save_dir: Path = Path("experiments")
    save_top_k: int = 3

    def __post_init__(self):
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()

        if "data" in raw:
            config.data = DataConfig(**raw["data"])
        if "augmentation" in raw:
            config.augmentation = AugmentationConfig(**raw["augmentation"])
        if "model" in raw:
            config.model = ModelConfig(**raw["model"])
        if "training" in raw:
            config.training = TrainingConfig(**raw["training"])
        if "loss" in raw:
            config.loss = LossConfig(**raw["loss"])
        if "inference" in raw:
            config.inference = InferenceConfig(**raw["inference"])
        if "logging" in raw:
            config.logging = LoggingConfig(**raw["logging"])

        return config

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict

        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        data = {}
        for key in ["data", "augmentation", "model", "training", "loss", "inference", "logging"]:
            section = asdict(getattr(self, key))
            data[key] = {k: convert(v) for k, v in section.items()}

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
