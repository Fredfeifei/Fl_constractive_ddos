from dataclasses import dataclass
@dataclass
class VAEConfig:
    input_dim: int = 86
    hidden_dim: int = 128
    latent_dim: int = 32
    learning_rate: float = 0.0001
    batch_size: int = 256
    num_epochs: int = 20
    beta: float = 0.3  # KL divergence weight

@dataclass
class ContrastiveConfig:
    feature_dim: int = 86
    temperature: float = 0.07
    batch_size: int = 32
    num_epochs: int = 256
    learning_rate: float = 0.001
    num_augmentations: int = 2

@dataclass
class TrainingConfig:
    device: str = 'cuda'
    data_root: str = 'data/processed'
    output_dir: str = 'outputs'
    seed: int = 42
    test_drop: bool = True
