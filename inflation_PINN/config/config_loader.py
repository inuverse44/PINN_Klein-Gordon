# config/config_loader.py

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float
    optimizer: str

@dataclass
class ModelConfig:
    hidden_layers: int
    hidden_units: int
    activation: str

@dataclass
class PhysicsConfig:
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    train: TrainConfig
    model: ModelConfig
    physics: PhysicsConfig

def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load and parse the YAML configuration file into a Config dataclass."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    
    return Config(
        train=TrainConfig(**raw_cfg["train"]),
        model=ModelConfig(**raw_cfg["model"]),
        physics=PhysicsConfig(**raw_cfg["physics"]),
    )
