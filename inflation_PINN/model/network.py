# NNアーキテクチャ定義（例: FCN）
# model/network.py

import torch
import torch.nn as nn
from typing import Type

class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: Type[nn.Module] = nn.Tanh
    ) -> None:
        """
        Args:
            input_dim (int): Dimension of input.
            output_dim (int): Dimension of output.
            hidden_dim (int): Number of units in each hidden layer.
            num_hidden_layers (int): Number of hidden layers.
            activation (Type[nn.Module]): Activation function class (default: nn.Tanh).
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
