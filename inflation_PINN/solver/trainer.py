# solver/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Callable, List, Tuple, Optional

from model.network import FullyConnectedNetwork
from model.pinn import compute_pinn_loss

class Trainer:
    def __init__(
        self,
        config: Any,
        physics_residual_func: Callable[..., torch.Tensor],
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        physics_inputs: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            config: Configuration object.
            physics_residual_func: Function to compute the physics residuals.
            x_data: Input tensor for training data (e.g., t).
            y_data: Output tensor for training data (e.g., φ(t)).
            physics_inputs: Dictionary of inputs needed for physics residual calculation.
            device: Device to train the model on.
        """
        self.config = config
        self.physics_residual_func = physics_residual_func
        self.device = device or torch.device('cpu')
        
        self.x_data = x_data.to(self.device)
        self.y_data = y_data.to(self.device)
        self.physics_inputs = {k: v.to(self.device) for k, v in physics_inputs.items()}
        self.parameters = config.physics.parameters
        
        # Build the model
        self.model = FullyConnectedNetwork(
            input_dim=self.x_data.shape[1],
            output_dim=self.y_data.shape[1],
            hidden_dim=config.model.hidden_units,
            num_hidden_layers=config.model.hidden_layers,
            activation=self._get_activation(config.model.activation)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.learning_rate)
        
        # History
        self.loss_history: List[float] = []

    def _get_activation(self, activation_name: str) -> Callable[[], nn.Module]:
        """Utility to get activation function."""
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
        }
        return activations.get(activation_name.lower(), nn.Tanh)

    def train(self) -> Tuple[nn.Module, List[float]]:
        """
        Train the PINN model.

        Returns:
            Trained model and loss history.
        """
        epochs = self.config.train.epochs
        physics_weight = self.config.train.physics_weight

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = compute_pinn_loss(
                model=self.model,
                x_data=self.x_data,
                y_data=self.y_data,
                physics_inputs=self.physics_inputs,
                physics_residual_func=self.physics_residual_func,
                parameters=self.parameters,
                physics_weight=physics_weight
            )
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            self.loss_history.append(loss_value)

            if epoch % 100 == 0:
                print(f"[Epoch {epoch}] Loss: {loss_value:.6e}")

        return self.model, self.loss_history

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)

    def save_loss_history(self, path: str) -> None:
        """Save the loss history as a simple text file."""
        with open(path, 'w') as f:
            for loss in self.loss_history:
                f.write(f"{loss}\n")
