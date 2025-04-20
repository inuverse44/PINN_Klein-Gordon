# PINNの損失関数（物理損失計算）
# model/pinn.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable

def compute_pinn_loss(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    physics_inputs: Dict[str, torch.Tensor],
    physics_residual_func: Callable[..., torch.Tensor],
    parameters: Dict[str, Any],
    physics_weight: float = 1e-4
) -> torch.Tensor:
    """
    Compute the total loss for PINN training.

    Args:
        model (nn.Module): Neural network model.
        x_data (torch.Tensor): Training input data tensor (e.g., (t)).
        y_data (torch.Tensor): Training output data tensor (e.g., φ(t)).
        physics_inputs (Dict[str, torch.Tensor]): Inputs for physics residual calculation (e.g., t, x).
        physics_residual_func (Callable): Function to compute physics residuals.
        parameters (Dict[str, Any]): Physical parameters dictionary.
        physics_weight (float): Weight for the physics loss term.

    Returns:
        torch.Tensor: Total loss (scalar).
    """
    # Data loss (MSE between model predictions and training data)
    y_pred = model(x_data)
    data_loss = torch.mean((y_pred - y_data)**2)

    # Physics loss
    residual = physics_residual_func(**physics_inputs, model=model, parameters=parameters)
    physics_loss = torch.mean(residual**2)

    # Total loss
    total_loss = data_loss + physics_weight * physics_loss

    return total_loss
