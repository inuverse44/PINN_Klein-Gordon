# physics/equation.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def klein_gordon_residual(
    t: torch.Tensor,
    model: nn.Module,
    parameters: Dict[str, Any],
    include_spatial: bool = False,
    x: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the residual of the Klein-Gordon equation during inflation.

    Args:
        t (torch.Tensor): Time variable tensor of shape (N, 1).
        model (nn.Module): Neural network model representing phi.
        parameters (Dict[str, Any]): Dictionary containing physical parameters, e.g., 'm', 'H'.
        include_spatial (bool): Whether to include spatial derivatives.
        x (torch.Tensor, optional): Spatial variable tensor of shape (N, 1). Required if include_spatial=True.

    Returns:
        torch.Tensor: Residual tensor of shape (N, 1).
    """
    if include_spatial and x is None:
        raise ValueError("Spatial variable x must be provided if include_spatial is True.")

    if include_spatial:
        inputs = torch.cat([x, t], dim=1)  # Shape: (N, 2)
    else:
        inputs = t  # Shape: (N, 1)

    # Predict phi
    phi = model(inputs)

    # Temporal derivatives
    dphi_dt = torch.autograd.grad(
        outputs=phi,
        inputs=t,
        grad_outputs=torch.ones_like(phi),
        create_graph=True
    )[0]
    d2phi_dt2 = torch.autograd.grad(
        outputs=dphi_dt,
        inputs=t,
        grad_outputs=torch.ones_like(dphi_dt),
        create_graph=True
    )[0]

    # Spatial derivatives (optional)
    if include_spatial:
        dphi_dx = torch.autograd.grad(
            outputs=phi,
            inputs=x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True
        )[0]
        d2phi_dx2 = torch.autograd.grad(
            outputs=dphi_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dphi_dx),
            create_graph=True
        )[0]
    else:
        d2phi_dx2 = 0.0

    # Extract physical parameters
    H = parameters.get("H", 0.0)  # Default H = 0 if not provided
    m = parameters["m"]

    # Potential derivative dV/dphi for V = 1/2 m^2 phi^2
    V_prime = m**2 * phi

    # Klein-Gordon residual
    residual = d2phi_dt2 - d2phi_dx2 + 3 * H * dphi_dt + V_prime

    return residual
