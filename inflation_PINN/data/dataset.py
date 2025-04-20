# data/dataset.py

import torch
from typing import Tuple, Dict, Any, Optional

def generate_dataset(
    domain_ranges: Dict[str, Tuple[float, float]],
    num_points: Dict[str, int],
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate training datasets.

    Args:
        domain_ranges: A dictionary specifying min and max for each input variable.
                       e.g., {"t": (0.0, 1.0), "x": (0.0, 1.0)}
        num_points: A dictionary specifying how many points to sample for each variable.
                    e.g., {"t": 100, "x": 50}
        device: Device to put the generated tensors on.

    Returns:
        A dictionary with tensors for each input variable.
        e.g., {"t": tensor of shape (N, 1), "x": tensor of shape (M, 1)}
    """
    dataset = {}
    for var, (vmin, vmax) in domain_ranges.items():
        n_points = num_points.get(var)
        if n_points is None:
            raise ValueError(f"Number of points for variable '{var}' is not specified.")
        
        values = torch.linspace(vmin, vmax, n_points).view(-1, 1)
        
        if device is not None:
            values = values.to(device)
        
        dataset[var] = values

    return dataset
