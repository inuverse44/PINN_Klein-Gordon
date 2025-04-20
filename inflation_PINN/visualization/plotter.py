# 学習曲線や予測結果の可視化
# visualization/plotter.py

import matplotlib.pyplot as plt
import torch
from typing import Optional

def plot_loss_history(
    loss_history: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the training loss history.

    Args:
        loss_history (torch.Tensor): Tensor or list of loss values.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss History")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_prediction_vs_exact(
    t: torch.Tensor,
    y_exact: torch.Tensor,
    model: torch.nn.Module,
    save_path: Optional[str] = None
) -> None:
    """
    Plot model predictions against the exact solution.

    Args:
        t (torch.Tensor): Time tensor (N, 1).
        y_exact (torch.Tensor): Exact solution tensor (N, 1).
        model (torch.nn.Module): Trained model.
        save_path (str, optional): Path to save the plot.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(t)
    
    plt.figure(figsize=(8, 5))
    plt.plot(t.cpu(), y_exact.cpu(), label="Exact", linestyle="--", linewidth=2)
    plt.plot(t.cpu(), y_pred.cpu(), label="Prediction", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Prediction vs Exact Solution")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
