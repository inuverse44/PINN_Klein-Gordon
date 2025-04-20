# __main__.py

import torch
import os

from config.config_loader import load_config
from data.dataset import generate_dataset
from physics.equation import klein_gordon_residual
from solver.trainer import Trainer
from visualization.plotter import plot_loss_history, plot_prediction_vs_exact

def main() -> None:
    # Load configuration
    config = load_config()

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create output directories
    os.makedirs("output/figures", exist_ok=True)
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)

    # Generate dataset
    dataset = generate_dataset(
        domain_ranges={"t": (0.0, 1.0)},
        num_points={"t": 500},
        device=device
    )
    x_data = dataset["t"]
    y_data = torch.sin(2 * torch.pi * x_data)  # (仮の解析解) 後で自由に変更可能
    physics_inputs = {"t": dataset["t"]}

    # Initialize trainer
    trainer = Trainer(
        config=config,
        physics_residual_func=klein_gordon_residual,
        x_data=x_data,
        y_data=y_data,
        physics_inputs=physics_inputs,
        device=device
    )

    # Train the model
    model, loss_history = trainer.train()

    # Save model and loss history
    trainer.save_model("output/models/pinn_model.pth")
    trainer.save_loss_history("output/logs/loss_history.txt")

    # Plot loss history
    plot_loss_history(loss_history, save_path="output/figures/loss_history.png")

    # Plot predictions vs exact solution
    plot_prediction_vs_exact(x_data, y_data, model, save_path="output/figures/prediction_vs_exact.png")

if __name__ == "__main__":
    main()
