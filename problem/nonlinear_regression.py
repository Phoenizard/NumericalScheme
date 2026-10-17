"""Noisy sine data, round-robin energy splitting, and clean test error."""

from pathlib import Path

import torch

from problem.config import PROBLEMS


def generate_data() -> Path:
    """Save fixed training and noise-free test data from the problem settings."""
    settings = PROBLEMS["nonlinear_regression"]
    dtype = getattr(torch, settings["dtype"])
    device = settings["device"]
    generator = torch.Generator(device=device).manual_seed(settings["noise_seed"])
    x_train = torch.linspace(
        settings["x_min"], settings["x_max"], settings["n_train"],
        dtype=dtype, device=device,
    ).reshape(-1, 1)
    noise = torch.randn(x_train.shape, generator=generator, dtype=dtype, device=device)
    y_train = torch.sin(2 * torch.pi * x_train) + settings["noise_std"] * noise
    x_test = torch.linspace(
        settings["x_min"], settings["x_max"], settings["n_test"],
        dtype=dtype, device=device,
    ).reshape(-1, 1)
    data_config = {
        key: settings[key] for key in (
            "n_train", "n_test", "x_min", "x_max", "noise_std", "noise_seed",
            "device", "dtype",
        )
    }
    data_config.update(
        sampling="linspace", target="sin(2*pi*x)", noise="normal",
        torch_version=str(torch.__version__),
    )
    path = Path(__file__).resolve().parents[1] / settings["data_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": torch.sin(2 * torch.pi * x_test),
        "config": data_config,
    }, path)
    return path


def split_energy_loss(model, x, y, m: int, lambda_reg: float = 0.0):
    """Return M energies for sorted (N, 1) sample pairs, normalized by 2N."""
    residual_squared = (model(x) - y).square()
    terms = [residual_squared[k::m].sum() / (2 * len(x)) for k in range(m)]
    if lambda_reg:
        regularization = lambda_reg * sum(p.square().sum() for p in model.parameters()) / 2
        terms = [term + regularization / m for term in terms]
    return terms


def test_mse(model, x, y) -> float:
    """Return mean squared prediction error against the noise-free target."""
    return (model(x) - y).square().mean().item()


if __name__ == "__main__":
    print(f"Data: {generate_data()}")
