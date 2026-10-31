"""Noisy sine data, nested random energy splitting, and clean test error."""

from pathlib import Path

import numpy as np
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
    if settings.get("data_rng") == "numpy":
        x = np.linspace(settings["x_min"], settings["x_max"], settings["n_train"])
        noise = np.random.RandomState(settings["noise_seed"]).standard_normal(len(x))
        x_train = torch.tensor(x, dtype=dtype, device=device).reshape(-1, 1)
        y_train = torch.tensor(
            np.sin(2 * np.pi * x) + settings["noise_std"] * noise,
            dtype=dtype, device=device,
        ).reshape(-1, 1)
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
        data_rng=settings.get("data_rng", "torch"),
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


def make_groups(n: int, m: int, seed: int, rng="torch"):
    """Bisect the largest group of a fixed random permutation to obtain nested partitions."""
    if not 1 <= m <= n:
        raise ValueError("Require 1 <= number of groups <= number of samples")
    generator = torch.Generator().manual_seed(seed)
    groups = [torch.randperm(n, generator=generator)]
    if rng == "numpy":
        groups = [torch.tensor(np.random.RandomState(seed).permutation(n))]
    elif rng != "torch":
        raise ValueError(f"Unsupported partition RNG: {rng}")
    while len(groups) < m:
        index = max(range(len(groups)), key=lambda i: len(groups[i]))
        groups[index:index + 1] = list(groups[index].tensor_split(2))
    return groups


def split_energy_loss(model, x, y, groups, lambda_reg: float = 0.0):
    """Return energies for disjoint index groups, all normalized by total sample count 2N."""
    residual_squared = (model(x) - y).square()
    terms = [residual_squared[group].sum() / (2 * len(x)) for group in groups]
    if lambda_reg:
        regularization = lambda_reg * sum(p.square().sum() for p in model.parameters()) / 2
        terms = [term + regularization / len(groups) for term in terms]
    return terms


def test_mse(model, x, y) -> float:
    """Return mean squared prediction error against the noise-free target."""
    return (model(x) - y).square().mean().item()


@torch.no_grad()
def controller_geometry(model, x, y, groups, method, total_shift, state):
    """Return the analytic sample Jacobian and the method's native B factor."""
    hidden = torch.tanh(x * model.hidden.weight.flatten() + model.hidden.bias)
    derivative = (1 - hidden.square()) * model.output.weight.flatten()
    jacobian = torch.cat((derivative * x, derivative, hidden, torch.ones_like(x)), dim=1)
    if method == "esav":
        return jacobian, jacobian.new_empty((jacobian.shape[1], 0))
    residual = (model(x) - y).flatten()
    if method == "sav":
        groups = [torch.arange(len(x))]
    energies = torch.stack([residual[group].square().sum() / (2 * len(x)) for group in groups])
    gradients = torch.stack([jacobian[group].T @ residual[group] / len(x) for group in groups], 1)
    if method in ("sav", "pb_sav"):
        columns = gradients / (2 * (energies + total_shift / len(groups))).sqrt()
    elif method == "pb_esav":
        references = state["initial_components"]
        ratios = (state["log_r_i"] - energies / (4 * references)).exp()
        ratios.masked_fill_(ratios == 0, torch.finfo(ratios.dtype).tiny)
        columns = gradients * (ratios / (4 * references).sqrt())
    else:
        raise ValueError(f"No curvature controller geometry for {method}")
    return jacobian, columns


if __name__ == "__main__":
    print(f"Data: {generate_data()}")
