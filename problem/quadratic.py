"""The 100-dimensional quadratic benchmark from PB-SAV, Section 5.1."""

import torch


def init_parameters(settings: dict):
    """Create a constant initial vector from the fixed problem settings."""
    return torch.full(
        (settings["dimension"],), settings["initial_value"],
        dtype=getattr(torch, settings["dtype"]), device=settings["device"],
        requires_grad=True,
    )


def split_energy_loss(x, settings: dict):
    """Split Energy Loss: return [E1, E2] for the two curvature classes.

    L2 regularization adds lambda_reg / 2 * ||x||^2,
    distributed across the same two coordinate groups. Use lambda_reg=0
    to reproduce the paper. Positive SAV shifts belong to the optimizer.
    """
    fast_weight, slow_weight = settings["energy_weights"]
    lambda_reg = settings["lambda_reg"]
    fast = (fast_weight + 0.5 * lambda_reg) * x[0::2].square().sum()
    slow = (slow_weight + 0.5 * lambda_reg) * x[1::2].square().sum()
    return [fast, slow]


def total_energy_loss(x, settings: dict):
    """Total Energy Loss: return E1 + E2, including the same regularization."""
    return sum(split_energy_loss(x, settings))
