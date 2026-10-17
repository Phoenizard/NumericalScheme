"""The 100-dimensional quadratic benchmark from PB-SAV, Section 5.1."""

import torch


def init_parameters(device="cpu"):
    """Start from x = (1, ..., 1), using double precision."""
    return torch.ones(100, dtype=torch.float64, device=device, requires_grad=True)


def split_energy_loss(x, lambda_reg=0.0):
    """Split Energy Loss: return [E1, E2] for the two curvature classes.

    L2 regularization adds lambda_reg / 2 * ||x||^2,
    distributed across the same two coordinate groups. Use lambda_reg=0
    to reproduce the paper. Positive SAV shifts belong to the optimizer.
    """
    fast = (1.0 + 0.5 * lambda_reg) * x[0::2].square().sum()
    slow = (0.01 + 0.5 * lambda_reg) * x[1::2].square().sum()
    return [fast, slow]


def total_energy_loss(x, lambda_reg=0.0):
    """Total Energy Loss: return E1 + E2, including the same regularization."""
    return sum(split_energy_loss(x, lambda_reg=lambda_reg))
