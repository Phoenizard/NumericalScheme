"""Tanh PINN for viscous Burgers; the inverse problem adds a trainable xi = log(nu)."""

import math

import torch
from torch import nn


class BurgersPINN(nn.Module):
    """Map (x, t) to u(x, t) with six tanh hidden layers (21,057 parameters at width 64).

    With inverse=True the log-viscosity xi is one more trainable parameter (21,058) and
    nu = exp(xi) > 0.
    """

    def __init__(self, settings: dict, inverse: bool = False):
        super().__init__()
        factory = {"dtype": getattr(torch, settings["dtype"]), "device": settings["device"]}
        width, depth = settings["width"], settings["hidden_layers"]
        # Restore the caller's CPU RNG; collocation sampling has its own streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings["init_seed"])
            layers, size = [], 2
            for _ in range(depth):
                layers.extend([nn.Linear(size, width, **factory), nn.Tanh()])
                size = width
            layers.append(nn.Linear(size, 1, **factory))
            self.net = nn.Sequential(*layers)
        self.xi = (nn.Parameter(torch.tensor(math.log(settings["nu_init"]), **factory))
                   if inverse else None)
        self.nu_fixed = settings["nu"]

    @property
    def viscosity(self) -> torch.Tensor:
        """Current viscosity: exp(xi) in the inverse problem, the fixed value otherwise."""
        return self.xi.exp() if self.xi is not None else torch.tensor(
            self.nu_fixed, dtype=self.net[0].weight.dtype, device=self.net[0].weight.device)

    def network_parameters(self):
        """Weights and biases of the network only (weight decay excludes xi)."""
        return self.net.parameters()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape != t.shape or x.ndim != 1:
            raise ValueError("x and t must be one-dimensional tensors of equal length")
        return self.net(torch.stack((x, t), dim=1)).squeeze(1)
