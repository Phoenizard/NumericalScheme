"""Convolutional-branch DeepONet with an exact homogeneous Dirichlet boundary."""

import torch
from torch import nn


class DeepONet(nn.Module):
    """Map fields (N, 1, H, W) and coordinates (P, 2) to solution values (N, P)."""

    def __init__(self, settings: dict):
        super().__init__()
        factory = {"dtype": getattr(torch, settings["dtype"]), "device": settings["device"]}
        channels = settings["conv_channels"]
        kernels = settings["conv_kernels"]
        strides = settings["conv_strides"]
        paddings = settings["conv_paddings"]
        if not len(channels) == len(kernels) == len(strides) == len(paddings) == 3:
            raise ValueError("Darcy DeepONet requires three configured convolution layers")
        width = settings["latent_dim"]
        # Restore the caller's CPU RNG; data sampling has its own NumPy streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings["init_seed"])
            branch = []
            input_channels = 1
            for output_channels, kernel, stride, padding in zip(
                    channels, kernels, strides, paddings):
                branch.extend([
                    nn.Conv2d(input_channels, output_channels, kernel, stride, padding,
                              **factory),
                    nn.ReLU(),
                ])
                input_channels = output_channels
            branch.extend([
                nn.AdaptiveAvgPool2d((settings["pool_size"], settings["pool_size"])),
                nn.Flatten(),
                nn.Linear(channels[-1] * settings["pool_size"] ** 2, width, **factory),
            ])
            self.branch = nn.Sequential(*branch)
            trunk = []
            input_width = 2
            for _ in range(settings["trunk_hidden_layers"]):
                trunk.extend([nn.Linear(input_width, width, **factory), nn.Tanh()])
                input_width = width
            trunk.append(nn.Linear(width, width, **factory))
            self.trunk = nn.Sequential(*trunk)
            self.output_bias = nn.Parameter(torch.zeros((), **factory))

    def forward(self, permeability: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        if permeability.ndim != 4 or permeability.shape[1] != 1:
            raise ValueError("permeability must have shape (N, 1, H, W)")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (P, 2)")
        x, y = coordinates.unbind(dim=1)
        boundary = 16 * x * (1 - x) * y * (1 - y)
        return (self.branch(permeability) @ self.trunk(coordinates).T + self.output_bias) * boundary
