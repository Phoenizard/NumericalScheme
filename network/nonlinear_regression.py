"""A scalar regression network with one tanh hidden layer."""

import torch
from torch import nn


class TanhRegressor(nn.Module):
    """Map inputs of shape (N, 1) to predictions of shape (N, 1)."""

    def __init__(self, settings: dict):
        super().__init__()
        width = settings["hidden_width"]
        dtype = getattr(torch, settings["dtype"])
        device = settings["device"]
        self.hidden = nn.Linear(1, width, dtype=dtype, device=device)
        self.output = nn.Linear(width, 1, dtype=dtype, device=device)
        generator = torch.Generator(device=device).manual_seed(settings["init_seed"])
        with torch.no_grad():
            for parameter in self.hidden.parameters():
                parameter.uniform_(
                    settings["hidden_init_min"], settings["hidden_init_max"],
                    generator=generator,
                )
            bound = width ** -0.5
            for parameter in self.output.parameters():
                parameter.uniform_(-bound, bound, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(x)))
