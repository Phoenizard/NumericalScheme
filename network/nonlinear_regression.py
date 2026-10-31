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
        self.centered = settings.get("initialization") == "centered_normal"
        with torch.no_grad():
            if self.centered:
                hidden = torch.randn(width, generator=generator, dtype=dtype, device=device)
                self.hidden.weight.copy_((hidden * settings["hidden_scale"]).reshape(-1, 1))
                self.hidden.bias.copy_((-0.5 * hidden) * settings["hidden_scale"])
                self.output.weight.copy_((torch.randn(
                    width, generator=generator, dtype=dtype, device=device,
                ) / width ** 0.5).reshape(1, -1))
                self.output.bias.zero_()
                return
            for parameter in self.hidden.parameters():
                parameter.uniform_(
                    settings["hidden_init_min"], settings["hidden_init_max"],
                    generator=generator,
                )
            bound = width ** -0.5
            for parameter in self.output.parameters():
                parameter.uniform_(-bound, bound, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.centered:
            hidden = torch.tanh(x * self.hidden.weight.flatten() + self.hidden.bias)
            return (hidden @ self.output.weight.flatten() + self.output.bias).reshape(-1, 1)
        return self.output(torch.tanh(self.hidden(x)))
