"""Full-batch SGD with optional heavy-ball momentum on the supplied objective."""

import math

import torch

from optimizer.gd import step as gd_step


def step(parameters, loss_fn, lr: float, *, momentum=0.0, state=None):
    """Update parameters in place and return a new, detached displacement state.

    state['p'] is the flat previous displacement; None means zero initial momentum.
    Coupled L2 regularization belongs in loss_fn. Zero momentum preserves plain GD.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("momentum must be finite and in [0, 1)")
    if momentum == 0:
        gd_step(parameters, loss_fn, lr)
        return None

    parameters = tuple(parameters)
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    gradients = torch.cat([
        gradient.reshape(-1) for gradient in torch.autograd.grad(energy, parameters)
    ])
    previous = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    displacement = torch.zeros_like(previous) if state is None else state["p"]
    if (displacement.shape != previous.shape or displacement.dtype != previous.dtype
            or displacement.device != previous.device):
        raise ValueError("SGD displacement must match parameter shape, dtype and device")
    candidate = previous - lr * gradients + momentum * displacement
    if not torch.isfinite(energy).all() or not torch.isfinite(candidate).all():
        raise FloatingPointError("Non-finite SGD objective, gradient or momentum update")
    with torch.no_grad():
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.copy_(candidate[offset:offset + size].reshape_as(parameter))
            offset += size
    return {"p": (candidate - previous).detach()}
