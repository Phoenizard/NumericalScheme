"""PB-ESAV with local auxiliary exp(E_i + 0.5 * log(E_i + C/M))."""

import math

import torch


def step(parameters, loss_fn, state, lr, scheme="plan1", relax=False, eta=None,
         *, curvature_shift):
    """Update parameters in place; return detached total and optional local log states.

    Initialize log_r = sum(E_i); Plan 2 needs log_r_i = E_i + 0.5 * log(E_i + C/M).
    Curvature uses E_i + curvature_shift / M, which must be positive.
    The total auxiliary and the objective remain unshifted.
    """
    if scheme not in ("plan1", "plan2") or relax or eta is not None:
        raise ValueError("PB-ESAV implements non-relaxed Plans 1 and 2 only")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    energies = torch.stack([term.detach() for term in terms])
    shifted = energies + curvature_shift / len(terms)
    if not torch.isfinite(shifted).all() or (shifted <= 0).any():
        raise ValueError("Each shifted component energy must be finite and positive")
    coefficients = 1 + 1 / (2 * shifted)
    ratio = (state["log_r"] - energies.sum()).exp()
    if scheme == "plan1":
        weights = ratio.expand_as(energies)
    else:
        if state["log_r_i"].shape != energies.shape:
            raise ValueError("The number of energy components must remain constant")
        local_energy = energies + 0.5 * shifted.log()
        weights = (state["log_r_i"] - local_energy).exp()
    if not all(torch.isfinite(value).all() for value in (energies, ratio, weights)):
        raise FloatingPointError("Non-finite PB-ESAV energy or auxiliary ratio")

    columns = []
    for i, term in enumerate(terms):
        derivatives = torch.autograd.grad(
            term, parameters, retain_graph=i < len(terms) - 1, allow_unused=True,
        ) if term.requires_grad else [None] * len(parameters)
        columns.append(torch.cat([
            (value if value is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter, value in zip(parameters, derivatives)
        ]))
    gradients = torch.stack(columns, dim=1)
    force = gradients.sum(dim=1)

    # B = V @ V.T; Plan 2 retains each component's evolving auxiliary ratio.
    V = gradients * (weights * coefficients).sqrt()

    rhs = -lr * ratio * force
    small = torch.eye(len(terms), dtype=V.dtype, device=V.device) + lr * (V.T @ V)
    increment = rhs - lr * V @ torch.linalg.solve(small, V.T @ rhs)
    next_state = {"log_r": (state["log_r"] + torch.dot(force, increment)).detach()}
    if scheme == "plan2":
        next_state["log_r_i"] = (
            state["log_r_i"] + coefficients * (gradients.T @ increment)
        ).detach()
    if not all(torch.isfinite(value).all() for value in (increment, *next_state.values())):
        raise FloatingPointError("Non-finite PB-ESAV update")

    with torch.no_grad():
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.add_(increment[offset:offset + size].reshape_as(parameter))
            offset += size
    return next_state
