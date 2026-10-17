"""PB-Exp-SAV Plans 1 and 2, with auxiliary variables stored as logarithms."""

import math

import torch


def step(parameters, loss_fn, state, lr, scheme="plan1", relax=False, eta=None):
    """Update parameters in place; return detached log_r and optional log_r_i.

    Initialize state with log_r = sum(E_i); Plan 2 also needs log_r_i = E_i.
    Gradients are evaluated before the step. Relaxation uses new energies
    after the parameter update, preserving a nonincreasing total log_r.
    Only relaxed Plan 1 accepts eta, the allowed fraction of dissipation
    added back to the trial log_r (default: 1, the original full budget).
    """
    if scheme not in ("plan1", "plan2"):
        raise ValueError(f"Unsupported PB-Exp-SAV scheme: {scheme}")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if eta is not None:
        if scheme != "plan1" or not relax:
            raise ValueError("eta is only supported by relaxed PB-Exp-SAV Plan 1")
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("eta must be finite and in [0, 1]")
    eta = 1.0 if eta is None else eta
    parameters = tuple(parameters)
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    energies = torch.stack([term.detach() for term in terms])
    ratio = (state["log_r"] - energies.sum()).exp()
    if scheme == "plan1":
        weights = ratio.expand_as(energies)
    else:
        if state["log_r_i"].shape != energies.shape:
            raise ValueError("The number of energy components must remain constant")
        weights = (state["log_r_i"] - energies).exp()
    if not all(torch.isfinite(value).all() for value in (energies, ratio, weights)):
        raise FloatingPointError("Non-finite PB-Exp-SAV energy or auxiliary ratio")

    columns = []
    for i, term in enumerate(terms):
        gradients = torch.autograd.grad(
            term, parameters, retain_graph=i < len(terms) - 1, allow_unused=True
        ) if term.requires_grad else [None] * len(parameters)
        columns.append(torch.cat([
            (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]))
    gradients = torch.stack(columns, dim=1)
    force = gradients.sum(dim=1)
    V = gradients * weights.sqrt()  # B = V @ V.T
    rhs = -lr * ratio * force
    small = torch.eye(len(terms), dtype=V.dtype, device=V.device) + lr * (V.T @ V)
    increment = rhs - lr * V @ torch.linalg.solve(small, V.T @ rhs)
    next_state = {"log_r": (state["log_r"] + torch.dot(force, increment)).detach()}
    if scheme == "plan2":
        next_state["log_r_i"] = (state["log_r_i"] + gradients.T @ increment).detach()
    if not all(torch.isfinite(value).all() for value in (increment, *next_state.values())):
        raise FloatingPointError("Non-finite PB-Exp-SAV update")

    with torch.no_grad():
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.add_(increment[offset:offset + size].reshape_as(parameter))
            offset += size
        if relax:
            new_terms = loss_fn()
            new_terms = [new_terms] if isinstance(new_terms, torch.Tensor) else list(new_terms)
            new_energies = torch.stack(new_terms)
            if new_energies.shape != energies.shape:
                raise ValueError("The number of energy components must remain constant")
            if not torch.isfinite(new_energies).all():
                raise FloatingPointError("Non-finite energy after PB-Exp-SAV parameter update")
            new_energy = new_energies.sum()
            if scheme == "plan1":
                if eta == 1:
                    # Preserve the original floating-point path at full relaxation.
                    upper = state["log_r"]
                else:
                    trial_log = next_state["log_r"]
                    dissipation = state["log_r"] - trial_log
                    upper = trial_log + eta * dissipation
                next_state["log_r"] = torch.minimum(new_energy, upper)
            else:
                if new_energy <= state["log_r"]:
                    next_state["log_r_i"] = new_energies
                else:
                    trial_logs = next_state["log_r_i"]
                    denominator = new_energy - trial_logs.sum()
                    if denominator <= 0:
                        raise FloatingPointError("Nonpositive PB-Exp-SAV relaxation denominator")
                    xi = (new_energy - state["log_r"]) / denominator
                    next_state["log_r_i"] = xi * trial_logs + (1 - xi) * new_energies
                next_state["log_r"] = next_state["log_r_i"].sum()
            if not all(torch.isfinite(value).all() for value in next_state.values()):
                raise FloatingPointError("Non-finite PB-Exp-SAV relaxed state")
    return next_state
