"""ESAV momentum with the SAV kinetic energy, component PB correction and full Relax."""

import math

import torch

from optimizer import pb_esav


def initialize(parameters, terms, *, mobility="identity", eps=1e-8):
    """Freeze positive Ei0; initialize total/component log states and unscaled momentum p."""
    if mobility not in ("identity", "amsgrad"):
        raise ValueError("PB-ESAV mobility must be identity or amsgrad")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("PB-ESAV mobility eps must be finite and positive")
    parameters = tuple(parameters)
    state = pb_esav.initialize(terms)
    flat = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    zero = flat.new_zeros(())
    state.update(
        p=torch.zeros_like(flat), learning_rate=zero.clone(),
        mobility=torch.full_like(flat, 1 / eps if mobility == "amsgrad" else 1),
        iteration=torch.zeros((), dtype=torch.int64, device=flat.device),
    )
    for name in ("kinetic_energy", "dissipation", "energy_law_error", "accepted_dissipation",
                 "mobility_increase", "auxiliary_increment",
                 "relax_log_correction"):
        state[name] = zero.clone()
    if mobility == "amsgrad":
        state.update(v=torch.zeros_like(flat), v_max=torch.zeros_like(flat))
    return state


def step(parameters, loss_fn, state, lr, *, momentum=0.9, alpha=0.5, relax=False,
         mobility="identity", beta2=0.999, eps=1e-8):
    """Update parameters in place and return detached state; leave input state unchanged.

    The displacement is M_next*p_next, as in PB-SAV. The dissipated energy is
    4*E0*log_r + p.T*M*p/(2*lr), with fixed lr. alpha multiplies the current PB
    matrix: alpha=0 gives ordinary ESAV momentum and alpha=1 gives full PB.
    Full Relax changes only total log_r; component log states follow their equations.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("PB-ESAV momentum requires a finite positive learning rate")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("PB-ESAV momentum requires 0 <= momentum < 1")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("PB-ESAV momentum requires 0 <= alpha <= 1")
    if mobility not in ("identity", "amsgrad"):
        raise ValueError("PB-ESAV mobility must be identity or amsgrad")
    if not math.isfinite(beta2) or not 0 <= beta2 < 1:
        raise ValueError("PB-ESAV mobility requires 0 <= beta2 < 1")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("PB-ESAV mobility eps must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    energies = torch.stack([term.detach() for term in terms])
    if not torch.isfinite(energies).all() or (energies < 0).any():
        raise FloatingPointError("PB-ESAV momentum requires finite nonnegative energies")
    pb_esav._validate_state(state, energies)
    references = state["initial_components"]
    gradients = pb_esav.component_gradients(terms, parameters)
    force = gradients.sum(dim=1)
    for key in ("p", "mobility") + (("v", "v_max") if mobility == "amsgrad" else ()):
        if (key not in state or state[key].shape != force.shape
                or state[key].dtype != force.dtype or state[key].device != force.device):
            raise ValueError(f"Initialize PB-ESAV {key} with matching shape, dtype and device")
    if (not all(torch.isfinite(value).all() for value in state.values())
            or (references <= 0).any() or state["initial_energy"] <= 0
            or (state["mobility"] <= 0).any()):
        raise FloatingPointError("Invalid PB-ESAV momentum state")
    if mobility == "identity" and not torch.equal(state["mobility"], torch.ones_like(force)):
        raise ValueError("Cannot switch mobility during PB-ESAV momentum evolution")
    if int(state["iteration"]) and state["learning_rate"] != force.new_tensor(lr):
        raise ValueError("PB-ESAV momentum requires the learning rate to remain fixed")
    old_metric = state["mobility"]
    iteration = state["iteration"] + 1
    metric = old_metric
    adaptive = {}
    if mobility == "amsgrad":
        if (state["v"] < 0).any() or (state["v_max"] < 0).any():
            raise ValueError("PB-ESAV requires nonnegative adaptive moments")
        v = beta2 * state["v"] + (1 - beta2) * force.square()
        v_max = torch.maximum(state["v_max"], v / (1 - beta2 ** int(iteration)))
        metric = (v_max.sqrt() + eps).reciprocal()
        adaptive = {"v": v, "v_max": v_max}
    ratio = (state["log_r"] - energies.sum() / (4 * state["initial_energy"])).exp()
    component_ratios = (state["log_r_i"] - energies / (4 * references)).exp()
    component_ratios.masked_fill_(component_ratios == 0, torch.finfo(energies.dtype).tiny)
    if (not torch.isfinite(ratio) or ratio <= 0 or not torch.isfinite(component_ratios).all()
            or not torch.isfinite(metric).all() or (metric <= 0).any()
            or (metric > old_metric).any()):
        raise FloatingPointError(f"Invalid PB-ESAV scaling at update {int(iteration)}")
    W = gradients * (math.sqrt(alpha) * component_ratios / (4 * references).sqrt())
    rhs = momentum * state["p"] - lr * ratio * force
    if alpha == 0:
        increment = metric * rhs
    else:
        # Include the RHS so historical momentum outside the gradient span is retained.
        metric_sqrt = metric.sqrt()
        columns = torch.cat((metric_sqrt[:, None] * W, (metric_sqrt * rhs)[:, None]), dim=1)
        basis, reduced = torch.linalg.qr(columns, mode="reduced")
        small = torch.eye(basis.shape[1], dtype=force.dtype, device=force.device)
        small = small + lr * (reduced[:, :-1] @ reduced[:, :-1].T)
        if not torch.isfinite(small).all():
            raise FloatingPointError(f"Nonfinite PB-ESAV linear system at update {int(iteration)}")
        try:
            increment = metric_sqrt * (basis @ torch.linalg.solve(small, reduced[:, -1]))
        except torch.linalg.LinAlgError as error:
            raise FloatingPointError(f"PB-ESAV solve failed at update {int(iteration)}") from error
    p_next = increment / metric
    log_change = ratio * force.dot(increment) / (4 * state["initial_energy"])
    trial_log_r = state["log_r"] + log_change
    next_log_r_i = state["log_r_i"] + (
        component_ratios * (gradients.T @ increment) / (4 * references)
    )
    old_kinetic = (old_metric * state["p"].square()).sum() / (2 * lr)
    new_kinetic = (metric * p_next.square()).sum() / (2 * lr)
    dissipation = (
        ((old_metric - metric) * state["p"].square()).sum()
        + (metric * (p_next - momentum * state["p"]).square()).sum()
        + (1 - momentum ** 2) * (metric * state["p"].square()).sum()
    ) / (2 * lr) + (W.T @ increment).square().sum()
    old_energy = 4 * state["initial_energy"] * state["log_r"] + old_kinetic
    law_error = old_energy - 4 * state["initial_energy"] * trial_log_r
    law_error = law_error - new_kinetic - dissipation
    if not all(torch.isfinite(value).all() for value in (
            increment, p_next, trial_log_r, next_log_r_i, new_kinetic, dissipation, law_error)):
        raise FloatingPointError(f"Nonfinite PB-ESAV candidate at update {int(iteration)}")
    previous = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        offset = 0
        candidates = []
        for parameter in parameters:
            size = parameter.numel()
            candidates.append(parameter + increment[offset:offset + size].reshape_as(parameter))
            offset += size
        if not all(torch.isfinite(value).all() for value in candidates):
            raise FloatingPointError(f"Nonfinite PB-ESAV parameters at update {int(iteration)}")
        for parameter, value in zip(parameters, candidates):
            parameter.copy_(value)
        next_log_r = trial_log_r
        if relax:
            try:
                new_terms = loss_fn()
                new_terms = [new_terms] if isinstance(new_terms, torch.Tensor) else list(new_terms)
                energies_next = torch.stack(new_terms)
                if not torch.isfinite(energies_next).all() or (energies_next < 0).any():
                    raise FloatingPointError("Full Relax requires finite nonnegative energies")
                budget = old_energy - new_kinetic
                next_log_r = torch.minimum(energies_next.sum(), budget)
                next_log_r = next_log_r / (4 * state["initial_energy"])
                if not torch.isfinite(next_log_r):
                    raise FloatingPointError("Invalid PB-ESAV full-Relax energy budget")
            except Exception:
                for parameter, value in zip(parameters, previous):
                    parameter.copy_(value)
                raise
    result = {
        "log_r": next_log_r, "log_r_i": next_log_r_i,
        "initial_energy": state["initial_energy"].clone(),
        "initial_components": references.clone(), "p": p_next,
        "mobility": metric, "iteration": iteration, "learning_rate": force.new_tensor(lr),
        "kinetic_energy": new_kinetic, "dissipation": dissipation,
        "energy_law_error": law_error,
        "accepted_dissipation": old_energy - 4 * state["initial_energy"] * next_log_r
        - new_kinetic,
        "mobility_increase": (metric - old_metric).max(),
        "auxiliary_increment": log_change, "relax_log_correction": next_log_r - trial_log_r,
        **adaptive,
    }
    return {key: value.detach() for key, value in result.items()}
