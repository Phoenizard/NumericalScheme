"""PB-SAV momentum with Euclidean or monotone AMSGrad-type diagonal mobility."""

import math

import torch


def initialize(parameters, terms, total_shift: float, *, mobility="identity", eps=1e-8) -> dict:
    """Initialize q=sqrt(sum(E_i)+C), zero displacement and energy diagnostics."""
    if not math.isfinite(total_shift) or total_shift < 0:
        raise ValueError("PB-SAV momentum requires a finite nonnegative total_shift")
    if mobility not in ("identity", "amsgrad"):
        raise ValueError("PB-SAV mobility must be identity or amsgrad")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("PB-SAV mobility eps must be finite and positive")
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    shifted = torch.stack([term.detach() for term in terms]) + total_shift / len(terms)
    if shifted.ndim != 1 or not torch.isfinite(shifted).all() or (shifted <= 0).any():
        raise ValueError("PB-SAV momentum requires positive finite shifted component energies")
    flat = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    zero = shifted.new_zeros(())
    state = {"q": shifted.sum().sqrt(), "p": torch.zeros_like(flat),
             "dissipation": zero, "energy_law_error": zero.clone()}
    if mobility == "amsgrad":
        state.update(v=torch.zeros_like(flat), v_max=torch.zeros_like(flat),
                     mobility=torch.full_like(flat, 1 / eps),
                     iteration=torch.zeros((), dtype=torch.int64, device=flat.device),
                     accepted_dissipation=zero.clone(), mobility_increase=zero.clone())
    return state


def step(parameters, loss_fn, state: dict, lr: float, *, total_shift: float,
         momentum: float = 0.9, alpha: float = 0.5, relax: bool = False,
         mobility="identity", beta2=0.999, eps=1e-8) -> dict:
    """Update parameters in place; return detached state without mutating input state.

    Equal component shifts sum to total_shift. The previous displacement is M*p.
    The dissipated energy is q**2 + p.T*M*p/(2*lr); lr must remain fixed.
    Optional full Relax includes kinetic energy in its scalar energy budget.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("PB-SAV momentum requires a finite positive learning rate")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("PB-SAV momentum requires 0 <= momentum < 1")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("PB-SAV momentum requires 0 <= alpha <= 1")
    if not math.isfinite(total_shift) or total_shift < 0:
        raise ValueError("PB-SAV momentum requires a finite nonnegative total_shift")
    if mobility not in ("identity", "amsgrad"):
        raise ValueError("PB-SAV mobility must be identity or amsgrad")
    if not math.isfinite(beta2) or not 0 <= beta2 < 1:
        raise ValueError("PB-SAV mobility requires 0 <= beta2 < 1")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("PB-SAV mobility eps must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    shifted = torch.stack([term.detach() for term in terms]) + total_shift / len(terms)
    if shifted.ndim != 1 or not torch.isfinite(shifted).all() or (shifted <= 0).any():
        raise FloatingPointError("PB-SAV momentum encountered nonpositive component energy")
    Q = shifted.sum().sqrt()
    columns = []
    for i, term in enumerate(terms):
        gradients = torch.autograd.grad(
            term, parameters, retain_graph=i < len(terms) - 1, allow_unused=True,
        ) if term.requires_grad else [None] * len(parameters)
        columns.append(torch.cat([
            (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]))
    gradients = torch.stack(columns, dim=1)
    force = gradients.sum(dim=1)
    q, p = state["q"], state["p"]
    if (q.ndim != 0 or p.shape != force.shape or p.dtype != force.dtype
            or p.device != force.device or q.dtype != force.dtype or q.device != force.device):
        raise ValueError("PB-SAV momentum state must match parameter shape, dtype and device")
    V = gradients / (2 * shifted).sqrt()
    c = shifted.sqrt() / Q
    # W W.T = (1-alpha) B_1 + alpha B_M, with only M columns (paper Eq. 48).
    W = math.sqrt(alpha) * V + (1 - math.sqrt(alpha)) * (V @ c)[:, None] * c
    rhs = momentum * p - lr * (q / Q) * force
    if mobility == "amsgrad":
        for key in ("v", "v_max", "mobility"):
            if key not in state or state[key].shape != force.shape:
                raise ValueError("Initialize PB-SAV with mobility='amsgrad' before stepping")
            if (state[key].dtype != force.dtype or state[key].device != force.device
                    or not torch.isfinite(state[key]).all()):
                raise ValueError("PB-SAV mobility state must be finite and match parameters")
        if (state["v"] < 0).any() or (state["v_max"] < 0).any() or (state["mobility"] <= 0).any():
            raise ValueError("PB-SAV requires nonnegative moments and positive mobility")
        iteration = state["iteration"] + 1
        v = beta2 * state["v"] + (1 - beta2) * force.square()
        v_max = torch.maximum(state["v_max"], v / (1 - beta2 ** int(iteration)))
        metric = (v_max.sqrt() + eps).reciprocal()
        metric_W = metric[:, None] * W
        metric_rhs = metric * rhs
        small = torch.eye(len(terms), dtype=force.dtype, device=force.device)
        small = small + lr * (W.T @ metric_W)
        increment = metric_rhs - lr * metric_W @ torch.linalg.solve(small, W.T @ metric_rhs)
        p_next = increment / metric
        old_kinetic = (state["mobility"] * p.square()).sum() / (2 * lr)
        new_kinetic = (metric * p_next.square()).sum() / (2 * lr)
        inertial_dissipation = (
            ((state["mobility"] - metric) * p.square()).sum()
            + (metric * (p_next - momentum * p).square()).sum()
            + (1 - momentum ** 2) * (metric * p.square()).sum()
        ) / (2 * lr)
    else:
        small = torch.eye(len(terms), dtype=force.dtype, device=force.device) + lr * (W.T @ W)
        increment = rhs - lr * W @ torch.linalg.solve(small, W.T @ rhs)
        p_next = increment
        old_kinetic = p.square().sum() / (2 * lr)
        new_kinetic = increment.square().sum() / (2 * lr)
        inertial_dissipation = ((increment - momentum * p).square().sum() / (2 * lr)
                               + (1 - momentum ** 2) * old_kinetic)
    scalar_increment = torch.dot(force, increment) / (2 * Q)
    q_trial = q + scalar_increment
    # Weighted variance evaluates increment.T @ (B_M-B_1) @ increment stably.
    component_slopes = gradients.T @ increment / shifted
    mean_slope = torch.dot(force, increment) / shifted.sum()
    curvature_gap = (shifted * (component_slopes - mean_slope).square()).sum() / 2
    dissipation = inertial_dissipation + scalar_increment.square() + alpha * curvature_gap
    energy_law_error = q.square() + old_kinetic - q_trial.square() - new_kinetic - dissipation
    if not all(torch.isfinite(value).all() for value in (
            q_trial, increment, p_next, dissipation, energy_law_error)):
        raise FloatingPointError("Non-finite PB-SAV momentum update or energy diagnostic")
    previous = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.add_(increment[offset:offset + size].reshape_as(parameter))
            offset += size
        if relax:
            try:
                new_terms = loss_fn()
                energy = new_terms if isinstance(new_terms, torch.Tensor) else sum(new_terms)
                budget = q.square() + old_kinetic - new_kinetic
                target = torch.minimum(energy + total_shift, budget)
                if not torch.isfinite(target) or target < 0:
                    raise FloatingPointError("Invalid PB-SAV momentum full-Relax energy budget")
                q_trial = target.sqrt()
            except Exception:
                for parameter, value in zip(parameters, previous):
                    parameter.copy_(value)
                raise
    next_state = {"q": q_trial.detach(), "p": p_next.detach(),
                  "dissipation": dissipation.detach(), "energy_law_error": energy_law_error.detach()}
    if mobility == "amsgrad":
        next_state.update(
            v=v.detach(), v_max=v_max.detach(), mobility=metric.detach(), iteration=iteration,
            accepted_dissipation=(q.square() + old_kinetic - q_trial.square() - new_kinetic).detach(),
            mobility_increase=(metric - state["mobility"]).max().detach(),
        )
    return next_state
