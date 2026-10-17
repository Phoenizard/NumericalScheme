"""First-order PB-SAV for L=0 and G=-I, with optional scalar relaxation."""

import torch


def step(parameters, loss_fn, q, lr, total_shift, relax=False):
    """Update parameters in place and return the detached scalar tracker q."""
    parameters = tuple(parameters)
    terms = loss_fn()
    if isinstance(terms, torch.Tensor):
        terms = [terms]

    energies = torch.stack([term.detach() for term in terms])
    shifted = energies + total_shift / len(terms)
    if not torch.isfinite(shifted).all() or (shifted <= 0).any():
        raise ValueError("Each shifted component energy must be finite and positive")
    Q = shifted.sum().sqrt()

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
    V = gradients / (2 * shifted).sqrt()  # B = V @ V.T
    rhs = -lr * (q / Q) * force
    small = torch.eye(len(terms), dtype=V.dtype, device=V.device) + lr * (V.T @ V)
    increment = rhs - lr * V @ torch.linalg.solve(small, V.T @ rhs)
    q_trial = q + torch.dot(force, increment) / (2 * Q)

    with torch.no_grad():
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.add_(increment[offset:offset + size].reshape_as(parameter))
            offset += size

        if relax:
            new_terms = loss_fn()
            new_energy = new_terms if isinstance(new_terms, torch.Tensor) else sum(new_terms)
            # For L=0 and delta_n=D_n, q_trial^2 + D_n = q_n^2.
            q_next = torch.minimum(new_energy + total_shift, q.square()).sqrt()
        else:
            q_next = q_trial
    return q_next.detach()
