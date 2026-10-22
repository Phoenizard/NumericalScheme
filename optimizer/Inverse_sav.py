"""Inverse SAV and a centered pullback extension for L=0 and G=-I."""

import math
from collections.abc import Callable, Iterable

import torch


def step(
    parameters: Iterable[torch.Tensor],
    loss_fn: Callable,
    r: torch.Tensor,
    lr: float,
    total_shift: float,
    relax: bool = False,
    *,
    pullback: bool = False,
) -> torch.Tensor:
    """Update parameters in place and return detached r without mutating its input.

    Initialize r=1/(E+C); the dissipating auxiliary energy is 1/r.
    The pullback variant uses centered component gradients and one total tracker.
    Its M=1 update is exactly the single-energy update. No line search is used here.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(total_shift) or total_shift < 0:
        raise ValueError("total_shift must be finite and nonnegative")
    if relax:
        raise ValueError("Inverse SAV does not use Relax")
    if r.numel() != 1 or not torch.isfinite(r).all() or (r <= 0).any():
        raise FloatingPointError("Inverse SAV requires a finite positive scalar r")

    parameters = tuple(parameters)
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms or any(term.numel() != 1 for term in terms):
        raise ValueError("loss_fn must return a scalar or a nonempty sequence of scalars")
    if not pullback:
        terms = [sum(terms)]
    shifted = torch.stack([term.detach() for term in terms]) + total_shift / len(terms)
    if not torch.isfinite(shifted).all() or (shifted <= 0).any():
        raise FloatingPointError("Inverse SAV requires every shifted energy to be positive")
    total = shifted.sum()

    columns = []
    for index, term in enumerate(terms):
        derivatives = torch.autograd.grad(
            term, parameters, retain_graph=index < len(terms) - 1, allow_unused=True,
        ) if term.requires_grad else [None] * len(parameters)
        columns.append(torch.cat([
            (value if value is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter, value in zip(parameters, derivatives)
        ]))
    gradients = torch.stack(columns, dim=1)
    force = gradients.sum(dim=1)
    direction = force
    if pullback and len(terms) > 1:
        # Centering removes the total-gradient correction already handled by r_next.
        centered = (gradients - force[:, None] * (shifted / total)) / shifted.sqrt()
        scaled = (r.detach().reciprocal() / total) * centered
        if not torch.isfinite(scaled).all():
            raise FloatingPointError("Nonfinite inverse-SAV pullback factor")
        # Solve (I + lr*scaled*scaled.T)d=force without forming its ill-conditioned Gram matrix.
        # A full orthogonal basis avoids subtracting almost equal Woodbury terms.
        try:
            basis, singular_values, _ = torch.linalg.svd(scaled, full_matrices=True)
        except torch.linalg.LinAlgError as error:
            raise FloatingPointError("Inverse-SAV pullback factorization failed") from error
        coordinates = basis.T @ force
        gains = torch.ones_like(force)
        denominators = 1 + lr * singular_values.square()
        if not torch.isfinite(denominators).all():
            raise FloatingPointError("Nonfinite inverse-SAV spectral denominator")
        gains[:singular_values.numel()] = 1 / denominators
        direction = basis @ (gains * coordinates)
        gradient_measure = torch.linalg.vector_norm(gains.sqrt() * coordinates)
    else:
        gradient_measure = torch.linalg.vector_norm(force)

    # Equivalent to the positive quadratic root, avoiding explicit squaring of r.
    root = torch.hypot(
        r.detach(), 2 * math.sqrt(lr) * (gradient_measure / total) / total.sqrt(),
    )
    next_r = 0.5 * r.detach() + 0.5 * root
    increment = -lr * (next_r.reciprocal() / total) * direction
    if not torch.isfinite(next_r).all() or not torch.isfinite(increment).all():
        raise FloatingPointError("Nonfinite inverse-SAV state or parameter increment")

    with torch.no_grad():
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.add_(increment[offset:offset + size].reshape_as(parameter))
            offset += size
    return next_r.detach()
