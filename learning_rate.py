"""Shared curvature damping and growth envelope for nonlinear auxiliary methods."""

import torch


def curvature_rate(previous_lr, geometry, settings):
    """Return a trial rate from first-order geometry; do not mutate optimizer state."""
    jacobian, columns = geometry
    gram = jacobian.T @ jacobian / len(jacobian)
    pullback = columns @ columns.T
    retained = (torch.trace(pullback) / torch.trace(gram)).item()
    safety = settings["damping_safety"] + (
        settings["maximum_safety"] - settings["damping_safety"]
    ) * min(1.0, max(0.0, retained))
    largest = torch.linalg.eigvalsh(gram - settings["implicit_factor"] * pullback)[-1].item()
    bound = min(settings["max_lr"], max(settings["min_lr"], safety / max(largest, 1e-12)))
    proposal = settings["initial_lr"] if previous_lr == 0 else previous_lr * settings["growth"]
    return min(proposal, bound)
