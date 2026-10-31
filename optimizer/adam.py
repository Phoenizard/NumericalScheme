"""Adam without weight decay, using the shared bias-corrected moment update."""

from optimizer.adamw import step as moment_step


def step(parameters, loss_fn, state, lr, *, betas, eps):
    """Update parameters in place and return new detached Adam moments."""
    return moment_step(parameters, loss_fn, state, lr, betas=betas, eps=eps, weight_decay=0.0)
