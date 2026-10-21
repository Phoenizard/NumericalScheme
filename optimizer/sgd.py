"""SGD baseline with full-batch gradients, zero momentum and zero weight decay."""

import math

from optimizer.gd import step as gd_step


def step(parameters, loss_fn, lr: float):
    """Update parameters in place using the supplied batch's total objective."""
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    gd_step(parameters, loss_fn, lr)
