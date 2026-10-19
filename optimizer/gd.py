"""Full-batch gradient descent on the sum of component energies."""

import torch


def step(parameters, loss_fn, lr: float):
    """Update parameters in place using one gradient of the total objective."""
    parameters = tuple(parameters)
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    gradients = torch.autograd.grad(energy, parameters)
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            parameter.add_(gradient, alpha=-lr)
