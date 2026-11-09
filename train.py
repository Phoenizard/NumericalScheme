"""Shared iteration loop for directly optimized tensors or model parameters."""

import math

import torch

from learning_rate import curvature_rate
from optimizer import pb_sav_momentum

Q_METHODS = ("sav", "pb_sav")


def _armijo_step(parameters, loss_fn, update_fn, initial_lr, settings, iteration):
    """Recompute each trial at its learning rate, restarting from initial_lr every update."""
    original = [parameter.detach().clone() for parameter in parameters]
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    objective = energy.item()
    gradients = torch.autograd.grad(energy, parameters, allow_unused=True)
    rejected = 0
    nonfinite = 0

    def restore_parameters():
        with torch.no_grad():
            for parameter, value in zip(parameters, original):
                parameter.copy_(value)

    lr = initial_lr
    while True:
        trial_objective = float("nan")
        slope = float("nan")
        bound = float("nan")
        finite = False
        try:
            trial_state = update_fn(lr)
            with torch.no_grad():
                terms = loss_fn()
                trial_energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
                trial_objective = trial_energy.item()
                slope = sum(
                    (gradient * (parameter - value)).sum().item()
                    for gradient, parameter, value in zip(gradients, parameters, original)
                    if gradient is not None
                )
                bound = objective + settings["armijo_c"] * slope
                states = trial_state.values() if isinstance(trial_state, dict) else [trial_state]
                finite = (
                    math.isfinite(trial_objective) and math.isfinite(slope)
                    and all(torch.isfinite(p).all().item() for p in parameters)
                    and all(torch.isfinite(s).all().item() for s in states if s is not None)
                )
        except FloatingPointError:
            # A nonfinite trial is counted and rejected; the accepted state is restored below.
            pass
        except Exception:
            restore_parameters()
            raise

        if finite and slope <= 0 and trial_objective <= bound:
            return trial_state, lr, rejected

        restore_parameters()
        rejected += 1
        nonfinite += int(not finite)
        if lr <= settings["min_lr"]:
            raise FloatingPointError(
                f"Armijo failed at update {iteration}, lr={lr:g}: "
                f"objective={objective:.16g}, trial={trial_objective:.16g}, "
                f"bound={bound:.16g}, slope={slope:.16g}; "
                f"rejected_trials={rejected}, nonfinite_trials={nonfinite}. State restored."
            )
        lr = max(lr * settings["shrink"], settings["min_lr"])


def train(parameters, loss_fn, step_fn, steps=300, lr=10.0,
          total_shift=None, relax=False, *, metrics_fn=None,
          method=None, line_search=None, history=None, initial_state=None,
          on_record=None, controller=None, geometry_fn=None, metrics_every=1):
    """Record initial/accepted states and pass a copy of each row to optional on_record."""
    if controller is not None and (geometry_fn is None or line_search is None):
        raise ValueError("Curvature damping requires geometry and Armijo settings")
    if method == "pb_sav_momentum" and line_search is not None:
        raise ValueError("PB momentum currently requires a fixed learning rate")
    if method in Q_METHODS or method == "pb_sav_momentum":
        total_shift = 1e-10 if total_shift is None else total_shift
    if line_search is not None:
        if not all(math.isfinite(value) for value in (lr, *line_search.values())):
            raise ValueError("Line-search settings must be finite")
        if not 0 < line_search["min_lr"] <= lr:
            raise ValueError("Require 0 < min_lr <= initial learning rate")
        if not 0 < line_search["shrink"] < 1:
            raise ValueError("Require 0 < shrink < 1")
        if not 0 < line_search["armijo_c"] < 1:
            raise ValueError("Require 0 < armijo_c < 1")
    parameters = tuple(parameters)
    history = [] if history is None else history
    q = None
    state = initial_state
    accepted_lr = 0.0
    backtracks = 0
    for iteration in range(steps + 1):
        with torch.no_grad():
            terms = loss_fn()
            terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
            energy = sum(terms)
            row = {"iteration": iteration, "objective": energy.item(),
                   "lr": accepted_lr, "backtracks": backtracks}
            if method == "pb_sav_momentum":
                if state is None:
                    state = pb_sav_momentum.initialize(parameters, terms, total_shift)
                row.update(r=state["q"].item(), r_current=(energy + total_shift).sqrt().item())
            elif method in Q_METHODS:
                current = (energy + total_shift).sqrt()
                if q is None:
                    q = current.detach().clone()
                row.update(r=q.item(), r_current=current.item())
            if len(terms) > 1:
                for i, value in enumerate(terms, start=1):
                    row[f"energy_{i}"] = value.item()
            if metrics_fn is not None and (iteration % metrics_every == 0 or iteration == steps):
                row.update(metrics_fn())
        if not all(math.isfinite(value) for value in row.values()):
            if on_record is not None:
                on_record(dict(row))
            raise FloatingPointError(f"Non-finite result at iteration {iteration}: {row}")
        history.append(row)
        if on_record is not None:
            on_record(dict(row))
        if iteration < steps:
            trial_lr = lr
            if controller is not None:
                trial_lr = curvature_rate(accepted_lr, geometry_fn(state), controller)
            accepted_lr = lr
            if line_search is not None:
                def trial_update(trial_lr):
                    if method == "gd":
                        return step_fn(parameters, loss_fn, trial_lr)
                    if method == "sgd":
                        return step_fn(parameters, loss_fn, trial_lr, state=state)
                    if method in ("adamw", "adam"):
                        return step_fn(parameters, loss_fn, state, trial_lr)
                    return step_fn(
                        parameters, loss_fn, q.clone(), trial_lr, total_shift, relax,
                    )

                accepted_state, accepted_lr, backtracks = _armijo_step(
                    parameters, loss_fn, trial_update, trial_lr, line_search, iteration + 1,
                )
                if method in Q_METHODS:
                    q = accepted_state
                elif method != "gd":
                    state = accepted_state
            elif method == "gd":
                step_fn(parameters, loss_fn, lr)
            elif method == "sgd":
                state = step_fn(parameters, loss_fn, lr, state=state)
            elif method == "pb_sav_momentum":
                state = step_fn(parameters, loss_fn, state, lr, relax=relax)
            elif method in ("adamw", "adam"):
                state = step_fn(parameters, loss_fn, state, lr)
            else:
                q = step_fn(parameters, loss_fn, q, lr, total_shift, relax)
    return history
