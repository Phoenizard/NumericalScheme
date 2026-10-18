"""Run one experiment using fixed settings from problem/config.py."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import torch

from network.nonlinear_regression import TanhRegressor
from optimizer import gd, pb_esav, pb_sav, sav
from problem import nonlinear_regression, quadratic
from problem.config import PROBLEMS
from train import train


def run_experiment(problem="quadratic", method="pb_sav", scheme=None,
                   m=None, loss=None, output=None):
    """Assemble one problem and optimizer, then save configuration and history."""
    settings = PROBLEMS[problem]
    config = settings.copy()
    root = Path(__file__).resolve().parent
    step_fn = {
        "gd": gd.step, "sav": sav.step, "pb_sav": pb_sav.step, "pb_esav": pb_esav.step,
    }[method]
    if method == "gd" and problem != "nonlinear_regression":
        raise ValueError("The GD baseline is currently configured for nonlinear_regression only")
    if scheme is not None and method != "pb_esav":
        raise ValueError("--scheme is only used by PB-ESAV")
    scheme = (scheme or "plan1") if method == "pb_esav" else None
    total_shift = settings["total_shift"] if method in ("sav", "pb_sav") else None
    relax = settings["relax"] if method != "gd" else False
    lr = settings["lr"]
    line_search = None
    metrics_fn = None
    name = method + (f"_{scheme}" if scheme is not None else "")

    if problem == "nonlinear_regression":
        line_search = settings["line_search"]
        if method == "sav" or loss is not None:
            raise ValueError("Use GD/PB-SAV/PB-ESAV with --m; --loss is only for quadratic")
        if m not in (1, 2, 4, 8, 16, 30):
            raise ValueError("Specify --m from 1, 2, 4, 8, 16, 30")
        data_path = root / settings["data_path"]
        data = torch.load(data_path, map_location=settings["device"], weights_only=True)
        dtype = getattr(torch, settings["dtype"])
        x, y = data["x_train"].to(dtype), data["y_train"].to(dtype)
        order = x[:, 0].argsort()
        x, y = x[order], y[order]
        x_test, y_test = data["x_test"].to(dtype), data["y_test"].to(dtype)
        groups = nonlinear_regression.make_groups(len(x), m, settings["partition_seed"])
        model = TanhRegressor(settings)
        parameters = tuple(model.parameters())
        loss_fn = partial(
            nonlinear_regression.split_energy_loss, model, x, y, groups,
            lambda_reg=settings["lambda_reg"],
        )
        metrics_fn = lambda: {
            "test_mse": nonlinear_regression.test_mse(model, x_test, y_test),
        }
        loss = "split"
        name += f"_m{m}"
        # Record the loaded data's settings rather than a later edit to the data configuration.
        config.update(data["config"])
        config.update(
            model="tanh", data=str(data_path), data_config=data["config"],
            n_train=len(x), n_test=len(x_test), batch_size=len(x), components=m,
            dimension=sum(p.numel() for p in parameters), initial_value=None,
            output_init_bound=settings["hidden_width"] ** -0.5,
            line_search=line_search,
            step_policy="fixed" if line_search is None else "armijo_lr_reset",
            grouping="nested_random_bisection", group_indices=[group.tolist() for group in groups],
            reproduction_status="declared_defaults_not_exact_author_replication",
        )
    else:
        if m is not None:
            raise ValueError("--m is only used by nonlinear_regression")
        loss = loss or "split"
        x = quadratic.init_parameters(settings)
        parameters = (x,)
        energy_fn = {
            "split": quadratic.split_energy_loss,
            "total": quadratic.total_energy_loss,
        }[loss]
        loss_fn = partial(energy_fn, x, settings)
        if method != "pb_esav" or loss == "total":
            name += f"_{loss}"
        config.update(
            model="none", data=None,
            components=2 if method != "sav" and loss == "split" else 1,
        )

    if relax:
        name += "_relaxed"
    config.update(
        problem=problem, method=method, loss=loss, total_shift=total_shift, relax=relax,
        device=str(parameters[0].device), dtype=str(parameters[0].dtype),
        momentum=False, lr=lr, torch_version=str(torch.__version__),
    )
    if method == "pb_esav":
        config.update(scheme=scheme, modified_energy="log_r")
        if scheme == "plan1" and settings["relax"]:
            config["eta"] = 1.0

    output = Path(output) if output is not None else root / "results" / problem / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8",
    )
    history = []
    try:
        train(
            parameters, loss_fn, step_fn, steps=settings["steps"], lr=lr,
            total_shift=total_shift, relax=relax, esav_scheme=scheme,
            metrics_fn=metrics_fn, method=method, line_search=line_search, history=history,
        )
    except FloatingPointError as error:
        (output / "failure.txt").write_text(str(error) + "\n", encoding="utf-8")
        raise
    finally:
        if history:
            with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
    print(f"{name}: final objective = {history[-1]['objective']:.8e}")
    if metrics_fn is not None:
        print(f"  Test MSE: {history[-1]['test_mse']:.8e}")
    print(f"  Results: {output.resolve()}")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=list(PROBLEMS), default="quadratic")
    parser.add_argument("--method", choices=["gd", "sav", "pb_sav", "pb_esav"],
                        default="pb_sav")
    parser.add_argument("--scheme", choices=["plan1", "plan2"],
                        help="PB-ESAV scheme (default: plan1)")
    parser.add_argument("--m", type=int, choices=[1, 2, 4, 8, 16, 30],
                        help="Number of nonlinear-regression energy components")
    parser.add_argument("--loss", choices=["split", "total"],
                        help="Quadratic energy splitting (default: split)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_experiment(**vars(args))


if __name__ == "__main__":
    main()
