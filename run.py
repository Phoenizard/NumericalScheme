"""Run one experiment using fixed settings from problem/config.py."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import torch

from network.nonlinear_regression import TanhRegressor
from optimizer import Inverse_sav, adamw, besav, esav, gd, mesav, naive_esav
from optimizer import pb_besav, pb_esav, pb_mesav, pb_naive_esav, pb_sav, sav, sgd
from problem import nonlinear_regression, quadratic
from problem.config import PROBLEMS
from train import train


def run_experiment(problem="quadratic", method="pb_sav", scheme=None,
                   m=None, loss=None, output=None, linearized=False, relax: bool | None = None):
    """Assemble one problem and optimizer, then save configuration and history."""
    settings = PROBLEMS[problem]
    config = settings.copy()
    root = Path(__file__).resolve().parent
    step_fn = {
        "gd": gd.step, "sav": sav.step, "pb_sav": pb_sav.step, "pb_esav": pb_esav.step,
        "pb_naive_esav": pb_naive_esav.step,
        "esav": esav.step, "csav": None,
        "besav": besav.step, "pb_besav": pb_besav.step,
        "mesav": mesav.step, "pb_mesav": pb_mesav.step,
        "naive_esav": naive_esav.step, "sgd": sgd.step, "adamw": adamw.step,
        "inverse_sav": Inverse_sav.step,
        "pb_inverse_sav": partial(Inverse_sav.step, pullback=True),
    }[method]
    if method == "csav":
        from optimizer.csav import step as step_fn
    is_esav = method in ("esav", "pb_esav")
    is_besav = method in ("besav", "pb_besav")
    is_mesav = method in ("mesav", "pb_mesav")
    if linearized and method != "naive_esav":
        raise ValueError("--linearized is only supported by the naiveESAV")
    if method == "gd" and problem != "nonlinear_regression":
        raise ValueError("The GD baseline is currently configured for nonlinear_regression only")
    if scheme is not None and method != "pb_naive_esav":
        raise ValueError("--scheme is only used by PB-naiveESAV")
    scheme = (scheme or "plan1") if method == "pb_naive_esav" else None
    total_shift = settings["total_shift"] if method in (
        "sav", "csav", "pb_sav", "inverse_sav", "pb_inverse_sav",
        "mesav", "pb_mesav",
    ) else None
    if relax is None:
        relax = settings["relax"] if method not in ("gd", "sgd", "adamw") else False
    elif relax and method in ("gd", "sgd", "adamw"):
        raise ValueError(f"{method} does not use Relax")
    if is_besav:
        config.update(
            total_auxiliary="exp(E/(E+E0))", auxiliary_state="log_r",
            modified_energy="log_r", reference_policy="frozen_initial_energy",
        )
        if method == "pb_besav":
            config["B_definition"] = "sum_i E_i0/(E_i+E_i0)^2 * grad(E_i) grad(E_i).T"
        if relax:
            config["relaxation_update"] = "log_r_next = min(E_next/(E_next+E0), log_r_old)"
    if is_esav:
        config.update(
            total_auxiliary="exp(E/(4*E0))", auxiliary_state="log_r",
            modified_energy="4*E0*log_r", reference_policy="frozen_initial_energy",
        )
        if method == "pb_esav":
            config["B_definition"] = "sum_i grad(E_i) grad(E_i).T / (4*E_i0)"
        if relax:
            config["relaxation_update"] = "log_r_next = min(E_next/(4*E0), log_r_old)"
    if is_mesav:
        step_fn = partial(step_fn, total_shift=total_shift)
        config.update(
            auxiliary_variant="modified_esav", auxiliary_state="log_r", modified_energy="log_r",
            total_auxiliary="exp(E/(E_initial+C))*sqrt(E+C)",
            reference_policy="frozen_initial_energy",
        )
        if relax:
            config["relaxation_update"] = (
                "log_r_next = min(E_next/(E_initial+C) + log(E_next+C)/2, log_r_old)"
            )
    if method == "naive_esav" and linearized and relax:
        raise ValueError("Full Relax requires naiveESAV (linearized=False)")
    if method in ("inverse_sav", "pb_inverse_sav"):
        if relax:
            raise ValueError("Inverse SAV does not use Relax")
        config.update(
            total_auxiliary="1/(E+C)", auxiliary_state="r", modified_energy="1/r",
            source="optimizer/InverseSAV.py",
            pullback="centered_component_covariance" if method == "pb_inverse_sav" else None,
        )
    elif method == "csav":
        if relax:
            raise ValueError("CSAV does not use post-step Relax")
        config.update(
            auxiliary_variant="recovery_bounded_feedback", total_auxiliary="E + C",
            auxiliary_state="r", modified_energy="r", source="optimizer/csav.py",
        )
    elif method == "pb_naive_esav":
        config["esav_variant"] = "naive"
    elif method == "naive_esav":
        step_fn = partial(step_fn, linearized=linearized)
        config.update(
            esav_variant="naive", linearized=linearized,
            total_auxiliary="exp(E)",
            auxiliary_state="r" if linearized else "log_r",
            source="https://arxiv.org/abs/1912.09263", source_equation="2.7",
        )
        if relax:
            config["relaxation_update"] = "log_r_next = min(E_next, log_r_old)"
    elif method == "adamw":
        step_fn = partial(step_fn, **settings["adamw"])
        config["optimizer_settings"] = settings["adamw"].copy()
    elif method == "sgd":
        config["optimizer_settings"] = {"momentum": 0.0, "weight_decay": 0.0}
    lr = settings["lr"]
    line_search = settings.get("line_search")
    if method == "adamw":
        lr, line_search = settings["reference_baseline_lr"], None
    name = method + (f"_{scheme}" if scheme is not None else "")
    if linearized:
        name += "_linearized"

    if problem == "nonlinear_regression":
        if method == "sav" or loss is not None:
            raise ValueError(
                "Use GD/PB-SAV/PB-ESAV/PB-naiveESAV with --m; --loss is only for quadratic"
            )
        if m not in (1, 2, 4, 8, 15, 16, 30):
            raise ValueError("Specify --m from 1, 2, 4, 8, 15, 16, 30")
        single_energy_methods = ("csav", "esav", "naive_esav", "inverse_sav", "mesav", "besav")
        if method in single_energy_methods and m != 1:
            raise ValueError("Single-energy methods require --m 1")
        data_path = root / settings["data_path"]
        data = torch.load(data_path, map_location=settings["device"], weights_only=True)
        dtype = getattr(torch, settings["dtype"])
        x, y = data["x_train"].to(dtype), data["y_train"].to(dtype)
        order = x[:, 0].argsort()
        x, y = x[order], y[order]
        groups = nonlinear_regression.make_groups(len(x), m, settings["partition_seed"])
        model = TanhRegressor(settings)
        parameters = tuple(model.parameters())
        loss_fn = partial(
            nonlinear_regression.split_energy_loss, model, x, y, groups,
            lambda_reg=settings["lambda_reg"],
        )
        loss = "split"
        name += f"_m{m}"
        # Record the loaded data's settings rather than a later edit to the data configuration.
        config.update(data["config"])
        config.update(
            model="tanh", data=str(data_path), data_config=data["config"],
            n_train=len(x), batch_size=len(x), components=m,
            dimension=sum(p.numel() for p in parameters), initial_value=None,
            output_init_bound=settings["hidden_width"] ** -0.5,
            line_search=line_search,
            step_policy="fixed" if line_search is None else "armijo_lr_reset",
            grouping="nested_random_bisection", group_indices=[group.tolist() for group in groups],
            reproduction_status="declared_defaults_not_exact_author_replication",
        )
        config.pop("n_test", None)
    else:
        if m is not None:
            raise ValueError("--m is only used by nonlinear_regression")
        loss = loss or (
            "total" if method in (
                "csav", "esav", "naive_esav", "inverse_sav", "sgd", "adamw", "mesav", "besav",
            ) else "split"
        )
        x = quadratic.init_parameters(settings)
        parameters = (x,)
        energy_fn = {
            "split": quadratic.split_energy_loss,
            "total": quadratic.total_energy_loss,
        }[loss]
        loss_fn = partial(energy_fn, x, settings)
        if method != "pb_naive_esav" or loss == "total":
            name += f"_{loss}"
        config.update(
            model="none", data=None,
            components=2 if method not in (
                "sav", "csav", "esav", "naive_esav", "inverse_sav", "mesav", "besav",
            ) and loss == "split" else 1,
            line_search=line_search,
            step_policy="fixed" if line_search is None else "armijo_lr_reset",
        )

    if is_besav:
        initializer = pb_besav if method == "pb_besav" else besav
        with torch.no_grad():
            initial_state = initializer.initialize(loss_fn())
        config.update(
            initial_energy=initial_state["initial_energy"].item(),
            initial_components=initial_state["initial_components"].tolist(),
        )
    if is_esav:
        initializer = pb_esav if method == "pb_esav" else esav
        with torch.no_grad():
            initial_state = initializer.initialize(loss_fn())
        config.update(
            initial_energy=initial_state["initial_energy"].item(),
            initial_components=initial_state["initial_components"].tolist()
            if "initial_components" in initial_state else None,
        )
    if is_mesav:
        initializer = pb_mesav if method == "pb_mesav" else mesav
        with torch.no_grad():
            initial_state = initializer.initialize(loss_fn(), total_shift)
        config.update(
            reference_energy=initial_state["reference_energy"].item(),
            reference_components=initial_state["reference_components"].tolist()
            if "reference_components" in initial_state else None,
        )
    if relax:
        name += "_relaxed"
    config.update(
        problem=problem, method=method, loss=loss, total_shift=total_shift, relax=relax,
        device=str(parameters[0].device), dtype=str(parameters[0].dtype),
        momentum=method == "adamw", lr=lr, torch_version=str(torch.__version__),
    )
    if method == "pb_naive_esav":
        config.update(scheme=scheme, modified_energy="log_r")
        if scheme == "plan1" and relax:
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
            method=method, line_search=line_search, history=history,
            naive_esav_linearized=linearized,
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
    print(f"  Results: {output.resolve()}")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=list(PROBLEMS), default="quadratic")
    parser.add_argument(
        "--method",
        choices=[
            "gd", "sgd", "adamw", "sav", "csav", "esav", "naive_esav", "pb_sav",
            "pb_esav", "pb_naive_esav", "besav", "pb_besav",
            "inverse_sav", "pb_inverse_sav", "mesav", "pb_mesav",
        ],
        default="pb_sav",
    )
    parser.add_argument("--relax", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable Relax (default: problem configuration)")
    parser.add_argument("--linearized", action="store_true",
                        help="Use a first-order Taylor expansion only in the naiveESAV r update")
    parser.add_argument("--scheme", choices=["plan1", "plan2"],
                        help="PB-naiveESAV scheme (default: plan1)")
    parser.add_argument("--m", type=int, choices=[1, 2, 4, 8, 15, 16, 30],
                        help="Number of nonlinear-regression energy components")
    parser.add_argument("--loss", choices=["split", "total"],
                        help="Quadratic energy splitting (CSAV/ESAV/SGD/AdamW default: total)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_experiment(**vars(args))


if __name__ == "__main__":
    main()
