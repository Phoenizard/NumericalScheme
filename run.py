"""Run one experiment using fixed settings from problem/config.py."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from network.nonlinear_regression import TanhRegressor
from optimizer import Inverse_sav, adamw, besav, esav, gd, mesav, naive_esav
from optimizer import pb_besav, pb_esav, pb_mesav, pb_naive_esav, pb_sav, sav, sgd
from optimizer import pb_esav_momentum, pb_sav_momentum
from problem import nonlinear_regression, quadratic
from problem.config import PROBLEMS
from train import train
from tracking import WandbRecorder, load_reference_run


def run_experiment(problem="quadratic", method="pb_sav", scheme=None,
                   m=None, loss=None, linearized=False, relax: bool | None = None,
                   mobility=None, reference_run=None):
    """Assemble one problem and optimizer, then stream its records to W&B."""
    settings = PROBLEMS[problem]
    config = settings.copy()
    root = Path(__file__).resolve().parent
    reference = None
    data_path = None
    initial_state = None
    is_pb_momentum = method in ("pb_sav_momentum", "pb_esav_momentum")
    if reference_run is not None and (not is_pb_momentum or problem != "darcy_deeponet"):
        raise ValueError("--reference-run is only used by Darcy PB momentum comparisons")
    if mobility is not None and not is_pb_momentum:
        raise ValueError("--mobility is only supported by PB-SAV/PB-ESAV momentum")
    mobility = mobility or "identity"
    if mobility not in ("identity", "amsgrad"):
        raise ValueError("Momentum mobility must be identity or amsgrad")
    if is_pb_momentum and method not in settings:
        raise ValueError(f"PB momentum is not configured for {problem}")
    if is_pb_momentum and mobility == "amsgrad" and "amsgrad" not in settings[method]:
        raise ValueError(f"AMSGrad mobility is not configured for {problem}")
    if "num_threads" in settings:
        torch.set_num_threads(settings["num_threads"])
    if problem == "darcy_deeponet":
        if method not in ("sgd", "pb_sav_momentum", "pb_esav_momentum"):
            raise ValueError("Darcy supports SGD, PB-SAV momentum or PB-ESAV momentum")
        if m is not None or loss is not None:
            raise ValueError("Darcy uses its fixed data + L2 objective; omit --m and --loss")
        if settings["device"] != "cpu" or settings["dtype"] != "float64":
            raise ValueError("The current Darcy implementation is configured for CPU float64")
        if settings.get("line_search") is not None:
            raise ValueError("The Darcy protocol requires a fixed learning rate")
    step_fn = {
        "gd": gd.step, "sav": sav.step, "pb_sav": pb_sav.step, "pb_esav": pb_esav.step,
        "pb_naive_esav": pb_naive_esav.step,
        "esav": esav.step, "csav": None,
        "besav": besav.step, "pb_besav": pb_besav.step,
        "mesav": mesav.step, "pb_mesav": pb_mesav.step,
        "naive_esav": naive_esav.step, "sgd": sgd.step, "adamw": adamw.step,
        "inverse_sav": Inverse_sav.step,
        "pb_inverse_sav": partial(Inverse_sav.step, pullback=True),
        "pb_sav_momentum": pb_sav_momentum.step,
        "pb_esav_momentum": pb_esav_momentum.step,
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
        "mesav", "pb_mesav", "pb_sav_momentum",
    ) else None
    if relax is None:
        if is_pb_momentum and mobility == "amsgrad":
            relax = settings[method]["amsgrad"]["relax"]
        else:
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
            config.update(
                auxiliary_state="log_r,log_r_i",
                component_auxiliary="exp(E_i/(4*E_i0))",
                B_definition=(
                    "sum_i (r_i/exp(E_i/(4*E_i0)))^2 "
                    "* grad(E_i) grad(E_i).T / (4*E_i0)"
                ),
            )
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
            source="optimizer/Inverse_sav.py",
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
        momentum = settings.get("sgd", {}).get("momentum", 0.0)
        step_fn = partial(step_fn, momentum=momentum)
        config["optimizer_settings"] = {
            "momentum": momentum, "weight_decay": 0.0,
            "dampening": 0.0, "nesterov": False,
        }
    elif is_pb_momentum:
        momentum = settings[method]["momentum"]
        alpha = settings[method]["alpha"]
        options = {"momentum": momentum, "alpha": alpha, "mobility": mobility}
        if mobility == "amsgrad":
            options.update(beta2=settings[method]["amsgrad"]["beta2"],
                           eps=settings[method]["amsgrad"]["eps"])
        config["optimizer_settings"] = options
        if method == "pb_sav_momentum":
            step_fn = partial(step_fn, total_shift=total_shift, **options)
            config.update(
                total_auxiliary="sqrt(E+C)",
                auxiliary_state="q,p,v,v_max,M" if mobility == "amsgrad" else "q,p",
                modified_energy="q^2 + p.T*M*p/(2*lr)",
                B_definition="(1-alpha)*g*g.T/(2*(E+C)) + alpha*sum_i gi*gi.T/(2*(Ei+Ci))",
            )
        else:
            step_fn = partial(step_fn, **options)
            config.update(
                total_auxiliary="exp(E/(4*E0))",
                component_auxiliary="exp(E_i/(4*E_i0))",
                auxiliary_state="log_r,log_r_i,p,M,learning_rate" + (
                    ",v,v_max,iteration" if mobility == "amsgrad" else ",iteration"),
                reference_policy="frozen_initial_energy",
                modified_energy="4*E0*log_r + p.T*M*p/(2*lr)",
                B_definition=("alpha*sum_i (r_i/exp(E_i/(4*E_i0)))^2 "
                              "* grad(E_i) grad(E_i).T / (4*E_i0)"),
                momentum_definition="p_next=inv(M_next)*(theta_next-theta)",
                momentum_weight="beta",
                kinetic_weight_policy="same_step_mobility",
                parameter_linearization="first_order_component_exponential_feedback",
            )
        if relax:
            config.update(relaxation_rho=1.0,
                          relaxation_update=("q_next^2 = min(E_next+C, q^2+K_old-K_next)"
                                             if method == "pb_sav_momentum" else
                                             "4*E0*log_r_next = min(E_next, "
                                             "4*E0*log_r_old+K_old-K_next)"))
        if mobility == "amsgrad":
            config.update(
                mobility_initialization="v=0,v_max=0,M=eps^(-1)*I",
                mobility_update=("v=beta2*v+(1-beta2)*g^2; "
                                 "v_max=max(v_max,v/(1-beta2^t)); M=1/(sqrt(v_max)+eps)"),
            )
    lr = settings["lr"]
    line_search = settings.get("line_search")
    if method == "adamw":
        lr, line_search = settings["reference_baseline_lr"], None
    elif method == "sgd" and "lr" in settings.get("sgd", {}):
        lr, line_search = settings["sgd"]["lr"], None
    elif is_pb_momentum:
        lr = (settings[method]["amsgrad"]["lr"] if mobility == "amsgrad"
              else settings[method].get("lr", lr))
        line_search = None
    name = (method.removesuffix("_momentum") if is_pb_momentum else method)
    if method in ("sav", "esav", "sgd", "adamw", "pb_sav", "pb_esav") or is_pb_momentum:
        name += f"_{'nonlinear' if problem == 'nonlinear_regression' else problem}"
    name += f"_{scheme}" if scheme is not None else ""
    if linearized:
        name += "_linearized"

    metrics_fn = None
    if problem == "darcy_deeponet":
        from network.darcy_deeponet import DeepONet
        from problem import darcy_deeponet

        data_path = root / settings["data_path"]
        if is_pb_momentum:
            reference_run = reference_run or settings[method]["reference_run"]
            reference = load_reference_run(reference_run)
            shared = ("grid_size", "n_train", "n_test", "correlation_length",
                      "permeability_values", "data_seed", "init_seed", "conv_channels",
                      "conv_kernels", "conv_strides", "conv_paddings", "pool_size",
                      "latent_dim", "trunk_hidden_layers", "lambda_reg", "steps", "lr",
                      "num_threads")
            if (reference["method"] != "sgd" or reference.get("status") != "completed"
                    or any(reference[key] != settings[key] for key in shared)
                    or reference["optimizer_settings"]["momentum"] != momentum
                    or reference["device"] != settings["device"]
                    or reference["dtype"].removeprefix("torch.") != settings["dtype"]):
                raise ValueError("PB momentum comparison must match the completed SGD reference")
        data = darcy_deeponet.load_data({**settings, "data_path": str(data_path)})
        model = DeepONet(settings)
        parameters = tuple(model.parameters())
        loss_fn = partial(
            darcy_deeponet.split_energy_loss, model, data["a_train"], data["coordinates"],
            data["u_train"], settings["lambda_reg"],
        )
        metrics_fn = partial(darcy_deeponet.metrics, model, data, settings["lambda_reg"])
        loss = "data_plus_l2"
        name += f"_momentum{momentum:g}_seed{settings['init_seed']}"
        config.update(
            model="darcy_deeponet", data=str(data_path.resolve()), data_config=data["config"],
            data_versions=data["versions"], batch_size=len(data["a_train"]), components=2,
            dimension=sum(p.numel() for p in parameters), initial_value=None,
            line_search=None, step_policy="fixed", momentum_initialization="zero_displacement",
            coupled_l2_in_loss=settings["lambda_reg"],
            test_metric="mean_per_field_relative_l2", normalization="none",
            objective_normalization=(
                "sum_squared_residuals/(2*N_train*N_grid) + lambda*||theta||^2/2"
            ),
            grouping="data_loss_and_l2", component_names=["data_loss", "regularization"],
            timing_scope="loop_and_metrics_excluding_synchronous_recording",
            reproduction_status="declared_defaults_not_exact_author_replication",
        )
    elif problem == "nonlinear_regression":
        if loss is not None:
            raise ValueError("--loss is only used by quadratic")
        if m not in (1, 2, 4, 8, 15, 16, 30):
            raise ValueError("Specify --m from 1, 2, 4, 8, 15, 16, 30")
        single_energy_methods = (
            "sav", "csav", "esav", "naive_esav", "inverse_sav", "mesav", "besav",
        )
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
        if method not in ("sav", "esav", "sgd", "adamw"):
            name += f"_m{m}"
        if is_pb_momentum:
            name += f"_momentum{momentum:g}_lr{lr:g}_alpha{alpha:g}"
        elif method in ("sav", "esav", "sgd", "adamw", "pb_sav", "pb_esav"):
            if method == "sgd" and config["optimizer_settings"].get("momentum", 0):
                name += f"_momentum{config['optimizer_settings']['momentum']:g}"
            name += "_armijo" if line_search is not None else f"_lr{lr:g}"
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
            objective_normalization="sum_squared_residuals/(2*N_train) + lambda*||theta||^2/2",
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
        if method in ("sav", "esav", "sgd", "adamw", "pb_sav", "pb_esav"):
            if method == "sgd" and config["optimizer_settings"].get("momentum", 0):
                name += f"_momentum{config['optimizer_settings']['momentum']:g}"
            name += "_armijo" if line_search is not None else f"_lr{lr:g}"
        elif method != "pb_naive_esav" or loss == "total":
            name += f"_{loss}"
        config.update(
            model="none", data=None,
            batch_size=None,
            objective_normalization="weighted_quadratic_sum",
            grouping="configured_quadratic_terms" if loss == "split" else "total_energy",
            group_indices=(
                [list(range(0, x.numel(), 2)), list(range(1, x.numel(), 2))]
                if loss == "split" else [list(range(x.numel()))]
            ),
            components=2 if method not in (
                "sav", "csav", "esav", "naive_esav", "inverse_sav", "mesav", "besav",
            ) and loss == "split" else 1,
            line_search=line_search,
            step_policy="fixed" if line_search is None else "armijo_lr_reset",
        )

    if is_pb_momentum:
        if mobility != "identity":
            name += f"_{mobility}"
        initial_options = {"mobility": mobility,
                           "eps": config["optimizer_settings"].get("eps", 1e-8)}
        with torch.no_grad():
            if method == "pb_sav_momentum":
                initial_state = pb_sav_momentum.initialize(
                    parameters, loss_fn(), total_shift, **initial_options,
                )
                config["component_shift"] = total_shift / config["components"]
            else:
                initial_state = pb_esav_momentum.initialize(
                    parameters, loss_fn(), **initial_options,
                )
                config.update(initial_energy=initial_state["initial_energy"].item(),
                              initial_components=initial_state["initial_components"].tolist(),
                              component_ratio_zero_epsilon=torch.finfo(
                                  initial_state["log_r"].dtype).tiny)
        config["momentum_initialization"] = "p=0; displacement=M*p"
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
        if method == "pb_esav":
            config["component_ratio_zero_epsilon"] = torch.finfo(parameters[0].dtype).tiny
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
        momentum=method == "adamw" or ((method == "sgd" or is_pb_momentum) and momentum > 0),
        lr=lr, torch_version=str(torch.__version__),
    )
    if method == "pb_naive_esav":
        config.update(scheme=scheme, modified_energy="log_r")
        if scheme == "plan1" and relax:
            config["eta"] = 1.0

    config.update(num_threads=torch.get_num_threads(), batch_policy="full_batch")
    if data_path is not None:
        config["data_path"] = str(data_path.resolve())

    recorder = WandbRecorder(config, name)
    history = []
    started = perf_counter()

    def result_summary():
        return {
            "training_seconds": max(0.0, perf_counter() - started - recorder.io_seconds),

        }

    try:
        train(
            parameters, loss_fn, step_fn, steps=settings["steps"], lr=lr,
            total_shift=total_shift, relax=relax, esav_scheme=scheme,
            method=method, line_search=line_search, history=history,
            naive_esav_linearized=linearized, metrics_fn=metrics_fn,
            initial_state=initial_state, on_record=recorder.record,
        )
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        try:
            recorder.finish(status, error=error, extra=result_summary())
        except BaseException as upload_error:
            error.add_note(f"W&B finalization also failed: {upload_error}")
        raise
    else:
        recorder.finish("completed", extra=result_summary())
    print(f"{name}: final objective = {history[-1]['objective']:.8e}")
    if problem == "darcy_deeponet":
        print(f"  Data loss = {history[-1]['data_loss']:.8e}; "
              f"test relative L2 = {history[-1]['test_relative_l2']:.8e}")
    print(f"  W&B: {recorder.run.url}")
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
            "pb_sav_momentum", "pb_esav_momentum",
        ],
        default="pb_sav",
    )
    parser.add_argument("--relax", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable Relax (default: problem configuration)")
    parser.add_argument("--mobility", choices=["identity", "amsgrad"],
                        help="PB momentum mobility; amsgrad uses its fixed problem settings")
    parser.add_argument("--linearized", action="store_true",
                        help="Use a first-order Taylor expansion only in the naiveESAV r update")
    parser.add_argument("--scheme", choices=["plan1", "plan2"],
                        help="PB-naiveESAV scheme (default: plan1)")
    parser.add_argument("--m", type=int, choices=[1, 2, 4, 8, 15, 16, 30],
                        help="Number of nonlinear-regression energy components")
    parser.add_argument("--loss", choices=["split", "total"],
                        help="Quadratic energy splitting (CSAV/ESAV/SGD/AdamW default: total)")
    parser.add_argument("--reference-run",
                        help="Completed SGD reference: entity/project/run_id (PB momentum only)")
    args = parser.parse_args()
    run_experiment(**vars(args))


if __name__ == "__main__":
    main()
