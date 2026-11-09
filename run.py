"""Run one experiment using fixed settings from problem/config.py."""

import argparse
import hashlib
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from network.nonlinear_regression import TanhRegressor
from optimizer import adam, adamw, gd, pb_sav, pb_sav_momentum, sav, sgd
from problem import nonlinear_regression, quadratic
from problem.config import PROBLEMS
from train import train
from tracking import make_recorder

METHODS = ("gd", "sgd", "adamw", "adam", "sav", "pb_sav", "pb_sav_momentum")
BASELINES = ("gd", "sgd", "adamw", "adam")
BURGERS = ("burgers_pinn", "burgers_inverse_pinn")


def run_experiment(problem="quadratic", method="pb_sav", m=None, loss=None,
                   relax: bool | None = None, mobility=None,
                   wandb: bool | None = None, seed=None):
    """Assemble one problem and optimizer; records go to W&B only when enabled."""
    settings = PROBLEMS[problem].copy()
    is_burgers = problem in BURGERS
    if seed is not None:
        if not is_burgers or seed not in settings["seeds"]:
            raise ValueError("--seed is only used by the Burgers problems, from their seeds")
        settings["init_seed"] = seed
    config = settings.copy()
    root = Path(__file__).resolve().parent
    data_path = None
    initial_state = None
    is_darcy = problem == "darcy_deeponet"
    is_pb_momentum = method == "pb_sav_momentum"
    if mobility is not None and not is_pb_momentum:
        raise ValueError("--mobility is only supported by PB momentum methods")
    mobility = mobility or (settings.get("default_mobility", "identity") if is_pb_momentum else "identity")
    if mobility not in ("identity", "amsgrad"):
        raise ValueError("Momentum mobility must be identity or amsgrad")
    if is_pb_momentum and method not in settings:
        raise ValueError(f"PB momentum is not configured for {problem}")
    if is_pb_momentum and mobility == "amsgrad" and "amsgrad" not in settings[method]:
        raise ValueError(f"AMSGrad mobility is not configured for {problem}")
    if "num_threads" in settings:
        torch.set_num_threads(settings["num_threads"])
    if is_darcy:
        if method not in ("pb_sav_momentum", "sgd", "adamw", "adam"):
            raise ValueError(f"{method} is not configured for {problem}")
        if m is not None or loss is not None:
            raise ValueError("Darcy uses its fixed data + L2 objective; omit --m and --loss")
        if settings["device"] != "cpu" or settings["dtype"] != "float64":
            raise ValueError("The current Darcy implementation is configured for CPU float64")
        if settings.get("line_search") is not None:
            raise ValueError("The Darcy protocol requires a fixed learning rate")
    if is_burgers:
        if method not in ("sgd", "adamw", "adam", "pb_sav_momentum"):
            raise ValueError(f"{method} is not configured for {problem}")
        if loss is not None:
            raise ValueError("Burgers uses its fixed PINN objective; omit --loss")
        if is_pb_momentum and m not in (1, 4):
            raise ValueError("Burgers PB-SAV momentum requires --m 1 (aggregate) or 4")
        if not is_pb_momentum and m is not None:
            raise ValueError("--m is only used by PB-SAV momentum on Burgers")
        if settings["device"] != "cpu" or settings["dtype"] != "float64":
            raise ValueError("The Burgers PINN is configured for CPU float64")
    step_fn = {
        "adam": adam.step, "gd": gd.step, "sav": sav.step, "pb_sav": pb_sav.step,
        "sgd": sgd.step, "adamw": adamw.step, "pb_sav_momentum": pb_sav_momentum.step,
    }[method]
    if method == "gd" and problem != "nonlinear_regression":
        raise ValueError("The GD baseline is currently configured for nonlinear_regression only")
    total_shift = settings["total_shift"] if method in ("sav", "pb_sav", "pb_sav_momentum") else None
    if relax is None:
        if is_pb_momentum and mobility == "amsgrad":
            relax = settings[method]["amsgrad"]["relax"]
        else:
            relax = settings["relax"] if method not in BASELINES else False
    elif relax and method in BASELINES:
        raise ValueError(f"{method} does not use Relax")
    if method == "adam":
        if "adam" not in settings:
            raise ValueError(f"Adam is not configured for {problem}")
        step_fn = partial(step_fn, **settings["adam"])
        config["optimizer_settings"] = settings["adam"].copy()
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
        step_fn = partial(step_fn, total_shift=total_shift, **options)
        config.update(
            total_auxiliary="sqrt(E+C)",
            auxiliary_state="q,p,v,v_max,M" if mobility == "amsgrad" else "q,p",
            modified_energy="q^2 + p.T*M*p/(2*lr)",
            B_definition="(1-alpha)*g*g.T/(2*(E+C)) + alpha*sum_i gi*gi.T/(2*(Ei+Ci))",
        )
        if relax:
            config.update(relaxation_rho=1.0,
                          relaxation_update="q_next^2 = min(E_next+C, q^2+K_old-K_next)")
        if mobility == "amsgrad":
            config.update(
                mobility_initialization="v=0,v_max=0,M=eps^(-1)*I",
                mobility_update=("v=beta2*v+(1-beta2)*g^2; "
                                 "v_max=max(v_max,v/(1-beta2^t)); M=1/(sqrt(v_max)+eps)"),
            )
    lr = settings["lr"]
    line_search = settings.get("line_search")
    if method in ("adamw", "adam"):
        lr, line_search = settings["reference_baseline_lr"], None
    elif method == "sgd" and "lr" in settings.get("sgd", {}):
        lr, line_search = settings["sgd"]["lr"], None
    elif is_pb_momentum:
        lr = (settings[method]["amsgrad"]["lr"] if mobility == "amsgrad"
              else settings[method].get("lr", lr))
        line_search = None
    if is_burgers and is_pb_momentum and m == 1 and mobility == "amsgrad":
        lr = settings[method]["amsgrad"].get("lr_one_component", lr)
    shift_weights = None
    controller = None
    geometry_fn = None
    if problem == "nonlinear_regression":
        if method in ("sav", "pb_sav"):
            controller = settings["controller"]
            lr = controller["initial_lr"]
        elif method == "gd":
            lr, line_search = settings["gd_lr"], None
        elif method == "adam":
            lr, line_search = settings["adam_lr"], None
    name = (method.removesuffix("_momentum") if is_pb_momentum else method)
    name += f"_{'nonlinear' if problem == 'nonlinear_regression' else problem}"

    metrics_fn = None
    if is_darcy:
        from network.darcy_deeponet import DeepONet
        from problem import darcy_deeponet

        data_path = root / settings["data_path"]
        data = darcy_deeponet.load_data({**settings, "data_path": str(data_path)})
        model = DeepONet(settings)
        parameters = tuple(model.parameters())
        loss_fn = partial(
            darcy_deeponet.split_energy_loss, model, data["a_train"], data["coordinates"],
            data["u_train"], settings["lambda_reg"],
        )
        coupled = method != "adamw"  # AdamW applies weight decay in the step instead
        if not coupled:
            full_loss = loss_fn
            loss_fn = lambda: full_loss()[:1]
        loss = "data_plus_l2"
        if method in ("sgd", "pb_sav_momentum"):
            name += f"_momentum{momentum:g}"
        name += f"_seed{settings['init_seed']}"
        config.update(
            model="darcy_deeponet", data=str(data_path.resolve()), data_config=data["config"],
            data_versions=data["versions"], batch_size=len(data["a_train"]),
            components=2 if coupled else 1,
            dimension=sum(p.numel() for p in parameters), initial_value=None,
            line_search=None, step_policy="fixed", momentum_initialization="zero_displacement",
            coupled_l2_in_loss=settings["lambda_reg"] if coupled else 0.0,
            test_metric="mean_per_field_relative_l2", normalization="none",
            objective_normalization=(
                "sum_squared_residuals/(2*N_train*N_grid) + lambda*||theta||^2/2"
            ),
            grouping="data_loss_and_l2", component_names=["data_loss", "regularization"],
            timing_scope="loop_and_metrics_excluding_synchronous_recording",
            reproduction_status="declared_defaults_not_exact_author_replication",
        )
        metrics_fn = partial(darcy_deeponet.metrics, model, data, settings["lambda_reg"])
    elif is_burgers:
        from network.burgers_pinn import BurgersPINN
        from problem import burgers_pinn

        inverse = problem == "burgers_inverse_pinn"
        data_path = root / settings["data_path"]
        data = burgers_pinn.load_data(settings)
        model = BurgersPINN(settings, inverse=inverse)
        parameters = tuple(model.parameters())
        points = burgers_pinn.sample_points(settings, settings["init_seed"], data, inverse)
        coupled = method != "adamw"  # AdamW applies weight decay in the step instead
        components = m if is_pb_momentum else 1
        loss_fn = partial(
            burgers_pinn.split_energy_loss, model, points, settings["lambda_reg"],
            inverse=inverse, components=components, coupled=coupled,
        )
        if is_pb_momentum:
            shift_weights = burgers_pinn.shift_weights(settings, inverse, components)
            step_fn = partial(step_fn, shift_weights=shift_weights)
        metrics_fn = burgers_pinn.Metrics(
            model, data, points, settings, inverse=inverse, coupled=coupled,
            every=settings["diagnostic_every"], steps=settings["steps"],
        )
        loss = "burgers_pinn_energy"
        if is_pb_momentum:
            name += f"_m{components}"
        if method in ("sgd", "pb_sav_momentum"):
            name += f"_momentum{momentum:g}"
        name += f"_seed{settings['init_seed']}"
        config.update(
            model="tanh_pinn", data=str(data_path.resolve()), n_train=settings["n_collocation"],
            data_sha256=data["sha256"], reference_refinement_error=
            data["refinement_relative_error"].item(),
            batch_size=settings["n_collocation"], components=components,
            component_names=burgers_pinn.component_names(inverse, components, coupled),
            dimension=sum(p.numel() for p in parameters), initial_value=None,
            line_search=None, step_policy="fixed", coupled_l2_in_loss=coupled,
            grouping="four_components" if components == 4 else "aggregate",
            objective_normalization="Eq. (76)-(79): MSE-based energies; E_wd = lambda*||theta||^2/2",
            objective_note="task objective = objective - coupled_regularization",
            reproduction_status="declared_defaults_not_exact_author_replication",
        )
    elif problem == "nonlinear_regression":
        if loss is not None:
            raise ValueError("--loss is only used by quadratic")
        if m not in (1, 2, 4, 8, 15, 16, 30):
            raise ValueError("Specify --m from 1, 2, 4, 8, 15, 16, 30")
        if method == "sav" and m != 1:
            raise ValueError("Single-energy methods require --m 1")
        data_path = root / settings["data_path"]
        data = torch.load(data_path, map_location=settings["device"], weights_only=True)
        dtype = getattr(torch, settings["dtype"])
        x, y = data["x_train"].to(dtype), data["y_train"].to(dtype)
        order = x[:, 0].argsort()
        x, y = x[order], y[order]
        groups = nonlinear_regression.make_groups(
            len(x), m, settings["partition_seed"], settings.get("partition_rng", "torch"),
        )
        model = TanhRegressor(settings)
        parameters = tuple(model.parameters())
        loss_fn = partial(
            nonlinear_regression.split_energy_loss, model, x, y, groups,
            lambda_reg=settings["lambda_reg"],
        )
        if controller is not None:
            if settings["lambda_reg"] != 0:
                raise ValueError("The current curvature geometry requires lambda_reg=0")
            geometry_fn = partial(nonlinear_regression.controller_geometry,
                                  model, x, y, groups, method, total_shift)
        metrics_fn = partial(nonlinear_regression.test_mse, model,
                             data["x_test"].to(dtype), data["y_test"].to(dtype))
        test_metric = metrics_fn
        metrics_fn = lambda: {"test_mse": test_metric()}
        config["input_sha256"] = hashlib.sha256(torch.cat((
            x.flatten(), y.flatten(), *[p.detach().flatten() for p in parameters],
        )).numpy().tobytes()).hexdigest()
        loss = "split"
        if method not in ("sav", "sgd", "gd", "adam", "adamw"):
            name += f"_m{m}"
        if is_pb_momentum:
            name += f"_momentum{momentum:g}_lr{lr:g}_alpha{alpha:g}"
        else:
            if method == "sgd" and config["optimizer_settings"].get("momentum", 0):
                name += f"_momentum{config['optimizer_settings']['momentum']:g}"
            name += "_curvature_damping" if controller is not None else f"_lr{lr:g}"
        # Record the loaded data's settings rather than a later edit to the data configuration.
        config.update(data["config"])
        config.update(
            model="tanh", data=str(data_path), data_config=data["config"],
            n_train=len(x), batch_size=len(x), components=m,
            dimension=sum(p.numel() for p in parameters), initial_value=None,
            output_init_bound=settings["hidden_width"] ** -0.5,
            line_search=line_search,
            step_policy="fixed" if controller is None else "curvature_damping",
            grouping="nested_random_bisection", group_indices=[group.tolist() for group in groups],
            objective_normalization="sum_squared_residuals/(2*N_train) + lambda*||theta||^2/2",
            reproduction_status="declared_defaults_not_exact_author_replication",
        )
        config.pop("n_test", None)
    else:
        if m is not None:
            raise ValueError("--m is only used by nonlinear_regression")
        loss = loss or ("total" if method in ("sgd", "adamw", "adam") else "split")
        x = quadratic.init_parameters(settings)
        parameters = (x,)
        energy_fn = {
            "split": quadratic.split_energy_loss,
            "total": quadratic.total_energy_loss,
        }[loss]
        loss_fn = partial(energy_fn, x, settings)
        if method in ("sav", "sgd", "adamw", "adam", "pb_sav"):
            if method == "sgd" and config["optimizer_settings"].get("momentum", 0):
                name += f"_momentum{config['optimizer_settings']['momentum']:g}"
            name += "_curvature_damping" if controller is not None else f"_lr{lr:g}"
        config.update(
            model="none", data=None,
            batch_size=None,
            objective_normalization="weighted_quadratic_sum",
            grouping="configured_quadratic_terms" if loss == "split" else "total_energy",
            group_indices=(
                [list(range(0, x.numel(), 2)), list(range(1, x.numel(), 2))]
                if loss == "split" else [list(range(x.numel()))]
            ),
            components=2 if method != "sav" and loss == "split" else 1,
            line_search=line_search,
            step_policy="fixed" if controller is None else "curvature_damping",
        )

    if controller is not None:
        config.update(controller=controller, step_policy="curvature_damping")
    if is_pb_momentum:
        if mobility != "identity":
            name += f"_{mobility}"
        initial_options = {"mobility": mobility,
                           "eps": config["optimizer_settings"].get("eps", 1e-8)}
        with torch.no_grad():
            initial_state = pb_sav_momentum.initialize(
                parameters, loss_fn(), total_shift, shift_weights=shift_weights, **initial_options,
            )
        config["component_shift"] = (
            [total_shift * weight for weight in shift_weights] if shift_weights is not None
            else total_shift / config["components"])
        config["momentum_initialization"] = "p=0; displacement=M*p"
    if relax:
        name += "_relaxed"
    config.update(
        problem=problem, method=method, loss=loss, total_shift=total_shift, relax=relax,
        device=str(parameters[0].device), dtype=str(parameters[0].dtype),
        momentum=method in ("adam", "adamw") or ((method == "sgd" or is_pb_momentum) and momentum > 0),
        lr=lr, torch_version=str(torch.__version__),
    )

    config.update(num_threads=torch.get_num_threads(), batch_policy="full_batch")
    if data_path is not None:
        config["data_path"] = str(data_path.resolve())

    recorder = make_recorder(config, name, wandb)
    history = []
    started = perf_counter()

    def result_summary():
        return {
            "training_seconds": max(0.0, perf_counter() - started - recorder.io_seconds),

        }

    try:
        train(
            parameters, loss_fn, step_fn, steps=settings["steps"], lr=lr,
            total_shift=total_shift, relax=relax,
            method=method, line_search=line_search, history=history,
            metrics_fn=metrics_fn,
            metrics_every=1,
            initial_state=initial_state, on_record=recorder.record,
            controller=controller, geometry_fn=geometry_fn,
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
    elif is_burgers:
        task = history[-1]["objective"] - history[-1]["coupled_regularization"]
        print(f"  Task objective = {task:.8e}; relative L2 = {history[-1]['relative_l2']:.8e}")
    if recorder.run is not None:
        print(f"  W&B: {recorder.run.url}")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=list(PROBLEMS), default="quadratic")
    parser.add_argument("--method", choices=METHODS, default="pb_sav")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                        help="Record to W&B (default: WANDB['enabled'] in problem/config.py, off)")
    parser.add_argument("--relax", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable Relax (default: problem configuration)")
    parser.add_argument("--mobility", choices=["identity", "amsgrad"],
                        help="PB momentum mobility; amsgrad uses its fixed problem settings")
    parser.add_argument("--m", type=int, choices=[1, 2, 4, 8, 15, 16, 30],
                        help="Energy components: nonlinear regression, or 1/4 for Burgers PB-SAV")
    parser.add_argument("--seed", type=int,
                        help="Burgers paired seed (default: first seed in the problem config)")
    parser.add_argument("--loss", choices=["split", "total"],
                        help="Quadratic energy splitting (SGD/AdamW default: total)")
    args = parser.parse_args()
    run_experiment(**vars(args))


if __name__ == "__main__":
    main()
