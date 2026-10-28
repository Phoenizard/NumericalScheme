"""Online experiment records, with SDK caches and upload staging under TMP."""

import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from uuid import uuid4


from problem.config import WANDB


ROOT = Path(__file__).resolve().parent
def source_paths(config, root=ROOT):
    """Include runtime sources (the runner imports all optimizers), excluding data and weights."""
    paths = [root / name for name in ("run.py", "train.py", "tracking.py", "requirements.txt")]
    for directory in ("optimizer", "problem", "network"):
        paths.extend(sorted((root / directory).glob("*.py")))
    return paths


def build_config(resolved):
    """Record only the selected mathematical parameters and experiment settings."""
    method = resolved["method"]
    problem = resolved["problem"]
    base = method.removeprefix("pb_").removesuffix("_momentum")
    expressions = {
        "sav": "sqrt(E+C)", "esav": "exp(E/(4*E_0))",
        "besav": "exp(E/(E+E_0))", "mesav": "exp(E/(E_0+C))*sqrt(E+C)",
        "naive_esav": "exp(E)", "inverse_sav": "1/(E+C)", "csav": "E+C",
    }
    algorithm = {}
    if base in expressions:
        auxiliary = {"expression": expressions[base]}
        if base in ("sav", "mesav", "inverse_sav", "csav"):
            auxiliary["C"] = resolved["total_shift"]
        if base in ("esav", "besav", "mesav"):
            auxiliary["E_0"] = resolved.get("initial_energy", resolved.get("reference_energy"))
            components = resolved.get("initial_components", resolved.get("reference_components"))
            if method.startswith("pb_") and components is not None:
                auxiliary["E_i0"] = components
        if base == "naive_esav":
            if resolved.get("scheme"):
                auxiliary["scheme"] = resolved["scheme"]
            if resolved.get("linearized"):
                auxiliary["linearized"] = True
        algorithm["auxiliary"] = auxiliary
    if method.startswith("pb_"):
        algorithm["pb"] = {"M": resolved["components"]}
    options = resolved.get("optimizer_settings", {})
    if options.get("momentum", 0):
        momentum = {"coefficient": options["momentum"]}
        if method.endswith("_momentum"):
            momentum["alpha"] = options["alpha"]
            if options["mobility"] == "identity":
                momentum["matrix"] = "I"
            else:
                momentum["adaptive"] = {
                    "method": options["mobility"], "beta2": options["beta2"],
                    "eps": options["eps"],
                }
        algorithm["momentum"] = momentum
    if resolved.get("relax"):
        algorithm["relax"] = {"enabled": True}
        for source, key in (("relaxation_rho", "rho"), ("eta", "eta")):
            if source in resolved:
                algorithm["relax"][key] = resolved[source]
    control = {"lr": resolved["lr"], "strategy": "fixed"}
    if resolved.get("line_search") is not None:
        control.update(strategy="armijo", armijo={
            **resolved["line_search"],
            "reset_each_iteration": resolved["step_policy"] == "armijo_lr_reset",
        })
    experiment = {}
    if problem == "quadratic":
        experiment.update(dimension=resolved["dimension"], initial_value=resolved["initial_value"])
    else:
        data_path = Path(resolved["data_path"])
        if data_path.is_absolute() and data_path.is_relative_to(ROOT):
            data_path = data_path.relative_to(ROOT)
        data = {"data_path": str(data_path), "n_train": resolved["n_train"]}
        network = {"model": resolved["model"]}
        if problem == "nonlinear_regression":
            data["noise"] = {
                "distribution": resolved["noise"], "std": resolved["noise_std"],
                "seed": resolved["noise_seed"],
            }
            network.update(hidden_width=resolved["hidden_width"], initialization={
                "seed": resolved["init_seed"],
                "hidden_range": [resolved["hidden_init_min"], resolved["hidden_init_max"]],
                "output_bound": resolved["output_init_bound"],
            })
        elif problem == "darcy_deeponet":
            for key in ("grid_size", "n_test", "correlation_length", "permeability_values",
                        "field_threshold", "gaussian_mode", "gaussian_truncate",
                        "reference_residual_tol", "data_seed"):
                data[key] = resolved[key]
            for key in ("conv_channels", "conv_kernels", "conv_strides", "conv_paddings",
                        "pool_size", "latent_dim", "trunk_hidden_layers"):
                network[key] = resolved[key]
            network["initialization"] = {"seed": resolved["init_seed"]}
        experiment.update(data=data, network=network)
        experiment["batch_policy"] = resolved["batch_policy"]
        if resolved["batch_policy"] != "full_batch":
            experiment["batch_size"] = resolved["batch_size"]
    for key in ("lambda_reg", "steps", "device", "num_threads"):
        experiment[key] = resolved[key]
    experiment["dtype"] = resolved["dtype"].removeprefix("torch.")
    return {"problem": problem, "method": method, "algorithm": algorithm,
            "control": control, "experiment": experiment}


def read_config(config):
    """Expose saved settings to existing plotting and reference-run consumers."""
    if "control" not in config:
        if "experiment" in config and "algorithm" in config:
            return {**config["experiment"], **config["algorithm"],
                    "method": config["method"], "problem": config["problem"]}
        return dict(config)
    algorithm, control = config.get("algorithm", {}), config["control"]
    result = {**config["experiment"], "method": config["method"], "problem": config["problem"],
              "lr": control["lr"], "relax": algorithm.get("relax", {}).get("enabled", False),
              "components": algorithm.get("pb", {}).get("M", 1)}
    search = control.get("armijo")
    result["line_search"] = ({k: v for k, v in search.items() if k != "reset_each_iteration"}
                             if search else None)
    result["step_policy"] = ("armijo_lr_reset" if search and search["reset_each_iteration"]
                             else "armijo" if search else "fixed")
    auxiliary = algorithm.get("auxiliary", {})
    if "C" in auxiliary:
        result["total_shift"] = auxiliary["C"]
    for key in ("scheme", "linearized"):
        if key in auxiliary:
            result[key] = auxiliary[key]
    if "eta" in algorithm.get("relax", {}):
        result["eta"] = algorithm["relax"]["eta"]
    data = result.pop("data", {})
    noise = data.get("noise")
    result.update({k: v for k, v in data.items() if k != "noise"})
    if noise:
        result.update(noise=noise["distribution"], noise_std=noise["std"], noise_seed=noise["seed"])
    network = result.pop("network", {})
    result.update({k: v for k, v in network.items() if k != "initialization"})
    initialization = network.get("initialization", {})
    if initialization:
        result["init_seed"] = initialization["seed"]
    if "hidden_range" in initialization:
        result["hidden_init_min"], result["hidden_init_max"] = initialization["hidden_range"]
        result["output_init_bound"] = initialization["output_bound"]
    momentum = algorithm.get("momentum", {})
    options = {"momentum": momentum.get("coefficient", 0)}
    if "alpha" in momentum:
        options["alpha"] = momentum["alpha"]
        adaptive = momentum.get("adaptive")
        options["mobility"] = adaptive["method"] if adaptive else "identity"
        if adaptive:
            options.update(beta2=adaptive["beta2"], eps=adaptive["eps"])
    result["optimizer_settings"] = options
    return result


def build_tags(config):
    """Build the independent method, plugin, learning-rate and case tags."""
    method = config["method"]
    if method in ("gd", "sgd", "adamw"):
        tags = ["baseline"]
    else:
        tags = [method.removesuffix("_momentum").replace("_", "-")]
    for plugin in ("momentum", "relax"):
        if plugin in config.get("algorithm", {}):
            tags.append(plugin)
    tags.append("armijo" if config["control"]["strategy"] == "armijo" else "fix_lr")
    tags.append("nonlinear" if config["problem"] == "nonlinear_regression" else config["problem"])
    return tags


def _sdk(root):
    """Configure local SDK scratch paths before importing W&B."""
    scratch = root / "TMP" / "wandb"
    for variable, folder in (
        ("WANDB_DIR", "runs"), ("WANDB_CACHE_DIR", "cache"), ("WANDB_DATA_DIR", "staging"),
        ("WANDB_ARTIFACT_DIR", "artifacts"), ("TMPDIR", "tmp"),
    ):
        path = scratch / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]
    import wandb
    return wandb


def load_reference_run(path, *, root=ROOT, sdk=None):
    """Read an explicitly identified completed W&B run, without downloading files."""
    prefix = f"{WANDB['entity']}/{WANDB['project']}/"
    if (not isinstance(path, str) or not path.startswith(prefix)
            or len(path.split("/")) != 3 or not path.split("/")[-1]):
        raise ValueError(f"Set reference_run to {prefix}<run_id>")
    sdk = sdk or _sdk(root)
    reference = sdk.Api(timeout=WANDB["init_timeout"]).run(path)
    if reference.state != "finished" or reference.summary.get("status") != "completed":
        raise ValueError("The W&B reference run must have completed successfully")
    config = dict(reference.config)
    return {**read_config(config), "status": "completed"}


def _json_value(value):
    """Keep nonfinite failure values explicit in JSON rather than silently dropping them."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class WandbRecorder:
    """Stream accepted records and upload complete run artifacts; never save under results/."""

    def __init__(self, resolved, name, *, root=ROOT, mode="online", sdk=None):
        self.root = Path(root)
        self.sdk = sdk or _sdk(self.root)
        if mode not in ("online", "offline"):
            raise ValueError("Recording mode must be online, or offline for SDK tests")
        self.mode = mode
        run_id = uuid4().hex[:12]
        self.stage = self.root / "TMP" / "wandb" / "uploads" / run_id
        self.stage.mkdir(parents=True, exist_ok=False)
        files = source_paths(resolved, self.root)
        for path in files:
            relative = path.relative_to(self.root)
            target = self.stage / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        self.config = build_config(resolved)
        self.run = self.sdk.init(
            entity=WANDB["entity"], project=WANDB["project"], id=run_id, name=name,
            tags=build_tags(self.config), job_type="experiment", resume="never",
            mode=mode, dir=str(self.stage), config=self.config,
            settings=self.sdk.Settings(
                init_timeout=WANDB["init_timeout"], finish_timeout=WANDB["finish_timeout"],
                finish_timeout_raises=True, console="off", disable_git=True, save_code=False,
                x_disable_meta=True,
            ),
        )
        if self.run is None:
            raise RuntimeError("W&B did not initialize; experiment not started")
        self.io_seconds = 0.0
        self.last = None
        self.invalid_row = None
        self.best_objective = None
        self.count = 0
        self.writer = None
        self.closed = False
        self.stream = None
        try:
            if mode == "online" and (self.run.offline or self.run.disabled):
                raise RuntimeError("Online W&B recording is required")
            self.run.define_metric("iteration")
            self.run.define_metric("*", step_metric="iteration")
            code = self.sdk.Artifact(f"source-{run_id}", type="experiment-source")
            code.add_dir(str(self.stage / "source"))
            self.run.log_artifact(code)
            self.stream = (self.stage / "history.csv").open("w", newline="", encoding="utf-8")
        except BaseException:
            self.run.finish(exit_code=1)
            raise

    def record(self, row):
        """Record iteration zero and each subsequent accepted update exactly once."""
        started = perf_counter()
        try:
            values = dict(row)
            if values.get("iteration") != self.count:
                raise ValueError("W&B iterations must start at zero and increase by one")
            if not all(isinstance(value, (int, float)) and math.isfinite(value)
                       for value in values.values()):
                self.invalid_row = values
                raise FloatingPointError("Nonfinite experiment record")
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(values))
                self.writer.writeheader()
            self.writer.writerow(values)
            self.stream.flush()
            self.last = values
            self.count += 1
            objective = values["objective"]
            self.best_objective = objective if self.best_objective is None else min(
                objective, self.best_objective,
            )
            self.run.log(values, step=values["iteration"])
        finally:
            self.io_seconds += perf_counter() - started

    def finish(self, status, *, error=None, extra=None):
        """Upload complete configuration/history and finish with the actual outcome."""
        if self.closed:
            raise RuntimeError("Experiment recorder already finished")
        self.closed = True
        self.stream.close()
        summary = {
            **(self.last or {}), "status": status, "best_objective": self.best_objective,
            **(extra or {}),
        }
        if error is not None:
            summary.update(error_type=type(error).__name__, error_message=str(error))
        if self.invalid_row is not None:
            summary["invalid_row"] = _json_value(self.invalid_row)
        payloads = {"config.json": self.config, "summary.json": summary}
        if error is not None:
            payloads["failure.json"] = {
                key: value for key, value in summary.items()
                if key in ("error_type", "error_message", "iteration", "invalid_row")
            }
        export_error = None
        try:
            for name, value in payloads.items():
                (self.stage / name).write_text(
                    json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8",
                )
            self.run.summary.update(_json_value(summary))
            records = self.sdk.Artifact(f"records-{self.run.id}", type="experiment-record")
            for name in (*payloads, "history.csv"):
                records.add_file(str(self.stage / name), name=name)
            self.run.log_artifact(records)
        except BaseException as caught:
            export_error = caught
            raise
        finally:
            self.run.finish(exit_code=0 if status == "completed" and export_error is None else 1)
        if self.mode == "online":
            shutil.rmtree(self.stage)
        return summary
