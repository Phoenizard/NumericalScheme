"""Viscous Burgers PINN (forward and inverse): reference solution, sampling and energies.

u_t + u u_x - nu u_xx = 0 on [-1, 1] x [0, 1], u(x, 0) = -sin(pi x), u(+-1, t) = 0.
Energies follow PB_SAV_Momentum.pdf Sec. 4.4 (Eq. 76-77) and Sec. 4.5 (Eq. 78-79).
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.stats import qmc
import torch

from problem.config import PROBLEMS

DATA_KEYS = ("nu", "x_min", "x_max", "t_max", "grid_nx", "grid_nt", "reference_refine",
             "reference_rtol", "reference_atol")


def data_config(settings):
    """Return the fixed reference specification used to reject stale datasets."""
    config = {key: settings[key] for key in DATA_KEYS}
    config.update(
        initial_condition="-sin(pi*x)", boundary="zero_dirichlet", dtype="float64",
        reference="method_of_lines_central_differences_BDF", grid="uniform_nodes_including_boundary",
    )
    return config


def _reference(nu, nx, nt, t_max, refine, rtol, atol):
    """Solve the semi-discrete Burgers system on (nx-1)*refine+1 nodes; return (nt, nx)."""
    nodes = (nx - 1) * refine + 1
    x = np.linspace(-1.0, 1.0, nodes)
    h = x[1] - x[0]

    def rhs(_, u):
        padded = np.concatenate(([0.0], u, [0.0]))
        ux = (padded[2:] - padded[:-2]) / (2 * h)
        uxx = (padded[2:] - 2 * u + padded[:-2]) / h ** 2
        return -u * ux + nu * uxx

    pattern = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(nodes - 2, nodes - 2), format="csr")
    solution = solve_ivp(
        rhs, (0.0, t_max), -np.sin(np.pi * x[1:-1]), method="BDF",
        t_eval=np.linspace(0.0, t_max, nt), rtol=rtol, atol=atol, jac_sparsity=pattern,
    )
    if not solution.success or solution.y.shape[1] != nt:
        raise FloatingPointError(f"Burgers reference solve failed: {solution.message}")
    full = np.zeros((nt, nodes))
    full[:, 1:-1] = solution.y.T
    return full[:, ::refine]


def generate_data(settings=None, output=None) -> Path:
    """Generate the reference solution and save it once; never overwrite an existing file."""
    settings = PROBLEMS["burgers_pinn"] if settings is None else settings
    root = Path(__file__).resolve().parents[1]
    path = Path(output) if output is not None else root / settings["data_path"]
    if path.exists():
        raise FileExistsError(f"Burgers data already exists: {path}")
    args = (settings["nu"], settings["grid_nx"], settings["grid_nt"], settings["t_max"])
    tolerances = (settings["reference_rtol"], settings["reference_atol"])
    refine = settings["reference_refine"]
    coarse = _reference(*args, refine, *tolerances)
    fine = _reference(*args, 2 * refine, *tolerances)
    error = np.linalg.norm(coarse - fine) / np.linalg.norm(fine)
    data = {
        "config": data_config(settings),
        "x": torch.linspace(settings["x_min"], settings["x_max"], settings["grid_nx"],
                            dtype=torch.float64),
        "t": torch.linspace(0.0, settings["t_max"], settings["grid_nt"], dtype=torch.float64),
        "u": torch.from_numpy(fine),
        "refinement_relative_error": torch.tensor(error, dtype=torch.float64),
        "versions": {"numpy": str(np.__version__), "scipy": str(scipy.__version__),
                     "torch": str(torch.__version__)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        torch.save(data, file)
    return path


def load_data(settings):
    """Load the reference solution and check its configuration, shape and boundary values."""
    path = Path(__file__).resolve().parents[1] / settings["data_path"]
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["config"] != data_config(settings):
        raise ValueError("Burgers data settings changed; generate a new dataset at a new path")
    nx, nt = settings["grid_nx"], settings["grid_nt"]
    u = data["u"]
    if u.shape != (nt, nx) or u.dtype != torch.float64 or not torch.isfinite(u).all():
        raise ValueError("Invalid Burgers reference shape or values")
    if torch.count_nonzero(u[:, [0, -1]]):
        raise ValueError("Burgers reference does not satisfy the zero boundary")
    if not torch.allclose(u[0], -torch.sin(torch.pi * data["x"]), atol=1e-14):
        raise ValueError("Burgers reference does not satisfy the initial condition")
    data["sha256"] = hashlib.sha256(
        json.dumps(data["config"], sort_keys=True).encode() + u.numpy().tobytes()).hexdigest()
    return data


def _sobol(dimension, count, rng):
    """Scrambled Sobol points in [0, 1)^d; unbalanced counts only trigger a SciPy warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qmc.Sobol(d=dimension, scramble=True, seed=rng).random(count)


def sample_points(settings, seed, data=None, inverse=False):
    """Draw the fixed training/validation points of one run and, if inverse, the observations.

    Four independent streams spawned from seed: collocation, validation, initial, boundary.
    Observations are 100 distinct interior reference nodes drawn with settings["data_seed"],
    so they are shared by all seeds and methods.
    """
    dtype = getattr(torch, settings["dtype"])
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    lo, hi = settings["x_min"], settings["x_max"]

    def space_time(count, rng):
        points = _sobol(2, count, rng)
        return (torch.tensor(lo + (hi - lo) * points[:, 0], dtype=dtype),
                torch.tensor(settings["t_max"] * points[:, 1], dtype=dtype))

    points = {}
    points["x_res"], points["t_res"] = space_time(settings["n_collocation"], streams[0])
    points["x_val"], points["t_val"] = space_time(settings["n_validation"], streams[1])
    points["x_ic"] = torch.tensor(
        lo + (hi - lo) * _sobol(1, settings["n_initial"], streams[2])[:, 0], dtype=dtype)
    points["t_bc"] = torch.tensor(
        settings["t_max"] * _sobol(1, settings["n_boundary"], streams[3])[:, 0], dtype=dtype)
    if inverse:
        nx, nt = settings["grid_nx"], settings["grid_nt"]
        interior = np.array([(j, i) for j in range(1, nt) for i in range(1, nx - 1)])
        chosen = np.random.default_rng(settings["data_seed"]).choice(
            len(interior), settings["n_observations"], replace=False)
        j, i = interior[np.sort(chosen)].T
        points["x_obs"] = data["x"][i].to(dtype)
        points["t_obs"] = data["t"][j].to(dtype)
        points["y_obs"] = data["u"][j, i].to(dtype)
    return points


def residual(model, x, t):
    """r_f = u_t + u u_x - nu u_xx at (x, t), differentiable in the model parameters."""
    with torch.enable_grad():
        x = x.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)
        u = model(x, t)
        u_x, u_t = torch.autograd.grad(u.sum(), (x, t), create_graph=True)
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        return u_t + u * u_x - model.viscosity * u_xx


def weight_decay(model, lambda_reg):
    """E_wd = lambda/2 ||theta||^2 over the network parameters (never xi)."""
    return lambda_reg * sum(p.square().sum() for p in model.network_parameters()) / 2


def physical_terms(model, points, inverse=False):
    """E_res, E_bc, E_ic (and E_data in the inverse problem) as in Eq. (76a-c) and (78)."""
    with torch.enable_grad():
        zero = torch.zeros_like(points["x_ic"])
        edge = torch.full_like(points["t_bc"], 1.0)
        terms = {
            "res": residual(model, points["x_res"], points["t_res"]).square().mean() / 2,
            "bc": (model(-edge, points["t_bc"]).square().mean()
                   + model(edge, points["t_bc"]).square().mean()) / 4,
            "ic": (model(points["x_ic"], zero) + torch.sin(torch.pi * points["x_ic"]))
                  .square().mean() / 2,
        }
        if inverse:
            terms["data"] = (model(points["x_obs"], points["t_obs"])
                             - points["y_obs"]).square().mean() / 2
        return terms


def component_names(inverse, components, coupled):
    """Names of the energy components: 1 aggregate or the four of Sec. 4.4 / 4.5."""
    if components == 1:
        return ["total"]
    names = ["residual", "boundary_initial", "data"] if inverse else [
        "residual", "boundary", "initial"]
    return names + (["regularization"] if coupled else [])


def split_energy_loss(model, points, lambda_reg, *, inverse=False, components=4, coupled=True):
    """Return the energy components used by the optimizers.

    components=1 sums everything into one term; components=4 keeps (res, bc, ic, wd) in the
    forward problem and (res, bc+ic, data, wd) in the inverse problem. coupled=False drops
    E_wd (for AdamW, whose weight decay is applied in the step).
    """
    terms = physical_terms(model, points, inverse)
    if inverse:
        parts = [terms["res"], terms["bc"] + terms["ic"], terms["data"]]
    else:
        parts = [terms["res"], terms["bc"], terms["ic"]]
    if coupled:
        parts.append(weight_decay(model, lambda_reg))
    if components == 1:
        return [sum(parts)]
    if components != 4:
        raise ValueError("Burgers PINN supports 1 aggregate or 4 components")
    return parts


def shift_weights(settings, inverse, components, coupled=True):
    """Paper rule: a quarter of the total shift to E_wd, the rest by sample counts."""
    if components == 1:
        return None
    counts = ([settings["n_collocation"], 2 * settings["n_boundary"] + settings["n_initial"],
               settings["n_observations"]] if inverse else
              [settings["n_collocation"], 2 * settings["n_boundary"], settings["n_initial"]])
    if not coupled:
        raise ValueError("PB-SAV shifts require the coupled regularization component")
    return [0.75 * n / sum(counts) for n in counts] + [0.25]


class Metrics:
    """Per-iteration diagnostics; the expensive ones run every `every` calls and at the end.

    Always: coupled_regularization (the part of the objective that is E_wd), so that
    task objective = objective - coupled_regularization. Every `every` calls: relative L2
    error on the reference grid, the residual energy on the validation points and, in the
    inverse problem, the viscosity and its relative error.
    """

    def __init__(self, model, data, points, settings, *, inverse, coupled, every, steps):
        self.model, self.data, self.points, self.settings = model, data, points, settings
        self.inverse, self.coupled, self.every, self.steps = inverse, coupled, every, steps
        x, t = torch.meshgrid(data["x"], data["t"], indexing="ij")
        self.grid = (x.T.reshape(-1).to(next(model.parameters()).dtype),
                     t.T.reshape(-1).to(next(model.parameters()).dtype))
        self.calls = 0

    def __call__(self):
        index, self.calls = self.calls, self.calls + 1
        regularization = weight_decay(self.model, self.settings["lambda_reg"]).item()
        row = {"coupled_regularization": regularization if self.coupled else 0.0}
        if index % self.every == 0 or index == self.steps:
            with torch.no_grad():
                prediction = self.model(*self.grid).reshape(self.data["u"].shape)
            reference = self.data["u"]
            row["relative_l2"] = ((prediction - reference).norm() / reference.norm()).item()
            row["validation_residual_energy"] = residual(
                self.model, self.points["x_val"], self.points["t_val"]
            ).detach().square().mean().item() / 2
            if self.inverse:
                nu = self.model.viscosity.item()
                row["viscosity"] = nu
                row["viscosity_relative_error"] = abs(nu - self.settings["nu"]) / self.settings["nu"]
        return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New dataset path; defaults to problem config")
    args = parser.parse_args()
    settings = PROBLEMS["burgers_pinn"]
    torch.set_num_threads(settings["num_threads"])
    print(f"Data: {generate_data(settings, args.output).resolve()}")
