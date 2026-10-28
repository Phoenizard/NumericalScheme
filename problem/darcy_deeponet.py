"""Two-phase Darcy data, conservative finite differences and supervised losses."""

import argparse
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import torch

from problem.config import PROBLEMS


def data_config(settings):
    """Return the fixed data specification used to reject stale datasets."""
    config = {key: settings[key] for key in (
        "grid_size", "n_train", "n_test", "correlation_length", "permeability_values",
        "field_threshold", "gaussian_mode", "gaussian_truncate", "reference_residual_tol",
        "data_seed",
    )}
    config.update(
        source=1.0, boundary="zero_dirichlet", face_average="harmonic",
        grid="uniform_nodes_including_boundary", normalization="none", dtype="float64",
        random_streams="SeedSequence.spawn(2)/PCG64", smoothing="ell/(sqrt(2)*h)",
    )
    return config


def solve_darcy(permeability, source=1.0, residual_tol=1e-10):
    """Solve -div(a grad u)=source on [0,1]^2 with zero boundary; return u, residual.

    a is a positive square nodal array. source is scalar or a matching nodal array.
    Interior values use C ordering and harmonic face coefficients.
    """
    permeability = np.asarray(permeability, dtype=np.float64)
    if (permeability.ndim != 2 or permeability.shape[0] != permeability.shape[1]
            or permeability.shape[0] < 3):
        raise ValueError("Darcy permeability must be a square array with at least 3 nodes")
    if not np.isfinite(permeability).all() or np.any(permeability <= 0):
        raise ValueError("Darcy permeability must be finite and positive")
    if not np.isfinite(residual_tol) or residual_tol <= 0:
        raise ValueError("reference residual tolerance must be finite and positive")
    source = np.broadcast_to(np.asarray(source, dtype=np.float64), permeability.shape)
    if not np.isfinite(source).all():
        raise ValueError("Darcy source must be finite")
    grid_size = len(permeability)
    h = 1 / (grid_size - 1)
    faces_x = 2 / (1 / permeability[:-1, :] + 1 / permeability[1:, :])
    faces_y = 2 / (1 / permeability[:, :-1] + 1 / permeability[:, 1:])
    left, right = faces_y[1:-1, :-1], faces_y[1:-1, 1:]
    lower, upper = faces_x[:-1, 1:-1], faces_x[1:, 1:-1]
    indices = np.arange((grid_size - 2) ** 2).reshape(grid_size - 2, grid_size - 2)
    rows = np.concatenate([
        indices.ravel(), indices[:, :-1].ravel(), indices[:, 1:].ravel(),
        indices[:-1, :].ravel(), indices[1:, :].ravel(),
    ])
    columns = np.concatenate([
        indices.ravel(), indices[:, 1:].ravel(), indices[:, :-1].ravel(),
        indices[1:, :].ravel(), indices[:-1, :].ravel(),
    ])
    coefficients = np.concatenate([
        (left + right + lower + upper).ravel(), -right[:, :-1].ravel(),
        -right[:, :-1].ravel(), -upper[:-1, :].ravel(), -upper[:-1, :].ravel(),
    ]) / h ** 2
    matrix = coo_matrix((coefficients, (rows, columns)),
                        shape=(indices.size, indices.size)).tocsr()
    rhs = source[1:-1, 1:-1].ravel()
    interior = spsolve(matrix, rhs)
    residual = float(np.linalg.norm(matrix @ interior - rhs)
                     / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny))
    if not np.isfinite(interior).all() or not np.isfinite(residual) or residual > residual_tol:
        raise FloatingPointError(f"Darcy reference solve failed: relative residual={residual:g}")
    solution = np.zeros_like(permeability)
    solution[1:-1, 1:-1] = interior.reshape(indices.shape)
    return solution, residual


def generate_data(settings=None, output=None) -> Path:
    """Generate and save one fixed dataset; never overwrite an existing file."""
    settings = PROBLEMS["darcy_deeponet"] if settings is None else settings
    root = Path(__file__).resolve().parents[1]
    path = Path(output) if output is not None else root / settings["data_path"]
    if path.exists():
        raise FileExistsError(f"Darcy data already exists: {path}")
    grid_size = settings["grid_size"]
    if grid_size < 3 or min(settings["n_train"], settings["n_test"]) < 1:
        raise ValueError("Require grid_size >= 3 and nonempty training/test datasets")
    if not np.isfinite(settings["correlation_length"]) or settings["correlation_length"] <= 0:
        raise ValueError("correlation_length must be finite and positive")
    low, high = settings["permeability_values"]
    if not 0 < low < high or not np.isfinite([low, high]).all():
        raise ValueError("Require two finite permeability values with 0 < low < high")
    h = 1 / (grid_size - 1)
    seeds = np.random.SeedSequence(settings["data_seed"]).spawn(2)
    data = {"config": data_config(settings), "reference_residuals": {}, "versions": {
        "numpy": str(np.__version__), "scipy": str(scipy.__version__),
        "torch": str(torch.__version__),
    }}
    for split, seed in zip(("train", "test"), seeds):
        generator = np.random.Generator(np.random.PCG64(seed))
        fields, solutions, residuals = [], [], []
        for sample in range(settings[f"n_{split}"]):
            smooth = gaussian_filter(
                generator.standard_normal((grid_size, grid_size)),
                sigma=settings["correlation_length"] / (np.sqrt(2) * h),
                mode=settings["gaussian_mode"], truncate=settings["gaussian_truncate"],
            )
            field = np.where(smooth >= settings["field_threshold"], high, low)
            try:
                solution, residual = solve_darcy(
                    field, residual_tol=settings["reference_residual_tol"],
                )
            except FloatingPointError as error:
                raise FloatingPointError(f"Darcy {split} sample {sample}: {error}") from error
            fields.append(field)
            solutions.append(solution.ravel())
            residuals.append(residual)
        data[f"a_{split}"] = torch.from_numpy(np.stack(fields)[:, None])
        data[f"u_{split}"] = torch.from_numpy(np.stack(solutions))
        data["reference_residuals"][split] = torch.tensor(residuals, dtype=torch.float64)
    axis = torch.linspace(0, 1, grid_size, dtype=torch.float64)
    data["coordinates"] = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1)
    data["coordinates"] = data["coordinates"].reshape(-1, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        torch.save(data, file)
    return path


def load_data(settings):
    """Load a matching CPU float64 dataset and check shapes, values and boundary."""
    path = Path(__file__).resolve().parents[1] / settings["data_path"]
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["config"] != data_config(settings):
        raise ValueError("Darcy data settings changed; generate a new dataset at a new path")
    grid_size = settings["grid_size"]
    axis = torch.linspace(0, 1, grid_size, dtype=torch.float64)
    coordinates = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    if data["coordinates"].dtype != torch.float64 or not torch.equal(
            data["coordinates"], coordinates):
        raise ValueError("Darcy data has inconsistent nodal coordinates")
    for split in ("train", "test"):
        fields, solutions = data[f"a_{split}"], data[f"u_{split}"]
        count = settings[f"n_{split}"]
        if fields.shape != (count, 1, grid_size, grid_size) or solutions.shape != (
                count, grid_size ** 2):
            raise ValueError(f"Invalid Darcy {split} data shapes")
        low, high = settings["permeability_values"]
        residuals = data["reference_residuals"][split]
        if fields.dtype != torch.float64 or solutions.dtype != torch.float64:
            raise ValueError(f"Darcy {split} data must be float64")
        if (not ((fields == low) | (fields == high)).all() or not torch.isfinite(solutions).all()
                or residuals.shape != (count,) or not torch.isfinite(residuals).all()
                or (residuals < 0).any() or (residuals > settings["reference_residual_tol"]).any()
                or (solutions.norm(dim=1) == 0).any()):
            raise ValueError(f"Invalid Darcy {split} values or reference residuals")
        grid = solutions.reshape(count, grid_size, grid_size)
        if torch.count_nonzero(grid[:, [0, -1], :]) or torch.count_nonzero(grid[:, :, [0, -1]]):
            raise ValueError(f"Darcy {split} reference does not satisfy the zero boundary")
    return data


def split_energy_loss(model, permeability, coordinates, target, lambda_reg):
    """Return data MSE/2 and coupled L2/2, with all trainable biases regularized."""
    data_loss = (model(permeability, coordinates) - target).square().mean() / 2
    regularization = lambda_reg * sum(p.square().sum() for p in model.parameters()) / 2
    return [data_loss, regularization]


@torch.no_grad()
def metrics(model, data, lambda_reg):
    """Report training terms and the mean of per-field test relative L2 errors."""
    data_loss, regularization = split_energy_loss(
        model, data["a_train"], data["coordinates"], data["u_train"], lambda_reg,
    )
    prediction = model(data["a_test"], data["coordinates"])
    relative = (prediction - data["u_test"]).norm(dim=1) / data["u_test"].norm(dim=1)
    return {"data_loss": data_loss.item(), "regularization": regularization.item(),
            "test_relative_l2": relative.mean().item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New dataset path; defaults to problem config")
    args = parser.parse_args()
    print(f"Data: {generate_data(output=args.output).resolve()}")
