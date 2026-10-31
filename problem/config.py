"""Fixed settings for each problem; research variables belong to the CLI."""

WANDB = {
    "entity": "pheonizard-university-of-nottingham",
    "project": "NumericalScheme",
    "init_timeout": 30,
    "finish_timeout": 60,
}

PROBLEMS = {
    "darcy_deeponet": {
        # 论文已披露参数及本地默认细节见 doc/experiments/darcy_deeponet.md。
        "grid_size": 33,
        "n_train": 128,
        "n_test": 64,
        "correlation_length": 0.12,
        "permeability_values": [3.0, 12.0],
        "field_threshold": 0.0,
        "gaussian_mode": "reflect",
        "gaussian_truncate": 4.0,
        "reference_residual_tol": 1e-10,
        "data_seed": 42,
        "data_path": "data/darcy_deeponet.pt",
        "conv_channels": [8, 16, 32],
        "conv_kernels": [5, 3, 3],
        "conv_strides": [1, 2, 2],
        "conv_paddings": [2, 1, 1],
        "pool_size": 4,
        "latent_dim": 64,
        "trunk_hidden_layers": 3,
        "init_seed": 42,
        "lr": 1e-2,
        "steps": 5000,
        "lambda_reg": 3e-5,
        "sgd": {"momentum": 0.9},
        "total_shift": 0.0,
        "pb_sav_momentum": {
            "momentum": 0.9,
            "alpha": 0.5,
            "reference_run": None,  # Set entity/project/run_id here or pass --reference-run.
            "amsgrad": {"lr": 3e-3, "relax": True, "beta2": 0.999, "eps": 1e-8},
        },
        "pb_esav_momentum": {
            "momentum": 0.9,
            "alpha": 0.5,
            "reference_run": None,  # Set entity/project/run_id here or pass --reference-run.
            "amsgrad": {"lr": 3e-3, "relax": True, "beta2": 0.999, "eps": 1e-8},
        },
        "relax": False,
        "line_search": None,
        "device": "cpu",
        "dtype": "float64",
        "num_threads": 4,
    },
    "quadratic": {
        # 问题与初值
        "dimension": 100,
        "initial_value": 1.0,
        "energy_weights": [1.0, 0.01],
        # 训练：无动量
        "lr": 100.0,
        "steps": 300,
        "lambda_reg": 0.0,
        "relax": False,
        # 与非线性回归相同的 Armijo 配置；每轮重新从 lr 开始
        "line_search": {
            "min_lr": 1e-6,
            "shrink": 0.5,
            "armijo_c": 1e-4,
        },
        # SAV 系列的总 shift；ESAV / PB-ESAV 使用冻结初始能量，不使用 shift
        "total_shift": 1e-10,
        "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        "reference_baseline_lr": 0.1,  # Fixed-rate SGD/AdamW reference in inverse-SAV research.
        # 计算
        "device": "cpu",
        "dtype": "float64",
        "num_threads": 4,
    },
    "nonlinear_regression": {
        # 数据：等距采样，y = sin(2πx) + Gaussian noise
        "n_train": 30,
        "n_test": 1001,
        "x_min": 0.0,
        "x_max": 1.0,
        "noise_std": 0.1,
        "noise_seed": 42,
        "partition_seed": 42,
        "data_path": "data/nonlinear_regression_r21.pt",
        "data_rng": "numpy",
        "partition_rng": "numpy",
        "initialization": "centered_normal",
        "hidden_scale": 0.72,
        "legacy_momentum": {
            "noise_seed": 1, "partition_seed": 3, "init_seed": 2,
            "data_path": "data/nonlinear_regression.pt", "data_rng": "torch",
            "partition_rng": "torch", "initialization": "uniform",
        },
        "controller": {
            "initial_lr": 1.0, "growth": 1.1, "min_lr": 1e-6, "max_lr": 100.0,
            "damping_safety": 0.5, "maximum_safety": 1.0, "implicit_factor": 2.0,
        },
        "gd_lr": 0.22,
        "adam_lr": 0.03,
        "adam": {"betas": (0.9, 0.999), "eps": 1e-8},
        # 网络：1 → hidden_width → 1，单隐层 tanh
        "hidden_width": 50,
        "hidden_init_min": -1.0,
        "hidden_init_max": 1.0,
        "init_seed": 42,
        # 当前输出层为 N(0, 1/hidden_width)；uniform 范围只供旧动量设置使用。
        # 训练：全量样本；600 次更新
        "lr": 100.0,  # 未迁移方法保留旧 Armijo 起点；新策略使用 controller.initial_lr。
        "steps": 600,
        "lambda_reg": 0.0,
        "relax": False,
        # 共同 Armijo 接受条件；新策略的试探起点由 controller 决定。
        "line_search": {
            "min_lr": 1e-6,
            "shrink": 0.5,
            "armijo_c": 1e-4,
        },
        # SAV 系列的总 shift；ESAV / PB-ESAV 使用冻结初始能量，不使用 shift
        "total_shift": 1e-10,
        "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        "reference_baseline_lr": 0.1,
        # 动量对照采用固定学习率；单位缩放、Relax 关闭。
        "sgd": {"lr": 0.1, "momentum": 0.9},
        "pb_sav_momentum": {"lr": 0.1, "momentum": 0.9, "alpha": 0.5},
        "pb_esav_momentum": {"lr": 0.1, "momentum": 0.9, "alpha": 0.5},
        # 计算
        "device": "cpu",
        "dtype": "float64",
        "num_threads": 4,
    },
}
