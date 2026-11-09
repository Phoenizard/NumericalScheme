"""Fixed settings for each problem; research variables belong to the CLI."""

WANDB = {
    "enabled": False,  # Default off; enable per run with --wandb.
    "entity": "pheonizard-university-of-nottingham",
    "project": "NumericalScheme",
    "init_timeout": 30,
    "finish_timeout": 60,
}

PROBLEMS = {
    "darcy_deeponet": {
        # 论文已披露参数与本地默认细节；尚无对应实验报告。
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
        "total_shift": 1e-12,  # 论文未给出；按约定取总 shift 1e-12
        "pb_sav_momentum": {
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
        # SAV 系列的总 shift
        "total_shift": 1e-10,
        "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        "reference_baseline_lr": 0.1,  # Fixed-rate SGD/AdamW reference.
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
        # SAV 系列的总 shift
        "total_shift": 1e-10,
        "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        "reference_baseline_lr": 0.1,
        # 动量对照采用固定学习率；单位缩放、Relax 关闭。
        "sgd": {"lr": 0.1, "momentum": 0.9},
        "pb_sav_momentum": {"lr": 0.1, "momentum": 0.9, "alpha": 0.5},
        # 计算
        "device": "cpu",
        "dtype": "float64",
        "num_threads": 4,
    },
}

# 二次型固定学习率平台：固定 0.9、3000 步、无线搜索；辅助变量方法开 Relax。
PROBLEMS["quadratic_fixed_lr"] = {
    **PROBLEMS["quadratic"],
    "lr": 0.9,
    "steps": 3000,
    "relax": True,
    "line_search": None,
    "reference_baseline_lr": 0.1,  # Fixed-rate AdamW / Adam reference.
    "adam": {"betas": (0.9, 0.999), "eps": 1e-8},
}

# 正向 Burgers PINN：PB_SAV_Momentum.pdf §4.4。种子、数据生成与 Sobol 细节为已声明的默认值。
# 首次生成参考解：python -B -m problem.burgers_pinn
PROBLEMS["burgers_pinn"] = {
    "nu": 0.01, "x_min": -1.0, "x_max": 1.0, "t_max": 1.0,
    "grid_nx": 256, "grid_nt": 101,
    "reference_refine": 8, "reference_rtol": 1e-10, "reference_atol": 1e-12,
    "data_path": "data/burgers_pinn.pt", "data_seed": 42,
    "n_collocation": 10000, "n_initial": 256, "n_boundary": 256, "n_validation": 10000,
    "hidden_layers": 6, "width": 64,
    "init_seed": 42, "seeds": [42, 2273, 2669, 5439, 8041],  # 同一种子控制初值与 Sobol 置乱
    "lr": 1e-3, "steps": 10000, "lambda_reg": 1e-6,
    "total_shift": 1e-12,  # 1/4 给权重衰减，其余按样本数分配
    "relax": False, "line_search": None, "diagnostic_every": 50,
    "default_mobility": "amsgrad",
    "sgd": {"lr": 1e-2, "momentum": 0.9},
    "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-6},
    "adam": {"betas": (0.9, 0.999), "eps": 1e-8}, "reference_baseline_lr": 1e-3,
    "pb_sav_momentum": {"momentum": 0.9, "alpha": 0.5,
                        "amsgrad": {"lr": 1e-3, "relax": True, "beta2": 0.999, "eps": 1e-8}},
    "device": "cpu", "dtype": "float64", "num_threads": 4,
}

# 反向 Burgers PINN：§4.5。未知粘性 nu = exp(xi)，初值 0.02；100 个无噪内部观测。
PROBLEMS["burgers_inverse_pinn"] = {
    **PROBLEMS["burgers_pinn"],
    "nu_init": 0.02, "n_observations": 100, "lambda_reg": 3e-7,
    "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 3e-7},
    "pb_sav_momentum": {"momentum": 0.9, "alpha": 0.5,
                        "amsgrad": {"lr": 1e-2, "lr_one_component": 3e-3, "relax": True,
                                    "beta2": 0.999, "eps": 1e-8}},
}

# Darcy v2 与报告共用三组独立样本、训练集 RMS 和 seed 0 初值。
# 首次生成数据：python -B -m problem.darcy_deeponet --problem darcy_deeponet_v2
PROBLEMS["darcy_deeponet_v2"] = {
    **PROBLEMS["darcy_deeponet"],
    "data_path": "data/darcy_deeponet_v2.pt", "data_seed": 20261106, "init_seed": 0,
    "n_validation": 32, "validation_every": 100,
    "default_mobility": "amsgrad",
    "pb_sav_momentum": {"momentum": 0.9, "alpha": 0.5,
                        "amsgrad": {"lr": 0.01, "relax": True, "beta2": 0.999, "eps": 1e-8}},
    "sgd": {"lr": 0.03, "momentum": 0.9},
    "adam": {"betas": (0.9, 0.999), "eps": 1e-8}, "reference_baseline_lr": 0.003,
}
