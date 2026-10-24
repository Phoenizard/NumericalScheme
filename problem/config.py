"""Fixed settings for each problem; research variables belong to the CLI."""

PROBLEMS = {
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
    },
    "nonlinear_regression": {
        # 数据：等距采样，y = sin(2πx) + Gaussian noise
        "n_train": 30,
        "n_test": 1001,
        "x_min": 0.0,
        "x_max": 1.0,
        "noise_std": 0.1,
        "noise_seed": 1,
        "partition_seed": 3,
        "data_path": "data/nonlinear_regression.pt",
        # 网络：1 → hidden_width → 1，单隐层 tanh
        "hidden_width": 50,
        "hidden_init_min": -1.0,
        "hidden_init_max": 1.0,
        "init_seed": 2,
        # 输出层初始化范围为 ±1 / sqrt(hidden_width)
        # 训练：全量样本，无动量；600 次接受的更新
        "lr": 100.0,  # 每轮 Armijo 回溯的初始学习率，也是学习率上限
        "steps": 600,
        "lambda_reg": 0.0,
        "relax": False,
        # 每轮从 lr 开始，拒绝后学习率减半并重新计算候选更新
        "line_search": {
            "min_lr": 1e-6,
            "shrink": 0.5,
            "armijo_c": 1e-4,
        },
        # SAV 系列的总 shift；ESAV / PB-ESAV 使用冻结初始能量，不使用 shift
        "total_shift": 1e-10,
        "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        "reference_baseline_lr": 0.1,
        # 计算
        "device": "cpu",
        "dtype": "float64",
    },
}
