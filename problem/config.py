"""Fixed settings for each problem; research variables belong to the CLI."""

PROBLEMS = {
    "quadratic": {
        # 问题与初值
        "dimension": 100,
        "initial_value": 1.0,
        "energy_weights": [1.0, 0.01],
        # 训练：无动量
        "lr": 10.0,
        "steps": 300,
        "lambda_reg": 0.0,
        "relax": False,
        # 算法：total_shift 仅用于 SAV / PB-SAV
        "total_shift": 1e-10,
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
        "data_path": "data/nonlinear_regression.pt",
        # 网络：1 → hidden_width → 1，单隐层 tanh
        "hidden_width": 50,
        "hidden_init_min": -1.0,
        "hidden_init_max": 1.0,
        "init_seed": 2,
        # 输出层初始化范围为 ±1 / sqrt(hidden_width)
        # 训练：全量样本，无动量
        "lr": 0.1,
        "steps": 600,
        "lambda_reg": 0.0,
        "relax": False,
        # 算法：total_shift 仅用于 SAV / PB-SAV，各分量使用 C / M
        "total_shift": 1e-10,
        # 计算
        "device": "cpu",
        "dtype": "float64",
    },
}
