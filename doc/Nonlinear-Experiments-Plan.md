# Nonlinear Experiments Plan

## 目标函数

$$
\boxed{
\min_{\theta\in\mathbb{R}^{151}} F(\theta),
\qquad
F(\theta)=\frac{1}{2N}\sum_{i=1}^{N}
\bigl(f_\theta(x_i)-y_i\bigr)^2
}
$$

$$
\begin{aligned}
N&=30,\qquad x_i\in[0,1],\\
y_i&=\sin(2\pi x_i)+\varepsilon_i,
\qquad \varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,\sigma^2).
\end{aligned}
$$

## 神经网络

```mermaid
flowchart LR
    X["x ∈ ℝ"] --> L1["Linear 1 → 50"]
    L1 --> H["tanh · 50"]
    H --> L2["Linear 50 → 1"]
    L2 --> Y["fθ(x) ∈ ℝ"]
```

$$
\begin{aligned}
f_\theta(x)&=\sum_{j=1}^{50}c_j\tanh(a_jx+b_j)+b_0,\\
\theta&=(a,b,c,b_0),\qquad \dim\theta=50+50+50+1=151.
\end{aligned}
$$

## 对照目标

$$
\text{论文 PB-SAV}\;\longrightarrow\;\text{本地 PB-SAV}\;\longrightarrow\;\text{PB-ESAV}.
$$

| 论文基准 | $M$ | 接受的更新次数 | 最终训练目标 |
| --- | --- | --- | --- |
| [PB SAV Momentum §4.2, Table 2](PB_SAV_Momentum.pdf#page=21) | 30 | 600 | $2.525\times10^{-3}$ |

PB-ESAV 是本项目扩展。未取得作者代码；以下明确区分论文设置与用户批准的本地默认值。

## 固定超参数

| 配置入口 | 设置 |
| --- | --- |
| [problem/config.py](../problem/config.py) | `PROBLEMS["nonlinear_regression"]` |

| 参数 | 设置 | 依据 |
| --- | --- | --- |
| 样本数 / 区间 / 噪声 | $N=30$；$[0,1]$；$\sigma=0.1$ | 论文 §4.2 |
| 网络 | $1\to50\to1$；tanh；151 个参数 | 论文 §4.2 |
| Batch / 精度 | 全量；`float64` | 论文 §4 |
| 参数更新次数 | $K=600$；自适应方法只统计接受的更新 | 论文 §4.2 |
| Momentum | Off；direct update，$\alpha=1$ | 论文 §4.2 |
| 正则化 | $\lambda=0$ | 论文目标函数 |
| 训练点 | $x_i=(i-1)/29$ | 本地默认 |
| 测试点 | $x_\ell=\ell/1000$；$\ell=0,\ldots,1000$ | 本地默认 |
| 测试目标 | $\sin(2\pi x)$，无噪声 | 论文 §4.2 |
| 隐藏层初始化 | $a_j,b_j\sim\mathcal U(-1,1)$ | 本地默认 |
| 输出层初始化 | $c_j,b_0\sim\mathcal U(-1/\sqrt{50},1/\sqrt{50})$ | 本地默认 |
| 数据 / 初始化 / 分组种子 | 1 / 2 / 3 | 本地默认 |
| SAV / PB-SAV shift | $C=10^{-10}$；$C_k=C/M$ | 本地默认；论文 §4.2 未给数值 |
| PB-ESAV / GD shift | 无 | 本项目定义 |
| Relax | Off | 沿用项目设置；论文 §4.2 未明确该开关 |
| 设备 | CPU | 本地设置；论文为 A100 GPU |

## 自适应学习率与 Armijo

$$
\Delta\theta_k(\eta)\;\text{由对应优化器按学习率 }\eta\text{ 计算},
\qquad g_k=\nabla F(\theta_k).
$$

$$
\boxed{
 g_k^\mathsf T\Delta\theta_k(\eta)\leq0,\qquad
 F(\theta_k+\Delta\theta_k(\eta))
 \leq F(\theta_k)+c\,g_k^\mathsf T\Delta\theta_k(\eta)
}
$$

| 设置 | GD / PB-SAV / PB-ESAV | 依据 |
| --- | --- | --- |
| 学习率范围 | $10^{-6}\leq\eta\leq100$ | 四种方法统一设置 |
| 每轮初始学习率 | $\eta\leftarrow100$；与上一轮接受值无关 | 当前回溯规则 |
| Armijo 系数 | $c=10^{-4}$ | 用户批准的常规设置 |
| 拒绝后重试 | $\eta\leftarrow\max(0.5\eta,10^{-6})$；重新计算 $\Delta\theta_k(\eta)$ | 当前回溯规则 |
| 接受后 | $\theta_{k+1}=\theta_k+\Delta\theta_k(\eta)$；提交对应辅助变量 | 原优化器完整更新 |
| 拒绝的试探步 | 恢复 $\theta_k$；保留原辅助变量；不计入 $K$ | 接受／拒绝实现 |
| 下限 | $\eta=10^{-6}$ 仍被拒绝则报错，保留已接受记录 | 实现约定 |
| GD 更新 | $\Delta\theta_k(\eta)=-\eta g_k$ | 使用相同 Armijo 回溯 |

## 控制变量与能量分组

$$
\mathcal M=\{1,2,4,8,16,30\},\qquad
\mathcal A=\{\mathrm{PB\text{-}SAV},\mathrm{PB\text{-}ESAV\ Plan\ 1},
\mathrm{PB\text{-}ESAV\ Plan\ 2},\mathrm{GD}\}.
$$

$$
E_k(\theta)=\frac1{2N}\sum_{i\in G_k}(f_\theta(x_i)-y_i)^2,
\qquad F(\theta)=\sum_{k=1}^M E_k(\theta).
$$

| 分组 | 设置 |
| --- | --- |
| 论文要求 | 嵌套随机分组；$M=30$ 时每组一个样本 |
| 本地具体规则 | 种子 3 生成样本排列；依次二分最大组，直到得到 $M$ 组 |
| 可复建记录 | `config.json` 保存 `partition_seed` 和 `group_indices` |
| 当前优先级 | 先核对 $M=30$ 的 PB-SAV 基线，再比较 PB-ESAV |

## 单组运行

```bash
conda activate numenv
python -B -m problem.nonlinear_regression
python -B run.py --problem nonlinear_regression --method pb_sav --m 30 --output results/nonlinear_regression/pb_sav_m30_armijo_lr100
```

| 数据 | 约定 |
| --- | --- |
| `data/nonlinear_regression.pt` | 训练、测试张量与生成参数；训练直接读取 |
| 重新生成 | 数据相关配置改变后执行生成命令；本次保留已有输入数据 |

PB-ESAV 和 GD 的单组入口：

```bash
python -B run.py --problem nonlinear_regression --method pb_esav --scheme plan1 --m 30 --output results/nonlinear_regression/pb_esav_plan1_m30_armijo_lr100
python -B run.py --problem nonlinear_regression --method pb_esav --scheme plan2 --m 30 --output results/nonlinear_regression/pb_esav_plan2_m30_armijo_lr100
python -B run.py --problem nonlinear_regression --method gd --m 30 --output results/nonlinear_regression/gd_m30_armijo_lr100
```

## 输出

| 文件 / 字段 | 内容 |
| --- | --- |
| `config.json` | 实际方法、数据参数、分组、步长规则与本地默认设置标识 |
| `history.csv` | 初始状态与每次接受后的状态；$n=0,\ldots,600$ |
| `objective` | $F(\theta)=\frac1{2N}\sum_i(f_\theta(x_i)-y_i)^2$ |
| `test_mse` | 无噪声测试点上的均方误差 |
| `initial_lr` / `accepted_lr` | 每轮初始学习率 100 / 接受的学习率；初始状态行为 0 |
| `rejected_trials` / `nonfinite_trials` | 该次接受前的拒绝次数 / 非有限试探次数 |
| `directional_derivative` | 接受增量对应的 $g_k^\mathsf T\Delta\theta_k(\eta)$ |
| `armijo_bound` | 接受步的 $F(\theta_k)+c\,g_k^\mathsf T\Delta\theta_k(\eta)$ |
| 辅助变量指标 | 沿用 $q,Q,q/Q$ 或 `log_r`、`log_r_i` 及一致性比值 |
| `failure.txt` | 仅在数值失败时保存失败条件；历史保留到最后接受状态 |

此前的非线性回归训练结果、图和文档结果表已全部作废并删除。

## M = 30：每轮从学习率 100 开始

$$
\eta_{k,0}=100,\qquad K=600,\qquad n=0,\ldots,600.
$$

| 方法 / 原始记录 | 最终训练目标 $F$ | 最终测试 MSE | 拒绝试探总数 | 接受的学习率范围 |
| --- | --- | --- | --- | --- |
| [PB-SAV](../results/nonlinear_regression/pb_sav_m30_armijo_lr100/history.csv) | $6.387098788\times10^{-3}$ | $7.311415189\times10^{-3}$ | 8 | $[25,100]$ |
| [PB-ESAV Plan 1](../results/nonlinear_regression/pb_esav_plan1_m30_armijo_lr100/history.csv) | $1.123444538\times10^{-1}$ | $1.793724985\times10^{-1}$ | 4814 | $[0.1953125,1.5625]$ |
| [PB-ESAV Plan 2](../results/nonlinear_regression/pb_esav_plan2_m30_armijo_lr100/history.csv) | $4.455623483\times10^{-2}$ | $6.485945268\times10^{-2}$ | 4930 | $[0.1953125,100]$ |
| [GD](../results/nonlinear_regression/gd_m30_armijo_lr100/history.csv) | $1.203258764\times10^{-1}$ | $1.928333337\times10^{-1}$ | 5349 | $[0.1953125,0.390625]$ |

每组均有 601 行有限数值记录。四种方法的接受步均满足 Armijo 条件；
拒绝试探不计入 600 次更新。PB-SAV 接受的学习率为 $25、50、100$。

| 对照 | 最终训练目标 |
| --- | --- |
| 论文 PB-SAV | $2.525\times10^{-3}$ |
| 本轮 PB-SAV | $6.387\times10^{-3}$ |

本轮 PB-SAV 尚未达到论文报告值；两个 PB-ESAV Plan 的最终训练目标均高于本轮 PB-SAV。

[M=30 训练目标与测试 MSE 曲线](../results/nonlinear_regression/training_m30_armijo_lr100.png)
