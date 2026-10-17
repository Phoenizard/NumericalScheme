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

## 固定超参数

| 配置入口 | 设置 |
| --- | --- |
| [problem/config.py](../problem/config.py) | `PROBLEMS["nonlinear_regression"]` |

| 参数 | 设置 |
| --- | --- |
| 样本数 | $N=30$ |
| 输入区间 | $x\in[0,1]$ |
| 噪声 | $\sigma=0.1$ |
| Batch size | $B=N=30$ |
| 参数更新次数 | $K=600$ |
| 正则化 | $\lambda=0$ |
| Momentum / Relax | Off / Off |
| 学习率 | $\eta=0.1$；固定步长 |
| SAV / PB-SAV shift | $C=10^{-10},\quad C_k=C/M$ |
| PB-ESAV shift | 无 |
| 训练点 | $x_i=(i-1)/29,\quad i=1,\ldots,30$ |
| 测试点 | $x_\ell^{\mathrm{test}}=\ell/1000,\quad \ell=0,\ldots,1000$；$N_{\mathrm{test}}=1001$ |
| 测试目标 | $y_\ell^{\mathrm{test}}=\sin(2\pi x_\ell^{\mathrm{test}})$ |
| 隐藏层初始化 | $a_j,b_j\overset{\mathrm{iid}}{\sim}\mathcal{U}(-1,1)$ |
| 输出层初始化 | $c_j,b_0\overset{\mathrm{iid}}{\sim}\mathcal{U}(-1/\sqrt{50},1/\sqrt{50})$ |
| 随机种子 | $s_{\mathrm{noise}}=1,\quad s_{\mathrm{init}}=2$ |
| 配对条件 | 全方法、全 $M$ 共用数据与 $\theta^0$ |
| 设备 / 精度 | CPU / `torch.float64` |

## 控制变量

$$
\mathcal{M}=\{1,2,4,8,16,30\}
$$

| 优化器 | 能量分量数 |
| --- | --- |
| PB-SAV | $M\in\mathcal{M}$ |
| PB-ESAV Plan 1 | $M\in\mathcal{M}$ |
| PB-ESAV Plan 2 | $M\in\mathcal{M}$ |

$$
x_{(1)}\leq x_{(2)}\leq\cdots\leq x_{(N)},
\qquad (x_{(i)},y_{(i)})\text{ 成对排序}.
$$

$$
k(i)=1+((i-1)\bmod M),
\qquad
G_k=\{i:k(i)=k\},\quad k=1,\ldots,M.
$$

$$
E_k(\theta)=\frac{1}{2N}\sum_{i\in G_k}
\bigl(f_\theta(x_{(i)})-y_{(i)}\bigr)^2,
\qquad
\boxed{F(\theta)=\sum_{k=1}^{M}E_k(\theta)}.
$$

| $M$ | 分组索引 |
| --- | --- |
| $2$ | $G_1=\{1,3,\ldots,29\}$；$G_2=\{2,4,\ldots,30\}$ |
| $4$ | $G_k=\{k,k+4,k+8,\ldots\}\cap\{1,\ldots,30\}$ |
| $30$ | $G_k=\{k\}$ |

## 实验矩阵

| 优化器 | $M=1$ | $M=2$ | $M=4$ | $M=8$ | $M=16$ | $M=30$ |
| --- | --- | --- | --- | --- | --- | --- |
| PB-SAV | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| PB-ESAV Plan 1 | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| PB-ESAV Plan 2 | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

| 基准 | 训练策略 |
| --- | --- |
| [PB SAV Momentum §4.2](PB_SAV_Momentum.pdf)：数据模型、网络、损失 | 本项目：固定步长、无动量、无 Relax、等距采样、均匀初始化、固定种子、排序后循环分组 |

## 单组运行

| 操作 | 命令 |
| --- | --- |
| 环境 | `conda activate numenv` |
| 生成数据；数据配置修改后重新执行 | `python -B -m problem.nonlinear_regression` |
| 数据文件 | `data/nonlinear_regression.pt` |

```bash
python run.py --problem nonlinear_regression --method pb_sav --m 4 --output results/nonlinear_regression/pb_sav_m4
python run.py --problem nonlinear_regression --method pb_esav --scheme plan1 --m 4 --output results/nonlinear_regression/pb_esav_plan1_m4
python run.py --problem nonlinear_regression --method pb_esav --scheme plan2 --m 4 --output results/nonlinear_regression/pb_esav_plan2_m4
```

| 输出 | 内容 |
| --- | --- |
| `config.json` | 实际训练设置、数据生成参数、方法、Plan、$M$ |
| `history.csv` | $n=0,\ldots,600$；`objective`、`test_mse`、辅助变量指标 |
| `objective` | $F(\theta)=\frac{1}{2N}\sum_i(f_\theta(x_i)-y_i)^2$ |
| `test_mse` | $\frac{1}{N_{\mathrm{test}}}\sum_\ell(f_\theta(x_\ell^{\mathrm{test}})-\sin(2\pi x_\ell^{\mathrm{test}}))^2$ |
| 当前进度 | 代码与数据已准备；18 组训练未运行 |
