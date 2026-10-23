# 非线性回归实验

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

$$
\begin{aligned}
f_\theta(x)&=\sum_{j=1}^{50}c_j\tanh(a_jx+b_j)+b_0,\\
\theta&=(a,b,c,b_0),\qquad \dim\theta=50+50+50+1=151.
\end{aligned}
$$

## 共同参数

| 配置入口 | 设置 |
| --- | --- |
| [problem/config.py](../../problem/config.py) | `PROBLEMS["nonlinear_regression"]` |

| 参数 | 设置 | 依据 |
| --- | --- | --- |
| 样本数 / 区间 / 噪声 | $N=30$；$[0,1]$；$\sigma=0.1$ | 论文 §4.2 |
| 网络 | $1\to50\to1$；tanh；151 个参数 | 论文 §4.2 |
| Batch / 精度 | 全量；`float64` | 论文 §4 |
| 参数更新次数 | $K=600$；自适应方法只统计接受的更新 | 论文 §4.2 |
| Momentum | Off；direct update，$\alpha=1$ | 论文 §4.2 |
| 正则化 | $\lambda=0$ | 论文目标函数 |
| 采样点 | $x_i=(i-1)/29$ | 本地默认 |
| 隐藏层初始化 | $a_j,b_j\sim\mathcal U(-1,1)$ | 本地默认 |
| 输出层初始化 | $c_j,b_0\sim\mathcal U(-1/\sqrt{50},1/\sqrt{50})$ | 本地默认 |
| 数据 / 初始化 / 分组种子 | 1 / 2 / 3 | 本地默认 |
| SAV / ESAV / MESAV 系列 shift | $C=10^{-10}$；$C_k=C/M$ | 本地默认；论文 §4.2 未给数值 |
| PB-naiveESAV / GD shift | 无 | 本项目定义 |
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

| Armijo 设置 | 值 | 依据 |
| --- | --- | --- |
| 学习率范围 | $10^{-6}\leq\eta\leq100$ | 用于下文 Armijo 方法 |
| 每轮初始学习率 | $\eta\leftarrow100$；与上一轮接受值无关 | 当前回溯规则 |
| Armijo 系数 | $c=10^{-4}$ | 用户批准的常规设置 |
| 拒绝后重试 | $\eta\leftarrow\max(0.5\eta,10^{-6})$；重新计算 $\Delta\theta_k(\eta)$ | 当前回溯规则 |
| 接受后 | $\theta_{k+1}=\theta_k+\Delta\theta_k(\eta)$；提交对应辅助变量 | 原优化器完整更新 |
| 拒绝的试探步 | 恢复 $\theta_k$；保留原辅助变量；不计入 $K$ | 接受／拒绝实现 |
| 下限 | $\eta=10^{-6}$ 仍被拒绝则报错，保留已接受记录 | 实现约定 |
| GD 更新 | $\Delta\theta_k(\eta)=-\eta g_k$ | 使用相同 Armijo 回溯 |

## 控制变量与能量分组

$$
M\in\{1,2,4,8,16,30\}.
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
| 对比范围 | 各节分别列出方法与 M；基础方法使用 $M=1$ |

## 1. naiveESAV 与 ESAV 的探索

本节沿用已有六组 M 的 Armijo 实验，Relax 关闭。

方法名称链接指向该次实验目录，包含 `config.json` 与 `history.csv`。

### M = 1

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_sav/) | $2.274832514\times10^{-1}$ | 12 | $[0.390625,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_naive_esav_plan_1/) | $1.187831012\times10^{-1}$ | 5038 | $[0.195312,0.78125]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_naive_esav_plan_2/) | $1.187831012\times10^{-1}$ | 5038 | $[0.195312,0.78125]$ |
| [PB-ESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_esav_plan_1/) | $1.201162212\times10^{-1}$ | 2328 | $[0.78125,100]$ |
| [PB-ESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_esav_plan_2/) | $1.198259576\times10^{-1}$ | 4646 | $[0.390625,0.78125]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m1/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=1 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m1.png)

### M = 2

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_sav/) | $2.167684229\times10^{-1}$ | 0 | $[100,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_naive_esav_plan_1/) | $1.182679391\times10^{-1}$ | 4972 | $[0.195312,0.78125]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_naive_esav_plan_2/) | $1.184271494\times10^{-1}$ | 4979 | $[0.195312,0.78125]$ |
| [PB-ESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_esav_plan_1/) | $1.206734918\times10^{-1}$ | 4731 | $[0.390625,100]$ |
| [PB-ESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_esav_plan_2/) | $1.184106383\times10^{-1}$ | 4976 | $[0.195312,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m2/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=2 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m2.png)

### M = 4

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_sav/) | $2.112231673\times10^{-1}$ | 0 | $[100,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_naive_esav_plan_1/) | $1.160201178\times10^{-1}$ | 4884 | $[0.195312,0.78125]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_naive_esav_plan_2/) | $1.164098738\times10^{-1}$ | 4896 | $[0.195312,0.78125]$ |
| [PB-ESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_esav_plan_1/) | $1.202661350\times10^{-1}$ | 4720 | $[0.390625,100]$ |
| [PB-ESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_esav_plan_2/) | $1.201690426\times10^{-1}$ | 4636 | $[0.390625,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m4/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=4 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m4.png)

### M = 8

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_sav/) | $1.704238631\times10^{-1}$ | 0 | $[100,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_naive_esav_plan_1/) | $1.168572436\times10^{-1}$ | 4920 | $[0.195312,0.78125]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_naive_esav_plan_2/) | $1.170055012\times10^{-1}$ | 4931 | $[0.195312,0.78125]$ |
| [PB-ESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_esav_plan_1/) | $2.815814043\times10^{-3}$ | 3506 | $[0.78125,100]$ |
| [PB-ESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_esav_plan_2/) | $3.081069787\times10^{-3}$ | 4093 | $[0.390625,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m8/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=8 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m8.png)

### M = 16

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_sav/) | $5.098776875\times10^{-2}$ | 2 | $[50,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_naive_esav_plan_1/) | $1.126629639\times10^{-1}$ | 4823 | $[0.195312,1.5625]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_naive_esav_plan_2/) | $1.122784692\times10^{-1}$ | 4829 | $[0.195312,1.5625]$ |
| [PB-ESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_esav_plan_1/) | $2.669282383\times10^{-3}$ | 3636 | $[0.78125,100]$ |
| [PB-ESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_esav_plan_2/) | $1.097032059\times10^{-2}$ | 342 | $[3.125,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m16/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=16 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m16.png)

### M = 30

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_sav/) | $6.387098788\times10^{-3}$ | 8 | $[25,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_naive_esav_plan_1/) | $1.123444538\times10^{-1}$ | 4814 | $[0.195312,1.5625]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_naive_esav_plan_2/) | $4.455623483\times10^{-2}$ | 4930 | $[0.195312,100]$ |
| [PB-ESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_esav_plan_1/) | $1.584532498\times10^{-3}$ | 0 | $[100,100]$ |
| [PB-ESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_esav_plan_2/) | $1.687399816\times10^{-3}$ | 0 | $[100,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m30/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=30 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m30.png)

### 论文参考

| 论文基准 | $M$ | 接受的更新次数 | 最终目标 |
| --- | --- | --- | --- |
| [PB SAV Momentum §4.2, Table 2](../paper/PB_SAV_Momentum.pdf#page=21) | 30 | 600 | $2.525\times10^{-3}$ |

论文设置与本地默认值见参数表；本报告不视为作者数值的严格复现。

## 2. MESAV 及其扩展

| 方法族 | 辅助变量与模式 |
| --- | --- |
| MESAV | $r=\exp(E/(E(\theta^0)+C))\sqrt{E+C}$；参考尺度固定 |
| PB-MESAV | 全局驱动与分量修正，只演化总辅助变量 |
| Relax | 参数更新后校正总辅助变量 |
| Adam | $\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$；历史 AdamW 的 weight decay 为 0，展示名称统一为 Adam |

† 表示未完成规定更新；所列为停止前最后有效目标，括号内为已完成 / 计划更新数。图中失败末点用 × 标记；发散超过纵轴上界时用 ▲ 标记。

### 2.1 基础 MESAV：固定学习率

$M=1$；四种方法均使用表中固定学习率，Relax 关闭。

| 学习率 | SAV | MESAV | SGD | Adam |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | [$1.2379550\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr0p1/sav/) | [$1.2385889\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr0p1/mesav/) | [$1.2362112\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr0p1/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr0p1/adam/) |
| 1 | [$2.2803681\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr1/sav/) | [$1.2186015\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr1/mesav/) | [$1.3384550\times10^{303}$ †（92/600）](../../results/nonlinear_regression/mesav/fixed_lr/single/lr1/sgd/) | [$2.9246782\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr1/adam/) |
| 10 | [$6.8930086\times10^{-1}$ †（179/600）](../../results/nonlinear_regression/mesav/fixed_lr/single/lr10/sav/) | [$1.2202762\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr10/mesav/) | [$4.9046823\times10^{304}$ †（57/600）](../../results/nonlinear_regression/mesav/fixed_lr/single/lr10/sgd/) | [$2.3934877\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/single/lr10/adam/) |

![single_fixed_lr](../../results/nonlinear_regression/mesav/figures/single_fixed_lr.png)

### 2.2 PB-MESAV：固定学习率

$M=30$；四种方法均使用表中固定学习率，Relax 关闭。

| 学习率 | PB-SAV | PB-MESAV | SGD | Adam |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | [$1.2371324\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr0p1/pb_sav/) | [$1.2379458\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr0p1/pb_mesav/) | [$1.2362112\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr0p1/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr0p1/adam/) |
| 1 | [$3.8888682\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr1/pb_sav/) | [$3.0027369\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr1/pb_mesav/) | [$1.3384550\times10^{303}$ †（92/600）](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr1/sgd/) | [$2.9246782\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr1/adam/) |
| 10 | [$8.5349745\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr10/pb_sav/) | [$2.4135097\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr10/pb_mesav/) | [$4.9046823\times10^{304}$ †（57/600）](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr10/sgd/) | [$2.3934877\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr10/adam/) |

![pb_fixed_lr](../../results/nonlinear_regression/mesav/figures/pb_fixed_lr.png)

PB-MESAV 的 Armijo 结果并入下一节的 M 扫描，不重复列出 M=30 的表格和曲线。

### 2.3 Armijo、Relax 与分组数 M

Adam 固定学习率 0.1，其余方法使用 Armijo。$M=1$ 对应单能量 SAV、MESAV 及其 Relax 版本。

| M | PB-SAV | PB-MESAV | Relax-PB-SAV | Relax-PB-MESAV | SGD | Adam |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [$2.2748325\times10^{-1}$ †（133/600）](../../results/nonlinear_regression/mesav/armijo/m1/pb_sav/) | [$1.1948802\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m1/pb_mesav/) | [$1.2041259\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m1/relax_pb_sav/) | [$1.2023788\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m1/relax_pb_mesav/) | [$1.2032588\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m1/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m1/adam/) |
| 2 | [$2.1676842\times10^{-1}$ †（168/600）](../../results/nonlinear_regression/mesav/armijo/m2/pb_sav/) | [$1.1954075\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m2/pb_mesav/) | [$1.2079924\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m2/relax_pb_sav/) | [$1.2024618\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m2/relax_pb_mesav/) | [$1.2032588\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m2/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m2/adam/) |
| 4 | [$2.1122317\times10^{-1}$ †（223/600）](../../results/nonlinear_regression/mesav/armijo/m4/pb_sav/) | [$1.1875136\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m4/pb_mesav/) | [$1.1065646\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m4/relax_pb_sav/) | [$1.0730230\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m4/relax_pb_mesav/) | [$1.2032588\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m4/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m4/adam/) |
| 8 | [$1.7042386\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m8/pb_sav/) | [$3.5056427\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m8/pb_mesav/) | [$2.6168745\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m8/relax_pb_sav/) | [$2.6076525\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m8/relax_pb_mesav/) | [$1.2032588\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m8/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m8/adam/) |
| 16 | [$5.0987769\times10^{-2}$](../../results/nonlinear_regression/mesav/armijo/m16/pb_sav/) | [$2.7894627\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m16/pb_mesav/) | [$2.5248100\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m16/relax_pb_sav/) | [$2.5239184\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m16/relax_pb_mesav/) | [$1.2032588\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/m16/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/m16/adam/) |
| 30 | [$6.3870988\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/pb_m30/pb_sav/) | [$2.0669102\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/pb_m30/pb_mesav/) | [$1.5726444\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/relax_comparison/relax_pb_sav/) | [$1.5646301\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/relax_comparison/relax_pb_mesav/) | [$1.2032588\times10^{-1}$](../../results/nonlinear_regression/mesav/armijo/relax_comparison/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/armijo/relax_comparison/adam/) |

PB-SAV 在 $M=1,2,4$ 时分别完成 133、168、223 步，随后辅助变量下溢并出现微小负值，触发失败检查。失败候选未提交，不续补到 600 步。

本节使用该批次的有效性检查及失败记录；第一节保留历史版本的原有记录，两批轨迹不拼接。

#### M=1

![armijo_m1](../../results/nonlinear_regression/mesav/figures/armijo_m1.png)

#### M=2

![armijo_m2](../../results/nonlinear_regression/mesav/figures/armijo_m2.png)

#### M=4

![armijo_m4](../../results/nonlinear_regression/mesav/figures/armijo_m4.png)

#### M=8

![armijo_m8](../../results/nonlinear_regression/mesav/figures/armijo_m8.png)

#### M=16

![armijo_m16](../../results/nonlinear_regression/mesav/figures/armijo_m16.png)

#### M=30

![armijo_m30](../../results/nonlinear_regression/mesav/figures/armijo_m30.png)

$M=30$ 的六项结果复用此前 PB / Relax 实验，表格与曲线统一引用同一份正式归档。

### 2.4 曲率信息：初始点诊断

参照 [PB-SAV Figure 4](../paper/PB_SAV_Momentum.pdf#page=22)。在共同初始参数处比较；本轮仅读取已保存矩阵误差，不计算新 Hessian。

$$
\frac{\|B-B_{\mathrm{GN}}\|_F}{\|B_{\mathrm{GN}}\|_F},\qquad
\frac{\|B-\nabla^2F\|_F}{\|\nabla^2F\|_F},\qquad
\max_{\eta\in\{0.1,1,10,100\}}
\frac{\|\Delta\theta_B(\eta)-\Delta\theta_{\mathrm{GN}}(\eta)\|_2}
{\|\Delta\theta_{\mathrm{GN}}(\eta)\|_2}.
$$

$$
B_{\mathrm{GN}}=\frac1N\sum_j\nabla f_{\theta^0}(x_j)\nabla f_{\theta^0}(x_j)^\top,
\qquad
\Delta\theta_B(\eta)=-\eta(I+\eta B)^{-1}\nabla F(\theta^0),
\qquad B_{\mathrm{MESAV}}^0=3B_{\mathrm{SAV}}^0.
$$

$$
B_{\mathrm{SAV}}^0=\sum_i
\frac{\nabla E_i(\theta^0)\nabla E_i(\theta^0)^\top}{2(E_i(\theta^0)+C/M)}.
$$

Relax 在初始点的矩阵和首步参数更新与非 Relax 重合。参照矩阵仅用于离线诊断，不传入优化器；各矩阵保持实际尺度。

| M | PB-SAV → GN | PB-MESAV → GN | PB-SAV → Hessian | PB-MESAV → Hessian | PB-SAV 步误差 | PB-MESAV 步误差 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8165991 | 0.5449504 | 0.8413426 | 0.6191234 | 0.9968959 | 0.8991767 |
| 2 | 0.785052 | 0.4580033 | 0.8145607 | 0.5537798 | 0.8315068 | 0.6545039 |
| 4 | 0.7671493 | 0.4098318 | 0.7994475 | 0.5195249 | 0.7815709 | 0.6463985 |
| 8 | 0.6203396 | 0.3137215 | 0.6784622 | 0.4560174 | 0.8667825 | 0.5446042 |
| 16 | 0.460264 | 0.7133371 | 0.5553318 | 0.7526172 | 0.7289595 | 0.7227819 |
| 30 | 6.921333e-08 | 2 | 0.3487888 | 1.90307 | 1.882295e-07 | 0.6415441 |

![MESAV 曲率信息对照](../../results/nonlinear_regression/mesav/curvature/curvature_vs_m.png)

[原始矩阵、误差与核查记录](../../results/nonlinear_regression/mesav/curvature/)。

### 归档与复现

| 内容 | 位置 |
| --- | --- |
| 第一节数据索引 | [esav_exploration/index.json](../../results/nonlinear_regression/esav_exploration/index.json) |
| MESAV 数据索引 | [mesav/index.json](../../results/nonlinear_regression/mesav/index.json) |
| 来源、原命名、逐文件 SHA256、复用关系 | [归档清单](../../results/archive_20261022.json) |
| 历史代码、输入数据与核查记录 | [复现归档](../../results/reproduction_20261022/README.md) |
| 理论推导 | [naiveESAV](../design/naive_esav.md)、[ESAV](../design/esav.md)、[MESAV / PB / Relax](../design/mesav.md) |

原始配置保留当时的方法名；目录、索引和本报告统一使用 MESAV 命名。全部图表来自已保存记录，本轮未重新优化。
