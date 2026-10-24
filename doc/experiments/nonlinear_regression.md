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
| 动量 | 直接更新方法关闭；Adam / AdamW 使用一、二阶矩 | 各节配置 |
| 正则化 | $\lambda=0$ | 论文目标函数 |
| 采样点 | $x_i=(i-1)/29$ | 本地默认 |
| 隐藏层初始化 | $a_j,b_j\sim\mathcal U(-1,1)$ | 本地默认 |
| 输出层初始化 | $c_j,b_0\sim\mathcal U(-1/\sqrt{50},1/\sqrt{50})$ | 本地默认 |
| 数据 / 初始化 / 分组种子 | 1 / 2 / 3 | 本地默认 |
| SAV / 旧版 ESAV / MESAV 的 shift | $C=10^{-10}$；$C_k=C/M$ | 本地默认；论文 §4.2 未给数值 |
| naiveESAV / ESAV / GD / SGD shift | 无 | 各节定义 |
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

## 1. naiveESAV 与旧版 ESAV：历史探索

本节保留历史实验；旧版 ESAV 已废弃，与第三节 ESAV 的定义不同。

| 历史方法 | 辅助变量定义 |
| --- | --- |
| naiveESAV | $r=\exp(E)$ |
| PB-naiveESAV | 总量 $r=\exp(E)$；局部 $r_i=\exp(E_i)$ |
| ESAV（旧版，已废弃） | $r=\exp(E)\sqrt{E+C}$ |
| PB-ESAV（旧版，已废弃） | 总量 $r=\exp(E)$；局部 $r_i=\exp(E_i)\sqrt{E_i+C/M}$ |

历史 PB Plan 1 演化总辅助变量；Plan 2 同时演化总量与局部量。上表局部定义不表示 Plan 1 保存独立局部状态。

### 单能量 ESAV（旧版，已废弃）：Armijo

$$
r=\exp\!\left(E+\frac12\log(E+C)\right),\qquad C=10^{-10}.
$$

不加 PB、Relax；使用该批记录中的历史源码，区别于旧 PB-ESAV 的 $M=1$ 格式。
沿用上述数据、初始化、600 次更新及 Armijo 配置；SAV、SGD、Adam 引用已有记录，未重跑。SAV 使用同配置的 PB-SAV $M=1$ 记录，该格式严格等于单能量 SAV。

| 方法 / 原始记录 | 学习率策略 | 最终目标 |
| --- | --- | ---: |
| [ESAV（旧版，已废弃）](../../results/nonlinear_regression/esav_exploration/armijo/single_esav_20261023/) | Armijo | $1.178178248\times10^{-1}$ |
| [SAV（已有，PB-SAV $M=1$）](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_sav/) | Armijo | $2.274832514\times10^{-1}$ |
| [SGD（已有）](../../results/nonlinear_regression/mesav/armijo/m1/sgd/) | Armijo | $1.203258764\times10^{-1}$ |
| [Adam（已有，weight decay 为 0）](../../results/nonlinear_regression/mesav/armijo/m1/adam/) | 固定 0.1 | $2.489698335\times10^{-3}$ |

![旧版 ESAV、SAV 与已有基线的目标曲线](../../results/nonlinear_regression/esav_exploration/figures/single_esav_labelled.png)

#### 辅助比值与后期进展诊断

| 指标 | ESAV（旧版） |
| --- | ---: |
| 初始目标 / 首步目标 | 0.279168461 / 0.233177499 |
| 第 500 / 600 步目标 | 0.120102585 / 0.117817825 |
| 最后 100 步目标相对下降 | 1.90234% |
| 第 100 / 600 步辅助比值 $r/\exp(E+\frac12\log(E+C))$ | 0.359076772 / 0.356964430 |
| 接受学习率范围 | $[0.1953125,1.5625]$ |
| 回溯拒绝次数 / 非有限试探次数 | 4387 / 0 |
| 目标值完全不变的更新次数 | 0 / 600 |

后期仍缓慢下降，未出现二次型上的完全停滞；比值未坍缩至零。核对与源码快照见[历史记录](../../results/nonlinear_regression/esav_exploration/armijo/single_esav_20261023/summary.json)。

以下 PB 对比沿用已有六组 M 的 Armijo 实验，Relax 关闭。

方法名称链接指向该次实验目录，包含 `config.json` 与 `history.csv`。

### M = 1

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_sav/) | $2.274832514\times10^{-1}$ | 12 | $[0.390625,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_naive_esav_plan_1/) | $1.187831012\times10^{-1}$ | 5038 | $[0.195312,0.78125]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_naive_esav_plan_2/) | $1.187831012\times10^{-1}$ | 5038 | $[0.195312,0.78125]$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_esav_plan_1/) | $1.201162212\times10^{-1}$ | 2328 | $[0.78125,100]$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m1/pb_esav_plan_2/) | $1.198259576\times10^{-1}$ | 4646 | $[0.390625,0.78125]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m1/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=1 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m1_labelled.png)

### M = 2

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_sav/) | $2.167684229\times10^{-1}$ | 0 | $[100,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_naive_esav_plan_1/) | $1.182679391\times10^{-1}$ | 4972 | $[0.195312,0.78125]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_naive_esav_plan_2/) | $1.184271494\times10^{-1}$ | 4979 | $[0.195312,0.78125]$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_esav_plan_1/) | $1.206734918\times10^{-1}$ | 4731 | $[0.390625,100]$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m2/pb_esav_plan_2/) | $1.184106383\times10^{-1}$ | 4976 | $[0.195312,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m2/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=2 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m2_labelled.png)

### M = 4

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_sav/) | $2.112231673\times10^{-1}$ | 0 | $[100,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_naive_esav_plan_1/) | $1.160201178\times10^{-1}$ | 4884 | $[0.195312,0.78125]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_naive_esav_plan_2/) | $1.164098738\times10^{-1}$ | 4896 | $[0.195312,0.78125]$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_esav_plan_1/) | $1.202661350\times10^{-1}$ | 4720 | $[0.390625,100]$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m4/pb_esav_plan_2/) | $1.201690426\times10^{-1}$ | 4636 | $[0.390625,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m4/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=4 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m4_labelled.png)

### M = 8

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_sav/) | $1.704238631\times10^{-1}$ | 0 | $[100,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_naive_esav_plan_1/) | $1.168572436\times10^{-1}$ | 4920 | $[0.195312,0.78125]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_naive_esav_plan_2/) | $1.170055012\times10^{-1}$ | 4931 | $[0.195312,0.78125]$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_esav_plan_1/) | $2.815814043\times10^{-3}$ | 3506 | $[0.78125,100]$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m8/pb_esav_plan_2/) | $3.081069787\times10^{-3}$ | 4093 | $[0.390625,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m8/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=8 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m8_labelled.png)

### M = 16

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_sav/) | $5.098776875\times10^{-2}$ | 2 | $[50,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_naive_esav_plan_1/) | $1.126629639\times10^{-1}$ | 4823 | $[0.195312,1.5625]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_naive_esav_plan_2/) | $1.122784692\times10^{-1}$ | 4829 | $[0.195312,1.5625]$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_esav_plan_1/) | $2.669282383\times10^{-3}$ | 3636 | $[0.78125,100]$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m16/pb_esav_plan_2/) | $1.097032059\times10^{-2}$ | 342 | $[3.125,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m16/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=16 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m16_labelled.png)

### M = 30

| 方法 / 原始记录 | 最终目标 $F(\theta^{600})$ | 拒绝试探总数 | 接受学习率范围 |
| --- | ---: | ---: | --- |
| [PB-SAV](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_sav/) | $6.387098788\times10^{-3}$ | 8 | $[25,100]$ |
| [PB-naiveESAV Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_naive_esav_plan_1/) | $1.123444538\times10^{-1}$ | 4814 | $[0.195312,1.5625]$ |
| [PB-naiveESAV Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_naive_esav_plan_2/) | $4.455623483\times10^{-2}$ | 4930 | $[0.195312,100]$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_esav_plan_1/) | $1.584532498\times10^{-3}$ | 0 | $[100,100]$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/nonlinear_regression/esav_exploration/armijo/m30/pb_esav_plan_2/) | $1.687399816\times10^{-3}$ | 0 | $[100,100]$ |
| [GD](../../results/nonlinear_regression/esav_exploration/armijo/m30/gd/) | $1.203258764\times10^{-1}$ | 5349 | $[0.195312,0.390625]$ |

![M=30 的优化目标曲线](../../results/nonlinear_regression/esav_exploration/figures/objective_m30_labelled.png)

### 论文参考

| 论文基准 | $M$ | 接受的更新次数 | 最终目标 |
| --- | --- | --- | --- |
| [PB SAV Momentum §4.2, Table 2](../paper/PB_SAV_Momentum.pdf#page=21) | 30 | 600 | $2.525\times10^{-3}$ |

论文设置与本地默认值见参数表；本报告不视为作者数值的严格复现。

## 2. MESAV 及其扩展：历史记录

本节只作历史比较，MESAV 不作为正式候选。

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

![single_fixed_lr](../../results/nonlinear_regression/mesav/figures/single_fixed_lr_labelled.png)

### 2.2 PB-MESAV：固定学习率

$M=30$；四种方法均使用表中固定学习率，Relax 关闭。

| 学习率 | PB-SAV | PB-MESAV | SGD | Adam |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | [$1.2371324\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr0p1/pb_sav/) | [$1.2379458\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr0p1/pb_mesav/) | [$1.2362112\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr0p1/sgd/) | [$2.4896983\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr0p1/adam/) |
| 1 | [$3.8888682\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr1/pb_sav/) | [$3.0027369\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr1/pb_mesav/) | [$1.3384550\times10^{303}$ †（92/600）](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr1/sgd/) | [$2.9246782\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr1/adam/) |
| 10 | [$8.5349745\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr10/pb_sav/) | [$2.4135097\times10^{-3}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr10/pb_mesav/) | [$4.9046823\times10^{304}$ †（57/600）](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr10/sgd/) | [$2.3934877\times10^{-1}$](../../results/nonlinear_regression/mesav/fixed_lr/pb_m30/lr10/adam/) |

![pb_fixed_lr](../../results/nonlinear_regression/mesav/figures/pb_fixed_lr_labelled.png)

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

![armijo_m1](../../results/nonlinear_regression/mesav/figures/armijo_m1_labelled.png)

#### M=2

![armijo_m2](../../results/nonlinear_regression/mesav/figures/armijo_m2_labelled.png)

#### M=4

![armijo_m4](../../results/nonlinear_regression/mesav/figures/armijo_m4_labelled.png)

#### M=8

![armijo_m8](../../results/nonlinear_regression/mesav/figures/armijo_m8_labelled.png)

#### M=16

![armijo_m16](../../results/nonlinear_regression/mesav/figures/armijo_m16_labelled.png)

#### M=30

![armijo_m30](../../results/nonlinear_regression/mesav/figures/armijo_m30_labelled.png)

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

![MESAV 曲率信息对照](../../results/nonlinear_regression/mesav/figures/curvature_labelled.png)

[原始矩阵、误差与核查记录](../../results/nonlinear_regression/mesav/curvature/)。

## 3. ESAV 及其扩展

$$
r=\exp\!\left(\frac{E}{4E^0}\right),\qquad
E^0=E(\theta^0)>0\ \text{冻结},\qquad
B^n=\sum_{i=1}^M\frac{\nabla E_i(\theta^n)\nabla E_i(\theta^n)^\top}{4E_i^0}.
$$

| 设置 | 值 |
| --- | --- |
| ESAV / PB-ESAV | 无 shift；只演化一个总辅助变量；$E^0,E_i^0$ 冻结 |
| Relax-PB-ESAV | $\log r^{n+1}=\min\{E(\theta^{n+1})/(4E^0),\log r^n\}$ |
| SAV 系列 | $C=10^{-10}$ |
| AdamW | $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$；weight decay 为 0 |
| 其他设置 | 沿用报告开头的目标、数据、初值、分组及 CPU float64 |

† 为未完成预算的失败记录，括号为已完成 / 计划更新数；表中取最后有效目标，曲线在该步结束。× 标记失败末点；▲ 标记发散曲线超出纵轴范围。‡ 为记录写满预算但已停滞或出现辅助变量异常。

### 3.1 基础 ESAV：固定学习率

$M=1$；PB、Relax、Armijo 关闭。每档所有方法（包括 AdamW）均使用该档学习率。

| 学习率 | ESAV | SAV | SGD | AdamW |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | [$1.23696065\times10^{-1}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr0p1/esav/) | [$1.23795501\times10^{-1}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr0p1/sav/) | [$1.23621123\times10^{-1}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr0p1/sgd/) | [$2.48969834\times10^{-3}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr0p1/adamw/) |
| 1 | [$1.21980107\times10^{-1}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr1/esav/) | [$2.28036812\times10^{-1}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr1/sav/) | [$1.33845503\times10^{303}$ †（92/600）](../../results/nonlinear_regression/esav/fixed_lr/single/lr1/sgd/) | [$2.92467819\times10^{-3}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr1/adamw/) |
| 10 | [$1.21750815\times10^{-1}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr10/esav/) | [$6.89300856\times10^{-1}$ †（179/600）](../../results/nonlinear_regression/esav/fixed_lr/single/lr10/sav/) | [$4.90468225\times10^{304}$ †（57/600）](../../results/nonlinear_regression/esav/fixed_lr/single/lr10/sgd/) | [$2.39348773\times10^{-1}$](../../results/nonlinear_regression/esav/fixed_lr/single/lr10/adamw/) |

![优化目标曲线](../../results/nonlinear_regression/esav/figures/single_fixed_lr.png)

### 3.2 基础 ESAV：Armijo

ESAV、SAV、SGD 使用 Armijo；AdamW 固定学习率 0.1。$M=1$，Relax 关闭。

| 方法 | 最后目标 / 原始记录 | 记录更新数 |
| --- | ---: | ---: |
| ESAV | [$1.18403201\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/single/esav/) | 600/600 |
| SAV | [$2.27483251\times10^{-1}$ ‡](../../results/nonlinear_regression/esav/armijo/single/sav/) | 600/600 |
| SGD | [$1.20325876\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/single/sgd/) | 600/600 |
| AdamW | [$2.48969834\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/single/adamw/) | 600/600 |

‡ SAV 有 591 次目标值不变的更新；辅助变量进入次正规数区间，比值曾为 $-2.7\times10^{-322}$。保留该历史记录的 600 步，不将其记作成功收敛，也不与下节第 133 步停止的 PB-SAV 轨迹拼接。

![优化目标曲线](../../results/nonlinear_regression/esav/figures/single_armijo.png)

### 3.3 PB 与 Relax-PB：Armijo

AdamW 固定学习率 0.1，其余方法使用 Armijo。PB 与 Relax-PB 共用下表，避免重复展示 PB 轨迹。

| M | PB-ESAV | Relax-PB-ESAV | PB-SAV | Relax-PB-SAV | SGD | AdamW |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [$1.18403201\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m1/pb_esav/) | [$1.20374474\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m1/relax_pb_esav/) | [$2.27483251\times10^{-1}$ †（133/600）](../../results/nonlinear_regression/esav/armijo/m1/pb_sav/) | [$1.20412593\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m1/relax_pb_sav/) | [$1.20325876\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m1/sgd/) | [$2.48969834\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m1/adamw/) |
| 2 | [$1.18193081\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m2/pb_esav/) | [$1.20334314\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m2/relax_pb_esav/) | [$2.16768423\times10^{-1}$ †（168/600）](../../results/nonlinear_regression/esav/armijo/m2/pb_sav/) | [$1.20799244\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m2/relax_pb_sav/) | [$1.20325876\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m2/sgd/) | [$2.48969834\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m2/adamw/) |
| 4 | [$1.17767090\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m4/pb_esav/) | [$1.20215439\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m4/relax_pb_esav/) | [$2.11223167\times10^{-1}$ †（223/600）](../../results/nonlinear_regression/esav/armijo/m4/pb_sav/) | [$1.10656462\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m4/relax_pb_sav/) | [$1.20325876\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m4/sgd/) | [$2.48969834\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m4/adamw/) |
| 8 | [$3.49847834\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m8/pb_esav/) | [$2.80924183\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m8/relax_pb_esav/) | [$1.70423863\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m8/pb_sav/) | [$2.61687447\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m8/relax_pb_sav/) | [$1.20325876\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m8/sgd/) | [$2.48969834\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m8/adamw/) |
| 16 | [$3.64898556\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m16/pb_esav/) | [$3.23443339\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m16/relax_pb_esav/) | [$5.09877687\times10^{-2}$](../../results/nonlinear_regression/esav/armijo/m16/pb_sav/) | [$2.52481005\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m16/relax_pb_sav/) | [$1.20325876\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m16/sgd/) | [$2.48969834\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m16/adamw/) |
| 30 | [$3.09197828\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m30/pb_esav/) | [$3.33947790\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m30/relax_pb_esav/) | [$6.38709879\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m30/pb_sav/) | [$1.57264435\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m30/relax_pb_sav/) | [$1.20325876\times10^{-1}$](../../results/nonlinear_regression/esav/armijo/m30/sgd/) | [$2.48969834\times10^{-3}$](../../results/nonlinear_regression/esav/armijo/m30/adamw/) |

$M=1$ 严格恢复单能量格式；$M=1,2,4$ 的 PB-SAV 分别在 133、168、223 步后因辅助变量下溢并出现负值而停止。

#### M=1

![优化目标曲线](../../results/nonlinear_regression/esav/figures/pb_relax_m1.png)

#### M=2

![优化目标曲线](../../results/nonlinear_regression/esav/figures/pb_relax_m2.png)

#### M=4

![优化目标曲线](../../results/nonlinear_regression/esav/figures/pb_relax_m4.png)

#### M=8

![优化目标曲线](../../results/nonlinear_regression/esav/figures/pb_relax_m8.png)

#### M=16

![优化目标曲线](../../results/nonlinear_regression/esav/figures/pb_relax_m16.png)

#### M=30

![优化目标曲线](../../results/nonlinear_regression/esav/figures/pb_relax_m30.png)

### 3.4 必要诊断

| 基础 ESAV 固定学习率 | 首步目标 / 初始目标 | 最终辅助比值 | 最后 100 步相对下降 | 零参数更新数 |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.758893 | 0.961072 | 0.2339% | 0 |
| 1 | 3.03124 | 0.168306 | 0.6399% | 0 |
| 10 | 10.2041 | 0.0179044 | 0.7172% | 0 |

| Relax-PB-ESAV：M | 最终辅助比值 | 最后 100 步相对下降 | 零参数更新数 |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 1.0858% | 0 |
| 2 | 1 | 1.1031% | 0 |
| 4 | 1 | 1.1401% | 0 |
| 8 | 1 | 3.8017% | 0 |
| 16 | 1 | 3.4398% | 0 |
| 30 | 1 | 3.1048% | 0 |

辅助比值定义为 $r^n/\exp(E(\theta^n)/(4E^0))$。Relax-PB-ESAV 的全部 Armijo 接受步比值为 1。

基础 ESAV 在固定学习率 1、10 时首步过冲、后期仍缓慢；Relax-PB-ESAV 在 $M=1,2,4$ 时停留在约 0.12，$M=8,16,30$ 时达到约 $0.003$，仍高于对应 Relax-PB-SAV。

各数值链接指向含原始配置、逐步记录、末态与失败材料的目录。

## 4. BESAV 及其扩展

$$r=\exp\!\left(\frac{E}{E+E^0}\right),\qquad E^0>0\ \text{冻结}.$$

### 4.1 共同参数与控制变量

目标函数、数据、初值、分组和更新预算沿用报告开头。CPU float64；BESAV 无 shift，SAV 总 shift 为 1e-10。基础固定学习率取 0.1、1、10；PB 与 Relax-PB 使用每轮从 100 开始的 Armijo。

AdamW 的 betas=(0.9,0.999)、eps=1e-8、weight_decay=0；固定学习率对比使用同档学习率，Armijo 对比使用固定 0.1。

下列 BESAV、尺度 ESAV、SAV、SGD、AdamW 均读取已有记录，本轮仅迁移接口、归档与重绘，不重跑完整实验。旧配置中的 bounded_exp、pb_bounded_exp、relax_pb_bounded_exp 分别展示为 BESAV、PB-BESAV、Relax-PB-BESAV，原配置与 CSV 不改写。

† 表示提前失败，括号为实际/计划更新数；比较使用最后有效目标值。曲线保留失败前轨迹，× 标记失败末点，▲ 标记超出纵轴的值。基础 Armijo 的 SAV 写满预算但出现停滞和辅助状态异常。

### 4.2 基础 BESAV：固定学习率

| 学习率 | BESAV | ESAV（尺度） | SAV | SGD | AdamW |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | [1.23681405e-01](../../results/nonlinear_regression/besav/fixed_lr/single/lr0p1/besav/) | [1.23696065e-01](../../results/nonlinear_regression/esav/fixed_lr/single/lr0p1/esav/) | [1.23795501e-01](../../results/nonlinear_regression/esav/fixed_lr/single/lr0p1/sav/) | [1.23621123e-01](../../results/nonlinear_regression/esav/fixed_lr/single/lr0p1/sgd/) | [2.48969834e-03](../../results/nonlinear_regression/esav/fixed_lr/single/lr0p1/adamw/) |
| 1 | [1.22015296e-01](../../results/nonlinear_regression/besav/fixed_lr/single/lr1/besav/) | [1.21980107e-01](../../results/nonlinear_regression/esav/fixed_lr/single/lr1/esav/) | [2.28036812e-01](../../results/nonlinear_regression/esav/fixed_lr/single/lr1/sav/) | [1.33845503e+303 †（92/600）](../../results/nonlinear_regression/esav/fixed_lr/single/lr1/sgd/) | [2.92467819e-03](../../results/nonlinear_regression/esav/fixed_lr/single/lr1/adamw/) |
| 10 | [1.35913378e+303 †（98/600）](../../results/nonlinear_regression/besav/fixed_lr/single/lr10/besav/) | [1.21750815e-01](../../results/nonlinear_regression/esav/fixed_lr/single/lr10/esav/) | [6.89300856e-01 †（179/600）](../../results/nonlinear_regression/esav/fixed_lr/single/lr10/sav/) | [4.90468225e+304 †（57/600）](../../results/nonlinear_regression/esav/fixed_lr/single/lr10/sgd/) | [2.39348773e-01](../../results/nonlinear_regression/esav/fixed_lr/single/lr10/adamw/) |

![基础固定学习率](../../results/nonlinear_regression/besav/figures/single_fixed_lr.png)

### 4.3 基础 BESAV：Armijo

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| BESAV | [1.18812828e-01](../../results/nonlinear_regression/besav/armijo/single/besav/) | 600/600 |
| ESAV (scale) | [1.18403201e-01](../../results/nonlinear_regression/esav/armijo/single/esav/) | 600/600 |
| SAV | [2.27483251e-01](../../results/nonlinear_regression/esav/armijo/single/sav/) | 600/600 |
| SGD | [1.20325876e-01](../../results/nonlinear_regression/esav/armijo/single/sgd/) | 600/600 |
| AdamW | [2.48969834e-03](../../results/nonlinear_regression/esav/armijo/single/adamw/) | 600/600 |

![基础 Armijo](../../results/nonlinear_regression/besav/figures/single_armijo.png)

### 4.4 PB 与 Relax-PB：Armijo

#### M=1

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| PB-BESAV | [1.18812828e-01](../../results/nonlinear_regression/besav/armijo/m1/pb_besav/) | 600/600 |
| Relax-PB-BESAV | [1.20363759e-01](../../results/nonlinear_regression/besav/armijo/m1/relax_pb_besav/) | 600/600 |
| PB-ESAV (scale) | [1.18403201e-01](../../results/nonlinear_regression/esav/armijo/m1/pb_esav/) | 600/600 |
| Relax-PB-ESAV (scale) | [1.20374474e-01](../../results/nonlinear_regression/esav/armijo/m1/relax_pb_esav/) | 600/600 |
| PB-SAV | [2.27483251e-01 †（133/600）](../../results/nonlinear_regression/esav/armijo/m1/pb_sav/) | 133/600 |
| Relax-PB-SAV | [1.20412593e-01](../../results/nonlinear_regression/esav/armijo/m1/relax_pb_sav/) | 600/600 |
| SGD | [1.20325876e-01](../../results/nonlinear_regression/esav/armijo/m1/sgd/) | 600/600 |
| AdamW | [2.48969834e-03](../../results/nonlinear_regression/esav/armijo/m1/adamw/) | 600/600 |

![M=1 优化目标](../../results/nonlinear_regression/besav/figures/pb_relax_m1.png)

#### M=2

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| PB-BESAV | [1.15631221e-01](../../results/nonlinear_regression/besav/armijo/m2/pb_besav/) | 600/600 |
| Relax-PB-BESAV | [1.20320941e-01](../../results/nonlinear_regression/besav/armijo/m2/relax_pb_besav/) | 600/600 |
| PB-ESAV (scale) | [1.18193081e-01](../../results/nonlinear_regression/esav/armijo/m2/pb_esav/) | 600/600 |
| Relax-PB-ESAV (scale) | [1.20334314e-01](../../results/nonlinear_regression/esav/armijo/m2/relax_pb_esav/) | 600/600 |
| PB-SAV | [2.16768423e-01 †（168/600）](../../results/nonlinear_regression/esav/armijo/m2/pb_sav/) | 168/600 |
| Relax-PB-SAV | [1.20799244e-01](../../results/nonlinear_regression/esav/armijo/m2/relax_pb_sav/) | 600/600 |
| SGD | [1.20325876e-01](../../results/nonlinear_regression/esav/armijo/m2/sgd/) | 600/600 |
| AdamW | [2.48969834e-03](../../results/nonlinear_regression/esav/armijo/m2/adamw/) | 600/600 |

![M=2 优化目标](../../results/nonlinear_regression/besav/figures/pb_relax_m2.png)

#### M=4

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| PB-BESAV | [1.15567698e-01](../../results/nonlinear_regression/besav/armijo/m4/pb_besav/) | 600/600 |
| Relax-PB-BESAV | [1.20119086e-01](../../results/nonlinear_regression/besav/armijo/m4/relax_pb_besav/) | 600/600 |
| PB-ESAV (scale) | [1.17767090e-01](../../results/nonlinear_regression/esav/armijo/m4/pb_esav/) | 600/600 |
| Relax-PB-ESAV (scale) | [1.20215439e-01](../../results/nonlinear_regression/esav/armijo/m4/relax_pb_esav/) | 600/600 |
| PB-SAV | [2.11223167e-01 †（223/600）](../../results/nonlinear_regression/esav/armijo/m4/pb_sav/) | 223/600 |
| Relax-PB-SAV | [1.10656462e-01](../../results/nonlinear_regression/esav/armijo/m4/relax_pb_sav/) | 600/600 |
| SGD | [1.20325876e-01](../../results/nonlinear_regression/esav/armijo/m4/sgd/) | 600/600 |
| AdamW | [2.48969834e-03](../../results/nonlinear_regression/esav/armijo/m4/adamw/) | 600/600 |

![M=4 优化目标](../../results/nonlinear_regression/besav/figures/pb_relax_m4.png)

#### M=8

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| PB-BESAV | [3.34115704e-03](../../results/nonlinear_regression/besav/armijo/m8/pb_besav/) | 600/600 |
| Relax-PB-BESAV | [2.91731977e-03](../../results/nonlinear_regression/besav/armijo/m8/relax_pb_besav/) | 600/600 |
| PB-ESAV (scale) | [3.49847834e-03](../../results/nonlinear_regression/esav/armijo/m8/pb_esav/) | 600/600 |
| Relax-PB-ESAV (scale) | [2.80924183e-03](../../results/nonlinear_regression/esav/armijo/m8/relax_pb_esav/) | 600/600 |
| PB-SAV | [1.70423863e-01](../../results/nonlinear_regression/esav/armijo/m8/pb_sav/) | 600/600 |
| Relax-PB-SAV | [2.61687447e-03](../../results/nonlinear_regression/esav/armijo/m8/relax_pb_sav/) | 600/600 |
| SGD | [1.20325876e-01](../../results/nonlinear_regression/esav/armijo/m8/sgd/) | 600/600 |
| AdamW | [2.48969834e-03](../../results/nonlinear_regression/esav/armijo/m8/adamw/) | 600/600 |

![M=8 优化目标](../../results/nonlinear_regression/besav/figures/pb_relax_m8.png)

#### M=16

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| PB-BESAV | [3.20394114e-03](../../results/nonlinear_regression/besav/armijo/m16/pb_besav/) | 600/600 |
| Relax-PB-BESAV | [2.86014906e-03](../../results/nonlinear_regression/besav/armijo/m16/relax_pb_besav/) | 600/600 |
| PB-ESAV (scale) | [3.64898556e-03](../../results/nonlinear_regression/esav/armijo/m16/pb_esav/) | 600/600 |
| Relax-PB-ESAV (scale) | [3.23443339e-03](../../results/nonlinear_regression/esav/armijo/m16/relax_pb_esav/) | 600/600 |
| PB-SAV | [5.09877687e-02](../../results/nonlinear_regression/esav/armijo/m16/pb_sav/) | 600/600 |
| Relax-PB-SAV | [2.52481005e-03](../../results/nonlinear_regression/esav/armijo/m16/relax_pb_sav/) | 600/600 |
| SGD | [1.20325876e-01](../../results/nonlinear_regression/esav/armijo/m16/sgd/) | 600/600 |
| AdamW | [2.48969834e-03](../../results/nonlinear_regression/esav/armijo/m16/adamw/) | 600/600 |

![M=16 优化目标](../../results/nonlinear_regression/besav/figures/pb_relax_m16.png)

#### M=30

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| PB-BESAV | [3.15322055e-03](../../results/nonlinear_regression/besav/armijo/m30/pb_besav/) | 600/600 |
| Relax-PB-BESAV | [2.65355240e-03](../../results/nonlinear_regression/besav/armijo/m30/relax_pb_besav/) | 600/600 |
| PB-ESAV (scale) | [3.09197828e-03](../../results/nonlinear_regression/esav/armijo/m30/pb_esav/) | 600/600 |
| Relax-PB-ESAV (scale) | [3.33947790e-03](../../results/nonlinear_regression/esav/armijo/m30/relax_pb_esav/) | 600/600 |
| PB-SAV | [6.38709879e-03](../../results/nonlinear_regression/esav/armijo/m30/pb_sav/) | 600/600 |
| Relax-PB-SAV | [1.57264435e-03](../../results/nonlinear_regression/esav/armijo/m30/relax_pb_sav/) | 600/600 |
| SGD | [1.20325876e-01](../../results/nonlinear_regression/esav/armijo/m30/sgd/) | 600/600 |
| AdamW | [2.48969834e-03](../../results/nonlinear_regression/esav/armijo/m30/adamw/) | 600/600 |

![M=30 优化目标](../../results/nonlinear_regression/besav/figures/pb_relax_m30.png)

## 原始记录与索引

| 内容 | 位置 |
| --- | --- |
| naiveESAV / 旧版 ESAV 历史数据 | [esav_exploration/index.json](../../results/nonlinear_regression/esav_exploration/index.json) |
| MESAV 历史数据 | [mesav/index.json](../../results/nonlinear_regression/mesav/index.json) |
| ESAV 数据 | [esav/index.json](../../results/nonlinear_regression/esav/index.json) |
| BESAV 数据 | [besav/index.json](../../results/nonlinear_regression/besav/index.json) |
| 既有历史归档 | [归档清单](../../results/archive_20261022.json)、[复现资料](../../results/reproduction_20261022/README.md) |
| 历史设计推导 | [naiveESAV](../design/naive_esav.md)、[旧版 ESAV](../design/esav_legacy.md)、[MESAV](../design/mesav.md) |
