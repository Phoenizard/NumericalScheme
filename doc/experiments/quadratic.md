# 二次型实验

## 目标函数

$$
\boxed{
\min_{\boldsymbol{\phi}\in\mathbb{R}^{100}} f(\boldsymbol{\phi}),
\qquad
f(\boldsymbol{\phi})
=\sum_{k=1}^{50}\phi_{2k-1}^{2}
+\frac{1}{100}\sum_{k=1}^{50}\phi_{2k}^{2}
}
$$

$$
\boldsymbol{\phi}^{\ast}=\boldsymbol{0},
\qquad f(\boldsymbol{\phi}^{\ast})=0.
$$

## 共同参数

| 参数 | 设置 |
| --- | --- |
| 初值 | $\boldsymbol{\phi}^{0}=(1,\ldots,1)$；$f(\boldsymbol{\phi}^{0})=50.5$ |
| 更新次数 | $K=300$ |
| 总 shift | $C=10^{-10}$ |
| 正则化 | $\lambda=0$ |

## 控制变量与 PB 分组
$$
\begin{aligned}
E_1(\boldsymbol{\phi})&=\sum_{k=1}^{50}\phi_{2k-1}^{2},\\
E_2(\boldsymbol{\phi})&=\frac{1}{100}\sum_{k=1}^{50}\phi_{2k}^{2},\\
f(\boldsymbol{\phi})&=E_1(\boldsymbol{\phi})+E_2(\boldsymbol{\phi}).
\end{aligned}
$$

| 共同设置 | 值 |
| --- | --- |
| 精度 / 设备 | float64 / CPU |
| 动量 | 关闭 |
| 分组 | 基础方法 $M=1$；PB 方法 $M=2$ |

Armijo：每轮从 100 开始，拒绝后乘 0.5，最低 $10^{-6}$，系数 $10^{-4}$；拒绝时回滚参数与辅助状态。

## 1. naiveESAV 与 ESAV 的探索

### PB 方法：固定学习率

Relax 关闭；固定学习率为 0.1、1、10。

方法名称链接指向该次实验目录，包含 `config.json` 与 `history.csv`。

#### $\Delta t=0.1$

| 方法 / 原始记录 | 最终目标 $f(\boldsymbol{\phi}^{300})$ |
| --- | ---: |
| [PB-SAV](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_sav/) | $1.687650689\times10^{-1}$ |
| [PB-naiveESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_naive_esav_plan_1/) | $2.491615377\times10^{-1}$ |
| [PB-naiveESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_naive_esav_plan_2/) | $2.492495415\times10^{-1}$ |
| [PB-ESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_esav_plan_1/) | $2.459861381\times10^{-1}$ |
| [PB-ESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_esav_plan_2/) | $2.457434468\times10^{-1}$ |

![固定学习率 0.1 的优化目标曲线](../../results/quadratic/esav_exploration/figures/objective_lr0p1.png)

#### $\Delta t=1$

| 方法 / 原始记录 | 最终目标 $f(\boldsymbol{\phi}^{300})$ |
| --- | ---: |
| [PB-SAV](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_sav/) | $4.597651866\times10^{-2}$ |
| [PB-naiveESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_naive_esav_plan_1/) | $3.572298599\times10^{-3}$ |
| [PB-naiveESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_naive_esav_plan_2/) | $3.550239817\times10^{-3}$ |
| [PB-ESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_esav_plan_1/) | $2.289379497\times10^{-3}$ |
| [PB-ESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_esav_plan_2/) | $2.167517990\times10^{-3}$ |

![固定学习率 1 的优化目标曲线](../../results/quadratic/esav_exploration/figures/objective_lr1.png)

#### $\Delta t=10$

| 方法 / 原始记录 | 最终目标 $f(\boldsymbol{\phi}^{300})$ |
| --- | ---: |
| [PB-SAV](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_sav/) | $9.308350881\times10^{-2}$ |
| [PB-naiveESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_naive_esav_plan_1/) | $7.055851969\times10^{-12}$ |
| [PB-naiveESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_naive_esav_plan_2/) | $2.610606321\times10^{-12}$ |
| [PB-ESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_esav_plan_1/) | $2.563976083\times10^{-11}$ |
| [PB-ESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_esav_plan_2/) | $1.655841109\times10^{-6}$ |

![固定学习率 10 的优化目标曲线](../../results/quadratic/esav_exploration/figures/objective_lr10.png)

### naiveESAV 失效诊断

沿用上述二次型、初值、300 次更新及 float64；本节采用整体能量，$M=1$。
关闭 Relax、指数一阶展开；固定学习率为 $0.01,0.1,1,10$。

#### 首步与最终目标

| 学习率 | 首步真实目标 | 首步后比值 $r^1/\exp(E(x^1))$ | 第 300 步真实目标 | 观察 |
| ---: | ---: | ---: | ---: | --- |
| 0.01 | 48.51980002 | 0.980198654 | 0.444281970 | 300 步持续下降，比值最终约 0.990427 |
| 0.1 | 32.498002 | 0.135335013 | 5.203671824 | 第 4 步后停滞 |
| 1 | 50.4802 | $1.38362\times10^{-87}$ | 50.4802 | 第 1 步后停滞，首步真实目标仍下降 |
| 10 | 18050.32 | $\exp(-20000.02)$，float64 下为 0 | 18050.32 | 首步目标上升，随后比值下溢并停滞 |

#### 学习率 0.1 的反馈过冲

| 更新次数 | $E(x^n)$ | $\log r^n$ | $r^n/\exp(E(x^n))$ |
| ---: | ---: | ---: | ---: |
| 0 | 50.5 | 50.5 | 1 |
| 1 | 32.498002 | 30.498 | 0.135335013 |
| 2 | 30.788888211 | 30.263524275 | 0.591340115 |
| 3 | 24.046453623 | 26.025905594 | 7.238774838 |
| 4 | 5.203671824 | -467.682722068 | $4.24667\times10^{-206}$ |

$$
\Delta t\frac{r^4}{\exp(E(x^4))}\approx4.24667\times10^{-207}.
$$

![naiveESAV 目标与反馈诊断](../../results/quadratic/esav_exploration/naive_esav_diagnosis/diagnosis.png)

#### 同批基线：最终目标

每行方法均使用该行固定学习率；AdamW 的 weight decay 为 0。

| 固定学习率 | SAV 第 300 步目标 | SGD 第 300 步目标 | AdamW 第 300 步目标 |
| ---: | ---: | ---: | ---: |
| 0.01 | 0.444478289 | 0.443726925 | $1.68621\times10^{-6}$ |
| 0.1 | 0.171506596 | 0.150416257 | $2.24730\times10^{-13}$ |
| 1 | 0.091941738 | 50.000002720 | $3.74714\times10^{-13}$ |
| 10 | 0.495173945 | 数值失败，未完成 | $1.17587\times10^{-10}$ |

SGD 在学习率 10 的第 120 步出现非有限值，未完成 300 步。

#### 停滞与 Armijo 诊断

| 情形 | 观察 |
| --- | --- |
| 固定学习率 0.1 | 比值先恢复至 7.23877，后降至约 $10^{-206}$；参数更新被舍入，辅助耗散平方量下溢 |
| 固定学习率 1 | 比值约 $10^{-87}$；参数与辅助更新小于 float64 可分辨尺度 |
| 固定学习率 10 | 比值下溢为 0 |
| Armijo | 每轮从 100 开始，乘 0.5，最低 $10^{-6}$，$c=10^{-4}$ |
| Armijo 首步 | 接受学习率 0.78125，目标 16.3048095703，比值 $9.67128\times10^{-54}$ |
| Armijo 后续 | 第 2–300 步均为零斜率、无参数变化的接受步 |

有限精度停滞不等于精确算术下的一般不收敛定理。公式见[设计文档](../design/naive_esav.md)。

#### 已归档验证

| 检查 | 已保存的结果 |
| --- | --- |
| naiveESAV 四档学习率各前 5 步 | 当前实现与历史匹配；能量及偏差恒等式最大绝对残差约 $1.14\times10^{-13}$ |
| SAV 标量递推与耗散恒等式 | 最大绝对残差约 $2.35\times10^{-13}$ |
| 500 位精度 | 学习率 0.1 的第 5 步最大变化约 $2.61\times10^{-207}$；学习率 1 约 $2.77\times10^{-87}$，均非零 |
| 学习率 0.1 的第 5 步辅助对数变化 | 约 $-3.41\times10^{-411}$ |

### 原始诊断与核查

| 内容 | 来源 |
| --- | --- |
| PB 图表数据与校验值 | [sources.json](../../results/quadratic/esav_exploration/index.json) |
| naiveESAV 四档固定学习率及 Armijo 记录 | [summary.json](../../results/quadratic/esav_exploration/naive_esav_diagnosis/summary.json)（含原路径与 SHA256） |
| 首步轨迹 / 基线 | [first_steps.csv](../../results/quadratic/esav_exploration/naive_esav_diagnosis/first_steps.csv)、[baseline_summary.csv](../../results/quadratic/esav_exploration/naive_esav_diagnosis/baseline_summary.csv) |
| 公式及精度检查 | [checks.json](../../results/quadratic/esav_exploration/naive_esav_diagnosis/checks.json)、[verification.json](../../results/quadratic/esav_exploration/naive_esav_diagnosis/verification.json) |
| 诊断脚本快照 | [checks.py.txt](../../results/quadratic/esav_exploration/naive_esav_diagnosis/checks.py.txt)、[analyze_history.py.txt](../../results/quadratic/esav_exploration/naive_esav_diagnosis/analyze_history.py.txt) |

历史配置 `esav_variant="original"` 对应 naiveESAV；上述验证为既有记录，本轮仅整理与重绘。

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
| 0.1 | [$1.7150660\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/single/lr0p1/sav/) | [$1.4746489\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/single/lr0p1/mesav/) | [$1.5041626\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/single/lr0p1/sgd/) | [$2.2473041\times10^{-13}$](../../results/quadratic/mesav/fixed_lr/single/lr0p1/adam/) |
| 1 | [$9.1941738\times10^{-2}$](../../results/quadratic/mesav/fixed_lr/single/lr1/sav/) | [$3.6077427\times10^{-5}$](../../results/quadratic/mesav/fixed_lr/single/lr1/mesav/) | [$5.0000003\times10^{1}$](../../results/quadratic/mesav/fixed_lr/single/lr1/sgd/) | [$3.7471366\times10^{-13}$](../../results/quadratic/mesav/fixed_lr/single/lr1/adam/) |
| 10 | [$4.9517395\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/single/lr10/sav/) | [$2.3609191\times10^{-5}$](../../results/quadratic/mesav/fixed_lr/single/lr10/mesav/) | [$1.1023691\times10^{306}$ †（119/300）](../../results/quadratic/mesav/fixed_lr/single/lr10/sgd/) | [$1.1758665\times10^{-10}$](../../results/quadratic/mesav/fixed_lr/single/lr10/adam/) |

![single_fixed_lr](../../results/quadratic/mesav/figures/single_fixed_lr.png)

### 2.2 PB-MESAV：固定学习率

$M=2$；四种方法均使用表中固定学习率，Relax 关闭。

| 学习率 | PB-SAV | PB-MESAV | SGD | Adam |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | [$1.6876507\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr0p1/pb_sav/) | [$1.4565377\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr0p1/pb_mesav/) | [$1.5041626\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr0p1/sgd/) | [$2.2473041\times10^{-13}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr0p1/adam/) |
| 1 | [$4.5976519\times10^{-2}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr1/pb_sav/) | [$3.2393717\times10^{-7}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr1/pb_mesav/) | [$5.0000003\times10^{1}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr1/sgd/) | [$3.7471366\times10^{-13}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr1/adam/) |
| 10 | [$9.3083509\times10^{-2}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr10/pb_sav/) | [$4.6623853\times10^{-22}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr10/pb_mesav/) | [$1.1023691\times10^{306}$ †（119/300）](../../results/quadratic/mesav/fixed_lr/pb_m2/lr10/sgd/) | [$1.1758665\times10^{-10}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr10/adam/) |

![pb_fixed_lr](../../results/quadratic/mesav/figures/pb_fixed_lr.png)

### 2.3 PB-MESAV：Armijo

单能量方法 $M=1$；PB 方法 $M=2$。Adam 固定学习率 0.1，其余方法使用 Armijo。

| 方法 | 最终目标 / 记录 | 完成更新 |
| --- | ---: | ---: |
| PB-SAV | [$3.2355673\times10^{-2}$](../../results/quadratic/mesav/armijo/pb_m2/pb_sav/) | 300 |
| PB-MESAV | [$3.8942764\times10^{-17}$](../../results/quadratic/mesav/armijo/pb_m2/pb_mesav/) | 300 |
| SGD | [$8.3546216\times10^{-7}$](../../results/quadratic/mesav/armijo/pb_m2/sgd/) | 300 |
| Adam | [$2.2473041\times10^{-13}$](../../results/quadratic/mesav/armijo/pb_m2/adam/) | 300 |

![pb_armijo](../../results/quadratic/mesav/figures/pb_armijo.png)

### 2.4 Relax 扩展

单能量方法 $M=1$；PB 方法 $M=2$。Adam 固定学习率 0.1，其余方法使用 Armijo。

| 方法 | 最终目标 / 记录 | 完成更新 |
| --- | ---: | ---: |
| Relax-SAV | [$6.3978095\times10^{-7}$](../../results/quadratic/mesav/armijo/relax_comparison/relax_sav/) | 300 |
| Relax-MESAV | [$7.4109009\times10^{-7}$](../../results/quadratic/mesav/armijo/relax_comparison/relax_mesav/) | 300 |
| Relax-PB-SAV | [$1.1241397\times10^{-16}$](../../results/quadratic/mesav/armijo/relax_comparison/relax_pb_sav/) | 300 |
| Relax-PB-MESAV | [$1.2181558\times10^{-16}$](../../results/quadratic/mesav/armijo/relax_comparison/relax_pb_mesav/) | 300 |
| SGD | [$8.3546216\times10^{-7}$](../../results/quadratic/mesav/armijo/relax_comparison/sgd/) | 300 |
| Adam | [$2.2473041\times10^{-13}$](../../results/quadratic/mesav/armijo/relax_comparison/adam/) | 300 |

![relax_armijo](../../results/quadratic/mesav/figures/relax_armijo.png)

### 归档与复现

| 内容 | 位置 |
| --- | --- |
| 第一节数据索引 | [esav_exploration/index.json](../../results/quadratic/esav_exploration/index.json) |
| MESAV 数据索引 | [mesav/index.json](../../results/quadratic/mesav/index.json) |
| 来源、原命名、逐文件 SHA256、复用关系 | [归档清单](../../results/archive_20261022.json) |
| 历史代码、输入数据与核查记录 | [复现归档](../../results/reproduction_20261022/README.md) |
| 理论推导 | [naiveESAV](../design/naive_esav.md)、[ESAV](../design/esav.md)、[MESAV / PB / Relax](../design/mesav.md) |

原始配置保留当时的方法名；目录、索引和本报告统一使用 MESAV 命名。全部图表来自已保存记录，本轮未重新优化。
