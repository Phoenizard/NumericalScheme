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
| SAV 与历史复合方案的 shift | $C=10^{-10}$ |
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
| 动量 | 直接更新方法关闭；AdamW 使用一、二阶矩 |
| 分组 | 基础方法 $M=1$；PB 方法 $M=2$ |

Armijo：每轮从 100 开始，拒绝后乘 0.5，最低 $10^{-6}$，系数 $10^{-4}$；拒绝时回滚参数与辅助状态。

## 1. naiveESAV 与旧版 ESAV：历史探索

本节保留历史实验；旧版 ESAV 已废弃，与第三节 ESAV 的定义不同。

| 历史方法 | 辅助变量定义 |
| --- | --- |
| naiveESAV | $r=\exp(E)$ |
| PB-naiveESAV | 总量 $r=\exp(E)$；局部 $r_i=\exp(E_i)$ |
| ESAV（旧版，已废弃） | $r=\exp(E)\sqrt{E+C}$ |
| PB-ESAV（旧版，已废弃） | 总量 $r=\exp(E)$；局部 $r_i=\exp(E_i)\sqrt{E_i+C/M}$ |

历史 PB Plan 1 演化总辅助变量；Plan 2 同时演化总量与局部量。上表局部定义不表示 Plan 1 保存独立局部状态。

### PB 方法：固定学习率

Relax 关闭；固定学习率为 0.1、1、10。

方法名称链接指向该次实验目录，包含 `config.json` 与 `history.csv`。

#### $\Delta t=0.1$

| 方法 / 原始记录 | 最终目标 $f(\boldsymbol{\phi}^{300})$ |
| --- | ---: |
| [PB-SAV](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_sav/) | $1.687650689\times10^{-1}$ |
| [PB-naiveESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_naive_esav_plan_1/) | $2.491615377\times10^{-1}$ |
| [PB-naiveESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_naive_esav_plan_2/) | $2.492495415\times10^{-1}$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_esav_plan_1/) | $2.459861381\times10^{-1}$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr0.1/pb_esav_plan_2/) | $2.457434468\times10^{-1}$ |

![固定学习率 0.1 的优化目标曲线](../../results/quadratic/esav_exploration/figures/objective_lr0p1_labelled.png)

#### $\Delta t=1$

| 方法 / 原始记录 | 最终目标 $f(\boldsymbol{\phi}^{300})$ |
| --- | ---: |
| [PB-SAV](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_sav/) | $4.597651866\times10^{-2}$ |
| [PB-naiveESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_naive_esav_plan_1/) | $3.572298599\times10^{-3}$ |
| [PB-naiveESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_naive_esav_plan_2/) | $3.550239817\times10^{-3}$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_esav_plan_1/) | $2.289379497\times10^{-3}$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr1/pb_esav_plan_2/) | $2.167517990\times10^{-3}$ |

![固定学习率 1 的优化目标曲线](../../results/quadratic/esav_exploration/figures/objective_lr1_labelled.png)

#### $\Delta t=10$

| 方法 / 原始记录 | 最终目标 $f(\boldsymbol{\phi}^{300})$ |
| --- | ---: |
| [PB-SAV](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_sav/) | $9.308350881\times10^{-2}$ |
| [PB-naiveESAV Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_naive_esav_plan_1/) | $7.055851969\times10^{-12}$ |
| [PB-naiveESAV Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_naive_esav_plan_2/) | $2.610606321\times10^{-12}$ |
| [PB-ESAV（旧版，已废弃）Plan 1](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_esav_plan_1/) | $2.563976083\times10^{-11}$ |
| [PB-ESAV（旧版，已废弃）Plan 2](../../results/quadratic/esav_exploration/fixed_lr/lr10/pb_esav_plan_2/) | $1.655841109\times10^{-6}$ |

![固定学习率 10 的优化目标曲线](../../results/quadratic/esav_exploration/figures/objective_lr10_labelled.png)

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

![naiveESAV 目标与反馈诊断](../../results/quadratic/esav_exploration/figures/naive_diagnosis_labelled.png)

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
| 0.1 | [$1.7150660\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/single/lr0p1/sav/) | [$1.4746489\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/single/lr0p1/mesav/) | [$1.5041626\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/single/lr0p1/sgd/) | [$2.2473041\times10^{-13}$](../../results/quadratic/mesav/fixed_lr/single/lr0p1/adam/) |
| 1 | [$9.1941738\times10^{-2}$](../../results/quadratic/mesav/fixed_lr/single/lr1/sav/) | [$3.6077427\times10^{-5}$](../../results/quadratic/mesav/fixed_lr/single/lr1/mesav/) | [$5.0000003\times10^{1}$](../../results/quadratic/mesav/fixed_lr/single/lr1/sgd/) | [$3.7471366\times10^{-13}$](../../results/quadratic/mesav/fixed_lr/single/lr1/adam/) |
| 10 | [$4.9517395\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/single/lr10/sav/) | [$2.3609191\times10^{-5}$](../../results/quadratic/mesav/fixed_lr/single/lr10/mesav/) | [$1.1023691\times10^{306}$ †（119/300）](../../results/quadratic/mesav/fixed_lr/single/lr10/sgd/) | [$1.1758665\times10^{-10}$](../../results/quadratic/mesav/fixed_lr/single/lr10/adam/) |

![single_fixed_lr](../../results/quadratic/mesav/figures/single_fixed_lr_labelled.png)

### 2.2 PB-MESAV：固定学习率

$M=2$；四种方法均使用表中固定学习率，Relax 关闭。

| 学习率 | PB-SAV | PB-MESAV | SGD | Adam |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | [$1.6876507\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr0p1/pb_sav/) | [$1.4565377\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr0p1/pb_mesav/) | [$1.5041626\times10^{-1}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr0p1/sgd/) | [$2.2473041\times10^{-13}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr0p1/adam/) |
| 1 | [$4.5976519\times10^{-2}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr1/pb_sav/) | [$3.2393717\times10^{-7}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr1/pb_mesav/) | [$5.0000003\times10^{1}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr1/sgd/) | [$3.7471366\times10^{-13}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr1/adam/) |
| 10 | [$9.3083509\times10^{-2}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr10/pb_sav/) | [$4.6623853\times10^{-22}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr10/pb_mesav/) | [$1.1023691\times10^{306}$ †（119/300）](../../results/quadratic/mesav/fixed_lr/pb_m2/lr10/sgd/) | [$1.1758665\times10^{-10}$](../../results/quadratic/mesav/fixed_lr/pb_m2/lr10/adam/) |

![pb_fixed_lr](../../results/quadratic/mesav/figures/pb_fixed_lr_labelled.png)

### 2.3 PB-MESAV：Armijo

单能量方法 $M=1$；PB 方法 $M=2$。Adam 固定学习率 0.1，其余方法使用 Armijo。

| 方法 | 最终目标 / 记录 | 完成更新 |
| --- | ---: | ---: |
| PB-SAV | [$3.2355673\times10^{-2}$](../../results/quadratic/mesav/armijo/pb_m2/pb_sav/) | 300 |
| PB-MESAV | [$3.8942764\times10^{-17}$](../../results/quadratic/mesav/armijo/pb_m2/pb_mesav/) | 300 |
| SGD | [$8.3546216\times10^{-7}$](../../results/quadratic/mesav/armijo/pb_m2/sgd/) | 300 |
| Adam | [$2.2473041\times10^{-13}$](../../results/quadratic/mesav/armijo/pb_m2/adam/) | 300 |

![pb_armijo](../../results/quadratic/mesav/figures/pb_armijo_labelled.png)

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

![relax_armijo](../../results/quadratic/mesav/figures/relax_armijo_labelled.png)

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
| 0.1 | [$1.55169932\times10^{-1}$](../../results/quadratic/esav/fixed_lr/single/lr0p1/esav/) | [$1.71506596\times10^{-1}$](../../results/quadratic/esav/fixed_lr/single/lr0p1/sav/) | [$1.50416257\times10^{-1}$](../../results/quadratic/esav/fixed_lr/single/lr0p1/sgd/) | [$2.24730408\times10^{-13}$](../../results/quadratic/esav/fixed_lr/single/lr0p1/adamw/) |
| 1 | [$4.10400709\times10^{-5}$](../../results/quadratic/esav/fixed_lr/single/lr1/esav/) | [$9.19417380\times10^{-2}$](../../results/quadratic/esav/fixed_lr/single/lr1/sav/) | [$5.00000027\times10^{1}$](../../results/quadratic/esav/fixed_lr/single/lr1/sgd/) | [$3.74713664\times10^{-13}$](../../results/quadratic/esav/fixed_lr/single/lr1/adamw/) |
| 10 | [$2.61888689\times10^{-5}$](../../results/quadratic/esav/fixed_lr/single/lr10/esav/) | [$4.95173945\times10^{-1}$](../../results/quadratic/esav/fixed_lr/single/lr10/sav/) | [$1.10236910\times10^{306}$ †（119/300）](../../results/quadratic/esav/fixed_lr/single/lr10/sgd/) | [$1.17586655\times10^{-10}$](../../results/quadratic/esav/fixed_lr/single/lr10/adamw/) |

![优化目标曲线](../../results/quadratic/esav/figures/single_fixed_lr.png)

### 3.2 基础 ESAV：Armijo

ESAV、SAV、SGD 使用 Armijo；AdamW 固定学习率 0.1。$M=1$，Relax 关闭。

| 方法 | 最后目标 / 原始记录 | 记录更新数 |
| --- | ---: | ---: |
| ESAV | [$9.21369124\times10^{-7}$](../../results/quadratic/esav/armijo/single/esav/) | 300/300 |
| SAV | [$4.53719836\times10^{-1}$ ‡](../../results/quadratic/esav/armijo/single/sav/) | 300/300 |
| SGD | [$8.35462161\times10^{-7}$](../../results/quadratic/esav/armijo/single/sgd/) | 300/300 |
| AdamW | [$2.24730408\times10^{-13}$](../../results/quadratic/esav/armijo/single/adamw/) | 300/300 |

‡ SAV 有 270 次目标值不变的更新，末辅助比值约 $3.23\times10^{-151}$。

![优化目标曲线](../../results/quadratic/esav/figures/single_armijo.png)

### 3.3 PB 与 Relax-PB：Armijo

AdamW 固定学习率 0.1，其余方法使用 Armijo。PB 与 Relax-PB 共用下表，避免重复展示 PB 轨迹。

| M | PB-ESAV | Relax-PB-ESAV | PB-SAV | Relax-PB-SAV | SGD | AdamW |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | [$2.86933412\times10^{-38}$](../../results/quadratic/esav/armijo/m2/pb_esav/) | [$1.36948386\times10^{-38}$](../../results/quadratic/esav/armijo/m2/relax_pb_esav/) | [$3.23556733\times10^{-2}$](../../results/quadratic/esav/armijo/m2/pb_sav/) | [$1.12413968\times10^{-16}$](../../results/quadratic/esav/armijo/m2/relax_pb_sav/) | [$8.35462161\times10^{-7}$](../../results/quadratic/esav/armijo/m2/sgd/) | [$2.24730408\times10^{-13}$](../../results/quadratic/esav/armijo/m2/adamw/) |

#### M=2

![优化目标曲线](../../results/quadratic/esav/figures/pb_relax_m2.png)

### 3.4 必要诊断

| 基础 ESAV 固定学习率 | 首步目标 / 初始目标 | 最终辅助比值 | 最后 100 步相对下降 | 零参数更新数 |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.672396 | 0.974777 | 32.3140% | 0 |
| 1 | 0.00972701 | 0.778793 | 95.6704% | 0 |
| 10 | 0.699084 | 0.0812467 | 96.2242% | 0 |

| Relax-PB-ESAV：M | 最终辅助比值 | 最后 100 步相对下降 | 零参数更新数 |
| ---: | ---: | ---: | ---: |
| 2 | 1 | 98.8685% | 0 |

辅助比值定义为 $r^n/\exp(E(\theta^n)/(4E^0))$。Relax-PB-ESAV 的全部 Armijo 接受步比值为 1。

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
| 0.1 | [1.50633064e-01](../../results/quadratic/besav/fixed_lr/single/lr0p1/besav/) | [1.55169932e-01](../../results/quadratic/esav/fixed_lr/single/lr0p1/esav/) | [1.71506596e-01](../../results/quadratic/esav/fixed_lr/single/lr0p1/sav/) | [1.50416257e-01](../../results/quadratic/esav/fixed_lr/single/lr0p1/sgd/) | [2.24730408e-13](../../results/quadratic/esav/fixed_lr/single/lr0p1/adamw/) |
| 1 | [3.13464133e-06](../../results/quadratic/besav/fixed_lr/single/lr1/besav/) | [4.10400709e-05](../../results/quadratic/esav/fixed_lr/single/lr1/esav/) | [9.19417380e-02](../../results/quadratic/esav/fixed_lr/single/lr1/sav/) | [5.00000027e+01](../../results/quadratic/esav/fixed_lr/single/lr1/sgd/) | [3.74713664e-13](../../results/quadratic/esav/fixed_lr/single/lr1/adamw/) |
| 10 | [3.93672828e-05](../../results/quadratic/besav/fixed_lr/single/lr10/besav/) | [2.61888689e-05](../../results/quadratic/esav/fixed_lr/single/lr10/esav/) | [4.95173945e-01](../../results/quadratic/esav/fixed_lr/single/lr10/sav/) | [1.10236910e+306 †（119/300）](../../results/quadratic/esav/fixed_lr/single/lr10/sgd/) | [1.17586655e-10](../../results/quadratic/esav/fixed_lr/single/lr10/adamw/) |

![基础固定学习率](../../results/quadratic/besav/figures/single_fixed_lr.png)

### 4.3 基础 BESAV：Armijo

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| BESAV | [3.88146050e-05](../../results/quadratic/besav/armijo/single/besav/) | 300/300 |
| ESAV (scale) | [9.21369124e-07](../../results/quadratic/esav/armijo/single/esav/) | 300/300 |
| SAV | [4.53719836e-01](../../results/quadratic/esav/armijo/single/sav/) | 300/300 |
| SGD | [8.35462161e-07](../../results/quadratic/esav/armijo/single/sgd/) | 300/300 |
| AdamW | [2.24730408e-13](../../results/quadratic/esav/armijo/single/adamw/) | 300/300 |

![基础 Armijo](../../results/quadratic/besav/figures/single_armijo.png)

### 4.4 PB 与 Relax-PB：Armijo

#### M=2

| 方法 | 最后有效目标 | 更新数 |
| --- | ---: | ---: |
| PB-BESAV | [9.91112510e-147](../../results/quadratic/besav/armijo/m2/pb_besav/) | 300/300 |
| Relax-PB-BESAV | [1.01677386e-135](../../results/quadratic/besav/armijo/m2/relax_pb_besav/) | 300/300 |
| PB-ESAV (scale) | [2.86933412e-38](../../results/quadratic/esav/armijo/m2/pb_esav/) | 300/300 |
| Relax-PB-ESAV (scale) | [1.36948386e-38](../../results/quadratic/esav/armijo/m2/relax_pb_esav/) | 300/300 |
| PB-SAV | [3.23556733e-02](../../results/quadratic/esav/armijo/m2/pb_sav/) | 300/300 |
| Relax-PB-SAV | [1.12413968e-16](../../results/quadratic/esav/armijo/m2/relax_pb_sav/) | 300/300 |
| SGD | [8.35462161e-07](../../results/quadratic/esav/armijo/m2/sgd/) | 300/300 |
| AdamW | [2.24730408e-13](../../results/quadratic/esav/armijo/m2/adamw/) | 300/300 |

![M=2 优化目标](../../results/quadratic/besav/figures/pb_relax_m2.png)

补充单能量 Relax：[1.20397957e-06](../../results/quadratic/besav/armijo/m1/relax_pb_besav/)，完成 300/300 步。该记录以 PB 的 M=1 执行，与 Relax-BESAV 等价；作为补充记录，不新增未对齐的基线比较。

### 4.5 同等级 SAV 实验矩阵与判断

✓ 表示 BESAV 最后有效目标值低于对应 SAV，☐ 表示未超过；SAV 提前失败仍使用最后有效目标值。先判断是否超过同等级 SAV，已经超过时不继续排名；两者均未超过时再比较 BESAV 与尺度 ESAV。

| 实验 | 对照 | BESAV 超过对照 |
| --- | --- | :---: |
| 基础，固定学习率 0.1 | SAV | ✓ |
| 基础，固定学习率 1 | SAV | ✓ |
| 基础，固定学习率 10 | SAV | ✓ |
| 基础，Armijo | SAV | ✓ |
| PB，M=2 | PB-SAV | ✓ |
| Relax-PB，M=2 | Relax-PB-SAV | ✓ |

本表已对齐的基础、PB、Relax-PB 实验均超过对应 SAV；按既定优先级，不进一步用这些实验争论两种指数方案谁更好。

### 4.6 辅助诊断与可复现记录

BESAV 辅助比值为 $r^n/\exp(E^n/(E^n+E^0))$，耗散量为 $\log r^n$；Relax 的 Armijo 接受态比值为 1。不把不同映射的修正能量数值直接比较。

[本问题原始索引](../../results/quadratic/besav/index.json) · [定义与证明](../design/besav.md) · [归档及验证](../../results/reproduction_besav_20261024/README.md)。

诊断误报中止记录独立保存在复现资料中，不计入 23 组正式实验。初始能量求和顺序、元数据修正及各执行源码版本均保留原记录。

## 原始记录与索引

| 内容 | 位置 |
| --- | --- |
| naiveESAV / 旧版 ESAV 历史数据 | [esav_exploration/index.json](../../results/quadratic/esav_exploration/index.json) |
| MESAV 历史数据 | [mesav/index.json](../../results/quadratic/mesav/index.json) |
| ESAV 数据 | [esav/index.json](../../results/quadratic/esav/index.json) |
| BESAV 数据 | [besav/index.json](../../results/quadratic/besav/index.json) |
| 既有历史归档 | [归档清单](../../results/archive_20261022.json)、[复现资料](../../results/reproduction_20261022/README.md) |
| 历史设计推导 | [naiveESAV](../design/naive_esav.md)、[旧版 ESAV](../design/esav_legacy.md)、[MESAV](../design/mesav.md) |

原始配置保留实验时的方法名和公式；展示标签按本报告区分。表格与曲线全部读取已有完整实验记录，不重新优化；旧图保留，实验源码及检查材料见[复现资料](../../results/reproduction_20261024/README.md)。
