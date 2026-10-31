# ESAV 实验分组与证据

范围：[W&B NumericalScheme](https://wandb.ai/pheonizard-university-of-nottingham/NumericalScheme) 于 2026-10-27 18:20（America/New_York）的全部 56 次实验。二次型 10 次，非线性回归 46 次；全部写满更新预算。

## 写作格式

| 项目 | 沿用 example.md 的写法 |
| --- | --- |
| 问题 | 目标函数用数学表达式；初值、数据、网络、归一化列入共同参数表 |
| 分组 | 每组对应一个问题；同一分量数单独列表、绘图 |
| 结果 | 固定更新预算的最终目标；方法名称直接链接原始 run |
| 曲线 | 横轴为 iteration，纵轴为优化目标；同组方法共图 |
| 诊断 | 辅助比值另图；停滞、异常直接标注 |
| 判断 | 每张表后一句，限定于该组设置 |

`example.md`（未纳入版本库） 作为表达样例；其中历史数值与旧 PB-ESAV 状态说明不进入本次分组。

## 板块与覆盖范围

| 板块 | 对照表 | 独立实验数 | 要回答的问题 |
| --- | --- | --- | --- |
| A 二次型基础方法 | A1 | 6 | ESAV 能否持续降低目标；Relax 改变了什么 |
| B 二次型 PB 扩展 | B1 | 4 | PB-ESAV 在分裂二次型上是否获得更低目标 |
| C 非线性基础方法 | C1 | 6 | 基础 ESAV 的改善能否延续到神经网络优化 |
| D 非线性 PB 与 Relax | D1–D5 | 20 | 分量数与 Relax 如何影响结果；优势的边界在哪里 |
| E 非线性动量扩展 | E1–E5 | 20 | 相同动量设置下 ESAV 与 SAV 如何比较 |
| 合计 | 13 张对照表 | 56 | 每次 run 只计入一个主组；基线和诊断可复用 |

## 共同设置

| 参数 | 二次型 | 非线性回归 |
| --- | --- | --- |
| 更新预算 | $K=300$ | $K=600$ |
| 初始目标 | $50.5$ | $0.2791684612274234$（分组求和有末位差异） |
| 设备与精度 | CPU / float64 | CPU / float64 |
| 正则化 | $\lambda=0$ | $\lambda=0$ |
| 数据与初值 | $\theta^0=(1,\ldots,1)\in\mathbb R^{100}$ | $N=30$；$[0,1]$ 等距采样；噪声标准差 $0.1$，种子 1 |
| 网络与初始化 | 直接优化参数 | $1\to50\to1$，tanh；初始化种子 2 |
| PB 分组 | 奇数与偶数坐标，$M=2$ | 嵌套随机分组，种子 3；$M=2,4,8,16,30$ |
| SAV shift | $C=10^{-10}$ | $C=10^{-10}$ |
| ESAV 参考尺度 | $E^0=50.5$，冻结，无 shift | $E^0,E_i^0$ 冻结，无 shift |

$$
\begin{aligned}
\text{二次型：}\quad
f(\theta)&=\sum_{k=1}^{50}\theta_{2k-1}^2
+\frac1{100}\sum_{k=1}^{50}\theta_{2k}^2,\\
\text{非线性：}\quad
E(\theta)&=\frac1{2N}\sum_{j=1}^{N}
\bigl(u_\theta(x_j)-y_j\bigr)^2,\qquad
y_j=\sin(2\pi x_j)+\varepsilon_j,\quad
\varepsilon_j\sim\mathcal N(0,0.1^2).
\end{aligned}
$$

| 对照条件 | 设置 |
| --- | --- |
| A–D 的 SAV、ESAV、PB 与 SGD | Armijo：每轮从 100 开始，收缩因子 0.5，下限 $10^{-6}$，$c=10^{-4}$ |
| AdamW | 固定学习率 0.1；$\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$；weight decay 为 0 |
| E 的动量方法 | 固定学习率 0.1；动量系数 0.9，$\alpha=0.5$，缩放矩阵 $I$；完全 Relax 的 $\rho=1$ |
| 结果字段 | 使用完整 CSV 的最后目标；不将 best_objective 与最终目标混用 |
| ‡ | 记录写满预算，但最后 100 步目标无下降；辅助状态另见诊断表 |

SAV 与 ESAV 的辅助反馈时间层不同。下表支持已实现完整算法在对应设置下的比较，尚不能将差异单独归因于指数映射。公式见 [ESAV](../design/esav.md) 与 [ESAV 动量](../design/esav-moment.md)。

## A 二次型基础方法

$M=1$，$K=300$；比较辅助变量形式及 Relax。AdamW 作为固定学习率参考。

### A1 基础 ESAV 与 Relax

| 方法与原始记录 | Relax | 学习率 | 最终目标 |
| --- | --- | --- | --- |
| [SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/20c9bc28cfa3) | 关 | Armijo | $4.537198\times10^{-1}$ ‡ |
| [ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bbac1a11c98c) | 关 | Armijo | $5.842314\times10^{-7}$ |
| [SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/640958e349a7) | 开 | Armijo | $6.397809\times10^{-7}$ |
| [ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/df873b70f847) | 开 | Armijo | $8.354622\times10^{-7}$ |
| [SGD](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/81cd960259c2) | 关 | Armijo | $8.354622\times10^{-7}$ |
| [AdamW](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/8914085ecaec) | 关 | 固定 0.1 | $2.247304\times10^{-13}$ |

判断：ESAV 持续下降，SAV 停滞；Relax-SAV 恢复下降，Relax-ESAV 与 SGD 的目标轨迹在浮点精度内重合。

## B 二次型 PB 扩展

$M=2$，$K=300$；Armijo。与 A1 的基础方法比较 PB 的作用，复用 A1 的 SGD、AdamW 基线。

### B1 PB 与 Relax 的组合

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/1288824d6201) | 关 | $3.235567\times10^{-2}$ ‡ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/b6b4f3c6580c) | 关 | $8.282665\times10^{-39}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/d3b4d07c7275) | 开 | $1.124140\times10^{-16}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bfee6830c639) | 开 | $4.736856\times10^{-39}$ |

判断：PB-ESAV 的最终目标约为 $10^{-39}$；在本组开、关 Relax 的两对比较中均低于对应 PB-SAV。

## C 非线性基础方法

$M=1$，$K=600$；与 A1 保持相同方法组合，检验结果能否延续到非线性优化。

### C1 基础 ESAV 的非线性表现

| 方法与原始记录 | Relax | 学习率 | 最终目标 |
| --- | --- | --- | --- |
| [SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/98ab403b9a60) | 关 | Armijo | $2.274833\times10^{-1}$ ‡ |
| [ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/2977d4c255df) | 关 | Armijo | $1.185830\times10^{-1}$ |
| [SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/748b782c7154) | 开 | Armijo | $1.204126\times10^{-1}$ |
| [ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/f394ef48a942) | 开 | Armijo | $1.203259\times10^{-1}$ |
| [SGD](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/b4caea89eaf7) | 关 | Armijo | $1.203259\times10^{-1}$ |
| [AdamW](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/a1d48545dfa6) | 关 | 固定 0.1 | $2.489698\times10^{-3}$ |

判断：ESAV 的最终目标低于 SAV；基础 ESAV 与 Relax-ESAV 仍处于 $10^{-1}$ 量级，AdamW 达到 $10^{-3}$ 量级。

## D 非线性 PB 与 Relax

$K=600$；Armijo；每个 $M$ 固定数据、网络、初值与分组，对照四种组合。C1 的 SGD 与 AdamW 可作为各图共用基线。

### D1 分量数 2

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/fbb4aad536d5) | 关 | $2.167684\times10^{-1}$ ‡ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5c8532a1e995) | 关 | $1.179550\times10^{-1}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5ff770f6618c) | 开 | $1.207992\times10^{-1}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bb4c930fc9d6) | 开 | $1.203632\times10^{-1}$ |

判断：关闭 Relax 时 PB-ESAV 更低；开启 Relax 没有降低 PB-ESAV 的最终目标。

### D2 分量数 4

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/908af20a4137) | 关 | $2.112232\times10^{-1}$ ‡ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/a1f98c1c8160) | 关 | $1.181725\times10^{-1}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/42e98dfb423b) | 开 | $1.106565\times10^{-1}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/cfa7a3cc86c1) | 开 | $1.202907\times10^{-1}$ |

判断：关闭 Relax 时 PB-ESAV 更低；开启 Relax 后 PB-SAV 更低，方法排序发生改变。

### D3 分量数 8

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/eb2848ac1045) | 关 | $1.704239\times10^{-1}$ ‡ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/ddb3a1415a5c) | 关 | $1.192191\times10^{-1}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5987a9a61dd0) | 开 | $2.616874\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/3a940eccc41a) | 开 | $1.196914\times10^{-1}$ |

判断：Relax-PB-SAV 达到 $2.62\times10^{-3}$，Relax-PB-ESAV 仍为 $1.20\times10^{-1}$；本组不支持 ESAV 在 Relax 后更优。

### D4 分量数 16

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/c35d2e4e83fb) | 关 | $5.098777\times10^{-2}$ ‡ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5ab903f3df71) | 关 | $1.455871\times10^{-2}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5eda45981333) | 开 | $2.524810\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/58b1d468b5e6) | 开 | $7.449043\times10^{-3}$ |

判断：PB-ESAV 降至 $1.46\times10^{-2}$，Relax 后进一步下降；对应 Relax-PB-SAV 的最终目标仍更低。

### D5 分量数 30

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/40f912e75865) | 关 | $6.387099\times10^{-3}$ ‡ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/24ef5f8af31c) | 关 | $3.493109\times10^{-3}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/00c28cf1a34e) | 开 | $1.572644\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/c6bc56347ca7) | 开 | $3.124943\times10^{-3}$ |

判断：PB-ESAV 与 Relax-PB-ESAV 均达到 $10^{-3}$ 量级；Relax-PB-SAV 在本组取得更低目标。

## E 非线性动量扩展

$K=600$；固定学习率 0.1，动量系数 0.9，$\alpha=0.5$，缩放矩阵 $I$。每组比较相同动量配置下的 SAV、ESAV 及其 Relax。

### E1 分量数 2

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/f9a954df51ca) | 关 | $1.153312\times10^{-1}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/24019ed86d38) | 关 | $3.228674\times10^{-3}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/83ec9760b0d1) | 开 | $3.126172\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/de017e333e33) | 开 | $3.134695\times10^{-3}$ |

判断：关闭 Relax 时 PB-SAV 为 $10^{-1}$ 量级，PB-ESAV 为 $10^{-3}$ 量级；Relax 后两者接近，PB-SAV 略低。

### E2 分量数 4

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/a71e47e477d5) | 关 | $1.130872\times10^{-1}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/ec740acccad4) | 关 | $3.225909\times10^{-3}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/311e0d1cde20) | 开 | $3.125702\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bd604d57dc43) | 开 | $3.134441\times10^{-3}$ |

判断：关闭 Relax 时 PB-ESAV 明显更低；Relax 后两者接近，PB-SAV 略低。

### E3 分量数 8

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/873df0bf8cc0) | 关 | $3.453054\times10^{-2}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/b6a804c7552a) | 关 | $3.199261\times10^{-3}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bae041585253) | 开 | $3.122640\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/8eb0c036f0f1) | 开 | $3.132205\times10^{-3}$ |

判断：关闭 Relax 时 PB-ESAV 约低一个数量级；Relax 后 PB-SAV 略低。

### E4 分量数 16

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/a2c6a0a67046) | 关 | $1.886777\times10^{-2}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/877b3447b2c4) | 关 | $3.191682\times10^{-3}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/0205570c823c) | 开 | $3.117632\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/617ec7e64ea9) | 开 | $3.131021\times10^{-3}$ |

判断：关闭 Relax 时 PB-ESAV 更低；Relax 后 PB-SAV 略低。

### E5 分量数 30

| 方法与原始记录 | Relax | 最终目标 |
| --- | --- | --- |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/b63e430be88d) | 关 | $7.379412\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/6e7a8d07ccd9) | 关 | $2.828360\times10^{-3}$ |
| [PB-SAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/09b8192545ac) | 开 | $3.107371\times10^{-3}$ |
| [PB-ESAV](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/571db84cacda) | 开 | $2.777804\times10^{-3}$ |

判断：开、关 Relax 时 PB-ESAV 均低于对应 PB-SAV；本组 Relax-PB-ESAV 为全部动量 ESAV 记录中的最低最终目标。

C1 的 AdamW 使用相同固定学习率 0.1，最终目标为 $2.489698\times10^{-3}$，可复用为参考。当前没有相同固定学习率与动量系数的 SGD 记录；D 与 E 同时改变学习率策略、动量和 PB 强度，不能据此单独量化动量的贡献。

## 辅助变量与停滞诊断

复用上述实验；用于解释目标曲线，不增加独立 run 数。下表均关闭 Relax。

$$
\text{SAV 辅助比值}=\frac{q^n}{\sqrt{E^n+C}},\qquad
\text{ESAV 辅助比值}=\frac{r^n}{\exp(E^n/(4E^0))}.
$$

| 问题 | $M$ | SAV 目标不变步数 | SAV 最终辅助比值 | ESAV 目标不变步数 | ESAV 最终辅助比值 |
| --- | --- | --- | --- | --- | --- |
| 二次型 | 1 | 270 | $3.230555\times10^{-151}$ | 0 | 0.471126 |
| 二次型 | 2 | 267 | $5.824034\times10^{-146}$ | 0 | 0.276417 |
| 非线性 | 1 | 591 | $8.201490\times10^{-322}$ | 0 | 0.729378 |
| 非线性 | 2 | 591 | $9.387247\times10^{-323}$ | 0 | 0.715786 |
| 非线性 | 4 | 588 | $-3.458460\times10^{-323}$ | 0 | 0.725160 |
| 非线性 | 8 | 562 | $1.151160\times10^{-235}$ | 0 | 0.486195 |
| 非线性 | 16 | 539 | $1.344485\times10^{-177}$ | 0 | 0.339387 |
| 非线性 | 30 | 524 | $2.841150\times10^{-128}$ | 0 | 0.412001 |

非线性 SAV、PB-SAV（$M=2,4$）分别从第 134、169、224 步首次出现非正辅助值；目标与辅助状态按原 CSV 保留。目标值不变步数不等同于零参数更新数。

## 报告中的图像安排

| 图组 | 实验来源 | 图像内容 | 对应论证 |
| --- | --- | --- | --- |
| 基础方法 | A1、C1 各一图 | 优化目标随更新次数变化；可另附辅助比值图 | 持续下降与停滞 |
| 二次型 PB | B1 与 A1 复用基线 | PB 四种组合的目标曲线 | PB 扩展的作用 |
| 非线性 PB | D1–D5 各一图 | 同一 $M$ 的四种组合与共用基线 | 分组和 Relax 的收益与边界 |
| 非线性动量 | E1–E5 各一图 | 同一 $M$ 的四种组合与 AdamW 参考 | 与动量结构组合后的表现 |

## 证据范围

| 论点 | 当前证据 |
| --- | --- |
| 关闭 Relax 的 ESAV 系列优于对应 SAV | A1、B1、C1、D1–D5、E1–E5 的 13 对最终目标均更低，限本次设置 |
| Relax 后仍全面优于 SAV | 不成立：二次型基础组、D2–D5、E1–E4 出现相反排序 |
| ESAV 与动量结构可以组合 | E1–E5 均完成 600 步并降低目标；独立动量收益尚缺同条件无动量对照 |
| 跨随机种子的稳定结论 | 非线性目前只有一套数据、初始化和分组种子 |
| BESAV、naiveESAV、MESAV、Darcy | 当前 W&B 清单中均为 0 次；不将 example.md 的历史链接计入本次在线实验 |

## 原始记录

| 内容 | 位置 |
| --- | --- |
| 在线实验 | 各结果表的方法链接 |
| 全部配置与在线摘要快照 | [inventory.json](../../TMP/experiment_report_audit/inventory.json) |
| 56 份完整 CSV 与配置 | [records](../../TMP/experiment_report_audit/records/) |
| 对应源码文件 | [source](../../TMP/experiment_report_audit/source/) |
| 分组与覆盖清单 | [groups.json](../../TMP/experiment_report_audit/groups.json) |
| 记录核对与诊断 | [audit.json](../../TMP/experiment_report_audit/audit.json) |

56 份 CSV 均从 iteration=0 连续记录至预算末步，共 30,656 行。读取的算法、问题、初始化与训练入口源码与当前对应文件一致。
