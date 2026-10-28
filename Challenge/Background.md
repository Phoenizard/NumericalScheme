# Background：ESAV Relax 研究背景与共同输入

原题编制日期：2026-10-28。本次仅按用户新增规则更新题面，未追加研究、实验或新算法。

## 1. 本题用途与优先规则

本题向不同 AI 提供同样的研究问题，通过解决问题的质量，辅助用户决定下个月继续使用 OpenAI 还是采用 Claude。评价依据是具体研究成果，不要求证明任何品牌优越。满足硬性要求后，实验效果优先于文档写作；不设置人为分数或权重。

三份共同题面为本文件、[Question.md](Question.md) 和 [Acceptance_Criteria.md](Acceptance_Criteria.md)。它们由出题方维护，参赛模型只能读取。新增规则优先于原材料中的项目运行惯例：本次挑战要求独立、本地实验，禁止 W&B；不沿用在线记录或等待另行实验批准的旧规定。

### 1.1 研究与交付的硬性要求

基本版本必须满足无条件辅助能量耗散，并在 `Handoff/Design.md` 中严格证明。对现行 ESAV 定义，须明确讨论对应的 $4E^0\ln r$；新辅助定义必须写明其实际辅助能量。“无条件”指在明确的数学假设下，对任意正学习率成立，不靠学习率上界、回溯接受条件或实验观察补足证明。仅证明带动能的组合量耗散，不满足基本版本的这一要求；动量扩展可另列性质。

必须交付基于实际执行的本地实验的 `Handoff/Result.md`。实验仅限本文的二次型和非线性回归；具体超参数、控制变量、预算与失败记录由模型科学设计。既有结果是背景，不是新方案已运行的证据。不预设真实目标无条件下降、任意学习率下参数稳定、普遍加速或所有配置获胜。

可以批评当前 ESAV 设计，但必须同时给出详细理由、实际结果、合理替代设计及其对应 Relax；替代设计仍须满足硬性要求。只提出批评或不可能性论证，不能代替本题所需的设计和实验交付。

### 1.2 独立工作目录

项目根目录为 `/Users/phoenizard/Workplace/NumericalScheme`。OpenAI 的唯一工作根目录是 `Challenge/OpenAI`，Claude 的唯一工作根目录是 `Challenge/CLAUDE`。下列 `Design.md`、`Result.md` 是待交付文件；目录骨架不代表这些成果已经完成。

```text
Challenge/
├── Background.md                 # 共同只读题面
├── Question.md                   # 共同只读题面
├── Acceptance_Criteria.md        # 共同只读题面
├── OpenAI/
│   ├── Handoff/
│   │   ├── Design.md
│   │   ├── Result.md
│   │   └── fig/                  # Markdown 引用的全部图片
│   ├── TmpResearchSpace/         # 独立代码、配置、验证与实验脚本
│   └── TmpData/                  # 输入副本、缓存、本地日志、原始结果
└── CLAUDE/
    ├── Handoff/
    │   ├── Design.md
    │   ├── Result.md
    │   └── fig/
    ├── TmpResearchSpace/
    └── TmpData/
```

两方不得读取、写入、执行或借用对方目录，也不得修改三个共同题面。除读取三份共同题面、§1.3 的有限只读参考、只读互联网搜索及 §1.5 的必要 Git 元数据操作外，不在各自工作目录之外操作。所有运行代码、配置、数据副本、缓存、临时文件、日志、结果和图都放在己方目录。临时目录与工具缓存显式指向己方 `TmpData/`，不得写入项目 `TMP/` 或系统临时目录。

代码和试验脚本放在 `TmpResearchSpace/`，数据、本地日志、原始结果放在 `TmpData/`。最终两文档放在 `Handoff/`。所有 Markdown 使用的图放在 `Handoff/fig/`，以相对路径引用；例如最终文档使用 `fig/example.png`。运行命令和重现说明也用己方目录内的相对路径。

### 1.3 代码、只读参考与环境边界

现有 ESAV、SAV 及其基线定义冻结。不得修改原仓库算法；独立副本也不得悄悄更改基线定义、时间层、初始化或 Relax 规则以取得优势。新 Relax、替代 ESAV 及其 Relax 必须另立名称和实现，与冻结基线清楚区分。

需要运行时，先在己方 `TmpResearchSpace/` 独立抄写或复制所需代码并核对。不得 import、动态加载、符号链接或调用原仓库研究代码；也不得执行原仓库入口。普通已安装第三方依赖可以使用，不属于借用仓库算法。不得安装新依赖或修改目录外环境。

唯一的目录外只读例外，是准备独立副本或参考文风所需的下列已有材料。优先使用本题已给出的内容，仅在确有需要时读取指定项；不执行原代码，不写原文件，不扩展为扫描仓库、读取其他历史或访问对方成果。

| 目的 | 允许的有限只读材料（路径均相对项目根目录） |
| --- | --- |
| 基线与更新定义 | `doc/design/esav.md`、`doc/design/ESAV-Moment.md`；所需的 `optimizer/esav.py`、`optimizer/pb_esav.py`、`optimizer/sav.py`、`optimizer/pb_sav.py`、`optimizer/sgd.py`、`optimizer/adamw.py`；研究动量时再读 `optimizer/pb_esav_momentum.py`、`optimizer/pb_sav_momentum.py`；控制器参考限 `train.py` |
| 当前两个问题及输入副本 | `problem/quadratic.py`、`problem/nonlinear_regression.py`、`network/nonlinear_regression.py`、`problem/config.py` 中这两个问题的设置，以及 `data/nonlinear_regression.pt`；实际运行使用己方目录内的独立副本或独立生成的数据 |
| 文档风格 | 上述两个设计文档及 `doc/experiments/esav_experiments.md`、`doc/experiments/esav_experiment_groups.md`；只参考简洁文字、数学表达式、表格和图像的组织方式 |

允许互联网搜索和在线阅读，仅用于查阅信息。不得通过搜索执行外部操作、上传或发布内容，也不得下载或拷贝大量文件、批量抓取资料或镜像网站。可在己方文档中记录来源链接和必要的简短摘录；本地文件访问仍遵守上述范围。

全部实验在本地运行并保存。禁止 W&B，以及替代性的在线实验追踪、结果上传或发布。本文保留的历史 run 链接只作来源标识，不授权访问在线实验账户或沿用历史上传流程。出题方单独维护共同题面的 Library 版本，不是参赛实验的保存渠道。

### 1.4 开始前的交流

每个模型在开始研究前最多与用户交流两轮；每轮可集中确认必要信息。不必问满两轮。两轮后不得依赖额外答复；未决条件须写成显式假设，且不能用假设扩大权限。若条件不足以完成硬性要求，披露具体缺口及未完成部分，不将其包装为已通过验收。

### 1.5 参赛分支与违规处理

参赛模型开始工作时必须自行创建并使用各自的独立 Git 分支，禁止在 main 上开展研究或提交，不得修改、合并或重置 main，也不得操作对方分支。此要求针对参赛阶段；共同题面由出题方在 main 维护。必要的 Git 元数据操作是工作目录限制的有限例外，只能服务于己方分支和允许修改的文件。不同分支不能隔离同一个工作区；不得在同一检出目录并行切换分支，应串行工作或使用出题方提供的独立检出。

任何违规情况均会被大幅扣分，包括但不限于：未说明理由而改变 ESAV 递推框架、擅自改变冻结的 SAV 递推框架、不满足无条件辅助能量耗散、干扰 main 或对方工作，以及违反目录或本地运行规定。允许有理由地提出另立名称的新方案，但须说明相对原式的改变、理由和依据，并完成所需证明与实验；这不授权修改冻结基线。不预设具体扣分数值。

## 2. 项目背景与阶段性判断

NumericalScheme 探索辅助变量数值优化方法，不预设算法优势、新颖性或可发表性。基础 SAV 是比较基线；现行 ESAV 与 BESAV 有阶段性推导和实验。项目另存 naiveESAV、MESAV 和旧版 ESAV 等历史工作，本文不混用它们的定义或实验。项目近期另有 Darcy 算子学习计划，但本题的数值依据来自二次型与非线性回归。

2026-10-28 01:39 UTC，指定对话“归类 ESAV 实验并规划报告”的最后一轮将以下结论写入 `doc/experiments/esav_experiments.md`：

> 在当前二次型与非线性回归实验中，ESAV 在对应的基础、PB 及动量配置下，相较未加 Relax 的 SAV 表现出更好的优化效果，显示出进一步研究的潜力。相比之下，Relax 对 ESAV 的改善有限，部分配置下反而使最终目标增大；其增益明显小于 Relax 对 SAV 的改善。因此，现有结果支持 ESAV 本身的有效性，但尚未显示 Relax-ESAV 具有稳定的额外优势。
>
> 上述比较针对完整算法配置；现有实验尚不能将优势单独归因于指数辅助变量。

这是本题需要进一步解释和检验的阶段性判断，不是要求研究者证明的既定结论。特别要保留 Relax 确实改善 ESAV 的配置，不能把“收益不稳定”改写成“所有配置均失败”。

## 3. 已知数学格式与实现

### 3.1 记号、来源与适用范围

沿用项目记号：参数为 $\theta\in\mathbb R^d$，真实目标为 $E(\theta)$，$E^n=E(\theta^n)$，$\nabla E^n=\nabla E(\theta^n)$，$\Delta\theta=\theta^{n+1}-\theta^n$。无动量格式使用正学习率 $\Delta t$；动量文档使用 $\eta$。$M$ 是能量分量数，$\mathbf M^n$ 是动量缩放矩阵，二者不同。

| 项目 | 本题定义 |
| --- | --- |
| 目标假设 | $E\in C^1$，$E(\theta)\ge0$，$E^0=E(\theta^0)>0$ |
| ESAV | $r=\exp(E/(4E^0))$；$E^0$ 初始化后冻结，无 shift |
| PB-ESAV | $E=\sum_{i=1}^M E_i$；$E_i\in C^1$、$E_i\ge0$、$E_i^0>0$；$r_i=\exp(E_i/(4E_i^0))$；各 $E_i^0$ 冻结 |
| 初始辅助状态 | $\ln r^0=\ln r_i^0=1/4$ |
| SAV 对照 | $r=\sqrt{E+C}$；本次实验 $C=10^{-10}$ |
| 不属于本题的同名或近名方案 | naiveESAV 为 $r=\exp(E)$；已弃用旧 ESAV 为 $r=\exp(E)\sqrt{E+C}$；MESAV 为 $r=\exp(E/(E^0+C))\sqrt{E+C}$ |

原始缩放指数构造见 Liu 与 Li，*The exponential scalar auxiliary variable (E-SAV) approach for phase field models and its explicit computing*，arXiv:1912.09263v2，§4，第 9 页，式 (4.1)–(4.3)。原文定义为 $r=\exp(E_1/C)$，这里的原文 $C$ 是指数尺度。项目令原文的线性算子为零、流算子为 $-I$，将整体优化目标对应到原文 $E_1$，并选取指数尺度 $4E^0$。这个尺度选择是项目设置，不是原文规定的唯一值，也不是 SAV 的加法 shift。[原文](https://arxiv.org/abs/1912.09263v2)

以下基础连续式和离散式已与上述原文、`doc/design/esav.md`、`optimizer/esav.py` 核对。PB 的全局驱动替换、分量指数一阶展开，以及 ESAV 的动量与 Relax 是项目扩展，不能直接称为文献原版 RE-SAV。

### 3.2 基础 ESAV

连续扩展系统为

$$
\dot\theta=-\frac{r}{\exp(E(\theta)/(4E^0))}\nabla E(\theta),
\qquad
\frac{d\ln r}{dt}
=\frac1{4E^0}\frac{r}{\exp(E(\theta)/(4E^0))}
\nabla E(\theta)^\top\dot\theta.
$$

在解存在且一致关系保持的条件下，一致初始化恢复 $r=\exp(E/(4E^0))$ 和原梯度流 $\dot\theta=-\nabla E$。其辅助能量满足

$$
\frac{d(4E^0\ln r)}{dt}
=-\left[\frac{r}{\exp(E/(4E^0))}\right]^2\|\nabla E\|^2.
$$

实际一阶格式在两条方程中均使用旧辅助比值、旧能量和旧梯度：

$$
\theta^{n+1}
=\theta^n-\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n,
$$

$$
\ln r^{n+1}
=\ln r^n+\frac1{4E^0}\frac{r^n}{\exp(E^n/(4E^0))}
(\nabla E^n)^\top\Delta\theta
=\ln r^n-\frac{\Delta t}{4E^0}
\left[\frac{r^n}{\exp(E^n/(4E^0))}\right]^2\|\nabla E^n\|^2.
$$

参数先更新，再更新对数辅助状态。精确算术下、各量有限时，已有恒等式为

$$
4E^0\ln\frac{r^{n+1}}{r^n}
=-\frac{\|\Delta\theta\|^2}{\Delta t}\le0.
$$

这个结论控制 $4E^0\ln r$，没有同时证明真实目标 $E$ 逐步下降、迭代收敛或辅助比值远离零。

### 3.3 现有基础 Relax

现有 Relax 保留上面的参数步，随后以新参数的真实目标替换辅助更新：

$$
\ln r^{n+1}
=\min\left\{\frac{E(\theta^{n+1})}{4E^0},\ \ln r^n\right\}.
$$

右侧第二项是旧状态，不是未 Relax 候选状态。这个公式不是在两个 $r$ 值之间作线性插值。只要新目标有限，辅助能量非增。

现有设计还给出条件性质：若 $r^n=\exp(E^n/(4E^0))$ 且 $E^{n+1}\le E^n$，则校正后仍一致。由此，在一致初始化且每步真实目标非增时，基础 Relax-ESAV 的参数步就是普通梯度步。第 4 节给出了与全量 SGD 的轨迹核对；这一现象不能未经分析地推广到 PB 或动量。

### 3.4 PB-ESAV 与现有 Relax-PB-ESAV

PB 推导先令参数方程使用新的分量辅助变量，而对数辅助方程的系数取旧状态；将得到的分量指数反馈作一阶展开，再把驱动项改为总辅助比值乘总梯度。最终实际更新为

$$
\mathbf B^n
=\sum_{i=1}^M
\left[\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}\right]^2
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{4E_i^0},
$$

$$
\theta^{n+1}
=\theta^n-\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
(I+\Delta t\,\mathbf B^n)^{-1}\nabla E^n,
$$

$$
\ln r^{n+1}
=\ln r^n+\frac1{4E^0}\frac{r^n}{\exp(E^n/(4E^0))}
(\nabla E^n)^\top\Delta\theta,
$$

$$
\ln r_i^{n+1}
=\ln r_i^n+\frac1{4E_i^0}\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top\Delta\theta.
$$

由 $\mathbf B^n\succeq0$，参数线性系统唯一可解，已有总辅助耗散恒等式为

$$
4E^0\ln\frac{r^{n+1}}{r^n}
=-\frac{\|\Delta\theta\|^2}{\Delta t}
-\Delta\theta^\top\mathbf B^n\Delta\theta\le0.
$$

各分量对数状态不要求分别单调。现有 Relax-PB-ESAV 只把总 $\ln r^{n+1}$ 替换成 §3.3 的 `min` 公式，分量 $\ln r_i^{n+1}$ 仍按上述递推。它没有逐步强制分量辅助一致性。

现行 PB 实现对所有分量数使用这套矩阵公式，$M=1$ 时也保留矩阵修正，因此不能把 PB-ESAV 的 $M=1$ 自动当成显式基础 ESAV。实现使用分量梯度矩阵的 reduced QR 和至多 $M$ 阶的系统，不显式构造完整 $d\times d$ 矩阵。若分量辅助比值的指数计算恰好下溢到零，实现用 `torch.finfo(dtype).tiny` 替代；总比值非正或非有限会报错。数学推导与这一浮点处理应分开评价。

### 3.5 动量扩展与现有 Relax 预算

动量位置借鉴 arXiv:2609.13569v1 §3.2.2 式 (25)–(29)，动能借鉴 §3.3 式 (38)–(42)，完全 Relax 预算借鉴 §3.4 式 (45)。使用 ESAV 的辅助映射和下述矩阵后，是项目的 ESAV 扩展。[原文](https://arxiv.org/abs/2609.13569v1)

令 $p^0=0$、固定 $\eta>0$、$0\le\beta<1$、$0\le\alpha\le1$，且 $0\prec\mathbf M^{n+1}\preceq\mathbf M^n$。项目取 $\mathbf B_\alpha^n=\alpha\mathbf B^n$，并按顺序计算

$$
\left[(\mathbf M^{n+1})^{-1}+\eta\mathbf B_\alpha^n\right]\Delta\theta
=\beta p^n-\eta\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n,
\qquad
p^{n+1}=(\mathbf M^{n+1})^{-1}\Delta\theta.
$$

未 Relax 的总对数增量与分量对数增量仍用 §3.4 的辅助方程，且都使用实际位移 $\Delta\theta$。总辅助能量与动能组成的量为

$$
4E^0\ln r^n+\frac{(p^n)^\top\mathbf M^np^n}{2\eta}.
$$

现有完全 Relax 仅校正总状态：

$$
4E^0\ln r^{n+1}
=\min\left\{
E(\theta^{n+1}),\quad
4E^0\ln r^n+
\frac{(p^n)^\top\mathbf M^np^n
-(p^{n+1})^\top\mathbf M^{n+1}p^{n+1}}{2\eta}
\right\}.
$$

已有项目推导证明上述条件下组合能量非增，不保证真实目标或总辅助变量分别非增。分量状态仍不 Relax。参数求解的 QR 子空间还纳入右端，系统至多为 $M+1$ 阶，以保留历史动量在当前分量梯度空间之外的部分。

本题引用的动量实验全部使用 $\mathbf M^n=I$。自适应缩放已有实现，但这些结果不提供其额外收益的证据。新方案若覆盖变学习率或其他缩放，须另给适用条件，不能直接套用固定学习率结论。

### 3.6 SAV 对照的实际差异

为解释比较结果，SAV / PB-SAV 当前无动量更新可写为

$$
\left[I+\Delta t\sum_{i=1}^{M}
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{2(E_i^n+C/M)}\right]\Delta\theta
=-\Delta t\,\frac{q^n}{\sqrt{E^n+C}}\nabla E^n,
$$

$$
q^{n+1}=q^n+
\frac{(\nabla E^n)^\top\Delta\theta}{2\sqrt{E^n+C}}.
$$

基础 SAV 取整体能量，即 $M=1$。打开 Relax 后，最后一式替换为

$$
q^{n+1}=\sqrt{\min\{E(\theta^{n+1})+C,(q^n)^2\}}.
$$

SAV 基础格式消元前使用新辅助变量反馈，ESAV 基础格式使用旧辅助比值，二者时间层不同。动量 SAV 的矩阵则是总梯度外积矩阵与分量矩阵的凸组合：

$$
(1-\alpha)\frac{\nabla E^n(\nabla E^n)^\top}{2(E^n+C)}
+\alpha\sum_{i=1}^{M}
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{2(E_i^n+C/M)}.
$$

它不同于 ESAV 的 $\alpha\mathbf B^n$。因此“同级配置”表示匹配问题、预算、分组及所列控制参数，不表示只改变了辅助变量映射。

## 4. 实验依据

### 4.1 问题、初始化与学习率控制

| 设置 | 二次型 | 非线性回归 |
| --- | --- | --- |
| 参数与预算 | $d=100$；$K=300$ 次接受更新 | $1\to50\to1$ tanh 网络，含两层偏置，共 151 个参数；$K=600$ |
| 初值或数据 | $\theta^0=(1,\ldots,1)$ | 30 个等距样本；全量训练 |
| 初始目标 | $50.5$ | $0.2791684612274234$，分组求和有末位差异 |
| 分组 | 奇数、偶数坐标两组，$M=2$ | 固定嵌套随机分组，$M=2,4,8,16,30$ |
| 随机种子 | 确定性初值 | 噪声 1、网络初始化 2、分组 3 |
| 正则化 | $\lambda=0$ | $\lambda=0$ |
| 设备与精度 | CPU、`torch.float64`、4 线程 | CPU、`torch.float64`、4 线程 |

二次型：

$$
E(\theta)=\sum_{k=1}^{50}\theta_{2k-1}^2
+\frac1{100}\sum_{k=1}^{50}\theta_{2k}^2.
$$

非线性回归：

$$
x_j=\frac{j-1}{29},\qquad
y_j=\sin(2\pi x_j)+\varepsilon_j,\qquad
\varepsilon_j\sim\mathcal N(0,0.1^2),\qquad j=1,\ldots,30,
$$

$$
E(\theta)=\frac1{2N}\sum_{j=1}^{N}(u_\theta(x_j)-y_j)^2,\qquad N=30.
$$

各分量也按总样本数归一化：第 $i$ 组只对其样本求和，但分母仍为 $2N$。噪声由 CPU `torch.Generator().manual_seed(1)` 和 `torch.randn((30,1), dtype=torch.float64)` 生成；保存的数据记录 PyTorch 版本为 2.14.1。网络用独立 seed 2 的生成器，按隐藏层权重、隐藏层偏置、输出层权重、输出层偏置的顺序初始化；隐藏层均匀分布于 $[-1,1]$，输出层均匀分布于 $[-1/\sqrt{50},1/\sqrt{50}]$。分组先以 seed 3 调用 `torch.randperm(30)`，从单组开始，反复取当前首个最大组，用 `tensor_split(2)` 二分，直至所需组数。

| 实验块 | 学习率及控制 |
| --- | --- |
| A–D 的 SAV、ESAV、PB、全量 SGD | Armijo；每轮重新从 100 开始，拒绝后乘 0.5，最低试到 $10^{-6}$；$c=10^{-4}$；无动量 |
| A、C 的 AdamW | 固定学习率 0.1；$\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$；weight decay 0 |
| E 的动量方法 | 固定学习率 0.1；$\beta=0.9$；PB 强度 $\alpha=0.5$；$\mathbf M=I$；完全 Relax |

Armijo 的接受条件使用实际候选位移：

$$
(\nabla E^n)^\top\Delta\theta\le0,
\qquad
E(\theta^n+\Delta\theta)
\le E^n+10^{-4}(\nabla E^n)^\top\Delta\theta.
$$

候选参数和辅助状态须有限。拒绝后恢复参数、保留旧辅助状态、按新学习率重新计算候选；拒绝试探不计入更新预算。最低学习率仍不接受则报错。基础和 PB 方法共用控制规则，但接受学习率和试探次数可以不同。项目的 SGD 在这些实验中使用全量梯度，不是小批随机梯度。

### 4.2 无动量最终目标

以下为各次运行最后一行的真实目标，不是历史最小值。数字均与保存的完整 CSV 核对；每个数字链接相应原始 run。A1、C1 为基础方法，其余为 PB。所有运行写满预算。符号 ‡ 表示最后 100 步目标无下降，不表示提前中止。

| 组别 | 分量数 | SAV，Relax 关 | SAV，Relax 开 | ESAV，Relax 关 | ESAV，Relax 开 |
| --- | --- | --- | --- | --- | --- |
| A1 二次型 | $1$ | [$4.537198\times10^{-1}$ ‡](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/20c9bc28cfa3) | [$6.397809\times10^{-7}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/640958e349a7) | [$5.842314\times10^{-7}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bbac1a11c98c) | [$8.354622\times10^{-7}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/df873b70f847) |
| B1 二次型 | $2$ | [$3.235567\times10^{-2}$ ‡](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/1288824d6201) | [$1.124140\times10^{-16}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/d3b4d07c7275) | [$8.282665\times10^{-39}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/b6b4f3c6580c) | [$4.736856\times10^{-39}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bfee6830c639) |
| C1 非线性 | $1$ | [$2.274833\times10^{-1}$ ‡](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/98ab403b9a60) | [$1.204126\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/748b782c7154) | [$1.185830\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/2977d4c255df) | [$1.203259\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/f394ef48a942) |
| D1 非线性 | $2$ | [$2.167684\times10^{-1}$ ‡](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/fbb4aad536d5) | [$1.207992\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5ff770f6618c) | [$1.179550\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5c8532a1e995) | [$1.203632\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bb4c930fc9d6) |
| D2 非线性 | $4$ | [$2.112232\times10^{-1}$ ‡](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/908af20a4137) | [$1.106565\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/42e98dfb423b) | [$1.181725\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/a1f98c1c8160) | [$1.202907\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/cfa7a3cc86c1) |
| D3 非线性 | $8$ | [$1.704239\times10^{-1}$ ‡](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/eb2848ac1045) | [$2.616874\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5987a9a61dd0) | [$1.192191\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/ddb3a1415a5c) | [$1.196914\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/3a940eccc41a) |
| D4 非线性 | $16$ | [$5.098777\times10^{-2}$ ‡](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/c35d2e4e83fb) | [$2.524810\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5eda45981333) | [$1.455871\times10^{-2}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/5ab903f3df71) | [$7.449043\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/58b1d468b5e6) |
| D5 非线性 | $30$ | [$6.387099\times10^{-3}$ ‡](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/40f912e75865) | [$1.572644\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/00c28cf1a34e) | [$3.493109\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/24ef5f8af31c) | [$3.124943\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/c6bc56347ca7) |

### 4.3 动量最终目标

下表均为非线性问题，600 次更新；设置见 §4.1。

| 组别 | 分量数 | PB-SAV，Relax 关 | PB-SAV，Relax 开 | PB-ESAV，Relax 关 | PB-ESAV，Relax 开 |
| --- | --- | --- | --- | --- | --- |
| E1 | $2$ | [$1.153312\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/f9a954df51ca) | [$3.126172\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/83ec9760b0d1) | [$3.228674\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/24019ed86d38) | [$3.134695\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/de017e333e33) |
| E2 | $4$ | [$1.130872\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/a71e47e477d5) | [$3.125702\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/311e0d1cde20) | [$3.225909\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/ec740acccad4) | [$3.134441\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bd604d57dc43) |
| E3 | $8$ | [$3.453054\times10^{-2}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/873df0bf8cc0) | [$3.122640\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/bae041585253) | [$3.199261\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/b6a804c7552a) | [$3.132205\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/8eb0c036f0f1) |
| E4 | $16$ | [$1.886777\times10^{-2}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/a2c6a0a67046) | [$3.117632\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/0205570c823c) | [$3.191682\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/877b3447b2c4) | [$3.131021\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/617ec7e64ea9) |
| E5 | $30$ | [$7.379412\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/b63e430be88d) | [$3.107371\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/09b8192545ac) | [$2.828360\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/6e7a8d07ccd9) | [$2.777804\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/571db84cacda) |

| 共同参照 | 二次型最终目标 | 非线性最终目标 |
| --- | --- | --- |
| 全量 SGD，无动量，Armijo | [$8.354622\times10^{-7}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/81cd960259c2) | [$1.203259\times10^{-1}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/b4caea89eaf7) |
| AdamW，固定学习率 0.1 | [$2.247304\times10^{-13}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/8914085ecaec) | [$2.489698\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/a1d48545dfa6) |
| 全量 SGD，动量 0.9，固定学习率 0.1 | 本题无对应记录 | [$3.137466\times10^{-3}$](https://forge.coreweave.com/wandb/pheonizard-university-of-nottingham/NumericalScheme/runs/8cffc061330b) |

D 与 E 同时改变了学习率控制、动量和 PB 强度，不能仅凭两表差异认定动量的独立贡献。AdamW 是不同完整优化器的参考，不是仅改变辅助变量的消融。

### 4.4 停滞、异常与证据限制

以下为关闭 Relax、无动量的既有诊断，SAV 比值为 $q^n/\sqrt{E^n+C}$，ESAV 比值为 $r^n/\exp(E^n/(4E^0))$。目标不变步数按相邻 CSV 行的目标值完全相同统计。

| 组别 | SAV 目标不变步数 | SAV 最终辅助比值 | ESAV 目标不变步数 | ESAV 最终辅助比值 |
| --- | --- | --- | --- | --- |
| A1 二次型基础 | 270 | $3.230555\times10^{-151}$ | 0 | 0.471126 |
| C1 非线性基础 | 591 | $8.201490\times10^{-322}$ | 0 | 0.729378 |
| D2 非线性，$M=4$ | 588 | $-3.458460\times10^{-323}$ | 0 | 0.725160 |
| D3 非线性，$M=8$ | 562 | $1.151160\times10^{-235}$ | 0 | 0.486195 |

C1 的 SAV、D1 的 PB-SAV、D2 的 PB-SAV 首次非正辅助值分别在第 134、169、224 步出现；记录仍完成预算，不能称作相同步数的提前失败。负号、零值和次正规浮点数保留为数值异常证据。目标不变不等于参数完全未更新。ESAV 没有相邻目标完全不变的步，也不等于已收敛到满意的目标。

直接核对基础 Relax-ESAV 与 SGD 的完整目标序列，A1 的最大绝对差为 $2.220446049250313\times10^{-16}$，C1 为 $5.551115123125783\times10^{-17}$。这是目标轨迹比较，没有保存参数轨迹以供逐元素比较。

本题覆盖最终报告的 57 条独立记录：10 条二次型、47 条非线性，共 31,257 行；均从第 0 步连续记录至预算末步，目标均有限。旧分组报告是 56 条记录的较早快照；最终报告后补了上述动量 SGD 基线。该数量是本题资料集的范围，不是实时在线项目的总数。

非线性只有一套数据、初始化和分组种子，尚无跨种子统计。当前读取的历史 CSV 主要包含 `iteration`、`objective`、`lr`、`r`、`r_current` 及分量能量；PB 分量对数状态、参数轨迹、梯度范数、动量及每步成本未完整保存。虽然算法在内存中维护这些状态，不能把文档所述诊断能力当成已有历史记录。因此，关于局部辅助漂移、矩阵作用或动能交换导致瓶颈的解释，目前仍需论证与针对性诊断。

## 5. “高效”的定义与已有成本线索

需要比较真实优化进展与实际计算代价，而不是只比较迭代数。可以选择相同接受更新预算、相同计算成本或达到目标阈值的成本等合理比较方式，不要求三种全部完成。具体实验由模型科学设计，说明预算、指标和停止规则即可。

| 当前执行路径 | 已确认的工作量 |
| --- | --- |
| 基础 ESAV 单次候选 | 一次目标前向和一次总梯度反向；参数及标量辅助更新 |
| PB-ESAV 单次候选 | 一次产生各分量能量的前向；逐分量调用自动微分，共 $M$ 次；分量梯度 QR 与小系统求解 |
| 当前 Relax | 在候选参数处再调用一次目标函数；不需要候选点梯度 |
| 当前 Armijo | 进入回溯前另求一次旧目标与总梯度；每个试探重新调用优化器；随后另算候选目标用于接受判定 |
| 当前记录循环 | 每个记录点另算目标；包含第 0 步 |
| 当前动量 PB-ESAV | 逐分量梯度；含右端的 QR 与至多 $M+1$ 阶求解；Relax 另算新目标 |

以上来自 `train.py` 与对应优化器源码；当前实现没有将旧梯度、QR 或 Relax 已求的新目标跨上述调用缓存复用。对正常完成、没有中途异常退出的试探，若一个接受更新共尝试 $J$ 个学习率，把一次返回全部能量项的 `loss_fn` 计为一次前向，并计入一个新状态记录，则基础无 Relax 是 $2J+2$ 次前向、$J+1$ 次总梯度反向；基础 Relax 是 $3J+2$ 次前向、$J+1$ 次反向。PB 的反向调用数变为 $JM+1$，并作 $J$ 次 QR 与求解。这是源码调用计数，不代表一次分量反向与一次总梯度反向具有相同耗时，也不是测得的耗时占比。

| 已有运行 | 无 Relax 的 `training_seconds` | 有 Relax 的 `training_seconds` |
| --- | --- | --- |
| C1 基础 ESAV | 1.269343 | 1.604148 |
| D3 PB-ESAV，$M=8$ | 4.025711 | 5.007818 |
| D4 PB-ESAV，$M=16$ | 7.031878 | 8.793103 |
| E5 动量 PB-ESAV，$M=30$ | 1.380973 | 1.475477 |

计时取对应 run 的 `summary.json`；`run.py` 从训练循环开始计时，扣除记录器累计的同步记录耗时，仍包含目标计算和指标开销。没有重复计时、逐步计时或函数调用实测日志，不能据此给出稳定加速比或纯 Relax 开销百分比。开关 Relax 还可能改变接受学习率与试探次数。

## 6. 相同输入、最小附件与证据索引

### 6.1 共同输入与当前任务

本组文件已包含核心格式、现有 Relax、问题定义、初始化规则、控制器、最终结果和成本限制。历史来源路径不等于额外访问授权；允许的只读范围以 §1.3 为准。不能把未读到的轨迹或源码内容当作已经验证。

双方使用相同共同题面和相同基线定义，各自设计科学合理的实验细节与预算。单个实验内部严格控制变量；跨模型比较时披露实际预算、工具与执行环境，不能假定两边计算成本相同。任何新增共同输入应由出题方同时提供；两轮交流之后不依赖继续补充材料。两方不得交换或访问对方的方案。

### 6.2 独立复现的最小材料

下表保留原题的最小材料范围。它不要求新附件打包，也不扩大 §1.3 的读取许可。未直接提供的原始材料按缺项披露；不得因此访问其他目录或在线账户。

| 用途 | 最小内容与使用方式 |
| --- | --- |
| 独立实现基线和控制器 | §1.3 指定的相关优化器与 `train.py`，只为准备己方独立副本；不得直接调用原文件。副本须与题面公式核对；新方案单独实现 |
| 独立实现两个问题 | §1.3 指定的问题、网络和固定配置；既有 `data/nonlinear_regression.pt` 约 19 KB，可准备己方输入副本以减少跨版本随机数差异，也可按明确记录的设置独立生成 |
| 如获提供，核对主要历史轨迹 | 原题建议的最小范围是 C1 的 6 条、D3 的 4 条、E5 的 ESAV 开/关 Relax 2 条，共 12 条 run；每条仅 `config.json`、`history.csv`、`summary.json`。只读取出题方已放入己方目录的相同副本，不自行下载或读取外部历史目录 |
| 文献来源 | arXiv:1912.09263v2 第 9 页、arXiv:2609.13569v1 §3.2.2–§3.4；保留引用用于溯源。已有公式见 §3，引用本身不增加目录外操作权限 |
| 实验执行与保存 | 独立入口、配置和脚本均在己方 `TmpResearchSpace/`，结果与日志在己方 `TmpData/`；不需要原仓库 `run.py`、`tracking.py` 或在线账户 |

不复制整个 Git 仓库、历史结果目录、SDK 缓存、私人会话或凭据。若某个归档版本或原始记录不能取得，列明缺项，不能以当前默认值冒充原始配置；新实验的独立设置应如实记录。

### 6.3 原题已核查的来源及未确定事项

下表是原题编制时的证据索引，不是参赛模型的额外读取许可。

| 来源 | 本题用途 |
| --- | --- |
| 指定对话 `01a17f70-aeb1-7910-9392-1cd01b271217`，“归类 ESAV 实验并规划报告”，最后两轮 | 已直接读取；确认用户认可并要求写入报告的阶段性结论 |
| `doc/experiments/esav_experiments.md`，A1–E5 与末尾阶段性结论 | 最终结果、后补动量 SGD 基线、报告结论 |
| `doc/experiments/esav_experiment_groups.md`，2026-10-27 18:20 America/New_York 快照 | 56 条旧清单、共同配置、诊断；其“缺动量 SGD”已被后续记录更新 |
| `doc/design/esav.md`；`doc/design/ESAV-Moment.md` | 当前基础、PB、动量与 Relax 公式及耗散对象 |
| 对应优化器、`train.py`、问题与网络模块 | 时间层、辅助状态、分组、初始化、回溯和调用成本 |
| `TMP/wandb_exports/<run_id>/` 下的完整 CSV、配置与摘要 | 57 条记录的预算、末行数字、异常与单次计时；原题编制时从保存的导出读取，未重新训练 |
| `TMP/experiment_report_audit/source/` 下的归档源码 | 抽查基础 ESAV、D3 Relax-PB-ESAV、E3 动量 ESAV 的相关更新文件，与当前对应优化器及所查训练循环一致 |
| `TMP/esav_relax_challenge/source_audit.json` | 原题编制时只读数据核对的行数、连续性、有限目标及末行值清单 |

旧 README 中仍有“PB-ESAV 仅演化总状态、$M=1$ 恢复基础 ESAV”的段落，与现行设计、优化器和所查归档源码不一致；本题采用 §3.4 的实际格式，未修改旧文件。

尚未核实或尚无证据的事项：所有运行的逐文件完整版本一致性、所有历史运行的精确软件环境、跨种子稳定性、可靠重复计时、每步调用实测、未保存的分量状态与参数轨迹，以及新 Relax 能否带来实质收益。数据文件记录的 PyTorch 版本不等于每条历史 run 的环境已逐项确认。上述题面编制与更新没有运行新训练或新 Relax 实验，也没有给出新 Relax 的解法；参赛交付仍须完成规定的独立本地实验。
