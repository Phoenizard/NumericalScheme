# NumericalScheme

## 目录结构与功能

```text
NumericalScheme/
├── problem/
│   ├── config.py                 # 按问题分区的固定超参数
│   ├── quadratic.py              # 二次问题与能量分裂
│   ├── nonlinear_regression.py   # 数据生成、分组损失、测试 MSE
│   └── darcy_deeponet.py         # Darcy 随机场、差分参考解、监督损失与评价
├── network/
│   ├── nonlinear_regression.py   # 单隐层 tanh 网络与初始化
│   └── darcy_deeponet.py         # 卷积 Branch + tanh Trunk，严格零边界
├── optimizer/
│   ├── gd.py                     # 总目标的普通梯度下降
│   ├── sgd.py                    # 全量 SGD，可选 heavy-ball 动量
│   ├── adamw.py                  # AdamW 基线，独立记录矩状态
│   ├── besav.py                  # BESAV，r = exp(E/(E+E0))
│   ├── pb_besav.py               # PB-BESAV，分量反馈与总辅助状态
│   ├── esav.py                   # 单能量 ESAV，r = exp(E/(4E0))
│   ├── mesav.py                  # MESAV，初始能量尺度归一化的 exponential 反馈
│   ├── pb_mesav.py               # PB-MESAV，全局驱动与分量修正
│   ├── naive_esav.py            # naiveESAV，支持完全 Relax 或 r 更新的一阶指数近似
│   ├── sav.py                    # SAV 单步更新
│   ├── csav.py                   # 项目 CSAV，单能量恢复与有界反馈
│   ├── InverseSAV.py             # 倒数辅助变量与中心化 PB 扩展
│   ├── pb_sav.py                 # PB-SAV 单步更新
│   ├── pb_sav_momentum.py        # PB-SAV 动量，欧氏/自适应缩放与完全Relax
│   ├── pb_esav_momentum.py       # ESAV/PB-ESAV 动量、自适应缩放与动能预算Relax
│   ├── pb_naive_esav.py         # PB-naiveESAV 两个 Plan，保留历史 Relax
│   └── pb_esav.py                # PB-ESAV，全局驱动与分量一阶修正
├── data/                         # 固定输入数据，默认不加入 Git
├── doc/                          # paper 论文、design 设计、experiments 实验报告
├── tracking.py                   # W&B 在线实验记录
├── TMP/                          # 项目内临时文件与工具缓存
├── run.py                        # 组装并运行一组实验
├── train.py                      # 公共迭代循环与指标采集
├── plot_figure.py                 # 读取已有结果绘图
└── requirements.txt
```

## 方法命名

| 名称 | 命令行方法 | 辅助变量 |
| --- | --- | --- |
| naiveESAV | `naive_esav` | 原始 $r=\exp(E)$ |
| BESAV / PB-BESAV | `besav` / `pb_besav` | $r=\exp(E/(E+E^0))$；冻结初始能量，无 shift |
| Relax-BESAV / Relax-PB-BESAV | `besav` / `pb_besav` 加 `--relax` | 同一定义，整体能量校正总 $\log r$ |
| ESAV | `esav` | $r=\exp(E/(4E^0))$，冻结初始能量 $E^0$ |
| Relax-ESAV / Relax-PB-ESAV | `esav` / `pb_esav` 加 `--relax` | 同一定义，参数更新后用整体能量校正总 $\ln r$ |
| MESAV（Modified-ESAV） | `mesav` / `pb_mesav` | $r=\exp(E/(E^0+C))\sqrt{E+C}$，冻结初始能量尺度 |
| Relax-MESAV / Relax-PB-MESAV | `mesav` / `pb_mesav` 加 `--relax` | 同一辅助变量，参数更新后用整体能量校正 |
| CSAV（项目方案） | `csav` | $r=E+C$，恢复型辅助更新与有界参数反馈 |
| Inverse SAV / PB-Inverse SAV | `inverse_sav` / `pb_inverse_sav` | $r=1/(E+C)$，辅助能量为 $1/r$ |
| PB-naiveESAV | `pb_naive_esav` | $r=\exp(E)$，$r_i=\exp(E_i)$ |
| PB-ESAV | `pb_esav` | 总 $r=\exp(E/(4E^0))$、分量 $r_i=\exp(E_i/(4E_i^0))$；冻结参考尺度，递推总量及分量辅助状态 |
| PB-ESAV动量 | `pb_esav_momentum` | 总及分量辅助状态；SAV动能，可选自适应缩放与组合能量Relax |

原始方法使用 `naive_esav`；ESAV 指 $r=\exp(E/(4E^0))$。
BESAV 单独使用 `besav` / `pb_besav`，不替换尺度 ESAV。
定义、基础 / PB / Relax 递推及耗散证明见 [ESAV 设计](doc/design/esav.md)。
指数与根号拼接的[旧版 ESAV](doc/design/esav_legacy.md) 已废弃；历史结果、配置、图片与个人 Notes 保留原样。
文献 RE-SAV 完整算法仍为待办，尚未实现。

MESAV 是已有 TMP 方案的归档实现，与原 `esav`、`pb_esav` 分开保留；当前仅作为实验记录和比较对象，不作为正式研究候选。科研评价约定见 [GOAL.md](GOAL.md#每次构造后的科研反思)。
定义、PB 构造、Relax 递推与耗散证明见 [MESAV 设计](doc/design/mesav.md)。
`mesav.py` 和 `pb_mesav.py` 均提供 `initialize(terms, total_shift)` 和原地参数更新的 `step`，返回新的辅助状态。
`step` 的 `total_shift` 和 `relax` 为关键字参数；`relax=True` 开启整体能量校正，
默认 `relax=False` 使用基础更新。PB 的 M=1 恢复相同 Relax 设置下的 MESAV。
只保存一个总 `log_r`；总参考尺度和 PB 分量参考尺度均固定为各自的初始能量加 shift，
记录在 `config.json` 的 `reference_energy`、`reference_components` 中。
命令行使用 `mesav` / `pb_mesav`，通过 `--relax` 选择 Relax，其余固定设置沿用 `PROBLEMS`。
运行入口统一记录实际目标和辅助诊断，不再计算测试 MSE；`mesav` 要求 `--m 1`。

```bash
python -B run.py --problem quadratic --method mesav
python -B run.py --problem nonlinear_regression --method pb_mesav --m 30
python -B run.py --problem nonlinear_regression --method pb_mesav --relax --m 30
```

Inverse SAV 与 PB 扩展的推导、初始化和无条件辅助耗散证明见
[Inverse SAV 设计](doc/design/inverse_sav.md)。PB 扩展只保存一个总辅助变量，
采用分量之间的中心化 pullback 修正；M=1 严格恢复单能量格式。
默认 `run.py` 沿用各问题的 Armijo 配置。

## 环境与配置

从项目根目录运行：

```bash
conda activate numenv
```

固定设置只在 [problem/config.py](problem/config.py) 的 `PROBLEMS` 字典中修改，
分别对应 `quadratic`、`nonlinear_regression` 和 `darcy_deeponet`。学习率、更新次数、shift、
随机种子、网络宽度等不通过命令行覆盖。

命令行选择问题、方法和研究变量；每次运行在 W&B 中创建独立 run。
BESAV 使用 `besav` / `pb_besav`，各自通过 `--relax` 开启 Relax；
非线性基础与 Relax-BESAV 要求 `--m 1`，PB 方法使用既有分组。
各支持 Relax 的方法统一使用 `--relax` / `--no-relax`；省略时读取问题配置的 `relax`。
Python 单步接口使用 `step(..., relax=True)`，默认关闭；`run_experiment(..., relax=None)`
读取问题配置。新记录保存基础/PB 的 `method` 与独立的布尔字段 `relax`。
`--scheme plan1/plan2` 仅用于 `--method pb_naive_esav`。
`--method esav/pb_esav` 使用 ESAV 的基础 / PB 格式，不使用 shift 或 Plan。
ESAV 保存总 `log_r`；PB-ESAV 另保存分量向量 `log_r_i`，所有分量数均采用 PB 更新。
加 `--relax` 校正总 `log_r`，PB 的分量状态仍按对数方程递推；
非线性回归的 `esav` 要求 `--m 1`。完整公式见 [ESAV 设计](doc/design/esav.md)。
`--method csav` 使用当前项目的 CSAV 候选，从 `total_shift` 读取 $C$，初始化
$r^0=E(x^0)+C$；先更新辅助变量，再用 $2r^{n+1}/[r^{n+1}+E(x^n)+C]$ 反馈参数。
使用现有 Armijo 回溯，不启用事后 Relax；不作为同名文献 CSAV 的复现。
`--method naive_esav` 使用 naiveESAV；`--linearized` 仅对其 r 更新的指数作一阶展开，
不改变 x 更新。原始分支保存 `log_r`，近似分支保存可为负数的 `r`，不作截断。
naiveESAV 开启 `relax` 时，在每步参数更新后执行
$\log r^{n+1}\leftarrow\min\{E(x^{n+1}),\log r^n\}$，参数保持本步计算值。
这是按 SAV 耗散预算思路扩展的完全 Relax；不可与 `--linearized` 同时使用。
W&B 配置的 `algorithm` 中记录 `relax` 及适用的 `relaxation_update`。
`--method sgd/adamw` 使用全量梯度基线；SGD 的 `sgd.momentum` 默认为0，Darcy 设为0.9。
SGD 的耦合正则写入问题损失；`step(..., momentum=0.9, state=state)` 返回新的位移状态。
AdamW 参数见问题配置中的 `adamw`。
AdamW 使用 `reference_baseline_lr=0.1` 的固定学习率，不启用 Armijo。
旧命令中的 `--lr`、`--steps`、`--lambda-reg`、`--total-shift`、
`--relax`、`--eta`、`--model`、`--data` 已移除。

## Darcy DeepONet

已接入SGD动量、PB-SAV动量及PB-ESAV动量；后两者支持欧氏或自适应缩放。配置与默认值来源见
[实验方案](doc/experiments/darcy_deeponet.md)。单种子42、CPU、float64，固定学习率0.01、
动量0.9、首步位移0、5000步；数据损失和耦合 $L^2$ 正则分别计算后相加。
PB 动量对照通过 `--reference-run` 指定本项目内已完成的 W&B SGD run，
核对共同配置与初始化种子；欧氏对照使用学习率0.01、动量0.9及相同预算；
`pb_sav_momentum.alpha=0.5`、总shift为0，默认无Relax、无自适应缩放。
这对应论文的欧氏动量更新，区别于论文Case3的完整自适应配置。
`--mobility amsgrad` 读取配置中的论文优化器设置：学习率0.003、beta2=0.999、epsilon=1e-8，
默认开启完全Relax（论文rho=1）；共同设置及初始化种子须与参考 run 一致。
运行示例（将 `<SGD_RUN_ID>` 替换为完成的在线 SGD run ID；已有数据无需再次生成）：

```bash
mkdir -p TMP/runtime
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/TMP/runtime"
export XDG_CACHE_HOME="$PWD/TMP/runtime/cache"
export MPLCONFIGDIR="$PWD/TMP/runtime/matplotlib"
python -B -m problem.darcy_deeponet
python -B run.py --problem darcy_deeponet --method sgd
python -B run.py --problem darcy_deeponet --method pb_sav_momentum --no-relax --reference-run 'pheonizard-university-of-nottingham/NumericalScheme/<SGD_RUN_ID>'
python -B run.py --problem darcy_deeponet --method pb_sav_momentum --mobility amsgrad --relax --reference-run 'pheonizard-university-of-nottingham/NumericalScheme/<SGD_RUN_ID>'
python -B run.py --problem darcy_deeponet --method pb_esav_momentum --mobility amsgrad --relax --reference-run 'pheonizard-university-of-nottingham/NumericalScheme/<SGD_RUN_ID>'
```

数据生成器保存 `data/darcy_deeponet.pt`，拒绝覆盖已有文件；数据设置变化时指定新路径并更新配置。
W&B 保存配置、完整逐步记录、摘要及源码。数据和模型权重不上传，仅记录数据路径。
历史列为 `iteration`、总目标 `objective`、`data_loss`、`regularization`、`test_relative_l2`，
包含第0步；图中绘制 `data_loss` 和逐测试场平均的相对 $L^2$。
PB-SAV动量另记 `q`、`q_over_Q`、`kinetic_energy`、`modified_energy`、`dissipation` 和 `energy_law_error`；
修正能量为 $q^2+p^\top M p/(2\eta)$，欧氏情形 $M=I$。
自适应缩放还记录缩放矩阵范围、逐坐标最大增量和Relax后的实际耗散；`dissipation`记录Relax前的五项耗散之和。
`--relax` 使用论文的包含动能预算的完全Relax；`--no-relax` 关闭该机制。
PB-ESAV动量采用[项目推导](doc/design/ESAV-Moment.md)中的SAV动量平衡；
其 `p` 与PB-SAV一致，实际位移为 $M^{n+1}p^{n+1}$，动能为 $p^\top M p/(2\eta)$。
保存总 `log_r` 和分量 `log_r_i`，$B_\alpha=\alpha B_{\mathrm{PB}}$；
α=0为普通ESAV动量，α=1为完整PB-ESAV动量。
`--mobility amsgrad` 读取固定学习率0.003、动量0.9、α=0.5、β₂=0.999、ε=1e-8；
`--relax` 采用包含动能的总对数辅助能量预算；组合能量为 $4E^0\log r+p^\top M p/(2\eta)$。
日志另记总与分量辅助状态、分量能量、辅助比值、Relax校正及耗散诊断。
`training_seconds` 包含指标计算，扣除同步记录耗时。
临时验证脚本、数据和记录位于 `TMP/darcy_sgd_validation_20261024/`，整目录可删除，正式模块不依赖它。

## 非线性回归

实验定义与设置来源见 [非线性回归实验报告](doc/experiments/nonlinear_regression.md)。
当前优先核对论文第 4.2 节的 PB-SAV、M=30 基线，再比较 PB-ESAV。
30 个带噪声正弦样本，1 → 50 → 1 tanh 网络，全量训练，CPU、float64。

GD、PB-SAV 与 PB-ESAV 每轮从学习率 `lr=100` 开始计算候选更新 `delta`，
检验 `F(theta+delta) <= F(theta)+1e-4*grad(F)·delta`，并要求 `grad(F)·delta <= 0`。
拒绝后恢复原参数、保留原辅助变量，将学习率减半并重新计算候选更新。
学习率最低试到 `1e-6`，仍不满足条件则报错；接受后提交参数和对应辅助变量。
下一轮重新从 `100` 开始。600 次接受的更新产生 601 行记录。
六组方法使用相同的学习率范围、减半规则和 Armijo 系数，设置集中在 `problem/config.py`。
论文未披露完整控制器、初始化和随机种子；缺失部分采用用户批准的本地默认值，
不能称为作者代码的逐数值严格复现。ESAV 系列通过方法名选择是否启用 Relax。

生成数据（仅首次或修改数据设置后）：

```bash
python -B -m problem.nonlinear_regression
```

数据保存至 `data/nonlinear_regression.pt`，记录生成参数。分组采用固定随机排列上的
嵌套二分，各组均按总样本数归一化；实际组索引保存进运行配置。
每次训练按相同种子重新创建网络。

PB-SAV 与 PB-ESAV 的动量对照使用固定学习率0.1、动量系数0.9、α=0.5，
单位缩放，默认关闭Relax；600步，4线程。配置位于各自的 `*_momentum` 字段。

每条命令运行一组：

```bash
python -B run.py --problem nonlinear_regression --method pb_sav --m 30
python -B run.py --problem nonlinear_regression --method pb_esav --m 30
python -B run.py --problem nonlinear_regression --method pb_esav --relax --m 30
python -B run.py --problem nonlinear_regression --method pb_sav_momentum --no-relax --m 30
python -B run.py --problem nonlinear_regression --method pb_esav_momentum --no-relax --m 30
python -B run.py --problem nonlinear_regression --method pb_naive_esav --scheme plan2 --m 30
python -B run.py --problem nonlinear_regression --method gd --m 30
```

`--m` 可取 `1, 2, 4, 8, 15, 16, 30`；更换 M 时同步更换输出目录。
PB-SAV 的 M=1 即 SAV。GD 对分量之和求梯度，不使用 shift 或辅助变量。
历史 GD、PB-SAV、PB-naiveESAV 与旧版 PB-ESAV 两个 Plan，以及 ESAV 的 PB / Relax-PB
结果均见实验文档；Nonlinear 按 M=1、2、4、8、16、30 分别列表与绘图。
历史目录与配置保留当时的名称。

## 二次问题

默认设置为 100 维、全 1 初值、两类能量系数 1 和 0.01、300 次接受更新、
CPU、float64、无正则化。Armijo 每轮从学习率 100 开始，拒绝后乘 0.5，
最低试到 1e-6，系数为 1e-4，与非线性回归相同；参数与辅助状态随拒绝步回滚。
固定参数来自配置中的 `quadratic` 分区。
`--loss split` 使用两个分量，`--loss total` 使用总能量。
CSAV、ESAV、naiveESAV、SGD、AdamW 默认使用总能量。
已整理的固定学习率结果与失效诊断见[二次型实验报告](doc/experiments/quadratic.md)。

```bash
python -B run.py --problem quadratic --method csav
python -B run.py --problem quadratic --method esav --loss total
python -B run.py --problem quadratic --method naive_esav
python -B run.py --problem quadratic --method naive_esav --linearized
python -B run.py --problem quadratic --method sgd
python -B run.py --problem quadratic --method adamw
```

```bash
python run.py --problem quadratic --method sav --loss total
python run.py --problem quadratic --method pb_sav --loss split
python run.py --problem quadratic --method pb_naive_esav --scheme plan1 --loss split
python run.py --problem quadratic --method pb_naive_esav --scheme plan2 --loss split
```

## 结果记录

新实验统一记录到 [W&B 项目](https://wandb.ai/pheonizard-university-of-nottingham/NumericalScheme)。
在线连接成功后才开始训练，每次独立 run；不再向 `results/` 写入结果。
配置采用顶层 `problem`、`method` 及嵌套的 `algorithm`、`control`、`experiment`；不记录版本号、比较组、校验值或 provenance。
完整方法参数清单、字段和参考 run 用法见 [实验记录](doc/design/experiment_tracking.md)。
上传 `config.json`、完整 `history.csv`、`summary.json` 和源码，SDK 缓存及暂存均位于 `TMP/wandb/`。
历史 `esav_variant` 中的 `original`、`naive` 对应 naiveESAV / PB-naiveESAV，
`exp_log_energy`、`sqrt_energy` 属于历史构造。ESAV 记录 `total_auxiliary`、
`initial_energy`、`initial_components` 和 PB 的 `B_definition`；PB 的 `auxiliary_state`
为 `log_r,log_r_i`，`component_auxiliary` 为 `exp(E_i/(4*E_i0))`。不使用 `curvature_shift`。
`iteration=0` 为初始状态；二次型 300 次接受更新对应 301 行，非线性回归 600 次对应 601 行。

| 字段 | 含义 |
| --- | --- |
| `objective` | 真实总目标；回归实验为残差平方和除以 2N |
| `test_mse` | 仅历史记录保留；当前入口不再计算 |
| `q`、`Q`、`q_over_Q` | SAV/PB-SAV 辅助变量、真实能量对应值及比值 |
| `log_r`、`r_over_exp_E` | naiveESAV / PB-naiveESAV，以及旧版 PB-ESAV 的历史字段 |
| `log_r`、`r_over_exp_auxiliary_energy` | ESAV 系列的对数辅助变量及 $r/\exp(E/(4E^0))$；历史构造按原配置解释 |
| `r` | naiveESAV 一阶指数近似分支的有符号辅助变量；该分支不记录对数修正能量 |
| `r`、`r_over_E_plus_C` | CSAV 的正辅助变量及一致性比值 $r/[E(x)+C]$ |
| `energy_i` | PB-ESAV 的分量能量，包含初始记录；历史 PB 指数方案亦使用此字段 |
| `log_r_i`、`r_i_over_exp_auxiliary_energy_i` | PB-ESAV 的分量对数状态及 $r_i/\exp(E_i/(4E_i^0))$，字段中的 i 为分量编号；历史记录按原配置解释 |
| `r_i_over_exp_E_i` | PB-naiveESAV Plan 2 相对于 $\exp(E_i)$ 的局部比值 |
| `modified_energy` | ESAV 系列为 $4E^0\ln r$；SAV/PB-SAV 为 q²，CSAV 为 r；naiveESAV/PB-naiveESAV 为 log_r，历史记录按原定义解释 |
| `initial_lr`、`accepted_lr` | 每轮初始学习率 100，以及实际接受的学习率；初始状态行为 0 |
| `rejected_trials`、`nonfinite_trials` | 当前接受步前的拒绝数、非有限试探数 |
| `directional_derivative`、`armijo_bound` | grad(F)·delta，以及 F(theta)+c*grad(F)·delta |

失败或中断时上传 `failure.json`，摘要记录结束状态、实际完成步数和最后有效结果。
非线性回归中的 ESAV、naiveESAV 使用 `--m 1`；新方法沿用该问题的 Armijo 回溯，
拒绝试探步时保留上一步辅助变量或 AdamW 矩状态。

GD 当前记录 `iteration`、`objective` 和上述 Armijo 回溯字段。
PB-SAV 使用总 shift，并均分到各分量；PB-naiveESAV 和 GD 不使用 shift。
PB-ESAV 的 $\mathbf B^n=\sum_i\nabla E_i^n(\nabla E_i^n)^\top/(4E_i^0)$，
仅演化总 $\ln r$，$M=1$ 严格恢复 ESAV。
Relax 更新为 $\ln r^{n+1}=\min\{E^{n+1}/(4E^0),\ln r^n\}$；
推导与无条件辅助耗散证明见 [ESAV 设计](doc/design/esav.md)。

运行入口不再接受 `--output`，结束后输出 W&B run 链接。

## 从已有结果绘图

正式报告通过官方 API 读取完整 records artifact，按 run ID 和版本固定比较组。
分组清单见 [esav_plot_groups.json](doc/experiments/esav_plot_groups.json)，
提取、校验与恢复规则见 [实验记录](doc/design/experiment_tracking.md#结果提取与绘图)。

```bash
conda activate numenv
python plot_figure.py --group B1
python plot_figure.py --all
python plot_figure.py --group B1 --offline
```

原始文件缓存于 `TMP/wandb_exports/<run_id>/`；各组的数值表、PNG、SVG 与版本清单
位于 `TMP/experiment_exports/<组号小写>/`，本次执行状态见 `status.json`。
首次下载后重复执行复用缓存；`--offline` 完全不连接 W&B。以上命令仅提取已有结果。

单个 records artifact 目录仍可直接绘图，兼容旧版配置：

```bash
python plot_figure.py --training --input TMP/downloaded_run --output TMP/downloaded_run/training.png
```

W&B 图表导出的 CSV 可直接绘制优化目标：

```bash
python plot_figure.py --wandb-csv --input TMP/wandb_export.csv \
  --output TMP/objective_log.png \
  --labels AdamW SGD Relax-ESAV Relax-SAV ESAV SAV
```

`--labels` 按 CSV 中目标列的顺序指定，省略时保留原始 run 名；`--title` 可设置标题。
读取 `iteration` 和 ` - objective` 列，保留全部点，不平滑；`__MIN`、`__MAX` 和
`_step` 列不作为独立曲线。正值使用对数轴，零或负值使用注明线性区间的 symlog；
缺失或非有限值明确报错。输出扩展名可使用 `.png`、`.svg` 或 `.pdf`。

`--input` 可传入同一 M 下已完成的多个实验目录。每个 M 单独作图，
当前结果绘制目标 loss；只有所有输入都含历史 `test_mse` 时才附加旧指标面板。
旧二次问题的单组能量图与多组辅助变量对比图入口保持不变。
`--modified-energy` 按学习率分别绘制二次问题的修正能量，保留负值；
SAV 使用 q²，CSAV 使用 r，BESAV 系列使用 log_r，ESAV 系列使用 $4E^0\ln r$，naiveESAV / PB-naiveESAV 使用 log_r；SGD、AdamW 绘制真实目标 E 作为参考，
在图例中区分。naiveESAV 一阶指数近似分支不绘制对数修正能量，不使用 log|r| 替代。
二次型目标或辅助比值含零时采用标注线性区间的 symlog 坐标，保留数值零点。

## 历史 Relax 实验

PB-naiveESAV 的历史 Relax 位于 `pb_naive_esav.py`；BESAV、ESAV、MESAV 的 Relax
统一集成于基础 / PB 模块的 `relax` 开关，现行公式以各自设计文档为准。
旧记录中的 `relax_` 方法名继续由绘图读取；原始配置、记录与归档源码保持原样。
旧二次问题记录包含 SAV/PB-SAV 的普通版与 Relax 版、PB-naiveESAV 两种 Plan，
以及 Plan 1 的 eta=1、0.95、0.1 三组 Relax 结果。旧数据的路径保持原样，
可继续使用绘图脚本读取。

Plan 1 的 Relax 对比图中，eta=1 为实线，0.95 为方形标记虚线，
0.1 为三角形标记点线；历史配置缺少 eta 时按 1 读取。


## 文档导航

- 论文：[paper](doc/paper/)。
- 设计：[BESAV / PB / Relax](doc/design/besav.md)、[naiveESAV](doc/design/naive_esav.md)、[ESAV / PB-ESAV / Relax](doc/design/esav.md)、[旧版 ESAV 与 Notes](doc/design/esav_legacy.md)、[MESAV / PB-MESAV / Relax](doc/design/mesav.md)、[Inverse SAV](doc/design/inverse_sav.md)。
- 实验：[二次型](doc/experiments/quadratic.md)、[非线性回归](doc/experiments/nonlinear_regression.md)。
- 正式数据归档：[results 导航](results/README.md)。两篇报告均含历史探索、MESAV、ESAV、BESAV 四个主节。BESAV 的 23 组原始记录已归档，推导与复现材料不依赖项目 TMP。

## BESAV 正式接口与结果

BESAV 在当前二次型和非线性回归实验背景下具有继续探索价值。评价先看是否超过
同等级 SAV；已超过时不继续排名，都未超过时再比较尺度 ESAV。SAV 提前失败使用
最后有效目标值。非线性固定学习率 10 标注为当前观察到的舒适区之外，原失败记录保留。

```bash
python -B run.py --problem quadratic --method besav
python -B run.py --problem quadratic --method besav --relax
python -B run.py --problem nonlinear_regression --method pb_besav --m 30
python -B run.py --problem nonlinear_regression --method pb_besav --relax --m 30
```

固定学习率实验在 `PROBLEMS` 中设置 `lr` 并将 `line_search` 设为 `None`；每次只运行一组，
每次运行自动创建新的 W&B run。

- [BESAV 定义、PB 推导与 Relax 耗散证明](doc/design/besav.md)。
- [二次型结果](doc/experiments/quadratic.md#4-besav-及其扩展)、[非线性回归结果](doc/experiments/nonlinear_regression.md#4-besav-及其扩展)。
- [迁移验证、源码与归档清单](results/reproduction_besav_20261024/README.md)。
