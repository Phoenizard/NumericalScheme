# NumericalScheme

## 目录结构与功能

```text
NumericalScheme/
├── problem/
│   ├── config.py                 # 按问题分区的固定超参数
│   ├── quadratic.py              # 二次问题与能量分裂
│   └── nonlinear_regression.py   # 数据生成、分组损失、测试 MSE
├── network/
│   └── nonlinear_regression.py   # 单隐层 tanh 网络与初始化
├── optimizer/
│   ├── gd.py                     # 总目标的普通梯度下降
│   ├── sgd.py                    # 全量 SGD 基线，无动量
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
│   ├── pb_naive_esav.py         # PB-naiveESAV 两个 Plan，保留历史 Relax
│   └── pb_esav.py                # PB-ESAV，全局驱动与分量一阶修正
├── data/                         # 固定输入数据，默认不加入 Git
├── doc/                          # paper 论文、design 设计、experiments 实验报告
├── results/                      # 每组实验的配置和逐步记录
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
| PB-ESAV | `pb_esav` | 总 $r=\exp(E/(4E^0))$；冻结 $E_i^0$，仅演化总辅助变量 |

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
python -B run.py --problem quadratic --method mesav --output results/quadratic/mesav_total
python -B run.py --problem nonlinear_regression --method pb_mesav --m 30 --output results/nonlinear_regression/pb_mesav_m30
python -B run.py --problem nonlinear_regression --method pb_mesav --relax --m 30 --output results/nonlinear_regression/relax_pb_mesav_m30
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
分别对应 `quadratic` 和 `nonlinear_regression`。学习率、更新次数、shift、
随机种子、网络宽度等不通过命令行覆盖。

命令行选择问题、方法和研究变量，并用 `--output` 指定结果目录。
BESAV 使用 `besav` / `pb_besav`，各自通过 `--relax` 开启 Relax；
非线性基础与 Relax-BESAV 要求 `--m 1`，PB 方法使用既有分组。
各支持 Relax 的方法统一使用 `--relax` / `--no-relax`；省略时读取问题配置的 `relax`。
Python 单步接口使用 `step(..., relax=True)`，默认关闭；`run_experiment(..., relax=None)`
读取问题配置。新记录保存基础/PB 的 `method` 与独立的布尔字段 `relax`。
`--scheme plan1/plan2` 仅用于 `--method pb_naive_esav`。
`--method esav/pb_esav` 使用 ESAV 的基础 / PB 格式，保存总 `log_r`，不使用 shift 或 Plan。
加 `--relax` 开启对应的整体能量 Relax；
非线性回归的 `esav` 要求 `--m 1`。完整公式见 [ESAV 设计](doc/design/esav.md)。
`--method csav` 使用当前项目的 CSAV 候选，从 `total_shift` 读取 $C$，初始化
$r^0=E(x^0)+C$；先更新辅助变量，再用 $2r^{n+1}/[r^{n+1}+E(x^n)+C]$ 反馈参数。
使用现有 Armijo 回溯，不启用事后 Relax；不作为同名文献 CSAV 的复现。
`--method naive_esav` 使用 naiveESAV；`--linearized` 仅对其 r 更新的指数作一阶展开，
不改变 x 更新。原始分支保存 `log_r`，近似分支保存可为负数的 `r`，不作截断。
naiveESAV 开启 `relax` 时，在每步参数更新后执行
$\log r^{n+1}\leftarrow\min\{E(x^{n+1}),\log r^n\}$，参数保持本步计算值。
这是按 SAV 耗散预算思路扩展的完全 Relax；不可与 `--linearized` 同时使用。
运行时指定新的 `--output` 目录，配置会记录 `relax` 及 `relaxation_update`。
`--method sgd/adamw` 使用全量梯度基线；SGD 无动量，AdamW 参数见问题配置中的 `adamw`。
AdamW 使用 `reference_baseline_lr=0.1` 的固定学习率，不启用 Armijo。
旧命令中的 `--lr`、`--steps`、`--lambda-reg`、`--total-shift`、
`--relax`、`--eta`、`--model`、`--data` 已移除。

## 非线性回归

实验定义与设置来源见 [非线性回归实验报告](doc/experiments/nonlinear_regression.md)。
当前优先核对论文第 4.2 节的 PB-SAV、M=30 基线，再比较 PB-ESAV。
30 个带噪声正弦样本，1 → 50 → 1 tanh 网络，全量训练，CPU、float64，无动量。

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

每条命令运行一组：

```bash
python -B run.py --problem nonlinear_regression --method pb_sav --m 30 --output results/nonlinear_regression/pb_sav_m30_armijo_lr100
python -B run.py --problem nonlinear_regression --method pb_esav --m 30 --output results/nonlinear_regression/pb_esav_m30_armijo
python -B run.py --problem nonlinear_regression --method pb_esav --relax --m 30 --output results/nonlinear_regression/relax_pb_esav_m30_armijo
python -B run.py --problem nonlinear_regression --method pb_naive_esav --scheme plan2 --m 30 --output results/nonlinear_regression/pb_naive_esav_plan2_m30_armijo_lr100
python -B run.py --problem nonlinear_regression --method gd --m 30 --output results/nonlinear_regression/gd_m30_armijo_lr100
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
python -B run.py --problem quadratic --method csav --output results/quadratic/csav_total_armijo_lr100
python -B run.py --problem quadratic --method esav --loss total --output results/quadratic/esav_total_armijo_lr100
python -B run.py --problem quadratic --method naive_esav --output results/quadratic/naive_esav_total
python -B run.py --problem quadratic --method naive_esav --linearized --output results/quadratic/naive_esav_linearized_total
python -B run.py --problem quadratic --method sgd --output results/quadratic/sgd_total
python -B run.py --problem quadratic --method adamw --output results/quadratic/adamw_total
```

```bash
python run.py --problem quadratic --method sav --loss total --output results/quadratic/sav_total
python run.py --problem quadratic --method pb_sav --loss split --output results/quadratic/pb_sav_split
python run.py --problem quadratic --method pb_naive_esav --scheme plan1 --loss split --output results/quadratic/pb_naive_esav_plan1
python run.py --problem quadratic --method pb_naive_esav --scheme plan2 --loss split --output results/quadratic/pb_naive_esav_plan2
```

## 结果记录

每组实验保存 `config.json` 与 `history.csv`。配置记录实际使用的方法、分组数、
固定参数、精度和 PyTorch 版本；非线性回归另记录数据路径及文件内的生成参数。
历史 `esav_variant` 中的 `original`、`naive` 对应 naiveESAV / PB-naiveESAV，
`exp_log_energy`、`sqrt_energy` 属于历史构造。ESAV 记录 `total_auxiliary`、
`initial_energy`、`initial_components` 和 PB 的 `B_definition`；不使用 `curvature_shift`。
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
| `energy_i` | 历史 PB 指数方案的分量能量 |
| `log_r_i`、`r_i_over_exp_auxiliary_energy_i` | 旧版 PB-ESAV Plan 2 的历史局部状态及比值 |
| `r_i_over_exp_E_i` | PB-naiveESAV Plan 2 相对于 $\exp(E_i)$ 的局部比值 |
| `modified_energy` | ESAV 系列为 $4E^0\ln r$；SAV/PB-SAV 为 q²，CSAV 为 r；naiveESAV/PB-naiveESAV 为 log_r，历史记录按原定义解释 |
| `initial_lr`、`accepted_lr` | 每轮初始学习率 100，以及实际接受的学习率；初始状态行为 0 |
| `rejected_trials`、`nonfinite_trials` | 当前接受步前的拒绝数、非有限试探数 |
| `directional_derivative`、`armijo_bound` | grad(F)·delta，以及 F(theta)+c*grad(F)·delta |

数值失败时另存 `failure.txt`，保留最后接受状态之前的历史。
非线性回归中的 ESAV、naiveESAV 使用 `--m 1`；新方法沿用该问题的 Armijo 回溯，
拒绝试探步时保留上一步辅助变量或 AdamW 矩状态。

GD 当前记录 `iteration`、`objective` 和上述 Armijo 回溯字段。
PB-SAV 使用总 shift，并均分到各分量；PB-naiveESAV 和 GD 不使用 shift。
PB-ESAV 的 $\mathbf B^n=\sum_i\nabla E_i^n(\nabla E_i^n)^\top/(4E_i^0)$，
仅演化总 $\ln r$，$M=1$ 严格恢复 ESAV。
Relax 更新为 $\ln r^{n+1}=\min\{E^{n+1}/(4E^0),\ln r^n\}$；
推导与无条件辅助耗散证明见 [ESAV 设计](doc/design/esav.md)。

默认输出目录为 `results/<problem>/<method及分组标识>/`。
同一路径再次运行会覆盖其配置和历史；保留旧结果时指定新的 `--output`。

## 从已有结果绘图

绘图只读取已有记录，自动从配置标注固定／自适应步长：

```bash
python plot_figure.py --training --input results/nonlinear_regression/pb_sav_m30_armijo_lr100 --output results/nonlinear_regression/pb_sav_m30_armijo_lr100/training.png
```

`--input` 可传入同一 M 下已完成的多个实验目录。每个 M 单独作图，
当前结果绘制目标 loss；只有所有输入都含历史 `test_mse` 时才附加旧指标面板。
旧二次问题的单组能量图与多组辅助变量对比图入口保持不变。
`--modified-energy` 按学习率分别绘制二次问题的修正能量，保留负值；
SAV 使用 q²，CSAV 使用 r，BESAV 系列使用 log_r，ESAV 系列使用 $4E^0\ln r$，naiveESAV / PB-naiveESAV 使用 log_r；SGD、AdamW 绘制真实目标 E 作为参考，
在图例中区分。naiveESAV 一阶指数近似分支不绘制对数修正能量，不使用 log|r| 替代。
二次型目标或辅助比值含零时采用标注线性区间的 symlog 坐标，保留数值零点。

## 历史 Relax 实验

PB-naiveESAV 的历史 Relax 位于 `pb_naive_esav.py`；BESAV、ESAV、MESAV 的 Relax
统一集成于基础 / PB 模块的 `relax` 开关，公式与历史数值行为保持不变。
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
python -B run.py --problem quadratic --method besav --output results/quadratic/besav_new_run
python -B run.py --problem quadratic --method besav --relax --output results/quadratic/relax_besav_new_run
python -B run.py --problem nonlinear_regression --method pb_besav --m 30 --output results/nonlinear_regression/pb_besav_m30_new_run
python -B run.py --problem nonlinear_regression --method pb_besav --relax --m 30 --output results/nonlinear_regression/relax_pb_besav_m30_new_run
```

固定学习率实验在 `PROBLEMS` 中设置 `lr` 并将 `line_search` 设为 `None`；每次只运行一组，
使用新的输出目录。无需重跑已有实验；原始记录和失败步数见两篇报告的 BESAV 节。

- [BESAV 定义、PB 推导与 Relax 耗散证明](doc/design/besav.md)。
- [二次型结果](doc/experiments/quadratic.md#4-besav-及其扩展)、[非线性回归结果](doc/experiments/nonlinear_regression.md#4-besav-及其扩展)。
- [迁移验证、源码与归档清单](results/reproduction_besav_20261024/README.md)。
