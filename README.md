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
│   ├── sav.py                    # SAV 单步更新
│   ├── pb_sav.py                 # PB-SAV 单步更新
│   ├── pb_naive_esav.py          # 原朴素 ESAV 两个 Plan，保留历史 Relax
│   └── pb_esav.py                # 指数内含对数能量的方案，非 Relax 的 Plan 1 / Plan 2
├── data/                         # 固定输入数据，默认不加入 Git
├── doc/                          # 论文、推导与实验设计
├── results/                      # 每组实验的配置和逐步记录
├── run.py                        # 组装并运行一组实验
├── train.py                      # 公共迭代循环与指标采集
├── plot_figure.py                 # 读取已有结果绘图
└── requirements.txt
```

## 环境与配置

从项目根目录运行：

```bash
conda activate numenv
```

固定设置只在 [problem/config.py](problem/config.py) 的 `PROBLEMS` 字典中修改，
分别对应 `quadratic` 和 `nonlinear_regression`。学习率、更新次数、shift、
随机种子、网络宽度等不通过命令行覆盖。

命令行选择问题、方法、PB-ESAV Plan 和研究变量，并用 `--output` 指定结果目录。
`--method pb_esav` 使用局部辅助变量 $r_i=\exp(E_i+\frac12\ln(E_i+C/M))$ 的方案；
`--method pb_naive_esav` 使用原朴素实现。两个方法均支持 `--scheme plan1/plan2`。
旧命令中的 `--lr`、`--steps`、`--lambda-reg`、`--total-shift`、
`--relax`、`--eta`、`--model`、`--data` 已移除。

## 非线性回归

实验定义与设置来源见 [Nonlinear Experiments Plan](doc/Nonlinear-Experiments-Plan.md)。
当前优先核对论文第 4.2 节的 PB-SAV、M=30 基线，再比较 PB-ESAV。
30 个带噪声正弦样本，1 → 50 → 1 tanh 网络，全量训练，CPU、float64，无动量。

GD、PB-SAV 与 PB-ESAV 每轮从学习率 `lr=100` 开始计算候选更新 `delta`，
检验 `F(theta+delta) <= F(theta)+1e-4*grad(F)·delta`，并要求 `grad(F)·delta <= 0`。
拒绝后恢复原参数、保留原辅助变量，将学习率减半并重新计算候选更新。
学习率最低试到 `1e-6`，仍不满足条件则报错；接受后提交参数和对应辅助变量。
下一轮重新从 `100` 开始。600 次接受的更新产生 601 行记录。
六组方法使用相同的学习率范围、减半规则和 Armijo 系数，设置集中在 `problem/config.py`。
论文未披露完整控制器、初始化和随机种子；缺失部分采用用户批准的本地默认值，
不能称为作者代码的逐数值严格复现。Relax 沿用项目的关闭设置。

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
python -B run.py --problem nonlinear_regression --method pb_esav --scheme plan1 --m 30 --output results/nonlinear_regression/pb_esav_plan1_m30_exp_log_energy
python -B run.py --problem nonlinear_regression --method pb_esav --scheme plan2 --m 30 --output results/nonlinear_regression/pb_esav_plan2_m30_exp_log_energy
python -B run.py --problem nonlinear_regression --method pb_naive_esav --scheme plan2 --m 30 --output results/nonlinear_regression/pb_naive_esav_plan2_m30_armijo_lr100
python -B run.py --problem nonlinear_regression --method gd --m 30 --output results/nonlinear_regression/gd_m30_armijo_lr100
```

`--m` 可取 `1, 2, 4, 8, 16, 30`；更换 M 时同步更换输出目录。
PB-SAV 的 M=1 即 SAV。GD 对分量之和求梯度，不使用 shift 或辅助变量。
当前 Armijo 设置下，M=1、2、4、8、16、30 的 GD、PB-SAV、PB Naive ESAV 两个 Plan
及新 PB-ESAV 两个 Plan 均已有结果，每个 M 单独列表与绘图，见实验文档。
历史目录与配置保留当时的名称。

## 二次问题

默认设置保持 100 维、全 1 初值、两类能量系数 1 和 0.01、步长 10、
300 次更新、CPU、float64、无正则化。固定参数来自配置中的 `quadratic` 分区。
`--loss split` 使用两个分量，`--loss total` 使用总能量。

```bash
python run.py --problem quadratic --method sav --loss total --output results/quadratic/sav_total
python run.py --problem quadratic --method pb_sav --loss split --output results/quadratic/pb_sav_split
python run.py --problem quadratic --method pb_naive_esav --scheme plan1 --loss split --output results/quadratic/pb_naive_esav_plan1
python run.py --problem quadratic --method pb_naive_esav --scheme plan2 --loss split --output results/quadratic/pb_naive_esav_plan2
```

## 结果记录

每组实验保存 `config.json` 与 `history.csv`。配置记录实际使用的方法、分组数、
固定参数、精度和 PyTorch 版本；非线性回归另记录数据路径及文件内的生成参数。
ESAV 配置用 `esav_variant` 区分 `exp_log_energy` / `naive`；历史平方根方案标记为
`sqrt_energy`。新 PB-ESAV 另记录 `curvature_shift` 和总、局部辅助变量的定义。
`iteration=0` 为初始状态，自适应训练的 600 次接受更新对应 601 行记录。

| 字段 | 含义 |
| --- | --- |
| `objective` | 真实总目标；回归实验为残差平方和除以 2N |
| `test_mse` | 回归实验在无噪声测试点上的均方误差 |
| `q`、`Q`、`q_over_Q` | SAV/PB-SAV 辅助变量、真实能量对应值及比值 |
| `log_r`、`r_over_exp_E` | PB-ESAV 总辅助变量的对数及一致性比值 |
| `energy_i` | PB-ESAV 各分量能量 |
| `log_r_i`、`r_i_over_exp_auxiliary_energy_i` | 新 PB-ESAV Plan 2 的局部对数状态及相对于 $\exp(E_i+\frac12\ln(E_i+C/M))$ 的比值 |
| `r_i_over_exp_E_i` | PB Naive ESAV Plan 2 相对于 $\exp(E_i)$ 的局部比值 |
| `modified_energy` | SAV/PB-SAV 的 q²，或 PB-ESAV 的 log_r |
| `initial_lr`、`accepted_lr` | 每轮初始学习率 100，以及实际接受的学习率；初始状态行为 0 |
| `rejected_trials`、`nonfinite_trials` | 当前接受步前的拒绝数、非有限试探数 |
| `directional_derivative`、`armijo_bound` | grad(F)·delta，以及 F(theta)+c*grad(F)·delta |

数值失败时另存 `failure.txt`，保留最后接受状态之前的历史。

GD 的历史包含 `iteration`、`objective`、`test_mse` 和上述 Armijo 回溯字段。
PB-SAV 使用总 shift，并均分到各分量；PB Naive ESAV 和 GD 不使用 shift。
新 PB-ESAV 从同一 `total_shift` 配置读取 $C$，用于局部辅助变量中的
$\frac12\ln(E_i+C/M)$ 及曲率权重 $1+1/[2(E_i+C/M)]$；
真实目标与总辅助变量 $r=\exp(E)$ 保持不变。该值保存为运行配置中的 `curvature_shift`。
Plan 1 仅演化总辅助变量，Plan 2 同时演化总、局部辅助变量，均保留对数状态。
Plan 2 初始化 $\ln r_i=E_i+\frac12\ln(E_i+C/M)$，每步增加
$(1+1/[2(E_i+C/M)])\boldsymbol{\nabla}E_i^\top\Delta\boldsymbol{\theta}$。
推导见 [PB-ESAV 方案](doc/PB-Exp-SAV-Plan.md)。

默认输出目录为 `results/<problem>/<method及分组标识>/`。
同一路径再次运行会覆盖其配置和历史；保留旧结果时指定新的 `--output`。

## 从已有结果绘图

绘图只读取已有记录，自动从配置标注固定／自适应步长：

```bash
python plot_figure.py --training --input results/nonlinear_regression/pb_sav_m30_armijo_lr100 --output results/nonlinear_regression/pb_sav_m30_armijo_lr100/training.png
```

`--input` 可传入同一 M 下已完成的多个实验目录。每个 M 单独作图，
左侧为训练目标，右侧为测试 MSE；是否写入文档按当前任务要求决定。
旧二次问题的单组能量图与多组辅助变量对比图入口保持不变。

## 历史 Relax 实验

已有 Relax 实现、配置与结果保留；ESAV 的历史 Relax 位于 `pb_naive_esav.py`，
新 `pb_esav.py` 不支持 Relax，当前研究不开展 Relax 参数扫描。
旧二次问题记录包含 SAV/PB-SAV 的普通版与 Relax 版、PB Naive ESAV 两种 Plan，
以及 Plan 1 的 eta=1、0.95、0.1 三组 Relax 结果。旧数据的路径保持原样，
可继续使用绘图脚本读取。

Plan 1 的 Relax 对比图中，eta=1 为实线，0.95 为方形标记虚线，
0.1 为三角形标记点线；历史配置缺少 eta 时按 1 读取。
