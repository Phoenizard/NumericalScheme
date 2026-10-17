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
│   ├── sav.py                    # SAV 单步更新
│   ├── pb_sav.py                 # PB-SAV 单步更新
│   └── pb_esav.py                # PB-ESAV Plan 1 / Plan 2 单步更新
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
旧命令中的 `--lr`、`--steps`、`--lambda-reg`、`--total-shift`、
`--relax`、`--eta`、`--model`、`--data` 已移除。

## 非线性回归

实验定义见 [Nonlinear Experiments Plan](doc/Nonlinear-Experiments-Plan.md)。
当前配置为 30 个带噪声正弦样本、1 → 50 → 1 的 tanh 网络、全量训练、
固定步长 0.1、600 次更新、CPU、float64，无动量、无正则化、无 Relax。

生成数据：

```bash
python -B -m problem.nonlinear_regression
```

数据保存至 `data/nonlinear_regression.pt`，包含训练张量、无噪声测试张量及
生成参数。修改样本数、区间、噪声或数据种子后，重新执行生成命令。
训练直接读取该文件，按输入排序后循环分组；各组均使用总样本数归一化。
每次训练使用配置中的同一初始化种子重新建立网络。

每条命令运行一组。以下三条可分别执行：

```bash
python run.py --problem nonlinear_regression --method pb_sav --m 4 --output results/nonlinear_regression/pb_sav_m4
python run.py --problem nonlinear_regression --method pb_esav --scheme plan1 --m 4 --output results/nonlinear_regression/pb_esav_plan1_m4
python run.py --problem nonlinear_regression --method pb_esav --scheme plan2 --m 4 --output results/nonlinear_regression/pb_esav_plan2_m4
```

`--m` 可取 `1, 2, 4, 8, 16, 30`；更换 M 时同步更换输出目录。
PB-SAV 的 M=1 即 SAV，无需单独增加一行实验。当前代码和数据已准备，
18 组训练尚未运行。

## 二次问题

默认设置保持 100 维、全 1 初值、两类能量系数 1 和 0.01、步长 10、
300 次更新、CPU、float64、无正则化。固定参数来自配置中的 `quadratic` 分区。
`--loss split` 使用两个分量，`--loss total` 使用总能量。

```bash
python run.py --problem quadratic --method sav --loss total --output results/quadratic/sav_total
python run.py --problem quadratic --method pb_sav --loss split --output results/quadratic/pb_sav_split
python run.py --problem quadratic --method pb_esav --scheme plan1 --loss split --output results/quadratic/pb_esav_plan1
python run.py --problem quadratic --method pb_esav --scheme plan2 --loss split --output results/quadratic/pb_esav_plan2
```

## 结果记录

每组实验保存 `config.json` 与 `history.csv`。配置记录实际使用的方法、分组数、
固定参数、精度和 PyTorch 版本；非线性回归另记录数据路径及文件内的生成参数。
`iteration=0` 为初始状态，600 次更新对应 601 行记录。

| 字段 | 含义 |
| --- | --- |
| `objective` | 真实总目标；回归实验为残差平方和除以 2N |
| `test_mse` | 回归实验在无噪声测试点上的均方误差 |
| `q`、`Q`、`q_over_Q` | SAV/PB-SAV 辅助变量、真实能量对应值及比值 |
| `log_r`、`r_over_exp_E` | PB-ESAV 总辅助变量的对数及一致性比值 |
| `energy_i` | PB-ESAV 各分量能量 |
| `log_r_i`、`r_i_over_exp_E_i` | PB-ESAV Plan 2 的局部辅助变量及比值 |
| `modified_energy` | SAV/PB-SAV 的 q²，或 PB-ESAV 的 log_r |

PB-SAV 使用总 shift，并均分到各分量；PB-ESAV 不使用 shift。
Plan 1 仅演化总辅助变量，Plan 2 同时演化总、局部辅助变量，均保留对数状态。
推导见 [PB-ESAV 方案](doc/PB-Exp-SAV-Plan.md)。

默认输出目录为 `results/<problem>/<method及分组标识>/`。
同一路径再次运行会覆盖其配置和历史；保留旧结果时指定新的 `--output`。

## 从已有结果绘图

```bash
python plot_figure.py --input results/nonlinear_regression/pb_esav_plan1_m4 --output results/nonlinear_regression/pb_esav_plan1_m4/figure.png
```

单组图显示目标能量和修正能量。传入多个目录可对比目标能量和辅助变量比值：

```bash
python plot_figure.py --input results/nonlinear_regression/pb_sav_m4 results/nonlinear_regression/pb_esav_plan1_m4 results/nonlinear_regression/pb_esav_plan2_m4 --output results/nonlinear_regression/comparison_m4.png
```

绘图只读取已完成的实验记录。当前绘图脚本不展示测试 MSE，该指标保存在 CSV 中。

## 历史 Relax 实验

已有 Relax 实现、配置与结果保留；当前研究不开展 Relax 参数扫描。
旧二次问题记录包含 SAV/PB-SAV 的普通版与 Relax 版、PB-ESAV 两种 Plan，
以及 Plan 1 的 eta=1、0.95、0.1 三组 Relax 结果。旧数据的路径保持原样，
可继续使用绘图脚本读取。

Plan 1 的 Relax 对比图中，eta=1 为实线，0.95 为方形标记虚线，
0.1 为三角形标记点线；历史配置缺少 eta 时按 1 读取。
