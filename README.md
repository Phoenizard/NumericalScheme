# NumericalScheme

## 目录结构与功能

```text
NumericalScheme/
├── problem/                  # 问题定义、参数初始化与损失函数
│   └── quadratic.py          # 二次问题；提供 Total Energy Loss 和 Split Energy Loss
├── optimizer/                # 优化算法的参数更新与辅助变量更新
│   ├── sav.py                # SAV 单步更新
│   ├── pb_sav.py             # PB-SAV 单步更新
│   └── pb_esav.py            # PB-Exp-SAV Plan 1 / Plan 2 单步更新
├── network/                  # 模型网络（预留）
├── data/                     # 数据集目录
├── doc/                      # 参考论文、算法推导与研究记录
│   └── PB-Exp-SAV-Plan.md     # PB-Exp-SAV 推导与方案文档
├── results/                  # 实验记录、运行参数与图表
├── run.py                    # 命令行入口：选择问题、方法与损失，保存结果
├── train.py                  # 统一迭代循环与过程记录
├── plot_figure.py            # 仅读取已有实验记录并绘图
├── requirements.txt          # Python 依赖
└── README.md                 # 目录与功能说明
```

## 算法库

在项目根目录激活环境：

```bash
conda activate numenv
```

以下以二次问题为例。`run.py` 负责运行和保存，`plot_figure.py` 负责绘图。
每个实验目录包含 `config.json`（运行配置）和 `history.csv`（逐步记录）。
`iteration` 表示 epoch，0 为初始状态；300 步实验包含 301 行数据。
SAV、PB-SAV 和 PB-Exp-SAV 添加 `--relax` 可启用各自的松弛更新。

### PB-Exp-SAV

依据 `doc/PB-Exp-SAV-Plan.md` 实现两种方案。Plan 1 只演化总辅助变量，
Plan 2 同时演化总辅助变量与局部辅助变量，均以对数形式保存。
算法不使用 shift，传入 `--total-shift` 会报错；添加 `--relax` 可启用
文档中的对数辅助变量修正。Plan 1 用 `--eta` 指定允许回补的耗散比例，
要求为有限值且位于 `[0, 1]`，省略时默认 1。只有 relaxed Plan 1 接受
`--eta`；Plan 2 保留共同混合系数修正各分量的旧规则。

```bash
python -B run.py --problem quadratic --method pb_esav --scheme plan1 --loss split --lr 10 --steps 300 --lambda-reg 0
python -B run.py --problem quadratic --method pb_esav --scheme plan2 --loss split --lr 10 --steps 300 --lambda-reg 0
```

默认分别保存到 `results/quadratic/pb_esav_plan1/` 和
`results/quadratic/pb_esav_plan2/`；可用 `--output` 指定实验目录。
两组均使用 100 维、全 1 初值、CPU 和 float64。

固定步长 10、300 步，运行三组 relaxed Plan 1 实验：

```bash
python -B run.py --problem quadratic --method pb_esav --scheme plan1 --loss split --lr 10 --steps 300 --lambda-reg 0 --relax --eta 1
python -B run.py --problem quadratic --method pb_esav --scheme plan1 --loss split --lr 10 --steps 300 --lambda-reg 0 --relax --eta 0.95
python -B run.py --problem quadratic --method pb_esav --scheme plan1 --loss split --lr 10 --steps 300 --lambda-reg 0 --relax --eta 0.1
```

默认保存到 `results/quadratic/` 下的 `pb_esav_plan1_relaxed/`、
`pb_esav_plan1_relaxed_eta_0p95/` 和 `pb_esav_plan1_relaxed_eta_0p1/`。
每个目录保存含实际 `eta` 的 `config.json` 和逐步 `history.csv`；同一设置
重跑时覆盖该目录记录。普通版和已保存的 relaxed Plan 2 记录不受影响。
每次循环的计算顺序与 relaxation 公式见 `doc/PB-Exp-SAV-Plan.md`。

记录字段：`objective` 为总能量，`energy_1`、`energy_2` 为分量能量；
`log_r` 与 `modified_energy` 均为总辅助变量的对数，
`r_over_exp_E` 为该步状态下的总辅助变量比值。
Plan 2 另保存 `log_r_1`、`log_r_2` 和对应的 `r_1_over_exp_E_1`、
`r_2_over_exp_E_2`。relaxed 版本沿用这些字段，记录修正完成后的同一步状态。
SAV/PB-SAV 保持原有记录字段，修正能量为 `q²`。

### 从已有结果绘图

```bash
python -B plot_figure.py --input results/quadratic/pb_esav_plan1 --output results/quadratic/pb_esav_plan1/figure.png
python -B plot_figure.py --input results/quadratic/pb_esav_plan2 --output results/quadratic/pb_esav_plan2/figure.png
```

每张图展示真实目标能量（对数纵轴）和修正能量（线性纵轴）。
绘图脚本也支持已有 SAV/PB-SAV 实验目录，不重新运行实验。

传入多个实验目录可生成对比图，左图显示目标能量，右图显示辅助变量比值
（SAV/PB-SAV 为 `q/Q`，PB-Exp-SAV 为 `r/exp(E)`）：

```bash
python -B plot_figure.py --input results/quadratic/sav results/quadratic/sav_relaxed results/quadratic/pb_sav results/quadratic/pb_sav_relaxed results/quadratic/pb_esav_plan1 results/quadratic/pb_esav_plan2 results/quadratic/pb_esav_plan1_relaxed results/quadratic/pb_esav_plan1_relaxed_eta_0p95 results/quadratic/pb_esav_plan1_relaxed_eta_0p1 --output results/quadratic/comparison.png
```

原论文复现记录中的 SAV/PB-SAV 设置为：100 维、全 1 初值、CPU、float64、
步长 10、300 步、无正则项、总 shift 为 `1e-10`，各含普通版与 relaxed 版。
PB-SAV 使用两个能量分量，SAV 使用总能量；PB-Exp-SAV 两种方案使用相同
公共实验设置与两个分量，不使用 shift。图中包含原四条 SAV/PB-SAV 曲线、
普通 Plan 1 / Plan 2，以及三组 relaxed Plan 1，共九条曲线。
relaxed Plan 2 不参与本次绘图，已有数据保留。
Plan 1 三组 relaxed 曲线均为绿色：eta=1 使用实线，eta=0.95 使用方形标记虚线，
eta=0.1 使用三角标记点线；图例标注 eta。历史 relaxed Plan 1 配置缺少 eta 时按 1 读取。

### PB-SAV

使用 Split Energy Loss：

```bash
python run.py --problem quadratic --method pb_sav --loss split --lr 10 --steps 300 --lambda-reg 0
```

### SAV

使用 Total Energy Loss：

```bash
python run.py --problem quadratic --method sav --loss total --lr 10 --steps 300 --lambda-reg 0
```
