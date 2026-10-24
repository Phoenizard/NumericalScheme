# 实验结果归档

| 问题 | 报告 | naiveESAV / 旧版 ESAV 历史探索 | MESAV 历史记录 | ESAV 及其扩展 | BESAV 及其扩展 |
| --- | --- | --- | --- | --- | --- |
| 二次型 | [报告](../doc/experiments/quadratic.md) | [索引](quadratic/esav_exploration/index.json) | [索引](quadratic/mesav/index.json) | [索引](quadratic/esav/index.json) | [索引](quadratic/besav/index.json) |
| 非线性回归 | [报告](../doc/experiments/nonlinear_regression.md) | [索引](nonlinear_regression/esav_exploration/index.json) | [索引](nonlinear_regression/mesav/index.json) | [索引](nonlinear_regression/esav/index.json) | [索引](nonlinear_regression/besav/index.json) |

| 展示名称 | 辅助变量定义与定位 |
| --- | --- |
| naiveESAV | $r=\exp(E)$ |
| ESAV（旧版，已废弃） | $r=\exp(E)\sqrt{E+C}$；仅保留历史实验 |
| PB-ESAV（旧版，已废弃） | 总量 $r=\exp(E)$；局部 $r_i=\exp(E_i)\sqrt{E_i+C/M}$；保留历史 Plan 1 / 2 |
| BESAV / PB-BESAV / Relax | $r=\exp(E/(E+E^0))$，冻结初始能量；当前实验背景下值得继续探索 |
| MESAV | $r=\exp(E/(E^0+C))\sqrt{E+C}$；历史比较对象，不作为正式候选 |
| ESAV / PB-ESAV / Relax-PB-ESAV | $r=\exp(E/(4E^0))$，$E^0=E(\theta^0)$ 冻结；无 shift |

[旧版单能量 ESAV / Nonlinear / Armijo](nonlinear_regression/esav_exploration/armijo/single_esav_20261023/summary.json) 属于第一节历史记录。

现行 BESAV、ESAV、MESAV 及其 PB 方法通过独立 `relax` 开关选择 Relax；
历史名称与原始配置保持原样，绘图兼容两种记录方式。
接口合并的短程等价性核查见 [Relax 开关迁移验证](reproduction_relax_switch_20261024/README.md)，不作为新增正式实验。

```text
results/
├── quadratic/
│   ├── esav_exploration/       # naiveESAV 与已废弃旧版 ESAV
│   ├── mesav/                  # MESAV 历史实验
│   ├── besav/                  # BESAV 基础 / PB / Relax，含失败记录
│   └── esav/                   # 基础固定学习率 / Armijo、PB / Relax-PB
├── nonlinear_regression/
│   ├── esav_exploration/       # naiveESAV 与已废弃旧版 ESAV
│   ├── mesav/                  # MESAV 历史实验与曲率诊断
│   ├── besav/                  # BESAV 基础 / PB / Relax，含失败记录
│   └── esav/                   # 基础固定学习率 / Armijo、六组 M 的 PB / Relax-PB
├── reproduction_20261022/      # 历史代码、数据及已有核查记录
├── reproduction_besav_20261024/ # BESAV 源码、推导、迁移与归档核查
├── reproduction_20261024/      # ESAV 实验源码、理论笔记及归档核查
├── archive_20261022.json       # 原路径、方法名映射、哈希与复用关系
├── report_sources_20261022.json
└── archive_checks_20261022.json
```

- 每组目录保留原始 `config.json`、`history.csv` 和原有末态、试探、摘要及失败文件。各方法族的 `index.json` 提供原始目录与展示名称。
- 历史 `normalized_exp`、`pb_normalized_exp` 对应 MESAV、PB-MESAV；2026-10-23 的 `scaled_exp`、`pb_scaled_exp`、`relax_pb_scaled_exp` 对应 ESAV、PB-ESAV、Relax-PB-ESAV。
- 原始配置和 CSV 不改写名称或数值。报告中的 Adam 对应 weight decay 为 0 的历史 Adam / AdamW。
- MESAV 的 M=30 复用保留原归档结构。ESAV 的 PB 与 Relax-PB 合并展示，同一条 PB 轨迹不重复列图表。
- 失败组保持停止时的有效轨迹与失败标记，不将末值视为完成全部更新的结果，不延长或填充曲线。
- 单能量 Armijo 的 SAV 历史记录写满预算但出现停滞／辅助变量异常，另用 ‡ 标记，不与提前失败的 PB-SAV 记录拼接。
- 重绘的历史图使用 `_labelled.png`；原图保留。报告只展示优化目标，必要诊断另列，不展示测试 MSE。
- BESAV 的 23 组实验（22 组完成、1 组失败）已归档；原记录及历史方法名不改写。本轮只迁移、核查与重绘，未重跑完整实验。
- 比较按同等级 SAV 的最后有效目标值判断，提前失败仍参与末值比较并标注步数；已超过 SAV 时不继续排名，都未超过时再比较 BESAV 与尺度 ESAV。
- 项目 TMP 在归档核查后清空；两组 SAV shift 试验按用户要求删除、不归档。

[归档核对](archive_checks_20261022.json) · [复现资料](reproduction_20261022/README.md)

[ESAV 补充复现资料](reproduction_20261024/README.md)

[BESAV 复现与迁移资料](reproduction_besav_20261024/README.md)
