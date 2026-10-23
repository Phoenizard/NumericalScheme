# 实验结果归档

| 问题 | 报告 | naiveESAV / ESAV 探索 | MESAV 及其扩展 |
| --- | --- | --- | --- |
| 二次型 | [报告](../doc/experiments/quadratic.md) | [索引](quadratic/esav_exploration/index.json) | [索引](quadratic/mesav/index.json) |
| 非线性回归 | [报告](../doc/experiments/nonlinear_regression.md) | [索引](nonlinear_regression/esav_exploration/index.json) | [索引](nonlinear_regression/mesav/index.json) |

```text
results/
├── quadratic/
│   ├── esav_exploration/       # 既有 PB 对比与 naiveESAV 失效诊断
│   └── mesav/                  # single / PB 固定学习率、Armijo、Relax
├── nonlinear_regression/
│   ├── esav_exploration/       # 既有六组 M 对比
│   └── mesav/                  # 固定学习率、Armijo M 扫描、曲率诊断
├── reproduction_20261022/      # 历史代码、数据及已有核查记录
├── archive_20261022.json       # 原路径、方法名映射、哈希与复用关系
├── report_sources_20261022.json
└── archive_checks_20261022.json
```

- 每组目录保留原始 `config.json`、`history.csv` 和原有的末态、试探、摘要及失败文件；`archive.json` 提供统一命名与来源。
- `normalized_exp`、`pb_normalized_exp` 及本轮 TMP 中的相应 ESAV 名称，分别映射为 MESAV、PB-MESAV 及其 Relax 版本。旧 ESAV 探索属于独立方案。
- 原始配置和 CSV 不改写名称或数值。报告中的 Adam 对应 weight decay 为 0 的历史 Adam / AdamW。
- M=30 的六项显式复用只保留一份正式实验目录。不同版本的配置、摘要和历史列保存在 `source_variants/`；相同末态及试探文件共用。
- 失败组保持停止时的有效轨迹与失败标记，不将末值视为完成全部更新的结果，不延长或填充曲线。
- 原 results 目录与 TMP 原件暂时保留；正式报告引用上述归档。本轮只复制、核对及重绘，没有重新优化。

[归档核对](archive_checks_20261022.json) · [复现资料](reproduction_20261022/README.md)
