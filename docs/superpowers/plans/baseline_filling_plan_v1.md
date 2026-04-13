# Baseline Filling Plan V1

## 1. 目标

下一轮只做一件事：

**把当前五个方法族补成“可正式比较”的研究基线，并给出足以支持方向判断的结果。**

明确不做的事：

- 不新增方法
- 不继续扩平台表面面积
- 不把主要时间花在 CLI、logger、report 样式、额外导出功能上

## 2. 执行原则

### 2.1 先证实传统基线，再决定是否继续押注扩散

当前 audited 结果里最强的是 `feature_memory + feature_distance`，不是 diffusion 家族。

因此下一轮必须遵守下面的顺序：

1. 先把传统基线做稳。
2. 再把 conditional classical 做成可信对照。
3. 之后再回到 diffusion static。
4. diffusion process 与 conditional-diffusion 放在更后面。

### 2.2 先修“合同不清”，再跑“官方矩阵”

在继续做研究比较前，必须先把“哪些配置真的生效”说清楚。否则后续矩阵结果会继续混入伪合同字段。

### 2.3 下一轮的官方 claim 只认一套设置

下一轮默认官方比较合同固定为：

- 数据集：`mvtec_bottle`、`mvtec_capsule`、`mvtec_grid`
- seed 数：`3`
- backend：`legacy`
- runtime：预算对齐版
- 输出：per-run report、matrix summary、grouped tables、failure analysis

## 3. 固定优先级顺序

### P0. 冻结官方 baseline 合同

目标：

- 明确 v1 官方结果到底认哪套配置。
- 清理或显式降级当前配置里的伪合同字段。

本阶段聚焦模块：

- `src/adrf/protocol/*`
- `src/adrf/runner/experiment_runner.py`
- `src/adrf/ablation/*`
- `configs/ablation/paper_baseline_matrix_v2_budgeted.yaml`
- `configs/ablation/paper_baseline_matrix_v3_audited.yaml`

必须完成的事：

- 决定并写死“运行时真实协议”：
  - v1 默认以 `train_epoch/evaluate` 为真实执行合同。
  - 不在本阶段优先做 protocol 大重构。
- 盘点并处理以下伪合同字段：
  - `output.summary_dir`
  - `output.keep_per_seed_runs`
  - `defaults.run_mode`
  - `defaults.save_checkpoint`
  - `defaults.export_report`
  - `defaults.export_predictions`
  - `defaults.logger`
- 对这些字段采用统一策略：
  - 要么接线并验证。
  - 要么从“官方 baseline config 合同”中明确移除。
- 明确 v1 正式比较只认 `legacy` backend。

本阶段完成标准：

- 看配置就能知道哪些字段是官方合同，哪些不是。
- 实现者不再需要猜“某字段是否真的生效”。

停止条件：

- 如果配置合同仍然含糊，禁止继续做方法对比矩阵。

### P1. 做稳 traditional baselines

目标：

- 把 `feature_memory + feature_distance` 与 `autoencoder + reconstruction_residual` 做成可信对照组。

本阶段聚焦模块：

- `src/adrf/representation/feature.py`
- `src/adrf/normality/feature_memory.py`
- `src/adrf/normality/autoencoder.py`
- `src/adrf/evidence/feature_distance.py`
- `src/adrf/evidence/reconstruction_residual.py`
- 对应 experiment/ablation configs

必须完成的事：

- 明确 `feature_memory` 是不是继续使用 `pretrained: false`。
  - 默认选择：不要再把它当官方强基线；官方比较应切到更可信的表征设置。
- 检查 autoencoder 的训练预算与输入尺寸是否已经足以代表“正式 baseline”，而不是 smoke 版本。
- 让这两条 classical baselines 在 3 个官方类别、3 seeds 下具备可重复结果。

本阶段完成标准：

- `feature_memory` 与 `autoencoder` 的结果不再依赖单 seed 或 fixture 设置。
- 它们成为后续 diffusion 家族的正式对照组。

停止条件：

- 如果 traditional baselines 仍不稳，禁止继续把主要精力投向 diffusion。

### P2. 做稳 conditional classical baseline

目标：

- 判断 `reference_basic + conditional_violation` 是否真能代表“条件化有收益”。

本阶段聚焦模块：

- `src/adrf/normality/reference_basic.py`
- `src/adrf/evidence/conditional_violation.py`
- 对应 experiment/ablation configs

必须完成的事：

- 校准 reference 使用方式，使其成为可信的条件对照。
- 不以“条件化概念成立”为目标，而以“条件化是否真的带来稳定收益”为目标。

本阶段完成标准：

- `reference_basic` 在官方矩阵里有清晰定位：
  - 要么成为可信条件基线。
  - 要么被证明当前收益不足，不再继续优先追加复杂条件化。

停止条件：

- 如果 `reference_basic` 仍未展现稳定收益，conditional-diffusion 继续后置。

### P3. 跑官方 3-seed budgeted matrix

目标：

- 产出第一版真正可用于研究判断的官方矩阵。

官方矩阵配置要求：

- 使用 `paper_baseline_matrix_v2_budgeted` 风格的预算对齐设置。
- 但执行时必须明确它是“official baseline matrix v1”。
- 使用 `3 datasets x 3 seeds`。
- 先不扩大到全量 15 类别。

必须产出的结果：

- per-run reports
- matrix summary
- grouped paper tables
- category mean table
- by-axis table

本阶段完成标准：

- 有一套 3-seed、预算对齐、3 类别的正式比较结果。
- 传统 / 条件 / diffusion static / diffusion process / conditional-diffusion 五条线能在同一合同下比较。

停止条件：

- 没有 3-seed 正式矩阵，禁止讨论“研究主线已经确定”。

### P4. 再投入 diffusion static

目标：

- 判断 `diffusion_basic + noise_residual` 是否值得继续保留为主线候选。

本阶段聚焦模块：

- `src/adrf/normality/diffusion_basic.py`
- `src/adrf/evidence/noise_residual.py`
- `src/adrf/diffusion/*`

必须完成的事：

- 先只做 `legacy` backend 下的正式比较。
- `diffusers` backend 只作为工程兼容项，不作为主结果来源。
- backend parity 验证要从“artifact shape 一致”升级到“score 或 metric 层面可比较”。

本阶段完成标准：

- diffusion static 在 official matrix 中至少能形成“弱可比基线”。

停止条件：

- 如果 diffusion static 仍显著弱于 traditional 且不稳，不再优先扩大它的研究投入。

### P5. 再投入 diffusion process

目标：

- 判断 `diffusion_inversion_basic + {path_cost, direction_mismatch}` 是否有独立研究价值。

本阶段聚焦模块：

- `src/adrf/normality/diffusion_inversion_basic.py`
- `src/adrf/evidence/path_cost.py`
- `src/adrf/evidence/direction_mismatch.py`

必须完成的事：

- 先解释当前 `path_cost` 与 `direction_mismatch` 为什么结果几乎并列。
- 明确 process evidence 到底提供了什么增量，而不是继续把两种 evidence 机械并列。

本阶段完成标准：

- process baseline 的价值被说清楚：
  - 要么有清晰增益。
  - 要么只是 diffusion static 的复杂弱化版。

停止条件：

- 如果 process family 仍无法证明增益，不应继续作为主线中心。

### P6. 最后投入 conditional-diffusion

目标：

- 判断 `reference_diffusion_basic + noise_residual` 是否真的比 unconditional diffusion 更值得保留。

本阶段聚焦模块：

- `src/adrf/normality/reference_diffusion_basic.py`
- `src/adrf/evidence/noise_residual.py`

必须完成的事：

- 只在 unconditional diffusion 已被做稳之后再做。
- 不允许把 conditional-diffusion 当作“跳过前面问题的替代方案”。

本阶段完成标准：

- 能明确回答“条件信息在扩散家族里有没有带来稳定增益”。

停止条件：

- 如果 unconditional diffusion 本身都不成立，则 conditional-diffusion 不再优先。

### P7. audited failure analysis 作为收尾，而不是替代主证据

目标：

- 用 failure analysis 解释 official matrix 结果，而不是替代它。

本阶段聚焦模块：

- `scripts/export_failure_analysis.py`
- `scripts/export_audit_tables.py`
- `outputs/ablations/<official_matrix>/...`

必须完成的事：

- 在 official matrix 完成后导出每个方法族的失败样本 bundle。
- 用它解释失败模式与方法边界。
- 不允许先做 audit 图，再倒推出研究结论。

本阶段完成标准：

- audit 成为“解释层”，不是“证据主层”。

## 4. 下一轮的固定交付物

下一轮结束时，至少要交付：

- 一套官方 3-seed budgeted baseline matrix 结果
- `paper_table.md`
- `paper_table_category_mean.md`
- `paper_table_by_axis.md`
- 每个方法族的 failure analysis bundle
- 一份明确的研究判断备忘录：
  - 哪条线最强
  - 哪条线最稳
  - 哪条线继续投入
  - 哪条线降级为对照

## 5. 验收门槛

只有同时满足下面条件，下一轮才允许说“baseline filling 已完成”：

- 平台测试仍全部通过。
- 官方比较使用的是明确生效的配置合同。
- official matrix 至少覆盖 `3 datasets x 3 seeds`。
- 传统基线与条件基线已经稳定。
- diffusion static/process/conditional-diffusion 至少完成一次同合同下正式比较。
- failure analysis 是基于 official matrix 的解释层输出，而不是临时样例展示。

## 6. 默认决策

为避免下一轮继续摇摆，本计划先锁定以下默认决策：

- 不优先重构平台大框架。
- 不优先扩全量数据类别。
- 不优先推广 `diffusers` backend 为正式结果 backend。
- 不优先扩新方法。
- 先把 strongest classical baseline 做成正式比较锚点。

## 7. 一句话执行顺序

**先清合同，再稳传统，再稳条件，再跑官方 3-seed 矩阵，之后才继续做 diffusion static、diffusion process、conditional-diffusion，最后用 audited failure analysis 做解释。**
