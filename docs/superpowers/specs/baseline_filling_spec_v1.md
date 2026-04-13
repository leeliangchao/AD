# Baseline Filling Spec V1

## 1. 文档目的

本文件用于回答当前 AD 仓库的两个问题：

1. 当前仓库到底已经完成到了什么程度。
2. 下一轮研究实现应该先补哪里，才能让基线结果具备正式决策价值。

本次判断只基于仓库内真实代码、配置、测试和已有输出，不基于 README 的目标描述做额外乐观推断。

## 2. 审计范围与证据

本次审计重点检查了以下区域：

- 主干层：`src/adrf/core/*`、`src/adrf/data/*`、`src/adrf/representation/*`、`src/adrf/normality/*`、`src/adrf/evidence/*`、`src/adrf/protocol/*`、`src/adrf/evaluation/*`
- 平台层：`src/adrf/runner/*`、`src/adrf/ablation/*`、`src/adrf/statistics/*`、`src/adrf/reporting/*`、`src/adrf/logging/*`、`src/adrf/checkpoint/*`、`src/adrf/benchmark/*`、`src/adrf/cli/*`、`src/adrf/utils/*`
- 配置层：`configs/experiment/*`、`configs/ablation/*`、`configs/runtime/*`、`configs/benchmark/*`
- 验证层：`tests/*`
- 结果层：`outputs/ablations/20260411_025101_paper_baseline_matrix_v3_audited/*`、`outputs/ablations/20260413_020619_diffusion_evidence_multiseed/*`

本次还执行了全量测试：

- `uv run pytest`
- 结果：`105 passed in 71.14s`

## 3. 直接回答 A-E

### A. 当前仓库的主干架构是否已经完整

结论：**已完整到“可运行研究管线”层面，但未完整到“研究决策级内核合同”层面。**

原因：

- 主干数据契约是清晰且闭环的：`Sample -> Representation -> NormalityArtifacts -> Evidence -> Evaluator`，见 `src/adrf/core/interfaces.py:13-74`。
- `OneClassProtocol` 真正在驱动训练与评估，见 `src/adrf/protocol/one_class.py:10-44`。
- `ExperimentRunner` 已能完成 registry 装配、runtime 解析、训练、评估、日志、checkpoint、report 导出，见 `src/adrf/runner/experiment_runner.py:48-168`。

但还存在一个重要不完整点：

- 抽象层里 `Protocol` 要求实现 `run(...)`，见 `src/adrf/core/interfaces.py:45-58`。
- 真实运行层却依赖 `BaseProtocol.train_epoch/evaluate`，见 `src/adrf/protocol/base.py:9-26` 与 `src/adrf/runner/experiment_runner.py:156-166`。
- 这意味着“运行时真实合同”与“抽象声明合同”没有完全收口。

所以答案不是“没架构”，而是：**架构骨架已成型，主干可跑，但核心合同尚未完全统一。**

### B. 哪些模块已经足够成熟，不该继续优先投入

优先级应下调的模块如下：

- `Sample` / `NormalityArtifacts` / evaluator 契约层：接口稳定、测试充分、已支撑所有方法族。
- `runner + registry + config instantiation`：实验装配链路已成熟。
- `logging + local report + checkpoint`：足够支撑研究迭代。
- `benchmark + ablation + multiseed aggregation + grouped table export`：平台能力已经超过“最小脚手架”，能稳定产出矩阵结果与论文表。
- `cli + runtime profile + device/dataloader runtime`：作为平台层已经够用，不该再优先扩表面面积。

这些层目前更适合“守住稳定性”，不适合继续作为主投入方向。

### C. 哪些模块虽然有实现，但仍是“研究极简版”

结论：**方法层几乎全部仍是研究极简版。**

具体包括：

- `feature_memory`：就是 memory bank + nearest distance，见 `src/adrf/normality/feature_memory.py:15-79`。
- `autoencoder`：两层卷积编码器 + 两层反卷积解码器，见 `src/adrf/normality/autoencoder.py:17-110`。
- `diffusion_basic`：单步噪声预测型最小 denoiser，不是完整扩散 AD 基线，见 `src/adrf/normality/diffusion_basic.py:19-157`。
- `diffusion_inversion_basic`：固定步数轨迹回放 + step cost，不是成熟的过程级扩散异常检测实现，见 `src/adrf/normality/diffusion_inversion_basic.py:15-104`。
- `reference_basic`：图像与 reference 直接拼接的最小条件模型，见 `src/adrf/normality/reference_basic.py:20-151`。
- `reference_diffusion_basic`：reference 条件噪声预测的最小版，见 `src/adrf/normality/reference_diffusion_basic.py:20-177`。
- `feature` 表征默认使用 `pretrained: false`、`freeze: true`，见 `configs/experiment/feature_baseline.yaml:11-15` 与 `src/adrf/representation/feature.py:18-39`，因此它更像“通路成立”而非“强表征基线”。
- 默认 experiment configs 全是开发/烟雾级预算：fixture 数据、`1 epoch`、小图输入，见 `configs/experiment/recon_baseline.yaml:15-23`、`configs/experiment/diffusion_baseline.yaml:15-24`、`configs/experiment/reference_baseline.yaml:16-23`、`configs/experiment/reference_diffusion_baseline.yaml:16-24`。

### D. 当前结果为什么还不足以正式拍板研究主线

原因不是“跑不通”，而是“证据不够强”。

主要有五点：

1. **官方 audited 结果只有 1 seed。**
   - `configs/ablation/paper_baseline_matrix_v3_audited.yaml:57` 明确是 `seeds: [0]`。
   - 这足以做平台审计，不足以做研究主线定夺。

2. **当前官方 paper 级覆盖只有 3 个类别，不是全量数据面。**
   - `paper_baseline_matrix_v3_audited` 只覆盖 `bottle/capsule/grid`，见 `configs/ablation/paper_baseline_matrix_v3_audited.yaml:6-12`。
   - 仓库里实际上已有更完整的 `data/mvtec` 数据目录，但当前官方结论没有利用它。

3. **当前 audited 结果显示 classical 明显领先，diffusion 家族整体偏弱。**
   - `feature_memory + feature_distance` 的 category mean 为 `0.815 +- 0.131` image AUROC、`0.805 +- 0.122` pixel AUROC，见 `outputs/ablations/20260411_025101_paper_baseline_matrix_v3_audited/paper_table_category_mean.md:9`。
   - `diffusion_basic + noise_residual` 为 `0.554 +- 0.026`，见同文件第 6 行。
   - `reference_diffusion_basic + noise_residual` 为 `0.543 +- 0.028`，见第 11 行。
   - `diffusion_inversion_basic` 两条 evidence 的 image AUROC 都只有约 `0.378`，见第 7-8 行。

4. **扩散家族在 multiseed 下呈现明显不稳定。**
   - `diffusion_evidence_multiseed` 的 bottle 结果中，image AUROC 标准差达到 `0.471`，见 `outputs/ablations/20260413_020619_diffusion_evidence_multiseed/paper_table.md:5-8`。
   - 这说明即便只在单一类别上，当前扩散家族也还没有形成可信基线。

5. **diffusers backend 目前只验证了“工程序列合同”，没有验证“研究结果等价性”。**
   - 现有测试只检查 artifact key 和 shape parity，见 `tests/test_diffusion_basic_backend_parity_smoke.py:17-56`、`tests/test_diffusion_inversion_backend_parity_smoke.py:17-56`。
   - 这说明 backend 切换的工程兼容性成立，但不说明指标层可以直接拿来做正式比较。

### E. 下一轮最值得投入的模块和顺序是什么

固定顺序如下：

1. **先冻结“官方基线合同”**
   - 明确哪些配置字段真生效，哪些只是声明未接线。
   - 明确 v1 正式 claim 只认 `legacy` backend。

2. **先补强传统基线**
   - `feature_memory + feature_distance`
   - `autoencoder + reconstruction_residual`

3. **再补强 conditional classical**
   - `reference_basic + conditional_violation`

4. **然后跑官方 3-seed、预算对齐、3 类别矩阵**
   - 先拿到可信的非扩散基线面。

5. **在此之后才继续投入 diffusion static**
   - `diffusion_basic + noise_residual`

6. **再投入 diffusion process**
   - `diffusion_inversion_basic + {path_cost, direction_mismatch}`

7. **最后再投入 conditional-diffusion**
   - `reference_diffusion_basic + noise_residual`

理由很直接：当前最强结果来自传统基线，而不是扩散家族；如果先继续深挖 diffusion，只会在“弱且不稳”的地基上反复调参。

## 4. 主干层审计

| 层 | 结论 | 判断 |
| --- | --- | --- |
| Sample | 已完成且稳定 | `Sample` 已统一 image/label/mask/reference/views/metadata 合同，足够支撑当前所有家族。 |
| Representation | 已完成但研究极简 | `pixel` 很稳定；`feature` 作为管线组件稳定，但默认不是强表征基线。 |
| Normality | 已完成但研究极简 | 所有方法族都已接入，但几乎全部是最小可运行实现。 |
| Artifacts | 已完成且稳定 | `NormalityArtifacts` 是本仓库最成功的中间合同之一。 |
| Evidence | 已完成但研究极简 | evidence 合同稳定，但具体 scoring 仍是最小启发式。 |
| Protocol | 已完成但抽象不收口 | one-class workflow 已稳定；抽象接口与运行接口仍有裂缝。 |
| Evaluation | 已完成且稳定 | image/pixel AUROC/AUPR 与 map aggregation 已够用。 |

## 5. 平台层审计

| 层 | 结论 | 判断 |
| --- | --- | --- |
| logging | 已完成且稳定 | 本地 `RunLogger` 已足够，SwanLab 也有退化兼容。 |
| checkpoint | 已完成但最小 | 只覆盖 trainable `state_dict` 保存/恢复，够用但不算丰富。 |
| benchmark | 已完成且稳定 | 顺序执行、结果收集、摘要导出都已成立。 |
| ablation | 已完成且稳定 | 组合展开、兼容性过滤、矩阵执行、聚合导出已形成闭环。 |
| multiseed | 已完成且稳定 | 已能按 group 聚合多 seed 结果。 |
| statistics | 已完成但最小 | 只有 mean/std/min/max/median 级聚合，没有显著性与置信区间。 |
| reporting | 已完成且稳定 | 单 run、benchmark、ablation summary、grouped paper table 都可用。 |
| cli | 已完成且稳定 | experiment/benchmark/ablation/report 子命令齐全。 |
| runtime | 已完成且稳定 | device、AMP、DataLoader runtime、runtime profile 已接好。 |

## 6. 方法族审计

| 方法族 | 当前对应实现 | 当前状态 | 是否应作为下一轮优先主线 |
| --- | --- | --- | --- |
| traditional | `feature_memory`、`autoencoder` | 可跑、最小、但已有较强结果 | 是 |
| diffusion static | `diffusion_basic` | 可跑、弱、目前不稳 | 否，排在传统之后 |
| diffusion process | `diffusion_inversion_basic` | 可跑、弱、结果波动更大 | 否，排在 diffusion static 之后 |
| conditional | `reference_basic` | 可跑、最小、当前不比传统强 | 是，但晚于传统 |
| conditional-diffusion | `reference_diffusion_basic` | 可跑、弱、目前未体现条件收益 | 否，最后处理 |

## 7. 稳定层

本轮审计认定以下模块属于“稳定层”：

- `src/adrf/core/sample.py`
- `src/adrf/core/artifacts.py`
- `src/adrf/evaluation/*`
- `src/adrf/runner/experiment_runner.py`
- `src/adrf/logging/*`
- `src/adrf/checkpoint/io.py`
- `src/adrf/benchmark/*`
- `src/adrf/ablation/*`
- `src/adrf/statistics/aggregate.py`
- `src/adrf/statistics/table_export.py`
- `src/adrf/reporting/*`
- `src/adrf/cli/*`
- `src/adrf/utils/runtime.py`

这些层的核心问题不是“没有实现”，而是“不要再继续优先扩平台”。

## 8. 极简层

本轮审计认定以下模块属于“已完成但仍是研究极简版”：

- `src/adrf/representation/feature.py`
- `src/adrf/normality/feature_memory.py`
- `src/adrf/normality/autoencoder.py`
- `src/adrf/normality/diffusion_basic.py`
- `src/adrf/normality/diffusion_inversion_basic.py`
- `src/adrf/normality/reference_basic.py`
- `src/adrf/normality/reference_diffusion_basic.py`
- `src/adrf/evidence/*`
- `configs/experiment/*`

这些层不是空白，但离“能支撑研究拍板”的成熟基线仍有距离。

## 9. 关键架构债

### 9.1 Protocol 合同分裂

- 抽象要求 `run(...)`，运行时依赖 `train_epoch/evaluate`。
- 这会导致新实现者读抽象接口时产生错误理解。

### 9.2 配置合同存在“伪生效字段”

在 `paper_baseline_matrix_v1/v2/v3` 配置里，存在多组声明性字段，但仓库执行链路并未真正使用它们：

- `output.summary_dir`
- `output.keep_per_seed_runs`
- `defaults.run_mode`
- `defaults.save_checkpoint`
- `defaults.export_report`
- `defaults.export_predictions`
- `defaults.logger`

这些字段在配置中出现，但在 `src` 与 `scripts` 中没有形成对应执行合同；它们当前更像“规划字段”，不是“真实合同”。

### 9.3 audit 仍是平台外脚本重放，不是一等运行产物

- `AblationRunner` 只负责记录 audit config 和导表，见 `src/adrf/ablation/runner.py:84-113`。
- 真正的 failure analysis 依赖 `scripts/export_failure_analysis.py` 对已完成 run 重新加载或重新训练后再导出，见 `scripts/export_failure_analysis.py:29-102`。
- 这说明 audit 目前是“后处理扩展”，不是“run 生命周期内的一等产物”。

### 9.4 统计层仍停留在聚合，不是推断

- 目前 multiseed 统计只有 `mean/std/min/max/median/count`，见 `src/adrf/statistics/aggregate.py:10-35`。
- 对研究拍板来说，这还不够。

## 10. 关键研究债

### 10.1 当前 audited 基线不是官方比较版

- 只有 1 seed。
- 只有 3 类别。
- 仍然不足以决定“研究主线是传统还是扩散”。

### 10.2 当前默认 experiment configs 仍是开发烟雾配置

- 它们的作用是“证明通路成立”，不是“证明方法有效”。

### 10.3 扩散家族当前缺乏可信基线表现

- diffusion static 目前接近弱基线。
- diffusion process 目前在 image-level 指标上尤其不稳。
- conditional-diffusion 还没有体现出 reference 条件带来的明确收益。

### 10.4 conditional 家族尚未证明“条件信息真的有用”

- `reference_basic` 当前能跑，但没有在 audited 结果上稳定压过传统最强项。
- 条件建模目前只是最小拼接式结构，不足以据此给条件化方向定主线。

## 11. 下一轮的固定判断基线

为了避免下一轮继续漂移，先锁定以下判断：

- 下一轮**不新增方法族**。
- 下一轮**不继续扩平台表面面积**。
- 下一轮的核心任务是**让现有五个方法族形成可信比较基线**。
- 下一轮的官方比较先固定在：
  - 数据：`bottle/capsule/grid`
  - seed：`3`
  - backend：`legacy`
  - 指标：`image_auroc/pixel_auroc/pixel_aupr/train_time/total_time`

## 12. 最终结论

当前仓库已经不是“空框架”，而是一个：

- 主干闭环已成立
- 平台能力已经相当完整
- 方法层全面接入但研究成熟度不足
- 可以做 baseline filling，但还不能据此拍板研究主线

下一轮的正确方向不是继续“加新东西”，而是先把现有基线做成可信证据。
