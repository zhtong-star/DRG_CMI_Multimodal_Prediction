# DRG_CMI_Multimodal_Prediction

[English](README.md) | 简体中文

本仓库提供 **LG-GCMN（Label-Guided Gated Cross-Modal Network，标签引导门控跨模态网络）** 的 PyTorch 实现，用于早期单标签多分类 DRG 预测和队列层面的病例组合指数（CMI）估计。模型融合早期住院临床文本与按小时整理的生理测量，分别支持 MS-DRG 和 APR-DRG 队列。

本实现和文档已对照用户提供的 2026 年 11 月 27 日论文版本及图 1。论文对照报告列出代码修正、论文内部冲突和未解决的实验溯源问题。软件检查或合成数据训练通过，不代表复现了论文报告的结果。

## 研究与数据

论文描述了一项使用 MIMIC-III v1.4 的模型开发与验证研究，数据来自 Beth Israel Deaconess Medical Center（BIDMC）2001—2012 年的 ICU 记录。论文报告最终多模态队列包含 19,789 次 MS-DRG 住院和 27,394 次 APR-DRG 住院，分别包含 611 和 908 个 DRG 类别，并排除了 3 和 20 次结构化数据不完整的住院。这些数字是论文报告值；代码没有写死这些数值，本仓库也不声称已独立复现。

论文纳入条件包括年龄至少 18 岁、单次 ICU 住院、ICU 住院时长至少 24 小时、至少一条时间位于医院入院至 ICU 入科后 48 小时范围内的合格笔记，以及可用的标签、权重和多模态输入。论文未讨论转科，本实现保留原代码中的无转科条件。APR-DRG 对重复描述按源代码的“保留首条记录”策略处理；若去重后仍存在冲突编码，则排除该住院。默认实现并不执行“选择最高严重度”的规则。

队列数量存在实质冲突。此前对本机可用元数据表进行只读筛查时，年龄、单次 ICU 和无转科筛选后得到 MS 21,938 次、APR 29,596 次住院。再执行论文所述 ICU LOS ≥24 小时条件后，分别减少到 18,144 和 24,673 次，且尚未筛除无合格笔记或结构化数据的住院。这两个上限都低于论文报告的最终队列数量。预处理代码遵循明确写出的 ≥24 小时方法，并记录实际数量；要解决差异，需要核对论文队列表和历史队列生成记录。详见[论文对照报告](docs/MANUSCRIPT_ALIGNMENT.md)。

DRG 标签和 2013 财年相对权重用于定义预测目标及计算 CMI，不会作为模型输入特征。本仓库不包含 MIMIC 数据、DRG 规则表、预训练模型权重或患者级输出。请仅在适用的数据访问条款下获取和使用这些资源。

## 任务与架构

模型根据每次住院在 24 或 48 小时预测时间点之前可获得的信息，预测一个 MS-DRG 或 APR-DRG 标签。分类以住院为单位。CMI 评估则只使用固定测试集中每位患者最早的合格医院住院；论文邻近段落中另有“全部住院”的表述，相关冲突记录在论文对照报告中。

```text
Notes from hospital admission through ICU admission + prediction horizon
  -> placeholder removal / adjacent-paragraph deduplication / tokenization
  -> Clinical-Longformer (up to 4096 tokens)
  -> masked label-wise attention -> D [batch, classes, hidden size]

29 hourly physiological variables before the prediction horizon
  -> hourly means / training-fold imputation and standardization
  -> 29 values + 29 observation indicators -> masked GRU -> T [batch, time, hidden size]

D as query; T as key/value -> masked scaled cross-attention -> S
z = sigmoid(Wg [D; S] + bg)
F = z * D + (1 - z) * S
  -> class-specific linear scores -> raw logits
  -> training: categorical cross-entropy(logits + lambda * log(class prior))
  -> inference: argmax(raw logits) -> DRG relative weight -> cohort CMI
```

模型按照图 1 和公式 1—10 实现标签级文本注意力、GRU 时间编码、标签引导的跨模态注意力、逐元素门控融合和类别专属评分。论文规定 `lambda = 1`；LG-GCMN 默认在训练损失中使用该调整，类别频率仅由对应折的训练集计算。推理时直接对原始 logits 取最大值，不再进行先验调整。门控值表示模型学习到的逐特征文本贡献，不是临床事件发生概率。

论文规定 29 个结构化变量。代码使用 `extract_structured.py` 中列出的 8 项生命体征和 21 项实验室检查，并为每个变量追加一个观测指示通道。代码先按原始时间筛选，再按 ICU 入科后的 `[0, T)` 小时区间聚合。同一变量、同一小时内的有限测量取平均；若存在 `ERROR` 列，则排除标记为 `ERROR=1` 的记录，并将华氏温度转换为摄氏度。

每一折仅使用训练集实际观测值拟合中位数、均值和总体标准差。序列起始处和整个训练折都缺失的值使用训练集对应中位数填补，后续缺失值向前填充。观测指示通道用于区分原始测量和填充值。有效时间掩码覆盖第 0 小时到预测窗口内最后一个存在观测的小时箱；之后的位置为零填充，并在 GRU 和跨模态注意力中屏蔽。论文没有说明具体的序列结束规则。若住院在第 24 小时前没有结构化观测、但之后有观测，该住院仍保留在共同的 48 小时队列中，其 24 小时结构化序列会全部被屏蔽。

文本处理在相邻段落清理完成前保留段落边界，去除 MIMIC 去标识化占位符，排除出院小结和行政诊断伪笔记，并按相对于 ICU 入科的记录时间排序。笔记可以早于 ICU 入科，但不得早于医院入院时间；预测时间窗以 ICU 入科为起点。只有当相邻段落的余弦相似度和归一化词级 Levenshtein 相似度都严格高于设定阈值时，代码才删除后一个段落。论文提到了阈值 `τ`，但未报告具体数值；`0.95` 是代码默认值，会写入队列和运行溯源信息，不能视为已找回的历史实验设置。文本会按配置长度截断或填充，并屏蔽填充位置。Longformer 使用首 token 全局注意力，这是代码实现细节。

对照模型为 GRU、CAML、Clinical-Longformer 和 Concat。当前实验调度器默认仅在 MS 队列运行论文表 3 的三种消融：将文本编码器替换为 ClinicalBERT、将标签级注意力替换为全局文本池化、将 GCM 替换为拼接。也可以显式设置为 APR 额外运行消融。基线和消融模型使用其对应队列最终确定的样本和患者级划分。

## 项目结构

```text
.
├── README.md / README_CN.md
├── requirements_multimodal.txt / check_setup.py
├── clinical_text.py             # 笔记筛选与段落清理
├── create_cohorts.py            # 标签、纳入条件和早期笔记队列
├── extract_structured.py        # 29 项生理变量及小时级数据
├── finalize_cohorts.py          # 完整性检查与患者级折叠
├── protocol.py                  # schema、划分、哈希和 CMI 工具
├── data_utils_multimodal.py     # 文本和结构化数据加载
├── models_multimodal.py         # LG-GCMN 和对照模型
├── train_utils_multimodal.py    # 训练、早停和推理
├── options_multimodal.py        # 模型与训练参数
├── main_multimodal.py           # 训练和已保存模型评估
├── run_experiments.py           # 多模型和消融命令调度
├── eval_utils.py / eval_cc_mcc.py
├── generate_report.py / visualize_attention.py
├── prepare_early_notes.py / preprocessing.py
├── docs/MANUSCRIPT_ALIGNMENT.md
├── docs/VALIDATION.md
└── tests/                       # 单元、合成集成和数据审计
```

`.gitignore` 会排除私有输入和运行产物。示例中的 `Clinical-Longformer/`、`ClinicalBERT/`、`rules/`、`mimic/`、`data_n/`、`embeddings/` 和 `results/` 都是本地资源路径，不是仓库分发文件。

## 环境

请使用 `requirements_multimodal.txt` 指定的依赖版本；当前 CPU 验证环境为 Python 3.13、PyTorch 2.9.0、Transformers 4.57.6、tokenizers 0.22.2、NumPy 2.2.6、pandas 2.3.3、scikit-learn 1.7.2 和 SciPy 1.16.3。请先根据本机 CPU/CUDA 环境安装合适的 PyTorch，再安装其余依赖。4096 token 模型训练需要较多资源；论文报告的环境是单张 24 GB 显存的 NVIDIA RTX 4090，这不保证其他环境也能运行。

```bash
python -m pip install -r requirements_multimodal.txt
python check_setup.py
python check_setup.py --model_dir Clinical-Longformer
```

模型目录需包含相互兼容的配置、tokenizer 和预训练权重文件。CAML 二进制 word2vec 加载功能需要额外安装 `gensim`；注意力图绘制功能需要额外安装 `matplotlib`。本仓库未提供模型权重或下载链接。

## 数据准备

将获得授权的 MIMIC-III v1.4 CSV 文件放在 `mimic/` 下，包括 `PATIENTS.csv`、`ADMISSIONS.csv`、`ICUSTAYS.csv`、`DRGCODES.csv`、`NOTEEVENTS.csv`、`CHARTEVENTS.csv` 和 `LABEVENTS.csv`。将本地准备的 2013 年规则表放在 `rules/MSDRG_RULE13.csv` 和 `rules/APRDRG_RULE13.csv`。每个规则表必须包含 `DRG_CODE`、`DESCRIPTION` 和正数 `WEIGHT`；若存在 `MDC` 列，代码也会使用。本仓库不包含这些数据和规则文件。

从仓库根目录运行。请使用新的 `data_n` 目录；脚本会拒绝覆盖或混合已有的预处理版本。

```bash
python create_cohorts.py --mimic_dir mimic --rule_dir rules --data_dir data_n --all_cohorts --note_similarity_threshold 0.95
python extract_structured.py --mimic_dir mimic --data_dir data_n --cohort ms
python extract_structured.py --mimic_dir mimic --data_dir data_n --cohort apr
python finalize_cohorts.py --data_dir data_n --rule_dir rules --cohort ms
python finalize_cohorts.py --data_dir data_n --rule_dir rules --cohort apr
```

每个队列目录先包含 `cohort_candidates.csv`、`text_raw/` 和 `structured/`；完成后还包括 `cohort.csv`、`splits.csv`、`test_stays.csv`、`exclusions.csv`、`input_manifest.csv`、`labels.json`、`cohort_audit.json`、`extraction_audit.json` 和 `finalization_audit.json`。审计文件保存本地实际筛选数量和排除原因，不会声称达到论文队列总数。schema 版本 4 要求在文本段落、时间戳或 padding 策略改变后重新生成数据。

## 配置与训练

`options_multimodal.py` 定义命令行默认值。论文明确给出的主要设置包括 Longformer 最多 4096 个 token、batch size 8、AdamW 文本编码器学习率 `1e-5`、其他模块学习率 `1e-3`、5 折患者级开发、固定 10% 患者测试集、24/48 小时评估，以及 LG-GCMN 的 `lambda=1`。代码按患者数的 10% 向上取整，默认划分 seed 为 1443。论文没有报告梯度累积步数；`main_multimodal.py` 默认值为 1，实验调度器要求显式填写。请记录实际使用的值。论文未规定的其他结构与优化器设置沿用代码默认值。

在一个队列上训练提出的模型：

```bash
python main_multimodal.py --data_dir data_n --rule_dir rules --cohort ms --model LGGCMN --bert_model_name Clinical-Longformer --logit_adjustment 1 --gradient_accumulation_steps 1 --output_dir results --run_name ms_lggcmn --verify_data_hashes
```

上面的命令使用当前代码默认的 1 个梯度累积步数，并不表示这是论文历史实验设置。省略 `--cohort ms` 时使用 MS 队列；选择 `apr` 则运行 APR 队列。不提供 `--fold` 时，入口会训练 5 个独立折；每折只用训练集拟合统计量和类别先验，依据验证集 macro-F1 选择最佳 epoch，并用同一固定测试患者分别评估两个时间窗。每组配置应使用新的 `--run_name`。

如需只打印对照模型和论文消融命令、不启动训练，需要提供本地 CAML 和 ClinicalBERT 资源路径：

```bash
python run_experiments.py --cohorts ms apr --ablations --ablation_cohorts ms --clinicalbert_dir ClinicalBERT --pretrained_word_vectors embeddings/clinical.bin --gradient_accumulation_steps 1 --run_prefix study_v1 --dry_run
```

删除 `--dry_run` 后会执行输出的实验。默认会在两个队列分别训练 5 种模型，并仅在 MS 队列额外运行 3 种消融。请将累积步数改成实验实际使用值，并和输出一同保存。`run_experiments.sh` 是调用同一 Python 调度器的 Bash 封装。

## 评估与输出

在保存的固定测试患者上重新评估 5 折模型：

```bash
python main_multimodal.py --eval_model results/ms_lggcmn --data_dir data_n --rule_dir rules --hours 24 48 --eval_cc_mcc --verify_data_hashes
```

`--eval_model` 也可以指定单个 `fold_N` 目录。评估过程会校验已保存队列、规则、划分和模型文件。分类指标使用所有测试住院：macro/micro F1、macro/micro AUC、top-1 和 top-5 accuracy。Macro-F1 使用完整已保存类别集合，缺失类别 F1 按 0 计。如果某类别缺少阳性样本或阴性样本，则对全部类别求平均的 macro-AUC 未定义，结果保存为 `null`；同时记录可定义类别的平均 AUC 和未定义类别数量。论文没有规定这种边界情形，因此代码将处理方式明确记录。

CMI 将预测和真实 DRG 标签映射到 2013 财年相对权重，然后计算测试集中每位患者首个合格住院的平均权重。`APE_CMI_percent` 定义为 `100 × |预测 CMI − 真实 CMI| / 真实 CMI`。额外的相对权重 MAE、RMSE 和 Spearman 相关系数使用全部测试住院。五折汇总是五个独立评估指标的算术平均，不是集成预测。

为某个已保存 LG-GCMN 病例生成报告时，请使用该折 `test.csv` 中的 `stay` 值：

```bash
python generate_report.py --model_path results/ms_lggcmn/fold_0 --stay TEST_STAY --hour 48
python visualize_attention.py --report results/ms_lggcmn/fold_0/case_reports/TEST_STAY_48h.json --output results/ms_lggcmn/fold_0/attention.png
python prepare_early_notes.py --data_dir data_n --cohort ms --hour 24 --note_similarity_threshold 0.95 --output results/early_notes_24h.csv
```

病例报告包含预测类别对应的文本和时间注意力、平均文本门控值、互不重叠的五词片段，以及最大注意力小时箱对应的原始单位小时均值和观测数量。`peak_time_hours` 是该小时箱的起始时间；报告另行保存注意力加权质心。论文表 4 给出的小数小时峰值没有说明如何从小时级输入得到，本实现不声称能够重建该结果。注意力和门控值是模型描述性指标，不代表因果解释或临床事件概率。患者级报告和导出的笔记必须保密。

典型运行输出如下：

```text
results/<run_name>/
  run_config.json  data_provenance.json  environment.json  initialization.json
  text_encoder_resources/  encoder_resources_sha256.json
  summary.json
  fold_0/ ... fold_4/
    config.json  train.csv  validation.csv  test.csv
    struct_stats.json       # 结构化模型
    vocabulary.json         # CAML
    checkpoint.pt  history.csv  training_summary.json  completed.json
    eval_24h/  eval_48h/
      predictions.csv  logits.npz  metrics.json
      cc_mcc_metrics.json  cc_mcc_cases.csv  # 可选 MS 分析
    case_reports/            # 可选私有报告
```

为保证可复现性，请保留 MIMIC 版本、2013 规则表、预处理审计、固定测试患者名单、折叠清单、笔记阈值、初始化资源哈希、软件环境、完整命令参数（包括梯度累积步数）、checkpoint、拟合统计量和逐折预测。`--verify_data_hashes` 可校验最终输入。中断后续训所需的 optimizer 状态不会保存。

## 验证与限制

```bash
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -v
python tests/smoke_pipeline.py
```

单元测试和合成集成检查使用人工生成的数据和随机初始化的小型编码器，用于检查软件流程，不代表临床预测表现。`tests/audit_local_data.py` 可选读取本机 MIMIC 表格，抽样检查事件和笔记，并只输出聚合审计数量；它不是完整队列重建或模型评估。当前版本实际执行的检查见[验证记录](docs/VALIDATION.md)。

已知限制包括上述队列数量冲突；历史数据划分、相似度阈值和梯度累积步数的溯源缺失；ITEMID 覆盖有限（尚未确认全面覆盖 CareVue）；本次审计所用数据目录缺少 `D_ITEMS.csv` 和 `D_LABITEMS.csv` 字典，因此无法独立复核 ITEMID 的临床语义；代码保留的无转科和 APR 标签处理策略；代码定义的时间序列结束规则；未定义类别的 AUC 处理方式；以及缺少表 4 的历史峰值时间映射。当前没有在真实数据上完成完整五折训练，也未复现论文报告的性能表。现有历史产物不能证明修正后的流程生成了论文结果。仓库不提交患者数据、checkpoint、预训练权重、私有运行产物或论文源文件。项目许可和第三方资源使用权尚未确定，公开前需由作者确认。

## 论文引用

Niu B, Zheng T, Zang S, Zhang R, Li S, Cai G. *A Label-Guided Gated Cross-Modal Network for Early DRG Prediction and CMI Estimation Using Multimodal Electronic Health Records: Development and Validation Study.* 用户提供的论文版本：2026 年 11 月 27 日。未提供期刊、发表状态或 DOI。
