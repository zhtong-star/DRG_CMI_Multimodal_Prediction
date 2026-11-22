# DRG_CMI_Multimodal_Prediction

English | [简体中文](README_CN.md)

This repository contains a PyTorch implementation of **LG-GCMN (Label-Guided Gated Cross-Modal Network)** for early, single-label multiclass DRG prediction and cohort-level case-mix index (CMI) estimation. It combines early-admission clinical notes with hourly physiological measurements and supports separate MS-DRG and APR-DRG cohorts.

The implementation and documentation were checked against the supplied November 27, 2026 manuscript and Figure 1. The alignment record lists corrected code paths, manuscript-internal conflicts, and unresolved experiment provenance. Passing software checks or synthetic training does not reproduce the manuscript's reported results.

## Study and Data

The manuscript describes a development and validation study using MIMIC-III v1.4 from Beth Israel Deaconess Medical Center (BIDMC), 2001–2012. It reports final multimodal cohorts of 19,789 MS-DRG admissions and 27,394 APR-DRG admissions, 611 and 908 DRG classes, and exclusion of 3 and 20 admissions with incomplete structured data. These are manuscript-reported figures; this repository does not hard-code them or claim to have reproduced them.

The manuscript eligibility criteria include age at least 18 years, one ICU stay, ICU length of stay (LOS) at least 24 hours, at least one eligible note from hospital admission through ICU admission plus 48 hours, and a usable label, weight, and multimodal input. The existing code also retains a no-transfer condition because the paper does not discuss transfers. APR-DRG duplicate descriptions follow the source code's first-record policy; admissions with conflicting final codes are excluded. The default does not implement a highest-severity selection rule.

There is a material cohort-count conflict. A read-only screening of the locally available metadata tables counted 21,938 MS and 29,596 APR admissions after age, single-ICU, and no-transfer screening. Enforcing the manuscript's additional ICU LOS ≥24-hour criterion reduced those counts to 18,144 and 24,673, before note and structured-data filtering. Those upper bounds are below the manuscript's reported final cohort sizes. The preprocessing code follows the explicit ≥24-hour method and records actual counts; resolving the discrepancy requires checking the manuscript's cohort table and the historical cohort-generation records. See [the alignment audit](docs/MANUSCRIPT_ALIGNMENT.md).

DRG labels and fiscal-year 2013 relative weights define prediction targets and the CMI calculation; they are not model input features. MIMIC data, DRG rule tables, pretrained model weights, and patient-level outputs are not included. Obtain and use them only under their applicable access terms.

## Task and Architecture

For each hospital admission, the model predicts one MS-DRG or APR-DRG label from information available by a 24- or 48-hour horizon. Clinical classification is admission-level. For CMI, the evaluation code uses only the first eligible hospital admission for each patient in the fixed test set; the manuscript also contains a nearby “all admissions” sentence, described in the alignment audit.

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

The model follows the label-wise text attention, GRU temporal encoder, label-driven cross-attention, element-wise gated fusion, and class-specific scoring shown in Figure 1 and equations 1–10. The manuscript specifies `lambda = 1`; the default LG-GCMN training loss applies this adjustment using class frequencies from that fold's training partition. Inference uses raw logits without another prior correction. The gate is a learned feature-wise text contribution, not a probability of a clinical event.

The manuscript defines 29 structured variables. The implementation uses the 8 vital-sign variables and 21 laboratory variables listed in `extract_structured.py`; it appends one observation indicator for each variable. Original event timestamps are filtered before hourly binning, using `[0, T)` hours after ICU admission. Finite measurements for the same variable and hour are averaged. The implementation excludes chart records marked `ERROR=1` when that column exists and converts Fahrenheit temperature to Celsius.

For each fold, medians, means, and population standard deviations are fitted only on observed training-partition values. Leading and entirely missing values use the training median; later gaps use forward filling. The observation indicators distinguish recorded from imputed values. The valid temporal mask covers hour 0 through the last observed hourly bin inside the requested horizon; later bins are zero padding and are masked in the GRU and cross-attention. The manuscript does not state the exact sequence-end convention. If an admission has no structured observations before hour 24 but has later observations, it remains in the common 48-hour cohort and its 24-hour structured sequence is fully masked.

Clinical notes retain paragraph boundaries until adjacent-paragraph cleaning, remove MIMIC deidentification placeholders, exclude discharge summaries and administrative-diagnosis pseudo-notes, and are ordered by ICU-relative chart time. Notes may precede ICU admission but must be at or after hospital admission; the prediction horizon is measured from ICU admission. The implementation removes an adjacent paragraph only when both cosine and normalized token-level Levenshtein similarities are strictly greater than the configured threshold. The manuscript names a threshold `τ` but gives no numeric value; `0.95` is an implementation default recorded in cohort and run provenance, not a recovered historical setting. Text is truncated or padded to the configured maximum, with padding masked. The Longformer path uses first-token global attention as an implementation detail.

Comparison models are GRU, CAML, Clinical-Longformer, and Concat. The current experiment runner applies the three manuscript ablations (ClinicalBERT, global text pooling instead of label-wise attention, and concatenation instead of GCM) to the MS cohort by default, matching Table 3. It can be explicitly configured to run extra APR ablations. The baselines and ablations use the finalized cohort and saved patient-level split for their cohort.

## Repository Layout

```text
.
├── README.md / README_CN.md
├── requirements_multimodal.txt / check_setup.py
├── clinical_text.py             # note filtering and paragraph cleaning
├── create_cohorts.py            # label, eligibility, and note cohort
├── extract_structured.py        # 29 hourly physiological variables
├── finalize_cohorts.py          # completeness checks and patient folds
├── protocol.py                  # schema, split, hash, and CMI helpers
├── data_utils_multimodal.py     # text/structured data loading
├── models_multimodal.py         # LG-GCMN and comparison models
├── train_utils_multimodal.py    # training, early stopping, inference
├── options_multimodal.py        # model and training options
├── main_multimodal.py           # training and saved-model evaluation
├── run_experiments.py           # multi-model/ablation command runner
├── eval_utils.py / eval_cc_mcc.py
├── generate_report.py / visualize_attention.py
├── prepare_early_notes.py / preprocessing.py
├── docs/MANUSCRIPT_ALIGNMENT.md
├── docs/VALIDATION.md
└── tests/                       # unit, synthetic integration, data audit
```

Private inputs and run outputs are excluded by `.gitignore`. The `Clinical-Longformer/`, `ClinicalBERT/`, `rules/`, `mimic/`, `data_n/`, `embeddings/`, and `results/` paths in examples refer to local resources, not files distributed by this repository.

## Environment

Use Python with the versions in `requirements_multimodal.txt`; the current CPU validation used Python 3.13, PyTorch 2.9.0, Transformers 4.57.6, tokenizers 0.22.2, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, and SciPy 1.16.3. Install the PyTorch build appropriate for your CPU/CUDA environment before installing the remaining requirements. Training the 4096-token model is resource-intensive; the manuscript's reported environment was one NVIDIA RTX 4090 with 24 GB memory, which is not a guarantee that another setup will fit.

```bash
python -m pip install -r requirements_multimodal.txt
python check_setup.py
python check_setup.py --model_dir Clinical-Longformer
```

The model directory must contain compatible configuration, tokenizer, and pretrained weight files. Optional CAML word2vec loading requires `gensim`; attention plotting requires `matplotlib`. Install these separately when using those features. No model weights or download links are supplied here.

## Data Preparation

Place authorized MIMIC-III v1.4 CSV files under `mimic/`: `PATIENTS.csv`, `ADMISSIONS.csv`, `ICUSTAYS.csv`, `DRGCODES.csv`, `NOTEEVENTS.csv`, `CHARTEVENTS.csv`, and `LABEVENTS.csv`. Place the locally prepared 2013 rule tables at `rules/MSDRG_RULE13.csv` and `rules/APRDRG_RULE13.csv`. Each rule table requires `DRG_CODE`, `DESCRIPTION`, and positive `WEIGHT`; `MDC` is used when available. These data and rule files are not included.

Run from the repository root. Use a fresh `data_n` directory; the scripts deliberately refuse to mix or overwrite previous preprocessing versions.

```bash
python create_cohorts.py --mimic_dir mimic --rule_dir rules --data_dir data_n --all_cohorts --note_similarity_threshold 0.95
python extract_structured.py --mimic_dir mimic --data_dir data_n --cohort ms
python extract_structured.py --mimic_dir mimic --data_dir data_n --cohort apr
python finalize_cohorts.py --data_dir data_n --rule_dir rules --cohort ms
python finalize_cohorts.py --data_dir data_n --rule_dir rules --cohort apr
```

Each cohort directory contains `cohort_candidates.csv`, `text_raw/`, `structured/`, then finalization outputs including `cohort.csv`, `splits.csv`, `test_stays.csv`, `exclusions.csv`, `input_manifest.csv`, `labels.json`, `cohort_audit.json`, `extraction_audit.json`, and `finalization_audit.json`. The audit files record actual local counts and exclusion reasons; they do not assert the manuscript's cohort totals. Schema version 4 requires regenerated data after the text paragraph, timestamp, or padding policy changes.

## Configuration and Training

`options_multimodal.py` contains CLI defaults. The key paper-specified settings are 4096 Longformer tokens, batch size 8, AdamW learning rates `1e-5` for the text encoder and `1e-3` for other components, five patient-level development folds, a fixed 10% patient-level test set, 24- and 48-hour evaluation, and `lambda=1` for LG-GCMN. The implementation uses a ceiling when selecting 10% of patients and defaults to split seed 1443. The paper does not specify the gradient accumulation count; `main_multimodal.py` defaults it to 1, while the experiment runner requires the value explicitly. Record the value used. Other architecture and optimizer settings not specified by the paper remain code defaults.

Train the proposed model on one cohort:

```bash
python main_multimodal.py --data_dir data_n --rule_dir rules --cohort ms --model LGGCMN --bert_model_name Clinical-Longformer --logit_adjustment 1 --gradient_accumulation_steps 1 --output_dir results --run_name ms_lggcmn --verify_data_hashes
```

The command above uses the current code default of one accumulation step; it is not presented as the historical manuscript setting. Omit `--cohort ms` or select `apr` for the APR cohort. Without `--fold`, the entry point trains five separate folds; each fold fits its own training statistics and class prior, selects the best epoch by validation macro-F1, and evaluates that checkpoint on the same fixed test patients at both horizons. Use a new `--run_name` for each configuration.

To print the comparison and manuscript ablation commands without starting training, provide local CAML and ClinicalBERT resources:

```bash
python run_experiments.py --cohorts ms apr --ablations --ablation_cohorts ms --clinicalbert_dir ClinicalBERT --pretrained_word_vectors embeddings/clinical.bin --gradient_accumulation_steps 1 --run_prefix study_v1 --dry_run
```

Remove `--dry_run` to run the listed experiments. The default runner trains five model configurations in each cohort and three additional ablations in MS only. Change the accumulation value to the one used in your experiment and preserve it with the outputs. `run_experiments.sh` invokes the same runner from Bash.

## Evaluation and Outputs

Re-evaluate saved five-fold checkpoints on the saved hold-out patients:

```bash
python main_multimodal.py --eval_model results/ms_lggcmn --data_dir data_n --rule_dir rules --hours 24 48 --eval_cc_mcc --verify_data_hashes
```

`--eval_model` also accepts an individual `fold_N` directory. Evaluation checks the saved cohort, rules, split, and model artifacts. Classification metrics use all test admissions: macro/micro F1, macro/micro AUC, top-1, and top-5 accuracy. Macro-F1 uses the full saved class set with zero for an absent class. If any class lacks positive or negative examples, the all-class macro-AUC is undefined and saved as `null`; the mean over defined classes and the number of undefined classes are also recorded. This edge-case convention is explicit in code because the manuscript does not state one.

CMI maps predicted and true DRG labels to fiscal-year 2013 relative weights and compares the mean weights for each patient's first eligible test admission. `APE_CMI_percent` is `100 × |predicted CMI − true CMI| / true CMI`. Additional relative-weight MAE, RMSE, and Spearman correlation use all test admissions. Five-fold summaries are arithmetic means of the five separately evaluated metrics; they are not ensemble predictions.

For a saved LG-GCMN case, use the `stay` value in that fold's `test.csv`:

```bash
python generate_report.py --model_path results/ms_lggcmn/fold_0 --stay TEST_STAY --hour 48
python visualize_attention.py --report results/ms_lggcmn/fold_0/case_reports/TEST_STAY_48h.json --output results/ms_lggcmn/fold_0/attention.png
python prepare_early_notes.py --data_dir data_n --cohort ms --hour 24 --note_similarity_threshold 0.95 --output results/early_notes_24h.csv
```

Reports include predicted-class text and temporal attention, the mean text gate, nonoverlapping five-word spans, original-unit hourly means and observation counts at the first maximum-attention hourly bin. `peak_time_hours` is the start of that bin; a separate attention-weighted centroid is also recorded. The manuscript's Table 4 lists subhour peak times but does not specify how they were derived from hourly inputs; this implementation does not claim to reconstruct them. Attention and gate values are descriptive model quantities, not causal explanations or clinical event probabilities. Patient-level reports and exported notes must remain private.

Typical run outputs are:

```text
results/<run_name>/
  run_config.json  data_provenance.json  environment.json  initialization.json
  text_encoder_resources/  encoder_resources_sha256.json
  summary.json
  fold_0/ ... fold_4/
    config.json  train.csv  validation.csv  test.csv
    struct_stats.json       # structured models
    vocabulary.json         # CAML
    checkpoint.pt  history.csv  training_summary.json  completed.json
    eval_24h/  eval_48h/
      predictions.csv  logits.npz  metrics.json
      cc_mcc_metrics.json  cc_mcc_cases.csv  # optional MS analysis
    case_reports/            # optional private reports
```

For reproducibility, retain the MIMIC version, 2013 rule tables, preprocessing audits, fixed test-patient list, fold manifest, note threshold, initialization hashes, software environment, exact CLI settings (including accumulation), checkpoints, fitted statistics, and per-fold predictions. `--verify_data_hashes` checks finalized inputs. Optimizer state is not saved for interrupted-run continuation.

## Verification and Limitations

```bash
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -v
python tests/smoke_pipeline.py
```

The unit and synthetic integration checks use artificial data and randomly initialized tiny encoders; they test software behavior, not clinical performance. `tests/audit_local_data.py` optionally reads locally available MIMIC tables, samples event/note rows, and writes aggregate audit counts only. It is not a full cohort rebuild or model evaluation. See [the validation record](docs/VALIDATION.md) for checks actually run in this release.

Known limitations include the cohort-count conflict above; unknown historical split, threshold, and gradient-accumulation provenance; limited ITEMID coverage (comprehensive CareVue coverage is not established); unavailable `D_ITEMS.csv` and `D_LABITEMS.csv` dictionaries, so ITEMID meanings were not independently verified in this audit; the retained no-transfer and APR-label policies; a code-defined temporal sequence-end rule; undefined-class AUC handling; and the missing historical Table 4 peak-time mapping. Full five-fold training on the real data and the reported performance tables were not reproduced here. Existing historical artifacts are not evidence that the corrected pipeline generated the manuscript results. No patient data, checkpoints, pretrained weights, private run artifacts, or source manuscript files are committed. No license or third-party resource rights are asserted; the authors must confirm them before public release.

## Citation

Niu B, Zheng T, Zang S, Zhang R, Li S, Cai G. *A Label-Guided Gated Cross-Modal Network for Early DRG Prediction and CMI Estimation Using Multimodal Electronic Health Records: Development and Validation Study.* Supplied manuscript version: November 27, 2026. Journal, publication status, and DOI were not provided.
