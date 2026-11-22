"""Five patient folds, one fixed hold-out, and auditable per-fold artifacts."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from data_utils_multimodal import MultimodalDrgDataset, fit_struct_stats, load_rule, canonical_code
from clinical_text import DEFAULT_NOTE_SIMILARITY_THRESHOLD, validate_note_policy
from eval_utils import evaluate_predictions, aggregate_fold_metrics
from models_multimodal import pick_model
from options_multimodal import parse_args
from preprocessing import build_caml_inputs
from protocol import read_json, write_json, sha256_file, fold_frames, validate_splits, SCHEMA_VERSION, artifact_hashes, verify_artifacts
from train_utils_multimodal import fit_model, predict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def resolve_device(requested):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested.isdigit():
        requested = 'cuda:' + requested
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return device

def build_optimizer(model, args):
    text = list(model.text_encoder.parameters()) if hasattr(model, 'text_encoder') else []
    ids = {id(p) for p in text}
    other = [p for p in model.parameters() if id(p) not in ids]
    groups = [{'params': other, 'lr': args.other_lr}]
    if text:
        groups.insert(0, {'params': text, 'lr': args.text_lr})
    return torch.optim.AdamW(groups, weight_decay=args.wd)

def prepare_class_prior(train, codes):
    counts = train.DRG_CODE.map(canonical_code).value_counts()
    values = np.array([counts.get(str(c), 0) for c in codes], dtype=float)
    # Absent training classes have a documented numerical floor, not test counts.
    values = np.maximum(values / values.sum(), 1e-12)
    return (values / values.sum()).tolist()

def environment_record():
    packages = {}
    for name in ('torch', 'transformers', 'tokenizers', 'numpy', 'pandas', 'scikit-learn', 'scipy'):
        packages[name] = importlib.metadata.version(name)
    return {'python': sys.version, 'packages': packages, 'cuda_available': torch.cuda.is_available(),
            'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'code_sha256': {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}

def load_inputs(args):
    base = Path(args.data_dir) / args.cohort
    rule_path = Path(args.rule_dir) / f'{args.cohort.upper()}DRG_RULE13.csv'
    audit_path = base / 'finalization_audit.json'
    if not audit_path.exists():
        raise FileNotFoundError('Finalized cohort missing: run create_cohorts.py, extract_structured.py, finalize_cohorts.py')
    audit = read_json(audit_path)
    validate_note_policy(
        audit,
        expected_threshold=getattr(args, 'note_similarity_threshold', DEFAULT_NOTE_SIMILARITY_THRESHOLD),
    )
    for file, expected in [('splits.csv', audit['splits_sha256']), ('labels.json', audit['labels_sha256']),
                           ('input_manifest.csv', audit['input_manifest_sha256'])]:
        if sha256_file(base / file) != expected:
            raise ValueError(f'Finalized data changed: {file}')
    if sha256_file(rule_path) != audit['rule_sha256']:
        raise ValueError('Rule weights changed after cohort finalization')
    manifest = pd.read_csv(base / 'splits.csv', dtype={'DRG_CODE': str, 'stay': str})
    validate_splits(manifest)
    codes = read_json(base / 'labels.json')
    if set(manifest.DRG_CODE) != set(codes):
        raise ValueError('Label ontology and cohort disagree')
    if args.verify_data_hashes:
        for row in pd.read_csv(base / 'input_manifest.csv').itertuples():
            if sha256_file(base / row.file) != row.sha256:
                raise ValueError('A prepared input changed after finalization')
    return base, rule_path, audit, manifest, codes

def evaluate_fold(model, args, fold_dir, test_df, rule_path, stats=None, vocabulary=None):
    fold_dir = Path(fold_dir)
    dataset = MultimodalDrgDataset(args, test_df, rule_path, stats, vocabulary=vocabulary)
    rules, _, _, _, weight_by_code = load_rule(rule_path)
    results = {}
    for hour in args.hours:
        dataset.load_data(hour)
        logits, labels, entries, _ = predict(model, dataset, args.batch_size, args.num_workers)
        if entries != test_df.stay.astype(str).tolist():
            raise ValueError('Prediction order differs from saved test metadata')
        metrics, rows = evaluate_predictions(logits, labels, test_df, args.label_codes, weight_by_code)
        out = fold_dir / f'eval_{hour}h'
        out.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out / 'predictions.csv', index=False)
        np.savez_compressed(out / 'logits.npz', logits=logits, labels=labels, stays=np.asarray(entries, dtype=str),
                            label_codes=np.asarray(args.label_codes, dtype=str))
        if args.eval_cc_mcc and args.cohort == 'ms':
            from eval_cc_mcc import conditional_metrics
            report, family_rows = conditional_metrics(logits, labels, entries, rules, args.label_codes)
            write_json(out / 'cc_mcc_metrics.json', report)
            family_rows.to_csv(out / 'cc_mcc_cases.csv', index=False)
        write_json(out / 'metrics.json', metrics)
        results[str(hour)] = metrics
    return results

def aggregate_run(run_dir, hours):
    run_dir = Path(run_dir)
    result = {'aggregation': 'arithmetic mean of five separately evaluated hold-out metric values; not an ensemble',
              'status': 'complete', 'hours': {}}
    for hour in hours:
        paths = [run_dir / f'fold_{i}' / f'eval_{hour}h' / 'metrics.json' for i in range(5)]
        if not all(p.exists() and (p.parent.parent / 'completed.json').exists() for p in paths):
            result['status'] = 'incomplete'
            result['hours'][str(hour)] = {'completed_folds': [i for i,p in enumerate(paths) if p.exists() and (p.parent.parent / 'completed.json').exists()]}
            continue
        result['hours'][str(hour)] = aggregate_fold_metrics([read_json(p) for p in paths])
        cc_paths = [p.parent / 'cc_mcc_metrics.json' for p in paths]
        if all(p.exists() for p in cc_paths):
            result['hours'][str(hour)]['CC_MCC'] = aggregate_fold_metrics([read_json(p) for p in cc_paths])
    write_json(run_dir / 'summary.json', result)
    return result

def load_fold(fold_dir, device='auto', overrides=None):
    fold_dir = Path(fold_dir)
    completed = read_json(fold_dir / 'completed.json')
    verify_artifacts(fold_dir, completed['artifact_sha256'])
    encoder_dir = fold_dir.parent / 'text_encoder_resources'
    if encoder_dir.exists():
        verify_artifacts(encoder_dir, read_json(fold_dir.parent / 'encoder_resources_sha256.json'))
    args = argparse.Namespace(**read_json(fold_dir / 'config.json'))
    if args.model not in {'GRU', 'CAML'}:
        args.bert_model_name = str(encoder_dir)
    for name, value in (overrides or {}).items():
        setattr(args, name, value)
    stats = read_json(fold_dir / 'struct_stats.json') if (fold_dir / 'struct_stats.json').exists() else None
    vocab = read_json(fold_dir / 'vocabulary.json') if args.model == 'CAML' else None
    embedding = np.zeros((len(vocab), args.embedding_dim), dtype=np.float32) if vocab is not None else None
    model = pick_model(args, embedding, len(stats['feature_names']) * 2 if stats else None, load_pretrained=False)
    state = torch.load(fold_dir / 'checkpoint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(resolve_device(device))
    return args, model, stats, vocab


def resource_record(args):
    """Record local initialization files; do not serialize credentials or model data."""
    if args.model in {'GRU', 'CAML'}:
        return {'word_vectors_sha256': sha256_file(args.pretrained_word_vectors) if args.pretrained_word_vectors else None}
    source = Path(args.bert_model_name)
    if not source.is_dir():
        raise ValueError('Use a local pretrained model directory to record immutable initialization hashes')
    suffixes = {'.json', '.txt', '.model', '.bin', '.safetensors'}
    files = [p.name for p in source.iterdir() if p.is_file() and p.suffix in suffixes]
    return {'pretrained_files_sha256': artifact_hashes(source, sorted(files))}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cli = parse_args(argv)
    if cli.eval_model:
        path = Path(cli.eval_model)
        folds = [path] if (path / 'config.json').exists() else [path / f'fold_{i}' for i in range(5)]
        for fold in folds:
            overrides = {}
            for name in ('data_dir', 'rule_dir', 'bert_model_name'):
                if any(token == '--' + name or token.startswith('--' + name + '=') for token in argv):
                    overrides[name] = getattr(cli, name)
            args, model, stats, vocab = load_fold(fold, cli.device, overrides)
            args.hours, args.eval_cc_mcc = cli.hours, cli.eval_cc_mcc
            args.verify_data_hashes = cli.verify_data_hashes
            _, rule, audit, manifest, codes = load_inputs(args)
            saved_provenance = read_json(fold.parent / 'data_provenance.json')
            if audit != saved_provenance or codes != args.label_codes:
                raise ValueError('Evaluation data differs from the training run provenance')
            test = pd.read_csv(fold / 'test.csv', dtype={'DRG_CODE': str, 'stay': str})
            expected = manifest[manifest.partition == 'test'].reset_index(drop=True)
            if not test.equals(expected):
                raise ValueError('Saved hold-out membership/metadata changed')
            evaluate_fold(model, args, fold, test, rule, stats, vocab)
            del model
        aggregate_run(folds[0].parent, cli.hours)
        print('Saved-checkpoint evaluation complete.')
        return

    base, rule, audit, manifest, codes = load_inputs(cli)
    cli.label_codes = codes
    # All routes use the same complete multimodal cohort for fair comparisons.
    run_name = cli.run_name or f'{cli.cohort}_{cli.model}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    if Path(run_name).name != run_name or run_name in {'.', '..'}:
        raise ValueError('run_name must be a directory name, not a path')
    run_dir = Path(cli.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    resources = resource_record(cli)
    environment = environment_record()
    common = vars(cli).copy()
    for k in ('fold', 'eval_model', 'device', 'verify_data_hashes'):
        common.pop(k, None)
    if (run_dir / 'run_config.json').exists():
        if read_json(run_dir / 'run_config.json') != common or read_json(run_dir / 'data_provenance.json') != audit:
            raise ValueError('Existing run has different settings/data; use a new run_name')
        if read_json(run_dir / 'initialization.json') != resources or read_json(run_dir / 'environment.json') != environment:
            raise ValueError('Code, environment, or initialization changed across folds; use a new run_name')
    else:
        write_json(run_dir / 'run_config.json', common)
        write_json(run_dir / 'data_provenance.json', audit)
        write_json(run_dir / 'environment.json', environment)
        write_json(run_dir / 'initialization.json', resources)
        if cli.model not in {'GRU', 'CAML'}:
            from transformers import AutoConfig, AutoTokenizer
            encoder_dir = run_dir / 'text_encoder_resources'
            AutoConfig.from_pretrained(cli.bert_model_name).save_pretrained(encoder_dir)
            AutoTokenizer.from_pretrained(cli.bert_model_name, use_fast=True).save_pretrained(encoder_dir)
            write_json(run_dir / 'encoder_resources_sha256.json', artifact_hashes(encoder_dir, [p.name for p in encoder_dir.iterdir()]))
    for fold in ([cli.fold] if cli.fold is not None else range(5)):
        fold_dir = run_dir / f'fold_{fold}'
        if fold_dir.exists():
            raise FileExistsError(f'{fold_dir} already exists; choose a fresh run_name or a different --fold')
        fold_dir.mkdir()
        a = copy.deepcopy(cli)
        a.fold = fold
        a.random_seed = cli.random_seed + fold
        set_seed(a.random_seed)
        train, val, test = fold_frames(manifest, fold)
        a.class_prior = prepare_class_prior(train, codes)
        stats = None
        if a.model in {'LGGCMN', 'Concat', 'GRU'}:
            stats = fit_struct_stats(base / 'structured', train, audit['feature_names'])
            write_json(fold_dir / 'struct_stats.json', stats)
        vocab = embedding = None
        if a.model == 'CAML':
            vocab, embedding = build_caml_inputs(a, train)
            a.embedding_dim = embedding.shape[1]
            write_json(fold_dir / 'vocabulary.json', vocab)
        if a.logit_adjustment is None:
            a.logit_adjustment = 1.0 if a.model == 'LGGCMN' else 0.0
        write_json(fold_dir / 'config.json', vars(a))
        for name, frame in [('train', train), ('validation', val), ('test', test)]:
            frame.to_csv(fold_dir / f'{name}.csv', index=False)
        datasets = [MultimodalDrgDataset(a, frame, rule, stats, vocabulary=vocab) for frame in (train, val)]
        model = pick_model(a, embedding, len(audit['feature_names']) * 2 if stats else None)
        model.to(resolve_device(a.device))
        print(f'fold={fold} training={len(train)} validation={len(val)} test={len(test)} lambda={a.logit_adjustment}', flush=True)
        fit_model(model, *datasets, build_optimizer(model, a), a, fold_dir)
        evaluate_fold(model, a, fold_dir, test, rule, stats, vocab)
        write_json(fold_dir / 'completed.json', {'fold': fold, 'artifact_sha256': artifact_hashes(fold_dir,
                   ['checkpoint.pt', 'config.json', 'struct_stats.json', 'vocabulary.json', 'train.csv', 'validation.csv', 'test.csv'])})
        del model, datasets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = aggregate_run(run_dir, cli.hours)
    print(f'Run {summary["status"]}: {run_dir}')

if __name__ == '__main__':
    main()
