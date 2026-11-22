"""Traceable case attribution using saved fold statistics and true hourly bins."""
import argparse
from bisect import bisect_left, bisect_right
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from main_multimodal import load_fold, load_inputs
from data_utils_multimodal import MultimodalDrgDataset, load_structured_frame
from extract_structured import FEATURE_UNITS
from protocol import write_json, sha256_file, read_json
from train_utils_multimodal import move_batch

def _valid_temporal_attention(temporal_attention, valid_mask):
    attention = np.asarray(temporal_attention, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if attention.ndim != 1 or valid.shape != attention.shape:
        raise ValueError('Temporal attention and mask must be matching one-dimensional arrays')
    indices = np.flatnonzero(valid)
    weights = attention[indices]
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Valid temporal attention must be finite and nonnegative')
    return indices, weights


def estimate_temporal_peak_time(temporal_attention, valid_mask):
    """First maximum valid hourly bin start, in hours since ICU admission.

    P137/Table 4 report historical noninteger peaks without specifying how
    they were derived. Hourly argmax is the peak the saved model supports;
    neither interpolation nor an attention centroid reconstructs those times.
    """
    indices, weights = _valid_temporal_attention(temporal_attention, valid_mask)
    if not len(indices) or not weights.any():
        return None
    return float(indices[int(np.argmax(weights))])


def estimate_attention_centroid_time(temporal_attention, valid_mask):
    """Attention-weighted mean of valid hourly bin starts; this is not a peak."""
    indices, weights = _valid_temporal_attention(temporal_attention, valid_mask)
    if not len(indices) or not weights.any():
        return None
    return float(np.dot(indices, weights / weights.sum()))


def top_word_spans(text, tokenizer, attention, max_length, top_k=5, ngram=5):
    """Rank contiguous complete words that are actually visible to the model."""
    if top_k < 1 or ngram < 1:
        raise ValueError('top_k and ngram must be positive')
    encoding = tokenizer(text, max_length=max_length, padding='max_length', truncation=True,
                         return_offsets_mapping=True, return_attention_mask=True)
    offsets = encoding['offset_mapping']
    mask = np.asarray(encoding['attention_mask'], dtype=bool)
    attention = np.asarray(attention, dtype=float)
    if attention.ndim != 1 or len(attention) != len(offsets) or mask.shape != attention.shape:
        raise ValueError('Text attention must align exactly with tokenizer offsets and mask')
    if not np.isfinite(attention[mask]).all() or (attention[mask] < 0).any():
        raise ValueError('Valid text attention must be finite and nonnegative')
    words = list(re.finditer(r'\S+', text))
    starts, ends = [w.start() for w in words], [w.end() for w in words]
    word_scores = np.zeros(len(words), dtype=float)
    covered = np.zeros(len(text), dtype=bool)
    for (start, stop), score, active in zip(offsets, attention, mask):
        if not active or start == stop:
            continue
        if not 0 <= start < stop <= len(text):
            raise ValueError('Tokenizer offsets lie outside the source text')
        # Bisect keeps long-note attribution from scanning every word per token.
        matches = range(bisect_right(ends, start), bisect_left(starts, stop))
        covered[start:stop] = True
        for i in matches:
            word_scores[i] += float(score) / len(matches)
    visible = np.asarray([covered[w.start():w.end()].all() for w in words], dtype=bool)
    candidates = []
    for i in range(max(0, len(words) - ngram + 1)):
        # A truncated subword must not expose the unseen remainder of a word.
        if visible[i:i+ngram].all():
            candidates.append((float(word_scores[i:i+ngram].sum()), i))
    selected, used = [], set()
    for score, i in sorted(candidates, key=lambda x: (-x[0], x[1])):
        indices = set(range(i, i+ngram))
        if used & indices:
            continue
        selected.append({'text': text[words[i].start():words[i+ngram-1].end()],
                         'start_char': words[i].start(), 'end_char': words[i+ngram-1].end(), 'attention_sum': score})
        used |= indices
        if len(selected) == top_k:
            break
    return selected

@torch.no_grad()
def make_case_report(fold_dir, stay, hour=48, device='auto', overrides=None, top_k=5):
    fold_dir = Path(fold_dir)
    args, model, stats, vocab = load_fold(fold_dir, device, overrides)
    if args.model != 'LGGCMN' or args.fusion_method != 'gcm':
        raise ValueError('Gate/cross-attention attribution requires LGGCMN with GCM')
    args.verify_data_hashes = True
    base, rule, provenance, _, _ = load_inputs(args)
    if provenance != read_json(fold_dir.parent / 'data_provenance.json'):
        raise ValueError('Dataset provenance differs from the training run')
    test = pd.read_csv(fold_dir / 'test.csv', dtype={'stay': str, 'DRG_CODE': str})
    selected = test[test.stay == str(stay)].reset_index(drop=True)
    if len(selected) != 1:
        raise ValueError('Select one admission from the saved hold-out test set')
    dataset = MultimodalDrgDataset(args, selected, rule, stats, vocabulary=vocab)
    dataset.load_data(hour)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model.eval()
    logits, attn = model(move_batch(batch, next(model.parameters()).device), return_attn=True)
    pred = int(logits.argmax(1))
    temporal = attn['temporal'][0, pred].cpu().numpy()
    valid = batch['structured_mask'][0].numpy()
    peak_time = estimate_temporal_peak_time(temporal, valid)
    peak = int(peak_time) if peak_time is not None else None
    centroid_time = estimate_attention_centroid_time(temporal, valid)
    frame = load_structured_frame(base / 'structured' / f'{stay}_struct.pk')
    evidence = []
    if peak is not None:
        for i, feature in enumerate(stats['feature_names']):
            z = float(batch['structured'][0, peak, i])
            observed = bool(batch['structured'][0, peak, len(stats['feature_names'])+i])
            record = {'feature': feature, 'unit': FEATURE_UNITS.get(feature, 'source unit'),
                      'model_z_value': z, 'imputed': not observed,
                      'model_value_original_units': z * stats['std'][i] + stats['mean'][i],
                      'observed_hourly_mean': float(frame.iloc[peak][feature]) if observed else None,
                      'source_observation_count': frame.attrs['observation_counts'][peak][i]}
            evidence.append(record)
        evidence.sort(key=lambda e: -abs(e['model_z_value']))
    text = dataset.read_text(stay, hour, selected.iloc[0].ADMITTIME)
    spans = top_word_spans(text, dataset.tokenizer, attn['text'][0,pred].cpu().numpy(), args.max_seq_length, top_k)
    return {
        'stay': str(stay), 'hour': hour, 'true_DRG': selected.iloc[0].DRG_CODE,
        'predicted_DRG': args.label_codes[pred], 'mean_text_gate': float(attn['gate'][0,pred].mean()),
        'peak_bin_index': peak, 'peak_interval_hours': [peak, min(peak+1, hour)] if peak is not None else None,
        'peak_time_hours': peak_time,
        'peak_time_method': 'first_maximum_valid_hourly_bin_start',
        'time_reference': 'hours_since_ICU_admission',
        'attention_centroid_hours': centroid_time,
        'text_attention_method': 'learned_label_wise' if args.text_pooling == 'label' else 'uniform_global_pooling',
        'temporal_attention': temporal.tolist(), 'valid_time_mask': valid.tolist(),
        'top_5gram_spans': spans, 'physiology_at_peak': evidence,
        'checkpoint_sha256': sha256_file(fold_dir / 'checkpoint.pt'),
        'statistics_sha256': sha256_file(fold_dir / 'struct_stats.json'),
        'text_input_sha256': sha256_file(base / 'text_raw' / f'{stay}.pk'),
        'structured_input_sha256': sha256_file(base / 'structured' / f'{stay}_struct.pk'),
        'attribution_policy': 'predicted-class attention; sum subwords within complete visible whitespace words; top nonoverlapping 5-word spans; peak is the first maximum valid hourly bin start; attention_centroid_hours is a separate weighted mean, not a peak',
        'limitation': 'Hourly means are not single raw measurements. Attention and gates are not causal explanations. The manuscript does not specify the mapping that produced historical noninteger peaks in Table 4; those times cannot be reconstructed from hourly attention alone.',
    }

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True, help='Saved fold directory')
    p.add_argument('--stay', required=True)
    p.add_argument('--hour', type=int, choices=[24,48], default=48)
    p.add_argument('--device', default='auto')
    p.add_argument('--data_dir')
    p.add_argument('--rule_dir')
    p.add_argument('--bert_model_name')
    p.add_argument('--output', help='Defaults to a private report inside the fold directory')
    a = p.parse_args(argv)
    overrides = {k: getattr(a,k) for k in ('data_dir', 'rule_dir', 'bert_model_name') if getattr(a,k) is not None}
    report = make_case_report(a.model_path, a.stay, a.hour, a.device, overrides)
    out = Path(a.output) if a.output else Path(a.model_path) / 'case_reports' / f'{a.stay}_{a.hour}h.json'
    write_json(out, report)
    print(f'Case report saved to {out}')

if __name__ == '__main__':
    main()
