"""Lazy datasets with one time origin, explicit missingness, and fold-fitted statistics."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from clinical_text import read_note_text, word_tokens
from protocol import SCHEMA_VERSION

def canonical_code(code):
    text = str(code).strip()
    try:
        value = float(text)
        if np.isfinite(value) and value.is_integer():
            return str(int(value))
    except ValueError:
        pass
    return text

def _lookup_rule_value(mapping, code, field_name='DRG_CODE'):
    key = canonical_code(code)
    if key in mapping:
        return mapping[key]
    raise KeyError(f'{field_name} {key!r} is absent from the rule mapping')

def load_rule(path):
    df = pd.read_csv(path, dtype={'DRG_CODE': str})
    required = {'DRG_CODE', 'DESCRIPTION', 'WEIGHT'}
    if not required.issubset(df.columns):
        raise ValueError(f'Rule table requires {sorted(required)}')
    df = df.dropna(subset=list(required)).copy()
    df['DRG_CODE'] = df.DRG_CODE.map(canonical_code)
    df = df[df.DRG_CODE.ne('') & df.DESCRIPTION.astype(str).str.strip().ne('')].copy()
    df['WEIGHT'] = pd.to_numeric(df.WEIGHT, errors='coerce')
    df = df[np.isfinite(df.WEIGHT) & df.WEIGHT.gt(0)]
    if (df.groupby('DRG_CODE').WEIGHT.nunique() > 1).any():
        raise ValueError('A DRG code has conflicting relative weights')
    df = df.drop_duplicates('DRG_CODE').reset_index(drop=True)
    codes = sorted(df.DRG_CODE, key=lambda x: (len(x), x))
    d2i = {code: i for i, code in enumerate(codes)}
    return df, d2i, {i: d for d, i in d2i.items()}, dict(zip(df.DRG_CODE, df.get('MDC', [''] * len(df)))), dict(zip(df.DRG_CODE, df.WEIGHT))

def load_structured_frame(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing structured input: {path}')
    frame = pd.read_pickle(path)
    if frame.attrs.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Legacy structured data has ambiguous padding/timing; re-run extraction with schema v{SCHEMA_VERSION}')
    names = frame.attrs.get('feature_names', [])
    if (not names or len(set(names)) != len(names) or
            list(frame.columns) != ['TIME', *names]):
        raise ValueError('Invalid structured feature schema')
    if not np.array_equal(frame.TIME.to_numpy(), np.arange(len(frame))) or len(frame) > 48 or not len(frame):
        raise ValueError('Structured frame must have contiguous hourly bin starts 0..L-1, 1<=L<=48')
    if (frame.attrs.get('valid_length') != len(frame) or
            frame.attrs.get('bin_interval') != '[h,h+1)' or
            frame.attrs.get('time_origin') != 'ICUSTAYS.INTIME'):
        raise ValueError('Structured timing metadata does not match hourly data')
    values = frame[names].to_numpy(float)
    counts = np.asarray(frame.attrs.get('observation_counts'), dtype=float)
    if (counts.shape != values.shape or not np.isfinite(counts).all() or
            (counts < 0).any() or (counts != np.floor(counts)).any() or
            not np.array_equal(counts > 0, np.isfinite(values))):
        raise ValueError('Structured observation counts disagree with missing values')
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError('Invalid or wholly absent structured measurements')
    return frame

def structured_window(frame, hour, feature_names=None):
    if hour not in (24, 48):
        raise ValueError('Supported prediction horizons are 24 and 48 hours')
    names = feature_names or frame.attrs['feature_names']
    if list(frame.attrs['feature_names']) != list(names):
        raise ValueError('Feature order differs from the fold statistics')
    window = frame[frame.TIME < hour].copy()
    observed_rows = window[names].notna().any(axis=1)
    length = int(window.loc[observed_rows, 'TIME'].max()) + 1 if observed_rows.any() else 0
    # Only observations preceding the horizon determine available length.
    return window.iloc[:length][names].to_numpy(np.float32), length

def fit_struct_stats(struct_dir, train_df, feature_names):
    if train_df.stay.duplicated().any():
        raise ValueError('Training statistics cannot count duplicate admissions')
    if 'partition' in train_df and train_df.partition.eq('test').any():
        raise ValueError('Test admissions cannot fit training statistics')
    parts = []
    for stay in train_df.stay:
        frame = load_structured_frame(Path(struct_dir) / f'{stay}_struct.pk')
        values, _ = structured_window(frame, 48, feature_names)
        parts.append(values)
    if not parts:
        raise ValueError('Cannot fit statistics on an empty training partition')
    matrix = np.vstack(parts)
    mean, std, median, counts = [], [], [], []
    for column in matrix.T:
        observed = column[np.isfinite(column)]
        counts.append(len(observed))
        mean.append(float(observed.mean()) if len(observed) else 0.0)
        scale = float(observed.std()) if len(observed) else 1.0
        std.append(scale if scale >= 1e-8 else 1.0)
        median.append(float(np.median(observed)) if len(observed) else 0.0)
    return {
        'schema_version': SCHEMA_VERSION, 'feature_names': list(feature_names),
        'feature_dim': len(feature_names), 'mean': mean, 'std': std, 'median': median,
        'observed_counts': counts, 'all_missing_training_features': [n for n, c in zip(feature_names, counts) if c == 0],
        'fit_stays': sorted(train_df.stay.astype(str).tolist()),
        'fit_horizon_hours': 48,
        'all_missing_fallback': 'median=mean=0, SD=1 when no training observation exists; no held-out values used',
        'fit_policy': 'finite observed hourly means in training partition only; population SD',
    }

def normalize_structured(frame, stats, hour, timesteps=48):
    names = stats['feature_names']
    if 'schema_version' in stats and stats['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Fold statistics schema differs from the current data schema')
    if not isinstance(timesteps, (int, np.integer)) or timesteps < hour:
        raise ValueError('timesteps must be an integer covering the prediction horizon')
    for key in ('mean', 'std', 'median'):
        vector = np.asarray(stats[key], dtype=float)
        if vector.shape != (len(names),) or not np.isfinite(vector).all():
            raise ValueError(f'Invalid training {key} statistics')
    if (np.asarray(stats['std']) <= 0).any():
        raise ValueError('Training SD must be positive')
    values, length = structured_window(frame, hour, names)
    out = np.zeros((timesteps, 2 * len(names)), dtype=np.float32)
    mask = np.arange(timesteps) < length
    if length:
        observed = np.isfinite(values)
        filled = pd.DataFrame(values).ffill().fillna(pd.Series(stats['median'])).to_numpy(np.float32)
        standardized = (filled - np.asarray(stats['mean'])) / np.asarray(stats['std'])
        out[:length, :len(names)] = standardized
        out[:length, len(names):] = observed.astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Non-finite model inputs after imputation')
    return out, mask

class MultimodalDrgDataset(Dataset):
    def __init__(self, args, df, rule_path, struct_stats=None, tokenizer=None, vocabulary=None):
        self.args = args
        self.df = df.copy().reset_index(drop=True)
        self.df['DRG_CODE'] = self.df.DRG_CODE.map(canonical_code)
        self.base = Path(args.data_dir) / args.cohort
        self.hour = 48
        self.structured = args.model in {'LGGCMN', 'Concat', 'GRU'}
        self.has_text = args.model != 'GRU'
        self.stats = struct_stats
        self.vocabulary = vocabulary
        _, _, _, _, self.d2w = load_rule(rule_path)
        self.d2i = {str(code): i for i, code in enumerate(args.label_codes)}
        self.Y = len(self.d2i)
        if not set(self.df.DRG_CODE).issubset(self.d2i):
            raise ValueError('Dataset contains labels outside the saved class order')
        if self.structured and struct_stats is None:
            raise ValueError('Structured models require training-fold statistics; fallback is forbidden')
        if struct_stats is not None:
            self.feature_names = list(struct_stats['feature_names'])
            self.struct_feature_dim = len(self.feature_names)
            self.struct_mean = np.asarray(struct_stats['mean'])
            self.struct_std = np.asarray(struct_stats['std'])
        self.tokenizer = tokenizer
        if self.has_text and args.model != 'CAML' and tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, use_fast=True)
        if args.model == 'CAML' and vocabulary is None:
            raise ValueError('CAML requires a vocabulary fitted on this training fold')

    def load_data(self, hour):
        if hour not in (24, 48):
            raise ValueError('hour must be 24 or 48')
        self.hour = hour

    def __len__(self):
        return len(self.df)

    def read_text(self, stay, hour, admission_time=None):
        return read_note_text(self.base / 'text_raw' / f'{stay}.pk', hour, admission_time,
                              self.args.note_similarity_threshold, require_schema=True)

    def read_structured(self, stay, hour):
        frame = load_structured_frame(self.base / 'structured' / f'{stay}_struct.pk')
        return normalize_structured(frame, self.stats, hour, self.args.struct_timesteps)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sample = {'entry': str(row.stay), 'drg': torch.tensor(self.d2i[row.DRG_CODE], dtype=torch.long)}
        if self.has_text:
            text = self.read_text(row.stay, self.hour, row.ADMITTIME)
            if self.args.model == 'CAML':
                ids = [self.vocabulary.get(t, 1) for t in word_tokens(text)][:self.args.max_seq_length] or [1]
                n = len(ids)
                ids += [0] * (self.args.max_seq_length - n)
                mask = [1] * n + [0] * (self.args.max_seq_length - n)
            else:
                encoded = self.tokenizer(text, max_length=self.args.max_seq_length, padding='max_length',
                                         truncation=True, return_attention_mask=True, return_token_type_ids=False)
                ids, mask = encoded['input_ids'], encoded['attention_mask']
            sample['text'] = torch.tensor(ids, dtype=torch.long)
            sample['text_mask'] = torch.tensor(mask, dtype=torch.bool)
        if self.structured:
            values, mask = self.read_structured(row.stay, self.hour)
            sample['structured'] = torch.from_numpy(values)
            sample['structured_mask'] = torch.from_numpy(mask)
        return sample

DrgTextDataset = MultimodalDrgDataset
