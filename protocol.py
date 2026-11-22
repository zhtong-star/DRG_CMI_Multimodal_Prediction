"""Persisted patient splits, source fingerprints, and JSON audit records."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SCHEMA_VERSION = 4

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [convert(v) for v in x]
        if isinstance(x, np.generic):
            return convert(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def artifact_hashes(directory, names):
    root = Path(directory)
    return {name: sha256_file(root / name) for name in names if (root / name).is_file()}


def verify_artifacts(directory, hashes):
    root = Path(directory)
    for name, expected in hashes.items():
        if not (root / name).is_file() or sha256_file(root / name) != expected:
            raise ValueError(f'Saved artifact changed or is missing: {name}')

def first_admission_mask(df):
    """Earliest eligible hospital admission, ties resolved by HADM_ID."""
    required = {'SUBJECT_ID', 'HADM_ID', 'ADMITTIME'}
    if not required.issubset(df.columns):
        raise ValueError(f'CMI requires {sorted(required)}')
    frame = df.reset_index(drop=True).copy()
    frame['ADMITTIME'] = pd.to_datetime(frame.ADMITTIME, errors='raise')
    if frame[list(required)].isna().any().any():
        raise ValueError('CMI selection metadata contains missing values')
    keep = frame.sort_values(['SUBJECT_ID', 'ADMITTIME', 'HADM_ID'], kind='stable').drop_duplicates('SUBJECT_ID').index
    mask = np.zeros(len(frame), dtype=bool)
    mask[keep] = True
    return mask

def build_splits(cohort, fixed_test_stays=None, seed=1443):
    """Fixed 10% patients; five patient folds on the remaining 90%."""
    df = cohort.copy().reset_index(drop=True)
    if df.stay.duplicated().any():
        raise ValueError('Cohort has duplicate admissions')
    patients = np.array(sorted(df.SUBJECT_ID.unique()))
    if len(patients) < 7:
        raise ValueError('At least seven patients are required for hold-out and five development folds')
    if fixed_test_stays is None:
        shuffled = np.random.default_rng(seed).permutation(patients)
        n_test = max(1, int(np.ceil(len(patients) * 0.1)))
        test_mask = df.SUBJECT_ID.isin(set(shuffled[:n_test]))
    else:
        fixed = set(map(str, fixed_test_stays))
        if fixed - set(df.stay.astype(str)):
            raise ValueError('Fixed test list contains admissions outside the final eligible cohort')
        test_mask = df.stay.astype(str).isin(fixed)
        if set(df.loc[test_mask, 'SUBJECT_ID']) & set(df.loc[~test_mask, 'SUBJECT_ID']):
            raise ValueError('Fixed test list splits admissions of the same patient')
    if not test_mask.any() or test_mask.all():
        raise ValueError('Test and development partitions must both be nonempty')
    dev_patients = np.array(sorted(set(df.loc[~test_mask, 'SUBJECT_ID'])))
    if len(dev_patients) < 5:
        raise ValueError('Development set needs at least five patients')
    df['partition'] = np.where(test_mask, 'test', 'development')
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(KFold(5, shuffle=True, random_state=seed).split(dev_patients)):
        df.loc[df.SUBJECT_ID.isin(dev_patients[val_idx]), 'fold'] = fold
    validate_splits(df)
    return df

def validate_splits(df):
    required = {'stay', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'partition', 'fold'}
    if not required.issubset(df.columns) or df.stay.duplicated().any():
        raise ValueError('Invalid split manifest schema or duplicate stays')
    if df[list(required)].isna().any().any():
        raise ValueError('Split manifest contains missing fields')
    if not set(df.partition).issubset({'development', 'test'}):
        raise ValueError('Unknown partition')
    if (df.groupby('SUBJECT_ID').partition.nunique() > 1).any():
        raise ValueError('Patient leakage between development and test')
    dev = df[df.partition == 'development']
    if set(dev.fold.unique()) != set(range(5)) or (dev.groupby('SUBJECT_ID').fold.nunique() > 1).any():
        raise ValueError('Invalid patient-level development folds')
    if not (df.loc[df.partition == 'test', 'fold'] == -1).all():
        raise ValueError('Test patients cannot belong to development folds')
    if not (df.partition == 'test').any():
        raise ValueError('Empty test set')

def fold_frames(manifest, fold):
    validate_splits(manifest)
    dev = manifest.partition == 'development'
    return tuple(x.reset_index(drop=True) for x in (
        manifest[dev & (manifest.fold != fold)],
        manifest[dev & (manifest.fold == fold)],
        manifest[manifest.partition == 'test'],
    ))
