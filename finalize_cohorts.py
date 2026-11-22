"""Exclude incomplete multimodal admissions before assigning patient folds."""
import argparse
from pathlib import Path
import pandas as pd
from clinical_text import read_note_text, NOTE_CLEANING_POLICY, validate_note_policy
from data_utils_multimodal import load_rule, load_structured_frame, canonical_code
from protocol import build_splits, write_json, sha256_file, first_admission_mask, SCHEMA_VERSION, read_json

def finalize(base, rule_path, fixed_test=None, seed=1443, note_similarity_threshold=None):
    base = Path(base)
    if (base / 'splits.csv').exists():
        raise FileExistsError('A finalized cohort already exists; prepare a new data_dir')
    candidate_audit = read_json(base / 'cohort_audit.json')
    candidate_threshold = validate_note_policy(candidate_audit)
    threshold_source = 'cohort_audit' if note_similarity_threshold is None else 'explicit_finalization_override'
    note_similarity_threshold = float(candidate_threshold if note_similarity_threshold is None else note_similarity_threshold)
    if not 0 < note_similarity_threshold <= 1:
        raise ValueError('note_similarity_threshold must be in (0, 1]')
    if sha256_file(rule_path) != candidate_audit.get('rule_sha256'):
        raise ValueError('Rule weights changed after candidate cohort construction')
    df = pd.read_csv(base / 'cohort_candidates.csv', dtype={'stay': str, 'DRG_CODE': str})
    df['DRG_CODE'] = df.DRG_CODE.map(canonical_code)
    _, d2i, _, _, d2w = load_rule(rule_path)
    keep, exclusions, inventory = [], [], []
    feature_names = None
    for row in df.itertuples():
        reason = None
        text_path = base / 'text_raw' / f'{row.stay}.pk'
        struct_path = base / 'structured' / f'{row.stay}_struct.pk'
        if row.DRG_CODE not in d2i:
            reason = 'missing_label_or_weight'
        elif not text_path.exists():
            reason = 'missing_text_file'
        elif not read_note_text(text_path, 48, row.ADMITTIME, note_similarity_threshold, require_schema=True):
            reason = 'no_eligible_text'
        elif not struct_path.exists():
            reason = 'missing_structured_file'
        else:
            frame = load_structured_frame(struct_path)
            names = frame.attrs['feature_names']
            if feature_names is None:
                feature_names = names
            if names != feature_names:
                raise ValueError('Structured feature order differs across admissions')
            if not frame[names].notna().any().any():
                reason = 'no_structured_measurement'
        if reason:
            exclusions.append({'stay': row.stay, 'reason': reason})
        else:
            keep.append(row.stay)
            for path in [text_path, struct_path]:
                inventory.append({'file': path.relative_to(base).as_posix(), 'sha256': sha256_file(path)})
    final = df[df.stay.isin(keep)].copy()
    final['WEIGHT'] = final.DRG_CODE.map(d2w)
    if not len(final):
        raise ValueError('No complete multimodal admissions')
    fixed = pd.read_csv(fixed_test).stay.astype(str) if fixed_test else None
    splits = build_splits(final, fixed, seed)
    final.to_csv(base / 'cohort.csv', index=False)
    splits.to_csv(base / 'splits.csv', index=False)
    splits[splits.partition == 'test'][['stay']].to_csv(base / 'test_stays.csv', index=False)
    pd.DataFrame(exclusions, columns=['stay', 'reason']).to_csv(base / 'exclusions.csv', index=False)
    pd.DataFrame(inventory).to_csv(base / 'input_manifest.csv', index=False)
    labels = sorted(final.DRG_CODE.unique(), key=lambda x: (len(x), x))
    write_json(base / 'labels.json', labels)
    test = splits[splits.partition == 'test']
    audit = {
        'schema_version': SCHEMA_VERSION, 'candidate_admissions': len(df),
        'note_similarity_threshold': float(note_similarity_threshold),
        'note_cleaning_policy': NOTE_CLEANING_POLICY,
        'note_similarity_threshold_source': threshold_source,
        'candidate_note_similarity_threshold': candidate_threshold,
        'cohort_audit_sha256': sha256_file(base / 'cohort_audit.json'),
        'count_provenance': 'Actual local finalization counts; manuscript cohort sizes are not asserted reproduced',
        'no_transfer_policy': candidate_audit.get('no_transfer_policy'),
        'complete_admissions': len(final), 'complete_patients': int(final.SUBJECT_ID.nunique()),
        'excluded_admissions': len(exclusions), 'exclusions_by_reason': pd.Series([e['reason'] for e in exclusions]).value_counts().to_dict(),
        'classes': len(labels), 'feature_names': feature_names, 'split_seed': seed,
        'fixed_test_list_supplied': fixed_test is not None, 'test_admissions': len(test),
        'cmi_test_patients': int(first_admission_mask(test).sum()),
        'rule_sha256': sha256_file(rule_path), 'splits_sha256': sha256_file(base / 'splits.csv'),
        'input_manifest_sha256': sha256_file(base / 'input_manifest.csv'),
        'labels_sha256': sha256_file(base / 'labels.json'),
        'folds': {i: {'train_admissions': int(((splits.fold >= 0) & (splits.fold != i)).sum()),
                      'validation_admissions': int((splits.fold == i).sum())} for i in range(5)},
    }
    write_json(base / 'finalization_audit.json', audit)
    return audit

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', default='data_n')
    p.add_argument('--rule_dir', default='rules')
    p.add_argument('--cohort', choices=['ms', 'apr'], default='ms')
    p.add_argument('--test_stays', help='Optional fixed list; every admission of a test patient must be included')
    p.add_argument('--split_seed', type=int, default=1443)
    p.add_argument('--note_similarity_threshold', type=float, default=None,
                   help='Override the threshold recorded by create_cohorts.py')
    a = p.parse_args(argv)
    print(finalize(Path(a.data_dir) / a.cohort, Path(a.rule_dir) / f'{a.cohort.upper()}DRG_RULE13.csv',
                   a.test_stays, a.split_seed, a.note_similarity_threshold))

if __name__ == '__main__':
    main()
