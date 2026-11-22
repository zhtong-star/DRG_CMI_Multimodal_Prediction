"""Construct candidate text cohorts. Final multimodal filtering precedes splitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from clinical_text import clean_note_paragraphs, DEFAULT_NOTE_SIMILARITY_THRESHOLD, NOTE_CLEANING_POLICY
from data_utils_multimodal import load_rule, canonical_code
from protocol import write_json, sha256_file, SCHEMA_VERSION

def count_stage(df):
    return {'patients': int(df.SUBJECT_ID.nunique()), 'admissions': int(df.HADM_ID.nunique()), 'rows': len(df)}

def select_drg(drg, cohort, apr_label_policy='source_first_per_description'):
    if cohort not in ('ms', 'apr'):
        raise ValueError('Unknown DRG cohort')
    kind = 'MS' if cohort == 'ms' else 'APR'
    df = drg[drg.DRG_TYPE.astype(str).str.strip().eq(kind)].copy()
    df = df.dropna(subset=['DRG_CODE', 'DESCRIPTION'])
    df['DRG_CODE'] = df.DRG_CODE.map(canonical_code)
    df = df.drop_duplicates(['HADM_ID', 'DRG_CODE', 'DESCRIPTION'])
    if cohort == 'apr' and apr_label_policy == 'source_first_per_description':
        # Preserve the source implementation's actual same-description policy.
        # Sorting descriptions did NOT select the highest severity in that code.
        df = df.drop_duplicates(['HADM_ID', 'DESCRIPTION'], keep='first')
    elif cohort == 'apr' and apr_label_policy != 'exclude_ambiguous':
        raise ValueError('Unknown APR label policy')
    df = df.drop_duplicates(['HADM_ID', 'DRG_CODE'])
    ambiguous = df.groupby('HADM_ID').DRG_CODE.nunique()
    return df[~df.HADM_ID.isin(ambiguous[ambiguous > 1].index)][['SUBJECT_ID', 'HADM_ID', 'DRG_CODE']]

def filter_cohort(labels, admissions, patients, icu):
    audit = {'labels_unambiguous': count_stage(labels)}
    df = labels.merge(admissions, on=['SUBJECT_ID', 'HADM_ID'], validate='one_to_one')
    df = df.merge(patients[['SUBJECT_ID', 'DOB']], on='SUBJECT_ID', validate='many_to_one')
    # Calendar birthdays enforce age >=18 exactly, including shifted ages >300.
    admitted = pd.to_datetime(df.ADMITTIME, errors='coerce', format='mixed')
    born = pd.to_datetime(df.DOB, errors='coerce', format='mixed')
    before_birthday = ((admitted.dt.month < born.dt.month) |
                       ((admitted.dt.month == born.dt.month) & (admitted.dt.day < born.dt.day)))
    df['AGE'] = admitted.dt.year - born.dt.year - before_birthday.astype(int)
    df = df[df.AGE >= 18].copy()
    audit['age_ge_18'] = count_stage(df)
    counts = icu.groupby('HADM_ID').size()
    single = icu[icu.HADM_ID.isin(counts[counts == 1].index)].copy()
    single_ids = set(single.HADM_ID)
    audit['single_icu'] = count_stage(df[df.HADM_ID.isin(single_ids)])
    single = single[(single.FIRST_WARDID == single.LAST_WARDID) & (single.FIRST_CAREUNIT == single.LAST_CAREUNIT)]
    cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS']
    df = df.merge(single[cols], on=['SUBJECT_ID', 'HADM_ID'], validate='one_to_one')
    audit['single_icu_no_transfer'] = count_stage(df)
    audit['no_transfer_policy'] = 'Retained from original code; P032 specifies single ICU stay but is silent about transfers'
    duration = (pd.to_datetime(df.OUTTIME, errors='coerce', format='mixed') - pd.to_datetime(df.INTIME, errors='coerce', format='mixed')).dt.total_seconds() / 3600
    df = df[duration >= 24].copy()
    audit['icu_ge_24h'] = count_stage(df)
    df['stay'] = df.SUBJECT_ID.astype(str) + '_' + df.HADM_ID.astype(str)
    return df, audit

def collect_notes(mimic_dir, cohort_df, max_hour=48, chunksize=100000, max_rows=None):
    if max_hour != 48:
        raise ValueError('Candidate cohort requires the 48-hour note horizon')
    parts = []
    raw_count = timestamp_count = eligible_count = 0
    cols = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'CATEGORY', 'TEXT']
    timing = cohort_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'INTIME']]
    for chunk in pd.read_csv(Path(mimic_dir) / 'NOTEEVENTS.csv', usecols=cols, chunksize=chunksize, nrows=max_rows):
        d = chunk.merge(timing, on=['SUBJECT_ID', 'HADM_ID'], validate='many_to_one')
        raw_count += len(d)
        d['CHARTTIME'] = pd.to_datetime(d.CHARTTIME, errors='coerce', format='mixed')
        timestamp_count += int(d.CHARTTIME.notna().sum())
        diff = (d.CHARTTIME - pd.to_datetime(d.INTIME)).dt.total_seconds() / 3600
        valid = d.CHARTTIME.ge(pd.to_datetime(d.ADMITTIME)) & diff.le(max_hour)
        valid &= ~d.CATEGORY.fillna('').str.strip().str.lower().isin(['discharge summary', 'admission_diag'])
        d = d.loc[valid, cols].copy()
        d['DIFFTIME'] = diff[valid]
        d['TEXT'] = d.TEXT.fillna('').map(clean_note_paragraphs)
        d = d[d.TEXT.ne('')]
        eligible_count += len(d)
        if len(d):
            parts.append(d)
    notes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols + ['DIFFTIME'])
    notes.attrs.update({'schema_version': SCHEMA_VERSION, 'paragraph_boundaries_preserved': True})
    return notes, {'matched_notes': raw_count, 'notes_with_charttime': timestamp_count, 'eligible_notes': eligible_count}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mimic_dir', default='mimic')
    p.add_argument('--data_dir', default='data_n')
    p.add_argument('--rule_dir', default='rules')
    p.add_argument('--cohort', choices=['ms', 'apr'], default='ms')
    p.add_argument('--all_cohorts', action='store_true')
    p.add_argument('--apr_label_policy', choices=['source_first_per_description', 'exclude_ambiguous'],
                   default='source_first_per_description', help='Explicit source-code policy; historical highest-severity claim was not implemented')
    p.add_argument('--threshold', type=int, default=48, choices=[48])
    p.add_argument('--note_similarity_threshold', type=float, default=DEFAULT_NOTE_SIMILARITY_THRESHOLD,
                   help='Recorded for downstream note cleaning; manuscript does not report numeric τ')
    a = p.parse_args(argv)
    if not 0 < a.note_similarity_threshold <= 1:
        raise ValueError('note_similarity_threshold must be in (0, 1]')
    root = Path(a.mimic_dir)
    drg = pd.read_csv(root / 'DRGCODES.csv')
    adm = pd.read_csv(root / 'ADMISSIONS.csv', usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME'])
    patients = pd.read_csv(root / 'PATIENTS.csv', usecols=['SUBJECT_ID', 'DOB'])
    icu = pd.read_csv(root / 'ICUSTAYS.csv')
    for cohort in (['ms', 'apr'] if a.all_cohorts else [a.cohort]):
        out = Path(a.data_dir) / cohort
        if (out / 'cohort_candidates.csv').exists():
            raise FileExistsError('Use a fresh data_dir to preserve previous cohort provenance')
        out.mkdir(parents=True, exist_ok=True)
        labels = select_drg(drg, cohort, a.apr_label_policy)
        df, audit = filter_cohort(labels, adm, patients, icu)
        rule_path = Path(a.rule_dir) / f'{cohort.upper()}DRG_RULE13.csv'
        rules, d2i, _, _, d2w = load_rule(rule_path)
        df = df[df.DRG_CODE.isin(d2i)].copy()
        df['WEIGHT'] = df.DRG_CODE.map(d2w)
        audit['complete_label_and_weight'] = count_stage(df)
        notes, note_audit = collect_notes(root, df)
        audit.update(note_audit)
        groups = notes.groupby('HADM_ID', sort=False)
        df = df[df.HADM_ID.isin(groups.groups)].copy()
        text_dir = out / 'text_raw'
        text_dir.mkdir(exist_ok=True)
        for row in df.itertuples():
            n = groups.get_group(row.HADM_ID).sort_values('CHARTTIME', kind='stable')
            n[['CHARTTIME', 'CATEGORY', 'DIFFTIME', 'TEXT']].to_pickle(text_dir / f'{row.stay}.pk')
        audit['note_by_48h'] = count_stage(df)
        # Persist metadata only; do not propagate administrative DIAGNOSIS as model input.
        df = df.drop(columns=['DOB'], errors='ignore')
        df.to_csv(out / 'cohort_candidates.csv', index=False)
        audit['rule_sha256'] = sha256_file(rule_path)
        audit['schema_version'] = SCHEMA_VERSION
        audit['note_similarity_threshold'] = a.note_similarity_threshold
        audit['note_cleaning_policy'] = NOTE_CLEANING_POLICY
        audit['apr_label_policy'] = a.apr_label_policy if cohort == 'apr' else None
        audit['policy'] = f'schema_v{SCHEMA_VERSION}; adjacent-paragraph cleaning; no ADMISSIONS.DIAGNOSIS; discharge-summary exclusion; ICU LOS >=24h'
        audit['count_provenance'] = 'Actual local filtering counts; manuscript Table 1 counts are not asserted reproduced'
        write_json(out / 'cohort_audit.json', audit)
        print(cohort, audit)

if __name__ == '__main__':
    main()
