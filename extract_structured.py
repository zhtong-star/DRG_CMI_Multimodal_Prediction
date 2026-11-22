"""Extract hourly physiological inputs without fitting any normalization statistics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import SCHEMA_VERSION, write_json

VITAL_ITEMS = {
    'heart_rate': [220045], 'sbp': [220050, 220179], 'dbp': [220051, 220180],
    'mbp': [220052, 220181, 225312], 'resp_rate': [220210, 224690],
    'temperature': [223761, 223762], 'spo2': [220277], 'glucose': [220621, 225664],
}
LAB_ITEMS = {
    'wbc': [51300, 51301], 'rbc': [51279], 'hemoglobin': [51222],
    'hematocrit': [51221], 'platelet': [51265], 'sodium': [50983],
    'potassium': [50971], 'chloride': [50902], 'bicarbonate': [50882],
    'bun': [51006], 'creatinine': [50912], 'glucose_lab': [50931],
    'calcium': [50893], 'magnesium': [50960], 'phosphate': [50970],
    'albumin': [50862], 'bilirubin': [50885], 'alt': [50861],
    'ast': [50878], 'alp': [50863], 'lactate': [50813],
}
FEATURE_NAMES = sorted([*VITAL_ITEMS, *LAB_ITEMS])
FEATURE_UNITS = {
    'heart_rate': 'beats/min', 'sbp': 'mmHg', 'dbp': 'mmHg', 'mbp': 'mmHg',
    'resp_rate': 'breaths/min', 'temperature': 'degC', 'spo2': '%',
    'glucose': 'mg/dL', 'glucose_lab': 'mg/dL', 'wbc': 'K/uL', 'rbc': 'M/uL',
    'hemoglobin': 'g/dL', 'hematocrit': '%', 'platelet': 'K/uL',
    'sodium': 'mEq/L', 'potassium': 'mEq/L', 'chloride': 'mEq/L',
    'bicarbonate': 'mEq/L', 'bun': 'mg/dL', 'creatinine': 'mg/dL',
    'calcium': 'mg/dL', 'magnesium': 'mg/dL', 'phosphate': 'mg/dL',
    'albumin': 'g/dL', 'bilirubin': 'mg/dL', 'alt': 'U/L', 'ast': 'U/L',
    'alp': 'U/L', 'lactate': 'mmol/L',
}

def process_events(chunk, cohort, mapping, hour=48):
    if hour not in (24, 48):
        raise ValueError('Supported structured horizons are 24 and 48 hours')
    lookup = {i: f for f, ids in mapping.items() for i in ids}
    if len(lookup) != sum(len(ids) for ids in mapping.values()):
        raise ValueError('A source ITEMID maps to more than one physiological variable')
    d = chunk[chunk.ITEMID.isin(lookup)].copy()
    if 'ERROR' in d:
        d = d[pd.to_numeric(d.ERROR, errors='coerce').fillna(0).ne(1)]
    d = d.merge(cohort[['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME']], on=['SUBJECT_ID', 'HADM_ID'], validate='many_to_one')
    d['CHARTTIME'] = pd.to_datetime(d.CHARTTIME, errors='coerce', format='mixed')
    d['TIME'] = (d.CHARTTIME - pd.to_datetime(d.INTIME)).dt.total_seconds() / 3600
    d['VALUENUM'] = pd.to_numeric(d.VALUENUM, errors='coerce')
    valid = d.TIME.ge(0) & d.TIME.lt(hour) & d.CHARTTIME.lt(pd.to_datetime(d.OUTTIME))
    valid &= np.isfinite(d.VALUENUM)
    d = d[valid].copy()
    d['FEATURE'] = d.ITEMID.map(lookup)
    # Item 223761 is temperature in Fahrenheit; 223762 is Celsius.
    fahrenheit = d.ITEMID.eq(223761)
    d.loc[fahrenheit, 'VALUENUM'] = (d.loc[fahrenheit, 'VALUENUM'] - 32) * 5 / 9
    # No shared clinical cutoff is defensible across 29 variables.
    d['TIME_BIN'] = np.floor(d.TIME).astype(int)
    return d

def aggregate_events(d):
    keys = ['SUBJECT_ID', 'HADM_ID', 'TIME_BIN', 'FEATURE']
    return d.groupby(keys, observed=True).agg(
        total=('VALUENUM', 'sum'), count=('VALUENUM', 'size'),
        first_time=('TIME', 'min'), last_time=('TIME', 'max')).reset_index()

def merge_aggregates(parts):
    if not parts:
        return pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID', 'TIME_BIN', 'FEATURE', 'total', 'count', 'first_time', 'last_time'])
    return pd.concat(parts, ignore_index=True).groupby(
        ['SUBJECT_ID', 'HADM_ID', 'TIME_BIN', 'FEATURE'], observed=True
    ).agg(total=('total', 'sum'), count=('count', 'sum'),
          first_time=('first_time', 'min'), last_time=('last_time', 'max')).reset_index()

def to_hourly_frame(aggregated, feature_names=FEATURE_NAMES):
    if aggregated.empty:
        return None
    d = aggregated.copy()
    if not set(d.FEATURE).issubset(feature_names) or len(set(feature_names)) != len(feature_names):
        raise ValueError('Unknown or duplicate structured feature name')
    if not (d.TIME_BIN.between(0, 47) & d.TIME_BIN.eq(np.floor(d.TIME_BIN))).all():
        raise ValueError('Hourly bins must be integer starts in [0, 48)')
    if not (d['count'] > 0).all():
        raise ValueError('Hourly means require positive observation counts')
    d['value'] = d.total / d['count']
    # Recording span is 0 through the last observed bin. No observations means no sequence.
    length = int(d.TIME_BIN.max()) + 1
    frame = d.pivot(index='TIME_BIN', columns='FEATURE', values='value').reindex(
        index=range(length), columns=feature_names)
    counts = d.pivot(index='TIME_BIN', columns='FEATURE', values='count').reindex(
        index=range(length), columns=feature_names).fillna(0)
    frame.index.name = 'TIME'
    frame = frame.reset_index()
    frame.attrs.update({
        'schema_version': SCHEMA_VERSION, 'feature_names': list(feature_names),
        'bin_interval': '[h,h+1)', 'time_origin': 'ICUSTAYS.INTIME',
        'valid_length': length, 'observation_counts': counts.to_numpy().astype(int).tolist(),
        'source_time_min': float(d.first_time.min()), 'source_time_max': float(d.last_time.max()),
        'temporal_mask_policy': 'valid from hour 0 through last observed bin before horizon; missing interior hours are imputed, tail is padding',
        'source_aggregation': 'arithmetic mean of finite observations per feature and hour',
    })
    return frame

def create_time_series_features(combined_df, stay, threshold_hour, all_features=FEATURE_NAMES):
    """Public helper also exercised by boundary regression tests."""
    if threshold_hour not in (24, 48):
        raise ValueError('Supported structured horizons are 24 and 48 hours')
    d = combined_df[(combined_df.SUBJECT_ID == stay['SUBJECT_ID']) &
                    (combined_df.HADM_ID == stay['HADM_ID']) &
                    (combined_df.TIME >= 0) & (combined_df.TIME < threshold_hour) &
                    np.isfinite(combined_df.VALUENUM)].copy()
    if d.empty:
        return None
    d['TIME_BIN'] = np.floor(d.TIME).astype(int)
    return to_hourly_frame(aggregate_events(d), all_features)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mimic_dir', default='mimic')
    p.add_argument('--data_dir', default='data_n')
    p.add_argument('--cohort', choices=['ms', 'apr'], default='ms')
    p.add_argument('--threshold', type=int, default=48, choices=[48])
    p.add_argument('--chunksize', type=int, default=500000)
    a = p.parse_args(argv)
    base = Path(a.data_dir) / a.cohort
    cohort = pd.read_csv(base / 'cohort_candidates.csv')
    out = base / 'structured'
    if out.exists() and any(out.glob('*_struct.pk')):
        raise FileExistsError('Use a fresh data_dir rather than mixing extraction versions')
    out.mkdir(parents=True, exist_ok=True)
    aggregate_parts = []
    audit = {}
    for filename, mapping in [('CHARTEVENTS.csv', VITAL_ITEMS), ('LABEVENTS.csv', LAB_ITEMS)]:
        count = 0
        path = Path(a.mimic_dir) / filename
        available = pd.read_csv(path, nrows=0).columns
        cols = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']
        if 'ERROR' in available:
            cols += ['ERROR']
        for i, chunk in enumerate(pd.read_csv(path, usecols=cols, chunksize=a.chunksize)):
            events = process_events(chunk, cohort, mapping, a.threshold)
            count += len(events)
            if len(events):
                aggregate_parts.append(aggregate_events(events))
            if len(aggregate_parts) >= 20:
                aggregate_parts = [merge_aggregates(aggregate_parts)]
            if i % 20 == 0:
                print(filename, 'chunks', i + 1, 'eligible observations', count, flush=True)
        audit[filename] = {'eligible_observations': count}
    combined = merge_aggregates(aggregate_parts)
    saved = 0
    for (subject, hadm), group in combined.groupby(['SUBJECT_ID', 'HADM_ID'], sort=False):
        frame = to_hourly_frame(group)
        frame.to_pickle(out / f'{int(subject)}_{int(hadm)}_struct.pk')
        saved += 1
    audit.update({'schema_version': SCHEMA_VERSION, 'feature_names': FEATURE_NAMES,
                  'features': {'vitals': VITAL_ITEMS, 'labs': LAB_ITEMS},
                  'feature_count': len(FEATURE_NAMES), 'feature_units': FEATURE_UNITS,
                  'mapping_provenance': 'Original source ITEMID mappings retained; P038 explicitly specifies 29 variables',
                  'window': '0 <= CHARTTIME-INTIME < 48h and CHARTTIME < OUTTIME',
                  'count_provenance': 'Actual extraction counts; manuscript counts are not asserted reproduced',
                  'extracted_admissions': saved, 'missing_structured_admissions': len(cohort) - saved,
                  'statistics_fitted': False})
    write_json(base / 'extraction_audit.json', audit)
    print('Structured extraction complete. Run finalize_cohorts.py before training.')

if __name__ == '__main__':
    main()
