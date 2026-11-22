"""Export the canonical ICU-relative note input for external baseline audits."""
import argparse
from pathlib import Path
import pandas as pd
from clinical_text import read_note_text, validate_note_policy, NOTE_CLEANING_POLICY
from protocol import read_json, write_json, sha256_file, SCHEMA_VERSION

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', default='data_n')
    p.add_argument('--cohort', choices=['ms', 'apr'], default='ms')
    p.add_argument('--hour', type=int, choices=[24,48], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--note_similarity_threshold', type=float, default=None,
                   help="Must match finalized cohort; defaults to its recorded threshold")
    a = p.parse_args(argv)
    base = Path(a.data_dir) / a.cohort
    audit = read_json(base / 'finalization_audit.json')
    threshold = validate_note_policy(audit, a.note_similarity_threshold)
    if sha256_file(base / 'splits.csv') != audit['splits_sha256']:
        raise ValueError('Finalized split manifest changed')
    split = pd.read_csv(base / 'splits.csv', dtype={'stay':str, 'DRG_CODE':str})
    selected = split[split.partition == 'test'].copy()
    selected['TEXT'] = [read_note_text(base / 'text_raw' / f'{r.stay}.pk', a.hour, r.ADMITTIME,
                                       threshold, require_schema=True) for r in selected.itertuples()]
    selected['prediction_hour_from_ICU'] = a.hour
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    selected[['stay','SUBJECT_ID','HADM_ID','DRG_CODE','prediction_hour_from_ICU','TEXT']].to_csv(a.output, index=False)
    write_json(str(a.output) + '.audit.json', {
        'schema_version': SCHEMA_VERSION, 'note_similarity_threshold': threshold,
        'note_cleaning_policy': NOTE_CLEANING_POLICY, 'prediction_hour_from_ICU': a.hour,
        'exported_admissions': len(selected), 'splits_sha256': audit['splits_sha256'],
        'finalization_audit_sha256': sha256_file(base / 'finalization_audit.json'),
        'output_sha256': sha256_file(a.output),
    })

if __name__ == '__main__':
    main()
