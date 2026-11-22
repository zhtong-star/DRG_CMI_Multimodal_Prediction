"""Read-only checks against local MIMIC tables; save aggregate counts only.

This samples event tables, so its counts are NOT a reconstruction of Table 1.
No identifiers, clinical text, model weights, or real observations are exported.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from create_cohorts import select_drg, filter_cohort, collect_notes, count_stage
from extract_structured import VITAL_ITEMS, LAB_ITEMS, FEATURE_NAMES, process_events, aggregate_events, merge_aggregates, to_hourly_frame
from data_utils_multimodal import load_rule, fit_struct_stats, normalize_structured
from models_multimodal import pick_model
from protocol import write_json, sha256_file, SCHEMA_VERSION
from clinical_text import NOTE_CLEANING_POLICY


def audit_metadata(root, rule_dir):
    """Read only small source tables; never access NOTEEVENTS or event tables."""
    root, rule_dir = Path(root), Path(rule_dir)
    drg = pd.read_csv(root/'DRGCODES.csv')
    admissions = pd.read_csv(root/'ADMISSIONS.csv', usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME'])
    patients = pd.read_csv(root/'PATIENTS.csv', usecols=['SUBJECT_ID','DOB'])
    icu = pd.read_csv(root/'ICUSTAYS.csv')
    summary = {
        'schema_version': SCHEMA_VERSION, 'full_training': False,
        'count_provenance': 'Actual local metadata filtering; not a reconstruction of manuscript Table 1',
        'cohort_stages': {}, 'source_sha256': {}, 'mapping_audit': {},
        'feature_count': len(FEATURE_NAMES), 'note_cleaning_policy': NOTE_CLEANING_POLICY,
        'cohort_policy': {'age': 'calendar age >=18', 'single_icu': True,
                          'no_transfer': 'original-code filter retained; manuscript P032 is silent',
                          'icu_duration': 'OUTTIME-INTIME >=24 hours',
                          'apr_label': 'source_first_per_description; residual ambiguity excluded'},
    }
    for name in ['DRGCODES.csv','ADMISSIONS.csv','PATIENTS.csv','ICUSTAYS.csv']:
        summary['source_sha256'][name] = sha256_file(root/name)
    frames = []
    for system in ['ms','apr']:
        labels = select_drg(drg, system)
        frame, stages = filter_cohort(labels, admissions, patients, icu)
        rule_path = rule_dir/f'{system.upper()}DRG_RULE13.csv'
        rules = load_rule(rule_path)
        frame = frame[frame.DRG_CODE.isin(rules[1])]
        stages['complete_label_and_weight'] = count_stage(frame)
        summary['cohort_stages'][system] = stages
        summary['source_sha256'][rule_path.name] = sha256_file(rule_path)
        frames.append(frame)
    for name, mapping in [('D_ITEMS.csv',VITAL_ITEMS),('D_LABITEMS.csv',LAB_ITEMS)]:
        path = root/name
        entry = {'dictionary_available': path.is_file(), 'mapped_features': len(mapping),
                 'configured_itemids': sorted(i for ids in mapping.values() for i in ids)}
        if path.is_file():
            dictionary = pd.read_csv(path)
            ids = set(pd.to_numeric(dictionary.ITEMID, errors='coerce').dropna().astype(int))
            entry['unmatched_itemids'] = sorted(set(entry['configured_itemids']) - ids)
            entry['dictionary_sha256'] = sha256_file(path)
            # Dictionary labels/units are public metadata, not patient observations.
            entry['matched_items'] = [
                {k: (None if pd.isna(v) else str(v)) for k, v in record.items()}
                for record in dictionary[dictionary.ITEMID.isin(entry['configured_itemids'])][
                    [c for c in ['ITEMID','LABEL','UNITNAME','FLUID'] if c in dictionary]].to_dict('records')]
        else:
            entry['verification'] = 'Unavailable locally; ITEMID semantics not independently verified'
        summary['mapping_audit'][name] = entry
    return summary, pd.concat(frames).drop_duplicates('HADM_ID')


def save_summary(output, summary):
    if output != '-':
        write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mimic_dir', required=True)
    p.add_argument('--rule_dir', required=True)
    p.add_argument('--event_rows', type=int, default=200000)
    p.add_argument('--note_rows', type=int, default=400000)
    p.add_argument('--output', default='.qa/real_data_summary.json', help='Use - for stdout only')
    p.add_argument('--metadata_only', action='store_true', help='Read small cohort/mapping tables only; no notes, events, or model execution')
    p.add_argument('--prior_summary', help='Compare common cohort stages with a prior aggregate audit without overwriting it')
    a = p.parse_args(argv)
    root = Path(a.mimic_dir)
    summary, cohort = audit_metadata(root, a.rule_dir)
    summary['metadata_only'] = a.metadata_only
    if a.prior_summary:
        prior = json.loads(Path(a.prior_summary).read_text(encoding='utf-8'))
        summary['prior_common_count_stages_equal'] = {
            system: {stage: counts == summary['cohort_stages'][system].get(stage)
                     for stage, counts in stages.items() if isinstance(counts, dict)}
            for system, stages in prior.get('cohort_stages', {}).items()}
        summary['prior_note_and_event_evidence'] = 'Not revalidated: text cleaning semantics changed; no prior model results asserted'
    if a.metadata_only:
        summary['not_evaluated'] = ['note eligibility and cleaning on real data', 'structured extraction',
                                    'final multimodal cohort sizes', 'training or clinical performance']
        save_summary(a.output, summary)
        return summary
    if a.event_rows <= 0 or a.note_rows <= 0:
        raise ValueError('Sample row limits must be positive')
    torch.set_num_threads(1)
    summary.update({'event_rows_limit_per_table': a.event_rows, 'note_rows_limit': a.note_rows})
    parts = []
    summary['sampled_events'] = {}
    for filename, mapping in [('CHARTEVENTS.csv',VITAL_ITEMS),('LABEVENTS.csv',LAB_ITEMS)]:
        cols = ['SUBJECT_ID','HADM_ID','ITEMID','CHARTTIME','VALUENUM']
        if filename == 'CHARTEVENTS.csv': cols += ['ERROR']
        chunk = pd.read_csv(root/filename, usecols=cols, nrows=a.event_rows)
        events = process_events(chunk, cohort, mapping)
        assert events.TIME.ge(0).all() and events.TIME.lt(48).all()
        summary['sampled_events'][filename] = {'rows_read':len(chunk),'eligible_observations':len(events)}
        parts.append(aggregate_events(events))
    agg = merge_aggregates(parts)
    assert not agg.empty
    Path('.qa').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='real_struct_',dir='.qa') as directory:
        folder = Path(directory)
        stays = []
        for (subject, hadm), group in agg.groupby(['SUBJECT_ID','HADM_ID']):
            stay = f'{int(subject)}_{int(hadm)}'
            to_hourly_frame(group).to_pickle(folder/f'{stay}_struct.pk')
            stays.append(stay)
        stats = fit_struct_stats(folder,pd.DataFrame({'stay':stays}),FEATURE_NAMES)
        batch = []
        masks = []
        for stay in stays[:2]:
            frame = pd.read_pickle(folder/f'{stay}_struct.pk')
            values, mask = normalize_structured(frame,stats,24)
            assert not mask[24:].any() and not values[24:].any()
            assert np.isfinite(values).all()
            batch.append(values); masks.append(mask)
        args = SimpleNamespace(model='GRU',label_codes=['synthetic_a','synthetic_b'],class_prior=[.5,.5],
                    logit_adjustment=0.,struct_hidden_size=8,struct_num_layers=1,bidirectional=False,
                    dropout_rate=0.,fusion_method='gcm')
        model = pick_model(args,struct_input_dim=58)
        with torch.no_grad():
            logits = model({'structured':torch.tensor(np.stack(batch)), 'structured_mask':torch.tensor(np.stack(masks))})
        assert torch.isfinite(logits).all()
        summary['real_structured_smoke'] = {'admissions_in_event_sample':len(stays),'batch_size':len(batch),
               'features':29,'channels':58,'finite_logits':True,'initialization':'random; no clinical performance evaluated'}
    notes, counts = collect_notes(root,cohort,chunksize=10000,max_rows=a.note_rows)
    assert len(notes)>0, 'Increase --note_rows to reach timestamped early notes'
    assert not notes.CATEGORY.str.lower().eq('discharge summary').any()
    assert not notes.TEXT.str.contains(r'\[\*\*',regex=True).any()
    summary['sampled_notes'] = counts
    save_summary(a.output, summary)
    return summary

if __name__ == '__main__':
    main()
