"""Conditional MS-DRG family severity analysis on saved common-cohort predictions."""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from data_utils_multimodal import load_rule
from eval_utils import validate_prediction_arrays
from protocol import write_json, read_json, sha256_file

def parse_severity(description):
    text = re.sub(r'\s+', ' ', str(description).upper()).strip()
    for suffix, level in [('WITHOUT CC/MCC', 0), ('W/O CC/MCC', 0), ('WITH MCC', 2),
                          ('W MCC', 2), ('WITH CC', 1), ('W CC', 1)]:
        if text.endswith(suffix):
            return text[:-len(suffix)].strip(' ,;-'), level
    # W/O MCC includes both CC and no-CC cases: it cannot mean "with CC".
    return None, None

def conditional_metrics(logits, labels, stays, rules, label_codes):
    logits, labels = validate_prediction_arrays(logits, labels, len(label_codes), allow_empty=True)
    if len(stays) != len(labels):
        raise ValueError('Severity predictions, labels, and admission IDs must align')
    code2idx = {str(c): i for i, c in enumerate(label_codes)}
    if len(code2idx) != len(label_codes):
        raise ValueError('Class codes must be unique')
    families, lookup = {}, {}
    ambiguous = 0
    for row in rules.itertuples():
        code = str(row.DRG_CODE)
        if code not in code2idx:
            continue
        base, level = parse_severity(row.DESCRIPTION)
        if base is None:
            ambiguous += 1
            continue
        families.setdefault(base, {})[code] = level
        lookup[code] = (base, level)
    families = {b: members for b, members in families.items() if len(set(members.values())) >= 2}
    records = []
    for scores, y, stay in zip(logits, labels, stays):
        true_code = str(label_codes[int(y)])
        base, true_level = lookup.get(true_code, (None, None))
        if base not in families:
            continue
        members = sorted(families[base])
        ranked = sorted(members, key=lambda c: (-float(scores[code2idx[c]]), c))
        # P126 evaluates three severity categories, not duplicate DRG codes
        # belonging to the same category. Preserve the existing top-code rule
        # while allowing each severity category only one top-k position.
        ranked_levels = list(dict.fromkeys(families[base][code] for code in ranked))
        level = ranked_levels[0]
        records.append({'stay': stay, 'family': base, 'true_DRG': true_code, 'predicted_DRG': ranked[0],
                        'true_level': true_level, 'predicted_level': level,
                        'correct_top2': true_level in ranked_levels[:2], 'major_error': abs(level - true_level) == 2})
    rows = pd.DataFrame(records, columns=['stay', 'family', 'true_DRG', 'predicted_DRG',
                                         'true_level', 'predicted_level', 'correct_top2', 'major_error'])
    report = {'scope': 'conditional_on_ground_truth_family', 'n_family_samples': len(rows),
              'n_families': int(rows.family.nunique()), 'unmapped_or_ambiguous_rule_codes': ambiguous,
              'top1_accuracy': None, 'top2_accuracy': None, 'major_error_rate': None,
              'family_mapping_policy': 'exact base description after unambiguous severity suffix removal; at least two distinct levels',
              'severity_ranking_policy': 'distinct severity levels ordered by highest raw member-code logit; ties by code'}
    confusion = np.zeros((3,3), dtype=int)
    if len(rows):
        report.update({'top1_accuracy': float((rows.true_level == rows.predicted_level).mean()),
                       'top2_accuracy': float(rows.correct_top2.mean()), 'major_error_rate': float(rows.major_error.mean())})
        for t, p in zip(rows.true_level, rows.predicted_level):
            confusion[int(t), int(p)] += 1
        for level in range(3):
            selected = rows[rows.true_level == level]
            report[f'accuracy_level_{level}'] = float((selected.predicted_level == level).mean()) if len(selected) else None
    report['confusion_matrix_rows_true_columns_pred'] = confusion.tolist()
    report['severity_order'] = ['without_CC_MCC', 'with_CC', 'with_MCC']
    return report, rows

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions_dir', required=True, help='fold_N/eval_24h or fold_N/eval_48h')
    p.add_argument('--rule_file', required=True, help='MS-DRG 2013 rule table')
    a = p.parse_args(argv)
    out = Path(a.predictions_dir)
    run_config = __import__('json').loads((out.parent / 'config.json').read_text(encoding='utf-8'))
    if run_config['cohort'] != 'ms':
        raise ValueError('CC/MCC family analysis is defined for MS-DRG, not APR-DRG severity subgroups')
    if sha256_file(a.rule_file) != read_json(out.parent.parent / 'data_provenance.json')['rule_sha256']:
        raise ValueError('Rule table differs from training-run provenance')
    saved = np.load(out / 'logits.npz', allow_pickle=False)
    if saved['label_codes'].tolist() != run_config['label_codes']:
        raise ValueError('Prediction class order differs from training configuration')
    test = pd.read_csv(out.parent / 'test.csv', dtype={'stay': str, 'DRG_CODE': str})
    if test.stay.tolist() != saved['stays'].tolist():
        raise ValueError('Prediction cases differ from the saved hold-out')
    code2idx = {str(code): i for i, code in enumerate(run_config['label_codes'])}
    expected_labels = test.DRG_CODE.map(code2idx)
    if expected_labels.isna().any() or not np.array_equal(expected_labels.to_numpy(), saved['labels']):
        raise ValueError('Prediction labels differ from the saved hold-out')
    rules = load_rule(a.rule_file)[0]
    report, cases = conditional_metrics(saved['logits'], saved['labels'], saved['stays'], rules, saved['label_codes'])
    write_json(out / 'cc_mcc_metrics.json', report)
    cases.to_csv(out / 'cc_mcc_cases.csv', index=False)
    print(report)

if __name__ == '__main__':
    main()
