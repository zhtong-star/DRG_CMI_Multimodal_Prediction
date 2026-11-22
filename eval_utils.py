"""Admission-level classification and first-eligible-admission CMI."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import spearmanr
from protocol import first_admission_mask

def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def validate_prediction_arrays(logits, labels, class_count=None, allow_empty=False):
    """Check the saved single-label class space without coercing invalid labels."""
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels)
    if logits.ndim != 2 or logits.shape[1] < 1 or not np.isfinite(logits).all():
        raise ValueError('Logits must be a finite admissions-by-classes matrix')
    if class_count is not None and logits.shape[1] != class_count:
        raise ValueError('Logits differ from the saved class order')
    if labels.ndim != 1 or len(labels) != len(logits) or (not len(labels) and not allow_empty):
        raise ValueError('One categorical label is required per prediction')
    if len(labels) and (not np.issubdtype(labels.dtype, np.integer)
                        or (labels < 0).any() or (labels >= logits.shape[1]).any()):
        raise ValueError('Labels must be integer indices in the saved class space')
    return logits, labels.astype(np.int64)


def score_f1(logits, y):
    logits, y = validate_prediction_arrays(logits, y)
    return f1_score(y, np.argmax(logits, axis=1), labels=np.arange(logits.shape[1]),
                    average='macro', zero_division=0)

def evaluate_predictions(logits, labels, metadata, label_codes, weight_by_code):
    logits, labels = validate_prediction_arrays(logits, labels, len(label_codes))
    metadata = metadata.reset_index(drop=True)
    if len(logits) != len(metadata):
        raise ValueError('Predictions and admission metadata must have equal nonzero lengths')
    if len(set(map(str, label_codes))) != len(label_codes):
        raise ValueError('Class codes must be unique')
    pred = logits.argmax(1)
    probabilities = softmax(logits)
    truth = np.eye(len(label_codes), dtype=np.int8)[labels]
    valid_auc = [i for i in range(len(label_codes)) if 0 < truth[:, i].sum() < len(labels)]
    per_auc = [roc_auc_score(truth[:, i], probabilities[:, i]) for i in valid_auc]
    order = np.argsort(-logits, axis=1, kind='stable')
    out = {
        'admission_count': len(labels), 'class_count': len(label_codes),
        'macro_F1': score_f1(logits, labels),
        'micro_F1': f1_score(labels, pred, labels=np.arange(len(label_codes)), average='micro', zero_division=0),
        # P097-P099 / Eq. 15 average over the full C-class space. An
        # undefined class AUC cannot silently be dropped from that denominator.
        'macro_AUC': float(np.mean(per_auc)) if len(valid_auc) == len(label_codes) else None,
        'macro_AUC_defined_classes_mean': float(np.mean(per_auc)) if per_auc else None,
        'macro_AUC_defined_classes': len(valid_auc),
        'macro_AUC_undefined_classes': len(label_codes) - len(valid_auc),
        'macro_AUC_undefined_class_codes': [str(c) for i, c in enumerate(label_codes) if i not in valid_auc],
        'macro_AUC_policy': 'mean over all C classes; null if any class lacks positive or negative examples; defined-class mean reported separately',
        'micro_AUC': roc_auc_score(truth.ravel(), probabilities.ravel()) if len(label_codes) > 1 else None,
        'top1_accuracy': float(np.mean(pred == labels)),
        'top5_accuracy': float(np.mean(np.any(order[:, :min(5, len(label_codes))] == labels[:, None], axis=1))),
    }
    weights = np.asarray([weight_by_code[str(code)] for code in label_codes], dtype=float)
    if weights.shape != (len(label_codes),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Each DRG requires a finite positive relative weight')
    true_rw, pred_rw = weights[labels], weights[pred]
    first = first_admission_mask(metadata)
    true_cmi, pred_cmi = float(true_rw[first].mean()), float(pred_rw[first].mean())
    out.update({
        'CMI_population': 'first_eligible_hospital_admission_per_test_patient',
        'CMI_patient_count': int(first.sum()), 'true_CMI': true_cmi, 'predicted_CMI': pred_cmi,
        'APE_CMI_percent': 100 * abs(pred_cmi - true_cmi) / true_cmi,
        'RW_MAE_all_admissions': float(np.mean(np.abs(pred_rw - true_rw))),
        'RW_RMSE_all_admissions': float(np.sqrt(np.mean((pred_rw - true_rw) ** 2))),
        'RW_Spearman_all_admissions': float(spearmanr(pred_rw, true_rw).statistic) if np.std(pred_rw) > 0 and np.std(true_rw) > 0 else None,
    })
    rows = metadata[['stay', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME']].copy()
    rows['true_DRG'] = [label_codes[i] for i in labels]
    rows['predicted_DRG'] = [label_codes[i] for i in pred]
    rows['true_relative_weight'] = true_rw
    rows['predicted_relative_weight'] = pred_rw
    rows['included_in_CMI'] = first
    return out, rows

def aggregate_fold_metrics(fold_metrics):
    if len(fold_metrics) != 5:
        raise ValueError('Five completed fold metrics are required; partial runs cannot be called fivefold means')
    common = set.intersection(*(set(x) for x in fold_metrics))
    summary = {}
    for name in sorted(common):
        vals = [m[name] for m in fold_metrics]
        if all(isinstance(v, (float, int, np.number)) for v in vals):
            summary[name] = {'mean': float(np.mean(vals)), 'std_population': float(np.std(vals)), 'fold_values': vals}
        elif any(v is None for v in vals):
            summary[name] = {'mean': None, 'fold_values': vals, 'reason': 'undefined in at least one fold'}
    return summary
