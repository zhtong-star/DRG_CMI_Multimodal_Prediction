import argparse
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn

from protocol import build_splits, fold_frames, first_admission_mask
from clinical_text import read_note_text, deduplicate_adjacent_paragraphs
from data_utils_multimodal import normalize_structured, fit_struct_stats, load_structured_frame
from extract_structured import create_time_series_features, process_events, VITAL_ITEMS
from models_multimodal import StructuredEncoder, GCMFusion, masked_softmax
from eval_utils import evaluate_predictions, aggregate_fold_metrics
from train_utils_multimodal import train_epoch
from create_cohorts import filter_cohort
from options_multimodal import parse_args
from eval_cc_mcc import parse_severity, conditional_metrics
from generate_report import estimate_temporal_peak_time, estimate_attention_centroid_time

torch.set_num_threads(1)

def cohort():
    rows = []
    for patient in range(30):
        for repeat in range(2 if patient % 3 == 0 else 1):
            rows.append({'stay': f'{patient}_{repeat}', 'SUBJECT_ID': patient,
                         'HADM_ID': patient*10+repeat, 'ADMITTIME': f'2100-01-{repeat+1:02d}',
                         'DRG_CODE': str(patient % 3 + 1)})
    return pd.DataFrame(rows)

def frame(times, values):
    d = pd.DataFrame({'SUBJECT_ID':1, 'HADM_ID':1, 'TIME':times, 'FEATURE':'x', 'VALUENUM':values})
    return create_time_series_features(d, {'SUBJECT_ID':1, 'HADM_ID':1}, 48, ['x'])

class ProtocolTests(unittest.TestCase):
    def test_training_loss_adjustment_and_raw_inference_are_distinct(self):
        from models_multimodal import pick_model
        args = parse_args(['--model','GRU','--logit_adjustment','0.7'])
        args.label_codes, args.class_prior = ['a','b'], [.9,.1]
        model = pick_model(args,struct_input_dim=2)
        logits = torch.tensor([[1.,2.],[3.,1.]])
        labels = torch.tensor([0,1])
        expected = nn.functional.cross_entropy(logits + .7 * torch.tensor([.9,.1]).log(),labels)
        torch.testing.assert_close(model.loss(logits,labels),expected)
        torch.testing.assert_close(model.loss(logits,labels,adjusted=False),nn.functional.cross_entropy(logits,labels))
        self.assertEqual(logits.argmax(1).tolist(),[1,0])

    def test_revised_manuscript_lambda_defaults_to_one_and_zero_remains_explicit(self):
        self.assertEqual(parse_args(['--model', 'LGGCMN']).logit_adjustment, 1.0)
        self.assertEqual(parse_args(['--model', 'LGGCMN', '--logit_adjustment', '0']).logit_adjustment, 0.0)
        self.assertEqual(parse_args(['--model', 'GRU']).logit_adjustment, 0.0)

    def test_temporal_peak_and_attention_centroid_are_separate(self):
        attention = np.array([0.1, 0.6, 0.3, 0.0])
        valid = np.array([True, True, True, False])
        self.assertEqual(estimate_temporal_peak_time(attention, valid), 1)
        self.assertAlmostEqual(estimate_attention_centroid_time(attention, valid), 1.2)
        self.assertIsNone(estimate_temporal_peak_time(attention, np.zeros(4, dtype=bool)))

    def test_class_prior_uses_only_training_partition(self):
        from main_multimodal import prepare_class_prior
        prior = prepare_class_prior(pd.DataFrame({'DRG_CODE':['1','1','2']}), ['1','2','3'])
        self.assertAlmostEqual(prior[0],2/3)
        self.assertAlmostEqual(prior[1],1/3)
        self.assertGreater(prior[2],0)
        self.assertLess(prior[2],1e-10)

    def test_undefined_auc_is_counted_not_fabricated(self):
        meta = pd.DataFrame({'stay':['a','b'],'SUBJECT_ID':[1,2],'HADM_ID':[10,20],
                             'ADMITTIME':['2100-01-01']*2})
        metrics,_ = evaluate_predictions(np.array([[3.,1.,0.],[1.,3.,0.]]),[0,1],meta,['a','b','c'],{'a':1,'b':2,'c':3})
        self.assertEqual(metrics['macro_AUC_defined_classes'],2)
        self.assertEqual(metrics['macro_AUC_undefined_classes'],1)
        self.assertIsNone(metrics['macro_AUC'])
        self.assertEqual(metrics['macro_AUC_defined_classes_mean'],1)
        self.assertAlmostEqual(metrics['macro_F1'],2/3)

    def test_apr_retains_actual_source_policy_not_highest_severity_claim(self):
        from create_cohorts import select_drg
        data = pd.DataFrame({'SUBJECT_ID': [1,1,2,2], 'HADM_ID': [10,10,20,20],
                             'DRG_TYPE': ['APR ']*4, 'DRG_CODE': [7201,7204,1001,2001],
                             'DESCRIPTION': ['same','same','one','other']})
        self.assertEqual(select_drg(data, 'apr').DRG_CODE.tolist(), ['7201'])
        self.assertTrue(select_drg(data, 'apr', 'exclude_ambiguous').empty)

    def test_artifact_change_is_detected(self):
        from protocol import artifact_hashes, verify_artifacts
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'config.json'
            p.write_text('{}')
            hashes = artifact_hashes(folder, ['config.json'])
            verify_artifacts(folder, hashes)
            p.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                verify_artifacts(folder, hashes)

    def test_folds_have_disjoint_patients_and_fixed_test(self):
        df = build_splits(cohort(), seed=1443)
        seen = []
        for i in range(5):
            train, val, test = fold_frames(df, i)
            sets = [set(d.SUBJECT_ID) for d in (train,val,test)]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            self.assertEqual(set(train.SUBJECT_ID)|set(val.SUBJECT_ID), set(df[df.partition=='development'].SUBJECT_ID))
            seen.extend(val.stay)
        self.assertEqual(sorted(seen), sorted(df[df.partition=='development'].stay))
        pd.testing.assert_frame_equal(df, build_splits(cohort(), seed=1443))

    def test_fixed_test_list_rejects_patient_leakage(self):
        with self.assertRaisesRegex(ValueError, 'same patient'):
            build_splits(cohort(), ['0_0'])

    def test_cmi_first_admission_is_chronological(self):
        meta = pd.DataFrame({'stay':['a','b','c'], 'SUBJECT_ID':[1,1,2],
                             'HADM_ID':[12,11,21], 'ADMITTIME':['2100-02-01','2100-01-01','2100-01-01']})
        scores = np.array([[9,0],[0,9],[9,0]])
        report, rows = evaluate_predictions(scores, np.array([0,0,1]), meta, ['a','b'], {'a':1,'b':3})
        self.assertEqual(rows.included_in_CMI.tolist(), [False,True,True])
        self.assertEqual(report['true_CMI'], 2)
        self.assertEqual(report['predicted_CMI'], 2)
        self.assertEqual(report['APE_CMI_percent'], 0)
        self.assertEqual(report['admission_count'], 3)

    def test_hour24_excludes_bin24_and_future_length(self):
        f = frame([0.2,2.4,24.0,24.5,47.9,48.0], [1,3,900,1000,5,9999])
        stats = {'feature_names':['x'],'mean':[0.0],'std':[1.0],'median':[2.0]}
        x, mask = normalize_structured(f, stats, 24)
        self.assertEqual(mask.sum(), 3)
        self.assertEqual(x[:3,0].tolist(), [1,1,3])
        self.assertFalse(x[3:].any())
        x48, r48 = normalize_structured(f, stats, 48)
        self.assertEqual(r48.sum(), 48)
        self.assertEqual(x48[24,0], 950)
        self.assertEqual(x48[47,0], 5)

    def test_leading_and_internal_missingness(self):
        f = frame([2.2,5.2],[4,10])
        x, mask = normalize_structured(f, {'feature_names':['x'],'mean':[2],'std':[2],'median':[2]},48)
        self.assertEqual(x[:6,0].tolist(), [0,0,1,1,1,4])
        self.assertEqual(x[:6,1].tolist(), [0,0,1,0,0,1])
        self.assertEqual(mask.sum(),6)
        self.assertFalse(x[6:].any())

    def test_train_statistics_do_not_read_test(self):
        with tempfile.TemporaryDirectory() as d:
            frame([0],[2]).to_pickle(Path(d)/'train_struct.pk')
            frame([0],[10000]).to_pickle(Path(d)/'test_struct.pk')
            stats = fit_struct_stats(d,pd.DataFrame({'stay':['train']}),['x'])
            self.assertEqual(stats['mean'],[2])
            self.assertEqual(stats['fit_stays'],['train'])

    def test_missing_or_legacy_structured_inputs_fail(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_structured_frame(Path(d)/'absent.pk')
            pd.DataFrame({'TIME':[0],'x':[1]}).to_pickle(Path(d)/'old.pk')
            with self.assertRaisesRegex(ValueError,'Legacy'):
                load_structured_frame(Path(d)/'old.pk')

    def test_text_policy_removes_admin_diagnosis_and_discharge(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({'DIFFTIME':[-100000,-1,1,24.5,2],
                          'CATEGORY':['admission_diag','Nursing','Physician','Nursing','Discharge summary'],
                          'TEXT':['secret_admin','old [**Name**] note','early','late','outcome']}).to_pickle(Path(d)/'x.pk')
            self.assertEqual(read_note_text(Path(d)/'x.pk',24),'old note early')
            self.assertIn('late',read_note_text(Path(d)/'x.pk',48))

    def test_adjacent_note_dedup_uses_both_similarity_measures(self):
        repeated = 'Vitals: HR 80 BP 120/70 RR 16. Assessment stable.'
        changed = 'Vitals: HR 80 BP 120/70 RR 16. Assessment stable. New fever.'
        text = '\n\n'.join([repeated, repeated, changed, repeated])
        result = deduplicate_adjacent_paragraphs(text, 0.95)
        self.assertEqual(result.count('Vitals:'), 3)
        self.assertIn('New fever.', result)

    def test_note_similarity_threshold_is_recorded_in_cli_config(self):
        args = parse_args(['--note_similarity_threshold', '0.87'])
        self.assertAlmostEqual(args.note_similarity_threshold, 0.87)

    def test_padded_suffix_cannot_change_bidirectional_gru(self):
        args = argparse.Namespace(struct_hidden_size=4, struct_num_layers=1,bidirectional=True,dropout_rate=0)
        encoder = StructuredEncoder(args,2).eval()
        x = torch.randn(2,6,2)
        mask = torch.tensor([[1,1,0,0,0,0],[1,1,1,1,0,0]],dtype=torch.bool)
        other = x.clone()
        other[~mask] = 999
        self.assertTrue(torch.allclose(encoder(x,mask),encoder(other,mask)))
        self.assertTrue(torch.equal(encoder(x,mask)[~mask],torch.zeros_like(encoder(x,mask)[~mask])))

    def test_gcm_equation_and_all_masked_sequence(self):
        fusion = GCMFusion(4).eval()
        D, T = torch.randn(2,3,4), torch.randn(2,6,4)
        mask = torch.tensor([[1,1,0,0,0,0],[0,0,0,0,0,0]],dtype=torch.bool)
        fused, alpha, gate = fusion(D,T,mask)
        self.assertFalse(alpha[:,:,2:].any())
        expected = gate*D+(1-gate)*torch.bmm(alpha,fusion.W_V(T))
        self.assertTrue(torch.allclose(fused,expected))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertFalse(alpha[1].any())

    def test_partial_fold_aggregation_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_fold_metrics([{'x':1}]*4)
        out=aggregate_fold_metrics([{'x':i} for i in range(5)])
        self.assertEqual(out['x']['mean'],2)

    def test_lambda_default_matches_revised_manuscript(self):
        self.assertEqual(parse_args([]).logit_adjustment, 1.0)
        self.assertEqual(parse_args(['--logit_adjustment','0.7']).logit_adjustment,0.7)

    def test_ambiguous_no_mcc_is_not_labeled_cc(self):
        self.assertEqual(parse_severity('SEPSIS W/O MCC'),(None,None))
        rules=pd.DataFrame({'DRG_CODE':['1','2','3'],'DESCRIPTION':['SEPSIS W/O CC/MCC','SEPSIS W CC','SEPSIS W MCC']})
        report,_=conditional_metrics(np.array([[0,1,9]]),[0],['x'],rules,['1','2','3'])
        self.assertEqual(report['major_error_rate'],1)

    def test_uneven_accumulation_matches_one_large_batch(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc=nn.Linear(2,2,bias=False)
            def forward(self,b):
                return self.fc(b['text'])
            def loss(self,x,y,adjusted=True):
                return nn.functional.cross_entropy(x,y)
        torch.manual_seed(3)
        m1=Model()
        m2=copy.deepcopy(m1)
        data=[{'text':torch.tensor([float(i),1.]),'drg':torch.tensor(i%2)} for i in range(5)]
        o1=torch.optim.SGD(m1.parameters(),lr=.1)
        o2=torch.optim.SGD(m2.parameters(),lr=.1)
        train_epoch(m1,data,o1,2,accumulation=3)
        train_epoch(m2,data,o2,5,accumulation=1)
        self.assertTrue(torch.allclose(m1.fc.weight,m2.fc.weight,atol=1e-6))

    def test_icu_length_filter_and_shifted_old_age(self):
        labels=pd.DataFrame({'SUBJECT_ID':[1,2], 'HADM_ID':[11,22], 'DRG_CODE':['1','2']})
        admissions=pd.DataFrame({'SUBJECT_ID':[1,2], 'HADM_ID':[11,22], 'ADMITTIME':['2150-01-01']*2})
        patients=pd.DataFrame({'SUBJECT_ID':[1,2], 'DOB':['1800-01-01','2100-01-01']})
        icu=pd.DataFrame({'SUBJECT_ID':[1,2],'HADM_ID':[11,22],'ICUSTAY_ID':[1,2],
                          'INTIME':['2150-01-01']*2,'OUTTIME':['2150-01-03','2150-01-01 12:00:00'],
                          'LOS':[2,.5], 'FIRST_WARDID':[1,1],'LAST_WARDID':[1,1],
                          'FIRST_CAREUNIT':['A','A'],'LAST_CAREUNIT':['A','A']})
        # Uniform time formatting avoids inferred-format ambiguity in pandas 2.
        icu['OUTTIME']=['2150-01-03 00:00:00','2150-01-01 12:00:00']
        eligible,_=filter_cohort(labels,admissions,patients,icu)
        self.assertEqual(eligible.SUBJECT_ID.tolist(),[1])

if __name__ == '__main__':
    unittest.main()
