"""Offline checks of manuscript equations and attribution using synthetic inputs."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import torch
from torch import nn
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, LongformerConfig, LongformerModel, BertConfig, BertModel

from models_multimodal import pick_model, GCMFusion, StructuredEncoder
from options_multimodal import parse_args
from train_utils_multimodal import predict, train_epoch
from eval_utils import evaluate_predictions, score_f1, aggregate_fold_metrics
from eval_cc_mcc import conditional_metrics, parse_severity
from generate_report import (estimate_temporal_peak_time, estimate_attention_centroid_time,
                             top_word_spans, make_case_report)


torch.set_num_threads(1)


def args_for(model='LGGCMN', *extra):
    args = parse_args(['--model', model, '--max_seq_length', '8', '--struct_hidden_size', '4',
                       '--struct_num_layers', '1', '--dropout_rate', '0', '--cnn_filter_maps', '4', *extra])
    args.label_codes = ['a', 'b', 'c']
    args.class_prior = [.98, .01, .01]
    return args


class TinyEncoder(nn.Module):
    """Known token states for checking equations without pretrained downloads."""
    def __init__(self, model_type='longformer'):
        super().__init__()
        self.config = argparse.Namespace(hidden_size=4, model_type=model_type)
        self.embedding = nn.Embedding(12, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(48).reshape(12, 4) / 30)
        self.inputs = None

    def forward(self, **inputs):
        self.inputs = inputs
        return argparse.Namespace(last_hidden_state=self.embedding(inputs['input_ids']))


def build_model(args=None, encoder=None):
    args = args or args_for()
    encoder = encoder or TinyEncoder(args.text_backbone)
    with patch('models_multimodal.AutoModel.from_pretrained', return_value=encoder):
        return pick_model(args, struct_input_dim=2).eval()


def batch():
    return {'text': torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]]),
            'text_mask': torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool),
            'structured': torch.tensor([[[1., 1.], [2., 1.], [3., 1.], [999., 999.]]]),
            'structured_mask': torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)}


def metadata():
    return pd.DataFrame({'stay': ['late', 'first', 'other'], 'SUBJECT_ID': [1, 1, 2],
                         'HADM_ID': [12, 11, 21],
                         'ADMITTIME': ['2100-02-01', '2100-01-01', '2100-01-01']})


def wordpiece_tokenizer():
    tokens = ['[PAD]', '[UNK]', 'one', 'two', 'three', 'four', 'kid', '##ney', 'six',
              'seven', 'eight', 'nine', 'ten']
    raw = Tokenizer(WordPiece({word: i for i, word in enumerate(tokens)}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', unk_token='[UNK]')


class ModelAlignmentTests(unittest.TestCase):
    def test_equations_1_2_4_5_6_7_and_padding(self):
        torch.manual_seed(11)
        model, inputs = build_model(), batch()
        logits, attention = model(inputs, return_attn=True)
        H = model.text_encoder.embedding(inputs['text'])
        A = torch.softmax(model.text_U(H[:, :3]).transpose(1, 2), dim=-1)
        D = torch.bmm(A, H[:, :3])
        T = model.struct_proj(model.struct_encoder(inputs['structured'], inputs['structured_mask']))[:, :3]
        cross = torch.softmax(torch.bmm(model.gcm.W_Q(D), model.gcm.W_K(T).transpose(1, 2)) / 2, dim=-1)
        S = torch.bmm(cross, model.gcm.W_V(T))
        gate = torch.sigmoid(model.gcm.W_g(torch.cat([D, S], dim=-1)))
        F = gate * D + (1 - gate) * S
        expected = torch.stack([F[:, i] @ model.fc.weight[i] + model.fc.bias[i]
                                for i in range(3)], dim=1)
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(attention['text'][:, :, :3], A)
        torch.testing.assert_close(attention['temporal'][:, :, :3], cross)
        torch.testing.assert_close(attention['gate'], gate)
        self.assertFalse(attention['text'][:, :, 3:].any())
        self.assertFalse(attention['temporal'][:, :, 3:].any())
        model.loss(logits, torch.tensor([1])).backward()
        self.assertFalse(model.text_encoder.embedding.weight.grad[0].any())
        self.assertTrue(torch.isfinite(model.gcm.W_Q.weight.grad).all())

    def test_training_prior_only_raw_predictions_and_categorical_loss(self):
        args = args_for()
        args.logit_adjustment = None
        model = build_model(args)
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.copy_(torch.tensor([0., 1., .5]))
        inputs = batch()
        scores = model(inputs)
        self.assertEqual(model.logit_adjustment, 1.)
        self.assertEqual(scores.argmax(1).item(), 1)
        self.assertEqual((scores + model.class_log_prior).argmax(1).item(), 0)
        expected = -torch.log_softmax(scores + torch.tensor(args.class_prior).log(), dim=1)[0, 1]
        torch.testing.assert_close(model.loss(scores, torch.tensor([1])), expected)
        item = {k: v[0] for k, v in inputs.items()}
        item.update(drg=torch.tensor(1), entry='synthetic')
        actual, labels, entries, val_loss = predict(model, [item])
        np.testing.assert_allclose(actual, scores.detach().numpy())
        self.assertEqual(entries, ['synthetic'])
        self.assertEqual(labels.tolist(), [1])
        self.assertAlmostEqual(val_loss, nn.functional.cross_entropy(scores, torch.tensor([1])).item())

    def test_model_defaults_and_explicit_overrides_remain_separate(self):
        for name in ['LGGCMN', 'Concat', 'Longformer', 'GRU', 'CAML']:
            self.assertEqual(args_for(name).logit_adjustment, 1. if name == 'LGGCMN' else 0.)
        self.assertEqual(args_for('LGGCMN', '--logit_adjustment', '0').logit_adjustment, 0)
        self.assertEqual(args_for('LGGCMN', '--logit_adjustment', '.7').logit_adjustment, .7)
        for bad in ([.5, float('nan'), .5], [[.3, .3, .4]], [.5, 0, .5]):
            args = args_for('GRU')
            args.class_prior = bad
            with self.assertRaisesRegex(ValueError, 'prior'):
                pick_model(args, struct_input_dim=2)

    def test_gates_represent_text_fraction_and_masked_time_has_no_gradient(self):
        torch.manual_seed(12)
        fusion = GCMFusion(4)
        D = torch.randn(2, 3, 4, requires_grad=True)
        T = torch.randn(2, 4, 4, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            fusion.W_g.weight.zero_()
            fusion.W_g.bias.fill_(30)
        fused, _, _ = fusion(D, T, mask)
        torch.testing.assert_close(fused, D)
        with torch.no_grad():
            fusion.W_g.bias.fill_(-30)
        fused, attention, _ = fusion(D, T, mask)
        torch.testing.assert_close(fused, torch.bmm(attention, fusion.W_V(T)), atol=1e-6, rtol=1e-6)
        fused.sum().backward()
        self.assertFalse(T.grad[~mask].any())
        self.assertFalse(attention[1].any())
        self.assertTrue(torch.isfinite(fused).all())

    def test_bidirectional_gru_and_all_masked_sequences_ignore_padding(self):
        args = args_for('GRU', '--bidirectional')
        encoder = StructuredEncoder(args, 2).eval()
        x = torch.randn(2, 4, 2)
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
        changed = x.clone()
        changed[~mask] = 500
        torch.testing.assert_close(encoder(x, mask), encoder(changed, mask))
        self.assertFalse(encoder(x, mask)[~mask].any())

    def test_caml_padding_cannot_enter_neighboring_convolutions(self):
        args = args_for('CAML')
        embeddings = np.arange(48, dtype=np.float32).reshape(12, 4) + 1
        model = pick_model(args, embeddings).eval()
        inputs, altered = batch(), batch()
        altered['text'][~altered['text_mask']] = 10
        torch.testing.assert_close(model(inputs), model(altered))
        _, attention = model(inputs, return_attn=True)
        self.assertFalse(attention['text'][:, :, 3:].any())

    def test_ablation_routes_global_pooling_and_concat(self):
        inputs = batch()
        global_model = build_model(args_for('LGGCMN', '--text_pooling', 'global'))
        _, attention = global_model(inputs, return_attn=True)
        torch.testing.assert_close(attention['text'][:, :, :3], torch.full((1, 3, 3), 1/3))
        torch.testing.assert_close(attention['temporal'][:, 0], attention['temporal'][:, 1])
        ablation = build_model(args_for('LGGCMN', '--fusion_method', 'concat'))
        baseline = build_model(args_for('Concat'))
        baseline.load_state_dict(ablation.state_dict())
        torch.testing.assert_close(ablation(inputs), baseline(inputs))
        self.assertIsNone(ablation(inputs, return_attn=True)[1]['gate'])
        text_only = build_model(args_for('Longformer'))
        self.assertEqual(text_only({k: inputs[k] for k in ['text', 'text_mask']}).shape, (1, 3))
        gru = pick_model(args_for('GRU'), struct_input_dim=2)
        self.assertEqual(gru({k: inputs[k] for k in ['structured', 'structured_mask']}).shape, (1, 3))

    def test_real_tiny_backbones_receive_masks_and_produce_finite_logits(self):
        configs = [LongformerConfig(vocab_size=12, hidden_size=4, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=8, max_position_embeddings=18,
                                   attention_window=[4], pad_token_id=0, hidden_dropout_prob=0,
                                   attention_probs_dropout_prob=0),
                   BertConfig(vocab_size=12, hidden_size=4, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=8, max_position_embeddings=16,
                              pad_token_id=0, hidden_dropout_prob=0, attention_probs_dropout_prob=0)]
        for config, constructor in zip(configs, [LongformerModel, BertModel]):
            args = args_for('LGGCMN', '--text_backbone', config.model_type)
            model = build_model(args, constructor(config))
            inputs, altered = batch(), batch()
            altered['text'][~altered['text_mask']] = 10
            scores, attention = model(inputs, return_attn=True)
            self.assertTrue(torch.isfinite(scores).all())
            torch.testing.assert_close(scores, model(altered))
            self.assertFalse(attention['text'][:, :, 3:].any())
            self.assertFalse(attention['temporal'][:, :, 3:].any())

    def test_uneven_accumulation_uses_training_adjustment(self):
        class LinearDRG(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 3)
            def forward(self, b):
                return self.fc(b['text'])
            def loss(self, logits, labels, adjusted=True):
                return nn.functional.cross_entropy(logits + torch.tensor([.8, .1, .1]).log(), labels)
        torch.manual_seed(8)
        model = LinearDRG()
        expected = copy.deepcopy(model)
        items = [{'text': torch.tensor([float(i), 1.]), 'drg': torch.tensor(i % 3)} for i in range(5)]
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        reference = torch.optim.SGD(expected.parameters(), lr=.01)
        train_epoch(model, items, optimizer, batch_size=2, accumulation=3)
        X = torch.stack([row['text'] for row in items])
        y = torch.stack([row['drg'] for row in items])
        nn.functional.cross_entropy(expected.fc(X) + torch.tensor([.8, .1, .1]).log(), y).backward()
        reference.step()
        torch.testing.assert_close(model.fc.weight, expected.fc.weight)
        torch.testing.assert_close(model.fc.bias, expected.fc.bias)


class MetricAlignmentTests(unittest.TestCase):
    def test_auc_missing_class_is_explicit_and_f1_keeps_full_class_space(self):
        scores = np.array([[3, 1, 0], [1, 3, 0]])
        report, _ = evaluate_predictions(scores, [0, 1], metadata().iloc[[1, 2]],
                                          ['a', 'b', 'c'], {'a': 1, 'b': 2, 'c': 3})
        self.assertIsNone(report['macro_AUC'])
        self.assertEqual(report['macro_AUC_defined_classes_mean'], 1)
        self.assertEqual(report['macro_AUC_undefined_class_codes'], ['c'])
        self.assertAlmostEqual(report['macro_F1'], 2/3)
        self.assertEqual(report['micro_F1'], 1)
        self.assertAlmostEqual(score_f1(scores, [0, 1]), 2/3)
        summary = aggregate_fold_metrics([report] * 5)
        self.assertIsNone(summary['macro_AUC']['mean'])
        self.assertEqual(summary['macro_AUC_defined_classes_mean']['mean'], 1)

    def test_all_class_auc_and_top5_are_computed_from_raw_logits(self):
        scores = np.eye(6) * 3
        scores[0] = [0, 6, 5, 4, 3, 2]
        meta = pd.DataFrame({'stay': list('abcdef'), 'SUBJECT_ID': range(6),
                             'HADM_ID': range(6), 'ADMITTIME': ['2100-01-01']*6})
        report, rows = evaluate_predictions(scores, np.arange(6), meta, list('abcdef'), dict.fromkeys('abcdef', 1))
        self.assertEqual(report['macro_AUC_defined_classes'], 6)
        self.assertIsNotNone(report['macro_AUC'])
        self.assertEqual(report['macro_AUC'], report['macro_AUC_defined_classes_mean'])
        self.assertAlmostEqual(report['top5_accuracy'], 5/6)
        self.assertAlmostEqual(report['micro_F1'], report['top1_accuracy'])
        self.assertEqual(rows.predicted_DRG.iloc[0], 'b')

    def test_cmi_is_first_admission_hard_assignment_and_classification_uses_all(self):
        scores = np.array([[0, 2], [1, 0], [1, 0]])
        report, rows = evaluate_predictions(scores, [0, 1, 0], metadata(), ['a', 'b'], {'a': 1, 'b': 3})
        self.assertEqual(rows.included_in_CMI.tolist(), [False, True, True])
        self.assertEqual(report['admission_count'], 3)
        self.assertEqual(report['CMI_patient_count'], 2)
        self.assertEqual(report['true_CMI'], 2)
        self.assertEqual(report['predicted_CMI'], 1)
        self.assertEqual(report['APE_CMI_percent'], 50)
        self.assertAlmostEqual(report['top1_accuracy'], 1/3)

    def test_invalid_labels_and_weights_cannot_silently_change_metrics(self):
        for labels in ([-1, 0, 1], [0., 1., .5], [0, 1, 2]):
            with self.assertRaises(ValueError):
                evaluate_predictions(np.ones((3, 2)), labels, metadata(), ['a', 'b'], {'a': 1, 'b': 2})
        with self.assertRaisesRegex(ValueError, 'weight'):
            evaluate_predictions(np.ones((3, 2)), [0, 1, 0], metadata(), ['a', 'b'], {'a': 1, 'b': float('nan')})

    def test_conditional_top2_means_distinct_severity_levels(self):
        rules = pd.DataFrame({'DRG_CODE': list('abcde'),
                              'DESCRIPTION': ['SEPSIS W/O CC/MCC', 'SEPSIS W CC', 'SEPSIS WITH CC',
                                              'SEPSIS W MCC', 'OTHER W MCC']})
        report, rows = conditional_metrics([[4, 5, 6, 1, 100], [10, 8, 7, 1, 100]],
                                            [0, 3], ['one', 'two'], rules, list('abcde'))
        self.assertEqual(report['top2_accuracy'], .5)
        self.assertEqual(report['major_error_rate'], .5)
        self.assertEqual(report['n_families'], 1)
        self.assertEqual(rows.predicted_DRG.tolist(), ['c', 'a'])
        self.assertEqual(report['confusion_matrix_rows_true_columns_pred'], [[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        self.assertEqual(parse_severity('SEPSIS W/O MCC'), (None, None))
        with self.assertRaisesRegex(ValueError, 'align'):
            conditional_metrics([[1, 2, 3, 4, 5]], [0], [], rules, list('abcde'))


class AttributionAlignmentTests(unittest.TestCase):
    def test_peak_and_centroid_have_distinct_meanings(self):
        attention, mask = [.1, .6, .3, 900], [True, True, True, False]
        self.assertEqual(estimate_temporal_peak_time(attention, mask), 1)
        self.assertAlmostEqual(estimate_attention_centroid_time(attention, mask), 1.2)
        self.assertEqual(estimate_temporal_peak_time([.5, .5, 0], [True]*3), 0)
        for weights, valid in [([0, 0], [True, True]), ([1, 2], [False, False])]:
            self.assertIsNone(estimate_temporal_peak_time(weights, valid))
            self.assertIsNone(estimate_attention_centroid_time(weights, valid))
        with self.assertRaises(ValueError):
            estimate_temporal_peak_time([1, 2], [True])
        with self.assertRaises(ValueError):
            estimate_temporal_peak_time([float('nan')], [True])

    def test_five_word_spans_sum_subwords_and_exclude_partial_trailing_word(self):
        text = 'one two three four kidney six'
        tokenizer = wordpiece_tokenizer()
        self.assertEqual(top_word_spans(text, tokenizer, [.1]*5, 5), [])
        spans = top_word_spans(text, tokenizer, [.1]*6, 6)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['text'], 'one two three four kidney')
        self.assertAlmostEqual(spans[0]['attention_sum'], .6)
        self.assertEqual(text[spans[0]['start_char']:spans[0]['end_char']], spans[0]['text'])
        self.assertEqual(top_word_spans('one two', tokenizer, [.1]*4, 4), [])
        with self.assertRaisesRegex(ValueError, 'align'):
            top_word_spans(text, tokenizer, [.1]*5, 6)

    def test_five_word_spans_are_contiguous_nonoverlapping_and_mask_padding(self):
        text = 'one two three four six seven eight nine ten one'
        spans = top_word_spans(text, wordpiece_tokenizer(), [.1]*10 + [900]*2, 12)
        self.assertEqual([s['text'] for s in spans],
                         ['one two three four six', 'seven eight nine ten one'])
        self.assertTrue(all(len(s['text'].split()) == 5 for s in spans))
        self.assertTrue(all(abs(s['attention_sum'] - .5) < 1e-9 for s in spans))

    def test_case_report_uses_predicted_class_same_peak_physiology_and_mean_gate(self):
        class CaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
            def forward(self, inputs, return_attn=False):
                return torch.tensor([[0., 2.]]), {
                    'temporal': torch.tensor([[[.2, .3, .5, 0], [.1, .6, .3, 0]]]),
                    'text': torch.full((1, 2, 5), .2),
                    'gate': torch.tensor([[[.1, .2], [.6, .8]]])}
        class CaseDataset:
            tokenizer = wordpiece_tokenizer()
            def __init__(self, *args, **kwargs):
                pass
            def load_data(self, hour):
                self.hour = hour
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return {'structured': torch.tensor([[.1, 1], [9, 1], [.3, 1], [0, 0]]),
                        'structured_mask': torch.tensor([True, True, True, False])}
            def read_text(self, *args):
                return 'one two three four six'
        args = args_for()
        args.label_codes, args.max_seq_length = ['a', 'b'], 5
        stats = {'feature_names': ['synthetic_feature'], 'mean': [10], 'std': [2]}
        frame = pd.DataFrame({'synthetic_feature': [10.2, 28, 10.6, np.nan]})
        frame.attrs['observation_counts'] = [[1], [2], [1], [0]]
        test = metadata().iloc[[1]].assign(DRG_CODE='a')
        with patch('generate_report.load_fold', return_value=(args, CaseModel(), stats, None)), \
             patch('generate_report.load_inputs', return_value=(Path('synthetic'), Path('rules'), {}, None, None)), \
             patch('generate_report.read_json', return_value={}), \
             patch('generate_report.pd.read_csv', return_value=test), \
             patch('generate_report.MultimodalDrgDataset', CaseDataset), \
             patch('generate_report.load_structured_frame', return_value=frame), \
             patch('generate_report.sha256_file', return_value='synthetic-hash'):
            report = make_case_report('synthetic_fold', 'first', device='cpu')
        self.assertEqual(report['predicted_DRG'], 'b')
        self.assertEqual(report['peak_time_hours'], 1)
        self.assertEqual(report['peak_bin_index'], 1)
        self.assertEqual(report['peak_interval_hours'], [1, 2])
        self.assertAlmostEqual(report['attention_centroid_hours'], 1.2)
        self.assertAlmostEqual(report['mean_text_gate'], .7)
        self.assertEqual(report['physiology_at_peak'][0]['model_z_value'], 9)
        self.assertEqual(report['physiology_at_peak'][0]['observed_hourly_mean'], 28)
        self.assertEqual(report['physiology_at_peak'][0]['source_observation_count'], 2)
        self.assertEqual(report['top_5gram_spans'][0]['text'], 'one two three four six')
        self.assertIn('cannot be reconstructed', report['limitation'])

    def test_plot_recomputes_peak_from_arrays_in_legacy_report(self):
        from visualize_attention import main
        report = {'hour': 48, 'valid_time_mask': [True, True, True, False],
                  'temporal_attention': [.1, .6, .3, 0], 'peak_time_hours': 1.2}
        fig, ax = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory, \
             patch('visualize_attention.read_json', return_value=report), \
             patch('matplotlib.pyplot.subplots', return_value=(fig, ax)), \
             patch('matplotlib.pyplot.close'):
            main(['--report', 'synthetic.json', '--output', str(Path(directory)/'attention.png')])
        lines = ax.axvline.call_args_list
        self.assertEqual(lines[0].args[0], 1)
        self.assertIn('peak bin start', lines[0].kwargs['label'])
        self.assertAlmostEqual(lines[1].args[0], 1.2)
        self.assertIn('centroid', lines[1].kwargs['label'])


if __name__ == '__main__':
    unittest.main()
