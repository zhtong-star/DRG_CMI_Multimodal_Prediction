"""CLI configuration. Importing this module never consumes sys.argv."""
import argparse
import math
from clinical_text import DEFAULT_NOTE_SIMILARITY_THRESHOLD

def build_parser():
    p = argparse.ArgumentParser(description='Fivefold development with a fixed patient hold-out')
    p.add_argument('--data_dir', default='data_n')
    p.add_argument('--cohort', choices=['ms', 'apr'], default='ms')
    p.add_argument('--rule_dir', default='rules')
    p.add_argument('--rule', default='13', choices=['13'])
    p.add_argument('--model', choices=['LGGCMN', 'Longformer', 'GRU', 'CAML', 'Concat'], default='LGGCMN')
    p.add_argument('--bert_model_name', default='Clinical-Longformer')
    p.add_argument('--text_backbone', choices=['longformer', 'bert'], default='longformer')
    p.add_argument('--text_pooling', choices=['label', 'global'], default='label')
    p.add_argument('--fusion_method', choices=['gcm', 'concat'], default='gcm')
    p.add_argument('--max_seq_length', type=int, default=4096)
    p.add_argument('--note_similarity_threshold', type=float, default=DEFAULT_NOTE_SIMILARITY_THRESHOLD,
                   help='Adjacent-note cosine and Levenshtein threshold; manuscript reports τ without a numeric value')
    p.add_argument('--struct_timesteps', type=int, default=48, choices=[48])
    p.add_argument('--struct_hidden_size', type=int, default=256)
    p.add_argument('--struct_num_layers', type=int, default=2)
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--dropout_rate', type=float, default=0.3)
    p.add_argument('--cnn_filter_maps', type=int, default=256)
    p.add_argument('--single_kernel_size', type=int, default=5)
    p.add_argument('--word_min_freq', type=int, default=3)
    p.add_argument('--pretrained_word_vectors', help='Local word2vec binary for CAML')
    p.add_argument('--allow_random_caml_embeddings', action='store_true')
    p.add_argument('--embedding_dim', type=int, default=200)
    p.add_argument('--text_lr', type=float, default=1e-5)
    p.add_argument('--other_lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--logit_adjustment', type=float, default=None,
                   help='Logit-adjustment strength; defaults to 1 for LGGCMN and 0 for other models. Zero explicitly disables it.')
    p.add_argument('--folds', type=int, default=5, choices=[5])
    p.add_argument('--fold', type=int, choices=range(5), help='Run one fold (0-4); default runs all five')
    p.add_argument('--random_seed', type=int, default=0)
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, or 0')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--output_dir', default='results')
    p.add_argument('--run_name', help='Run directory name; completed folds are never overwritten')
    p.add_argument('--eval_model', help='Run or fold directory to re-evaluate')
    p.add_argument('--hours', nargs='+', type=int, choices=[24, 48], default=[24, 48])
    p.add_argument('--eval_cc_mcc', action='store_true')
    p.add_argument('--verify_data_hashes', action='store_true')
    return p

def parse_args(argv=None):
    a = build_parser().parse_args(argv)
    for name in ('batch_size', 'gradient_accumulation_steps', 'epochs', 'patience',
                 'struct_hidden_size', 'struct_num_layers', 'max_seq_length'):
        if getattr(a, name) < 1:
            raise ValueError(f'{name} must be positive')
    if a.logit_adjustment is None:
        a.logit_adjustment = 1.0 if a.model == 'LGGCMN' else 0.0
    if a.logit_adjustment is not None and (not math.isfinite(a.logit_adjustment) or a.logit_adjustment < 0):
        raise ValueError('logit_adjustment must be finite and nonnegative')
    if not math.isfinite(a.note_similarity_threshold) or not 0 < a.note_similarity_threshold <= 1:
        raise ValueError('note_similarity_threshold must be finite and in (0, 1]')
    if a.text_backbone == 'bert' and a.max_seq_length > 512:
        raise ValueError('ClinicalBERT ablation requires --max_seq_length 512 (or smaller for a smoke test)')
    if a.text_backbone == 'longformer' and a.model not in {'GRU', 'CAML'} and a.max_seq_length > 4096:
        raise ValueError('Clinical-Longformer supports at most 4096 input tokens')
    if a.single_kernel_size % 2 != 1:
        raise ValueError('single_kernel_size must be odd')
    return a
