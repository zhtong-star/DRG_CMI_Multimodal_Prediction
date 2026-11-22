"""Cross-platform comparison/ablation runner; every selected model runs five folds."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohorts', nargs='+', choices=['ms','apr'], default=['ms','apr'])
    p.add_argument('--models', nargs='+', choices=['LGGCMN','GRU','Longformer','CAML','Concat'],
                   default=['LGGCMN','GRU','Longformer','CAML','Concat'])
    p.add_argument('--ablations', action='store_true')
    p.add_argument('--ablation_cohorts', nargs='+', choices=['ms', 'apr'], default=['ms'],
                   help='Table 3 uses MS-DRG only; explicitly select apr for additional experiments')
    p.add_argument('--logit_adjustment', type=float, default=1.0,
                   help='LGGCMN lambda; the revised manuscript specifies 1. Zero explicitly disables adjustment.')
    p.add_argument('--gradient_accumulation_steps', type=int, required=True, help='Explicit choice; historical value unreported')
    p.add_argument('--clinicalbert_dir', help='Local ClinicalBERT resources for ablation')
    p.add_argument('--pretrained_word_vectors', help='Local binary word2vec for CAML')
    p.add_argument('--run_prefix', required=True)
    p.add_argument('--dry_run', action='store_true')
    a, common = p.parse_known_args(argv)
    forbidden = ('--model','--cohort','--run_name','--fold','--eval_model','--text_backbone', '--fusion_method', '--text_pooling')
    if any(x.split('=')[0] in forbidden for x in common):
        p.error('Model/fold/run/ablation flags are controlled by this runner')
    if a.ablations and not a.clinicalbert_dir:
        p.error('--ablations requires --clinicalbert_dir')
    if 'CAML' in a.models and not a.pretrained_word_vectors:
        p.error('CAML requires --pretrained_word_vectors (use the main CLI for explicit random smoke initialization)')
    if a.gradient_accumulation_steps < 1:
        p.error('--gradient_accumulation_steps must be positive')
    variants = [(m,m,[]) for m in a.models]
    ablations = []
    if a.ablations:
        ablations = [('LGGCMN','no_label_attention',['--text_pooling','global']),
                     ('LGGCMN','no_gcm',['--fusion_method','concat']),
                     ('LGGCMN','clinicalbert',['--text_backbone','bert','--bert_model_name',a.clinicalbert_dir,'--max_seq_length','512'])]
    for cohort in a.cohorts:
        selected = variants + (ablations if cohort in a.ablation_cohorts else [])
        for model, suffix, extra in selected:
            cmd = [sys.executable, str(Path(__file__).with_name('main_multimodal.py')), *common,
                   '--cohort',cohort,'--model',model,'--run_name',f'{a.run_prefix}_{cohort}_{suffix}',
                   '--logit_adjustment',str(a.logit_adjustment if model == 'LGGCMN' else 0),
                   '--gradient_accumulation_steps',str(a.gradient_accumulation_steps), *extra]
            if model == 'CAML':
                cmd += ['--pretrained_word_vectors',a.pretrained_word_vectors]
            print(subprocess.list2cmdline(cmd) if os.name == 'nt' else shlex.join(cmd), flush=True)
            if not a.dry_run:
                subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
