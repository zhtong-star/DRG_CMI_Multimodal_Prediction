"""Check installed dependencies and optional private resources without training."""
import argparse
import importlib
import importlib.metadata
from pathlib import Path
from clinical_text import DEFAULT_NOTE_SIMILARITY_THRESHOLD


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_dir', help='Optional local transformer resource directory')
    p.add_argument('--data_dir', help='Optional finalized dataset root')
    p.add_argument('--rule_dir', default='rules')
    p.add_argument('--cohort', choices=['ms', 'apr'], default='ms')
    p.add_argument('--require_cuda', action='store_true')
    p.add_argument('--verify_data_hashes', action='store_true')
    p.add_argument('--note_similarity_threshold', type=float, default=DEFAULT_NOTE_SIMILARITY_THRESHOLD)
    a = p.parse_args(argv)
    for package, module in [('torch','torch'), ('transformers','transformers'), ('tokenizers','tokenizers'),
                            ('numpy','numpy'), ('pandas','pandas'), ('scikit-learn','sklearn'), ('scipy','scipy')]:
        importlib.import_module(module)
        print(package, importlib.metadata.version(package))
    import torch
    print('CUDA available:', torch.cuda.is_available())
    if a.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this check but is unavailable')
    if a.model_dir:
        from transformers import AutoConfig, AutoTokenizer
        root = Path(a.model_dir)
        config = AutoConfig.from_pretrained(root, local_files_only=True)
        AutoTokenizer.from_pretrained(root, local_files_only=True, use_fast=True)
        if not (list(root.glob('*.safetensors')) or list(root.glob('pytorch_model*.bin'))):
            raise FileNotFoundError('No transformer weights found; tokenizer/config alone cannot initialize training')
        print('Model config/tokenizer/weight presence:', config.model_type, '(weight tensors not loaded by this check)')
    if a.data_dir:
        from main_multimodal import load_inputs
        _, _, audit, _, _ = load_inputs(a)
        print('Finalized dataset:', audit['complete_admissions'], 'admissions;', audit['classes'], 'classes')
    print('Setup checks passed.')

if __name__ == '__main__':
    main()
