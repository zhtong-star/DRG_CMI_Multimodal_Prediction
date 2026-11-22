"""Training and inference with exact sample-weighted gradient accumulation."""
import copy
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from eval_utils import score_f1
from protocol import write_json

def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

def train_epoch(model, dataset, optimizer, batch_size, accumulation=1, num_workers=0):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if not len(loader):
        raise ValueError('Empty training partition')
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    accumulated_samples = 0
    total_loss = total_samples = 0
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        loss = model.loss(model(batch), batch['drg'], adjusted=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        n = len(batch['drg'])
        (loss * n).backward()
        accumulated_samples += n
        total_samples += n
        total_loss += float(loss.detach()) * n
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
    return total_loss / total_samples

@torch.no_grad()
def predict(model, dataset, batch_size=8, num_workers=0):
    model.eval()
    device = next(model.parameters()).device
    logits, labels, entries = [], [], []
    total_loss = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        batch = move_batch(batch, device)
        scores = model(batch)
        loss = model.loss(scores, batch['drg'], adjusted=False)
        total_loss += float(loss) * len(batch['drg'])
        logits.append(scores.cpu().numpy())
        labels.append(batch['drg'].cpu().numpy())
        entries.extend(batch['entry'])
    if not logits:
        raise ValueError('Empty inference partition')
    return np.concatenate(logits), np.concatenate(labels), entries, total_loss / len(entries)

def fit_model(model, train, val, optimizer, args, fold_dir):
    fold_dir = Path(fold_dir)
    history = []
    best_score = -float('inf')
    best_state = None
    best_epoch = -1
    for epoch in range(args.epochs):
        start = time.monotonic()
        loss = train_epoch(model, train, optimizer, args.batch_size, args.gradient_accumulation_steps, args.num_workers)
        logits, labels, _, val_loss = predict(model, val, args.batch_size, args.num_workers)
        score = score_f1(logits, labels)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, fold_dir / 'checkpoint.pt')
        history.append({'epoch': epoch + 1, 'training_loss': loss, 'validation_loss_raw_logits': val_loss,
                        'validation_macro_F1': score, 'seconds': time.monotonic() - start})
        pd.DataFrame(history).to_csv(fold_dir / 'history.csv', index=False)
        print(f'epoch={epoch + 1} train_loss={loss:.5f} val_macro_F1={score:.5f}', flush=True)
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError('Training did not produce a checkpoint')
    model.load_state_dict(best_state)
    write_json(fold_dir / 'training_summary.json', {
        'best_epoch': best_epoch + 1, 'best_validation_macro_F1': best_score,
        'epochs_completed': len(history), 'criterion': 'maximum validation macro-F1 over the fixed class space',
    })
    return history
