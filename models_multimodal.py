"""Paper equations (1)-(10), comparison models, and specified ablation switches."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoConfig, AutoModel

def masked_softmax(scores, mask, dim=-1):
    mask = mask.to(device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=dim) * mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)

def masked_mean(sequence, mask):
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    return (sequence * weights).sum(1) / weights.sum(1).clamp_min(1)

class StructuredEncoder(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, args.struct_hidden_size, args.struct_num_layers,
                          batch_first=True, bidirectional=args.bidirectional,
                          dropout=args.dropout_rate if args.struct_num_layers > 1 else 0.0)
        self.output_dim = args.struct_hidden_size * (2 if args.bidirectional else 1)

    def forward(self, x, temporal_mask):
        lengths = temporal_mask.sum(1)
        expected = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(expected, temporal_mask.bool()):
            raise ValueError('Temporal mask must be a contiguous valid prefix')
        # Packing prevents even the backward GRU from processing padded values.
        clean = x * temporal_mask.unsqueeze(-1)
        packed = pack_padded_sequence(clean, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        states, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        return states * temporal_mask.unsqueeze(-1)

class GCMFusion(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_g = nn.Linear(2 * hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)

    def forward(self, D, T, temporal_mask):
        scores = torch.bmm(self.W_Q(D), self.W_K(T).transpose(1, 2)) / self.scale
        cross = masked_softmax(scores, temporal_mask.unsqueeze(1))
        S = torch.bmm(cross, self.W_V(T))
        z = torch.sigmoid(self.W_g(torch.cat([D, S], dim=-1)))
        # Exactly Eq. 6: do not add an undocumented post-fusion LayerNorm.
        return z * D + (1 - z) * S, cross, z

class DRGModel(nn.Module):
    def __init__(self, args, embedding=None, struct_input_dim=None, load_pretrained=True):
        super().__init__()
        self.args = args
        self.model_name = args.model
        self.num_classes = len(args.label_codes)
        self.has_text = args.model != 'GRU'
        self.has_struct = args.model in {'LGGCMN', 'Concat', 'GRU'}
        adjustment = args.logit_adjustment
        if adjustment is None:
            adjustment = 1.0 if args.model == 'LGGCMN' else 0.0
        self.logit_adjustment = float(adjustment)
        if not math.isfinite(self.logit_adjustment) or self.logit_adjustment < 0:
            raise ValueError('logit_adjustment must be finite and nonnegative')
        prior = torch.as_tensor(args.class_prior, dtype=torch.float32)
        if (prior.shape != (self.num_classes,) or self.num_classes < 1
                or not torch.isfinite(prior).all() or (prior <= 0).any()):
            raise ValueError('A finite positive training-partition prior is required for every class')
        self.register_buffer('class_log_prior', prior.log())
        if self.has_text:
            if args.model == 'CAML':
                if embedding is None:
                    raise ValueError('CAML initialization matrix is required')
                self.embed = nn.Embedding.from_pretrained(torch.as_tensor(embedding, dtype=torch.float32), freeze=False, padding_idx=0)
                self.conv = nn.Conv1d(embedding.shape[1], args.cnn_filter_maps, args.single_kernel_size,
                                      padding=args.single_kernel_size // 2)
                self.dropout = nn.Dropout(args.dropout_rate)
                d = args.cnn_filter_maps
            else:
                self.text_encoder = (AutoModel.from_pretrained(args.bert_model_name) if load_pretrained else
                                     AutoModel.from_config(AutoConfig.from_pretrained(args.bert_model_name)))
                model_type = self.text_encoder.config.model_type
                expected = 'longformer' if args.text_backbone == 'longformer' else 'bert'
                if model_type != expected:
                    raise ValueError(f'Expected {expected} backbone, found {model_type}')
                if args.gradient_checkpointing:
                    self.text_encoder.gradient_checkpointing_enable()
                d = self.text_encoder.config.hidden_size
            self.text_U = nn.Linear(d, self.num_classes, bias=False)
        else:
            d = args.struct_hidden_size
        if self.has_struct:
            self.struct_encoder = StructuredEncoder(args, struct_input_dim)
            self.struct_proj = nn.Linear(self.struct_encoder.output_dim, d)
        self.fusion_method = 'concat' if args.model == 'Concat' else args.fusion_method
        if self.has_struct and self.has_text:
            if self.fusion_method == 'gcm':
                self.gcm = GCMFusion(d)
            else:
                self.concat = nn.Linear(2 * d, d)
        self.fc = nn.Linear(d, self.num_classes)

    def loss(self, logits, targets, adjusted=True):
        if adjusted and self.logit_adjustment:
            logits = logits + self.logit_adjustment * self.class_log_prior
        return F.cross_entropy(logits, targets)

    def forward(self, batch, return_attn=False):
        D = text_attention = temporal_attention = z = None
        if self.has_text:
            ids, mask = batch['text'], batch['text_mask']
            if self.model_name == 'CAML':
                # Mask before convolution: padded embeddings must not enter a
                # neighboring real token's receptive field.
                embeddings = self.embed(ids.masked_fill(~mask.bool(), 0))
                embeddings = embeddings.masked_fill(~mask.bool().unsqueeze(-1), 0)
                H = torch.tanh(self.conv(self.dropout(embeddings).transpose(1, 2))).transpose(1, 2)
            else:
                inputs = {'input_ids': ids, 'attention_mask': mask.long()}
                if self.text_encoder.config.model_type == 'longformer':
                    global_mask = torch.zeros_like(ids)
                    global_mask[:, 0] = mask[:, 0]
                    inputs['global_attention_mask'] = global_mask
                H = self.text_encoder(**inputs).last_hidden_state
            if self.args.text_pooling == 'global' and self.model_name != 'CAML':
                D = masked_mean(H, mask).unsqueeze(1).expand(-1, self.num_classes, -1)
                text_attention = mask.to(H.dtype).unsqueeze(1).expand(-1, self.num_classes, -1)
                text_attention = text_attention / text_attention.sum(-1, keepdim=True).clamp_min(1)
            else:
                text_attention = masked_softmax(self.text_U(H).transpose(1, 2), mask.unsqueeze(1))
                D = torch.bmm(text_attention, H)
        if self.has_struct:
            r = batch['structured_mask']
            T = self.struct_proj(self.struct_encoder(batch['structured'], r)) * r.unsqueeze(-1)
            if self.has_text:
                if self.fusion_method == 'gcm':
                    fused, temporal_attention, z = self.gcm(D, T, r)
                else:
                    pooled = masked_mean(T, r).unsqueeze(1).expand(-1, self.num_classes, -1)
                    fused = self.concat(torch.cat([D, pooled], dim=-1))
                logits = (self.fc.weight * fused).sum(-1) + self.fc.bias
            else:
                logits = self.fc(masked_mean(T, r))
        else:
            logits = (self.fc.weight * D).sum(-1) + self.fc.bias
        if return_attn:
            return logits, {'text': text_attention, 'temporal': temporal_attention, 'gate': z}
        return logits

def pick_model(args, embedding=None, struct_input_dim=None, load_pretrained=True):
    return DRGModel(args, embedding, struct_input_dim, load_pretrained)

LGGCMNModel = DRGModel
