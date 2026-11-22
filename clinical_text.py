"""One clinical-note policy shared by every text model."""
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from protocol import SCHEMA_VERSION

PLACEHOLDER = re.compile(r'\[\*\*.*?\*\*\]', re.DOTALL)
DEFAULT_NOTE_SIMILARITY_THRESHOLD = 0.95
NOTE_CLEANING_POLICY = {
    'paragraphs': 'blank-line boundaries preserved until adjacent-paragraph cleaning',
    'cosine': 'full-paragraph clinical-token frequency vectors',
    'levenshtein': 'full-paragraph token edit distance / maximum original token count',
    'remove_when': 'both similarities strictly greater than note_similarity_threshold',
    'adjacency': 'compare with the last retained paragraph in chronological note order',
    'threshold_provenance': 'implementation setting; manuscript P037 does not specify numeric tau',
    'clinical_content': 'retain punctuation and numbers; remove deidentification placeholders',
}


def validate_note_policy(audit, expected_threshold=None):
    """Reject incompatible or unspecified prepared-note provenance."""
    if audit.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Prepared notes require schema v{SCHEMA_VERSION}; regenerate the cohort')
    if 'note_similarity_threshold' not in audit:
        raise ValueError('Prepared notes lack explicit note_similarity_threshold provenance')
    threshold = float(audit['note_similarity_threshold'])
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('note_similarity_threshold must be finite and in (0, 1]')
    if audit.get('note_cleaning_policy') != NOTE_CLEANING_POLICY:
        raise ValueError('Prepared note_cleaning_policy differs; regenerate the cohort')
    if expected_threshold is not None and float(expected_threshold) != threshold:
        raise ValueError('Prepared note-cleaning threshold differs from the requested threshold')
    return threshold


def clean_text(text):
    return re.sub(r'\s+', ' ', PLACEHOLDER.sub(' ', str(text))).strip()


def _similarity_tokens(text):
    """Preserve decimals and clinical punctuation in similarity comparisons."""
    return re.findall(r"\d+(?:\.\d+)*|[^\W\d_]+(?:['’-][^\W\d_]+)*|[^\w\s]|_",
                      clean_text(text).lower())


def cosine_similarity(text_a, text_b):
    """Cosine similarity of normalized word-frequency vectors."""
    left, right = Counter(_similarity_tokens(text_a)), Counter(_similarity_tokens(text_b))
    if not left or not right:
        return 1.0 if not left and not right else 0.0
    common = set(left).intersection(right)
    dot = sum(left[token] * right[token] for token in common)
    norm_left = sum(value * value for value in left.values()) ** 0.5
    norm_right = sum(value * value for value in right.values()) ** 0.5
    return float(dot / (norm_left * norm_right)) if norm_left and norm_right else 0.0


def levenshtein_similarity(text_a, text_b):
    """Exact normalized token edit similarity over both complete paragraphs.

    Removing an equal prefix/suffix reduces work without changing edit distance;
    normalization still uses the full original lengths. No prefix cap is used.
    """
    left, right = _similarity_tokens(text_a), _similarity_tokens(text_b)
    denominator = max(len(left), len(right))
    if left == right:
        return 1.0
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    left, right = left[prefix:], right[prefix:]
    suffix = 0
    while suffix < min(len(left), len(right)) and left[-suffix-1] == right[-suffix-1]:
        suffix += 1
    if suffix:
        left, right = left[:-suffix], right[:-suffix]
    if len(right) > len(left):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, token_left in enumerate(left, 1):
        current = [i]
        for j, token_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (token_left != token_right)))
        previous = current
    return float(1.0 - previous[-1] / denominator)


def _adjacent_paragraphs(text):
    raw = str(text).replace('\r\n', '\n').replace('\r', '\n')
    raw = PLACEHOLDER.sub(' ', raw)
    return [clean_text(part) for part in re.split(r'\n\s*\n+', raw) if clean_text(part)]


def clean_note_paragraphs(text):
    """Clean whitespace within paragraphs while preserving their boundaries."""
    return '\n\n'.join(_adjacent_paragraphs(text))


def deduplicate_adjacent_paragraphs(text, threshold=DEFAULT_NOTE_SIMILARITY_THRESHOLD):
    """Remove adjacent template paragraphs using cosine and Levenshtein checks.

    The manuscript specifies a threshold τ but does not provide its numeric
    value. The default is therefore an explicit implementation setting and is
    recorded in the cohort/run provenance; historical results cannot be
    inferred from this default alone.
    """
    if threshold is None:
        return clean_text(text)
    threshold = float(threshold)
    if not 0 < threshold <= 1:
        raise ValueError('note similarity threshold must be in (0, 1]')
    kept = []
    for paragraph in _adjacent_paragraphs(text):
        if kept:
            cosine = cosine_similarity(kept[-1], paragraph)
            if cosine <= threshold:
                kept.append(paragraph)
                continue
            levenshtein = levenshtein_similarity(kept[-1], paragraph)
            if levenshtein > threshold:
                continue
        kept.append(paragraph)
    return '\n\n'.join(kept)


def select_notes(notes, hour, admission_time=None):
    required = {'DIFFTIME', 'CATEGORY', 'TEXT'}
    if not required.issubset(notes.columns):
        raise ValueError('Prepared notes must have DIFFTIME, CATEGORY and TEXT')
    if hour not in (24, 48):
        raise ValueError('Supported note horizons are 24 and 48 hours')
    category = notes.CATEGORY.fillna('').str.strip().str.lower()
    relative = pd.to_numeric(notes.DIFFTIME, errors='coerce')
    mask = relative.le(hour) & relative.notna() & relative.abs().ne(float('inf'))
    mask &= ~category.isin(['discharge summary', 'admission_diag'])
    if admission_time is not None:
        if 'CHARTTIME' not in notes:
            raise ValueError('CHARTTIME is required to enforce the hospital-admission lower bound')
        start = pd.Timestamp(admission_time)
        if pd.isna(start):
            raise ValueError('Hospital-admission time is missing')
        mask &= pd.to_datetime(notes.CHARTTIME, errors='coerce', format='mixed').ge(start)
    selected = notes.loc[mask].copy()
    # Relative times share one ICU origin and also order mixed timestamp strings correctly.
    selected['DIFFTIME'] = relative.loc[mask]
    return selected.sort_values('DIFFTIME', kind='stable')


def read_note_text(path, hour, admission_time=None,
                   similarity_threshold=DEFAULT_NOTE_SIMILARITY_THRESHOLD, *, require_schema=False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing text input: {path}')
    notes = pd.read_pickle(path)
    if require_schema and (notes.attrs.get('schema_version') != SCHEMA_VERSION or
                           notes.attrs.get('paragraph_boundaries_preserved') is not True):
        raise ValueError(f'Prepared notes require schema v{SCHEMA_VERSION} with preserved paragraphs; regenerate the cohort')
    selected = select_notes(notes, hour, admission_time)
    blocks = [clean_note_paragraphs(t) for t in selected.TEXT.dropna() if clean_text(t)]
    return clean_text(deduplicate_adjacent_paragraphs('\n\n'.join(blocks), similarity_threshold))


def word_tokens(text):
    """Deterministic CAML word tokenizer; fitted vocabulary uses training notes only."""
    return re.findall(r"\b\w+\b", clean_text(text).lower())
