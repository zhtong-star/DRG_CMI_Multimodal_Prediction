"""Training-fold vocabulary and optional pretrained CAML embeddings."""
from collections import Counter
from pathlib import Path
import numpy as np
from clinical_text import read_note_text, word_tokens

def build_caml_inputs(args, train_df):
    counter = Counter()
    base = Path(args.data_dir) / args.cohort / 'text_raw'
    for row in train_df.itertuples():
        counter.update(word_tokens(read_note_text(base / f'{row.stay}.pk', 48, row.ADMITTIME,
                                                  args.note_similarity_threshold)))
    vocab = {'<pad>': 0, '<unk>': 1}
    for word in sorted(w for w, n in counter.items() if n >= args.word_min_freq):
        if word not in vocab:
            vocab[word] = len(vocab)
    rng = np.random.default_rng(args.random_seed)
    if args.pretrained_word_vectors:
        from gensim.models import KeyedVectors
        vectors = KeyedVectors.load_word2vec_format(args.pretrained_word_vectors, binary=True)
        dim = vectors.vector_size
    else:
        if not args.allow_random_caml_embeddings:
            raise ValueError('CAML needs --pretrained_word_vectors or explicit --allow_random_caml_embeddings')
        vectors, dim = None, args.embedding_dim
    embedding = rng.normal(0, 0.02, (len(vocab), dim)).astype(np.float32)
    embedding[0] = 0
    if vectors is not None:
        for word, i in vocab.items():
            if i > 1 and word in vectors:
                embedding[i] = vectors[word]
    return vocab, embedding
