import numpy as np
from collections import Counter
from pathlib import Path

def build_vocab(data_dir, min_freq=2, max_vocab_size=20000):
    counter = Counter()

    for txt_file in Path(data_dir).rglob("*.txt"):
        text = txt_file.read_text(encoding="utf-8").lower()
        words = text.split()
        counter.update(words)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, count in counter.most_common(max_vocab_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def text_to_ids(text, vocab, max_length=500):
    words = text.lower().split()
    ids = []

    for word in words:
        ids.append(vocab.get(word, vocab["<UNK>"]))

    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        ids = ids + [vocab["<PAD>"]] * (max_length - len(ids))
    
    return ids

def build_matrix(vocab, path, embedding_dim=100):
    matrix = np.random.normal(0, 0.1, (len(vocab), embedding_dim)).astype(np.float32)

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)

    return matrix