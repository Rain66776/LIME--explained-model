import re
from collections import Counter
from torch.utils.data import Dataset


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    return tokens


def build_vocab(texts, max_vocab_size):
    word_counts = Counter()
    for text in texts:
        tokens = tokenize(text)
        word_counts.update(tokens)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def text_to_indices(text, vocab, max_len):
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = text_to_indices(text, self.vocab, self.max_len)
        return indices, label
