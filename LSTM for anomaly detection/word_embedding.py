# data_preparation.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(file_path):
    df = pd.read_csv(file_path)
    sequences = df["sequence"].tolist()
    labels = df["label"].tolist()
    return sequences, labels

# Fonction pour créer le vocabulaire
def build_vocab(sequences_train, sequences_test):
    all_events = set()
    for seq in sequences_train + sequences_test:
        all_events.update(seq.split())
    vocab = {event: idx+1 for idx, event in enumerate(sorted(all_events))}
    vocab_size = len(vocab) + 1  # +1 pour le padding (index 0)
    return vocab, vocab_size

# Dataset PyTorch
class LogDataset(Dataset):
    def __init__(self, sequences, labels, vocab, window_size=50):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.window_size = window_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].split()
        seq_idx = [self.vocab.get(e, 0) for e in seq]  # Mapping avec le vocab
        if len(seq_idx) < self.window_size:
            seq_idx += [0] * (self.window_size - len(seq_idx))  # Padding
        else:
            seq_idx = seq_idx[:self.window_size]  # Truncation
        return torch.tensor(seq_idx, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Charger les données
def prepare_data(train_file, test_file, window_size=50):
    sequences_train, labels_train = load_data(train_file)
    sequences_test, labels_test = load_data(test_file)
    
    vocab, vocab_size = build_vocab(sequences_train, sequences_test)
    
    train_dataset = LogDataset(sequences_train, labels_train, vocab, window_size)
    test_dataset = LogDataset(sequences_test, labels_test, vocab, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    return train_loader, test_loader, vocab_size
