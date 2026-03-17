#!/usr/bin/env python3
"""
Train TextCNN model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os

from src.model import TextCNN
from src.preprocess import build_vocab, TextDataset


BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LENGTH = 64
EMBEDDING_DIM = 128
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
DROPOUT = 0.5
CLASSES = ['World', 'Sports', 'Business', 'Sci/Tech']


def load_data():
    print("Loading dataset...")
    try:
        from datasets import load_dataset
        train_data = load_dataset('ag_news', split='train')
        test_data = load_dataset('ag_news', split='test')
        
        train_texts = [item['text'] for item in train_data][:15000]
        train_labels = [item['label'] for item in train_data][:15000]
        test_texts = [item['text'] for item in test_data][:2000]
        test_labels = [item['label'] for item in test_data][:2000]
    except:
        sample_texts = [
            "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling breed of risky asset investors, are seeing their fortunes revive after a nearly decade-long ban on betting against a company.",
            "NBA Players Help Kick Off Finals Week (AP) AP - The NBA Finals are getting a different kind of assist from some well-known players.",
            "Stocks End Mixed in Quiet Trading (Reuters) Reuters - Stocks finished mixed in quiet trading Tuesday, with few major economic reports on the calendar.",
            "Space Station Malfunction Reveals Flaw (AP) AP - NASA said Wednesday it has corrected a problem with the international space station's cooling system.",
        ] * 2500
        sample_labels = [0, 1, 2, 3] * 2500
        train_texts = sample_texts[:8000]
        train_labels = sample_labels[:8000]
        test_texts = sample_texts[8000:]
        test_labels = sample_labels[8000:]
    
    return train_texts, train_labels, test_texts, test_labels


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for indices, labels in loader:
        indices = indices.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(indices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for indices, labels in loader:
            indices = indices.to(device)
            labels = labels.to(device)
            outputs = model(indices)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def main():
    train_texts, train_labels, test_texts, test_labels = load_data()
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, MAX_VOCAB_SIZE)
    print(f"Vocabulary size: {len(vocab)}")
    
    train_dataset = TextDataset(train_texts, train_labels, vocab, MAX_SEQ_LENGTH)
    test_dataset = TextDataset(test_texts, test_labels, vocab, MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        num_classes=len(CLASSES),
        filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS,
        dropout=DROPOUT
    )
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/text_cnn_model.pth')
    with open('models/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Model saved to models/")


if __name__ == '__main__':
    main()
