#!/usr/bin/env python3
"""
LIME Explanation
"""

import torch
import numpy as np
import pickle

from src.model import TextCNN
from src.lime_utils import create_predict_fn


MAX_SEQ_LENGTH = 64
EMBEDDING_DIM = 128
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
DROPOUT = 0.5
CLASSES = ['World', 'Sports', 'Business', 'Sci/Tech']


def load_data():
    try:
        from datasets import load_dataset
        test_data = load_dataset('ag_news', split='test')
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
        test_texts = sample_texts[8000:]
        test_labels = sample_labels[8000:]
    return test_texts, test_labels


def main():
    print("Loading model and data...")
    
    with open('models/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        num_classes=len(CLASSES),
        filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS,
        dropout=DROPOUT
    )
    model.load_state_dict(torch.load('models/text_cnn_model.pth', weights_only=True))
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    test_texts, test_labels = load_data()
    
    predict_fn = create_predict_fn(model, vocab, MAX_SEQ_LENGTH, device)
    
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=CLASSES)
    
    print("\nLIME Explanation Results")
    print("=" * 50)
    
    sample_indices = [0, 5, 10, 15]
    for idx in sample_indices:
        text = test_texts[idx]
        true_label = test_labels[idx]
        
        proba = predict_fn([text])[0]
        pred_label = np.argmax(proba)
        
        explanation = explainer.explain_instance(text, predict_fn, num_features=10)
        
        print(f"\n--- Sample {idx} ---")
        print(f"Text: {text[:100]}...")
        print(f"True Label: {CLASSES[true_label]}")
        print(f"Predicted Label: {CLASSES[pred_label]}")
        print(f"Top Features:")
        
        for word, weight in explanation.as_list():
            direction = "+" if weight > 0 else "-"
            print(f"  {direction} {word}: {weight:.4f}")


if __name__ == '__main__':
    main()
