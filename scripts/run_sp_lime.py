#!/usr/bin/env python3
"""
Submodular Pick with LIME
"""

import argparse
import torch
import numpy as np
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TextCNN
from src.lime_utils import create_predict_fn
from submodular_pick import SubmodularPick


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
    parser = argparse.ArgumentParser(description='Submodular Pick with LIME')
    parser.add_argument('--num-samples', type=int, default=2000, help='Number of samples to process')
    parser.add_argument('--num-exps', type=int, default=12, help='Number of explanations to select')
    parser.add_argument('--num-features', type=int, default=10, help='Number of features per explanation')
    args = parser.parse_args()
    
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
    
    print("\n" + "=" * 50)
    print("Submodular Pick - SP-LIME")
    print(f"Samples: {args.num_samples}, Select: {args.num_exps}")
    print("=" * 50)
    
    sample_data = test_texts[:args.num_samples]
    sample_labels = test_labels[:args.num_samples]
    
    print("Generating explanations...")
    sp = SubmodularPick(
        explainer=explainer,
        data=np.array(sample_data),
        predict_fn=predict_fn,
        method='full',
        num_exps_desired=args.num_exps,
        num_features=args.num_features
    )
    
    print(f"\nSelected {len(sp.sp_explanations)} representative samples:")
    print("-" * 50)
    
    for i, exp in enumerate(sp.sp_explanations):
        orig_idx = sp.V[i]
        text = sample_data[orig_idx]
        true_label = sample_labels[orig_idx]
        proba = predict_fn([text])[0]
        pred_label = np.argmax(proba)
        
        exp_list = exp.as_list(label=pred_label)
        
        status = "OK" if pred_label == true_label else "NG"
        
        print(f"\n[Selected {i+1}] Index: {orig_idx}")
        print(f"Text: {text[:80]}...")
        print(f"Predicted: {CLASSES[pred_label]}")
        print(f"True: {CLASSES[true_label]} [{status}]")
        print(f"Features:")
        
        for word, weight in exp_list[:5]:
            direction = "+" if weight > 0 else "-"
            print(f"  {direction} {word}: {weight:.4f}")


if __name__ == '__main__':
    main()
