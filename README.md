# TextCNN + LIME + Submodular Pick

AG News Classification with 1D CNN and LIME Explanation with Submodular Pick (SP-LIME).

## Project Structure

```
text_project2/
├── README.md                # This file
├── requirements.txt         # Dependencies
├── submodular_pick.py      # SP-LIME algorithm implementation
├── src/
│   ├── __init__.py
│   ├── model.py           # TextCNN model
│   ├── preprocess.py       # Data preprocessing
│   └── lime_utils.py      # LIME prediction function
├── models/
│   ├── text_cnn_model.pth # Trained model weights
│   └── vocab.pkl          # Vocabulary
└── scripts/
    ├── train_model.py     # Train the model
    ├── run_lime.py        # LIME explanation
    └── run_sp_lime.py    # Submodular Pick
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model (Optional - model is already provided)

```bash
python scripts/train_model.py
```

### 2. LIME Explanation

```bash
python scripts/run_lime.py
```

### 3. Submodular Pick

```bash
# Default: 2000 samples, select 12 representative samples
python scripts/run_sp_lime.py

# Custom parameters
python scripts/run_sp_lime.py --num-samples 500 --num-exps 5 --num-features 10
```

## Arguments

### run_sp_lime.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-samples` | 2000 | Number of samples to process |
| `--num-exps` | 12 | Number of representative samples to select |
| `--num-features` | 10 | Number of features per explanation |

## Model Configuration

- **Vocabulary Size**: 30,000
- **Max Sequence Length**: 64
- **Embedding Dimension**: 128
- **Filter Sizes**: [3, 4, 5]
- **Number of Filters**: 100
- **Dropout**: 0.5

## Classes

- World
- Sports
- Business
- Sci/Tech

## Device Support

Automatically detects: CUDA (GPU) → MPS (Apple Silicon) → CPU

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- LIME
- scikit-learn
- datasets (HuggingFace)
