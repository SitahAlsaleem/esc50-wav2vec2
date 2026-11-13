#!/usr/bin/env bash
# Full reproduction pipeline (run from the project root, in Colab with a GPU).
set -e
git clone https://github.com/karolpiczak/ESC-50.git || true
pip install -q -r requirements.txt

python check_data.py --root ESC-50

# Model 1: Mel-spectrogram 2D-CNN from scratch
python train.py --model mel_cnn --root ESC-50

# Models 2 and 3: frozen Wav2Vec 2.0 (cache features once, then train heads)
python cache_features.py --root ESC-50 --layer 12
python train.py --model w2v_linear --layer 12
python train.py --model w2v_attn --layer 12

# Evaluation on the held-out test fold (fold 5)
python evaluate.py --model mel_cnn --root ESC-50
python evaluate.py --model w2v_linear --root ESC-50 --layer 12
python evaluate.py --model w2v_attn --root ESC-50 --layer 12
