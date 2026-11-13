# Environmental Sound Classification on ESC-50 with Wav2Vec 2.0 (Track 3)

Compares three models on the same split (official folds: train 1-3, val 4, test 5):

| # | Model | Idea |
|---|-------|------|
| 1 | 2D-CNN on log-Mel spectrograms | baseline, no pre-training |
| 2 | Frozen Wav2Vec 2.0 + linear head | value of pre-training |
| 3 | Frozen Wav2Vec 2.0 + temporal attention head | value of attention |

## Run in Google Colab (GPU: Runtime > Change runtime type > T4)
```
!git clone https://github.com/SitahAlsaleem/esc50-wav2vec2.git
%cd esc50-wav2vec2
!bash run.sh
```
Or step by step: `check_data.py` -> `train.py` -> `cache_features.py` -> `evaluate.py`.
Demo: open `esc50_demo.ipynb` after training.

## Structure
`data/` datasets and cached features, `models/` Mel-CNN and Wav2Vec heads, `utils/` seeds,
`train.py`, `evaluate.py`, `cache_features.py`, `run.sh`, `requirements.txt`.
Seeds are fixed (42); logs go to TensorBoard (`runs/`); results to `results/`.

## Results (seed 42, test fold 5)
| Model | Test accuracy | Macro F1 |
|-------|---------------|----------|
| 1. Mel-CNN | 0.7525 | 0.741 |
| 2. Wav2Vec (L12) + linear | 0.4350 | 0.422 |
| 3. Wav2Vec (L12) + attention | 0.3825 | 0.374 |
