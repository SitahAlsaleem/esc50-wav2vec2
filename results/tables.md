## Main + ablation results (mean ± std over seeds)

| Configuration | Params | Seeds | Test acc (%) | Macro-F1 (%) | Val acc (%) | Train acc (%) |
|---|---|---|---|---|---|---|
| Mel-CNN | 1,185,492 | 5 | 75.7 ± 1.6 | 74.1 ± 1.8 | 81.9 ± 1.3 | 99.7 ± 0.3 |
| W2V linear (layer 12) | 39,986 | 5 | 43.3 ± 0.9 | 42.0 ± 0.9 | 48.5 ± 0.6 | 84.1 ± 3.0 |
| W2V uniform-weight (layer 12) | 39,986 | 5 | 43.5 ± 0.8 | 41.9 ± 0.8 | 48.9 ± 0.4 | 87.3 ± 2.5 |
| W2V attention (layer 12) | 237,107 | 5 | 38.6 ± 0.6 | 37.8 ± 0.6 | 48.3 ± 1.8 | 89.5 ± 4.0 |
| Mel-CNN (no augmentation) | 1,185,492 | 5 | 75.2 ± 2.3 | 73.5 ± 2.5 | 81.9 ± 2.1 | 99.8 ± 0.5 |

## Tuning log (seed 42; choose by validation accuracy)

| Model | LR | Dropout | Val acc (%) | Best epoch | Test acc (%) |
|---|---|---|---|---|---|
| w2v_attn | 0.001 | 0.5 | 49.2 | 26 | 39.0 |
| w2v_attn | 0.001 | 0.3 | 46.2 | 15 | 38.2 |
| w2v_attn | 0.0003 | 0.5 | 45.2 | 48 | 38.2 |
| w2v_attn | 0.0003 | 0.3 | 43.8 | 42 | 37.0 |
| w2v_attn | 0.0001 | 0.3 | 40.0 | 39 | 34.0 |
| w2v_attn | 0.0001 | 0.5 | 39.2 | 46 | 36.0 |
| w2v_linear | 0.001 | 0.5 | 48.8 | 35 | 44.2 |
| w2v_linear | 0.001 | 0.3 | 48.5 | 22 | 43.5 |
| w2v_linear | 0.0003 | 0.3 | 45.5 | 54 | 43.5 |
| w2v_linear | 0.0003 | 0.5 | 44.0 | 40 | 42.0 |
| w2v_linear | 0.0001 | 0.3 | 41.2 | 50 | 35.0 |
| w2v_linear | 0.0001 | 0.5 | 39.2 | 52 | 34.8 |
