#!/usr/bin/env bash
# Records the exact environment for the reproducibility section of the paper.
mkdir -p results
{
  echo "date: $(date -u +%FT%TZ)"
  echo "commit: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  python --version
  python - << 'PY'
import torch, torchaudio, transformers, sklearn, numpy
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("torchaudio", torchaudio.__version__, "| transformers", transformers.__version__)
print("scikit-learn", sklearn.__version__, "| numpy", numpy.__version__)
PY
} | tee results/environment.txt
pip freeze > requirements.lock.txt && echo "wrote requirements.lock.txt"
