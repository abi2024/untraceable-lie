#!/bin/bash
set -e
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --extra-index-url https://pypi.org/simple
export HF_HOME=/workspace/hf-cache
export MISTRAL_2503_PATH=$(find /workspace/hf-cache/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots -maxdepth 1 -mindepth 1 -type d | head -1)
python -c "import torch,transformers,pandas,sklearn,scipy,pyarrow; print('env OK:', torch.__version__, transformers.__version__, torch.cuda.is_available())"
