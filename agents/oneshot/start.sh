#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate agent

# Environment checks (ignore permission issues on /private for nonroot user)
ls /private 2>/dev/null 
ls /home/data

# Detect GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
else
  HARDWARE="CPU"
fi
export HARDWARE

echo "Running 1-shot LLM agent with hardware: $HARDWARE"

# Run the 1-shot agent
python /home/agent/oneshot_agent.py \
    --data-dir /home/data \
    --submission-dir /home/submission \
    --code-dir /home/code \
    --logs-dir /home/logs \
    "$@"
