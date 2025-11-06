#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate agent

cd "${AGENT_DIR}"

# Fixed dataset mount point provided by the runner
DATA_DIR="/home/data"

# Environment checks (ignore permission issues on /private for nonroot user)
ls /private >/dev/null 2>&1 || true
ls "${DATA_DIR}" >/dev/null 2>&1

# Detect GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
else
  HARDWARE="CPU"
fi
export HARDWARE

# Run the 1-shot agent
python "${AGENT_DIR}/oneshot_agent.py" \
    --data-dir "${DATA_DIR}" \
    --submission-dir "${SUBMISSION_DIR}" \
    --code-dir "${CODE_DIR}" \
    --logs-dir "${LOGS_DIR}" \
    "$@"
