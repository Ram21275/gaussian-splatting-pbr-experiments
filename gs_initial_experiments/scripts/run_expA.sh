#!/usr/bin/env bash
# Experiment A (A0 direct / A1 diffusion / A2 rectified flow): train + evaluate.
# Usage: scripts/run_expA.sh <config.json> <split: val|test> [seed] [nfe list...]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=${PY:-../py}
CFG=$1; SPLIT=$2; SEED=${3:-0}; shift 3 || true
NFES=${*:-4 8 16}
ROOT=$($PY -c "import json;print(json.load(open('$CFG'))['output_root'])")
for V in A0 A2 A1; do
  if [ -f "$ROOT/${V}_seed$SEED/manifest.json" ]; then
    $PY -m gs_meeting.cli train --config "$CFG" --variant $V --seed $SEED --device cuda --resume
  else
    $PY -m gs_meeting.cli train --config "$CFG" --variant $V --seed $SEED --device cuda
  fi
done
$PY -m gs_meeting.cli evaluate --config "$CFG" --checkpoint "$ROOT/A0_seed$SEED/checkpoint.pt" --split $SPLIT --nfe 1 --samples 8 --lpips --overwrite
for V in A2 A1; do for N in $NFES; do
  $PY -m gs_meeting.cli evaluate --config "$CFG" --checkpoint "$ROOT/${V}_seed$SEED/checkpoint.pt" --split $SPLIT --nfe $N --samples 8 --lpips --overwrite
done; done
$PY -m gs_meeting.cli summarize --root "$ROOT" --split $SPLIT
