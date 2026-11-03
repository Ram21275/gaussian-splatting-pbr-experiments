#!/usr/bin/env bash
# Experiment 0 (v2 data) feature extraction. train then test on the Ada (GPU0, waits for >=10GB free); val on the A400 (GPU1).
cd /home/kavinder/RAM/gaussian-splatting
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( CUDA_VISIBLE_DEVICES=1 ./py -m gsm.exp0_extract --split val --bs 2 > outputs/exp0/logs/extract_val.log 2>&1 ) &
( for s in train test; do
    until [ $(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits) -gt 10000 ]; do sleep 30; done
    CUDA_VISIBLE_DEVICES=0 ./py -m gsm.exp0_extract --split $s --bs 6 > outputs/exp0/logs/extract_$s.log 2>&1
  done ) &
wait
