#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=/media/kavinder/hdd2/hf_cache
exec /media/kavinder/hdd2/RAM/gaussian-splatting/envs/gsm/bin/python "$@"
