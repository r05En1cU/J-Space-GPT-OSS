#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"
OUT="${OUT:-gpt_oss_20b_jspace_dictionary.pt}"
LAYERS="${LAYERS:-0,4,8,12,16,20}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

python jspace_gpt_oss.py inspect-model \
  --model-id "$MODEL_ID" \
  --torch-dtype "$DTYPE" \
  --device-map "$DEVICE_MAP"

python jspace_gpt_oss.py build-dictionary \
  --model-id "$MODEL_ID" \
  --prompts-file calibration_prompts.txt \
  --candidates-file candidate_concepts.txt \
  --layers "$LAYERS" \
  --max-prompts 8 \
  --max-length 128 \
  --max-pairs 1 \
  --position-mode last \
  --torch-dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  --out "$OUT"

python jspace_gpt_oss.py readout \
  --dictionary "$OUT" \
  --model-id "$MODEL_ID" \
  --prompt "A spider builds a" \
  --layer 12 \
  --position -1 \
  --top-k 20 \
  --torch-dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  > readout_spider_layer12.json

python jspace_gpt_oss.py decompose \
  --dictionary "$OUT" \
  --model-id "$MODEL_ID" \
  --prompt "A spider builds a" \
  --layer 12 \
  --position -1 \
  --k 25 \
  --torch-dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  > decompose_spider_layer12.json

python jspace_gpt_oss.py intervene \
  --dictionary "$OUT" \
  --model-id "$MODEL_ID" \
  --prompt "The animal crawled across the" \
  --layer 12 \
  --position -1 \
  --mode steer \
  --token " spider" \
  --alpha 2.0 \
  --steps 32 \
  --torch-dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  > steer_spider_layer12.json

printf 'Wrote %s, readout_spider_layer12.json, decompose_spider_layer12.json, steer_spider_layer12.json\n' "$OUT"
