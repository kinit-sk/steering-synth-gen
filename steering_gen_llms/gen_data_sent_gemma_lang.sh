#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LANG="$1"
LANG_RES="$3"
LANG_HUMAN="$6"

MODEL_ID="google/gemma-2-9b-it"

DIM="good"

python steering_inference_it.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors_flor/${LANG}" \
  --dim $DIM \
  --layer $5 \
  --alpha $2 \
  --max_new_tokens 512 \
  --num_samples 50 \
  --language $4 \
  --labels "positive" "negative" \
  --human_data "human_data/${LANG}-sentiment/${LANG_HUMAN}_train.csv" \
  --dtype "bfloat16" \
  --output_path "outputs_flor_l$5_9b/${LANG}/sent/${LANG_RES}_$2/generated_good.csv"