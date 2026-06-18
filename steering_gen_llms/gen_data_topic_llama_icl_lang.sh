#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LANG="$1"
LANG_RES="$3"
LANG_HUMAN="$6"

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" # "google/gemma-2-9b-it"

python steering_inference_it_topic_llama_icl.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors_flor_lang" \
  --dim $LANG \
  --layer $5 \
  --alpha $2 \
  --max_new_tokens 512 \
  --num_samples 20 \
  --language $4 \
  --label "science/technology" "travel" "politics" "sports" "health" "entertainment" "geography" \
  --human_data "human_data/${LANG}-sib200/${LANG_HUMAN}_train.csv" \
  --dtype 'bfloat16' \
  --use_lang_vec True \
  --output_path "outputs_flor_l$5_llama3_8b_icl_lang/${LANG}/topic/${LANG_RES}_$2/generated_good.csv"