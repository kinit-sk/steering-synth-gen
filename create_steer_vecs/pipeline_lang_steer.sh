#!/bin/sh

MODEL_ID="meta-llama/Llama-3.1-70B-Instruct"

python collect_lang_activations_and_create_steer_vecs.py \
  --model_name $MODEL_ID \
  --batch_size 1 \
  --data_dir "contrast_data" \
  --languages "am,de,da,cs,he,id,sk,sl,en,jv,su,mt" \
  --dim "good" \
  --output_dir "vectors_flor_lang"