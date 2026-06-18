#!/bin/sh

# -----------------------
# Usage: ./script.sh <LANG_CODE>
# Example: ./script.sh de-DE
# -----------------------

if [ -z "$1" ]; then
  echo "Usage: $0 <LANG_CODE>"
  exit 1
fi

LANG="$1"

export CUDA_VISIBLE_DEVICES=0 #,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

 
MODEL_ID=meta-llama/Llama-3.1-70B-Instruct #"meta-llama/Llama-3.1-8B-Instruct" #etc. other LLMs

DIM="good"
python collect_activations_qual.py \
  --model_name $MODEL_ID \
  --batch_size 1 \
  --dataset_path "contrast_data/pairs_flor_${LANG}.json" \
  --dim $DIM \
  --output_dir "activations_flor/${LANG}"

DIM="bad"
python collect_activations_qual.py \
  --model_name $MODEL_ID \
  --batch_size 1 \
  --dataset_path "contrast_data/pairs_flor_${LANG}.json" \
  --dim $DIM \
  --output_dir "activations_flor/${LANG}"

python create_steer_vector_qual.py \
  --model_name $MODEL_ID \
  --input_dir "activations_flor/${LANG}" \
  --dims "good" "bad" \
  --output_dir "vectors_flor/${LANG}"