#!/bin/bash
seeds=(11 12 13 14 15 16 17 18 19 20)
for seed in ${seeds[@]}; do
    python finetune.py --no_epochs 50 --base_model_type FacebookAI/xlm-roberta-base --batch_size 16 --batch_size_eval 512 --repeat 10 --train_csv $1 --results_dir $2 --orig_data_csv $3 --seed=$seed --val_csv $4 --max_token_len 512 --early_stopping_patience 10
done