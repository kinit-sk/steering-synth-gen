#!/bin/bash
seeds=(1 2 3 4 5 6 7 8 9 10)
for seed in ${seeds[@]}; do
    python finetune.py --no_epochs 50 --base_model_type FacebookAI/xlm-roberta-base --batch_size 16 --batch_size_eval 512 --repeat 10 --train_csv $1 --results_dir $2 --orig_data_csv $3 --seed=$seed --val_csv $4 --max_token_len 512 --early_stopping_patience 10
done