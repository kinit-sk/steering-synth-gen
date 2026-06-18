#!/bin/bash
# run this e.g. 'run_all_gen_and_finetuning_llama_lang_vecs_for_one_lang.sh sl sl-SL sl Slovenian 7'
# This is the main loop how we collected and preprocessed the data and then finetuned
# modify this for running gemma and zero-shot
set -e  # stop on first error (recommenend for long sweeps)
LANG="$1"
LANG_EVAL="$2"
LANG_SHORT_EVAL="$3"
LANG_EVAL_EXPL="$4"

# -----------------------------
# GRID SEARCH VALUES
# -----------------------------

if [ "$5" -eq 7 ]; then
  ALPHAS=(0.0 1.0 2.0 3.0 4.0)
else
  ALPHAS=(1.0 2.0 3.0 4.0)
fi

TOKEN_PROBS=(0.01)

for ALPHA in "${ALPHAS[@]}"; do
  for TOKEN_PROB in "${TOKEN_PROBS[@]}"; do

    echo "========================================"
    echo "Running LANG=${LANG} | ALPHA=${ALPHA} | TOKEN_PROB=${TOKEN_PROB}"
    echo "========================================"

    sh gen_data_topic_llama_icl_lang.sh "${LANG}" "${ALPHA}" "${LANG_SHORT_EVAL}" "${LANG_EVAL_EXPL}" $5 "${LANG_EVAL}"

    python preprocess_topic.py --directory_path "outputs_flor_l$5_llama3_8b_icl_lang/${LANG}/topic/${LANG_SHORT_EVAL}_${ALPHA}/"
    
    sh finetuning.sh \
      "outputs_flor_l$5_llama3_8b_icl_lang/${LANG}/topic/${LANG_SHORT_EVAL}_${ALPHA}/generated_good.csv" \
      "finetuning_results_flor_l$5_llama3_8b_icl_lang/${LANG}/topic/${LANG_SHORT_EVAL}/good_${TOKEN_PROB}_${ALPHA}/" \
      "human_data/${LANG_SHORT_EVAL}-sib200/${LANG_EVAL}_test.csv" \
      "human_data/${LANG_SHORT_EVAL}-sib200/${LANG_EVAL}_val.csv"

    sh finetuning_2.sh \
      "outputs_flor_l$5_llama3_8b_icl_lang/${LANG}/topic/${LANG_SHORT_EVAL}_${ALPHA}/generated_good.csv" \
      "finetuning_results_flor_l$5_llama3_8b_icl_lang/${LANG}/topic/${LANG_SHORT_EVAL}/good_${TOKEN_PROB}_${ALPHA}/" \
      "human_data/${LANG_SHORT_EVAL}-sib200/${LANG_EVAL}_test.csv" \
      "human_data/${LANG_SHORT_EVAL}-sib200/${LANG_EVAL}_val.csv"

    sh gen_data_sent_llama_icl_lang.sh "${LANG}" "${ALPHA}" "${LANG_SHORT_EVAL}" "${LANG_EVAL_EXPL}" $5 "${LANG_EVAL}"

    python preprocess_sentiment.py --directory_path "outputs_flor_l$5_llama3_8b_icl_lang/${LANG}/sent/${LANG_SHORT_EVAL}_${ALPHA}/"
    
    sh finetuning.sh \
      "outputs_flor_l$5_llama3_8b_icl_lang/${LANG}/sent/${LANG_SHORT_EVAL}_${ALPHA}/generated_good.csv" \
      "finetuning_results_flor_l$5_llama3_8b_icl_lang/${LANG}/sent/${LANG_SHORT_EVAL}/good_${TOKEN_PROB}_${ALPHA}/" \
      "human_data/${LANG_SHORT_EVAL}-sentiment/${LANG_EVAL}_test.csv" \
      "human_data/${LANG_SHORT_EVAL}-sentiment/${LANG_EVAL}_val.csv"

    sh finetuning_2.sh \
      "outputs_flor_l$5_llama3_8b_icl_lang/${LANG}/sent/${LANG_SHORT_EVAL}_${ALPHA}/generated_good.csv" \
      "finetuning_results_flor_l$5_llama3_8b_icl_lang/${LANG}/sent/${LANG_SHORT_EVAL}/good_${TOKEN_PROB}_${ALPHA}/" \
      "human_data/${LANG_SHORT_EVAL}-sentiment/${LANG_EVAL}_test.csv" \
      "human_data/${LANG_SHORT_EVAL}-sentiment/${LANG_EVAL}_val.csv"
  done
done
