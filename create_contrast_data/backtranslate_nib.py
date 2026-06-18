import argparse
import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import random

# -----------------------------
# CONFIGURATION
# ----------------------------- 
INTERMEDIATE_LANGS = ["deu_Latn", "zho_Hans"]  # intermediate languages for backtranslation # cym_Latn #eng_Latn
BATCH_SIZE = 64 
DEVICE = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# -----------------------------
# PIPELINE PRELOADING
# -----------------------------
def preload_pipelines(src_lang, intermediates):
    """
    Preload all translation pipelines needed for forward and backward backtranslation.
    Returns a dict {(src, tgt): pipeline}.
    """
    pipelines = {}

    # Build forward + backward directions
    path = [src_lang] + intermediates
    reverse_path = list(reversed(path))
    directions = set()

    # forward
    for a, b in zip(path[:-1], path[1:]):
        directions.add((a, b))
    # backward
    for a, b in zip(reverse_path[:-1], reverse_path[1:]):
        directions.add((a, b))
    # final translation back to source
    directions.add((intermediates[0], src_lang))

    # preload pipelines
    for src, tgt in directions:
        print(f"🔧 Loading pipeline: {MODEL_NAME}, {src} -> {tgt}")
        pipelines[(src, tgt)] = pipeline(
            task="translation",
            model=MODEL_NAME,
            src_lang=src,
            tgt_lang=tgt,
            device=DEVICE,
            dtype=torch.float16 if DEVICE >= 0 else torch.float32,
            batch_size=BATCH_SIZE
        )

    return pipelines

# -----------------------------
# TRANSLATION FUNCTION
# -----------------------------
def translate_batch(texts, pipeline_obj, token_corruption_prob=0.0):
    """Translate texts using the pipeline and optionally corrupt tokens."""
    translations = pipeline_obj(texts, max_length=512)
    translations = [t['translation_text'] for t in translations]

    # optional token corruption
    corrupted = []
    for text in translations:
        tokens = text.split()
        for i in range(len(tokens)):
            if random.random() < token_corruption_prob:
                op = random.choice(["drop", "swap", "repeat"])
                if op == "drop":
                    tokens[i] = ""
                elif op == "swap" and i < len(tokens) - 1:
                    tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
                elif op == "repeat":
                    tokens[i] = tokens[i] + " " + tokens[i]
        corrupted.append(" ".join(t for t in tokens if t))
    return corrupted

# -----------------------------
# BACKTRANSLATION PATH
# -----------------------------
def backtranslate_texts(texts, src_lang, pipelines, token_corruption_prob=0.3):
    """Perform chained backtranslation using preloaded pipelines."""
    current_texts = texts
    path = INTERMEDIATE_LANGS

    # forward through intermediates
    prev_lang = src_lang
    for lang in path:
        current_texts = translate_batch(current_texts, pipelines[(prev_lang, lang)], token_corruption_prob)
        prev_lang = lang

    # backward through intermediates (reverse)
    for lang in reversed(path[:-1]):
        current_texts = translate_batch(current_texts, pipelines[(prev_lang, lang)], token_corruption_prob)
        prev_lang = lang

    # final translation back to original source language
    current_texts = translate_batch(current_texts, pipelines[(prev_lang, src_lang)], token_corruption_prob)

    return current_texts

# -----------------------------
# DATASET BACKTRANSLATION
# -----------------------------
def backtranslate_dataset(source_csv, target_csv, src_lang, token_corruption_prob, text_column="text"):
    df = pd.read_csv(source_csv).dropna().head(180000)
    texts = df[text_column].tolist()
    gem_ids = df["gem_id"].tolist()
    labels = df["label"].tolist() if "label" in df.columns else [None] * len(texts)
    indices = df["index"].tolist() if "index" in df.columns else list(range(len(texts)))

    # preload all pipelines once
    pipelines = preload_pipelines(src_lang, INTERMEDIATE_LANGS)

    results = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Backtranslating"):
        batch = texts[i:i + BATCH_SIZE]
        bt = backtranslate_texts(batch, src_lang, pipelines, token_corruption_prob)
        results.extend(bt)

    out_df = pd.DataFrame({
        "original_text": texts,
        "label": labels,
        "index": indices,
        "backtranslated_text": results,
        "gem_id": gem_ids
    })
    out_df.to_csv(target_csv, index=False)

# -----------------------------
# ARGUMENT PARSING
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Backtranslate a dataset using NLLB pipeline.")
    parser.add_argument("--source_csv", required=True, help="Path to source CSV file (input dataset).")
    parser.add_argument("--target_csv", required=True, help="Path to target CSV file (output).")
    parser.add_argument("--src_lang", required=True, help="Source language code, e.g. 'eng_Latn', 'tha_Thai'.")
    parser.add_argument("--text_column", default="text", help="Column name containing input text.")
    parser.add_argument("--token_corruption_prob", type=float, default=0.3, help="Token corruption probability.")
    return parser.parse_args()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    backtranslate_dataset(
        source_csv=args.source_csv,
        target_csv=args.target_csv,
        src_lang=args.src_lang,
        token_corruption_prob=args.token_corruption_prob,
        text_column=args.text_column
    )
