#!/usr/bin/env python
import os
import json
import argparse
import torch
from tqdm import tqdm

from sae_lens import HookedSAETransformer


# -----------------------------
# Dataset loader
# -----------------------------
def load_json_dim_dataset(path: str, dim: str, n: int | None = None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = data[dim]
    return texts[:n] if n is not None else texts


# -----------------------------
# Batch helper
# -----------------------------
def batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]


# -----------------------------
# Compute mean activations per language
# -----------------------------
def compute_language_mean(model, texts, batch_size, n_layers, d_model):

    device = next(model.parameters()).device

    accum = torch.zeros(n_layers, d_model, dtype=torch.float32, device=device)
    total_tokens = 0

    needed_hook_names = [
        f"blocks.{layer}.hook_resid_post"
        for layer in range(n_layers)
    ]

    for batch_texts in tqdm(list(batches(texts, batch_size))):

        toks = model.to_tokens(batch_texts, prepend_bos=False)

        pad_mask = (toks != model.tokenizer.pad_token_id).long()
        pad_mask_expanded = pad_mask.unsqueeze(-1)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks,
                stop_at_layer=n_layers,
                names_filter=lambda name: name in needed_hook_names,
            )

        total_tokens += int(pad_mask.sum())

        for layer_idx in range(n_layers):
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            resid = cache[hook_name]

            masked = resid * pad_mask_expanded.to(resid.device)
            summed = masked.sum(dim=(0, 1))

            accum[layer_idx] += summed

        del cache
        torch.cuda.empty_cache()

    mean = accum / max(total_tokens, 1)
    return mean.cpu()


# -----------------------------
# Compute language steering vectors
# -----------------------------
def compute_language_steering(lang_means, output_path, model_name, file_name="language"):

    langs = list(lang_means.keys())

    # Stack: [num_langs, n_layers, d_model]
    V = torch.stack([lang_means[l] for l in langs])
    num_langs = V.shape[0]

    # Global mean
    global_mean = V.mean(dim=0)

    # Compute diff vectors
    diff_vectors = {}

    for i, lang in enumerate(langs):
        v_L = V[i]

        others_mean = (global_mean * num_langs - v_L) / (num_langs - 1)
        diff = v_L - others_mean

        # Normalize per layer
        norms = torch.norm(diff, dim=1, keepdim=True)
        diff_normed = diff / norms.clamp(min=1e-12)

        diff_vectors[lang] = diff_normed

    # Convert to your format
    n_layers = V.shape[1]
    all_svectors = []

    for layer_index in range(n_layers):
        print("layer:", layer_index)

        svectors = {}

        for lang in langs:
            vec = diff_vectors[lang][layer_index]
            svectors[lang] = vec.cpu().numpy()

        all_svectors.append(svectors)

    # Save (same format as your previous pipeline)
    out_path = os.path.join(output_path, model_name)
    os.makedirs(out_path, exist_ok=True)

    torch.save(
        all_svectors,
        f"{out_path}/{file_name}_vectors_diffmean"
    )

    print("✅ Language steering vectors saved.")


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, required=True)  # "sk,cs,de"
    parser.add_argument("--dim", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load model
    # -----------------------------
    print("Loading model...")

    model: HookedSAETransformer = HookedSAETransformer.from_pretrained_no_processing(
        args.model_name,
        device=device,
        torch_dtype=dtype,
    )

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    print(f"Model: {n_layers} layers | d_model={d_model}")

    # -----------------------------
    # Parse languages
    # -----------------------------
    languages = [l.strip() for l in args.languages.split(",")]
    print(f"Languages: {languages}")

    # -----------------------------
    # Compute means
    # -----------------------------
    lang_means = {}

    for lang in languages:
        print(f"\nProcessing language: {lang}")

        path = os.path.join(args.data_dir, f"pairs_flor_{lang}.json")
        texts = load_json_dim_dataset(path, args.dim, args.max_samples)

        print(f"Loaded {len(texts)} samples")

        mean_vec = compute_language_mean(
            model,
            texts,
            args.batch_size,
            n_layers,
            d_model
        )

        lang_means[lang] = mean_vec

    # Optional: save raw means
    out_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    torch.save(
        lang_means,
        os.path.join(out_dir, f"language_means.{args.dim}.pt")
    )

    # -----------------------------
    # Compute steering vectors
    # -----------------------------
    compute_language_steering(
        lang_means,
        args.output_dir,
        args.model_name,
        file_name="language"
    )


if __name__ == "__main__":
    main()