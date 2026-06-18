#!/usr/bin/env python
import os
import json
import argparse
import multiprocessing as mp

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
# Main
# -----------------------------
def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
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

    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---------------------------------------------------
    # MODEL (multi-GPU via device_map)
    # ---------------------------------------------------
    print("Loading model with device_map='cuda' ...")

    model: HookedSAETransformer = HookedSAETransformer.from_pretrained_no_processing(
        args.model_name,
        device=device,          # 🔥 key for 27B
        torch_dtype=dtype,
    )

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    print(f"Model loaded: {n_layers} layers | d_model={d_model}")

    # ---------------------------------------------------
    # DATASET
    # ---------------------------------------------------
    texts = load_json_dim_dataset(args.dataset_path, args.dim, args.max_samples)
    print(f"Loaded {len(texts)} samples")

    # ---------------------------------------------------
    # ACCUMULATORS (CPU, explicit dtype)
    # ---------------------------------------------------
    model_resid_post_over_zero = torch.zeros(
        n_layers, d_model, dtype=torch.float32, device=device
    )

    model_resid_post_over_zero_binary = torch.zeros(
        n_layers, d_model, dtype=torch.int64, device=device
    )

    total_tokens = 0

    # ---------------------------------------------------
    # Hook names (no SAE dependency)
    # ---------------------------------------------------
    needed_hook_names = [
        f"blocks.{layer}.hook_resid_post"
        for layer in range(n_layers)
    ]

    def batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i + bs]

    # ---------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------
    print("Collecting resid_post activations...")

    for batch_texts in tqdm(
        list(batches(texts, args.batch_size)),
        total=(len(texts) + args.batch_size - 1) // args.batch_size,
    ):

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

        # ---------------------------------------------------
        # Accumulate per layer
        # ---------------------------------------------------
        for layer_idx in range(n_layers):

            hook_name = f"blocks.{layer_idx}.hook_resid_post"

            resid = cache[hook_name]  # (batch, seq, d_model)

            tmp_resid = resid * pad_mask_expanded.to(resid.device)

            # Move only reduced tensors to CPU
            summed = tmp_resid.to(torch.float32).sum(dim=(0, 1))
            binary = (tmp_resid > 0).to(torch.int64).sum(dim=(0, 1))

            model_resid_post_over_zero[layer_idx] += summed
            model_resid_post_over_zero_binary[layer_idx] += binary

        del cache
        torch.cuda.empty_cache()

    # ---------------------------------------------------
    # SAVE
    # ---------------------------------------------------
    out = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out, exist_ok=True)

    torch.save(
        dict(n=total_tokens, over_zero=model_resid_post_over_zero.to("cpu")),
        f"{out}/model_resid_post_activation.{args.dim}",
    )

    torch.save(
        dict(n=total_tokens, over_zero=model_resid_post_over_zero_binary.to("cpu")),
        f"{out}/model_resid_post_activation_binary.{args.dim}",
    )

    print("✅ Done.")


if __name__ == "__main__":
    main()