#!/usr/bin/env python
import os
import argparse
from functools import partial
import torch
import pandas as pd
import random
import numpy as np

from sae_lens import SAE, HookedSAETransformer
from transformer_lens.hook_points import HookPoint

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

def apply_hooks(model, layer, steer_vec, alpha=1):
    def steer_func(output, hook, steer_vec, alpha=1):
        steer_vec = steer_vec.to(output.dtype).unsqueeze(0).unsqueeze(0)
        return output + steer_vec * alpha

    hook = (f"blocks.{layer}.hook_resid_post", partial(steer_func, steer_vec=steer_vec, alpha=alpha))
    print(hook)
    model.add_hook(hook[0], hook[1])

def sample_short_texts(df: pd.DataFrame, label: int, random_state: int | None = None) -> list[str]:
    """
    Takes a DataFrame with a 'text' column and:
    1. Calculates text lengths
    2. Sorts rows from shortest to longest
    3. Keeps the 50 shortest rows
    4. Randomly samples 5 rows from those 50
    5. Returns a list of the sampled text values

    Args:
        df: Input DataFrame containing a 'text' column
        random_state: Optional seed for reproducible sampling

    Returns:
        List of 5 sampled text strings
    """

    # Create a copy to avoid modifying the original DataFrame
    temp_df = df.copy()
    temp_df = temp_df[temp_df['label'] == label]

    # Calculate text lengths
    temp_df["text_length"] = temp_df["text"].astype(str).str.len()

    # Sort by shortest texts and keep top 50
    shortest_50 = temp_df.sort_values("text_length").head(50)

    # Randomly sample 5 rows
    sampled_rows = shortest_50.sample(
        n=min(5, len(shortest_50)),
        random_state=random_state
    )

    # Return only the text column as a list
    return sampled_rows["text"].tolist()

def reset_hooks(model):
    model.reset_hooks()
    model.reset_saes()

def load_steer_vec(vectors_path, dim, layer):
    file_name = "model_resid_post_activation"
    all_svectors = torch.load(f"{vectors_path}/{file_name}_vectors_diffmean", weights_only=False)
    return torch.Tensor(all_svectors[layer][dim]).to(device)

def load_steer_vec_lang(vectors_path, language, layer):
    file_name = "language_vectors_diffmean"
    all_svectors = torch.load(f"{vectors_path}/{file_name}", weights_only=False)
    return torch.Tensor(all_svectors[layer][language]).to(device)

def main():
    parser = argparse.ArgumentParser(description="Instruction-Tuned LLM Generation Script")

    parser.add_argument("--model_name", type=str, required=True, help="Instruction-tuned LLM name")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--freq_penalty", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--labels", nargs="+", required=True, help="List of sentiment labels")
    parser.add_argument("--human_dataset", type=str, required=True, help="Human dataset for ICL")
    parser.add_argument("--language", type=str, required=True, help="Output language")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output CSV or Parquet")

    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--dim", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--use_lang_vec", type=bool, default=False)

    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    torch.set_grad_enabled(False)

    # Load model
    print(f"Loading model {args.model_name}...")
    model = HookedSAETransformer.from_pretrained(args.model_name, device=device, torch_dtype=dtype)


    vectors_path = os.path.join(args.dataset_path, args.model_name)
    if args.use_lang_vec:
        steer_vec = load_steer_vec_lang(vectors_path, args.dim, args.layer).to(dtype)
    else:
        steer_vec = load_steer_vec(vectors_path, args.dim, args.layer).to(dtype)
    if args.alpha > 0:
        apply_hooks(model, args.layer, steer_vec, alpha=args.alpha)

    # Prepare output accumulation
    generated_samples = []
    seen_texts = set()
    df_human = pd.read_csv(args.human_dataset)

    for idx, label in enumerate(args.labels):
        print(f"\n=== Generating samples for label: {label} ===")
        examples = sample_short_texts(df_human, idx, random_state=42)
        str_examples = "\n".join(examples)

        base_prompt = f"Generate a long wiki style sentence on the topic of {label} in {args.language} language based on examples. Output only the text in {args.language}. Examples: {str_examples}"
        print(base_prompt)
        instruction_prompt = f"### Instruction:\n{base_prompt}\n\n### Response:\n"
        i = 0
        base_seed = 0
        while i <= args.num_samples:
            
            input_ids = model.to_tokens(instruction_prompt).to(device)

            output_tokens = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                freq_penalty=args.freq_penalty,
                top_p=args.top_p,
                stop_at_eos=True,
            )

            text = model.tokenizer.decode(output_tokens[0]).strip()
            #print(text)

            if "### Response:" in text:
                text = text.split("### Response:")[1].strip().split("\n")[0]

            print(text)

            if text not in seen_texts:
                generated_samples.append({
                    "text": text,
                    "label": label,
                    "language": args.language
                })
                seen_texts.add(text)
                i += 1
                base_seed += 1
            else:
                print("Duplicate detected, skipping.")
                base_seed += 1 # to avoid infinite loop essentially

    reset_hooks(model)

    # Ensure directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(generated_samples)

    # Save
    if args.output_path.endswith(".csv"):
        df.to_csv(args.output_path, index=False)
    elif args.output_path.endswith(".parquet"):
        df.to_parquet(args.output_path, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")

    print(f"\nSaved {len(generated_samples)} new samples → {args.output_path}")
    print(f"Total rows in file: {len(df)}")

if __name__ == "__main__":
    main()