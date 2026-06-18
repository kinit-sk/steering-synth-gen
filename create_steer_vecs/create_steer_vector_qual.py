import torch
import argparse
import os

parser = argparse.ArgumentParser(
    description="Create steering vectors from resid_post activations"
)

parser.add_argument(
    "--model_name",
    type=str,
    required=True,
)

parser.add_argument(
    "--dims",
    nargs='+',
    required=True,
    help="Dimension keys (e.g. good bad)",
)

parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Directory where activation stats are stored",
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
)

args = parser.parse_args()


# -----------------------------
# Mappings
# -----------------------------
id2dim = {i: dim for i, dim in enumerate(args.dims)}
dim2id = {dim: i for i, dim in enumerate(args.dims)}

# -----------------------------
# Input path (UPDATED)
# -----------------------------
input_path = os.path.join(args.input_dir, args.model_name)


# -----------------------------
# Load data
# -----------------------------
def get_data(file_name="model_resid_post_activation"):
    n, over_zero = [], []

    for dim in args.dims:
        data = torch.load(f"{input_path}/{file_name}.{dim}")

        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n, dtype=torch.float32)

    over_zero = torch.stack(over_zero, dim=-1)

    over_zero = torch.nan_to_num(
        over_zero,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    num_layers, d_model, dim_num = over_zero.size()
    print(f"Loaded: layers={num_layers}, d_model={d_model}, dims={dim_num}")

    return n, over_zero


# -----------------------------
# Steering vector computation
# -----------------------------
def activation(file_name="model_resid_post_activation"):

    n, over_zero = get_data(file_name)

    # Mean activation per dimension
    activation_probs = over_zero / n

    # Mean of all OTHER dimensions
    n_sum = n.sum() - n
    dim_sum = over_zero.sum(dim=2, keepdim=True) - over_zero

    activation_sum_probs = dim_sum / n_sum

    # Difference
    diff_mean = activation_probs - activation_sum_probs

    # Normalize per layer
    norms = torch.norm(diff_mean, dim=1, keepdim=True)
    diff_mean_normalized = diff_mean / norms.clamp(min=1e-12)

    # -----------------------------
    # Convert to steering vectors
    # -----------------------------
    all_svectors = []

    for layer_index in range(diff_mean_normalized.shape[0]):
        print("layer:", layer_index)

        svectors = diff_mean_normalized[layer_index].T  # [dims, d_model]
        svectors = svectors.cpu().numpy()

        svectors = {
            id2dim[i]: svectors[i]
            for i in range(svectors.shape[0])
        }

        all_svectors.append(svectors)

    # -----------------------------
    # Save
    # -----------------------------
    out_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out_path, exist_ok=True)

    torch.save(
        all_svectors,
        f"{out_path}/{file_name}_vectors_diffmean"
    )


# -----------------------------
# Run
# -----------------------------
activation("model_resid_post_activation")