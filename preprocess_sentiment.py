import os
import argparse
import pandas as pd


def convert_labels_in_directory(directory_path):
    # List all CSV files
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found.")
        return

    for filename in csv_files:
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {file_path}")

        # Load CSV
        df = pd.read_csv(file_path)

        # Check if label column exists
        if "label" not in df.columns:
            print(f"Skipping {filename}: no 'label' column found.")
            continue

        # Convert labels
        df["label"] = df["label"].map({
            "positive": 1,
            "negative": 0
        })

        df["text"] = df["text"].str.replace('"', '')
        df["text"] = df["text"].str.replace("'", '')

        # Optional: warn about unmapped values
        if df["label"].isna().any():
            print(f"Warning: unmapped label values in {filename}")

        # Save back to CSV
        df.to_csv(file_path, index=False)
        print(f"Saved updated file: {file_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert sentiment labels in all CSV files within a directory."
    )
    parser.add_argument(
        "--directory_path",
        required=True,
        help="Path to the directory containing CSV files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_labels_in_directory(args.directory_path)