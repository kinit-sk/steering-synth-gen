import os
import pandas as pd
from collections import defaultdict
import json
import argparse

# -----------------------
# Parse command-line arguments
# -----------------------
parser = argparse.ArgumentParser(description="Convert CSV to JSON with good/bad data.")
parser.add_argument("--lang", type=str, required=True, help="Language code for CSV (e.g., az-AZ)")
parser.add_argument("--source_file", type=str, default="trans_2")
parser.add_argument("--target_file", type=str, required=True, help="Path to save the output JSON file")
args = parser.parse_args()

# -----------------------
# Define file paths
# -----------------------
base_dir = "massive_for_pairs"
csv_file = os.path.join(base_dir, f"{args.source_file}.csv")

# -----------------------
# Load CSV and process
# -----------------------
df = pd.read_csv(csv_file).dropna(subset=["backtranslated_text"])

# Assuming the CSV has columns "original_text" and "backtranslated_text"
new_data = defaultdict(list)
new_data["good"].extend(df["original_text"])
new_data["bad"].extend(df["backtranslated_text"])

# -----------------------
# Save to JSON
# -----------------------
with open(args.target_file, "w", encoding="utf8") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print(f"Processed '{csv_file}' and saved JSON to '{args.target_file}'.")