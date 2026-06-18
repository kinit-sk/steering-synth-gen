import pandas as pd
import os
import logging
import argparse
import random
import torch
import numpy as np
import datasets

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import f1_score

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Select device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ------------------------------
# Arguments
# ------------------------------
parser = argparse.ArgumentParser(description='Train on ONE CSV and evaluate ONLY on orig_data_csv.')

parser.add_argument('--no_epochs', type=int, const=50, default=50, nargs='?',
                    help='No. epochs for training.')
parser.add_argument('--seed', type=int, const=0, default=0, nargs='?',
                    help='Seed to be used for shuffling.')
parser.add_argument('--batch_size', type=int, const=32, default=32, nargs='?',
                    help='Training batch size.')
parser.add_argument('--batch_size_eval', type=int, const=512, default=512, nargs='?',
                    help='Eval batch size.')

parser.add_argument('--sampling', action='store_true',
                    help='Whether to enable sampling. Defaults to False.')

parser.add_argument('--sampling_size_per_label', type=int, const=100, default=100, nargs='?',
                    help='Number of samples per label when sampling is enabled.')

parser.add_argument('--text_column', type=str, const='text', default='text', nargs='?',
                    help='Name of the text column in the CSV.')

parser.add_argument('--train_csv', type=str, required=True,
                    help='Single training CSV file.')
parser.add_argument('--val_csv', type=str, required=True,
                    help='Validation CSV file for early stopping.')
parser.add_argument('--orig_data_csv', type=str, required=True,
                    help='The CSV file with original human data used ONLY for evaluation.')

parser.add_argument('--results_dir', type=str, required=True,
                    help='Directory where result CSV will be saved.')
parser.add_argument('--repeat', type=int, const=10, default=10, nargs='?',
                    help='How many times training is repeated.')
parser.add_argument('--base_model_type', type=str, required=True,
                    help='Model type (e.g. roberta-base).')

parser.add_argument('--early_stopping_patience', type=int, default=5,
                    help='Number of epochs with no improvement before stopping.')
parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                    help='Minimum F1 improvement to be considered as progress.')

parser.add_argument('--max_token_len', type=int, default=256,
                    help='Max token length for the classifier.')

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained(args.base_model_type)
random.seed(args.seed)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# ------------------------------
# Data preparation
# ------------------------------
def prepare_data(df, text_column):
    if args.sampling:
        df = (df.groupby('label', group_keys=False)
              .apply(
                  lambda x: x.sample(
                      n=min(len(x), args.sampling_size_per_label),
                      random_state=args.seed
                  )).reset_index(drop=True))

    dataset = datasets.Dataset.from_pandas(df[[text_column, 'label']])
    textos = [str(t) for t in dataset[text_column]]
    #print(len(textos))

    tokenized = tokenizer(
        textos,
        padding=True,
        return_tensors='pt',
        truncation=True,
        max_length=args.max_token_len,
        add_special_tokens=True
    )

    tokenized['label'] = dataset['label']
    tokenized['text'] = dataset[text_column]

    ds = datasets.Dataset.from_dict(tokenized).with_format("torch")
    return ds


# ------------------------------
# Evaluation
# ------------------------------
def eval_loop(model, ds, eval_batch_size):
    model.eval().to(device)
    loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False)

    all_preds = []
    all_corrs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs[1], 1)

            all_preds.extend(predicted.cpu().tolist())
            all_corrs.extend(labels.cpu().tolist())

    return all_preds, all_corrs


def eval_on_dataframe(model, df):
    ds_val = prepare_data(df, args.text_column)
    preds, corrs = eval_loop(model, ds_val, args.batch_size_eval)
    return f1_score(corrs, preds, average='macro')


def run_eval_and_log_results_orig(model, df):
    ds_test = prepare_data(df, args.text_column)
    preds, corrs = eval_loop(model, ds_test, args.batch_size_eval)
    return f1_score(corrs, preds, average='macro')


# ------------------------------
# Training loop with early stopping
# ------------------------------
def train_loop(model, ds, df_val, batch_size=16, num_epochs=1):
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, num_workers=0)

    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_f1 = -np.inf
    best_state_dict = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_f1 = eval_on_dataframe(model, df_val)

        logging.info(
            f"Epoch {epoch} | Train loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1 + args.early_stopping_min_delta:
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logging.info(f"New best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            logging.info(
                f"No improvement. Patience {patience_counter}/{args.early_stopping_patience}"
            )

            if patience_counter >= args.early_stopping_patience:
                logging.info("Early stopping triggered.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


# ------------------------------
# Main logic
# ------------------------------
random_state = args.seed

df_train = pd.read_csv(args.train_csv).drop_duplicates().dropna()
df_val = pd.read_csv(args.val_csv).drop_duplicates().dropna()
df_test_orig = pd.read_csv(args.orig_data_csv).drop_duplicates().dropna()

ds_train = prepare_data(df_train, args.text_column)
ds_dct = datasets.DatasetDict({"train": ds_train})

model = AutoModelForSequenceClassification.from_pretrained(
    args.base_model_type,
    num_labels=len(set(df_train["label"])),
    classifier_dropout=0.2
)

model = train_loop(
    model,
    ds_dct,
    df_val,
    batch_size=args.batch_size,
    num_epochs=args.no_epochs
)

logging.info("Running evaluation ONLY on original human data.")

dct_res = {'res_f1': [], 'seed': []}

res = run_eval_and_log_results_orig(model, df_test_orig)

dct_res['res_f1'].append(res)
dct_res['seed'].append(random_state)

os.makedirs(args.results_dir, exist_ok=True)
save_path = os.path.join(args.results_dir, "results_base.csv")

res_df = pd.DataFrame.from_dict(dct_res)

if not os.path.exists(save_path):
    res_df.to_csv(save_path, index=False)
else:
    res_df.to_csv(save_path, mode='a', header=False, index=False)
