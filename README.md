# Want Better Synthetic Data? Steer It
## Activation Steering for Low-Resource Language Generation

This repository contains the code accompanying the paper:

> **Want Better Synthetic Data? Steer It: Activation Steering for Low-Resource Language Generation**

- **Language Steering** — steering generation toward the linguistic identity of a target language.
- **Quality Steering** — steering generation toward higher-quality / more natural generations using contrastive activation vectors.

The repository includes:
- scripts for creating contrastive datasets,
- scripts for collecting activations and building steering vectors,
- generation pipelines for steered and non-steered generation,
- downstream fine-tuning scripts,
- preprocessing utilities.

---

# Repository Structure

```text
.
├── create_contrast_data/
├── create_steer_vecs/
├── finetune_downstream/
├── steering_gen_llms/
├── create_steer_vector_alt.py
├── preprocess_sentiment.py
├── preprocess_topic.py
├── requirements.txt
└── Synthetic_data_steering_vectors.pdf
```

---

# Important Notes

## Missing Data / Vectors

The repository intentionally omits several large files due to repository and attachment size limits:

- Generated synthetic datasets
- Human-written datasets
- Precomputed steering vectors
- Full experimental outputs

These resources will be added after acceptance.

## Generation Scripts

The generation scripts provided in this repository are intended as **examples/templates** demonstrating how experiments were run.

The repository does **not** contain exhaustive scripts for every configuration reported in the paper. Instead, it includes representative examples for:

- Zero-shot generation
- Few-shot generation
- Language steering vectors
- Quality steering vectors

These can be adapted for additional languages, models, and experimental settings.

---

# Installation

## 1. Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Main Components

# 1. Contrastive Data Creation

Directory:

```text
create_contrast_data/
```

This directory contains scripts used to create contrastive data pairs for steering vector construction.

## Files

### `create_data.py`

Creates paired datasets used for activation contrast experiments.

Typical usage:
- constructing positive/negative examples,
- preparing text pairs,
- formatting data for activation extraction.

### `backtranslate_nib.py`

Performs backtranslation-based processing used for quality steering experiments.

This script is used to generate lower-quality or transformed variants of text for contrastive activation collection.

---

# 2. Steering Vector Creation

Directory:

```text
create_steer_vecs/
```

This directory contains the core pipeline for:
- collecting activations,
- computing steering vectors,
- preparing vectors for generation.

## Files

### `collect_activations_qual.py`

Collects hidden-state activations for quality steering experiments.

Used to:
- process contrastive datasets,
- extract activations from target LLMs,
- save activations for later vector computation.

### `collect_lang_activations_and_create_steer_vecs.py`

Collects activations for language steering and creates steering vectors directly.

Used for:
- language identity steering,
- multilingual activation extraction,
- vector generation.

### `create_steer_vector_qual.py`

Builds quality steering vectors from previously collected activations.

Typically computes:
- mean activation differences,
- contrastive steering directions.

### `pipeline_lang_steer.sh`

Example end-to-end pipeline for:
1. collecting activations,
2. computing language steering vectors.

### `pipeline_qual_steer.sh`

Example end-to-end pipeline for:
1. creating contrastive pairs,
2. collecting activations,
3. computing quality steering vectors.

## `contrast_data/`

Contains example contrastive data pairs used for language steering experiments.

Example files:

```text
pairs_flor_de.json
pairs_flor_cs.json
pairs_flor_he.json
...
```

These correspond to language-specific contrastive examples.

---

# 3. Steering-Based Generation

Directory:

```text
steering_gen_llms/
```

This directory contains scripts for generating synthetic datasets using steered LLM inference.

The scripts support:
- Gemma-based models,
- LLaMA-based models,
- zero-shot prompting,
- few-shot prompting,
- language steering,
- quality steering.

## Shell Scripts

### Sentiment Generation

#### `gen_data_sent_gemma_lang.sh`

Example sentiment-generation pipeline using:
- Gemma models,
- language steering vectors.

#### `gen_data_sent_llama.sh`

Example sentiment-generation pipeline using:
- LLaMA models.

### Topic Generation

#### `gen_data_topic_gemma_icl.sh`

Example topic-generation pipeline using:
- Gemma,
- in-context learning (few-shot prompting).

#### `gen_data_topic_llama_icl_lang.sh`

Example topic-generation pipeline using:
- LLaMA,
- language steering,
- few-shot prompting.

### Full Experimental Pipelines

#### `run_all_gen_and_finetuning_gemma_qual_vecs_for_one_lang.sh`

Example orchestration script for:
1. generation,
2. steering,
3. downstream fine-tuning,

using:
- Gemma,
- quality steering vectors,
- one target language.

#### `run_all_gen_and_finetuning_llama_lang_vecs_for_one_lang.sh`

Equivalent orchestration pipeline for:
- LLaMA,
- language steering vectors.

## Python Inference Scripts

### `steering_inference_it_gemma.py`

Inference script for steered generation with Gemma models.

### `steering_inference_it_gemma_icl.py`

Gemma inference with:
- steering,
- in-context learning / few-shot prompting.

### `steering_inference_it_llama.py`

Inference script for steered generation with LLaMA models.

### `steering_inference_it_llama_icl.py`

LLaMA inference with:
- steering,
- few-shot prompting.

---

# 4. Downstream Fine-Tuning

Directory:

```text
finetune_downstream/
```

Contains scripts for training downstream classifiers on generated datasets.

## Files

### `finetune.py`

Main fine-tuning script used for downstream classification experiments.

Typical tasks include:
- sentiment classification,
- topic classification.

### `finetuning.sh`

Example fine-tuning command script.

### `finetuning_2.sh`

Additional fine-tuning examples/configurations.

---

# 5. Preprocessing Utilities

## `preprocess_sentiment.py`

Preprocessing utilities for sentiment datasets.

Typical functionality:
- formatting labels,
- cleaning generations,
- converting datasets into training format.

## `preprocess_topic.py`

Preprocessing utilities for topic-classification datasets.

---
