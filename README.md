# MFML

Code for **Predictive Business Process Monitoring Based on Multi-View Fusion and Multi-Task Learning**.

MFML combines a Medium-BERT semantic view, a Transformer-LSTM order view, and a residual CNN attribute view through gated fusion. Two heads jointly predict the next activity and remaining time. This repository includes preprocessing, training, and evaluation; datasets and pretrained weights are external inputs.

## Installation

Use Python 3.11. From the project directory, create and activate a virtual environment, then install the runtime dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU training, replace the CPU PyTorch installation with the PyTorch 2.5.1 build appropriate for your CUDA environment. The validation baseline is Python 3.11, PyTorch 2.5.1 CPU, and Transformers 4.46.3. GPU training and complete reproduction of the eight-dataset paper results are not claimed here.

## Quick start

Download Helpdesk from the table below and place your CSV export at `data/Helpdesk.csv`. CSV, XES, and compressed `.xes.gz` inputs are supported. Download [Medium-BERT](https://huggingface.co/prajjwal1/bert-medium), or use an existing local model directory:

```bash
huggingface-cli download prajjwal1/bert-medium --local-dir resources/bert-medium
python preprocess.py --dataset Helpdesk --data data/Helpdesk.csv --output prepared/Helpdesk.pt
python train.py --prepared prepared/Helpdesk.pt --bert-model resources/bert-medium --output runs/Helpdesk --local-files-only
python evaluate.py --checkpoint runs/Helpdesk/best.pt --prepared prepared/Helpdesk.pt
```

Use the dataset name from the table with the matching source file. For example, an original Production XES file can be prepared with:

```bash
python preprocess.py --dataset Production --data data/Production.xes --output prepared/Production.pt
```

No JSON configuration is required. Dataset column mappings, narrative templates, and defaults live in `settings.py`. Missing required fields or conflicting source columns cause an error instead of silently changing the input schema. Production uses **`Complete Timestamp`**, rather than `time:timestamp`, as its event time. Source exports must match the registered dataset schema; downloading a public log alone does not establish equivalence to the paper's processed input.

For a new log with `case`, `activity`, `timestamp`, and `resource` columns, use `--dataset custom`. For another schema, add or edit a `DatasetSpec` registration in `settings.DATASETS`, including its categorical attributes and text templates. Only use attributes available at prediction time.

## Training and evaluation

Default training uses seed 42, physical batch size 16, accumulation over 2 steps, at most 50 epochs, and early-stopping patience 8. AdamW uses learning rate `1e-5` for BERT and `3e-4` for other parameters; BERT embeddings and its bottom four layers are frozen.

Supported training options include `--device`, `--seed`, `--epochs`, `--batch-size`, `--accumulation-steps`, `--lambda1`, `--lambda2`, `--no-amp`, `--local-files-only`, and `--threads`. See `python train.py --help` for values and defaults. AMP applies on CUDA; use `--device cpu` for CPU execution. For example:

```bash
python train.py --prepared prepared/Helpdesk.pt --bert-model resources/bert-medium --output runs/Helpdesk-short --epochs 2 --device cpu --threads 2 --local-files-only
```

The objective is `(lambda1 * CE + lambda2 * SmoothL1) / (lambda1 + lambda2)`, with both weights defaulting to 1. CE uses inverse-square-root class-frequency weights and label smoothing 0.05. SmoothL1 uses beta 0.5 on the standardized `log1p(remaining_days)` target. Accumulation accounts for the actual group size, including the final incomplete group.

The prepared file stores split data and fitted preprocessing. Training writes `best.pt`, `history.csv`, `metrics.json`, `predictions.csv` (including the three gate weights), and tokenizer assets under the run directory. The best checkpoint is selected using validation data before test evaluation. Reports include Accuracy, support-weighted Precision/F1, and MAE/RMSE in days; validation selection uses the existing joint macro-F1/MAE score.

Each evaluation writes an independent `reevaluation-<timestamp>` directory without overwriting training results. Keep the prepared file and complete run directory together when transferring an experiment. Reevaluation needs these saved artifacts, but does not require the original source log or original BERT folder. Load only trusted prepared files and checkpoints.

## Data protocol

- Cases are ordered by start time: the last 30% form the test split; the first 70% are divided 80%/20% into training/validation, approximately 56%/14%/30% overall.
- Every eligible prefix satisfies `2 <= k < n`. All prefixes of a case remain in one split. Inputs contain observed events only; remaining time is computed from the observed prefix end to the case end.
- Vocabularies, CBOW embeddings, numeric statistics, and target scaling are fitted on training data only. CBOW is required: dimension 32, window 5, 20 epochs; initialized embeddings remain trainable. Inputs are capped at 256 events and 512 text tokens.
- Numeric event features are seconds since the preceding event and since case start, transformed with `log1p` and standardized. Remaining-time predictions are inverted to days for reporting.
- All valid prefixes are scored. Unseen true activity names are retained, so predicting an unknown token does not count as a correct prediction of an unseen activity.

## Datasets

| Dataset | Download page |
| --- | --- |
| BPIC_2013_C | [BPI Challenge 2013 - closed problems](https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11) |
| Receipt | [CoSeLoG receipt phase](https://doi.org/10.4121/uuid:a07386a5-7be3-4367-9535-70bc9e77dbe6) |
| Helpdesk | [Help desk log of an Italian company](https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb) |
| BPIC_2017_O | [BPI Challenge 2017 - Offer log](https://doi.org/10.4121/12705737) |
| BPIC_2020_Re | [BPI Challenge 2020 - Request For Payment](https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030) |
| BPIC_2020_Pr | [BPI Challenge 2020 - Prepaid Travel Costs](https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165) |
| Production | [Production Analysis with Process Mining Technology](https://doi.org/10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399) |
| MIP | [Author repository - mip.csv](https://github.com/Sergey-Zeltyn/MIP-dataset) |

## Files and validation

`MFML.py` combines the model views defined in `semantic.py`, `order.py`, and `attribute.py`. `settings.py` defines dataset schemas; `preprocess.py`, `train.py`, and `evaluate.py` provide the workflow. Keep these Python files in the same directory.

Validated in an independent Python 3.11 / PyTorch 2.5.1 CPU environment: dependency installation and `pip check`, 30 regression tests, preprocessing and safe reload of all eight local datasets, and CSV/XES/XES.gz input coverage. A real Medium-BERT run completed two epochs on a 20-case synthetic log with batch size 4; after moving the run and prepared file, offline reevaluation reproduced the metrics exactly and predictions byte-for-byte. All three command-line scripts passed help checks from another working directory, and the four model files remain unchanged; these checks do not constitute full eight-dataset training or GPU validation.

Code is provided under the [MIT License](LICENSE). Dataset and pretrained-model licenses remain with their respective providers.
