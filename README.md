# MFML

Model implementation for **Predictive Business Process Monitoring Based on Multi-View Fusion and Multi-Task Learning**.

MFML combines a Medium-BERT semantic view, a Transformer–LSTM order view, and a residual CNN attribute view through gated fusion. Two prediction heads jointly predict the next activity and remaining time. `MFML.py` combines the three views and provides the prediction heads.

## Files

- `MFML.py`: main model, gated fusion, and prediction heads.
- `semantic.py`: Medium-BERT semantic view.
- `order.py`: Transformer-LSTM order view.
- `attribute.py`: residual CNN attribute view.

Keep these four Python files in the same directory.

## Requirements

Tested with Python 3.11, PyTorch 2.5.1, and Transformers 4.46.3.

```bash
pip install torch==2.5.1 transformers==4.46.3
```

Use [Medium-BERT](https://huggingface.co/prajjwal1/bert-medium) or an existing local copy through `bert_model_name_or_path`.

## Usage

```python
import torch
from MFML import MFML

model = MFML(
    categorical_cardinalities={"activity": 12, "resource": 20},
    num_activities=10,
    bert_model_name_or_path="prajjwal1/bert-medium",  # or a local directory
)

# Shape example; replace these tensors with your preprocessed event prefixes.
model.eval()
with torch.no_grad():
    output = model(
        input_ids=torch.tensor([[101, 2054, 102]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        categorical_ids=torch.tensor([[[2, 2], [3, 4]]]),
        numeric_features=torch.zeros(1, 2, 2),
        event_mask=torch.ones(1, 2, dtype=torch.bool),
    )
print(output["activity_logits"].shape)  # [1, 10]
print(output["remaining_time_z"].shape)  # [1]
```

## Inputs and outputs

- Text tensors have shape `[batch, tokens]` and describe only observed prefix events. Categorical IDs have shape `[batch, events, attributes]`, following the attribute order passed to the constructor; reserve `0` for padding and `1` for unknown values. Events are right-padded.
- Numeric inputs are `[seconds_since_last, seconds_since_start]`, each transformed with `log1p` and standardized using training-set statistics. For the paper setting, pass training-set CBOW matrices of shape `[vocab_size, embedding_dim]` (including PAD/UNK rows) through `embedding_initial_weights`; otherwise embeddings initialize randomly.
- Outputs are next-activity logits, `remaining_time_z`, and three view weights (`semantic`, `order`, `attribute`). Time is predicted in standardized `log1p(days)` space: `days = torch.expm1(z * train_std + train_mean).clamp_min(0)`.

Architecture parameters are defined directly in the Python files. Prepare your own inputs and training loop; no dataset or configuration files are required by this model-only release.

## Datasets

Download the original event logs from the sources below and preprocess them separately. Datasets are not bundled with this repository.

| Dataset | Download page |
| --- | --- |
| BPIC_2013_C | [BPI Challenge 2013 — closed problems](https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11) |
| Receipt | [CoSeLoG receipt phase](https://doi.org/10.4121/uuid:a07386a5-7be3-4367-9535-70bc9e77dbe6) |
| Helpdesk | [Help desk log of an Italian company](https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb) |
| BPIC_2017_O | [BPI Challenge 2017 — Offer log](https://doi.org/10.4121/12705737) |
| BPIC_2020_Re | [BPI Challenge 2020 — Request For Payment](https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030) |
| BPIC_2020_Pr | [BPI Challenge 2020 — Prepaid Travel Costs](https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165) |
| Production | [Production Analysis with Process Mining Technology](https://doi.org/10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399) |
| MIP | [Author repository — mip.csv](https://github.com/Sergey-Zeltyn/MIP-dataset) |

For Production, use `Complete Timestamp` as the event time.
