## Predictive Business Process Monitoring Based on Multi-Semantic Fusion and Multi-Task Learning

### Introduction
> In this paper, we propose a predictive business process monitoring framework based on multi-semantic fusion and multi-task learning. First, large language models generate semantic representations of event sequences by converting event log trajectories into semantic narratives. Subsequently, a Transformer encoder captures global contextual information, while an LSTM decoder models fine-grained temporal dynamics within events. Finally, residual convolutional networks extract semantic features reflecting implicit attribute correlations among events. These diverse representations are fused and fed into a multi-task learning framework for the simultaneous prediction of the next activity and remaining time.

### Environment Requirment
> This code has been tested running undeer Python 3.9.23
> The Required packages are as follows:
> - torch == 2.1.1+cu129
> - numpy == 1.26.4
> - transformers == 4.38.2
> - pandas == 2.0.3
> - tqdm == 4.66.2
> - scipy == 1.12.0
> - jinjia2 == 3.1.3
> - scikit-learn == 1.1.3
> - imbalanced-learn == 0.12.3

##### NOTE: The medium-BERT model you can download from`https://huggingface.co/prajjwal1/bert-medium`

### Project Layout & Artifacts
```
.
├─ MFML.py # main training / evaluation script
├─ data/
│ └─ {csv_log}.csv # your source event log (CSV)
├─ utility/
│ ├─ log_config.py # dataset templates & field config (required)
│ └─ Bert-medium/ # tokenizer + weights (or use a HF model name)
├─ log_history/{csv_log}/ # semantic story caches & labels
│ ├─ {csv_log}_train_all.pkl
│ ├─ {csv_log}_test_all.pkl
│ ├─ {csv_log}_label_train_all.pkl
│ ├─ {csv_log}_label_test_all.pkl
│ ├─ {csv_log}_label_rt_train_all.pkl
│ ├─ {csv_log}_label_rt_test_all.pkl
│ ├─ {csv_log}_id2label_all.pkl
│ └─ {csv_log}_label2id_all.pkl
├─ pro_data/ # attribute tensors & dataframes
│ ├─ {csv_log}_train_df.pkl
│ ├─ {csv_log}_test_df.pkl
│ ├─ {csv_log}_train_data_x.npy # [N, T, M, D]
│ ├─ {csv_log}_train_data_y.npy
│ ├─ {csv_log}_test_data_x.npy
│ └─ {csv_log}_test_data_y.npy
├─ w2v_model/ # per‑attribute Word2Vec models
│ └─ {csv_log}_{attribute}_w2v_model.h5
└─ models/
└─ {csv_log}_all.pth # best checkpoint (by validation)
```

###### The _all suffix comes from TYPE='all'. If you change TYPE, corresponding filenames change accordingly.

### Your Data (CSV) & Mandatory Columns
Place the source log at data/{csv_log}.csv. The CSV must contain at least:

- case — case identifier

- activity — event name (classification target)

- timestamp — parsable by pandas.to_datetime

- Additional attribute columns matching ATTRIBUTES used by the script 

- remaining_time — remaining time to case end in days (target for RTP)

### How to Run (Quickstart)

- Prepare CSV in data/ with columns above, including remaining_time (days)

- Configure templates in utility/log_config.py (key must equal csv_log)

- Point BERT to a valid local folder or HF model name

- Install deps (Section 3)

- Tune hyperparams if needed (Section 6)


Launch:

```
python MFML.py
```

First run will:

1. Split cases chronologically into train/test (e.g., 70/30)

2. Build semantic story caches under log_history/

3. Train per‑attribute Word2Vec encoders & export attribute tensors into pro_data/

4. Train with early stopping; save best checkpoint to models/

5. Evaluate on test set and print metrics


### Tips & Troubleshooting

- FileNotFoundError: utility/Bert-medium/
  Use a valid HF model name or provide a local folder; update both tokenizer and model paths.
- KeyError: 'remaining_time' or unusually large MAE
  Add/verify remaining_time in days. See Section 4.1.
- OOM / out of GPU memory
  Reduce BATCH_SIZE, MAX_LEN, Windows_size, or ENCODING_LENGTH (e.g., 32 → 16).

- Attribute names don’t match
  Ensure ATTRIBUTES exactly matches CSV column names.

- Caches inconsistent after changing settings
  Remove log_history/ & pro_data/ and re‑run to rebuild caches.

### License

-This project is released under the [**MIT License**](https://opensource.org/license/MIT%7D%7B%5Ctextbf%7BMIT).
-The complete text of the license can be found in the [LICENSE](LICENSE) file in the root directory of this repository.

