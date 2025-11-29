# -*- coding: utf-8 -*-
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
from jinja2 import Template
from tqdm import tqdm
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
from utility import log_config as lg
# -------------------- 基础设置 --------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
seed = 42
set_seed(seed)
csv_log = 'mip'
TYPE = 'all'
EPOCHS = 50
BATCH_SIZE = 16
MAX_LEN = 512
Windows_size = 30
ENCODING_MODE = 'w2v'
ENCODING_LENGTH = 32
LEARNING_RATE = 4e-5
HIDDEN_DIM = 384
ATTRIBUTES = ("activity", "resource", "session_id")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------- 多任务权重 --------------------
LAMBDA_CLS = 1
LAMBDA_RT  = 0.02#0.01~0.1
class CsvEventLog:
    def __init__(self, log_add, attributes, encoding_length=32, encoding_mode='embedding', min_length=3, random_seed=42, glove_path=None):
        self.attributes = list(attributes)               # M 个属性
        self.encoding_mode = encoding_mode
        self.min_length = int(min_length)
        self.random_seed = int(random_seed)
        self.dataset = os.path.splitext(os.path.basename(log_add))[0]
        df0 = pd.read_csv(log_add)
        self.log_df = df0
        self.attributes_values = {
            att: sorted(df0[att].astype(str).unique().tolist())
            for att in self.attributes
        }
        self.attributes_values_nums = {att: len(vs) for att, vs in self.attributes_values.items()}
        self.attribute_encodings = {}
        self.encoding_length = int(encoding_length)
        self.trace_lists = self._corpus_build()
        self._w2v_model_build(train_epoch=10)
        for attribute in self.attributes:
            self.attribute_encodings[attribute] = self._w2v_encoding(attribute)
    def _corpus_build(self):
        traces = {att: [] for att in self.attributes}
        for _, g in self.log_df.groupby('case', sort=False):
            g = g[self.attributes].astype(str)
            for att in self.attributes:
                traces[att].append(g[att].tolist())
        return traces
    def _w2v_model_build(self, train_epoch=10):
        os.makedirs('w2v_model', exist_ok=True)
        for att in self.attributes:
            model = Word2Vec(
                vector_size=self.encoding_length,
                window=5, min_count=1, workers=1,
                sg=0, seed=self.random_seed
            )
            model.build_vocab(self.trace_lists[att], min_count=1)
            model.train(self.trace_lists[att], total_examples=model.corpus_count, epochs=train_epoch)
            model.save(f"w2v_model/{self.dataset}_{att}_w2v_model.h5")
    def _w2v_encoding(self, attribute_name):
        vec_model = Word2Vec.load(f"w2v_model/{self.dataset}_{attribute_name}_w2v_model.h5")
        enc = {}
        for v in self.attributes_values[attribute_name]:
            key = str(v)
            enc[key] = vec_model.wv[key].astype(np.float32)
        return enc

    # ---------- 编码 df_split → [N,K,M,D], y ----------
    def encode_df(self, df_split, label2id_activity, time_steps, pad_value=0.0):
        """
        固定窗口 K = time_steps，左侧补零。
        返回：
          X: [N, K, M, D]（float32）
          y: [N]（int64）
        """
        M, D = len(self.attributes), self.encoding_length
        pad_vec = [pad_value] * D

        def _left_pad_K(prefix):
            need = time_steps - len(prefix)
            if need > 0:
                pad = [[[pad_value] * D for _ in range(M)] for _ in range(need)]
                return pad + prefix
            return prefix

        def _lookup(att, val):
            table = self.attribute_encodings[att]
            key = str(val)
            if key in table:
                vec = table[key]
                return vec if isinstance(vec, list) else np.asarray(vec, dtype=np.float32).tolist()
            return pad_vec

        X, y = [], []
        for _, g in df_split.groupby('case', sort=False):
            enc, acts = [], []
            for _, row in g.iterrows():
                evt = [_lookup(att, row[att]) for att in self.attributes]  # [M, D]
                enc.append(evt)
                acts.append(str(row['activity']))
            n = len(enc)
            if n < time_steps + 1:
                continue
            for t in range(time_steps, n):
                start = t - time_steps
                win = enc[start:t]
                win = _left_pad_K(win)
                X.append(win)
                y.append(label2id_activity[acts[t]])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
# -------------------- 数据预处理：保存属性侧序列 --------------------
def data_pro():
    df_train = pd.read_pickle(f'pro_data/{csv_log}_train_df.pkl')
    df_test  = pd.read_pickle(f'pro_data/{csv_log}_test_df.pkl')

    log_attr = CsvEventLog(
        log_add=f'data/{csv_log}.csv',
        attributes=ATTRIBUTES,
        encoding_length=ENCODING_LENGTH,
        encoding_mode=ENCODING_MODE,
    )
    with open(f'log_history/{csv_log}/{csv_log}_label2id_{TYPE}.pkl', 'rb') as f:
        label2id_all = pickle.load(f)
    label2id_activity = label2id_all['activity']

    x_tr, y_tr = log_attr.encode_df(df_train, label2id_activity, time_steps=Windows_size)
    x_te, y_te = log_attr.encode_df(df_test,  label2id_activity, time_steps=Windows_size)

    out = f'pro_data/{csv_log}'
    np.save(out + '_train_data_x.npy',  x_tr)   # [N, K, M, D]
    np.save(out + '_train_data_y.npy',  y_tr)
    np.save(out + '_test_data_x.npy',   x_te)
    np.save(out + '_test_data_y.npy',   y_te)

# -------------------- 语义故事与标签生成 --------------------
class Log():
    def __init__(self, log, setting):
        self.__log_name = log
        self.__log = pd.read_csv('data/'+log+'.csv')
        self.__train = []
        self.__test = []
        self.__len_prefix_test = []
        self.__history_train = []
        self.__history_test = []
        self.__dict_label_train = []
        self.__dict_label_test = []
        self.__id2label = {}
        self.__label2id = {}
        self.__setting = setting
        self.__window_size = Windows_size
        self.__max_length = 0
        self.__split_log()

    def __gen_prefix_history(self, df):
        list_seq = []
        list_len_prefix = []
        sequence = df.groupby('case', sort=False)
        event_template = Template(lg.log[self.__log_name]['event_template'])
        trace_template = Template(lg.log[self.__log_name]['trace_template'])

        # 标签
        dict_event_label = {v: [] for v in lg.log[self.__log_name]['event_attribute']}
        dict_event_label['remaining_time'] = []
        dict_len_label = {i: [] for i in range(self.__max_length)}

        for _, group_data in sequence:
            n_tr = len(group_data)
            if n_tr < self.__window_size + 1:
                continue
            event_dict_hist, trace_dict_hist = {}, {}
            ev_snips = []
            base_idx = len(list_seq)

            for _, row in group_data.iterrows():
                for v in lg.log[self.__log_name]['event_attribute']:
                    value = row[v]
                    event_dict_hist[v] = value.replace(' ', '') if isinstance(value, str) else value
                snippet = event_template.render(event_dict_hist)
                ev_snips.append(snippet)

                start = max(0, len(ev_snips) - self.__window_size)
                win_snips = ev_snips[start:]
                event_text = ' '.join(win_snips) + ' '

                for w in lg.log[self.__log_name]['trace_attribute']:
                    value = row[w]
                    trace_dict_hist[w] = value.replace(' ', '') if isinstance(value, str) else value
                trace_text = trace_template.render(trace_dict_hist)

                prefix_hist = event_text + trace_text
                if len(ev_snips) >= self.__window_size:
                    list_seq.append(prefix_hist)
                    list_len_prefix.append(len(win_snips))

            # 删除最后一个（没有“下一事件”）
            if len(list_seq) > base_idx:
                list_seq.pop()
                list_len_prefix.pop()

            # 下一事件的分类标签
            k = self.__window_size

            # 下一事件（分类标签）
            next_act_series = group_data['activity'].shift(-1).dropna().iloc[(k - 1):]
            dict_len_label[0].extend([self.__label2id['activity'][a] for a in next_act_series.tolist()])
            for v in lg.log[self.__log_name]['event_attribute']:
                vals = group_data[v].shift(-1).dropna().iloc[(k - 1):]
                dict_event_label[v].extend(vals.tolist())
            rt_series = group_data['remaining_time'].iloc[:-1].iloc[(k - 1):]
            dict_event_label['remaining_time'].extend(rt_series.tolist())
        return list_seq, dict_event_label, list_len_prefix, dict_len_label

    def __extract_timestamp_features(self, group):
        timestamp_col = 'timestamp'
        group = group.sort_values(timestamp_col, ascending=True)
        start_date = group[timestamp_col].iloc[0]
        timesincelastevent = group[timestamp_col].diff()
        timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
        group["timesincelastevent"] = timesincelastevent.apply(lambda x: float(x / np.timedelta64(1, 's')))
        elapsed = group[timestamp_col] - start_date
        elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
        group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))
        return group

    def __split_log(self):
        self.__log['resource'] = self.__log['resource'].astype(str)
        self.__log['resource'] = self.__log['resource'].str.replace(' ', '').str.replace('+','').str.replace('-','').str.replace('_','')
        self.__log.fillna('UNK', inplace=True)
        self.__log['timestamp'] = pd.to_datetime(self.__log['timestamp'])
        for c in lg.log[self.__log_name]['event_attribute']:
            if c not in ('timesincecasestart', 'remaining_time'):
                ALL_LABEL = list(self.__log[c].unique())
                self.__id2label[c] = {k: l for k, l in enumerate(ALL_LABEL)}
                self.__label2id[c] = {l: k for k, l in enumerate(ALL_LABEL)}
        cont_trace = self.__log['case'].value_counts(dropna=False)
        self.__max_length = max(cont_trace)

        self.__log = self.__log.groupby('case', group_keys=False).apply(self.__extract_timestamp_features)
        self.__log = self.__log.reset_index(drop=True)
        self.__log['timesincecasestart'] = self.__log['timesincecasestart'].astype(int)

        grouped = self.__log.groupby("case")
        start_timestamps = grouped["timestamp"].min().reset_index().sort_values("timestamp", ascending=True, kind="mergesort")
        train_ids = list(start_timestamps["case"])[:int(0.7 * len(start_timestamps))]
        self.__train = self.__log[self.__log["case"].isin(train_ids)].sort_values("timestamp", ascending=True, kind='mergesort')
        self.__test  = self.__log[~self.__log["case"].isin(train_ids)].sort_values("timestamp", ascending=True, kind='mergesort')

        os.makedirs("pro_data", exist_ok=True)
        self.__train.to_pickle(f"pro_data/{self.__log_name}_train_df.pkl")
        self.__test.to_pickle(f"pro_data/{self.__log_name}_test_df.pkl")

        self.__history_train, self.__dict_label_train, self.__len_prefix_train, _ = self.__gen_prefix_history(self.__train)
        self.__history_test,  self.__dict_label_test,  self.__len_prefix_test,  _ = self.__gen_prefix_history(self.__test)

        for v in self.__dict_label_train:
            if v in ('timesincecasestart', 'remaining_time'):
                self.__dict_label_train[v] = torch.tensor(self.__dict_label_train[v], dtype=torch.float32).view(-1, 1)
            else:
                temp_list = [self.__label2id[v].get(key) for key in self.__dict_label_train[v]]
                self.__dict_label_train[v] = torch.tensor(temp_list, dtype=torch.long)
        for v in self.__dict_label_test:
            if v in ('timesincecasestart', 'remaining_time'):
                self.__dict_label_test[v] = torch.tensor(self.__dict_label_test[v], dtype=torch.float32).view(-1, 1)
            else:
                temp_list = [self.__label2id[v].get(key) for key in self.__dict_label_test[v]]
                self.__dict_label_test[v] = torch.tensor(temp_list, dtype=torch.long)

        os.makedirs(f'log_history/{self.__log_name}', exist_ok=True)
        # 语义故事
        self.__serialize_object(self.__history_train, 'train')
        self.__serialize_object(self.__history_test, 'test')
        # 标签
        self.__serialize_object(self.__dict_label_train[lg.log[self.__log_name]['target']], 'label_train')
        self.__serialize_object(self.__dict_label_test[lg.log[self.__log_name]['target']], 'label_test')
        self.__serialize_object(self.__dict_label_train['remaining_time'], 'label_rt_train')
        self.__serialize_object(self.__dict_label_test['remaining_time'],  'label_rt_test')

        with open(f'log_history/{self.__log_name}/{self.__log_name}_id2label_{self.__setting}.pkl', 'wb') as f:
            pickle.dump(self.__id2label, f)
        with open(f'log_history/{self.__log_name}/{self.__log_name}_label2id_{self.__setting}.pkl', 'wb') as f:
            pickle.dump(self.__label2id, f)
    def __serialize_object(self, lista, type):
        with open(f'log_history/{self.__log_name}/{self.__log_name}_{type}_{self.__setting}.pkl', 'wb') as f:
            pickle.dump(lista, f)
# -------------------- 数据集 --------------------
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, attr_data=None):
        self.texts = texts
        self.labels = {k: v for k, v in labels.items()}  # dict: '0'（分类）, 'rt'（回归，单位“天”，无标准化）
        self.tokenizer = tokenizer
        self.max_len = max_len
        if attr_data is not None and not isinstance(attr_data, torch.Tensor):
            attr_data = torch.from_numpy(attr_data).float()
        self.attr_data = attr_data  # [N, T, M, D] or None
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }
        # 分类标签
        item_labels = {}
        if '0' in self.labels:
            item_labels['0'] = torch.as_tensor(self.labels['0'][idx], dtype=torch.long)
        # 回归标签
        if 'rt' in self.labels:
            rt_val = self.labels['rt'][idx]
            item_labels['rt'] = torch.as_tensor(rt_val, dtype=torch.float32).view(1)
        item['labels'] = item_labels
        if self.attr_data is not None:
            item['attr_input'] = self.attr_data[idx]   # [T, M, D]
        return item

# -------------------- 模型 --------------------
class ResCell(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1)
        self.b1 = nn.BatchNorm2d(mid_channel)
        self.c2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1)
        self.b2 = nn.BatchNorm2d(mid_channel)
        self.c3 = nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1)
        self.b3 = nn.BatchNorm2d(out_channel)
        self.c4 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
        self.b4 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = F.relu(self.b2(self.c2(y)))
        y = self.b3(self.c3(y))
        x = self.b4(self.c4(x))
        return F.relu(x + y)

class ResBlock(nn.Module):
    def __init__(self, att_channel, output_dim):
        super().__init__()
        self.res1 = ResCell(att_channel, 64, 64, stride=2)
        self.res2 = ResCell(64, 128, 128, stride=2)
        self.res3 = ResCell(128, 256, 256, stride=2)
        self.res4 = ResCell(256, 512, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):                    # [B, M, T, D]
        y = self.res1(x)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = torch.squeeze(self.pool(y), (2, 3))  # [B, 512]
        return F.relu(self.fc(y))

class OrderBranch_PaperPlain(nn.Module):
    def __init__(self, input_dim, lstm_hidden=128, nhead=8, dim_ff=256,
                 t_layers=3, num_layers=2, dropout=0, out_dim=HIDDEN_DIM):
        super().__init__()
        d_model = input_dim
        while d_model % nhead != 0:
            d_model += 1
        self.input_proj = nn.Linear(input_dim, d_model) if d_model != input_dim else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.trans = nn.TransformerEncoder(enc_layer, num_layers=t_layers)

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.to_fuse = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x_cont):          # [B, T, input_dim]
        pad_mask = (x_cont.abs().sum(dim=-1) == 0)
        z = self.input_proj(x_cont)
        z = self.trans(z, src_key_padding_mask=pad_mask)
        y, _ = self.lstm(z)
        last = y[:, -1, :]
        return F.relu(self.to_fuse(last))

class OutputHeadsMTL(nn.Module):
    """
    输出两个头：
      - head_cls: 下一事件分类
      - head_rt : 剩余时间回归（单位：天，非负约束）
    """
    def __init__(self, gpt_model, num_classes,
                 attribute_num=len(ATTRIBUTES), hidden_dim=HIDDEN_DIM,
                 seq_enc_dim=ENCODING_LENGTH,  # 会用真实 D 覆盖
                 seq_lstm_hidden=128, seq_nhead=8, seq_ff=256, seq_tlayers=3):
        super().__init__()
        self.gpt_model = gpt_model
        self.res_block = ResBlock(attribute_num, hidden_dim)
        self.seq_input_dim = attribute_num * seq_enc_dim
        self.order_branch = OrderBranch_PaperPlain(
            input_dim=self.seq_input_dim, lstm_hidden=seq_lstm_hidden,
            nhead=seq_nhead, dim_ff=seq_ff, t_layers=seq_tlayers,
            num_layers=2, dropout=0, out_dim=hidden_dim
        )
        fuse_in = gpt_model.config.hidden_size + hidden_dim + hidden_dim
        self.head_cls = nn.Linear(fuse_in, num_classes)
        self.head_rt  = nn.Linear(fuse_in, 1)  # 回归

    def forward(self, input_ids, attention_mask, attr_input):
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask)
        h_sem = outputs.pooler_output
        h_attr = self.res_block(attr_input.permute(0, 2, 1, 3))      # [B, M, T, D]
        B, T, M, D = attr_input.shape
        x_cont = attr_input.reshape(B, T, M*D)
        h_seq  = self.order_branch(x_cont)
        h = torch.cat([h_sem, h_seq, h_attr], dim=1)
        logits_cls = self.head_cls(h)
        pred_rt = F.relu(self.head_rt(h))  # 非负约束
        return logits_cls, pred_rt

# -------------------- 训练 & 评估 --------------------
def train_fn(model, train_loader, optimizer, device, criterion):
    model.train()
    total = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        attr_input     = batch['attr_input'].to(device)
        logits_cls, pred_rt = model(input_ids, attention_mask, attr_input)

        loss_cls = criterion['cls'](logits_cls, batch['labels']['0'].to(device))
        loss_rt  = criterion['rt'](pred_rt, batch['labels']['rt'].to(device).view(-1,1))
        loss = LAMBDA_CLS * loss_cls + LAMBDA_RT * loss_rt

        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / len(train_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            attr_input     = batch['attr_input'].to(device)
            logits_cls, pred_rt = model(input_ids, attention_mask, attr_input)
            loss_cls = criterion['cls'](logits_cls, batch['labels']['0'].to(device))
            loss_rt  = criterion['rt'](pred_rt, batch['labels']['rt'].to(device).view(-1,1))
            loss = LAMBDA_CLS * loss_cls + LAMBDA_RT * loss_rt
            total += float(loss.item())
    return total / len(data_loader)

def train_llm(model, train_data_loader, valid_data_loader, optimizer, EPOCHS, criterion, device):
    best_valid_loss = float("inf")
    early_stop_counter = 0
    patience = 5
    best_model = model
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_fn(model, train_data_loader, optimizer, device, criterion)
        valid_loss = evaluate_fn(model, valid_data_loader, criterion, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {valid_loss:.4f}")
        if early_stop_counter >= patience:
            print(f"Validation loss hasn't improved for {patience} epochs. Early stopping...")
            break
    return best_model

AVERAGING = 'weighted'
def classification_metrics(y_true, y_pred, average=AVERAGING):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall':    recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1':        f1_score(y_true, y_pred, average=average, zero_division=0),
        'gmean':     geometric_mean_score(y_true, y_pred, average=average)
    }

# -------------------- 训练主流程 --------------------
def train_model():
    MODEL_DIR = "models"
    print(f"使用设备: {device}")
    # --- 读取文本与分类标签 ---
    with open(f'log_history/{csv_log}/{csv_log}_id2label_{TYPE}.pkl', 'rb') as f:
        id2label = pickle.load(f)
    with open(f'log_history/{csv_log}/{csv_log}_train_{TYPE}.pkl', 'rb') as f:
        train_texts = pickle.load(f)
    with open(f'log_history/{csv_log}/{csv_log}_label_train_{TYPE}.pkl', 'rb') as f:
        y_activity_train = pickle.load(f)
    with open(f'log_history/{csv_log}/{csv_log}_label_rt_train_{TYPE}.pkl', 'rb') as f:
        y_rt_train_tensor = pickle.load(f)  # shape [N,1]
    with open(f'log_history/{csv_log}/{csv_log}_label_rt_test_{TYPE}.pkl', 'rb') as f:
        y_rt_test_tensor = pickle.load(f)

    x_train_attr_all = np.load(f'pro_data/{csv_log}_train_data_x.npy')  # [N, T, M, D]
    N = len(train_texts)
    assert x_train_attr_all.shape[0] == N
    assert len(y_activity_train) == N
    assert y_rt_train_tensor.shape[0] == N

    # ====== 训练/验证 切分 ======
    cut = int(0.7 * N)
    idx_tr = np.arange(cut)
    idx_val = np.arange(cut, N)

    train_input = [train_texts[i] for i in idx_tr]
    val_input   = [train_texts[i] for i in idx_val]

    y_cls_all = np.asarray(y_activity_train if not isinstance(y_activity_train, torch.Tensor)
                           else y_activity_train.numpy())
    y_rt_all = y_rt_train_tensor.numpy().reshape(-1)
    train_label = {'0': y_cls_all[idx_tr], 'rt': y_rt_all[idx_tr]}
    val_label   = {'0': y_cls_all[idx_val], 'rt': y_rt_all[idx_val]}
    x_attr_tr  = x_train_attr_all[idx_tr]
    x_attr_val = x_train_attr_all[idx_val]
    enc_dim_real = int(x_attr_tr.shape[-1])
    tokenizer     = AutoTokenizer.from_pretrained('utility/Bert-medium/', truncation_side='left')
    bert_backbone = AutoModel.from_pretrained('utility/Bert-medium/').to(device)

    train_dataset = CustomDataset(train_input, train_label, tokenizer, MAX_LEN, attr_data=x_attr_tr)
    val_dataset   = CustomDataset(val_input,   val_label,   tokenizer, MAX_LEN, attr_data=x_attr_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # --- 模型（BERT + 属性CNN + 顺序Transformer→LSTM） ---
    num_activities = len(id2label['activity'])
    model = OutputHeadsMTL(
        bert_backbone, num_classes=num_activities,
        attribute_num=len(ATTRIBUTES), hidden_dim=HIDDEN_DIM,
        seq_enc_dim=enc_dim_real,          # 用真实 D
        seq_lstm_hidden=128, seq_nhead=8, seq_ff=256, seq_tlayers=3
    ).to(device)

    criterion = {
        'cls': torch.nn.CrossEntropyLoss(),
        'rt':  torch.nn.L1Loss()  # MAE 基损失
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print('TRAINING START...')
    startTime = time.time()
    best_model = train_llm(model, train_loader, val_loader, optimizer, EPOCHS, criterion, device)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(MODEL_DIR, f"{csv_log}_{TYPE}.pth"))
    executionTime = (time.time() - startTime)
    with open(f'{csv_log}_{TYPE}.txt', 'w') as file_time:
        file_time.write(str(executionTime))

    with open(f'log_history/{csv_log}/{csv_log}_test_{TYPE}.pkl', 'rb') as f:
        test_texts = pickle.load(f)
    with open(f'log_history/{csv_log}/{csv_log}_label_test_{TYPE}.pkl', 'rb') as f:
        y_activity_test = pickle.load(f)
    x_test_attr_all = np.load(f'pro_data/{csv_log}_test_data_x.npy')  # [N_te, T, M, D]
    # 回归真值
    y_rt_test_raw = y_rt_test_tensor.numpy().reshape(-1)
    y_test_cls = np.asarray(y_activity_test if not isinstance(y_activity_test, torch.Tensor)
                            else y_activity_test.numpy())

    test_dataset = CustomDataset(
        test_texts,
        {'0': y_test_cls, 'rt': y_rt_test_raw},   # 不标准化
        tokenizer, MAX_LEN, attr_data=x_test_attr_all
    )
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cls_targets, cls_preds = [], []
    rt_targets_raw, rt_preds_raw = [], []
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            attr_input     = batch['attr_input'].to(device)
            logits_cls, pred_rt = best_model(input_ids, attention_mask, attr_input)
            # 分类
            preds  = logits_cls.argmax(dim=1).cpu().numpy()
            labels = batch['labels']['0'].cpu().numpy()
            cls_preds.extend(preds)
            cls_targets.extend(labels)
            # 回归（直接使用模型输出为“天”）
            y_pred = pred_rt.cpu().numpy().reshape(-1)
            y_true = batch['labels']['rt'].cpu().numpy().reshape(-1)
            rt_preds_raw.extend(y_pred.tolist())
            rt_targets_raw.extend(y_true.tolist())
    # ====== 1) NEP：五指标 ======
    m_cls = classification_metrics(cls_targets, cls_preds)
    print(f"\n[NEP] TEST（average={AVERAGING}）:")
    print(f"{'Accuracy':<12}: {m_cls['accuracy']:.4f}")
    print(f"{'Precision':<12}: {m_cls['precision']:.4f}")
    print(f"{'Recall':<12}: {m_cls['recall']:.4f}")
    print(f"{'F1':<12}: {m_cls['f1']:.4f}")
    print(f"{'G-mean':<12}: {m_cls['gmean']:.4f}")

    # ====== 2) 剩余时间：仅报告 MAE ======
    mae = mean_absolute_error(rt_targets_raw, rt_preds_raw)
    print(f"\n[Remaining Time] 指标（仅 MAE）:")
    print(f"{'MAE(days)':<12}: {mae:.4f}")

# -------------------- 主入口 --------------------
def main():
    Log(csv_log, TYPE)
    data_pro()
    train_model()

if __name__ == '__main__':
    main()
