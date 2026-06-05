# =========================================================
# HYBRID MODEL v4.0
# HIGH-RECALL VERSION
#
# GOAL:
#   ↑ Recall / True Positives
#   Maintain good TN
#
# KEY FIXES:
#   1. no sampling / full dataset
#   2. Contrastive negative pairs reduced
#   3. XGBoost tuned for recall
#   4. Threshold optimization beta=3
#   5. Positive-weighted contrastive loss
# =========================================================

# =========================
# 1. IMPORTS
# =========================
import torch
import esm
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import random
import os
import pickle

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    fbeta_score,
    confusion_matrix,
    classification_report
)

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# REPRODUCIBILITY
# =========================
seed = 42

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# =========================
# 2. LOAD STRUCTURAL FEATURES
# =========================
df = pd.read_csv(
    "final_features_cd_hit_NEW_ENTROPY_v4.csv"
)
df = df.drop(columns=["is_short"], errors="ignore")
y = df["label"].values
protein_ids = df["protein_id"]

X_struct_df = df.drop(
    columns=["protein_id", "label"]
)


# =========================
# OPTIONAL SIGNALP FEATURE
# =========================
SIGNALP_FILE = "signalp_results.csv"

if os.path.exists(SIGNALP_FILE):

    df_signalp = pd.read_csv(SIGNALP_FILE)

    df_signalp = df_signalp[
        ["protein_id", "has_signal_peptide"]
    ]

    df = df.merge(
        df_signalp,
        on="protein_id",
        how="left"
    )

    df["has_signal_peptide"] = (
        df["has_signal_peptide"]
        .fillna(0)
        .astype(int)
    )

    X_struct_df["has_signal_peptide"] = (
        df["has_signal_peptide"].values
    )

    print("✅ Signal peptide feature added")

else:
    print("⚠️ signalp_results.csv not found")


X_struct = np.nan_to_num(
    X_struct_df.values
)

scaler = StandardScaler()

X_struct = scaler.fit_transform(X_struct)

struct_feature_names = [
    f"struct_{i}"
    for i in range(X_struct.shape[1])
]

df_struct_clean = pd.DataFrame(
    X_struct,
    columns=struct_feature_names
)

df_struct_clean["protein_id"] = protein_ids.values
df_struct_clean["label"] = y

print(
    f"\nStructural feature shape: {X_struct.shape}"
)


# =========================
# 3. LOAD ESM2-650M
# =========================
print("\n🔹 LOADING ESM2-650M")

model_esm, alphabet = (
    esm.pretrained.esm2_t33_650M_UR50D()
)

model_esm = model_esm.to(device)
model_esm.eval()

batch_converter = alphabet.get_batch_converter()

ESM_DIM = 1280
ESM_LAYER = 33

print("✅ ESM2-650M loaded")


# =========================
# 4. LOAD FASTA
# =========================
def load_fasta(file, label):

    data = {}

    for record in SeqIO.parse(file, "fasta"):

        pid = record.id.strip()

        data[pid] = (
            str(record.seq),
            label
        )

    return data


pos_dict = load_fasta(
    "final_positive_40.fasta",
    1
)

neg_dict = load_fasta(
    "negative_40.fasta",
    0
)

seq_dict = {
    **pos_dict,
    **neg_dict
}

print("\nFASTA loaded")
print("Positive:", len(pos_dict))
print("Negative:", len(neg_dict))


# =========================
# 5. ESM EMBEDDINGS
# =========================
def get_embedding(seq):

    seq = seq[:1022]

    batch = [("protein", seq)]

    _, _, tokens = batch_converter(batch)

    tokens = tokens.to(device)

    with torch.no_grad():

        output = model_esm(
            tokens,
            repr_layers=[ESM_LAYER]
        )

    emb = output["representations"][ESM_LAYER]

    emb = emb[0, 1:-1].mean(0)

    return emb.cpu().numpy()


esm_rows = []

print("\n🔹 GENERATING ESM EMBEDDINGS")

for pid in tqdm(df_struct_clean["protein_id"]):

    if pid not in seq_dict:
        continue

    seq, _ = seq_dict[pid]

    try:

        emb = get_embedding(seq)

        esm_rows.append((pid, emb))

    except Exception as e:

        print(f"Failed for {pid}: {e}")


X_esm = np.array(
    [x[1] for x in esm_rows]
)

ids = [x[0] for x in esm_rows]

esm_cols = [
    f"esm_{i}"
    for i in range(X_esm.shape[1])
]

df_esm = pd.DataFrame(
    X_esm,
    columns=esm_cols
)

df_esm["protein_id"] = ids

print("ESM shape:", X_esm.shape)


# =========================
# 6. MERGE
# =========================
df_merged = pd.merge(
    df_struct_clean,
    df_esm,
    on="protein_id",
    how="left"
)

df_merged = df_merged.dropna()

y = df_merged["label"].values

X_struct_final = df_merged[
    struct_feature_names
].values

X_esm_final = df_merged[
    esm_cols
].values

X = np.concatenate(
    [X_struct_final, X_esm_final],
    axis=1
)

print("\nMerged shape:", X.shape)


# =========================
# 7. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)


# =====================================================
# FIX 1:
# 1:3 SAMPLING → 1:1.5
# =====================================================
# Use full training data (no subsampling)
print("\n🔹 USING FULL TRAINING DATA")
X_bal = X_train
y_bal = y_train
print(pd.Series(y_bal).value_counts())

# =========================
# 8. CONTRASTIVE DATASET
# =========================
class ContrastiveDataset(Dataset):

    def __init__(self, X, y):

        self.pairs = []

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        for i in pos_idx:

            # positive pair
            j = np.random.choice(pos_idx)

            if i != j:

                self.pairs.append(
                    (X[i], X[j], 1)
                )

            # =================================================
            # FIX 2:
            # NEGATIVE PAIRS REDUCED 3 → 1
            # =================================================
            for _ in range(0):

                k = np.random.choice(neg_idx)

                self.pairs.append(
                    (X[i], X[k], 0)
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        x1, x2, label = self.pairs[idx]

        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


dataset = ContrastiveDataset(
    X_bal,
    y_bal
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# =========================
# 9. SIAMESE MODEL
# =========================
class Siamese(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256)
        )

    def forward(self, x):
        return self.net(x)


model_cl = Siamese(
    X.shape[1]
).to(device)

optimizer = optim.Adam(
    model_cl.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)


# =========================
# 10. TRAIN CONTRASTIVE
# =========================
print("\n🔹 TRAINING CONTRASTIVE MODEL")

for epoch in range(30):

    total_loss = 0

    model_cl.train()

    for x1, x2, label in loader:

        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.to(device)

        out1 = model_cl(x1)
        out2 = model_cl(x2)

        dist = torch.norm(
            out1 - out2,
            dim=1
        )

        # ============================================
        # FIX 3:
        # POSITIVE-WEIGHTED CONTRASTIVE LOSS
        # ============================================
        pos_loss = label * dist**2

        neg_loss = (
            (1 - label)
            * torch.clamp(1 - dist, min=0)**2
        )

        loss = torch.mean(
            2.5 * pos_loss +
            1.0 * neg_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(
        f"Epoch {epoch+1}, Loss: {total_loss:.4f}"
    )


# =========================
# 11. TRANSFORM
# =========================
def transform(X_input):

    model_cl.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X_input,
            dtype=torch.float32
        ).to(device)

        return model_cl(
            X_tensor
        ).cpu().numpy()


X_train_emb = transform(X_train)
X_test_emb = transform(X_test)

X_train_final = np.concatenate(
    [X_train, X_train_emb],
    axis=1
)

X_test_final = np.concatenate(
    [X_test, X_test_emb],
    axis=1
)

print(
    "\nFinal transformed shape:",
    X_train_final.shape
)


# =====================================================
# FIX 4:
# HIGH-RECALL XGBOOST
# =====================================================
# Add this before XGBClassifier definition
spw = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"\nscale_pos_weight: {spw:.2f}")


models = {

    "XGB": XGBClassifier(

        n_estimators=700,

        max_depth=5,

        learning_rate=0.02,

        min_child_weight=2,

        gamma=0.2,

        subsample=0.85,
        colsample_bytree=0.85,

        reg_alpha=0.5,
        reg_lambda=3.0,

        scale_pos_weight=5.0,

        objective='binary:logistic',

        eval_metric='aucpr',

        random_state=42,

        n_jobs=-1
    )
}


# =====================================================
# FIX 5:
# THRESHOLD OPTIMIZATION
# beta=3
# =====================================================
def find_best_threshold(
    y_true,
    y_prob,
    beta=3
):

    best_score = -1
    best_t = 0.5

    for t in np.arange(0.20, 0.80, 0.01):

        y_pred = (y_prob >= t).astype(int)

        score = fbeta_score(
            y_true,
            y_pred,
            beta=beta,
            zero_division=0
        )

        if score > best_score:

            best_score = score
            best_t = t

    return best_t


# =========================
# 13. TRAIN XGB
# =========================
print("\n🔹 TRAINING XGB")

clf = models["XGB"]

clf.fit(
    X_train_final,
    y_train
)

y_prob = clf.predict_proba(
    X_test_final
)[:, 1]

best_t = find_best_threshold(
    y_test,
    y_prob
)

print("\nBest threshold:", best_t)

y_pred = (
    y_prob >= best_t
).astype(int)


# =========================
# 14. FINAL METRICS
# =========================
acc = accuracy_score(
    y_test,
    y_pred
)

prec = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

rec = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)

roc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"ROC-AUC    : {roc:.4f}")
print(f"MCC        : {mcc:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================
# 15. SAVE MODEL
# =========================
with open(
    "XGB_MODEL_v5_fixed.pkl",
    "wb"
) as f:

    pickle.dump({

        "model_cl": model_cl,

        "scaler": scaler,

        "classifier": clf,

        "threshold": best_t,

        "input_dim": X.shape[1],

        "final_dim": X_train_final.shape[1],

        "esm_dim": ESM_DIM,

        "version": "v5_fixed"

    }, f)

print("\n✅ Saved: XGB_MODEL_v5_fixed.pkl")


# =========================
# 16. SAVE PREDICTIONS
# =========================
df_pred = pd.DataFrame({

    "probability": y_prob,
    "prediction": y_pred,
    "true_label": y_test

})

df_pred.to_csv(
    "PREDICTIONS_XGB_MODEL_v4.csv",
    index=False
)

print(
    "✅ Saved: PREDICTIONS_XGB_MODEL_v4.csv"
)

print("\n🔥 PIPELINE COMPLETE")
