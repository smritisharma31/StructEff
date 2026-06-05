
# =========================================================
# HYBRID MODEL v3
# INDEPENDENT TEST PIPELINE
# Compatible with:
#   XGB_MODEL_v3.pkl
#   SVM_MODEL_v3.pkl
#   LR_MODEL_v3.pkl
#   RF_MODEL_v3.pkl
# =========================================================

# =========================
# 1. IMPORTS
# =========================
import pickle
import torch
import esm
import numpy as np
import pandas as pd
import torch.nn as nn

from Bio import SeqIO
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =========================
# 2. SELECT MODEL
# =========================
# MODEL_FILE = "LR_MODEL_v3.pkl"
MODEL_FILE = "XGB_MODEL_v5_hardneg.pkl"
# MODEL_FILE = "SVM_MODEL_v3.pkl"
# MODEL_FILE = "RF_MODEL_v3.pkl"

# =========================
# 3. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

# =========================
# 4. SIAMESE MODEL v3
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

# =========================
# 5. LOAD SAVED MODEL
# =========================
print("\n🔹 LOADING MODEL")

with open(MODEL_FILE, "rb") as f:
    saved = pickle.load(f)

model_cl   = saved["model_cl"]
scaler     = saved["scaler"]
classifier = saved["classifier"]
threshold  = saved["threshold"]

print("✅ MODEL LOADED")
print("Threshold:", threshold)

model_cl = model_cl.to(device)
model_cl.eval()

# =========================
# 6. LOAD FEATURES
# =========================
print("\n🔹 LOADING FEATURES")

df = pd.read_csv("independent_aligned_features_v4.csv")
df = df.drop(columns=["is_short"], errors="ignore")
protein_ids = df["protein_id"]

# ======================================
# CREATE LABELS
# ======================================

y = np.array([
    1 if pid.startswith("Fung") else 0
    for pid in protein_ids
])

# ======================================
# STRUCTURAL FEATURES
# ======================================

training_columns = [

    'phi_mean', 'phi_std',
    'psi_mean', 'psi_std',

    'rsa_mean', 'rsa_std',

    'contact_number_mean',
    'contact_number_std',

    'long_range_contacts_mean',

    'betweenness_centrality_mean',

    'core_mean',

    'plddt_mean_mean',
    'plddt_mean_std',

    'disorder_proxy_mean',

    'is_helix_mean',
    'is_sheet_mean',
    'is_coil_mean',

    'long_range_norm',

    'charged_fraction',
    'polar_fraction',
    'hydrophobic_fraction',

    'contact_order',

    'graph_density',

    'total_sasa',

    'length',

    'contact_density',

    'mean_entropy',
    'variance_entropy',
    'std_entropy',
    'skew_entropy',
    'pct_conserved',
    'pct_variable',
    'cysteine_count',
    'cysteine_fraction',
    'has_4plus_cys'
]


X_struct_df = df[training_columns]

X_struct = np.nan_to_num(X_struct_df.values)

# IMPORTANT
# Use TRAINING scaler
X_struct = scaler.transform(X_struct)

print("Structural feature shape:", X_struct.shape)

# =========================
# OPTIONAL SIGNAL PEPTIDE
# =========================
# If your v3 training used signal peptide feature,
# uncomment this section and append it.

"""
signalp_df = pd.read_csv("signalp_results.csv")

signalp_df = signalp_df[
    ["protein_id", "has_signal_peptide"]
]

df = df.merge(
    signalp_df,
    on="protein_id",
    how="left"
)

df["has_signal_peptide"] = (
    df["has_signal_peptide"]
    .fillna(0)
    .astype(int)
)

signal_feature = df[
    ["has_signal_peptide"]
].values

X_struct = np.concatenate(
    [X_struct, signal_feature],
    axis=1
)
"""

# =========================
# 7. LOAD ESM2-650M
# =========================
print("\n🔹 LOADING ESM2-650M")

model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

model_esm = model_esm.to(device)
model_esm.eval()

batch_converter = alphabet.get_batch_converter()

ESM_REPR_LAYER = 33

print("✅ ESM2-650M loaded")

# =========================
# 8. LOAD FASTA
# =========================
print("\n🔹 LOADING FASTA")

seq_dict = {}

for record in SeqIO.parse("independent.fasta", "fasta"):
    seq_dict[record.id] = str(record.seq)

print("Sequences loaded:", len(seq_dict))

# =========================
# 9. ESM EMBEDDINGS
# =========================
def get_embedding(seq):

    seq = seq[:1022]

    batch = [("protein", seq)]

    _, _, tokens = batch_converter(batch)

    tokens = tokens.to(device)

    with torch.no_grad():

        output = model_esm(
            tokens,
            repr_layers=[ESM_REPR_LAYER]
        )

    emb = output["representations"][ESM_REPR_LAYER][0, 1:-1].mean(0)

    return emb.cpu().numpy()

print("\n🔹 GENERATING ESM EMBEDDINGS")

esm_rows = []

for pid in tqdm(protein_ids):

    if pid not in seq_dict:
        continue

    seq = seq_dict[pid]

    try:
        emb = get_embedding(seq)

        esm_rows.append((pid, emb))

    except Exception as e:

        print(f"Failed for {pid}: {e}")

        continue

# =========================
# 10. CREATE ESM DATAFRAME
# =========================
X_esm = np.array([x[1] for x in esm_rows])

ids = [x[0] for x in esm_rows]

esm_cols = [f"esm_{i}" for i in range(X_esm.shape[1])]

df_esm = pd.DataFrame(X_esm, columns=esm_cols)

df_esm["protein_id"] = ids

print("ESM shape:", X_esm.shape)

# =========================
# 11. FINAL MERGE
# =========================
print("\n🔹 FINAL MERGE")

struct_cols = [
    f"struct_{i}"
    for i in range(X_struct.shape[1])
]

df_struct = pd.DataFrame(
    X_struct,
    columns=struct_cols
)

df_struct["protein_id"] = protein_ids.values
df_struct["label"] = y

df_final = pd.merge(
    df_struct,
    df_esm,
    on="protein_id",
    how="left"
)

df_final = df_final.fillna(0)

print("Merged shape:", df_final.shape)

# =========================
# 12. FINAL FEATURES
# =========================
y_true = df_final["label"].values

X_struct_final = df_final[struct_cols].values

X_esm_final = df_final[esm_cols].values

X = np.concatenate(
    [X_struct_final, X_esm_final],
    axis=1
)

print("\nFinal feature dimension:", X.shape)

# ======================================
# SAFETY CHECK
# ======================================

EXPECTED_INPUT_DIM = saved.get("input_dim", 1315)
print(f"\nExpected input dim : {EXPECTED_INPUT_DIM}")

if X.shape[1] != EXPECTED_INPUT_DIM:

    raise ValueError(
        f"Feature mismatch! "
        f"Expected {EXPECTED_INPUT_DIM}, "
        f"got {X.shape[1]}"
    )

# =========================
# 13. CONTRASTIVE TRANSFORM
# =========================
print("\n🔹 CONTRASTIVE TRANSFORM")

with torch.no_grad():

    X_emb = model_cl(
        torch.tensor(
            X,
            dtype=torch.float32
        ).to(device)
    ).cpu().numpy()

X_final = np.concatenate(
    [X, X_emb],
    axis=1
)

print("Final transformed shape:", X_final.shape)

# ======================================
# FINAL DIMENSION CHECK
# ======================================

EXPECTED_FINAL_DIM = saved.get("final_dim", 1571)
print(f"Expected final dim : {EXPECTED_FINAL_DIM}")
if X_final.shape[1] != EXPECTED_FINAL_DIM:

    raise ValueError(
        f"Final feature mismatch! "
        f"Expected {EXPECTED_FINAL_DIM}, "
        f"got {X_final.shape[1]}"
    )

# =========================
# 14. PREDICTION
# =========================
print("\n🔹 PREDICTION")

y_prob = classifier.predict_proba(X_final)[:, 1]

print("\nProbability statistics")
print("Min :", y_prob.min())
print("Max :", y_prob.max())
print("Mean:", y_prob.mean())

# =========================
# 15. THRESHOLD OPTIMIZATION
# =========================
# =========================
# 15. THRESHOLD SELECTION
# =========================

# Biological discovery mode:
# prioritize recall over MCC

best_t = 0.64

print("\nUsing fixed threshold:", best_t)

# =========================
# 16. FINAL PREDICTIONS
# =========================
y_pred = (y_prob >= best_t).astype(int)

# =========================
# 17. RESULTS
# =========================
print("\n===================================")
print("FINAL INDEPENDENT RESULTS")
print("===================================")

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc = roc_auc_score(y_true, y_prob)
mcc = matthews_corrcoef(y_true, y_pred)

print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"ROC-AUC    : {roc:.4f}")
print(f"MCC        : {mcc:.4f}")

# =========================
# 18. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix")
print(cm)

# =========================
# 19. CLASSIFICATION REPORT
# =========================
print("\nClassification Report")

print(classification_report(y_true, y_pred))

# =========================
# 20. SAVE PREDICTIONS
# =========================
pred_df = pd.DataFrame({

    "protein_id": df_final["protein_id"],

    "true_label": y_true,

    "probability": y_prob,

    "prediction": y_pred
})

pred_df = pred_df.sort_values(
    by="probability",
    ascending=False
)

save_name = (
    f"PREDICTIONS_"
    f"{MODEL_FILE.replace('.pkl','')}.csv"
)

pred_df.to_csv(
    save_name,
    index=False
)

print("\n✅ Predictions saved:", save_name)

# =========================
# 21. TOP PREDICTIONS
# =========================
print("\nTop 20 probabilities")

print(
    pred_df[
        [
            "protein_id",
            "probability",
            "true_label"
        ]
    ].head(20)
)

# =========================
# 22. FULL PROBABILITIES
# =========================
pred_df.to_csv(
    "FULL_PROBABILITIES_v5.csv",
    index=False
)


# =========================
# 23. THRESHOLD SWEEP
# =========================
print("\nThreshold sweep")

print(
    f"{'t':>6}  "
    f"{'TP':>4}  "
    f"{'FP':>4}  "
    f"{'FN':>4}  "
    f"{'Prec':>6}  "
    f"{'Rec':>6}  "
    f"{'F1':>6}  "
    f"{'MCC':>6}"
)

print("-" * 60)

for t in np.arange(0.30, 0.98, 0.02):

    y_pred_t = (y_prob >= t).astype(int)

    tp = (
        (y_pred_t == 1) &
        (y_true == 1)
    ).sum()

    fp = (
        (y_pred_t == 1) &
        (y_true == 0)
    ).sum()

    fn = (
        (y_pred_t == 0) &
        (y_true == 1)
    ).sum()

    prec = precision_score(
        y_true,
        y_pred_t,
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred_t,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred_t,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred_t
    )

    print(
        f"{t:.2f}  "
        f"{tp:>4}  "
        f"{fp:>4}  "
        f"{fn:>4}  "
        f"{prec:>6.3f}  "
        f"{rec:>6.3f}  "
        f"{f1:>6.3f}  "
        f"{mcc:>6.3f}"
    )

print("\n🔥 PIPELINE COMPLETE")

pred_df.to_csv(
    "PREDICTIONS_XGB_MODEL_v5.csv",
    index=False
)

print("\n✅ Predictions saved: PREDICTIONS_XGB_MODEL_v5.csv")

# FALSE POSITIVES
fp_df = pred_df[
    (pred_df.true_label == 0) &
    (pred_df.prediction == 1)
]

fp_df = fp_df.sort_values(
    "probability",
    ascending=False
)

fp_df.to_csv(
    "TOP_FALSE_POSITIVES.csv",
    index=False
)

print("\nTop False Positives")
print(fp_df.head(30))

print("\n🔥 PIPELINE COMPLETE")
