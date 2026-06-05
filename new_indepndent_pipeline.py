
# =========================================================
# HYBRID INDEPENDENT PIPELINE (FINAL CORRECTED VERSION)
# =========================================================

import os
import pickle
import torch
import esm
import numpy as np
import pandas as pd

from Bio import SeqIO
from tqdm import tqdm

# =========================================================
# CHECKPOINT CONTROL
# =========================================================
# 1 = start from FASTA
# 2 = start from MMSEQS
# 3 = start from FEATURES
# 4 = start from MERGING
# 5 = start from MODEL LOADING
# 6 = start from ESM
# 7 = start from FINAL MERGE
# 8 = start from PREDICTION

START_FROM_STEP = 4

# =========================================================
# DEVICE
# =========================================================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

# =========================================================
# SETTINGS
# =========================================================
PDB_DIR = "independent_216"

UNIREF_DB = "/media/ravi/RAVI_DATA/Smriti/FOLDSEEK/mmseqs_db/uniref50"

# =========================================================
# STEP 1 : PDB → FASTA
# =========================================================
if START_FROM_STEP <= 1:

    print("\n🔹 STEP 1: PDB → FASTA")

    os.system(
        f"python independentpdb_to_fasta.py --pdb_dir {PDB_DIR}"
    )

# =========================================================
# STEP 2 : MMSEQS
# =========================================================
if START_FROM_STEP <= 2:

    print("\n🔹 STEP 2: MMSEQS")

    os.system("mkdir -p tmp_mmseqs")

    os.system(
        "mmseqs createdb independent.fasta indepDB"
    )

    os.system(
        f"""
        mmseqs search indepDB {UNIREF_DB} resultDB tmp_mmseqs \
        --threads 16 -s 7.5 --max-seqs 300
        """
    )

    os.system(
        """
        mmseqs convertalis \
        indepDB \
        /media/ravi/RAVI_DATA/Smriti/FOLDSEEK/mmseqs_db/uniref50 \
        resultDB \
        independent_alignment.tsv
        """
    )

# =========================================================
# STEP 3 : FEATURES
# =========================================================
if START_FROM_STEP <= 3:

    print("\n🔹 STEP 3: FEATURES")

    os.system("python independent_entropy_script.py")

    os.system("python independent_dssp_script.py")

    os.system("python independent_struct_features_script.py")

    os.system("python independent_residue_script.py")

    os.system("python independent_graph_script.py")

# =========================================================
# STEP 4 : BASIC FEATURE MERGE
# =========================================================
if START_FROM_STEP <= 4:

    print("\n🔹 STEP 4: BASIC FEATURE MERGE")

    df_struct = pd.read_csv(
        "independent_struct_features.csv"
    )

    df_entropy = pd.read_csv(
        "independent_entropy_features.csv"
    )

    df_graph = pd.read_csv(
        "independent_graph_features.csv"
    )

    # =====================================================
    # MERGE FEATURES
    # =====================================================
    df = df_graph.merge(
        df_entropy,
        on="protein_id",
        how="left"
    )

    df = df.merge(
        df_struct,
        on="protein_id",
        how="left"
    )

    df = df.fillna(0)

    print("Merged shape:", df.shape)

    df.to_csv(
        "independent_features.csv",
        index=False
    )

# =========================================================
# SIAMESE MODEL CLASS
# =========================================================
import torch.nn as nn

class Siamese(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128)
        )

    def forward(self, x):

        return self.net(x)

# =========================================================
# STEP 5 : LOAD MODEL
# =========================================================
if START_FROM_STEP <= 5:

    print("\n🔹 STEP 5: LOAD MODEL")

    saved = pickle.load(
        open("FINAL_HYBRID_2.0_MODEL.pkl", "rb")
    )

    model_cl = saved["model_cl"].to(device)

    model_cl.eval()

    scaler = saved["scaler"]

    clf = saved["classifier"]

    threshold = saved["threshold"]

    print("✅ MODEL LOADED")

# =========================================================
# STEP 6 : ESM EMBEDDINGS
# =========================================================
if START_FROM_STEP <= 6:

    print("\n🔹 STEP 6: ESM")

    model_esm, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

    model_esm = model_esm.to(device)

    model_esm.eval()

    batch_converter = alphabet.get_batch_converter()

    def get_embedding(seq):

        batch = [("protein", seq)]

        _, _, tokens = batch_converter(batch)

        tokens = tokens.to(device)

        with torch.no_grad():

            out = model_esm(
                tokens,
                repr_layers=[6]
            )

        return out["representations"][6][0,1:-1] \
            .mean(0) \
            .cpu() \
            .numpy()

    ids = []

    embs = []

    for record in tqdm(
        SeqIO.parse("independent.fasta", "fasta")
    ):

        try:

            emb = get_embedding(str(record.seq))

            ids.append(record.id)

            embs.append(emb)

        except Exception as e:

            print("Embedding failed:", record.id)
            print(e)

            continue

    df_esm = pd.DataFrame(embs)

    df_esm.columns = [
        f"esm_{i}"
        for i in range(df_esm.shape[1])
    ]

    df_esm["protein_id"] = ids

    df_esm.to_csv(
        "independent_esm_embeddings.csv",
        index=False
    )

    print("✅ ESM SAVED")

# =========================================================
# STEP 7 : FINAL MERGE
# =========================================================
if START_FROM_STEP <= 7:

    print("\n🔹 STEP 7: FINAL MERGE")

    # =====================================================
    # LOAD BASIC FEATURES
    # =====================================================
    df = pd.read_csv(
        "independent_features.csv"
    )

    # =====================================================
    # LOAD ESM
    # =====================================================
    df_esm = pd.read_csv(
        "independent_esm_embeddings.csv"
    )

    # =====================================================
    # MERGE ESM
    # =====================================================
    df = df.merge(
        df_esm,
        on="protein_id",
        how="left"
    )

    print("Shape AFTER ESM merge:", df.shape)

    # =====================================================
    # FIX DUPLICATE LABELS
    # =====================================================
    if "label_x" in df.columns:
        df["label"] = df["label_x"]

    elif "label_y" in df.columns:
        df["label"] = df["label_y"]

    drop_cols = []

    for col in ["label_x", "label_y"]:

        if col in df.columns:
            drop_cols.append(col)

    if len(drop_cols) > 0:

        df = df.drop(
            columns=drop_cols
        )

    # =====================================================
    # REMOVE DUPLICATE COLUMNS
    # =====================================================
    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    # =====================================================
    # REQUIRED STRUCT FEATURES
    # =====================================================
    REQUIRED_FEATURES = [

        'phi_mean',
        'phi_std',

        'psi_mean',
        'psi_std',

        'rsa_mean',
        'rsa_std',

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

        'mean_entropy',
        'variance_entropy',
        'std_entropy',
        'skew_entropy',

        'pct_conserved',
        'pct_variable',

        'charged_fraction',
        'polar_fraction',
        'hydrophobic_fraction',

        'contact_order',
        'graph_density',
        'total_sasa',
        'length',
        'contact_density'
    ]

    # =====================================================
    # ADD MISSING STRUCT FEATURES
    # =====================================================
    missing_cols = []

    for col in REQUIRED_FEATURES:

        if col not in df.columns:

            df[col] = 0

            missing_cols.append(col)

    print("\nMissing STRUCT columns added:")
    print(missing_cols)

    # =====================================================
    # KEEP ONLY REQUIRED STRUCT FEATURES
    # + ESM FEATURES
    # =====================================================
    esm_cols = [
        c for c in df.columns
        if c.startswith("esm_")
    ]

    final_columns = (
        ["protein_id", "label"]
        + REQUIRED_FEATURES
        + esm_cols
    )

    df = df[final_columns]

    print(
        "\nFinal aligned shape:",
        df.shape
    )

    print(
        "ESM feature count:",
        len(esm_cols)
    )

    print("\n✅ Feature alignment complete")

    # =====================================================
    # SAVE FINAL ALIGNED FEATURES
    # =====================================================
    df.to_csv(
        "independent_aligned_features.csv",
        index=False
    )

    print("\n✅ Saved aligned features")

    print(
        "\nFinal aligned shape:",
        df.shape
    )

    print(
        "ESM feature count:",
        len(esm_cols)
    )

    print("\n✅ Feature alignment complete")
# =========================================================
# STEP 8 : PREDICTION
# =========================================================
if START_FROM_STEP <= 8:

    print("\n🔹 STEP 8: PREDICTION")

    # =====================================================
    # LOAD FINAL ALIGNED DATA
    # =====================================================
    df = pd.read_csv(
        "independent_aligned_features.csv"
    )

    print("Input dataframe shape:")
    print(df.shape)

    # =====================================================
    # TRUE LABELS
    # =====================================================
    y_true = df["label"].values

    # =====================================================
    # REQUIRED STRUCT FEATURES
    # =====================================================
    struct_cols = [

        'phi_mean',
        'phi_std',

        'psi_mean',
        'psi_std',

        'rsa_mean',
        'rsa_std',

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

        'mean_entropy',
        'variance_entropy',
        'std_entropy',
        'skew_entropy',

        'pct_conserved',
        'pct_variable',

        'charged_fraction',
        'polar_fraction',
        'hydrophobic_fraction',

        'contact_order',
        'graph_density',
        'total_sasa',
        'length',
        'contact_density'
    ]

    # =====================================================
    # FIND ESM COLUMNS
    # =====================================================
    esm_cols = [

        c for c in df.columns

        if c.startswith("esm_")
    ]

    print("\nNumber of ESM features:")
    print(len(esm_cols))

    # =====================================================
    # STRUCT FEATURES
    # =====================================================
    X_struct = scaler.transform(
        df[struct_cols].values
    )

    print("\nStructural feature shape:")
    print(X_struct.shape)

    # =====================================================
    # ESM FEATURES
    # =====================================================
    X_esm = df[esm_cols].values

    print("\nESM feature shape:")
    print(X_esm.shape)

    # =====================================================
    # CONCATENATE
    # =====================================================
    X = np.concatenate(
        [X_struct, X_esm],
        axis=1
    )

    print("\nFinal Siamese input shape:")
    print(X.shape)

    # =====================================================
    # CONTRASTIVE EMBEDDINGS
    # =====================================================
    with torch.no_grad():

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        ).to(device)

        X_emb = model_cl(
            X_tensor
        ).cpu().numpy()

    print("\nContrastive embedding shape:")
    print(X_emb.shape)

    # =====================================================
    # FINAL CLASSIFIER INPUT
    # =====================================================
    X_final = np.concatenate(
        [X, X_emb],
        axis=1
    )

    print("\nFinal classifier input shape:")
    print(X_final.shape)

    # =====================================================
    # PREDICTION
    # =====================================================
    y_prob = clf.predict_proba(
        X_final
    )[:, 1]

    # =====================================================
    # USE TRAINED THRESHOLD
    # =====================================================
    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================
    df_out = pd.DataFrame({

        "protein_id": df["protein_id"].values,

        "probability": y_prob,

        "prediction": y_pred,

        "true_label": y_true
    })

    df_out.to_csv(
        "independent_predictions.csv",
        index=False
    )

    print("\n✅ Predictions saved")

    # =====================================================
    # PROBABILITY STATS
    # =====================================================
    print("\nProbability statistics")

    print("Min :", y_prob.min())
    print("Max :", y_prob.max())
    print("Mean:", y_prob.mean())

    print("\nTop 20 probabilities")

    top_idx = np.argsort(y_prob)[-20:]

    for i in reversed(top_idx):

        print(
            df_out["protein_id"].iloc[i],
            y_prob[i],
            y_true[i]
        )
# =========================================================
# STEP 9 : EVALUATION
# =========================================================
print("\n🔹 STEP 9: EVALUATION")

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

# =========================================================
# METRICS
# =========================================================
acc = accuracy_score(y_true, y_pred)

prec = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

rec = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_prob
)

mcc = matthews_corrcoef(
    y_true,
    y_pred
)

cm = confusion_matrix(
    y_true,
    y_pred
)

# =========================================================
# PRINT RESULTS
# =========================================================
print("\n===================================")
print("FINAL INDEPENDENT TEST RESULTS")
print("===================================")

print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"ROC-AUC    : {auc:.4f}")
print(f"MCC        : {mcc:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        y_true,
        y_pred
    )
)

print("\n🔥 PIPELINE COMPLETE")
