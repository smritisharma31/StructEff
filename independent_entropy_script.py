import pandas as pd
import numpy as np
from scipy.stats import skew

# =========================
# FILE
# =========================
ALIGNMENT_FILE = "independent_alignment.tsv"

# =========================
# LOAD MMSEQS ALIGNMENT
# =========================
def load_alignment(file_path):

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None
    )

    # MMseqs convertalis columns
    # query,target,pident,alnlen,mismatch,
    # gapopen,qstart,qend,tstart,tend,
    # evalue,bits

    if df.shape[1] < 12:
        raise ValueError(
            "Alignment TSV does not contain enough columns."
        )

    df = df.iloc[:, :12]

    df.columns = [
        "query",
        "target",
        "pident",
        "alnlen",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "tstart",
        "tend",
        "evalue",
        "bits"
    ]

    return df

# =========================
# ENTROPY FEATURES
# =========================
def compute_features(group):

    identities = (
        group["pident"]
        .astype(float)
        .values
    )

    # normalize
    probs = identities / (
        identities.sum() + 1e-9
    )

    entropy_vals = -probs * np.log2(
        probs + 1e-9
    )

    return {

        "mean_entropy":
            np.mean(entropy_vals),

        "variance_entropy":
            np.var(entropy_vals),

        "std_entropy":
            np.std(entropy_vals),

        "skew_entropy":
            skew(entropy_vals),

        "pct_conserved":
            np.sum(identities > 70)
            / len(identities),

        "pct_variable":
            np.sum(identities < 30)
            / len(identities)
    }

# =========================
# PROCESS
# =========================
def process(df):

    grouped = df.groupby("query")

    results = []

    for protein, group in grouped:

        feats = compute_features(group)

        feats["protein_id"] = protein

        results.append(feats)

    return pd.DataFrame(results)

# =========================
# MAIN
# =========================
print("🚀 Computing entropy features...")

df = load_alignment(ALIGNMENT_FILE)

final_df = process(df)

final_df.to_csv(
    "independent_entropy_features.csv",
    index=False
)

print("\n✅ DONE")
print(final_df.shape)
print(final_df.head())
