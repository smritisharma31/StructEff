import pandas as pd
import numpy as np

print("🚀 Loading residue-level features...")

df = pd.read_csv(
    "independent_residue_features.csv",
    low_memory=False
)

print("Input shape:", df.shape)

# =========================
# CLEAN
# =========================
df = df.dropna(subset=["rsa", "phi", "psi"])

# =========================
# TYPE OPTIMIZATION
# =========================
float_cols = [
    "rsa",
    "phi",
    "psi",
    "contact_number",
    "long_range_contacts",
    "betweenness_centrality",
    "plddt_mean"
]

for col in float_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).astype(np.float32)

df["label"] = df["label"].astype(np.int8)

# =========================
# AGGREGATION
# =========================
print("🔄 Aggregating...")

agg = df.groupby(
    "protein_id",
    sort=False
).agg({

    "phi": ["mean","std"],

    "psi": ["mean","std"],

    "rsa": ["mean","std","min","max"],

    "contact_number": ["mean","std","max"],

    "long_range_contacts": ["mean","sum"],

    "betweenness_centrality": ["mean","max"],

    "core": ["mean"],

    "plddt_mean": ["mean","std"],

    "disorder_proxy": ["mean"],

    "is_helix": ["mean"],

    "is_sheet": ["mean"],

    "is_coil": ["mean"],

    "label": "first"
})

# =========================
# FLATTEN
# =========================
agg.columns = [
    "_".join(col).strip()
    for col in agg.columns
]

agg = agg.rename(
    columns={"label_first": "label"}
)

agg = agg.reset_index()

# =========================
# DERIVED FEATURE
# =========================
agg["long_range_norm"] = (
    agg["long_range_contacts_sum"] /
    (agg["contact_number_mean"] + 1e-6)
)

# =========================
# CLEAN
# =========================
agg = agg.replace(
    [np.inf, -np.inf],
    np.nan
)

agg = agg.fillna(0)

# =========================
# SAVE
# =========================
agg.to_csv(
    "independent_graph_features.csv",
    index=False
)

print("\n✅ GRAPH FEATURES DONE")
print(agg.shape)
