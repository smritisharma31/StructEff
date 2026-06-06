"""
structeff/features/graph.py
Aggregates residue-level features to protein-level.
"""

import numpy as np
import pandas as pd


def aggregate_to_protein(df_residue):
    print("\n🔹 Aggregating residue features to protein level...")

    df = df_residue.copy()
    df = df.dropna(subset=["rsa", "phi", "psi"])

    float_cols = [
        "rsa", "phi", "psi",
        "contact_number", "long_range_contacts",
        "betweenness_centrality", "plddt_mean"
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    agg = df.groupby("protein_id", sort=False).agg({
        "phi"                   : ["mean", "std"],
        "psi"                   : ["mean", "std"],
        "rsa"                   : ["mean", "std", "min", "max"],
        "contact_number"        : ["mean", "std", "max"],
        "long_range_contacts"   : ["mean", "sum"],
        "betweenness_centrality": ["mean", "max"],
        "core"                  : ["mean"],
        "plddt_mean"            : ["mean", "std"],
        "disorder_proxy"        : ["mean"],
        "is_helix"              : ["mean"],
        "is_sheet"              : ["mean"],
        "is_coil"               : ["mean"],
    })

    agg.columns = ["_".join(col).strip() for col in agg.columns]
    agg         = agg.reset_index()

    agg["long_range_norm"] = (
        agg["long_range_contacts_sum"] /
        (agg["contact_number_mean"] + 1e-6)
    )

    agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"  Proteins aggregated: {len(agg)}")
    return agg
