"""
structeff/merge.py
Merges all feature DataFrames into final aligned feature matrix.
"""

import numpy as np
import pandas as pd


TRAINING_COLUMNS = [
    'phi_mean', 'phi_std', 'psi_mean', 'psi_std',
    'rsa_mean', 'rsa_std',
    'contact_number_mean', 'contact_number_std',
    'long_range_contacts_mean',
    'betweenness_centrality_mean',
    'core_mean',
    'plddt_mean_mean', 'plddt_mean_std',
    'disorder_proxy_mean',
    'is_helix_mean', 'is_sheet_mean', 'is_coil_mean',
    'long_range_norm',
    'charged_fraction', 'polar_fraction',
    'hydrophobic_fraction',
    'contact_order',
    'graph_density',
    'total_sasa',
    'length',
    'contact_density',
    'mean_entropy', 'variance_entropy',
    'std_entropy', 'skew_entropy',
    'pct_conserved', 'pct_variable',
    'cysteine_count', 'cysteine_fraction',
    'has_4plus_cys'
]


def merge_all_features(df_graph, df_struct, df_entropy, df_bio, protein_ids):
    print("\n🔹 Merging all features...")

    df_base = pd.DataFrame({"protein_id": protein_ids})
    df      = df_base.merge(df_graph,   on="protein_id", how="left")
    df      = df.merge(df_struct,  on="protein_id", how="left")
    df      = df.merge(df_entropy, on="protein_id", how="left")
    df      = df.merge(df_bio,     on="protein_id", how="left")

    print(f"  Merged shape: {df.shape}")

    missing = [c for c in TRAINING_COLUMNS if c not in df.columns]
    if missing:
        print(f"  ⚠️  Missing columns (filling with 0): {missing}")
        for col in missing:
            df[col] = 0.0

    df_final = df[["protein_id"] + TRAINING_COLUMNS].copy()
    df_final = df_final.fillna(0.0)

    print(f"  Final shape: {df_final.shape}")
    return df_final
