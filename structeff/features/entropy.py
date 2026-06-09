"""
structeff/features/entropy.py
Computes sequence conservation features from MMseqs2 alignment.
"""

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import skew


ENTROPY_COLUMNS = [
    "mean_entropy", "variance_entropy", "std_entropy",
    "skew_entropy", "pct_conserved", "pct_variable",
]


def run_mmseqs2(fasta_file, db_path, output_dir, threads=8):
    os.makedirs(output_dir, exist_ok=True)
    query_db  = os.path.join(output_dir, "queryDB")
    result_db = os.path.join(output_dir, "resultDB")
    tmp_dir   = os.path.join(output_dir, "tmp")
    aln_file  = os.path.join(output_dir, "alignment.tsv")

    print("\n🔹 Running MMseqs2 search...")

    subprocess.run(["mmseqs", "createdb", fasta_file, query_db], check=True)
    subprocess.run([
        "mmseqs", "search",
        query_db, db_path, result_db, tmp_dir,
        "--threads", str(threads), "-s", "7.5", "--max-seqs", "300"
    ], check=True)
    subprocess.run([
        "mmseqs", "convertalis",
        query_db, db_path, result_db, aln_file
    ], check=True)

    print(f"  Alignment saved: {aln_file}")
    return aln_file


def compute_entropy_features(group):
    identities   = group["pident"].astype(float).values
    probs        = identities / (identities.sum() + 1e-9)
    entropy_vals = -probs * np.log2(probs + 1e-9)
    return {
        "mean_entropy"    : np.mean(entropy_vals),
        "variance_entropy": np.var(entropy_vals),
        "std_entropy"     : np.std(entropy_vals),
        "skew_entropy"    : skew(entropy_vals),
        "pct_conserved"   : np.sum(identities > 70) / len(identities),
        "pct_variable"    : np.sum(identities < 30) / len(identities),
    }


def _zero_feats(protein_id):
    feats = {c: 0.0 for c in ENTROPY_COLUMNS}
    feats["protein_id"] = protein_id
    return feats


def extract_entropy_features(aln_file, protein_ids=None):
    """
    Parse the MMseqs2 alignment and compute per-protein conservation features.

    Robust to the common no-homolog case: if the alignment file is missing or
    empty (0 hits passed the e-value threshold), every query gets zero-filled
    entropy features instead of crashing — identical to the --no-mmseqs path.
    Proteins present in `protein_ids` but absent from the alignment are also
    zero-filled, so the output always covers every query.
    """
    print("\n🔹 Computing entropy features...")

    # Empty or missing alignment -> all-zero features for every query.
    if (not os.path.exists(aln_file)) or os.path.getsize(aln_file) == 0:
        print("  No alignment hits — filling entropy features with zeros")
        ids = list(protein_ids) if protein_ids is not None else []
        df_out = pd.DataFrame([_zero_feats(pid) for pid in ids])
        print(f"  Proteins with entropy features: {len(df_out)}")
        return df_out

    df = pd.read_csv(aln_file, sep="\t", header=None).iloc[:, :12]
    df.columns = [
        "query","target","pident","alnlen",
        "mismatch","gapopen","qstart","qend",
        "tstart","tend","evalue","bits"
    ]

    results = []
    hit_ids = set()
    for protein, group in df.groupby("query"):
        feats = compute_entropy_features(group)
        feats["protein_id"] = protein
        results.append(feats)
        hit_ids.add(protein)

    # Zero-fill any query that had no hits so the feature table is complete.
    if protein_ids is not None:
        for pid in protein_ids:
            if pid not in hit_ids:
                results.append(_zero_feats(pid))

    df_out = pd.DataFrame(results)
    print(f"  Proteins with entropy features: {len(df_out)}")
    return df_out


def get_default_entropy_features(protein_ids):
    print("\n⚠️  MMseqs2 not used — filling entropy features with zeros")
    return pd.DataFrame([_zero_feats(pid) for pid in protein_ids])
