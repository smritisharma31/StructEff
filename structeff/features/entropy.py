"""
structeff/features/entropy.py
Computes sequence conservation features from MMseqs2 alignment.
"""

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import skew


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


def extract_entropy_features(aln_file):
    print("\n🔹 Computing entropy features...")
    df = pd.read_csv(aln_file, sep="\t", header=None).iloc[:, :12]
    df.columns = [
        "query","target","pident","alnlen",
        "mismatch","gapopen","qstart","qend",
        "tstart","tend","evalue","bits"
    ]
    results = []
    for protein, group in df.groupby("query"):
        feats = compute_entropy_features(group)
        feats["protein_id"] = protein
        results.append(feats)
    df_out = pd.DataFrame(results)
    print(f"  Proteins with entropy features: {len(df_out)}")
    return df_out


def get_default_entropy_features(protein_ids):
    print("\n⚠️  MMseqs2 not used — filling entropy features with zeros")
    rows = []
    for pid in protein_ids:
        rows.append({
            "protein_id"      : pid,
            "mean_entropy"    : 0.0,
            "variance_entropy": 0.0,
            "std_entropy"     : 0.0,
            "skew_entropy"    : 0.0,
            "pct_conserved"   : 0.0,
            "pct_variable"    : 0.0,
        })
    return pd.DataFrame(rows)
