"""
StructEff — Fungal Effector Prediction Tool
============================================
Usage:
    python predict.py --fasta input.fasta --outdir results/
    python predict.py --fasta input.fasta --outdir results/ --mmseqs-db /path/to/uniref50
    python predict.py --fasta input.fasta --outdir results/ --no-mmseqs
"""

import os
import sys
import gc
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import esm
from Bio import SeqIO
from tqdm import tqdm

from structeff.fasta_utils       import prepare_fasta
from structeff.esmfold           import fold_fasta
from structeff.features.residue  import extract_residue_features
from structeff.features.graph    import aggregate_to_protein
from structeff.features.struct   import extract_struct_features
from structeff.features.entropy  import (
    run_mmseqs2,
    extract_entropy_features,
    get_default_entropy_features
)
from structeff.features.bio      import extract_bio_features
from structeff.merge             import merge_all_features, TRAINING_COLUMNS
from structeff.siamese import Siamese

MODEL_PATH = os.path.join(os.path.dirname(__file__), "XGB_MODEL_v5_hardneg.pkl")
ESM_LAYER  = 33
THRESHOLD  = 0.2


def parse_args():
    parser = argparse.ArgumentParser(
        description="StructEff: Fungal Effector Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --fasta proteins.fasta --outdir results/
  python predict.py --fasta proteins.fasta --outdir results/ --no-mmseqs
  python predict.py --fasta proteins.fasta --outdir results/ --mmseqs-db /path/to/uniref50
        """
    )
    parser.add_argument("--fasta",     required=True,               help="Input FASTA file")
    parser.add_argument("--outdir",    default="results",           help="Output directory")
    parser.add_argument("--mmseqs-db", default=None,                help="Path to MMseqs2 UniRef50 database")
    parser.add_argument("--threads",   default=8,      type=int,    help="Threads for MMseqs2")
    parser.add_argument("--threshold", default=THRESHOLD, type=float, help="Prediction threshold (default: 0.2)")
    parser.add_argument("--no-mmseqs", action="store_true",         help="Skip MMseqs2 entropy features")
    return parser.parse_args()


def generate_esm_embeddings(fasta_file, device):
    print("\n🔹 Loading ESM2-650M...")
    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.to(device)
    model_esm.eval()
    batch_converter = alphabet.get_batch_converter()
    print("✅ ESM2-650M loaded")

    records  = list(SeqIO.parse(fasta_file, "fasta"))
    esm_rows = []

    print(f"\n🔹 Generating ESM2 embeddings for {len(records)} sequences...")
    for record in tqdm(records):
        pid = record.id.strip()
        # Input FASTA was already sanitized by prepare_fasta at ingestion, so
        # sequences are tokenizer-safe here. The 1022 cap is ESM2's hard
        # architectural limit (1024 positions minus BOS/EOS) and must stay.
        seq = str(record.seq)[:1022]
        batch = [("protein", seq)]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            output = model_esm(tokens, repr_layers=[ESM_LAYER])
        emb = output["representations"][ESM_LAYER][0, 1:-1].mean(0).cpu().numpy()
        esm_rows.append((pid, emb))

    del model_esm
    gc.collect()

    X_esm    = np.array([x[1] for x in esm_rows])
    ids      = [x[0] for x in esm_rows]
    esm_cols = [f"esm_{i}" for i in range(X_esm.shape[1])]
    df_esm   = pd.DataFrame(X_esm, columns=esm_cols)
    df_esm["protein_id"] = ids

    print(f"  ESM shape: {X_esm.shape}")
    return df_esm, esm_cols


def load_model(model_path):
    print(f"\n🔹 Loading StructEff model...")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Make sure XGB_MODEL_v5_hardneg.pkl is in the same directory as predict.py")
        sys.exit(1)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    print("✅ Model loaded")
    print(f"   Version  : {bundle.get('version', 'unknown')}")
    print(f"   Threshold: {bundle['threshold']}")
    return bundle


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print("  StructEff — Fungal Effector Prediction Tool v1.0")
    print(f"{'='*55}")
    print(f"Device   : {device}")
    print(f"Input    : {args.fasta}")
    print(f"Output   : {args.outdir}")
    print(f"Threshold: {args.threshold}")
    print(f"MMseqs2  : {'disabled' if args.no_mmseqs else args.mmseqs_db or 'not provided'}")

    os.makedirs(args.outdir, exist_ok=True)
    pdb_dir   = os.path.join(args.outdir, "pdb_structures")
    mmseq_dir = os.path.join(args.outdir, "mmseqs_results")

    # Load model
    bundle     = load_model(MODEL_PATH)
    model_cl   = bundle["model_cl"]
    scaler     = bundle["scaler"]
    classifier = bundle["classifier"]
    threshold  = args.threshold
    print(f"   Active threshold: {threshold} (bundle stored: {bundle['threshold']})")
    model_cl.eval()

    # Step 0: parse + sanitize input FASTA once. Every downstream step reads
    # this cleaned file, so character cleaning happens in exactly one place.
    print(f"\n{'='*55}")
    print("STEP 0: Parsing and sanitizing input FASTA")
    print(f"{'='*55}")
    clean_fasta = os.path.join(args.outdir, "cleaned_input.fasta")
    report = prepare_fasta(args.fasta, clean_fasta)
    print(f"  Read     : {report['n_in']} sequences")
    print(f"  Kept     : {report['n_out']}")
    print(f"  Modified : {len(report['modified'])} (non-standard chars cleaned)")
    print(f"  Dropped  : {len(report['dropped'])}")
    if report["dropped"]:
        drop_path = os.path.join(args.outdir, "dropped_sequences.txt")
        with open(drop_path, "w") as fh:
            for pid, reason in report["dropped"]:
                fh.write(f"{pid}\t{reason}\n")
        print(f"  Dropped IDs written to: {drop_path}")
    if report["n_out"] == 0:
        print("\n[FATAL] No usable sequences after cleaning. Check the input FASTA.")
        sys.exit(1)

    # Get protein IDs (from the cleaned FASTA, so they match every later step)
    records     = list(SeqIO.parse(clean_fasta, "fasta"))
    protein_ids = [r.id.strip() for r in records]
    print(f"\n📋 Proteins to predict: {len(protein_ids)}")

    # Step 1: ESMFold
    print(f"\n{'='*55}")
    print("STEP 1: Generating 3D structures with ESMFold (local)")
    print(f"{'='*55}")
    fold_fasta(clean_fasta, pdb_dir)

    # Step 2: Residue features
    print(f"\n{'='*55}")
    print("STEP 2: Extracting residue-level features")
    print(f"{'='*55}")
    df_residue = extract_residue_features(pdb_dir)
    if df_residue is None or df_residue.empty or "rsa" not in df_residue.columns:
        print("\n[FATAL] No parseable residue features were produced.")
        print("        All structures failed to fold or DSSP could not read them.")
        print("        Check results/pdb_structures/ — PDBs must start with a CRYST1 record.")
        sys.exit(1)

    # Step 3: Aggregate
    print(f"\n{'='*55}")
    print("STEP 3: Aggregating to protein level")
    print(f"{'='*55}")
    df_graph = aggregate_to_protein(df_residue)

    # Step 4: Structural features
    print(f"\n{'='*55}")
    print("STEP 4: Extracting structural features")
    print(f"{'='*55}")
    df_struct = extract_struct_features(pdb_dir)

    # Step 5: Entropy features
    print(f"\n{'='*55}")
    print("STEP 5: Conservation features")
    print(f"{'='*55}")
    if args.no_mmseqs or args.mmseqs_db is None:
        df_entropy = get_default_entropy_features(protein_ids)
    else:
        aln_file   = run_mmseqs2(clean_fasta, args.mmseqs_db, mmseq_dir, args.threads)
        df_entropy = extract_entropy_features(aln_file, protein_ids)

    # Step 6: Bio features
    print(f"\n{'='*55}")
    print("STEP 6: Extracting biological features")
    print(f"{'='*55}")
    df_bio = extract_bio_features(pdb_dir)

    # Step 7: Merge
    print(f"\n{'='*55}")
    print("STEP 7: Merging all features")
    print(f"{'='*55}")
    df_features = merge_all_features(df_graph, df_struct, df_entropy, df_bio, protein_ids)

    # Scale structural features
    X_struct    = np.nan_to_num(df_features[TRAINING_COLUMNS].values)
    X_struct    = scaler.transform(X_struct)
    struct_cols = [f"struct_{i}" for i in range(X_struct.shape[1])]

    # Step 8: ESM2 embeddings
    print(f"\n{'='*55}")
    print("STEP 8: Generating ESM2 embeddings")
    print(f"{'='*55}")
    df_esm, esm_cols = generate_esm_embeddings(clean_fasta, device)

    # Merge struct + ESM
    df_struct_df = pd.DataFrame(X_struct, columns=struct_cols)
    df_struct_df["protein_id"] = df_features["protein_id"].values

    df_merged    = df_struct_df.merge(df_esm, on="protein_id", how="left").dropna()
    X            = np.concatenate([
        df_merged[struct_cols].values,
        df_merged[esm_cols].values
    ], axis=1)

    print(f"\n  Final feature dimension: {X.shape}")

    # Safety check: feature count must match what the model was trained on.
    # Catches feature-pipeline drift before it reaches the classifier and
    # produces silently-wrong probabilities.
    expected_input = bundle.get("input_dim", 1315)
    if X.shape[1] != expected_input:
        print(f"\n[FATAL] Feature dimension mismatch: built {X.shape[1]}, "
              f"model expects {expected_input}.")
        print("        A feature step changed shape — predictions would be invalid.")
        sys.exit(1)

    # Contrastive transform
    print("\n🔹 Applying contrastive transform...")
    model_cl = model_cl.to(device)
    model_cl.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        X_emb    = model_cl(X_tensor).cpu().numpy()

    X_final = np.concatenate([X, X_emb], axis=1)
    print(f"  Transformed shape: {X_final.shape}")

    expected_final = bundle.get("final_dim", 1571)
    if X_final.shape[1] != expected_final:
        print(f"\n[FATAL] Transformed dimension mismatch: got {X_final.shape[1]}, "
              f"model expects {expected_final}.")
        sys.exit(1)

    # Predict
    print(f"\n🔹 Predicting with threshold={threshold}...")
    y_prob = classifier.predict_proba(X_final)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Save results
    df_results = pd.DataFrame({
        "protein_id" : df_merged["protein_id"].values,
        "probability": np.round(y_prob, 4),
        "prediction" : y_pred,
        "label"      : ["Effector" if p == 1 else "Non-effector" for p in y_pred]
    })

    output_path = os.path.join(args.outdir, "structeff_predictions.csv")
    df_results.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'='*55}")
    print("  STRUCTEFF PREDICTION COMPLETE")
    print(f"{'='*55}")
    print(f"Total proteins          : {len(df_results)}")
    print(f"Predicted effectors     : {y_pred.sum()}")
    print(f"Predicted non-effectors : {(y_pred==0).sum()}")
    print(f"\n✅ Results saved: {output_path}")

    print("\nTop effector predictions:")
    top = df_results[df_results["prediction"]==1].sort_values(
        "probability", ascending=False
    ).head(10)
    print(top[["protein_id","probability","label"]].to_string(index=False))

    print("\n🔥 PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
