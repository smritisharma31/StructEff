"""
structeff/features/bio.py
Computes biological features from PDB files.
(cysteine count, cysteine fraction, has_4plus_cys)
"""

import os
import pandas as pd
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)


def compute_bio_features(pdb_path, protein_id):
    try:
        structure      = parser.get_structure(protein_id, pdb_path)
        residues       = list(structure.get_residues())
        total_res      = len(residues)
        cysteine_count = sum(1 for r in residues if r.get_resname() == "CYS")
        cysteine_frac  = cysteine_count / total_res if total_res > 0 else 0
        has_4plus_cys  = 1 if cysteine_count >= 4 else 0

        return {
            "protein_id"       : protein_id,
            "cysteine_count"   : cysteine_count,
            "cysteine_fraction": cysteine_frac,
            "has_4plus_cys"    : has_4plus_cys,
        }

    except Exception as e:
        print(f"[ERROR] {protein_id}: {e}")
        return {
            "protein_id"       : protein_id,
            "cysteine_count"   : 0,
            "cysteine_fraction": 0.0,
            "has_4plus_cys"    : 0,
        }


def extract_bio_features(pdb_dir):
    data      = []
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    print(f"\n🔹 Extracting biological features from {len(pdb_files)} PDB files...")

    for file in pdb_files:
        protein_id = file.replace(".pdb", "")
        pdb_path   = os.path.join(pdb_dir, file)
        feats      = compute_bio_features(pdb_path, protein_id)
        data.append(feats)

    return pd.DataFrame(data)
