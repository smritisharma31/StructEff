"""
structeff/features/residue.py
Extracts residue-level structural features from PDB files.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

DSSP_EXE = "mkdssp"
parser   = PDBParser(QUIET=True)


def compute_contact_graph(structure, cutoff=8.0):
    model    = structure[0]
    residues = [res for res in model.get_residues() if 'CA' in res]
    G        = nx.Graph()

    for i, res1 in enumerate(residues):
        G.add_node(i)
        coord1 = res1['CA'].get_coord()
        for j, res2 in enumerate(residues[i+1:], start=i+1):
            coord2 = res2['CA'].get_coord()
            if np.linalg.norm(coord1 - coord2) <= cutoff:
                G.add_edge(i, j)

    return G, residues


def process_pdb(pdb_path, protein_id):
    rows = []
    try:
        structure   = parser.get_structure(protein_id, pdb_path)
        model       = structure[0]
        dssp        = DSSP(model, pdb_path, dssp=DSSP_EXE, file_type="PDB")
        G, residues = compute_contact_graph(structure)
        centrality  = nx.betweenness_centrality(G)

        for idx, res in enumerate(residues):
            chain_id = res.get_parent().id
            res_id   = res.get_id()
            dssp_key = (chain_id, res_id)

            if dssp_key not in dssp:
                continue

            dssp_data = dssp[dssp_key]
            ss  = dssp_data[2]
            rsa = dssp_data[3]
            phi = dssp_data[4]
            psi = dssp_data[5]

            contact_number      = len(list(G.neighbors(idx)))
            long_range_contacts = sum(1 for n in G.neighbors(idx) if abs(n - idx) > 12)
            core                = 1 if rsa < 0.25 else 0
            mean_b              = np.mean([atom.get_bfactor() for atom in res])
            disorder            = 1 if mean_b < 70 else 0

            rows.append({
                "protein_id"            : protein_id,
                "chain"                 : chain_id,
                "res_seq"               : res_id[1],
                "res_index"             : idx,
                "secondary_structure"   : ss,
                "rsa"                   : rsa,
                "phi"                   : phi,
                "psi"                   : psi,
                "contact_number"        : contact_number,
                "long_range_contacts"   : long_range_contacts,
                "betweenness_centrality": centrality.get(idx, 0),
                "core"                  : core,
                "plddt_mean"            : mean_b,
                "disorder_proxy"        : disorder,
                "is_helix"              : int(ss in ["H","G","I"]),
                "is_sheet"              : int(ss in ["E","B"]),
                "is_coil"               : int(ss not in ["H","G","I","E","B"]),
            })

    except Exception as e:
        print(f"[ERROR] {protein_id}: {e}")

    return rows


def extract_residue_features(pdb_dir):
    all_rows  = []
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    print(f"\n🔹 Extracting residue features from {len(pdb_files)} PDB files...")

    for file in pdb_files:
        protein_id = file.replace(".pdb", "")
        pdb_path   = os.path.join(pdb_dir, file)
        rows       = process_pdb(pdb_path, protein_id)
        all_rows.extend(rows)
        print(f"  [OK] {protein_id} — {len(rows)} residues")

    return pd.DataFrame(all_rows)
