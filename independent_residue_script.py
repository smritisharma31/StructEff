# Updated `independent_residue_script.py`
import os
import numpy as np
import pandas as pd
import networkx as nx

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

# ==================================
# SETTINGS
# ==================================
PDB_DIR = "independent_216"
OUTPUT_FILE = "independent_residue_features.csv"
DSSP_EXE = "mkdssp"

parser = PDBParser(QUIET=True)
all_rows = []

# ==================================
# CONTACT GRAPH
# ==================================
def compute_contact_graph(structure, cutoff=8.0):
    """
    Create residue contact graph using CA atoms.
    """

    model = structure[0]

    residues = [
        res for res in model.get_residues()
        if 'CA' in res
    ]

    G = nx.Graph()

    for i, res1 in enumerate(residues):

        G.add_node(i)

        coord1 = res1['CA'].get_coord()

        for j, res2 in enumerate(residues[i + 1:], start=i + 1):

            coord2 = res2['CA'].get_coord()

            distance = np.linalg.norm(coord1 - coord2)

            if distance <= cutoff:
                G.add_edge(i, j)

    return G, residues

# ==================================
# PROCESS SINGLE PDB
# ==================================
def process_pdb(file):

    protein_id = file.replace(".pdb", "")
    filepath = os.path.join(PDB_DIR, file)

    # ==============================
    # LABEL ASSIGNMENT
    # ==============================
    if file.startswith("Fung"):
        label = 1

    elif file.startswith("Non"):
        label = 0

    else:
        print(f"[SKIPPED] Unknown label format: {file}")
        return

    print(f"Processing: {protein_id}")

    try:

        # ==============================
        # LOAD STRUCTURE
        # ==============================
        structure = parser.get_structure(protein_id, filepath)
        model = structure[0]

        # ==============================
        # RUN DSSP
        # ==============================
        dssp = DSSP(
            model,
            filepath,
            dssp=DSSP_EXE,
            file_type="PDB"
        )

        # ==============================
        # BUILD CONTACT GRAPH
        # ==============================
        G, residues = compute_contact_graph(structure)

        # Graph centrality
        centrality = nx.betweenness_centrality(G)

        # ==============================
        # RESIDUE LOOP
        # ==============================
        for idx, res in enumerate(residues):

            chain_id = res.get_parent().id
            res_id = res.get_id()

            dssp_key = (chain_id, res_id)

            # Skip if residue absent in DSSP
            if dssp_key not in dssp:
                continue

            dssp_data = dssp[dssp_key]

            # DSSP features
            ss = dssp_data[2]
            rsa = dssp_data[3]
            phi = dssp_data[4]
            psi = dssp_data[5]

            # ==============================
            # GRAPH FEATURES
            # ==============================
            contact_number = len(list(G.neighbors(idx)))

            long_range_contacts = sum(
                1 for n in G.neighbors(idx)
                if abs(n - idx) > 12
            )

            # ==============================
            # CORE / SURFACE
            # ==============================
            core = 1 if rsa < 0.25 else 0

            # ==============================
            # pLDDT / B-FACTOR
            # ==============================
            mean_b = np.mean([
                atom.get_bfactor()
                for atom in res
            ])

            # Higher pLDDT usually means more ordered
            disorder = 1 if mean_b < 70 else 0

            # ==============================
            # SECONDARY STRUCTURE FLAGS
            # ==============================
            is_helix = int(ss in ["H", "G", "I"])
            is_sheet = int(ss in ["E", "B"])
            is_coil = int(ss not in ["H", "G", "I", "E", "B"])

            # ==============================
            # STORE FEATURES
            # ==============================
            all_rows.append({

                "protein_id": protein_id,
                "chain": chain_id,
                "res_seq": res_id[1],
                "res_index": idx,

                # DSSP
                "secondary_structure": ss,
                "rsa": rsa,
                "phi": phi,
                "psi": psi,

                # Graph
                "contact_number": contact_number,
                "long_range_contacts": long_range_contacts,
                "betweenness_centrality": centrality.get(idx, 0),

                # Structural
                "core": core,
                "plddt_mean": mean_b,
                "disorder_proxy": disorder,

                # One-hot SS
                "is_helix": is_helix,
                "is_sheet": is_sheet,
                "is_coil": is_coil,

                # Label
                "label": label
            })

    except Exception as e:
        print(f"[ERROR] {protein_id}: {e}")

# ==================================
# MAIN
# ==================================
if __name__ == "__main__":

    pdb_files = [
        f for f in os.listdir(PDB_DIR)
        if f.endswith(".pdb")
    ]

    print(f"Total PDB files found: {len(pdb_files)}")

    for file in pdb_files:
        process_pdb(file)

    # ==============================
    # SAVE CSV
    # ==============================
    df = pd.DataFrame(all_rows)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ RESIDUE FEATURES EXTRACTION DONE")
    print(f"Final shape: {df.shape}")
    print(f"Saved to: {OUTPUT_FILE}")


