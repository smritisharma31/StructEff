import os
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser
from collections import Counter

parser = PDBParser(QUIET=True)

# =========================
# SETTINGS
# =========================
PDB_DIR = "independent_216"

OUTPUT_FILE = "independent_struct_features.csv"

# =========================
# BASIC FEATURES
# =========================
def compute_basic(structure):

    coords = np.array([
        atom.coord
        for atom in structure.get_atoms()
    ])

    length = len(
        list(structure.get_residues())
    )

    if len(coords) == 0:
        return 0,0,0,0

    total_sasa = len(coords) * 1.5

    centroid = coords.mean(axis=0)

    distances = np.linalg.norm(
        coords - centroid,
        axis=1
    )

    contact_density = np.mean(distances)

    contact_order = np.std(distances)

    return (
        total_sasa,
        length,
        contact_density,
        contact_order
    )

# =========================
# COMPOSITION FEATURES
# =========================
def compute_composition(structure):

    residues = [
        res.get_resname()
        for res in structure.get_residues()
    ]

    hydrophobic = [
        'ALA','VAL','LEU','ILE',
        'MET','PHE','TRP','PRO'
    ]

    polar = [
        'SER','THR','ASN',
        'GLN','TYR','CYS'
    ]

    charged = [
        'ASP','GLU',
        'LYS','ARG','HIS'
    ]

    total = (
        len(residues)
        if len(residues) > 0
        else 1
    )

    counts = Counter(residues)

    return (

        sum(
            counts[r]
            for r in hydrophobic
            if r in counts
        ) / total,

        sum(
            counts[r]
            for r in polar
            if r in counts
        ) / total,

        sum(
            counts[r]
            for r in charged
            if r in counts
        ) / total
    )

# =========================
# GRAPH FEATURES
# =========================
def compute_graph_features(structure):

    coords = np.array([
        atom.coord
        for atom in structure.get_atoms()
    ])

    if len(coords) < 10:
        return 0,0

    sample = coords[
        np.random.choice(
            len(coords),
            min(200, len(coords)),
            replace=False
        )
    ]

    dists = np.linalg.norm(
        sample[:, None]
        - sample[None, :],
        axis=-1
    )

    threshold = 8

    edges = (
        np.sum(dists < threshold)
        - len(sample)
    )

    possible = (
        len(sample)**2
        - len(sample)
    )

    density = (
        edges / possible
        if possible > 0
        else 0
    )

    avg_dist = np.mean(dists)

    return density, avg_dist

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(
    pdb_path,
    protein_id,
    label
):

    try:

        structure = parser.get_structure(
            "X",
            pdb_path
        )

        (
            total_sasa,
            length,
            contact_density,
            contact_order
        ) = compute_basic(structure)

        (
            hydrophobic,
            polar,
            charged
        ) = compute_composition(structure)

        (
            graph_density,
            avg_shortest
        ) = compute_graph_features(structure)

        return {

            "protein_id": protein_id,

            "total_sasa": total_sasa,

            "charged_fraction": charged,

            "polar_fraction": polar,

            "hydrophobic_fraction": hydrophobic,

            "length": length,

            "contact_density": contact_density,

            "contact_order": contact_order,

            "graph_density": graph_density,

            "avg_shortest_path": avg_shortest,

            "label": label
        }

    except Exception as e:

        print(
            f"[ERROR] {protein_id}: {e}"
        )

        return None

# =========================
# MAIN LOOP
# =========================
data = []

print("\n🚀 Processing independent dataset...")

for file in os.listdir(PDB_DIR):

    if not file.endswith(".pdb"):
        continue

    protein_id = file.replace(
        ".pdb",
        ""
    )

    path = os.path.join(
        PDB_DIR,
        file
    )

    # ======================================
    # NEW LABEL LOGIC
    # ======================================
    if protein_id.startswith("Fung-"):
        label = 1

    elif protein_id.startswith("Non-"):
        label = 0

    else:
        print(f"[SKIP] Unknown format: {file}")
        continue

    feat = extract_features(
        path,
        protein_id,
        label
    )

    if feat:
        data.append(feat)

        print(f"[OK] {protein_id}")

# =========================
# SAVE
# =========================
df = pd.DataFrame(data)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n✅ DONE")
print(df.shape)

print(
    f"📁 Saved: {OUTPUT_FILE}"
)
