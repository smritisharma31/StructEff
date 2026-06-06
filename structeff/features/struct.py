"""
structeff/features/struct.py
Extracts structural/physicochemical features from PDB files.
"""

import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from collections import Counter

parser      = PDBParser(QUIET=True)
HYDROPHOBIC = ['ALA','VAL','LEU','ILE','MET','PHE','TRP','PRO']
POLAR       = ['SER','THR','ASN','GLN','TYR','CYS']
CHARGED     = ['ASP','GLU','LYS','ARG','HIS']


def compute_basic(structure):
    coords  = np.array([atom.coord for atom in structure.get_atoms()])
    length  = len(list(structure.get_residues()))
    if len(coords) == 0:
        return 0, 0, 0, 0
    total_sasa      = len(coords) * 1.5
    centroid        = coords.mean(axis=0)
    distances       = np.linalg.norm(coords - centroid, axis=1)
    contact_density = np.mean(distances)
    contact_order   = np.std(distances)
    return total_sasa, length, contact_density, contact_order


def compute_composition(structure):
    residues = [res.get_resname() for res in structure.get_residues()]
    total    = len(residues) if len(residues) > 0 else 1
    counts   = Counter(residues)
    return (
        sum(counts[r] for r in HYDROPHOBIC if r in counts) / total,
        sum(counts[r] for r in POLAR       if r in counts) / total,
        sum(counts[r] for r in CHARGED     if r in counts) / total,
    )


def compute_graph_features(structure):
    coords = np.array([atom.coord for atom in structure.get_atoms()])
    if len(coords) < 10:
        return 0, 0
    sample   = coords[np.random.choice(len(coords), min(200, len(coords)), replace=False)]
    dists    = np.linalg.norm(sample[:, None] - sample[None, :], axis=-1)
    edges    = np.sum(dists < 8) - len(sample)
    possible = len(sample)**2 - len(sample)
    density  = edges / possible if possible > 0 else 0
    return density, np.mean(dists)


def extract_struct_features(pdb_dir):
    data      = []
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    print(f"\n🔹 Extracting structural features from {len(pdb_files)} PDB files...")

    for file in pdb_files:
        protein_id = file.replace(".pdb", "")
        pdb_path   = os.path.join(pdb_dir, file)
        try:
            structure = parser.get_structure("X", pdb_path)
            total_sasa, length, contact_density, contact_order = compute_basic(structure)
            hydrophobic, polar, charged = compute_composition(structure)
            graph_density, _            = compute_graph_features(structure)

            data.append({
                "protein_id"          : protein_id,
                "total_sasa"          : total_sasa,
                "charged_fraction"    : charged,
                "polar_fraction"      : polar,
                "hydrophobic_fraction": hydrophobic,
                "length"              : length,
                "contact_density"     : contact_density,
                "contact_order"       : contact_order,
                "graph_density"       : graph_density,
            })
            print(f"  [OK] {protein_id}")

        except Exception as e:
            print(f"  [ERROR] {protein_id}: {e}")

    return pd.DataFrame(data)
