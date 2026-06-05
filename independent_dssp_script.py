import os
import pandas as pd
from Bio.PDB import PDBParser, DSSP

# =========================
# SETTINGS
# =========================
BASE_DIR = "independent_216"
DSSP_EXE = "mkdssp"
OUTPUT_CSV = "independent_dssp_features.csv"

HELIX = {"H", "G", "I"}
STRAND = {"E", "B"}

parser = PDBParser(QUIET=True)
rows = []

# =========================
# LOOP THROUGH PDB FILES
# =========================
for file in sorted(os.listdir(BASE_DIR)):

    if not file.endswith(".pdb"):
        continue

    pdb_path = os.path.join(BASE_DIR, file)
    protein_id = file.replace(".pdb", "")

    if file.startswith("Fung"):
        label = 1

    elif file.startswith("Non"):
        label = 0

    else:
        continue

    try:
        structure = parser.get_structure(protein_id, pdb_path)
        model = structure[0]

        dssp = DSSP(
            model,
            pdb_path,
            dssp=DSSP_EXE,
            file_type="PDB"
        )

        if len(dssp) == 0:
            continue

        ss_list = []
        rsa_list = []

        for prop in dssp.property_list:
            ss = prop[2]
            rsa = prop[3]

            ss_list.append(ss)
            rsa_list.append(rsa)

        total = len(ss_list)

        helix_count = sum(ss in HELIX for ss in ss_list)
        strand_count = sum(ss in STRAND for ss in ss_list)
        coil_count = total - helix_count - strand_count

        rows.append({
            "protein_id": protein_id,

            "length_dssp": total,

            "helix_frac": helix_count / total,
            "strand_frac": strand_count / total,
            "coil_frac": coil_count / total,

            "mean_rsa": sum(rsa_list) / total,

            "helix_count": helix_count,
            "strand_count": strand_count,
            "coil_count": coil_count,

            "label": label
        })

        print(f"[OK] {protein_id}")

    except Exception as e:
        print(f"[CRASH] {protein_id}: {e}")

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ DSSP DONE")
print(df.shape)
