# =========================================================
# COMPUTE CYSTEINE + LENGTH FEATURES FOR INDEPENDENT SET
# Matches the style of your existing independent feature scripts.
#
# Input:  independent_216/ folder (Fung-*.pdb, Non-*.pdb)
# Output: adds cysteine_count, cysteine_fraction,
#         has_4plus_cys, is_short to independent_aligned_features.csv
#
# Final output: independent_aligned_features_v4.csv
# =========================================================

import os
import pandas as pd
from Bio.PDB import PDBParser


# =========================
# SETTINGS
# =========================
BASE_DIR   = "independent_216"          # folder with Fung-*.pdb and Non-*.pdb
INPUT_CSV  = "independent_aligned_features.csv"
OUTPUT_CSV = "independent_aligned_features_v4.csv"

parser = PDBParser(QUIET=True)


# =========================
# FEATURE EXTRACTION
# =========================
def get_cys_count(filepath, protein_id):
    """
    Count CYS residues using same CA-based logic as training pipeline.
    Returns (seq_length, cys_count)
    """
    structure = parser.get_structure(protein_id, filepath)
    model = structure[0]

    seen = set()
    cys_count = 0

    for chain in model:
        for res in chain:
            if 'CA' not in res:
                continue
            res_key = (chain.id, res.get_id()[1])
            if res_key not in seen:
                seen.add(res_key)
                if res.get_resname() == "CYS":
                    cys_count += 1

    return len(seen), cys_count


# =========================
# LOOP THROUGH PDB FILES
# =========================
rows = []

print(f"\n🔹 Processing PDB files in: {BASE_DIR}")
print("-" * 55)

for file in sorted(os.listdir(BASE_DIR)):
    if not file.endswith(".pdb"):
        continue

    protein_id = file.replace(".pdb", "")
    pdb_path   = os.path.join(BASE_DIR, file)

    # Assign label same way as your existing script
    if file.startswith("Fung"):
        label = 1
    elif file.startswith("Non"):
        label = 0
    else:
        print(f"  [SKIP] {protein_id} — unknown prefix")
        continue

    try:
        seq_len, cys = get_cys_count(pdb_path, protein_id)

        rows.append({
            "protein_id":        protein_id,
            "cysteine_count":    cys,
            "cysteine_fraction": cys / seq_len if seq_len > 0 else 0,
            "has_4plus_cys":     1 if cys >= 4 else 0,
            "is_short":          1 if seq_len <= 400 else 0
        })

        print(f"  ✅ {protein_id:25s}  len={seq_len:4d}  cys={cys:3d}  "
              f"is_short={1 if seq_len<=400 else 0}  label={label}")

    except Exception as e:
        print(f"  ⚠️  [CRASH] {protein_id}: {e}")


# =========================
# BUILD DATAFRAME
# =========================
df_cys = pd.DataFrame(rows)
print(f"\n✅ Features computed for {len(df_cys)} proteins")


# =========================
# LOAD EXISTING FEATURES
# =========================
print(f"\n🔹 Loading {INPUT_CSV}...")
df_orig = pd.read_csv(INPUT_CSV)
print(f"   Original shape : {df_orig.shape}")


# =========================
# MERGE
# =========================
df_merged = pd.merge(
    df_orig,
    df_cys[["protein_id", "cysteine_count",
            "cysteine_fraction", "has_4plus_cys", "is_short"]],
    on="protein_id",
    how="left"
)

# Fill any unmatched with 0
df_merged["cysteine_count"]    = df_merged["cysteine_count"].fillna(0).astype(int)
df_merged["cysteine_fraction"] = df_merged["cysteine_fraction"].fillna(0)
df_merged["has_4plus_cys"]     = df_merged["has_4plus_cys"].fillna(0).astype(int)
df_merged["is_short"]          = df_merged["is_short"].fillna(0).astype(int)

unmatched = df_merged[df_merged["cysteine_count"] == 0]["protein_id"].tolist()
if unmatched:
    print(f"\n⚠️  {len(unmatched)} proteins had no PDB match — filled with 0:")
    for pid in unmatched[:10]:
        print(f"    {pid}")

print(f"\n   New shape      : {df_merged.shape}")
print(f"   Columns        : {list(df_merged.columns)}")


# =========================
# SANITY CHECK BY LABEL
# =========================
if "label" in df_merged.columns:
    print("\nFeature statistics by label:")
    print("-" * 65)
    for feat in ["cysteine_count", "cysteine_fraction",
                 "has_4plus_cys", "is_short"]:
        pos  = df_merged[df_merged["label"] == 1][feat].mean()
        neg  = df_merged[df_merged["label"] == 0][feat].mean()
        diff = pos - neg
        flag = "✅" if abs(diff) > 0.02 else "⚠️ "
        print(f"  {flag} {feat:22s}  Effector={pos:.3f}  "
              f"Non-effector={neg:.3f}  diff={diff:+.3f}")
    print("-" * 65)


# =========================
# SAVE
# =========================
df_merged.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved → {OUTPUT_CSV}")
print(f"   Shape : {df_merged.shape}")
print(f"\nNext steps:")
print(f"  1. Update independent script to load '{OUTPUT_CSV}'")
print(f"  2. Add new columns to training_columns list:")
print(f"     'cysteine_count', 'cysteine_fraction', 'has_4plus_cys', 'is_short'")
print(f"  3. Update struct_cols: range(32) → range(36)")
print(f"  4. Update EXPECTED_DIM: 1312 → 1316  (36 + 1280)")
print(f"  5. Update EXPECTED_FINAL_DIM: 1568 → 1572  (1316 + 256)")
