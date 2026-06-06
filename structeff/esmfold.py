"""
structeff/esmfold.py
Generates PDB structures from sequences using ESMFold API.
"""

import os
import time
import requests
from tqdm import tqdm
from Bio import SeqIO


def fold_sequence(sequence, protein_id, output_dir, max_retries=3):
    url     = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    seq     = sequence[:400]

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=seq, timeout=60)

            if response.status_code == 200:
                pdb_path = os.path.join(output_dir, f"{protein_id}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(response.text)
                return pdb_path

            elif response.status_code == 429:
                print(f"[RATE LIMIT] {protein_id} — waiting 10s...")
                time.sleep(10)

            else:
                print(f"[ERROR] {protein_id}: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"[RETRY {attempt+1}] {protein_id}: {e}")
            time.sleep(5)

    print(f"[FAILED] {protein_id} after {max_retries} attempts")
    return None


def fold_fasta(fasta_file, output_dir, delay=1.5):
    os.makedirs(output_dir, exist_ok=True)
    records = list(SeqIO.parse(fasta_file, "fasta"))
    print(f"\n🔹 Folding {len(records)} sequences with ESMFold API...")

    results = {}
    failed  = []

    for record in tqdm(records):
        protein_id = record.id.strip()
        sequence   = str(record.seq)

        pdb_path = os.path.join(output_dir, f"{protein_id}.pdb")
        if os.path.exists(pdb_path):
            print(f"[SKIP] {protein_id} already exists")
            results[protein_id] = pdb_path
            continue

        path = fold_sequence(sequence, protein_id, output_dir)

        if path:
            results[protein_id] = path
        else:
            failed.append(protein_id)

        time.sleep(delay)

    print(f"\n✅ Folded: {len(results)} | Failed: {len(failed)}")
    if failed:
        print(f"Failed: {failed}")

    return results
