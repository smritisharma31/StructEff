"""
structeff/fasta_utils.py
Input FASTA parsing / sanitization for StructEff.

This is the SINGLE place sequence character-cleaning is defined. The pipeline
cleans the input ONCE at ingestion (prepare_fasta) and writes a sanitized
FASTA that every downstream step (folding, MMseqs2, ESM2 embeddings) reads —
so the same cleaning never has to be re-implemented per consumer.

Scope: character cleaning only. Length caps are model-specific and stay where
they belong (ESMFold cap in esmfold.py; ESM2's 1022-residue architectural
limit in predict.py).
"""
import os
from Bio import SeqIO

# The 20 standard amino acids. 'X' (unknown) is also accepted by both the
# ESMFold and ESM2 tokenizers. Everything else (stop codons '*', gaps '-'/'.',
# whitespace, digits) is removed; rare ambiguous residues are mapped below.
_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
_AMBIGUOUS_MAP = {
    "U": "C",  # selenocysteine -> cysteine
    "O": "K",  # pyrrolysine    -> lysine
    "B": "X",  # Asx (N/D)      -> unknown
    "Z": "X",  # Glx (Q/E)      -> unknown
    "J": "X",  # Leu/Ile        -> unknown
}

# Sequences with fewer valid residues than this after cleaning are dropped:
# they cannot fold or embed meaningfully.
MIN_LEN = 5


def clean_sequence(sequence):
    """
    Normalize one raw sequence for the ESMFold / ESM2 tokenizers:
      - uppercase, strip whitespace
      - drop stop codons '*', gaps '-'/'.', and any other junk characters
      - map ambiguous residues (U,O,B,Z,J) to a standard residue or 'X'
      - keep 'X' (both tokenizers handle unknown residues)
    Returns the cleaned string (may be empty if nothing valid remains).
    Idempotent: cleaning an already-clean sequence returns it unchanged.
    """
    seq = "".join(str(sequence).split()).upper()
    out = []
    for ch in seq:
        if ch in _STANDARD_AA or ch == "X":
            out.append(ch)
        elif ch in _AMBIGUOUS_MAP:
            out.append(_AMBIGUOUS_MAP[ch])
        # else: dropped
    return "".join(out)


def prepare_fasta(raw_fasta, out_fasta):
    """
    Read raw_fasta, clean every sequence, and write a sanitized FASTA to
    out_fasta. Returns a report dict:

        {
          "clean_path": out_fasta,
          "n_in":  <records read>,
          "n_out": <records written>,
          "kept":     [protein_id, ...],
          "modified": [protein_id, ...],   # sequence changed by cleaning
          "dropped":  [(protein_id, reason), ...],
        }

    Dropped: sequences shorter than MIN_LEN after cleaning, and duplicate IDs
    (first occurrence kept). Writing a single cleaned FASTA means every
    downstream step operates on identical, tokenizer-safe input.
    """
    out_dir = os.path.dirname(os.path.abspath(out_fasta))
    os.makedirs(out_dir, exist_ok=True)

    kept, modified, dropped = [], [], []
    seen = set()
    n_in = 0

    with open(out_fasta, "w") as out:
        for record in SeqIO.parse(raw_fasta, "fasta"):
            n_in += 1
            pid = record.id.strip()
            raw = str(record.seq)
            cleaned = clean_sequence(raw)

            if pid in seen:
                dropped.append((pid, "duplicate id"))
                continue
            if len(cleaned) < MIN_LEN:
                dropped.append((pid, f"too short after cleaning (len={len(cleaned)})"))
                continue

            seen.add(pid)
            kept.append(pid)
            if cleaned != "".join(raw.split()).upper():
                modified.append(pid)

            out.write(f">{pid}\n{cleaned}\n")

    return {
        "clean_path": out_fasta,
        "n_in": n_in,
        "n_out": len(kept),
        "kept": kept,
        "modified": modified,
        "dropped": dropped,
    }
