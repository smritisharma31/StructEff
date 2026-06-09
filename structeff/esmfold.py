"""
structeff/esmfold.py
Generates PDB structures from sequences using a LOCAL ESMFold model
(HuggingFace transformers: facebook/esmfold_v1).

Drop-in replacement for the old API-based version. The public contract is
unchanged so the rest of the pipeline (STEP 2 onward) works as before:

    fold_fasta(fasta_file, output_dir) -> {protein_id: pdb_path}

    - writes one "{protein_id}.pdb" per successfully folded sequence
    - skips proteins whose .pdb already exists (caching)
    - never raises on a single failure; collects them in a `failed` list

Install (pin a known-good combo; adjust torch build to your CUDA):
    pip install "transformers>=4.40" accelerate
    # torch must already be installed for your CUDA version

First run downloads ~2.5 GB of weights to your HF cache (once).
"""
import os
import gc
import torch
from tqdm import tqdm
from Bio import SeqIO

# ----------------------------------------------------------------------
# Tunables. Adjust MAX_LEN / LONG_SEQ_THRESHOLD to your GPU's VRAM.
# ESMFold peak memory scales ~quadratically with sequence length.
# ----------------------------------------------------------------------
MAX_LEN = 400                 # hard cap; matches the old API truncation
LONG_SEQ_THRESHOLD = 350      # above this, shrink chunk size to save VRAM
CHUNK_SIZE_LONG = 64          # trunk chunk size for long sequences (None = off)
MODEL_NAME = "facebook/esmfold_v1"

# Module-level cache so the 2.5 GB model loads only once per process.
_MODEL = None
_TOKENIZER = None
_DEVICE = None


def _load_model():
    """Lazy-load ESMFold once and keep it cached at module scope."""
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    from transformers import AutoTokenizer, EsmForProteinFolding

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔹 Loading local ESMFold ({MODEL_NAME}) on {_DEVICE}...")

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(MODEL_NAME)
    model = model.to(_DEVICE)
    model.eval()

    # Keep the language-model stem in fp32 on CPU; on GPU, fp16 on the
    # folding trunk saves a lot of memory with negligible accuracy cost.
    if _DEVICE.type == "cuda":
        model.esm = model.esm.half()

    _MODEL = model
    print("✅ ESMFold loaded")
    return _MODEL, _TOKENIZER, _DEVICE


def _fold_once(model, tokenizer, device, seq):
    """Run one sequence through ESMFold and return PDB text."""
    # Long sequences: trade speed for lower peak memory.
    if len(seq) > LONG_SEQ_THRESHOLD and CHUNK_SIZE_LONG is not None:
        try:
            model.trunk.set_chunk_size(CHUNK_SIZE_LONG)
        except Exception:
            pass
    else:
        try:
            model.trunk.set_chunk_size(None)
        except Exception:
            pass

    inputs = tokenizer(
        [seq], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    with torch.no_grad():
        output = model.infer_pdb(seq) if hasattr(model, "infer_pdb") else None
        if output is not None:
            return output
        # Fallback for versions without the infer_pdb convenience method:
        result = model(inputs)
        return model.output_to_pdb(result)[0]


# Standard dummy unit cell. DSSP (and many PDB parsers) require a CRYST1
# record before the ATOM records; predicted structures have no real cell,
# so a 1 Å P1 box is the accepted convention.
_CRYST1 = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"


def _sanitize_pdb(pdb_text):
    """
    Make HuggingFace ESMFold PDB text parseable by DSSP/BioPython:
      - drop unsupported PARENT records and blank leading lines
      - guarantee a CRYST1 record appears before the first ATOM
    """
    lines = [ln for ln in pdb_text.splitlines() if ln.strip() != ""]
    lines = [ln for ln in lines if not ln.startswith("PARENT")]

    if not any(ln.startswith("CRYST1") for ln in lines):
        # insert CRYST1 just before the first ATOM/HETATM record
        for i, ln in enumerate(lines):
            if ln.startswith(("ATOM", "HETATM")):
                lines.insert(i, _CRYST1.rstrip("\n"))
                break
        else:
            lines.insert(0, _CRYST1.rstrip("\n"))

    return "\n".join(lines) + "\n"


def fold_sequence(sequence, protein_id, output_dir):
    """
    Fold one sequence locally and write {protein_id}.pdb.
    Returns the pdb path on success, or None on failure (never raises).
    Tries GPU first; on CUDA OOM, retries the same sequence on CPU.
    """
    model, tokenizer, device = _load_model()
    seq = str(sequence)[:MAX_LEN]
    pdb_path = os.path.join(output_dir, f"{protein_id}.pdb")

    try:
        pdb_text = _fold_once(model, tokenizer, device, seq)
    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] {protein_id} ({len(seq)} aa) — retrying on CPU...")
        torch.cuda.empty_cache()
        gc.collect()
        try:
            cpu_model = model.to("cpu")
            pdb_text = _fold_once(cpu_model, tokenizer, torch.device("cpu"), seq)
            model.to(device)  # move back for subsequent sequences
        except Exception as e:
            print(f"[FAILED] {protein_id} on CPU fallback: {e}")
            return None
    except Exception as e:
        print(f"[FAILED] {protein_id}: {e}")
        return None

    pdb_text = _sanitize_pdb(pdb_text)
    with open(pdb_path, "w") as f:
        f.write(pdb_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pdb_path


def fold_fasta(fasta_file, output_dir, delay=0.0):
    """
    Fold every sequence in `fasta_file`, writing one PDB per protein into
    `output_dir`. Skips proteins whose .pdb already exists.

    `delay` is kept for signature compatibility but unused locally (no
    rate limit to respect).

    Returns: {protein_id: pdb_path} for successfully folded proteins.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = list(SeqIO.parse(fasta_file, "fasta"))
    print(f"\n🔹 Folding {len(records)} sequences locally with ESMFold...")

    results = {}
    failed = []

    for record in tqdm(records):
        protein_id = record.id.strip()
        sequence = str(record.seq)
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

    print(f"\n✅ Folded: {len(results)} | Failed: {len(failed)}")
    if failed:
        print(f"Failed: {failed}")
    return results
