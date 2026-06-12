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

from structeff.fasta_utils import clean_sequence

# ----------------------------------------------------------------------
# Folding length / memory policy (tuned for ~16 GB VRAM, RTX 5070 Ti).
# ESMFold peak GPU memory scales ~quadratically with sequence length.
#
#   <= CHUNK_THRESHOLD : fold with no chunking (fastest)
#   CHUNK_THRESHOLD..FOLD_CAP : fold, stepping chunk size down on OOM
#   > FOLD_CAP         : TRUNCATE to FOLD_CAP, then fold (cap-fold) — the
#                        protein still gets a (partial, N-terminal) structure
#                        and a prediction rather than being dropped.
#
# If folding still OOMs at the smallest chunk size, that single protein is
# skipped and logged (NOT sent to CPU — CPU folding of large proteins can
# exhaust system RAM and get the whole run killed by the OOM killer).
# ----------------------------------------------------------------------
FOLD_CAP = 800                 # truncate sequences longer than this before folding
CHUNK_THRESHOLD = 350          # above this length, use chunked attention
CHUNK_STEPS = [128, 64, 32]    # chunk sizes tried in order on repeated OOM
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


def _fold_once(model, tokenizer, device, seq, chunk_size):
    """Run one sequence through ESMFold at a given trunk chunk size.

    chunk_size=None disables chunking (fastest, highest memory). A smaller
    integer lowers peak memory at the cost of speed.
    """
    try:
        model.trunk.set_chunk_size(chunk_size)
    except Exception:
        pass

    with torch.no_grad():
        if hasattr(model, "infer_pdb"):
            return model.infer_pdb(seq)
        # Fallback for versions without the infer_pdb convenience method:
        inputs = tokenizer(
            [seq], return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)
        result = model(inputs)
        return model.output_to_pdb(result)[0]


def _chunk_schedule(seq_len):
    """Chunk sizes to try, in order, for a given (already-capped) length."""
    if seq_len <= CHUNK_THRESHOLD:
        return [None] + CHUNK_STEPS      # try fast path first, then shrink
    return list(CHUNK_STEPS)             # long: go straight to chunked


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
    Sequences longer than FOLD_CAP are truncated; on CUDA OOM the chunk
    size is stepped down, and the protein is skipped if it still won't fit.
    """
    model, tokenizer, device = _load_model()

    seq = clean_sequence(sequence)
    if len(seq) < 5:
        print(f"[SKIP] {protein_id}: no valid residues after cleaning "
              f"(len={len(seq)}); original had non-standard characters")
        return None

    orig_len = len(seq)
    if orig_len > FOLD_CAP:
        seq = seq[:FOLD_CAP]
        print(f"[TRUNCATE] {protein_id}: {orig_len} aa -> {FOLD_CAP} aa "
              f"(too long for full-length folding; predicting on N-terminal {FOLD_CAP})")
        with open(os.path.join(output_dir, "truncated_folds.txt"), "a") as tf:
            tf.write(f"{protein_id}\t{orig_len}\t{FOLD_CAP}\n")

    pdb_path = os.path.join(output_dir, f"{protein_id}.pdb")

    # Try folding, stepping the chunk size down on each CUDA OOM. Only give up
    # (skip + log) if even the smallest chunk size runs out of memory. No CPU
    # fallback: CPU folding of large proteins can exhaust system RAM and get
    # the whole run killed by the OS OOM killer.
    pdb_text = None
    for chunk in _chunk_schedule(len(seq)):
        try:
            pdb_text = _fold_once(model, tokenizer, device, seq, chunk)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            pdb_text = None
            continue
        except Exception as e:
            print(f"[FAILED] {protein_id}: {e}")
            return None

    if pdb_text is None:
        print(f"[SKIP] {protein_id}: out of GPU memory at all chunk sizes "
              f"(len={len(seq)}) — too large to fold on this card")
        torch.cuda.empty_cache()
        gc.collect()
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
        failed_path = os.path.join(output_dir, "failed_folds.txt")
        with open(failed_path, "w") as fh:
            fh.write("\n".join(failed) + "\n")
        preview = ", ".join(failed[:10])
        more = f" (+{len(failed) - 10} more)" if len(failed) > 10 else ""
        print(f"   Failed IDs: {preview}{more}")
        print(f"   Full list written to: {failed_path}")
    return results
