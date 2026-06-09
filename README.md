# StructEff — Fungal Effector Prediction Tool

StructEff predicts fungal effector proteins from amino-acid sequence by
combining predicted 3D structure (local ESMFold), residue/structural/biological
features, MMseqs2 conservation entropy, and ESM2-650M embeddings, transformed
through a contrastive (Siamese) network and classified with XGBoost.

## Pipeline

```
FASTA
 ├─ STEP 1  ESMFold (local)        → 3D structures (PDB)
 ├─ STEP 2  residue features       (DSSP)
 ├─ STEP 3  aggregate to protein
 ├─ STEP 4  structural features
 ├─ STEP 5  conservation entropy   (MMseqs2 vs UniRef50)
 ├─ STEP 6  biological features
 ├─ STEP 7  merge
 ├─ STEP 8  ESM2-650M embeddings
 └─ contrastive transform → XGBoost → prediction
```

## Repository layout

```
StructEff/
├── predict.py
├── structeff/                  # pipeline package
│   ├── esmfold.py              # local ESMFold folding
│   ├── siamese.py              # contrastive network (needed to load the model)
│   ├── merge.py
│   └── features/               # residue, graph, struct, entropy, bio
├── XGB_MODEL_v5_hardneg.pkl    # trained model bundle (committed)
├── test_run.fasta                  # tiny example input for the verification run
├── requirements.txt
├── tools/                      # MMseqs2 binary      (gitignored — install per machine)
├── mmseqs_db/                  # UniRef50 DB ~150 GB (gitignored — build per machine)
└── results/                    # pipeline outputs    (gitignored)
```

`tools/`, `mmseqs_db/`, and `results/` are gitignored on purpose — they are
large and/or machine-specific and must never be committed.

---

## System requirements

| Component | Requirement |
|-----------|-------------|
| OS        | Linux (x86-64). Tested on Ubuntu. |
| Python    | 3.10 (required — cu128 torch wheels are cp310) |
| GPU       | NVIDIA CUDA GPU. ~16 GB VRAM is sufficient at the default 400-residue fold cap. |
| GPU driver| CUDA 12.8-capable. **Blackwell cards (RTX 50-series, sm_120) require a cu128 torch build** — see PyTorch step below. |
| CPU       | AVX2 recommended (for the standard MMseqs2 binary; an SSE4.1 build exists otherwise) |
| RAM       | 16 GB+ (MMseqs2 search against UniRef50 peaks high; it splits automatically if memory is limited) |
| Disk      | **~180+ GB free**: UniRef50 MMseqs DB (~150 GB) + ESMFold weights (~8.4 GB) + ESM2-650M (~2.5 GB) + structures/outputs |
| Network   | First run downloads model weights; UniRef50 download is large and multi-hour |

CPU-only operation is possible but ESMFold folding will be very slow; a GPU is
strongly recommended.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/smritisharma31/StructEff.git
cd StructEff
```

All subsequent commands assume you are inside this directory.

### 2. Environment (Python 3.10 required)

```bash
conda create -n StructEff python=3.10 -y
conda activate StructEff
```

> A Python 3.8 env will not work — the cu128 torch wheels are cp310.

### 3. PyTorch — install FIRST (before the other deps)

On NVIDIA Blackwell GPUs (RTX 50-series, e.g. 5070 Ti = compute capability
sm_120) you must use a CUDA 12.8 build. Standard cu121/cu124 wheels lack sm_120
kernels and fail at runtime even though `cuda.is_available()` returns True.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
# if the stable cu128 wheel lacks sm_120, use the nightly:
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# verify sm_120 is present and a real kernel launches:
python -c "import torch; print(torch.cuda.get_arch_list()); \
x=torch.randn(2000,2000,device='cuda'); print('ok', tuple((x@x).shape))"
```

For non-Blackwell GPUs, use the CUDA build matching your driver (see
https://pytorch.org/get-started/locally/).

### 4. Python dependencies

```bash
pip install -r requirements.txt
```

### 5. External tools (not pip)

**DSSP** — required for STEP 2 residue features:

```bash
conda install -c salilab dssp -y     # provides the `mkdssp` binary
```

**MMseqs2** — required for STEP 5 conservation entropy. Install the static
binary into `tools/` inside the repo:

```bash
mkdir -p tools
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xzf mmseqs-linux-avx2.tar.gz -C tools/
rm mmseqs-linux-avx2.tar.gz
ls tools/mmseqs/bin/mmseqs            # confirm the binary exists

# put it on PATH (use the ABSOLUTE path to your clone)
echo "export PATH=$(pwd)/tools/mmseqs/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
mmseqs version
```

> If `mmseqs version` reports "Illegal instruction", your CPU lacks AVX2 —
> download the `mmseqs-linux-sse41` build instead.

### 6. Download the UniRef50 database (~150+ GB free, several hours)

```bash
mkdir -p mmseqs_db
mmseqs databases UniRef50 mmseqs_db/uniref50 mmseqs_db/tmp --threads 8
rm -rf mmseqs_db/tmp                  # reclaim temp space when done
```

The first `predict.py` run also downloads model weights once (cached after):
ESMFold ~8.4 GB, ESM2-650M ~2.5 GB.

---

## Verify the installation

First, check that all dependencies are met:

```bash
python check_setup.py
```

This reports PASS / WARN / FAIL for Python version, required packages,
torch+CUDA (including Blackwell sm_120), MMseqs2, DSSP, and the model file. It
exits non-zero if a hard requirement is missing. WARN items are optional
(e.g. MMseqs2 is only needed without `--no-mmseqs`; a GPU is recommended but
CPU works).

Then run the committed `test_run.fasta` end to end:

```bash
# full pipeline (with conservation features):
python predict.py --fasta test_run.fasta --outdir results/ \
    --mmseqs-db mmseqs_db/uniref50

# OR quick check without MMseqs2 (entropy zero-filled — skips the long search):
python predict.py --fasta test_run.fasta --outdir results/ --no-mmseqs
```

A successful run prints STEP 1–8 banners and writes
`results/structeff_predictions.csv`. The two example sequences have no UniRef50
homologs, so their entropy features are zero-filled and both are predicted
non-effectors — this confirms the plumbing works. For real accuracy testing,
run a FASTA of known effectors.

---

## Usage

```bash
# recommended (real conservation features):
python predict.py --fasta proteins.fasta --outdir results/ \
    --mmseqs-db mmseqs_db/uniref50

# fast, no MMseqs2 (entropy zero-filled — testing only):
python predict.py --fasta proteins.fasta --outdir results/ --no-mmseqs
```

| Option        | Default            | Description                              |
|---------------|--------------------|------------------------------------------|
| `--fasta`     | (required)         | Input FASTA file                         |
| `--outdir`    | `results`          | Output directory                         |
| `--mmseqs-db` | none               | Path to MMseqs2 UniRef50 DB prefix       |
| `--threads`   | `8`                | Threads for MMseqs2                      |
| `--threshold` | `0.2`              | Prediction threshold (high-recall)       |
| `--no-mmseqs` | off                | Skip MMseqs2; zero-fill entropy features |

Default threshold is **0.2**, matching the high-recall objective the model was
tuned for. Override with `--threshold`.

Output `structeff_predictions.csv` columns: `protein_id`, `probability`,
`prediction` (0/1), `label` (Effector / Non-effector).

---

## Notes

- **GPU memory:** ESMFold and ESM2-650M load at different pipeline stages, so a
  16 GB card is sufficient at the default 400-residue fold cap.
- **scikit-learn is pinned to 1.6.1** to match the StandardScaler in the model
  bundle; other versions may emit warnings or alter the scaling.
