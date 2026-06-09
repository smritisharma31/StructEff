# StructEff — Fungal Effector Prediction Tool

StructEff predicts fungal effector proteins from protein sequences using a hybrid deep learning pipeline. Input sequences are first folded into 3D PDB structures using ESMFold, from which **35 structural and physicochemical features are manually computed** from the PDB structures (phi/psi torsion angles, RSA, contact maps, graph topology, betweenness centrality, pLDDT, cysteine content, secondary structure fractions, etc.). Sequences are additionally used for **MMseqs2-based conservation features** (entropy, pct_conserved) and **ESM2-650M sequence embeddings** (1280-dim). All features are combined and trained via a **dynamic hard negative contrastive Siamese network** followed by XGBoost classification.

---

## Performance

Benchmarked on an independent dataset of **213 proteins (53 effectors, 160 non-effectors)**.

| Method | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | MCC |
|--------|----|----|----|----|----------|-----------|--------|----|-----|
| **StructEff** | **43** | 143 | 17 | **10** | 87.3% | 0.717 | **0.811** | **0.761** | 0.678 |
| PPEPFinder | 36 | **152** | **8** | 17 | 88.3% | **0.818** | 0.679 | 0.742 | 0.672 |
| EffectorP 3.0 | 36 | 148 | 12 | 17 | 86.4% | 0.750 | 0.679 | 0.713 | 0.620 |
| EffectorP-fungi 3.0 | 33 | **159** | **1** | 20 | **90.1%** | **0.971** | 0.623 | 0.759 | **0.726** |
| DeepRedEff | 32 | 91 | 69 | 21 | 57.7% | 0.317 | 0.604 | 0.416 | 0.149 |

**Key highlights:**
- StructEff detects the **most true effectors (TP=43)** — 7 more than PPEPFinder and EffectorP 3.0
- StructEff achieves the **highest recall (0.811)** and **best F1-score (0.761)**
- StructEff achieves **ROC-AUC = 0.8979** — best overall discrimination
- EffectorP-fungi 3.0 is most conservative (FP=1) but misses 20 effectors
- DeepRedEff performs poorly with 69 false positives on this dataset

---

## How It Works

```
Input: FASTA file (protein sequences)
        ↓
Step 1: ESMFold API → 3D PDB structures
        ↓
Step 2: Structural feature extraction
        (phi/psi angles, RSA, contact maps,
         graph features, pLDDT, DSSP, etc.)
        ↓
Step 3: MMseqs2 → sequence conservation features
        (entropy, pct_conserved, pct_variable)
        ↓
Step 4: ESM2-650M → sequence embeddings (1280-dim)
        ↓
Step 5: Siamese contrastive transform → XGBoost
        ↓
Output: CSV with protein_id, probability, prediction
```

---

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

CPU-only operation is possible but ESMFold folding will be very slow; a GPU is strongly recommended.

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

#> A Python 3.8 env will not work — the cu128 torch wheels are cp310.

### 3. Python dependencies

```bash
pip install -r requirements.txt
```
First, check that torch is installed:

```bash
python check_setup.py
```

Note: Torch might require cuda/GPU specific installation, if check_setup.py returns a successful torch import; skip step4.


### 4. PyTorch — cuda/GPU specific installation

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
conda activate StructEff
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
