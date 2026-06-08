# StructEff 🔬

**A Hybrid Contrastive Learning Framework for Fungal Effector Prediction**

StructEff predicts fungal effector proteins from amino acid sequences using a hybrid deep learning pipeline combining ESM2 protein language model embeddings with manually engineered structural and physicochemical features, trained via dynamic hard negative contrastive learning.

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

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/smritisharma31/StructEff.git
cd StructEff
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install DSSP
```bash
# Ubuntu/Debian
sudo apt-get install dssp

# Or via conda
conda install -c salilab dssp
```

### 4. Install MMseqs2 (optional but recommended)
```bash
# Ubuntu
sudo apt install mmseqs2

# Or download binary
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH
```

### 5. Download UniRef50 database (for MMseqs2)
```bash
mmseqs databases UniRef50 uniref50 tmp
```

---

## Usage

### Basic usage (without MMseqs2):
```bash
python predict.py \
    --fasta your_proteins.fasta \
    --outdir results/ \
    --no-mmseqs
```

### Full pipeline (with MMseqs2):
```bash
python predict.py \
    --fasta your_proteins.fasta \
    --outdir results/ \
    --mmseqs-db /path/to/uniref50 \
    --threads 16
```

### Input format:
```
>protein_1
MSTLVPSLCLAAALVHAAAAASAPETPKPQVLKGSDMQNVSESPHPEPQIELNPGALSTPTPALLLQG
>protein_2
MKLVPSLFLAAALVHAAAAASAPETPKPQVLKGSDMQ
```

### Output:
```
protein_id,probability,prediction,label
protein_1,0.923,1,Effector
protein_2,0.043,0,Non-effector
```

---

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--fasta` | Input FASTA file | required |
| `--outdir` | Output directory | `results/` |
| `--mmseqs-db` | Path to UniRef50 MMseqs2 database | None |
| `--threads` | Number of threads for MMseqs2 | 8 |
| `--threshold` | Prediction threshold | 0.64 |
| `--no-mmseqs` | Skip MMseqs2 entropy features | False |

---

## Model Architecture

StructEff uses a three-stage hybrid architecture:

1. **Feature Engineering (35 features)**
   - Structural: phi/psi angles, RSA, contact maps, betweenness centrality
   - Physicochemical: charged/polar/hydrophobic fractions, cysteine content
   - Conservation: sequence entropy from MMseqs2 alignment
   - Graph: contact density, graph density, total SASA

2. **ESM2-650M Embeddings (1280-dim)**
   - Mean pooled representations from ESM2 650M parameter model

3. **Siamese Contrastive Network + XGBoost**
   - Dynamic hard negative mining during contrastive training
   - 256-dimensional transformed embeddings
   - XGBoost classifier on concatenated features

---

## Citation

If you use StructEff in your research, please cite:

```
Sharma S, et al. StructEff: A Hybrid Contrastive Learning Framework
Integrating Protein Language Models and Structural Features for
Fungal Effector Prediction. (2025)
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- ESM2 (fair-esm)
- XGBoost >= 1.6
- Biopython >= 1.79
- MMseqs2 (optional)
- mkdssp

---

## License

MIT License — see LICENSE file for details.

---

## Contact

Smriti Sharma — smriti1831@gmail.com
GitHub: https://github.com/smritisharma31
