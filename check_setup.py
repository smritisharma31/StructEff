#!/usr/bin/env python3
"""
check_setup.py — verify StructEff dependencies before running the pipeline.

Run from the repo root:
    python check_setup.py

Reports PASS / WARN / FAIL for each requirement. Exits non-zero if any
hard requirement (FAIL) is missing, so it can be used in CI.
WARN items are optional/contextual (e.g. MMseqs2 is only needed without
--no-mmseqs; GPU is recommended but not strictly required).
"""
import importlib
import os
import shutil
import sys

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

results = []  # (level, label, detail)


def record(level, label, detail=""):
    results.append((level, label, detail))


# ---- Python version --------------------------------------------------
v = sys.version_info
if v.major == 3 and v.minor == 10:
    record(PASS, "Python 3.10", f"{v.major}.{v.minor}.{v.micro}")
else:
    record(FAIL, "Python 3.10 required",
           f"found {v.major}.{v.minor}.{v.micro} — cu128 torch wheels are cp310")


# ---- Required Python packages ---------------------------------------
# import_name -> friendly/pip name
required = {
    "torch": "torch",
    "transformers": "transformers",
    "esm": "fair-esm",
    "Bio": "biopython",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "tqdm": "tqdm",
}
for mod, pipname in required.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        record(PASS, f"import {mod}", str(ver))
    except Exception as e:
        record(FAIL, f"import {mod}", f"missing — pip install {pipname} ({e})")


# ---- scikit-learn version pin (matches the pickled scaler) ----------
try:
    import sklearn
    if sklearn.__version__ == "1.6.1":
        record(PASS, "scikit-learn == 1.6.1", sklearn.__version__)
    else:
        record(WARN, "scikit-learn version",
               f"{sklearn.__version__} (bundle pinned to 1.6.1; mismatch may warn or alter scaling)")
except Exception:
    pass  # already recorded as FAIL above


# ---- Torch / CUDA / Blackwell sm_120 --------------------------------
try:
    import torch
    if torch.cuda.is_available():
        try:
            arches = torch.cuda.get_arch_list()
            name = torch.cuda.get_device_name(0)
            # real kernel launch — proves the build actually runs on this GPU
            x = torch.randn(64, 64, device="cuda")
            _ = (x @ x).sum().item()
            torch.cuda.synchronize()
            record(PASS, "CUDA GPU usable", f"{name} | arches={arches}")
            # Blackwell check: if the device is sm_120, the build must list it
            cap = torch.cuda.get_device_capability(0)
            cap_tag = f"sm_{cap[0]}{cap[1]}"
            if cap_tag == "sm_120" and "sm_120" not in arches:
                record(FAIL, "Blackwell sm_120 kernels",
                       "GPU is sm_120 but torch build lacks sm_120 — install cu128 build")
        except Exception as e:
            record(FAIL, "CUDA kernel launch", f"GPU present but compute failed: {e}")
    else:
        record(WARN, "CUDA GPU", "not available — CPU works but ESMFold folding is very slow")
except Exception:
    pass  # torch import already recorded


# ---- External binaries ----------------------------------------------
if shutil.which("mmseqs"):
    record(PASS, "mmseqs on PATH", shutil.which("mmseqs"))
else:
    record(WARN, "mmseqs on PATH",
           "not found — needed for conservation features (omit only with --no-mmseqs)")

# DSSP binary is usually 'mkdssp', sometimes 'dssp'
dssp = shutil.which("mkdssp") or shutil.which("dssp")
if dssp:
    record(PASS, "DSSP (mkdssp/dssp)", dssp)
else:
    record(FAIL, "DSSP (mkdssp/dssp)",
           "not found — required for STEP 2 residue features (conda install -c salilab dssp)")


# ---- Model bundle ----------------------------------------------------
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "XGB_MODEL_v5_hardneg.pkl")
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1e6
    record(PASS, "model bundle present", f"{os.path.basename(model_path)} ({size_mb:.1f} MB)")
else:
    record(FAIL, "model bundle present",
           "XGB_MODEL_v5_hardneg.pkl not found next to this script")


# ---- Report ----------------------------------------------------------
icon = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌"}
print("\nStructEff dependency check")
print("=" * 60)
for level, label, detail in results:
    line = f"{icon[level]} {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
print("=" * 60)

n_fail = sum(1 for r in results if r[0] == FAIL)
n_warn = sum(1 for r in results if r[0] == WARN)

if n_fail:
    print(f"\n{n_fail} hard requirement(s) missing. Fix the ❌ items above.")
    sys.exit(1)
elif n_warn:
    print(f"\nAll hard requirements met. {n_warn} optional warning(s) — "
          "fine for --no-mmseqs / CPU runs.")
    sys.exit(0)
else:
    print("\nAll checks passed. You're ready to test predict.py.")
    sys.exit(0)
