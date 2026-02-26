# SNMF

**SNMF: Integrated learning of mutational signatures and prediction of DNA repair deficiencies**

SNMF is a supervised non-negative matrix factorization method that jointly learns  
(i) mutational signatures shared across samples and  
(ii) a predictor of DNA damage repair (DDR) deficiency labels.

This repository contains the SNMF implementation, preprocessing utilities, and
notebooks to reproduce all experiments reported in the manuscript.

---

## Installation

SNMF is distributed as research code and is intended to be run inside a conda
environment.

git clone https://github.com/joanagoncalveslab/SNMF.git
cd SNMF
conda env create -f environment.yml
conda activate snmf_env

---

## Quickstart

Run SNMF from the repository root using the provided command-line interface:

python -m SNMF.run_snmf \
  --x-train data/processed/zou2021/splits_new/split_10/XbootM_train_all.txt \
  --y-train data/processed/zou2021/splits_new/split_10/YbootM_train_all.txt \
  --x-test  data/processed/zou2021/splits_new/split_10/XbootM_test.txt \
  --y-test  data/processed/zou2021/splits_new/split_10/YbootM_test.txt \
  --outdir results/quickstart/split_00 \
  --k 5 \
  --reps 10 \
  --lambda-c 0.1 \
  --lambda-p 1e-4 \
  --lr 5e-3 \
  --no-plots

---

## Input data

SNMF currently supports input_type="text".

Mutation profiles (X*.txt)
- Rows correspond to samples
- Columns correspond to mutation features (e.g. SBS96 channels)
- Values are non-negative mutation counts
- Format: tab-separated text file

Labels (Y*.txt)
- One label per sample
- Sample order must match the mutation profile matrix
- Labels correspond to DDR gene or pathway deficiency classes

The datasets used in the manuscript are available under:
data/processed/zou2021/splits_new/

---

## Output

All results are written to the directory specified by --outdir. Outputs include:
- learned mutational signatures
- exposure matrices
- predicted labels on test data
- evaluation metrics
- run configuration and logs

Exact filenames depend on the SNMF configuration and replicate settings.

---

## Reproducing manuscript results

All experiments reported in the manuscript can be reproduced using the notebooks
located in:
notebooks/reproduce/

Main notebooks:
- 0_preprocess_celline.ipynb
- 0_preprocess_TCGA.ipynb
- 1_benchmark_cellline.ipynb
- 2_SNMF_TCGA_test.ipynb

---

## Repository structure

SNMF/
  SNMF/                 core SNMF Python package
    sigpro.py           training and testing API
    nmf_cpu.py
    run_snmf.py         command-line interface
  src/
    processing/         preprocessing and bootstrapping utilities
  notebooks/
    reproduce/          notebooks reproducing manuscript experiments
  data/
    raw/                raw input resources
    processed/          processed and bootstrapped datasets
  results/              generated results and figures

---

## License

This repository includes code adapted from SigProfiler (Alexandrov Lab) and is
licensed under the BSD 2-Clause License. See LICENSE.txt for details.
