# Reassessing Protein Function Prediction Performance of Alignment-based Approaches

Protein Function Prediction (PFP) is a critical task in bioinformatics, where the goal is to predict the function of proteins based on their sequences and annotations.

Current methods often rely on machine learning and deep learning techniques, with alignement-based methods being a common baseline.  
This repository shows that their implementation in the literature is often sub-optimal and does not represent real-world scenarios.
It provides a pipeline to run these methods using more realistic settings, showing that they can match state-of-the-art algorithms.


## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Reproduction Steps](#reproduction-steps)
  - [1. Data Downloading & Preparation](#1-data-downloading--preparation)
  - [2. Annotation Propagation](#2-annotation-propagation)
  - [3. Alignment](#3-alignment)
  - [4. Preparing the evaluation](#4-preparing-the-evaluation)
  - [5. Running Baselines](#5-running-baselines)
  - [6. Evaluation](#6-evaluation)
- [Datasets](#datasets)
- [Notes](#notes)
---

## Overview

This repository provides a pipeline for running baseline alignment-based methods (e.g. Diamond) for Protein Function Prediction (PFP) tasks.  
It covers all steps from data downloading to evaluation, including parsing, annotation propagation, alignment, running baselines, format conversion, and final evaluation.

## Requirements

The following Python packages are required to run the scripts in this repository:
```sh
pip install obonet networkx tqdm pandas scipy biopython matplotlib seaborn
```

Additionally, the Diamond software is required for sequence alignment.  
You can download it from the [Diamond GitHub repository](http://github.com/bbuchfink/diamond) or simply execute the following command on Linux-based systems:

```sh
wget http://github.com/bbuchfink/diamond/releases/download/v2.1.11/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
```

## Reproduction Steps

### 1. Data Downloading & Preparation

Download all necessary information (e.g., protein sequences, GO annotations) from UniProt (SwissProt) at different releases and parse them into the required formats.

```sh
python download_swissprot.py
```

### 2. Annotation Propagation
Propagate GO annotations using the ontology structure:

```sh
python propagate_swissprot_terms.py
```
This step uses the GO ontology to propagate annotations from parent to child terms, ensuring that all relevant annotations are included.  
Note that propagating terms can take a while.

### 3. Alignment
Run sequence alignment using Diamond on the most up-to-date SwissProt database:

```sh
echo "Creating Diamond database..."
diamond makedb --in data/2024_01/swissprot_2024_01.fasta -d data/2024_01/swissprot_2024_01_proteins_set

echo "Running Diamond blast on protein sequences against themselves..."
diamond blastp --very-sensitive --db data/swissprot/2024_01/swissprot_2024_01_proteins_set.dmnd --query data/swissprot/2024_01/swissprot_2024_01.fasta --out data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv -e 0.001
```
This step creates a Diamond database from the SwissProt protein sequences and performs a sequence alignment to find similar proteins. The output will be stored in `data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv`.  
Note that as the 2024 release of SwissProt contains over 570,000 proteins, the all-vs-all alignment step can be rather long (about 1 hour).

### 4. Preparing the evaluation
To evaluate the performance of a method, $IC$-weighted scores are used. These scores are computed based on the Information Content ($IC$) of the GO terms, which is derived from the background distribution of GO terms in the dataset.  
Background files must be generated prior to running the baselines.  

Example for the ATGO dataset (uses dedicated train/test split files):
```sh
python background.py \
  --cco ./data/ATGO/ATGO_CCO_train_annotations.tsv \
  --bpo ./data/ATGO/ATGO_BPO_train_annotations.tsv \
  --mfo ./data/ATGO/ATGO_MFO_train_annotations.tsv \
  --output ./data/ATGO/background_ATGO.pkl \
  --test_cco ./data/ATGO/ATGO_CCO_test_annotations.tsv \
  --test_bpo ./data/ATGO/ATGO_BPO_test_annotations.tsv \
  --test_mfo ./data/ATGO/ATGO_MFO_test_annotations.tsv
```

For the **H30** dataset the full SwissProt 2024_01 experimental annotations serve as the training background, with H30 test proteins automatically excluded from the IC calculation:
```sh
python background.py \
  --bpo ./data/swissprot/2024_01/swissprot_2024_01_BPO_exp_annotations.tsv \
  --cco ./data/swissprot/2024_01/swissprot_2024_01_CCO_exp_annotations.tsv \
  --mfo ./data/swissprot/2024_01/swissprot_2024_01_MFO_exp_annotations.tsv \
  --test_bpo ./data/H30/H30_BPO_test_annotations.tsv \
  --test_cco ./data/H30/H30_CCO_test_annotations.tsv \
  --test_mfo ./data/H30/H30_MFO_test_annotations.tsv \
  --output ./data/H30/background_H30.pkl
```

### 5. Running Baselines
Run the baseline methods (e.g., Naive, DiamondKNN, AlignmentScore) using the prepared data and alignment results. The following command runs the baselines on the ATGO dataset, under the constrained setup:
```sh
python main.py \
  --db_versions '' \
  --dataset ATGO \
  --alignment_dir ./data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv \
  --k_values 1 3 5 10 15 20 \
  --aspects BPO CCO MFO
```
`--dataset` can be set to either `D1` (BeProf D1 dataset), `H30` (low-homology dataset), `ATGO` or `CAFA3`.  
To greatly speed up the process, you can skip the Naive baseline by uncommenting the corresponding line in the `main.py` script.  

`--db_versions` can be set to one or more SwissProt versions, e.g. `2024_01` or `2024_01 2021_01`. If not set, the script will iterate over all available versions.  
`--alignment_dir` specifies the path to the Diamond alignment file generated in step 3.  
`--k_values` specifies the k values to use for the KNN baseline.  
`--aspects` specifies the GO subontologies to consider (BPO, CCO, MFO). Defaults to all three aspects.  
`--experimental_only` restricts training annotations to experimentally validated GO terms. Leaving this flag unset includes all manually curated annotations.  

#### Experimental Setups
![alt text](setups.png)

To execute annotation transfer from the train set to the test set (i.e. the usual setup in the literature, referred to as 'Benchmark'), set `--db_versions` to an empty string `''`.  
`--one_vs_all` will run the baselines in a 'One-vs-All' setup, where each test protein can receive annotations from the rest of the proteins in the dataset (excluding themselves), regardless of their train/test split.  
`--annotations_2024_01` will freeze annotations to the 2024_01 SwissProt release, using only proteins present in the specified `--db_versions`. This is referred to as the 'Up-to-date' setup.  
Not including any of these flags will run the baselines in the 'SwissProt' setup, where annotations are transferred from the specified SwissProt version(s) to the test proteins.

Example usage on the ATGO dataset, applied to all SwissProt versions, using experimental annotations and a one-vs-all setup:

```sh
python main.py \
  --dataset ATGO \
  --alignment_dir ./data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv \
  --k_values 1 3 5 10 15 20 \
  --aspects BPO CCO MFO \
  --experimental_only \
  --one_vs_all
```

#### STRING DB integration

For proteins that have no Diamond alignment hit (particularly relevant in the H30 low-homology dataset), STRING DB interaction scores can supplement or replace alignment-based predictions via `--stringdb`.

Required files (placed under `data/swissprot/2024_01/`):
- `swissprot_stringdb.tsv` — STRING DB interactions with `combined_score` column (0–1000)
- `idmapping_swissprot_stringdb.tsv` — mapping from UniProt accession (`From`) to STRING DB ID (`To`)

Two modes are available:

| Mode | Behaviour |
|------|-----------|
| `rescue` | Use STRING DB predictions only for proteins that received no alignment hit |
| `merge <weight>` | Linearly blend alignment and STRING DB predictions for **all** proteins; `weight` ∈ [0, 1] controls the STRING DB contribution |

```sh
# Rescue unaligned proteins with StringDB
python main.py \
  --dataset H30 \
  --alignment_dir ./data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv \
  --k_values 1 3 5 10 15 20 \
  --experimental_only \
  --stringdb rescue

# Blend alignment (70%) and StringDB (30%) for all proteins
python main.py \
  --dataset H30 \
  --alignment_dir ./data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv \
  --k_values 1 3 5 10 15 20 \
  --experimental_only \
  --stringdb merge 0.3
```

### 6. Evaluation
Evaluating predictions runs automatically when executing `main.py`.  
It can also be triggered manually with `evaluation.py`:

```sh
python evaluation.py \
  --input_dir './results/ATGO/baselines_ATGO_2024_01_BPO_exp_one_vs_all' \
  --aspect BPO \
  --k_values 1 3 5 10 15 20
```

The script selects ground truth and background files automatically based on the directory name.  

**Handling proteins without predictions:** test proteins absent from a prediction file are automatically inserted as empty predictions before evaluation. This ensures they are counted as zero-recall cases (CAFA-style penalisation) rather than being silently ignored, which would artificially inflate scores. The number of such proteins is logged during evaluation.

See the [BeProf evaluation script github](https://github.com/CSUBioGroup/BeProf/tree/main) for details on the evaluation metrics and file formats.

---

## Datasets

| Dataset | Description | Test set definition |
|---------|-------------|---------------------|
| `ATGO` | ATGO benchmark | Pre-split train/val/test files |
| `CAFA3` | CAFA3 benchmark | Pre-split train/val/test files |
| `D1` | BeProf D1 dataset | Pre-split train/test files |
| `H30` | Low-homology dataset | Proteins with no pairwise sequence identity > 30 % to any other SwissProt 2024_01 protein; includes proteins absent from the Diamond alignment entirely |

The H30 dataset is generated by `notebooks/generate_H30.ipynb`. The notebook:
1. Loads the all-vs-all Diamond alignment for SwissProt 2024_01
2. Collects all proteins from the full SwissProt 2024_01 annotation file (including proteins with zero alignment hits)
3. Removes any protein that shares > 30 % sequence identity with at least one other protein
4. Filters the resulting set against the per-ontology experimental annotation files

---

### Notes
Adjust file paths and parameters as needed for your specific setup.  
Plots can be reproduced using the Jupyter notebooks in the `notebooks/` folder.  
Additional performance could likely be further improved by tuning the scoring functions and parameters according to: [A large-scale assessment of sequence database search tools for homology-based protein function prediction](https://doi.org/10.1093/bib/bbae349).
