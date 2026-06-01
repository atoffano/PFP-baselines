# Protein Function Prediction: Capabilities and Limitations of Alignment-based Approaches

![Annotation Transfer](annotation_transfer.svg)

This repository contains the code and data processing pipelines for our study reassessing the capabilities of alignment-based annotation transfer in Protein Function Prediction (PFP). 

Computational prediction of protein function is increasingly dominated by deep learning architectures. However, our findings demonstrate that homology-based methods can surpass state-of-the-art deep learning models when provided with appropriate contextual data. This repository provides the framework we used to explore and mitigate the primary failure point of homology methods (_i.e._ when there is either no protein to transfer from, or no function to transfer).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Reproduction Steps](#reproduction-steps)
  - [1. Data Downloading & Preparation](#1-data-downloading--preparation)
  - [2. Annotation Propagation](#2-annotation-propagation)
  - [3. Alignment](#3-alignment)
  - [4. Preparing the evaluation](#4-preparing-the-evaluation)
  - [5. Running Baselines & Proposed Enhancements](#5-running-baselines--proposed-enhancements)
  - [6. Evaluation](#6-evaluation)
- [Datasets](#datasets)
- [Notes](#notes)

---

## Overview

This repository provides a pipeline covering preprocessing steps (data downloading, annotation propagation, alignment using Diamond), running prediction algorithms and evaluating the alignment-based scoring methods discussed in the article:
- **IDScore (Sequence Identity-Based)**: Transferring GO annotations from the highest sequence identity match.
- **DiamondScore (Bit-Score Aggregation)**: A multi-template method aggregating annotations weighted by bitscores.
- **DiamondKNN**: A top-$k$ constrained variant of DiamondScore.

Advanced configurations like One-vs-All (OVA) alignment, integration of curated Labels (CUR), and STRING interactions are also supported through simple input flags.

## Requirements

The following Python packages are required:
```sh
pip install obonet networkx tqdm pandas scipy biopython matplotlib seaborn pickle
```

Additionally, the [Diamond software](http://github.com/bbuchfink/diamond) is required for sequence alignment.
On Linux-based systems:

```sh
wget http://github.com/bbuchfink/diamond/releases/download/v2.1.11/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
```

## Reproduction Steps

### 1. Data Downloading & Preparation

Download necessary information (e.g., protein sequences, GO annotations) from UniProt (SwissProt) at different releases and parse them into required formats:

```sh
python download_swissprot.py
```

If you wish to use functionalities that rely on STRING protein interaction data, download with:
```sh
python download_stringdb.py
```
*Note: This can take a while.*

### 2. Annotation Propagation

Propagate GO annotations from specific terms to broader parent terms using the ontology structure:

```sh
python propagate_swissprot_terms.py
```

### 3. Alignment

Run sequence alignment using Diamond. This step finds homologous proteins required for annotation transfer:

```sh
echo "Creating Diamond database..."
diamond makedb --in data/2024_01/swissprot_2024_01.fasta -d data/2024_01/swissprot_2024_01_proteins_set

echo "Running Diamond blast on protein sequences against themselves..."
diamond blastp --very-sensitive --db data/swissprot/2024_01/swissprot_2024_01_proteins_set.dmnd --query data/swissprot/2024_01/swissprot_2024_01.fasta --out data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv -e 0.001
```
*Note: The all-vs-all alignment for the entire SwissProt database (~570,000 proteins) takes about 1 hour, depending on hardware.*

### 4. Preparing the evaluation

To more accurately evaluate the predictive performance, we use $IC$-weighted scores based on the Information Content ($IC$) of the GO terms within the dataset. Background files must be generated prior to running predictions.

Example for the ATGO dataset:
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



### 5. Running Algorithms

```sh
python main.py \
  --db_versions '' \
  --dataset ATGO \
  --k_values 1 3 5 \
  --aspects BPO CCO MFO
```

**Parameters**:
- `--dataset`: `D1` (BeProf D1), `H30` (low-homology), `ATGO`, or `CAFA3`.
- `--db_versions`: One or more SwissProt versions (e.g., `2024_01`). Set to empty string `''` to execute annotation transfer strictly from the benchmark train set to the test set. Note that not passing the flag will run on all available SwissProt versions, based on the `SWISSPROT_VERSIONS` constant.
- `--k_values`: Limits for KNN baseline (e.g., `DiamondKNN(k=1)`).
- `--aspects`: Restrict GO subontologies (`BPO`, `CCO`, `MFO`).
- `--experimental_only`: Restricts transfered annotations to experimental GO terms.
- `--eval`: Evaluation script to use, can be either `cafa` or `beprof`. Use cafa if in doubt.
- `--norm`: Normalization mode used by CAFA evaluator, can be either `cafa` or `pred` or `gt`.

To reproduce enhancements proposed in the paper, add the following flags:
- **DS – OVA** (One-vs-all): Include `--one_vs_all` to allow all proteins in the dataset to contribute annotations to the target.
- **DS – CUR** (Curated): Omit `--experimental_only` to incorporate non-experimental manually reviewed labels.
- **DS – STRING**: Add `--stringdb_weight` followed by the weight (between 0 and 1) to balance STRING interaction contribution predictions.

Example:
```sh
# Blend alignment (50%) and StringDB (50%) for all proteins
# Will run DiamondScore, IDScore and DiamondKNN(k=1,3,5) on the H30 dataset at release 2024_01
python main.py \
  --dataset H30 \
  --db_versions 2024_01 \
  --k_values 1 3 5 \
  --stringdb_weight 0.5
  --experimental_only
```

### 6. Evaluation

Evaluation of predictions is run automatically during `main.py`, logging $F_{max}$ and $S_{min}$ for each method.

You can also manually evaluate:

```sh
python evaluation.py \
  --input_dir './results/ATGO/baselines_ATGO_2024_01_BPO_exp_one_vs_all' \
  --aspect BPO \
  --k_values 1 3 5
```
