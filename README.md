# GenoPheno_Embedding_Validator

Idea: Haplograph --> Evo2 genomic model

Builds a haploblock co-occurrence graph per chromosome from the [1000 Genomes haplograph](https://data.haploblocks.org/haplograph/1000G/), turns its nodes into embeddings with a GNN or XGBoost, and fine-tunes Evo 2 (7B) on those embeddings.

For more see: [haploblocks.org](haploblocks.org)

## Dependencies

- Python 3.11 or 3.12, Linux, NVIDIA GPU with CUDA 12.1+ (80 GB recommended for step 4)
- numpy, pandas, scipy (all steps)
- torch (GNN embeddings, Evo 2)
- xgboost (XGBoost embeddings)
- evo2 + flash-attn (step 4)

## Install

```bash
pip install numpy pandas scipy torch xgboost
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
```

## Usage

Run the scripts in order. Each one writes to `haplograph/` (change with `--workdir`).

```bash
python 1_build_graph.py                       # download data, build one graph per chromosome
python 2_ancestry_labels.py                   # ancestry labels (needed for xgboost or --objective ancestry)
python 3_node_embeddings.py --method gnn      # or --method xgboost
python 4_finetune_evo2.py   --method gnn      # --objective next (default) or ancestry
```

Add `--chroms chr21,chr22` to any step for a quick test run, and `--help` to see all options.