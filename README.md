# Haplograph → Evo 2

Builds a genome-wide (or per-chromosome-selection) haploblock co-occurrence graph from the [1000 Genomes haplograph](https://data.haploblocks.org/haplograph/1000G/), turns its nodes into embeddings with node2vec, a GNN or XGBoost, and fine-tunes Evo 2 (7B) on those embeddings.

## Dependencies

- Python 3.11 or 3.12, Linux, NVIDIA GPU with CUDA 12.1+ (80 GB recommended for step 4)
- numpy, pandas, scipy (all steps)
- torch (node2vec and GNN embeddings, Evo 2)
- xgboost (XGBoost embeddings)
- evo2 + flash-attn (step 4)

## Install

```bash
pip install numpy pandas scipy torch xgboost
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
```

## Usage

Run the scripts in order. Step 1 writes the graph to `haplograph/genome/`; steps 2–4 read and write there (change with `--graph-dir`).

```bash
python 1_build_graph.py                            # whole genome (or --chroms chr21,chr22)
python 2_ancestry_labels.py                        # needed for xgboost or --objective ancestry
python 3_node_embeddings.py --method node2vec      # or gnn / xgboost
python 4_finetune_evo2.py   --method node2vec      # --objective next (default) or ancestry
```

For a quick test, build a small graph with `python 1_build_graph.py --chroms chr21,chr22` and pass `--graph-dir haplograph/chr21_chr22` to steps 2–4. Each script's header describes its input and output file formats, and `--help` lists all options.