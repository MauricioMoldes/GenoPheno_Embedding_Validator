# Haplograph → Evo 2

<<<<<<< HEAD
Builds a genome-wide (or per-chromosome-selection) haploblock co-occurrence graph from the [1000 Genomes haplograph](https://data.haploblocks.org/haplograph/1000G/), turns its nodes into embeddings with node2vec, a GNN or XGBoost, and fine-tunes Evo 2 (7B) on those embeddings.
=======
Genome Cluster Engine Project

Established and optimized the haploblocks method for discretizing genomic data (https://osf.io/xhkc3)
Generated binary strings from haploblock clusters and did some basic linear modeling. Leveraged XGBoost and deep learning. Results were promising but not conclusive.
Processed the 1000 genomes data set with the haploblocks method (paper submitted)
Established and optimized the haplographs method (manuscript in preparation)

Here’s what we are currently working on.

Establishing what use cases respond better to a simpler clustering method (linclust)
Establishing what use cases respond best to:
Associated binary strings of haploblocks going straight into a DNA foundation model
Using XGBoost to come up with cogent groups of haploblocks during analysis
Building a haplograph and using node2vec to process the phenotype-genotype relationships
Building a haplograph and using a GNN (PyTorch) to isolate subnetworks and create contextualized vectors
Using XGBoost to come up with cogent groups of haploblocks during analysis of haplograph data
Testing all of the above conditions in both cancer and germline disease
Testing in what conditions early fusion or late fusion of proteomics data with genomic data and phenotype but is more effective


## Testing pertubations of the Haplograph --> Evo2 genomic model to see what works best in particular settings.  

As a POC, builds a haploblock co-occurrence graph per chromosome from the [1000 Genomes haplograph](https://data.haploblocks.org/haplograph/1000G/), turns its nodes into embeddings with a GNN or XGBoost, and fine-tunes Evo 2 (7B) on those embeddings

For more see: [haploblocks.org](haploblocks.org)
>>>>>>> a9981ef5e84966f20c41f9de122e5e1646b4b54d

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

<<<<<<< HEAD
For a quick test, build a small graph with `python 1_build_graph.py --chroms chr21,chr22` and pass `--graph-dir haplograph/chr21_chr22` to steps 2–4. Each script's header describes its input and output file formats, and `--help` lists all options.
=======
Add `--chroms chr21,chr22` to any step for a quick test run, and `--help` to see all options.
>>>>>>> a9981ef5e84966f20c41f9de122e5e1646b4b54d
