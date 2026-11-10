"""
Step 2 of 4 - ancestry labels for the individuals and the nodes of the graph from 1_build_graph.py.

Needed for: 3_node_embeddings.py --method xgboost, and 4_finetune_evo2.py --objective ancestry.
Skip it if you only use node2vec / gnn with --objective next.

  per individual : super-population (the "ancestry" phenotype)
  per node       : the super-population most ENRICHED among the cluster's carriers
                   (enrichment = share of carriers from a population / that population's share of
                   all individuals; enrichment, not majority, so the largest population doesn't win
                   everywhere)

INPUT
  --graph-dir/individuals.txt   I lines, one individual (IND* column name) per line
  --graph-dir/carriers.npz      scipy.sparse CSR uint8 [N, I], node x individual carriers
  phenotypes_real.csv           (--phenotypes, or downloaded from the dataset root)
                                CSV of real labels from IGSR (ancestry, population, sex).
                                Long format (one row per individual-phenotype pair: sample id,
                                phenotype name, value) or wide format (one column per phenotype)
                                are both accepted; columns are auto-detected and can be set with
                                --pheno-id-col / --pheno-name-col / --pheno-value-col.
  --sample-map (optional)       CSV with 2 columns: IND* column name, phenotype sample id;
                                needed only if the phenotype ids differ from the IND* names.

OUTPUT (in --graph-dir)
  individual_ancestry.csv  CSV, I rows in individuals.txt order:
                             individual  IND* column name
                             ancestry    super-population label, empty if unknown
  labels.npz               numpy archive:
                             labels      int64 [N], index into classes, -1 = no labelled carriers
                             classes     str [P], sorted population labels
                             enrichment  float32 [N, P], per-node enrichment of each population

USAGE
  python 2_ancestry_labels.py
  python 2_ancestry_labels.py --graph-dir haplograph/chr21_chr22 --phenotypes phenotypes_real.csv

REQUIRES  numpy, pandas, scipy.  Run 1_build_graph.py first.
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

PHENO_URL = "https://data.haploblocks.org/haplograph/1000G/phenotypes_real.csv"


def log(msg=""):
    print(msg, flush=True)


def gfile(args, name):
    return os.path.join(args.graph_dir, name)


def download(url, dest):
    if os.path.exists(dest):
        return dest
    log(f"  downloading {url}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def individual_ancestry(args, individuals):
    """Ancestry label per individual, aligned to `individuals` (None where unknown)."""
    pheno_path = args.phenotypes or download(PHENO_URL, gfile(args, "phenotypes_real.csv"))
    ph = pd.read_csv(pheno_path, dtype=str)
    inds = set(individuals)
    to_col = None
    if args.sample_map:
        m = pd.read_csv(args.sample_map, dtype=str)
        to_col = dict(zip(m.iloc[:, 1], m.iloc[:, 0]))

    def overlap(col):
        vals = ph[col].dropna()
        if to_col is not None:
            vals = vals.map(to_col).dropna()
        return vals.isin(inds).mean()

    id_col = args.pheno_id_col or max(ph.columns, key=overlap)
    if overlap(id_col) == 0:
        raise SystemExit(
            "No phenotype rows match the individual columns of nodes.csv.gz "
            f"(e.g. {individuals[:3]}). Pass --pheno-id-col and/or --sample-map.")

    ids = ph[id_col] if to_col is None else ph[id_col].map(to_col)
    key = args.ancestry_key.lower()
    others = [c for c in ph.columns if c != id_col]
    wide = [c for c in others if c.lower() == key]
    if wide:                                   # wide format: a column called e.g. "ancestry"
        values = ph[wide[0]]
    else:                                      # long format: (id, phenotype name, value)
        name_col = args.pheno_name_col or max(
            others, key=lambda c: ph[c].astype(str).str.lower().eq(key).sum())
        value_col = args.pheno_value_col or next(c for c in others if c != name_col)
        rows = ph[name_col].astype(str).str.lower().eq(key)
        if not rows.any():
            raise SystemExit(f"no rows with phenotype '{args.ancestry_key}' in column '{name_col}'. "
                             f"Values seen: {ph[name_col].unique()[:10]}. Use --ancestry-key.")
        ids, values = ids[rows], ph.loc[rows, value_col]

    lookup = dict(zip(ids, values))
    return np.array([lookup.get(i) if isinstance(lookup.get(i), str) else None for i in individuals],
                    dtype=object)


def node_labels(carriers, anc):
    known = np.array([a is not None for a in anc])
    classes = np.array(sorted(set(anc[known])))
    onehot = np.zeros((len(anc), len(classes)), np.float32)
    onehot[np.where(known)[0], np.searchsorted(classes, anc[known])] = 1
    counts = np.asarray(carriers @ onehot)                     # [N, P] carriers per population
    frac = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    enrichment = (frac / (onehot.sum(0) / onehot.sum())).astype(np.float32)
    labels = np.where(counts.sum(1) > 0, enrichment.argmax(1), -1).astype(np.int64)
    return labels, classes, enrichment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", default="haplograph/genome", help="output folder of 1_build_graph.py")
    ap.add_argument("--phenotypes", help="local phenotypes_real.csv (downloaded if omitted)")
    ap.add_argument("--ancestry-key", default="ancestry", help="phenotype to use as the label")
    ap.add_argument("--pheno-id-col"); ap.add_argument("--pheno-name-col"); ap.add_argument("--pheno-value-col")
    ap.add_argument("--sample-map", help="csv: IND* column name, phenotype sample id")
    ap.add_argument("--force", action="store_true", help="recompute existing labels")
    args = ap.parse_args()

    for f in ("individuals.txt", "carriers.npz"):
        if not os.path.exists(gfile(args, f)):
            raise SystemExit(f"missing {f} in {args.graph_dir} - run 1_build_graph.py first")
    if not args.force and all(os.path.exists(gfile(args, f)) for f in ("individual_ancestry.csv", "labels.npz")):
        log(f"labels already in {args.graph_dir} (use --force to recompute)")
        return

    individuals = open(gfile(args, "individuals.txt")).read().split("\n")
    anc = individual_ancestry(args, individuals)
    pd.DataFrame({"individual": individuals,
                  "ancestry": [a or "" for a in anc]}).to_csv(gfile(args, "individual_ancestry.csv"), index=False)
    log(f"individuals: ancestry known for {sum(a is not None for a in anc):,}/{len(individuals):,}")

    carriers = sp.load_npz(gfile(args, "carriers.npz")).astype(np.float32)
    labels, classes, enrichment = node_labels(carriers, anc)
    np.savez(gfile(args, "labels.npz"), labels=labels, classes=classes, enrichment=enrichment)
    dist = pd.Series(classes[labels[labels >= 0]]).value_counts().to_dict()
    log(f"nodes: {dist}  (unlabelled {(labels < 0).sum():,})")
    log("done.")


if __name__ == "__main__":
    main()