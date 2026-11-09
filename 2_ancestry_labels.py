"""
Step 2 of 4 - ancestry labels from phenotypes_real.csv.

Needed for: 3_node_embeddings.py --method xgboost, and 4_finetune_evo2.py --objective ancestry.
(Skip it if you only use --method gnn with --objective next.)

  per individual : super-population ("ancestry" phenotype) -> <workdir>/individual_ancestry.csv
  per node       : the super-population most ENRICHED among the cluster's carriers (enrichment,
                   not majority, so the largest population doesn't win everywhere)
                   -> <workdir>/<chrom>/labels.npz

phenotypes_real.csv columns are auto-detected (long or wide format). If its sample ids don't
match the IND* columns of nodes.csv.gz, pass --sample-map (csv: node column name, sample id).

Usage:
  python 2_ancestry_labels.py
  python 2_ancestry_labels.py --phenotypes phenotypes_real.csv --chroms chr21,chr22

Requires: numpy, pandas, scipy.  Run 1_build_graph.py first.
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_URL = "https://data.haploblocks.org/haplograph/1000G"
ALL_CHROMS = [f"chr{i}" for i in range(1, 23)]  # chrX/chrY are not in the dataset


def log(msg=""):
    print(msg, flush=True)


def gpath(args, name):
    """File shared by all chromosomes."""
    return os.path.join(args.workdir, name)


def cpath(args, chrom, name):
    """Per-chromosome file."""
    d = os.path.join(args.workdir, chrom)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def done(args, chrom, *names):
    return not args.force and all(os.path.exists(cpath(args, chrom, n)) for n in names)


def download(url, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log(f"  downloading {url}")
    tmp = dest + ".part"

    def hook(blocks, bs, total):
        if total > 0:
            sys.stdout.write(f"\r    {min(blocks * bs, total) / 1e6:9.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.replace(tmp, dest)
    log()
    return dest


def require(args, chrom, *names, hint):
    missing = [n for n in names if not os.path.exists(cpath(args, chrom, n))]
    if missing:
        raise SystemExit(f"{chrom}: missing {', '.join(missing)} in {args.workdir}/{chrom}/ - run {hint} first")


def individual_ancestry(args):
    """Ancestry label per individual, aligned to individuals.txt (None if unknown). Cached."""
    individuals = open(gpath(args, "individuals.txt")).read().split("\n")
    cache = gpath(args, "individual_ancestry.csv")
    if os.path.exists(cache) and not args.force:
        a = pd.read_csv(cache, dtype=str, keep_default_na=False)
        return individuals, np.array([v if v else None for v in a["ancestry"]], dtype=object)

    pheno_path = args.phenotypes or download(f"{BASE_URL}/phenotypes_real.csv",
                                             gpath(args, "raw/phenotypes_real.csv"))
    ph = pd.read_csv(pheno_path, dtype=str)
    inds = set(individuals)
    to_col = None
    if args.sample_map:  # optional csv: node-file column name, phenotype sample id
        m = pd.read_csv(args.sample_map, dtype=str)
        to_col = dict(zip(m.iloc[:, 1], m.iloc[:, 0]))

    def overlap(col):
        vals = ph[col].dropna()
        if to_col is not None:
            vals = vals.map(to_col).dropna()
        return vals.isin(inds).mean()

    id_col = args.pheno_id_col or max(ph.columns, key=overlap)
    if overlap(id_col) == 0:
        raise ValueError(
            "No phenotype rows match the individual columns in nodes.csv.gz "
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
            raise ValueError(f"no rows with phenotype '{args.ancestry_key}' in column '{name_col}'. "
                             f"Values seen: {ph[name_col].unique()[:10]}. Use --ancestry-key.")
        ids, values = ids[rows], ph.loc[rows, value_col]

    lookup = dict(zip(ids, values))
    anc = np.array([lookup.get(i) for i in individuals], dtype=object)
    pd.DataFrame({"individual": individuals,
                  "ancestry": [a if a is not None else "" for a in anc]}).to_csv(cache, index=False)
    known = sum(a is not None for a in anc)
    log(f"  ancestry known for {known:,}/{len(individuals):,} individuals")
    return individuals, anc


def stage_labels(args, chrom):
    if done(args, chrom, "labels.npz"):
        log(f"[labels {chrom}] cached"); return
    _, anc = individual_ancestry(args)
    carriers = sp.load_npz(cpath(args, chrom, "carriers.npz")).astype(np.float32)
    known = np.array([a is not None for a in anc])
    classes = np.array(sorted(set(anc[known])))
    onehot = np.zeros((len(anc), len(classes)), np.float32)
    onehot[np.where(known)[0], np.searchsorted(classes, anc[known])] = 1

    counts = np.asarray(carriers @ onehot)                     # [N, P] carriers per population
    frac = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    enrichment = frac / (onehot.sum(0) / onehot.sum())
    labels = np.where(counts.sum(1) > 0, enrichment.argmax(1), -1)
    np.savez(cpath(args, chrom, "labels.npz"), labels=labels, classes=classes)
    dist = pd.Series(classes[labels[labels >= 0]]).value_counts().to_dict()
    log(f"[labels {chrom}] {dist}  (unlabelled {(labels < 0).sum():,})")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chroms", default=",".join(ALL_CHROMS), help="comma-separated, default chr1-chr22")
    ap.add_argument("--workdir", default="haplograph", help="shared folder used by all four scripts")
    ap.add_argument("--force", action="store_true", help="recompute existing labels")
    ap.add_argument("--phenotypes", help="local phenotypes_real.csv (downloaded if omitted)")
    ap.add_argument("--ancestry-key", default="ancestry", help="phenotype to use as the label")
    ap.add_argument("--pheno-id-col"); ap.add_argument("--pheno-name-col"); ap.add_argument("--pheno-value-col")
    ap.add_argument("--sample-map", help="csv mapping node-file individual column -> phenotype sample id")
    args = ap.parse_args()
    args.chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    if not os.path.exists(gpath(args, "individuals.txt")):
        raise SystemExit(f"{args.workdir}/individuals.txt not found - run 1_build_graph.py first")
    for chrom in args.chroms:
        require(args, chrom, "carriers.npz", hint="1_build_graph.py")
        stage_labels(args, chrom)
    log("done.")


if __name__ == "__main__":
    main()
