"""
Step 1 of 4 - build the haplograph: GENOME-WIDE by default (chr1-chr22 merged into one graph),
or for selected chromosomes only (--chroms).

INPUT  (downloaded from https://data.haploblocks.org/haplograph/1000G/<chrom>/, or read from
        --local-dir/<chrom>/ if you already have them)
  nodes.csv.gz   gzipped CSV, one row per haploblock cluster (= graph node):
                   id             cluster id "<chrom>_<start>-<end>_cluster<N>",
                                  e.g. chr21_14215892-14284114_cluster1
                   high_dim_edge  the haploblock region "<chrom>_<start>-<end>"
                   IND1 ... INDk  one 0/1 column per individual (2,548 in 1000G),
                                  1 = this individual carries the cluster
  edges.csv.gz   gzipped CSV, one row per co-occurring pair of clusters (same chromosome only):
                   source, target  cluster ids (as in nodes.csv.gz)
                   weight          int, number of individuals carrying both clusters
                 (--edges lift5 reads edges_lift_above_threshold.csv.gz instead: same columns
                  plus a lift column, only pairs with lift >= 5; ~16 MB instead of ~1 GB)

OUTPUT (in --out; default <workdir>/genome, or <workdir>/<chrA>_<chrB>... for a selection)
  nodes.csv.gz     gzipped CSV, N rows; row i describes node i in every other file:
                     node     int, 0..N-1
                     id       cluster id
                     chrom    e.g. chr21
                     region   haploblock region
                     start, end   int, bp, parsed from region
                     support  int, number of individuals carrying the cluster
  individuals.txt  I lines, one individual (IND* column name) per line; column j of
                   carriers.npz is line j. Same order for every chromosome.
  carriers.npz     scipy.sparse CSR, uint8, shape [N, I]; 1 = individual carries the cluster
  graph.npz        scipy.sparse CSR, float32, shape [N, N]; symmetric, zero diagonal;
                   value = log(1 + lift), lift = weight / (support_A * support_B / I),
                   as the dataset README recommends. Pairs with lift < --min-lift are dropped.
                   The data only has within-chromosome edges, so the genome-wide graph is made
                   of one disconnected block per chromosome.
  graph_info.json  chromosomes, settings, node and edge counts

  Load with:  scipy.sparse.load_npz("graph.npz"),  pandas.read_csv("nodes.csv.gz")

Parsed chromosomes are cached in <workdir>/parts/<chrom>/ (an interrupted genome-wide run resumes
where it stopped, and a later selection reuses them); raw downloads go to <workdir>/raw/<chrom>/.

USAGE
  python 1_build_graph.py                                  # whole genome -> haplograph/genome
  python 1_build_graph.py --chroms chr21,chr22             # -> haplograph/chr21_chr22
  python 1_build_graph.py --edges lift5                    # small pre-filtered edge files

REQUIRES  numpy, pandas, scipy
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_URL = "https://data.haploblocks.org/haplograph/1000G"
ALL_CHROMS = [f"chr{i}" for i in range(1, 23)]  # chrX/chrY are not in the dataset
REGION_RE = re.compile(r"^(?P<chrom>.+)_(?P<start>\d+)-(?P<end>\d+)$")
OUTPUTS = ("nodes.csv.gz", "individuals.txt", "carriers.npz", "graph.npz", "graph_info.json")


def log(msg=""):
    print(msg, flush=True)


def chrom_key(c):
    m = re.fullmatch(r"chr(\d+)", c)
    return (0, int(m.group(1)), "") if m else (1, 0, c)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

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


def raw_file(args, chrom, name):
    """Use <local-dir>/<chrom>/<name> if present, otherwise download into <workdir>/raw/<chrom>/."""
    if args.local_dir:
        p = os.path.join(args.local_dir, chrom, name)
        if os.path.exists(p):
            return p
    return download(f"{BASE_URL}/{chrom}/{name}", os.path.join(args.workdir, "raw", chrom, name))


# ---------------------------------------------------------------------------
# Parse one chromosome
# ---------------------------------------------------------------------------

def read_nodes(nodes_path, chunksize=20_000):
    header = pd.read_csv(nodes_path, nrows=0).columns.tolist()
    inds = [c for c in header if c not in ("id", "high_dim_edge")]
    dtypes = {c: np.uint8 for c in inds}
    dtypes.update({"id": str, "high_dim_edge": str})
    ids, regions, blocks = [], [], []
    for chunk in pd.read_csv(nodes_path, dtype=dtypes, chunksize=chunksize):
        ids.append(chunk["id"].to_numpy())
        regions.append(chunk["high_dim_edge"].to_numpy())
        blocks.append(sp.csr_matrix(chunk[inds].to_numpy(dtype=np.uint8)))
    ids = np.concatenate(ids)
    if len(pd.unique(ids)) != len(ids):
        raise ValueError(f"{nodes_path}: duplicate cluster ids")
    return ids, np.concatenate(regions), sp.vstack(blocks).tocsr(), inds


def read_edges(edges_path, ids, support, n_ind, min_lift, chunksize):
    """Stream an edges file, map ids -> row indices, compute lift, keep lift >= min_lift."""
    index = pd.Index(ids)
    S, T, L = [], [], []
    n_read = n_unknown = 0
    for chunk in pd.read_csv(edges_path, usecols=["source", "target", "weight"],
                             dtype={"source": str, "target": str, "weight": np.float64},
                             chunksize=chunksize):
        s = index.get_indexer(chunk["source"])
        t = index.get_indexer(chunk["target"])
        w = chunk["weight"].to_numpy()
        n_read += len(chunk)
        n_unknown += int(((s < 0) | (t < 0)).sum())
        ok = (s >= 0) & (t >= 0) & (s != t)
        s, t, w = s[ok], t[ok], w[ok]
        expected = support[s].astype(np.float64) * support[t] / n_ind
        lift = np.divide(w, expected, out=np.zeros_like(w), where=expected > 0)
        keep = lift >= min_lift
        S.append(s[keep].astype(np.int64)); T.append(t[keep].astype(np.int64))
        L.append(lift[keep].astype(np.float32))
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.empty(0, dt)
    return cat(S, np.int64), cat(T, np.int64), cat(L, np.float32), n_read, n_unknown


def chrom_part(args, chrom):
    """Parse one chromosome; cached in <workdir>/parts/<chrom>/."""
    d = os.path.join(args.workdir, "parts", chrom)
    os.makedirs(d, exist_ok=True)
    edges_name = "edges_lift_above_threshold.csv.gz" if args.edges == "lift5" else "edges.csv.gz"
    f_meta, f_car = os.path.join(d, "nodes.npz"), os.path.join(d, "carriers.npz")
    f_edges = os.path.join(d, f"edges_{args.edges}_minlift{args.min_lift:g}.npz")

    if not args.force and os.path.exists(f_meta) and os.path.exists(f_car):
        meta = np.load(f_meta)
        ids, regions, inds = meta["ids"], meta["regions"], meta["individuals"].tolist()
        carriers = sp.load_npz(f_car).tocsr()
    else:
        ids, regions, carriers, inds = read_nodes(raw_file(args, chrom, "nodes.csv.gz"))
        sp.save_npz(f_car, carriers)
        np.savez(f_meta, ids=ids.astype(str), regions=regions.astype(str),
                 individuals=np.array(inds, dtype=str))

    support = np.asarray(carriers.sum(axis=1)).ravel()
    if not args.force and os.path.exists(f_edges):
        e = np.load(f_edges)
        s, t, lift = e["s"], e["t"], e["lift"]
        status = "cached"
    else:
        s, t, lift, n_read, n_unk = read_edges(raw_file(args, chrom, edges_name), ids, support,
                                              carriers.shape[1], args.min_lift, args.chunksize)
        np.savez(f_edges, s=s, t=t, lift=lift)
        status = f"{len(s):,}/{n_read:,} edge rows kept" + (f", {n_unk:,} with unknown ids" if n_unk else "")
    return ids, regions, carriers, inds, s, t, lift, status


# ---------------------------------------------------------------------------
# Merge into one graph
# ---------------------------------------------------------------------------

def build(args):
    chroms = sorted(set(args.chroms), key=chrom_key)
    out = args.out
    if not args.force and all(os.path.exists(os.path.join(out, f)) for f in OUTPUTS):
        log(f"graph already built in {out} (use --force to rebuild)")
        return
    os.makedirs(out, exist_ok=True)
    log(f"building graph for {len(chroms)} chromosome(s) -> {out}")

    ref_inds = None
    ids_all, regions_all, chrom_all, carrier_blocks, S, T, W = [], [], [], [], [], [], []
    offset = 0
    for chrom in chroms:
        t0 = time.time()
        ids, regions, carriers, inds, s, t, lift, status = chrom_part(args, chrom)
        if ref_inds is None:
            ref_inds = inds
        elif inds != ref_inds:
            if set(inds) != set(ref_inds):
                raise ValueError(f"{chrom}: individual columns differ from {chroms[0]}")
            pos = {c: j for j, c in enumerate(inds)}
            carriers = carriers[:, [pos[c] for c in ref_inds]].tocsr()
        ids_all.append(ids); regions_all.append(regions); chrom_all.append(np.full(len(ids), chrom))
        carrier_blocks.append(carriers)
        S.append(s + offset); T.append(t + offset); W.append(np.log1p(lift))
        offset += len(ids)
        log(f"  {chrom}: {len(ids):,} nodes, {status}  ({time.time() - t0:.0f}s)")

    n = offset
    A = sp.coo_matrix((np.concatenate(W).astype(np.float32),
                       (np.concatenate(S), np.concatenate(T))), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A = A.maximum(A.T).tocsr()          # co-occurrence is symmetric
    A.setdiag(0); A.eliminate_zeros(); A.sort_indices()
    carriers = sp.vstack(carrier_blocks).tocsr().astype(np.uint8)

    regions = np.concatenate(regions_all)
    parsed = pd.Series(regions).str.extract(REGION_RE)
    nodes = pd.DataFrame({
        "node": np.arange(n), "id": np.concatenate(ids_all), "chrom": np.concatenate(chrom_all),
        "region": regions,
        "start": pd.to_numeric(parsed["start"]).astype("Int64"),
        "end": pd.to_numeric(parsed["end"]).astype("Int64"),
        "support": np.asarray(carriers.sum(axis=1)).ravel(),
    })
    if nodes["start"].isna().any():
        log(f"  warning: {int(nodes['start'].isna().sum()):,} regions don't match <chrom>_<start>-<end>")

    nodes.to_csv(os.path.join(out, "nodes.csv.gz"), index=False)
    with open(os.path.join(out, "individuals.txt"), "w") as f:
        f.write("\n".join(ref_inds))
    sp.save_npz(os.path.join(out, "carriers.npz"), carriers)
    sp.save_npz(os.path.join(out, "graph.npz"), A)
    deg = np.diff(A.indptr)
    info = {"chromosomes": chroms, "edges_file": args.edges, "min_lift": args.min_lift,
            "edge_value": "log1p(lift)", "n_nodes": int(n), "n_edges": int(A.nnz // 2),
            "n_individuals": len(ref_inds), "isolated_nodes": int((deg == 0).sum())}
    with open(os.path.join(out, "graph_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    log(f"  graph: {n:,} nodes, {A.nnz // 2:,} edges, {len(ref_inds):,} individuals, "
        f"mean degree {deg.mean():.1f}, isolated {int((deg == 0).sum()):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chroms", help="comma-separated selection, e.g. chr21,chr22 (default: whole genome)")
    ap.add_argument("--workdir", default="haplograph", help="root folder for downloads, cache and graphs")
    ap.add_argument("--out", help="output folder (default <workdir>/genome or <workdir>/<chroms>)")
    ap.add_argument("--edges", choices=["all", "lift5"], default="all",
                    help="all = edges.csv.gz (~1 GB per chromosome); lift5 = pre-filtered lift>=5 file")
    ap.add_argument("--min-lift", type=float, default=2.0, help="keep edges with lift >= this")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="edge rows read at a time")
    ap.add_argument("--local-dir", help="folder with already-downloaded <chrom>/nodes.csv.gz etc.")
    ap.add_argument("--force", action="store_true", help="re-parse cached chromosomes and rebuild")
    args = ap.parse_args()

    if args.chroms:
        args.chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
        name = "_".join(sorted(set(args.chroms), key=chrom_key))
    else:
        args.chroms, name = ALL_CHROMS, "genome"
    args.out = args.out or os.path.join(args.workdir, name)
    build(args)
    log("done.")


if __name__ == "__main__":
    main()