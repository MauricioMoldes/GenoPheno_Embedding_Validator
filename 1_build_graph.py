"""
Step 1 of 4 - build one haplograph PER CHROMOSOME from the 1000G haploblock data.

Data: https://data.haploblocks.org/haplograph/1000G/<chrom>/{nodes,edges}.csv.gz  (chr1-chr22)

  node     = haploblock cluster (id like chr21_14215892-14284114_cluster1)
  features = which of the 2,548 individuals carry it (sparse 0/1 matrix)
  edge     = co-occurrence, weighted by log(1 + lift), where
             lift = weight / (support_A * support_B / N_individuals)
             as the dataset README recommends; edges below --min-lift are dropped.

Outputs (per chromosome, in <workdir>/<chrom>/):
  graph.npz      sparse symmetric adjacency (scipy CSR)
  carriers.npz   sparse [nodes x individuals] carrier matrix
  nodes.csv.gz   id, region, start, end, support
and <workdir>/individuals.txt (same individual order for every chromosome).

Usage:
  python 1_build_graph.py                          # all chromosomes
  python 1_build_graph.py --chroms chr21,chr22     # quick test
  python 1_build_graph.py --edges lift5            # small pre-filtered edge files

Requires: numpy, pandas, scipy
"""
import argparse
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


def raw_file(args, chrom, name):
    """Use <local-dir>/<chrom>/<name> if present, otherwise download into <workdir>/raw/."""
    if args.local_dir:
        p = os.path.join(args.local_dir, chrom, name)
        if os.path.exists(p):
            return p
    return download(f"{BASE_URL}/{chrom}/{name}", os.path.join(args.workdir, "raw", chrom, name))


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
    return np.concatenate(ids), np.concatenate(regions), sp.vstack(blocks).tocsr(), inds


def read_edges(edges_path, ids, support, n_ind, min_lift, chunksize):
    """Stream an edges file, map ids -> indices, compute lift, keep lift >= min_lift."""
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


def stage_graph(args, chrom):
    if done(args, chrom, "graph.npz", "carriers.npz", "nodes.csv.gz"):
        log(f"[graph {chrom}] cached"); return
    t0 = time.time()
    edges_name = "edges_lift_above_threshold.csv.gz" if args.edges == "lift5" else "edges.csv.gz"
    ids, regions, carriers, inds = read_nodes(raw_file(args, chrom, "nodes.csv.gz"))

    # every chromosome must use the same individuals, in the same order
    ind_file = gpath(args, "individuals.txt")
    if os.path.exists(ind_file):
        ref = open(ind_file).read().split("\n")
        if set(ref) != set(inds):
            raise ValueError(f"{chrom}: individual columns differ from {ind_file}")
        pos = {c: i for i, c in enumerate(inds)}
        carriers = carriers[:, [pos[c] for c in ref]]
    else:
        with open(ind_file, "w") as f:
            f.write("\n".join(inds))

    support = np.asarray(carriers.sum(axis=1)).ravel()
    n, n_ind = carriers.shape
    s, t, lift, n_read, n_unk = read_edges(raw_file(args, chrom, edges_name), ids, support,
                                          n_ind, args.min_lift, args.chunksize)
    A = sp.coo_matrix((np.log1p(lift), (s, t)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A = A.maximum(A.T).tocsr()          # co-occurrence is symmetric
    A.setdiag(0); A.eliminate_zeros()

    parsed = pd.Series(regions).str.extract(REGION_RE)
    nodes = pd.DataFrame({
        "id": ids, "region": regions,
        "start": pd.to_numeric(parsed["start"]), "end": pd.to_numeric(parsed["end"]),
        "support": support,
    })
    if nodes["start"].isna().any():
        log(f"  warning: {nodes['start'].isna().sum():,} regions don't match <chrom>_<start>-<end>")

    sp.save_npz(cpath(args, chrom, "graph.npz"), A)
    sp.save_npz(cpath(args, chrom, "carriers.npz"), carriers)
    nodes.to_csv(cpath(args, chrom, "nodes.csv.gz"), index=False)
    deg = np.diff(A.indptr)
    log(f"[graph {chrom}] {n:,} nodes, {A.nnz // 2:,} edges ({len(s):,}/{n_read:,} rows kept"
        f"{f', {n_unk:,} with unknown ids' if n_unk else ''}), mean degree {deg.mean():.1f}, "
        f"isolated {int((deg == 0).sum()):,}  ({time.time() - t0:.0f}s)")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chroms", default=",".join(ALL_CHROMS), help="comma-separated, default chr1-chr22")
    ap.add_argument("--workdir", default="haplograph", help="shared folder used by all four scripts")
    ap.add_argument("--force", action="store_true", help="rebuild chromosomes already done")
    ap.add_argument("--edges", choices=["all", "lift5"], default="all",
                    help="all = edges.csv.gz (~1 GB per chromosome); lift5 = pre-filtered lift>=5 file")
    ap.add_argument("--min-lift", type=float, default=2.0, help="keep edges with lift >= this")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--local-dir", help="folder with already-downloaded <chrom>/nodes.csv.gz etc.")
    args = ap.parse_args()
    args.chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    for chrom in args.chroms:
        stage_graph(args, chrom)
    log("done.")


if __name__ == "__main__":
    main()
