"""
Step 3 of 4 - node embeddings for each chromosome's graph, one model per chromosome.

  --method gnn     : 2-layer GCN graph autoencoder (link reconstruction with negative sampling);
                     input features = SVD of the carrier matrix.
  --method xgboost : XGBoost predicts each node's ancestry label (from 2_ancestry_labels.py) from
                     [own carrier profile, neighbourhood-averaged profile]; the leaves each node
                     falls into across all trees are one-hot encoded and compressed with SVD.

Output: <workdir>/<chrom>/node_emb_<method>.npy   [nodes x --dim]

Usage:
  python 3_node_embeddings.py --method gnn
  python 3_node_embeddings.py --method xgboost --chroms chr21,chr22

Requires: numpy, pandas, scipy, torch (gnn) or xgboost (xgboost).
Run 1_build_graph.py first (and 2_ancestry_labels.py for xgboost).
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

ALL_CHROMS = [f"chr{i}" for i in range(1, 23)]  # chrX/chrY are not in the dataset


def log(msg=""):
    print(msg, flush=True)


def cpath(args, chrom, name):
    """Per-chromosome file."""
    d = os.path.join(args.workdir, chrom)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def done(args, chrom, *names):
    return not args.force and all(os.path.exists(cpath(args, chrom, n)) for n in names)


def require(args, chrom, *names, hint):
    missing = [n for n in names if not os.path.exists(cpath(args, chrom, n))]
    if missing:
        raise SystemExit(f"{chrom}: missing {', '.join(missing)} in {args.workdir}/{chrom}/ - run {hint} first")


def carrier_features(args, chrom):
    """SVD-compressed, standardized carrier profiles [N, svd_dim] (cached)."""
    f = cpath(args, chrom, f"features_svd{args.svd_dim}.npy")
    if os.path.exists(f) and not args.force:
        return np.load(f)
    C = sp.load_npz(cpath(args, chrom, "carriers.npz")).astype(np.float32)
    k = min(args.svd_dim, min(C.shape) - 1)
    U, S, _ = svds(C, k=k, random_state=args.seed)
    X = (U * S).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    np.save(f, X)
    return X


def normalized_adjacency(A):
    A_hat = A + sp.identity(A.shape[0], dtype=np.float32, format="csr")
    d = np.asarray(A_hat.sum(1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    return (Dm @ A_hat @ Dm).tocsr().astype(np.float32)


def leaf_embedding(leaves, dim, seed=0):
    """One-hot encode leaf indices per tree ([N, n_trees]) and compress with SVD -> [N, dim]."""
    n, n_trees = leaves.shape
    cols, offset = np.empty_like(leaves, dtype=np.int64), 0
    for j in range(n_trees):
        _, inv = np.unique(leaves[:, j], return_inverse=True)
        cols[:, j] = inv + offset
        offset += inv.max() + 1
    rows = np.repeat(np.arange(n), n_trees)
    M = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols.ravel())), shape=(n, offset))
    k = min(dim, min(M.shape) - 1)
    U, S, _ = svds(M, k=k, random_state=seed)
    return (U * S)[:, ::-1].astype(np.float32).copy()


def embed_xgboost(args, chrom, X, A_norm):
    import xgboost as xgb
    lab = np.load(cpath(args, chrom, "labels.npz"), allow_pickle=True)
    y, classes = lab["labels"], lab["classes"]

    feats = np.hstack([X, A_norm @ X])          # own profile + neighbourhood average
    idx = np.where(y >= 0)[0]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = max(1, int(0.1 * len(idx)))
    val, tr = idx[:n_val], idx[n_val:]

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    clf = xgb.XGBClassifier(
        n_estimators=args.xgb_trees, max_depth=args.xgb_depth, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=device,
        objective="multi:softprob", eval_metric="mlogloss", random_state=args.seed,
    )
    clf.fit(feats[tr], y[tr], eval_set=[(feats[val], y[val])], verbose=False)
    acc = (clf.predict(feats[val]) == y[val]).mean()
    log(f"  XGBoost node-ancestry accuracy (held-out nodes) {acc:.3f}, "
        f"majority-class baseline {np.bincount(y[tr]).max() / len(tr):.3f}")
    leaves = clf.apply(feats)                   # leaf index per tree, for every node
    return leaf_embedding(leaves.reshape(len(feats), -1), args.dim, args.seed)


def embed_gnn(args, X, A, A_norm):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = A.shape[0]
    An = A_norm.tocoo()
    A_t = torch.sparse_coo_tensor(np.vstack([An.row, An.col]), An.data, (n, n)).coalesce().to(dev)
    Xt = torch.from_numpy(X).to(dev)
    up = sp.triu(A, k=1).tocoo()
    pos = np.vstack([up.row, up.col]).astype(np.int64)
    if pos.shape[1] == 0:
        raise ValueError("graph has no edges; lower --min-lift or use --edges all")

    class GCN(nn.Module):
        def __init__(s, d_in, d_hid, d_out):
            super().__init__()
            s.l1, s.l2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_out)

        def forward(s, X, A):
            H = F.dropout(F.relu(torch.sparse.mm(A, s.l1(X))), 0.1, s.training)
            return torch.sparse.mm(A, s.l2(H))

    model = GCN(X.shape[1], args.gnn_hidden, args.dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.gnn_lr)
    batch = min(args.edge_batch, pos.shape[1])
    for ep in range(1, args.gnn_epochs + 1):
        model.train(); opt.zero_grad()
        Z = model(Xt, A_t)
        p = torch.from_numpy(pos[:, rng.choice(pos.shape[1], batch, replace=False)]).to(dev)
        q = torch.from_numpy(rng.integers(0, n, (2, batch))).to(dev)
        pl, nl = (Z[p[0]] * Z[p[1]]).sum(-1), (Z[q[0]] * Z[q[1]]).sum(-1)
        loss = (F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl))
                + F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl)))
        loss.backward(); opt.step()
        if ep == 1 or ep % 100 == 0 or ep == args.gnn_epochs:
            acc = (((pl > 0).float().mean() + (nl < 0).float().mean()) / 2).item()
            log(f"    epoch {ep:4d}  loss {loss.item():.4f}  link acc {acc:.3f}")
    model.eval()
    with torch.no_grad():
        return model(Xt, A_t).cpu().numpy().astype(np.float32)


def stage_embed(args, chrom):
    out = f"node_emb_{args.method}.npy"
    if done(args, chrom, out):
        log(f"[embed {chrom}] cached"); return
    log(f"[embed {chrom}] {args.method}")
    A = sp.load_npz(cpath(args, chrom, "graph.npz")).astype(np.float32)
    A_norm = normalized_adjacency(A)
    X = carrier_features(args, chrom)
    Z = embed_gnn(args, X, A, A_norm) if args.method == "gnn" else embed_xgboost(args, chrom, X, A_norm)
    np.save(cpath(args, chrom, out), Z)
    log(f"  saved {Z.shape} embeddings")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["gnn", "xgboost"], required=True)
    ap.add_argument("--chroms", default=",".join(ALL_CHROMS), help="comma-separated, default chr1-chr22")
    ap.add_argument("--workdir", default="haplograph", help="shared folder used by all four scripts")
    ap.add_argument("--force", action="store_true", help="recompute existing embeddings")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=64, help="embedding size")
    ap.add_argument("--svd-dim", type=int, default=128, help="carrier-profile features after SVD")
    ap.add_argument("--gnn-hidden", type=int, default=256)
    ap.add_argument("--gnn-epochs", type=int, default=300)
    ap.add_argument("--gnn-lr", type=float, default=0.01)
    ap.add_argument("--edge-batch", type=int, default=500_000)
    ap.add_argument("--xgb-trees", type=int, default=100)
    ap.add_argument("--xgb-depth", type=int, default=6)
    args = ap.parse_args()
    args.chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    for chrom in args.chroms:
        require(args, chrom, "graph.npz", "carriers.npz", hint="1_build_graph.py")
        if args.method == "xgboost":
            require(args, chrom, "labels.npz", hint="2_ancestry_labels.py")
        stage_embed(args, chrom)
    log("done.")


if __name__ == "__main__":
    main()
