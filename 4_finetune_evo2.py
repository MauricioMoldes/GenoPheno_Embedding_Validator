"""
Step 4 of 4 - fine-tune Evo 2 7B on the graph embeddings (no DNA sequence is used).

  - A training example is one individual on one chromosome: the clusters that person carries,
    ordered by genomic position, each replaced by its graph embedding (from 3_node_embeddings.py).
  - A per-chromosome linear layer projects those embeddings to Evo 2's hidden size, and they are fed
    into Evo 2 IN PLACE OF its nucleotide token embeddings (a hook swaps the output of Evo 2's
    embedding layer). Per-chromosome, because each chromosome's embeddings come from its own model.
  - Evo 2 runs up to --layer (default blocks.28.mlp.l3); the last --train-blocks blocks before it are
    unfrozen, the rest stay frozen.
  - --objective next     (default): predict the NEXT cluster along the chromosome (softmax over all
                          clusters of that chromosome), mirroring Evo 2's next-token pre-training.
    --objective ancestry : predict the individual's super-population (needs 2_ancestry_labels.py).
  - Individuals are split train/validation; validation accuracy is reported before and after.

Output: <workdir>/evo2_<objective>_<method>.pt  (unfrozen blocks, projections, head, config)

Usage:
  python 4_finetune_evo2.py --method gnn
  python 4_finetune_evo2.py --method xgboost --objective ancestry --chroms chr21,chr22
  python 4_finetune_evo2.py --method gnn --train-blocks 0     # Evo 2 frozen: baseline comparison

Requires: numpy, pandas, scipy, torch, evo2 (pip install evo2; Linux, CUDA 12.1+, Python 3.11/3.12;
the 7B models run in bf16 without Transformer Engine; an 80 GB GPU is recommended).

Note: the official Evo 2 training paths are Savanna and NVIDIA BioNeMo. This script uses the `evo2`
inference package in plain PyTorch and, after the first step, checks that gradients reach the input
projection and the unfrozen blocks, stopping with a clear message if they don't.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

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


def require(args, chrom, *names, hint):
    missing = [n for n in names if not os.path.exists(cpath(args, chrom, n))]
    if missing:
        raise SystemExit(f"{chrom}: missing {', '.join(missing)} in {args.workdir}/{chrom}/ - run {hint} first")


def load_individual_ancestry(args, individuals):
    """Ancestry per individual from 2_ancestry_labels.py, aligned to individuals.txt."""
    f = gpath(args, "individual_ancestry.csv")
    if not os.path.exists(f):
        raise SystemExit(f"{f} not found - run 2_ancestry_labels.py first (needed for --objective ancestry)")
    a = pd.read_csv(f, dtype=str, keep_default_na=False)
    lookup = dict(zip(a["individual"], a["ancestry"]))
    return np.array([lookup.get(i) or None for i in individuals], dtype=object)


class ChromData:
    """Per-chromosome node embeddings + each individual's clusters in genomic order."""

    def __init__(self, args, chrom):
        Z = np.load(cpath(args, chrom, f"node_emb_{args.method}.npy")).astype(np.float32)
        self.Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        nodes = pd.read_csv(cpath(args, chrom, "nodes.csv.gz"))
        order = np.lexsort((np.arange(len(nodes)), nodes["start"].fillna(0).to_numpy()))
        self.rank = np.empty(len(nodes), np.int64)
        self.rank[order] = np.arange(len(nodes))
        self.by_ind = sp.load_npz(cpath(args, chrom, "carriers.npz")).T.tocsr()  # [I, N]

    def tokens(self, i):
        """Node indices carried by individual i, sorted by genomic position."""
        cols = self.by_ind.indices[self.by_ind.indptr[i]:self.by_ind.indptr[i + 1]]
        return cols[np.argsort(self.rank[cols], kind="stable")]


def find_embedding_module(net, name):
    import torch.nn as nn
    mods = dict(net.named_modules())
    if name:
        return mods[name]
    for n, m in mods.items():
        if n.endswith("embedding_layer"):
            return m
    for n, m in mods.items():
        if isinstance(m, nn.Embedding):
            return m
    raise ValueError("could not find Evo 2's token embedding layer; pass --embed-module")


def stage_evo2(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from evo2 import Evo2

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = {c: ChromData(args, c) for c in args.chroms}
    individuals = open(gpath(args, "individuals.txt")).read().split("\n")
    n_ind = len(individuals)

    y_ind, classes = None, None
    if args.objective == "ancestry":
        anc = load_individual_ancestry(args, individuals)
        known = np.array([a is not None for a in anc])
        classes = np.array(sorted(set(anc[known])))
        y_ind = np.full(n_ind, -1)
        y_ind[known] = np.searchsorted(classes, anc[known])
        pool = np.where(known)[0]
    else:
        pool = np.arange(n_ind)

    perm = rng.permutation(pool)
    n_val = max(1, int(args.val_frac * len(perm)))
    val_inds, tr_inds = perm[:n_val], perm[n_val:]
    examples = lambda inds: [(c, i) for c in args.chroms for i in inds if len(data[c].tokens(i)) >= 2]
    train_ex, val_ex = examples(tr_inds), examples(val_inds)
    if len(val_ex) > args.val_examples:
        val_ex = [val_ex[j] for j in rng.choice(len(val_ex), args.val_examples, replace=False)]
    log(f"[evo2] objective={args.objective}, embeddings={args.method}, {len(args.chroms)} chromosomes, "
        f"{len(tr_inds):,} train / {len(val_inds):,} val individuals, {len(train_ex):,} training sequences")

    evo = Evo2(args.evo2_model)
    net = evo.model
    dev = next(net.parameters()).device

    # ---- freeze all, unfreeze the last `train_blocks` blocks up to the read-out layer
    m = re.match(r"blocks\.(\d+)\.", args.layer)
    if not m:
        raise ValueError("--layer must look like blocks.<i>.<...>, e.g. blocks.28.mlp.l3")
    last = int(m.group(1))
    for p in net.parameters():
        p.requires_grad_(False)
    trainable = []
    for i in range(max(0, last - args.train_blocks + 1), last + 1) if args.train_blocks > 0 else []:
        for name, p in net.blocks[i].named_parameters():
            p.requires_grad_(True); trainable.append((f"blocks.{i}.{name}", p))

    # ---- hooks: replace token embeddings with our projected graph embeddings; stop at --layer
    class StopForward(Exception):
        pass

    state = {"soft": None, "h": None, "embed_shape": None}

    def embed_hook(_m, _inp, out):
        state["embed_shape"] = tuple(out.shape)
        if state["soft"] is None:
            return None
        if state["soft"].shape != out.shape:
            raise RuntimeError(f"soft input {tuple(state['soft'].shape)} != embedding output {tuple(out.shape)}")
        return state["soft"].to(out.dtype)

    def layer_hook(_m, _inp, out):
        state["h"] = out[0] if isinstance(out, tuple) else out
        raise StopForward

    find_embedding_module(net, args.embed_module).register_forward_hook(embed_hook)
    dict(net.named_modules())[args.layer].register_forward_hook(layer_hook)

    def run(soft):
        """soft: [T, hidden] -> Evo 2 hidden states at --layer, [T, hidden_out] float32."""
        state["soft"] = soft.unsqueeze(0)
        dummy = torch.zeros(1, soft.shape[0], dtype=torch.int, device=dev)  # ids are ignored
        try:
            net(dummy)
        except StopForward:
            pass
        state["soft"] = None
        return state.pop("h").float().squeeze(0)

    # discover sizes with one plain forward pass
    with torch.no_grad():
        try:
            net(torch.zeros(1, 8, dtype=torch.int, device=dev))
        except StopForward:
            pass
    hidden_in, hidden_out = state["embed_shape"][-1], state["h"].shape[-1]
    state["h"] = None

    # ---- new trainable layers: per-chromosome input projections and output heads
    dim = next(iter(data.values())).Z.shape[1]
    proj_in = nn.ModuleDict({c: nn.Sequential(nn.Linear(dim, hidden_in), nn.LayerNorm(hidden_in))
                             for c in args.chroms}).to(dev)
    if args.objective == "next":
        head = nn.ModuleDict({c: nn.Sequential(nn.LayerNorm(hidden_out), nn.Linear(hidden_out, dim))
                              for c in args.chroms}).to(dev)
    else:
        head = nn.Sequential(nn.LayerNorm(hidden_out), nn.Linear(hidden_out, len(classes))).to(dev)
    Z_t = {c: torch.from_numpy(d.Z).to(dev) for c, d in data.items()}
    Zn_t = {c: F.normalize(z, dim=-1) for c, z in Z_t.items()}

    opt = torch.optim.AdamW([
        {"params": [p for _, p in trainable], "lr": args.lr},
        {"params": list(proj_in.parameters()) + list(head.parameters()), "lr": args.head_lr},
    ], weight_decay=0.01)
    log(f"  {args.evo2_model}: hidden {hidden_in}, read-out {args.layer}, unfrozen blocks "
        f"{f'{last - args.train_blocks + 1}-{last}' if trainable else 'none'} "
        f"({sum(p.numel() for _, p in trainable) / 1e6:.0f}M params)")

    def window(tok, train):
        if len(tok) <= args.max_len:
            return tok
        s = rng.integers(0, len(tok) - args.max_len + 1) if train else (len(tok) - args.max_len) // 2
        return tok[s:s + args.max_len]

    def example_loss(chrom, i, train):
        """Returns (loss, n_correct, n_predictions) for one individual-chromosome sequence."""
        tok = torch.from_numpy(window(data[chrom].tokens(i), train)).to(dev)
        h = run(proj_in[chrom](Z_t[chrom][tok]))                       # [T, hidden_out]
        if args.objective == "next":
            pred = F.normalize(head[chrom](h[:-1]), dim=-1)             # predict token t+1 from <= t
            logits = pred @ Zn_t[chrom].T / args.temperature            # over all clusters of chrom
            target = tok[1:]
            loss = F.cross_entropy(logits, target)
            return loss, (logits.argmax(-1) == target).sum().item(), len(target)
        logits = head(h.mean(0, keepdim=True))
        target = torch.tensor([y_ind[i]], device=dev)
        return F.cross_entropy(logits, target), int(logits.argmax(-1).item() == y_ind[i]), 1

    def evaluate():
        net.eval(); proj_in.eval(); head.eval()
        tot_loss = correct = count = 0
        with torch.no_grad():
            for c, i in val_ex:
                l, k, n = example_loss(c, i, False)
                tot_loss += l.item(); correct += k; count += n
        net.train(); proj_in.train(); head.train()
        return tot_loss / len(val_ex), correct / max(count, 1)

    what = "next-cluster top-1" if args.objective == "next" else "ancestry"
    l0, a0 = evaluate()
    log(f"  before training: val loss {l0:.4f}, val {what} accuracy {a0:.3f}")

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_ex))
        for b in range(0, len(order) - args.batch + 1, args.batch):
            opt.zero_grad()
            batch_loss = 0.0
            for j in order[b:b + args.batch]:            # gradient accumulation over the batch
                loss, _, _ = example_loss(*train_ex[j], True)
                (loss / args.batch).backward()
                batch_loss += loss.item() / args.batch

            if step == 0:
                if all(p.grad is None for p in proj_in.parameters()):
                    raise RuntimeError("No gradient reached the input projection: this evo2 version's "
                                       "forward pass is not differentiable. Fine-tune via BioNeMo instead.")
                if trainable and all(p.grad is None for _, p in trainable):
                    raise RuntimeError("No gradient reached the unfrozen Evo 2 blocks; try --train-blocks 0 "
                                       "(trains only the projection + head through frozen Evo 2).")
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable] + list(proj_in.parameters())
                                           + list(head.parameters()), 1.0)
            opt.step(); step += 1
            if step % args.log_every == 0:
                log(f"  epoch {epoch + 1} step {step}: loss {batch_loss:.4f}  "
                    f"{(time.time() - t0) / step:.1f}s/step")
            if step % args.eval_every == 0:
                l, a = evaluate()
                log(f"  [eval] val loss {l:.4f}, val {what} accuracy {a:.3f}")

    l, a = evaluate()
    log(f"  final: val loss {l:.4f} (was {l0:.4f}), val {what} accuracy {a:.3f} (was {a0:.3f})")
    out = gpath(args, f"evo2_{args.objective}_{args.method}.pt")
    torch.save({
        "evo2_model": args.evo2_model, "layer": args.layer, "objective": args.objective,
        "graph_method": args.method, "chroms": args.chroms,
        "classes": None if classes is None else classes.tolist(),
        "trained_blocks": {n: p.detach().cpu() for n, p in trainable},
        "proj_in": proj_in.state_dict(), "head": head.state_dict(),
        "val_individuals": [individuals[i] for i in val_inds],
    }, out)
    log(f"  saved to {out} (load Evo2('{args.evo2_model}'), copy trained_blocks into evo.model, "
        f"and rebuild proj_in/head with the same shapes)")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["gnn", "xgboost"], required=True,
                    help="which embeddings from 3_node_embeddings.py to use")
    ap.add_argument("--chroms", default=",".join(ALL_CHROMS), help="comma-separated, default chr1-chr22")
    ap.add_argument("--workdir", default="haplograph", help="shared folder used by all four scripts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", choices=["next", "ancestry"], default="next")
    ap.add_argument("--evo2-model", default="evo2_7b_base", help="evo2_7b_base (8K ctx) or evo2_7b")
    ap.add_argument("--layer", default="blocks.28.mlp.l3", help="read-out layer; forward stops here")
    ap.add_argument("--embed-module", help="name of Evo 2's token embedding module (auto-detected)")
    ap.add_argument("--train-blocks", type=int, default=4, help="0 = Evo 2 frozen, train projection + head")
    ap.add_argument("--max-len", type=int, default=1024, help="max clusters per training sequence")
    ap.add_argument("--batch", type=int, default=8, help="sequences per optimizer step")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5, help="for unfrozen Evo 2 blocks")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="for projection + head")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of individuals held out")
    ap.add_argument("--val-examples", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()
    args.chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    if not os.path.exists(gpath(args, "individuals.txt")):
        raise SystemExit(f"{args.workdir}/individuals.txt not found - run 1_build_graph.py first")
    for chrom in args.chroms:
        require(args, chrom, "carriers.npz", "nodes.csv.gz", hint="1_build_graph.py")
        require(args, chrom, f"node_emb_{args.method}.npy", hint=f"3_node_embeddings.py --method {args.method}")
    stage_evo2(args)
    log("done.")


if __name__ == "__main__":
    main()
