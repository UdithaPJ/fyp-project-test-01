"""
=============================================================================
STEP 3: CPU Baseline Implementations – All Core Algorithms
=============================================================================
FYP: Mid-Range GPU-Accelerated Framework for Multi-Scale Biological Network Analysis
Milestone 1.3 – Baseline CPU Implementations

ALGORITHMS IMPLEMENTED:
  1. BFS  (Breadth-First Search)
  2. PageRank
  3. RWR  (Random Walk with Restart)
  4. HITS (Hyperlink-Induced Topic Search)
  5. MCE  (Mean Conditional Entropy)  ← on expression data
  6. Louvain Community Detection       ← via python-louvain / networkx
  7. MCL  (Markov Clustering)

EACH ALGORITHM:
  - Pure NumPy/SciPy implementation (no GPU)
  - Timing wrapper
  - Saves results as .npy / .csv

HOW TO RUN:
  pip install numpy scipy networkx python-louvain tqdm pandas

  # Run all algorithms on the preprocessed graph:
  python step3_cpu_baselines.py \
      --adj output/adj_csr.npz \
      --expr output/expr_matrix_log2.npy \
      --genes output/genes_filtered.npy \
      --outdir output/cpu_results

  # Run a single algorithm:
  python step3_cpu_baselines.py --adj output/adj_csr.npz --algo pagerank
=============================================================================
"""

import os
import sys
import time
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from collections import deque

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/cpu_baselines.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Timing decorator
# ─────────────────────────────────────────────────────────────
def timed(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info(f"[TIMING] {func.__name__:30s}  {elapsed:.4f}s")
        return result, elapsed
    return wrapper


# ═══════════════════════════════════════════════════════════════
# 1. BFS  ─  Breadth-First Search
# ═══════════════════════════════════════════════════════════════
@timed
def bfs(adj_csr: sparse.csr_matrix, source: int = 0):
    """
    Level-synchronous BFS on CSR adjacency.
    Returns distance array (INF = unreachable).

    Complexity: O(V + E)
    """
    n = adj_csr.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    frontier = deque([source])

    while frontier:
        u = frontier.popleft()
        start = adj_csr.indptr[u]
        end   = adj_csr.indptr[u + 1]
        for v in adj_csr.indices[start:end]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                frontier.append(v)

    return dist


@timed
def multi_source_bfs(adj_csr: sparse.csr_matrix, n_sources: int = 5):
    """
    Run BFS from n_sources random seed nodes.
    Useful for diameter estimation in biological networks.
    """
    n = adj_csr.shape[0]
    sources = np.random.choice(n, size=n_sources, replace=False)
    all_dists = np.full((n_sources, n), -1, dtype=np.int32)
    for i, src in enumerate(sources):
        result, _ = bfs.__wrapped__(adj_csr, src)
        all_dists[i] = result
    return all_dists, sources


# ═══════════════════════════════════════════════════════════════
# 2. PageRank
# ═══════════════════════════════════════════════════════════════
@timed
def pagerank(adj_csr: sparse.csr_matrix, damping: float = 0.85,
             tol: float = 1e-6, max_iter: int = 100):
    """
    Power-iteration PageRank on CSR adjacency.
    Uses column-stochastic transition matrix.

    Complexity: O(k × E)  where k ≈ 20-50 iterations
    """
    n = adj_csr.shape[0]

    # Build column-stochastic matrix: divide each column by its degree
    # out-degree = column sum for undirected; for directed = row sum
    out_deg = np.array(adj_csr.sum(axis=1)).flatten().astype(np.float64)
    out_deg[out_deg == 0] = 1.0  # dangling nodes

    # Normalise rows (not columns) – row-stochastic for outgoing prob
    inv_deg = 1.0 / out_deg
    # Build transition matrix T = D^-1 A (row-stochastic)
    D_inv = sparse.diags(inv_deg)
    T = D_inv @ adj_csr  # T[u,v] = A[u,v] / out_deg[u]

    pr = np.ones(n, dtype=np.float64) / n
    teleport = (1.0 - damping) / n

    for iteration in range(max_iter):
        pr_new = damping * (T.T @ pr) + teleport
        # Dangling correction
        dangling_sum = pr[out_deg == 1.0].sum()
        pr_new += damping * dangling_sum / n

        err = np.abs(pr_new - pr).sum()
        pr = pr_new
        if err < tol:
            log.info(f"  PageRank converged at iteration {iteration+1}, err={err:.2e}")
            break

    return pr


# ═══════════════════════════════════════════════════════════════
# 3. RWR  ─  Random Walk with Restart
# ═══════════════════════════════════════════════════════════════
@timed
def rwr(adj_csr: sparse.csr_matrix, seed_nodes: list,
        restart_prob: float = 0.15, tol: float = 1e-6, max_iter: int = 100):
    """
    Random Walk with Restart from one or more seed nodes.
    Returns steady-state probability vector p.

    p = (1-r) W p + r p0
    W is column-stochastic transition matrix.

    Complexity: O(k × E)
    """
    n = adj_csr.shape[0]

    # Column-stochastic W
    col_sums = np.array(adj_csr.sum(axis=0)).flatten().astype(np.float64)
    col_sums[col_sums == 0] = 1.0
    W = adj_csr.astype(np.float64)
    # divide each column by its sum
    W = W @ sparse.diags(1.0 / col_sums)

    # Seed vector (uniform over seeds)
    p0 = np.zeros(n, dtype=np.float64)
    for s in seed_nodes:
        p0[s] = 1.0 / len(seed_nodes)

    p = p0.copy()
    for iteration in range(max_iter):
        p_new = (1.0 - restart_prob) * (W @ p) + restart_prob * p0
        err = np.abs(p_new - p).sum()
        p = p_new
        if err < tol:
            log.info(f"  RWR converged at iteration {iteration+1}, err={err:.2e}")
            break

    return p


# ═══════════════════════════════════════════════════════════════
# 4. HITS  ─  Hyperlink-Induced Topic Search
# ═══════════════════════════════════════════════════════════════
@timed
def hits(adj_csr: sparse.csr_matrix, tol: float = 1e-6, max_iter: int = 100):
    """
    HITS algorithm.
    h = A × a,  a = Aᵀ × h,  both normalised at each step.

    In miRNA-target networks:
      - miRNAs → hub nodes (high hub score)
      - target genes → authority nodes (high authority score)

    Complexity: O(k × (V + E))
    """
    n = adj_csr.shape[0]
    A = adj_csr.astype(np.float64)
    AT = A.T.tocsr()

    h = np.ones(n, dtype=np.float64)
    a = np.ones(n, dtype=np.float64)

    for iteration in range(max_iter):
        a_new = AT @ h
        norm = np.sqrt((a_new ** 2).sum())
        if norm > 0:
            a_new /= norm

        h_new = A @ a_new
        norm = np.sqrt((h_new ** 2).sum())
        if norm > 0:
            h_new /= norm

        err = max(np.abs(a_new - a).max(), np.abs(h_new - h).max())
        a, h = a_new, h_new
        if err < tol:
            log.info(f"  HITS converged at iteration {iteration+1}, err={err:.2e}")
            break

    return h, a  # hub scores, authority scores


# ═══════════════════════════════════════════════════════════════
# 5. MCE  ─  Mean Conditional Entropy
# ═══════════════════════════════════════════════════════════════
def _conditional_entropy(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """
    H(Y|X): conditional entropy of y given x using histogram binning.
    Lower value = x is a strong regulator of y.
    """
    x_bins = np.digitize(x, np.linspace(x.min(), x.max() + 1e-8, n_bins + 1)) - 1
    x_bins = np.clip(x_bins, 0, n_bins - 1)
    h_y_given_x = 0.0
    for b in range(n_bins):
        mask = x_bins == b
        p_x = mask.sum() / len(x)
        if p_x == 0:
            continue
        y_cond = y[mask]
        if len(y_cond) < 2:
            continue
        # entropy of y | x=b
        hist, _ = np.histogram(y_cond, bins=n_bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        h = -np.sum(probs * np.log2(probs))
        h_y_given_x += p_x * h
    return h_y_given_x


@timed
def mce(expr_matrix: np.ndarray, gene_ids: np.ndarray,
        regulator_indices: list = None, n_bins: int = 10):
    """
    Mean Conditional Entropy for regulatory influence analysis.
    expr_matrix: (n_genes × n_samples), rows = genes
    regulator_indices: subset of rows treated as regulators (e.g. miRNAs)
                       If None, all genes are tested.

    Returns MCE score per regulator (lower = stronger regulator).
    Complexity: O(M × G × S) per regulator
    """
    n_genes, n_samples = expr_matrix.shape

    if regulator_indices is None:
        regulator_indices = list(range(min(n_genes, 200)))  # limit for speed
        log.info(f"  MCE: testing first {len(regulator_indices)} genes as regulators")

    mce_scores = {}
    for reg_idx in regulator_indices:
        x = expr_matrix[reg_idx]
        ce_values = []
        for tgt_idx in range(n_genes):
            if tgt_idx == reg_idx:
                continue
            y = expr_matrix[tgt_idx]
            ce = _conditional_entropy(x, y, n_bins=n_bins)
            ce_values.append(ce)
        mce_scores[gene_ids[reg_idx]] = np.mean(ce_values) if ce_values else np.nan

    return mce_scores


# ═══════════════════════════════════════════════════════════════
# 6. Louvain Community Detection
# ═══════════════════════════════════════════════════════════════
@timed
def louvain(adj_csr: sparse.csr_matrix):
    """
    Louvain community detection via python-louvain (community package)
    or NetworkX fallback.
    Returns partition dict: {node_id: community_id}
    """
    try:
        import community as community_louvain
        import networkx as nx
        G = nx.from_scipy_sparse_array(adj_csr)
        partition = community_louvain.best_partition(G)
        n_communities = len(set(partition.values()))
        log.info(f"  Louvain: {n_communities} communities detected")
        return partition
    except ImportError:
        log.warning("python-louvain not installed. Using NetworkX greedy_modularity.")
        import networkx as nx
        G = nx.from_scipy_sparse_array(adj_csr)
        communities = nx.community.greedy_modularity_communities(G)
        partition = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid
        log.info(f"  Louvain fallback: {len(communities)} communities detected")
        return partition


# ═══════════════════════════════════════════════════════════════
# 7. MCL  ─  Markov Clustering
# ═══════════════════════════════════════════════════════════════
@timed
def mcl(adj_csr: sparse.csr_matrix, expansion: int = 2, inflation: float = 2.0,
        prune_threshold: float = 1e-4, max_iter: int = 100, tol: float = 1e-5):
    """
    Markov Clustering Algorithm.
    Works on column-stochastic representation.

    1. Expand  : M = M^e  (matrix power)
    2. Inflate : M[i,j] = M[i,j]^r, then re-normalise columns
    3. Prune   : zero out entries below prune_threshold
    4. Repeat until convergence

    WARNING: MCL on large dense graphs is very memory-intensive.
    We work with dense float32 for matrices up to ~2000 nodes.
    For larger graphs, use chunked sparse MCL or reduce to the
    largest connected component first.

    Complexity: O(iter × n^2) dense, or O(iter × nnz²) sparse
    """
    n = adj_csr.shape[0]

    if n > 3000:
        log.warning(f"MCL: {n} nodes is large for CPU dense MCL. "
                    f"Using largest component or sub-sampling.")
        # Sub-sample: keep top-degree nodes
        degrees = np.array(adj_csr.sum(axis=1)).flatten()
        top_idx = np.argsort(degrees)[-2000:]
        top_idx = np.sort(top_idx)
        adj_csr = adj_csr[top_idx][:, top_idx]
        n = adj_csr.shape[0]
        log.info(f"MCL: reduced to {n} top-degree nodes")

    # Convert to dense float32
    M = adj_csr.toarray().astype(np.float32)

    # Add self-loops to ensure irreducibility
    np.fill_diagonal(M, M.diagonal() + 1.0)

    # Column-normalise (column-stochastic)
    col_sums = M.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    M /= col_sums

    for iteration in range(max_iter):
        M_old = M.copy()

        # Expansion
        M = np.linalg.matrix_power(M.astype(np.float64), expansion).astype(np.float32)

        # Inflation
        M = np.power(M, inflation)
        col_sums = M.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        M /= col_sums

        # Prune
        M[M < prune_threshold] = 0.0
        col_sums = M.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        M /= col_sums

        # Convergence check
        diff = np.abs(M - M_old).max()
        if diff < tol:
            log.info(f"  MCL converged at iteration {iteration+1}, diff={diff:.2e}")
            break

    # Extract clusters: attractor nodes are those with M[i,i] > 0
    clusters = {}
    attractors = np.where(M.diagonal() > 0.01)[0]
    for node in range(n):
        best_attractor = attractors[np.argmax(M[attractors, node])] \
            if len(attractors) > 0 else node
        clusters[node] = int(best_attractor)

    n_clusters = len(set(clusters.values()))
    log.info(f"  MCL: {n_clusters} clusters detected")
    return clusters, M


# ═══════════════════════════════════════════════════════════════
# Result Saving Helpers
# ═══════════════════════════════════════════════════════════════
def save_scores(scores: np.ndarray, gene_ids: np.ndarray,
                filename: str, score_name: str = "score"):
    df = pd.DataFrame({
        "gene_id": gene_ids,
        score_name: scores
    }).sort_values(score_name, ascending=False)
    df.to_csv(filename, index=False)
    log.info(f"Saved: {filename}  (top gene: {df.iloc[0]['gene_id']} = {df.iloc[0][score_name]:.6f})")


def save_partition(partition: dict, gene_ids: np.ndarray,
                   filename: str):
    rows = [{"gene_id": gene_ids[k] if k < len(gene_ids) else str(k),
             "community": v}
            for k, v in partition.items()]
    df = pd.DataFrame(rows).sort_values("community")
    df.to_csv(filename, index=False)
    log.info(f"Saved: {filename}")


def save_bfs_results(dist: np.ndarray, gene_ids: np.ndarray,
                     source: int, filename: str):
    df = pd.DataFrame({
        "gene_id": gene_ids,
        "bfs_distance": dist
    })
    df.to_csv(filename, index=False)
    log.info(f"Saved BFS results: {filename}  "
             f"(source={gene_ids[source]}, reachable={np.sum(dist >= 0)})")


# ═══════════════════════════════════════════════════════════════
# MAIN – run all baselines
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CPU Baseline Implementations for FYP Graph Algorithms")
    parser.add_argument("--adj", required=True,
                        help="Path to adj_csr.npz (scipy sparse CSR)")
    parser.add_argument("--expr",
                        help="Path to expr_matrix_log2.npy (for MCE)")
    parser.add_argument("--genes",
                        help="Path to genes_filtered.npy")
    parser.add_argument("--algo", default="all",
                        choices=["all", "bfs", "pagerank", "rwr",
                                 "hits", "mce", "louvain", "mcl"])
    parser.add_argument("--bfs_source", type=int, default=0,
                        help="Source node for BFS (default: 0)")
    parser.add_argument("--rwr_seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seed nodes for RWR")
    parser.add_argument("--outdir", default="output/cpu_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    log.info("=" * 60)
    log.info("FYP Milestone 1.3 – CPU Baseline Implementations")
    log.info("=" * 60)

    # ── Load graph ─────────────────────────────────────────
    log.info(f"Loading adjacency matrix: {args.adj}")
    adj = load_npz(args.adj)
    n = adj.shape[0]
    log.info(f"Graph: {n} nodes, {adj.nnz} edges")

    # ── Load gene IDs ──────────────────────────────────────
    if args.genes and os.path.exists(args.genes):
        gene_ids = np.load(args.genes, allow_pickle=True)
    else:
        gene_ids = np.array([f"gene_{i}" for i in range(n)])
    gene_ids = gene_ids[:n]  # ensure alignment

    # -- Load expression ----------------------------------------
    expr = None
    if args.expr and os.path.exists(args.expr):
        n_genes_expr = len(gene_ids)
        file_bytes = os.path.getsize(args.expr)
        n_samples_expr = file_bytes // (n_genes_expr * 4)
        expr = np.memmap(args.expr, dtype=np.float32, mode="r",
                         shape=(n_genes_expr, n_samples_expr))
        log.info(f"Expression matrix: {expr.shape} (memmap)")


    timing_log = {}

    # ──────────────────────────────────────────────────────
    # 1. BFS
    # ──────────────────────────────────────────────────────
    if args.algo in ["all", "bfs"]:
        log.info("\n--- BFS ---")
        dist, t = bfs(adj, source=args.bfs_source)
        timing_log["bfs"] = t
        save_bfs_results(dist, gene_ids, args.bfs_source,
                         os.path.join(args.outdir, "bfs_distances.csv"))
        np.save(os.path.join(args.outdir, "bfs_distances.npy"), dist)

    # ──────────────────────────────────────────────────────
    # 2. PageRank
    # ──────────────────────────────────────────────────────
    if args.algo in ["all", "pagerank"]:
        log.info("\n--- PageRank ---")
        pr, t = pagerank(adj)
        timing_log["pagerank"] = t
        save_scores(pr, gene_ids,
                    os.path.join(args.outdir, "pagerank_scores.csv"),
                    score_name="pagerank")
        np.save(os.path.join(args.outdir, "pagerank_scores.npy"), pr)

    # ──────────────────────────────────────────────────────
    # 3. RWR
    # ──────────────────────────────────────────────────────
    if args.algo in ["all", "rwr"]:
        log.info("\n--- RWR ---")
        seeds = [s for s in args.rwr_seeds if s < n]
        rwr_scores, t = rwr(adj, seed_nodes=seeds)
        timing_log["rwr"] = t
        save_scores(rwr_scores, gene_ids,
                    os.path.join(args.outdir, "rwr_scores.csv"),
                    score_name="rwr_relevance")
        np.save(os.path.join(args.outdir, "rwr_scores.npy"), rwr_scores)

    # ──────────────────────────────────────────────────────
    # 4. HITS
    # ──────────────────────────────────────────────────────
    if args.algo in ["all", "hits"]:
        log.info("\n--- HITS ---")
        (hub_scores, auth_scores), t = hits(adj)
        timing_log["hits"] = t
        save_scores(hub_scores, gene_ids,
                    os.path.join(args.outdir, "hits_hub_scores.csv"),
                    score_name="hub_score")
        save_scores(auth_scores, gene_ids,
                    os.path.join(args.outdir, "hits_authority_scores.csv"),
                    score_name="authority_score")

    # ──────────────────────────────────────────────────────
    # 5. MCE
    # ──────────────────────────────────────────────────────
    if args.algo in ["all", "mce"] and expr is not None:
        log.info("\n--- MCE ---")
        n_genes_expr = expr.shape[0]
        # Test first 50 genes as regulators (full MCE is O(M×G×S))
        regs = list(range(min(50, n_genes_expr)))
        mce_scores, t = mce(expr, gene_ids[:n_genes_expr],
                                  regulator_indices=regs)
        timing_log["mce"] = t
        mce_df = pd.DataFrame(list(mce_scores.items()),
                               columns=["gene_id", "mce_score"])
        mce_df = mce_df.sort_values("mce_score")
        mce_df.to_csv(os.path.join(args.outdir, "mce_scores.csv"), index=False)
        log.info(f"MCE saved. Top regulator: {mce_df.iloc[0]['gene_id']} "
                 f"(MCE={mce_df.iloc[0]['mce_score']:.4f})")
    elif args.algo in ["all", "mce"] and expr is None:
        log.warning("MCE skipped: --expr not provided")

    # ──────────────────────────────────────────────────────
    # 6. Louvain
    # ──────────────────────────────────────────────────────
    if args.algo in ["all", "louvain"]:
        log.info("\n--- Louvain Community Detection ---")
        partition, t = louvain(adj)
        timing_log["louvain"] = t
        save_partition(partition, gene_ids,
                       os.path.join(args.outdir, "louvain_communities.csv"))

    # ──────────────────────────────────────────────────────
    # 7. MCL
    # ──────────────────────────────────────────────────────
    if args.algo in ["all", "mcl"]:
        log.info("\n--- MCL ---")
        (clusters, _), t = mcl(adj)
        timing_log["mcl"] = t
        # Build partition dict using gene_ids
        n_mcl = len(clusters)
        mcl_gene_ids = gene_ids[:n_mcl] if len(gene_ids) >= n_mcl else \
            np.array([f"gene_{i}" for i in range(n_mcl)])
        save_partition(clusters, mcl_gene_ids,
                       os.path.join(args.outdir, "mcl_clusters.csv"))

    # ──────────────────────────────────────────────────────
    # Summary timing report
    # ──────────────────────────────────────────────────────
    timing_path = os.path.join(args.outdir, "cpu_timing.csv")
    timing_df = pd.DataFrame(
        [(k, v) for k, v in timing_log.items()],
        columns=["algorithm", "cpu_time_seconds"]
    )
    timing_df.to_csv(timing_path, index=False)

    log.info(f"\n{'='*60}")
    log.info("CPU Baseline Summary:")
    log.info(f"{'='*60}")
    for algo, t in timing_log.items():
        log.info(f"  {algo:20s} {t:>10.4f}s")
    log.info(f"\nAll results saved to: {os.path.abspath(args.outdir)}/")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()