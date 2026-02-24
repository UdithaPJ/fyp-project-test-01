r"""
=============================================================================
STEP 4: Parallel CPU Implementations + Benchmark vs Single-Thread
=============================================================================
FYP: Mid-Range GPU-Accelerated Framework for Multi-Scale Biological
     Network Analysis
Milestone 2.1 / 3.2 - Parallel CPU vs Single-thread CPU Comparison

WHAT THIS DOES:
  Runs parallel versions of all 7 algorithms on the ALREADY-PREPROCESSED
  outputs from step1/step3. No data loading or preprocessing is repeated.
  Generates a side-by-side benchmark report with speedup ratios.

PARALLELISM STRATEGY PER ALGORITHM:
  BFS      - multiprocessing.Pool: each BFS source in its own process.
             True CPU parallelism (bypasses GIL). Speedup = min(sources, workers).

  PageRank - scipy SpMV (T.T @ pr) is automatically multi-threaded via
             OpenBLAS when OMP_NUM_THREADS is set. The convergence norm
             is split across threads with ThreadPoolExecutor.

  RWR      - multiprocessing.Pool: each seed node's random walk runs in
             its own process. Speedup = min(n_seeds, workers).

  HITS     - ThreadPoolExecutor: matrix is row-partitioned, each thread
             computes its rows of A*a and AT*h independently. Thread-safe
             because output slices are disjoint.

  MCE      - multiprocessing.Pool: each regulator gene is processed by
             a separate worker. Workers open the mmap file independently
             (OS page-cache shared, no RAM duplication).
             Speedup = near-linear up to n_workers.

  Louvain  - ThreadPoolExecutor: runs n_workers independent Louvain
             restarts in parallel, picks the best modularity result.

  MCL      - ThreadPoolExecutor for inflation (row chunks) + column
             normalisation. BLAS expansion is OpenBLAS-threaded.

HOW TO RUN:
  Step 1 - Make sure step3 has run and produced cpu_timing.csv:
      output\cpu_results\cpu_timing.csv

  Step 2 - Run step4:
      python step4_parallel_cpu.py ^
          --adj            output\adj_csr.npz ^
          --expr           output\expr_mmap.npy ^
          --genes          output\genes_filtered.npy ^
          --single_timing  output\cpu_results\cpu_timing.csv ^
          --workers        8 ^
          --outdir         output\parallel_results

  Run a single algorithm only:
      python step4_parallel_cpu.py --adj output\adj_csr.npz ^
          --genes output\genes_filtered.npy --algo pagerank --workers 8

  Quick test (2 workers, all algorithms):
      python step4_parallel_cpu.py --adj output\adj_csr.npz ^
          --genes output\genes_filtered.npy --workers 2

OUTPUTS in --outdir:
  par_bfs_distances.csv          BFS distances (multi-source)
  par_pagerank_scores.csv        PageRank scores
  par_rwr_scores.csv             RWR relevance scores
  par_hits_hub_scores.csv        HITS hub scores
  par_hits_authority_scores.csv  HITS authority scores
  par_mce_scores.csv             MCE regulatory scores (needs --expr)
  par_louvain_communities.csv    Louvain community assignments
  par_mcl_clusters.csv           MCL cluster assignments
  parallel_timing.csv            Parallel wall-clock times per algorithm
  benchmark_comparison.csv       Single vs parallel times + speedup
  benchmark_report.txt           Human-readable summary report
=============================================================================
"""

import os
import sys
import io
import time
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from collections import deque
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# Windows-safe UTF-8 logging
if hasattr(sys.stdout, "buffer"):
    _utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
else:
    _utf8_stdout = sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(_utf8_stdout),
        logging.FileHandler("output/parallel_cpu.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# Timing decorator (identical to step3 for consistent comparison)
# =============================================================================
def timed(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info(f"[TIMING] {func.__name__:35s}  {elapsed:.4f}s")
        return result, elapsed
    return wrapper


# =============================================================================
# Set OpenBLAS/MKL thread count so scipy sparse matmul uses all cores
# Must be called BEFORE importing numpy in worker processes (set env vars early)
# =============================================================================
def configure_blas_threads(n_threads: int):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    log.info(f"BLAS thread count set to {n_threads}")


# =============================================================================
# 1. PARALLEL BFS
#    Strategy: multiprocessing.Pool - one process per BFS source.
#    Each worker receives the raw CSR arrays (picklable numpy arrays).
#    Speedup: min(n_sources, n_workers) - embarrassingly parallel.
# =============================================================================
def _bfs_worker(args):
    """Single-source BFS worker. Receives raw CSR arrays, not sparse matrix."""
    indptr, indices, n, source = args
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    frontier = deque([source])
    while frontier:
        u = frontier.popleft()
        for v in indices[indptr[u]: indptr[u + 1]]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                frontier.append(v)
    return source, dist


@timed
def parallel_bfs(adj_csr: sparse.csr_matrix, n_sources: int = 10,
                 n_workers: int = 8):
    n = adj_csr.shape[0]
    rng = np.random.default_rng(42)
    sources = rng.choice(n, size=min(n_sources, n), replace=False).tolist()

    # Pass raw arrays - numpy arrays are picklable and efficient to copy
    indptr  = adj_csr.indptr
    indices = adj_csr.indices
    worker_args = [(indptr, indices, n, src) for src in sources]

    all_dists = np.full((len(sources), n), -1, dtype=np.int32)
    log.info(f"  BFS: {len(sources)} sources across {n_workers} workers")

    with Pool(processes=n_workers) as pool:
        for src, dist in pool.imap_unordered(_bfs_worker, worker_args):
            idx = sources.index(src)
            all_dists[idx] = dist

    reachable = int(np.sum(all_dists[0] >= 0))
    log.info(f"  BFS source[0]: {reachable}/{n} nodes reachable")
    return all_dists, sources


# =============================================================================
# 2. PARALLEL PAGERANK
#    Strategy: OpenBLAS multi-threading for SpMV (dominant cost).
#    The sparse matmul T.T @ pr uses however many threads OMP_NUM_THREADS
#    specifies - this is the primary source of speedup.
#    Convergence norm split across ThreadPoolExecutor for additional gain.
# =============================================================================
@timed
def parallel_pagerank(adj_csr: sparse.csr_matrix, damping: float = 0.85,
                      tol: float = 1e-6, max_iter: int = 100,
                      n_workers: int = 8):
    n = adj_csr.shape[0]
    out_deg = np.array(adj_csr.sum(axis=1)).flatten().astype(np.float64)
    out_deg[out_deg == 0] = 1.0

    D_inv = sparse.diags(1.0 / out_deg)
    T = D_inv @ adj_csr          # row-stochastic transition matrix

    pr       = np.ones(n, dtype=np.float64) / n
    teleport = (1.0 - damping) / n
    chunk    = max(1, n // n_workers)

    for iteration in range(max_iter):
        # Primary parallelism: OpenBLAS-threaded sparse matmul
        pr_new = damping * (T.T @ pr) + teleport

        dangling_sum = pr[out_deg == 1.0].sum()
        pr_new += damping * dangling_sum / n

        # Secondary parallelism: thread-parallel L1 norm computation
        diff = pr_new - pr
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(lambda s, e: np.abs(diff[s:e]).sum(),
                          i, min(i + chunk, n))
                for i in range(0, n, chunk)
            ]
            err = sum(f.result() for f in as_completed(futures))

        pr = pr_new
        if err < tol:
            log.info(f"  PageRank converged at iter {iteration+1}, err={err:.2e}")
            break

    return pr


# =============================================================================
# 3. PARALLEL RWR
#    Strategy: multiprocessing.Pool - one process per seed node.
#    Each worker reconstructs the transition matrix from raw arrays,
#    runs its own full RWR, and returns its steady-state vector.
#    Final result is the mean over all seed vectors (ensemble RWR).
#    Speedup: min(n_seeds, n_workers) - embarrassingly parallel.
# =============================================================================
def _rwr_worker(args):
    """Single-seed RWR worker. Reconstructs sparse matrix from raw arrays."""
    indptr, indices, data, n, seed, restart_prob, tol, max_iter = args

    W_raw = sparse.csr_matrix(
        (data.astype(np.float64), indices, indptr), shape=(n, n)
    )
    col_sums = np.array(W_raw.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0
    W = W_raw @ sparse.diags(1.0 / col_sums)   # column-stochastic

    p0 = np.zeros(n, dtype=np.float64)
    p0[seed] = 1.0
    p = p0.copy()

    for _ in range(max_iter):
        p_new = (1.0 - restart_prob) * (W @ p) + restart_prob * p0
        if np.abs(p_new - p).sum() < tol:
            break
        p = p_new

    return seed, p


@timed
def parallel_rwr(adj_csr: sparse.csr_matrix, seed_nodes: list,
                 restart_prob: float = 0.15, tol: float = 1e-6,
                 max_iter: int = 100, n_workers: int = 8):
    n = adj_csr.shape[0]
    # Copy arrays for pickling - shared memory not available across processes
    indptr  = adj_csr.indptr.copy()
    indices = adj_csr.indices.copy()
    data    = adj_csr.data.astype(np.float32).copy()

    worker_args = [
        (indptr, indices, data, n, s, restart_prob, tol, max_iter)
        for s in seed_nodes
    ]

    log.info(f"  RWR: {len(seed_nodes)} seeds across {n_workers} workers")
    seed_results = {}
    with Pool(processes=n_workers) as pool:
        for seed, p in pool.imap_unordered(_rwr_worker, worker_args):
            seed_results[seed] = p

    # Aggregate: mean over all seed steady-state vectors
    all_vecs = np.stack([seed_results[s] for s in seed_nodes], axis=0)
    return all_vecs.mean(axis=0)


# =============================================================================
# 4. PARALLEL HITS
#    Strategy: ThreadPoolExecutor with row-partitioned SpMV.
#    Each thread computes a disjoint row-slice of the output vector.
#    Thread-safe: no shared write locations between threads.
#    Also benefits from OpenBLAS threading within each thread's slice.
# =============================================================================
@timed
def parallel_hits(adj_csr: sparse.csr_matrix, tol: float = 1e-6,
                  max_iter: int = 100, n_workers: int = 8):
    n     = adj_csr.shape[0]
    A     = adj_csr.astype(np.float64)
    AT    = A.T.tocsr()
    chunk = max(1, n // n_workers)

    h = np.ones(n, dtype=np.float64)
    a = np.ones(n, dtype=np.float64)

    def spmv_rows(mat, vec, out, start, end):
        """Write mat[start:end] @ vec into out[start:end]. Disjoint write."""
        out[start:end] = mat[start:end] @ vec

    for iteration in range(max_iter):
        # Parallel authority update: a_new = AT @ h
        a_new = np.zeros(n, dtype=np.float64)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(spmv_rows, AT, h, a_new, i, min(i + chunk, n))
                for i in range(0, n, chunk)
            ]
            for f in as_completed(futures):
                f.result()
        norm = np.sqrt((a_new ** 2).sum())
        if norm > 0:
            a_new /= norm

        # Parallel hub update: h_new = A @ a_new
        h_new = np.zeros(n, dtype=np.float64)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(spmv_rows, A, a_new, h_new, i, min(i + chunk, n))
                for i in range(0, n, chunk)
            ]
            for f in as_completed(futures):
                f.result()
        norm = np.sqrt((h_new ** 2).sum())
        if norm > 0:
            h_new /= norm

        err = max(np.abs(a_new - a).max(), np.abs(h_new - h).max())
        a, h = a_new, h_new
        if err < tol:
            log.info(f"  HITS converged at iter {iteration+1}, err={err:.2e}")
            break

    return h, a


# =============================================================================
# 5. PARALLEL MCE
#    Strategy: multiprocessing.Pool - one process per regulator gene.
#    Workers open the mmap file independently at the given path.
#    The OS page-cache de-duplicates physical memory across all workers
#    so RAM usage does not multiply by n_workers.
#    Speedup: near-linear up to n_workers (embarrassingly parallel).
# =============================================================================
def _mce_worker(args):
    """Compute MCE for one regulator gene against all target genes."""
    reg_idx, expr_path, n_genes, n_samples, gene_id, n_bins = args

    # Open mmap read-only inside worker (no pickling of array needed)
    expr  = np.memmap(expr_path, dtype=np.float32, mode="r",
                      shape=(n_genes, n_samples))
    x     = np.array(expr[reg_idx], dtype=np.float64)
    edges = np.linspace(x.min(), x.max() + 1e-8, n_bins + 1)
    x_bin = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    total_ce = 0.0
    count    = 0
    for tgt_idx in range(n_genes):
        if tgt_idx == reg_idx:
            continue
        y          = np.array(expr[tgt_idx], dtype=np.float64)
        h_y_given_x = 0.0
        for b in range(n_bins):
            mask = x_bin == b
            p_x  = mask.sum() / len(x)
            if p_x == 0:
                continue
            y_cond = y[mask]
            if len(y_cond) < 2:
                continue
            hist, _ = np.histogram(y_cond, bins=n_bins)
            probs    = hist / hist.sum()
            probs    = probs[probs > 0]
            h_y_given_x += p_x * (-np.sum(probs * np.log2(probs)))
        total_ce += h_y_given_x
        count    += 1

    return gene_id, (total_ce / count if count > 0 else np.nan)


@timed
def parallel_mce(expr_path: str, gene_ids: np.ndarray,
                 regulator_indices: list = None,
                 n_bins: int = 10, n_workers: int = 8):
    file_bytes = os.path.getsize(expr_path)
    n_genes    = len(gene_ids)
    n_samples  = file_bytes // (n_genes * 4)

    if regulator_indices is None:
        regulator_indices = list(range(min(50, n_genes)))

    log.info(f"  MCE: {len(regulator_indices)} regulators across {n_workers} workers")

    worker_args = [
        (reg_idx, expr_path, n_genes, n_samples, str(gene_ids[reg_idx]), n_bins)
        for reg_idx in regulator_indices
    ]

    mce_scores = {}
    with Pool(processes=n_workers) as pool:
        for gene_id, val in pool.imap_unordered(_mce_worker, worker_args):
            mce_scores[gene_id] = val

    return mce_scores


# =============================================================================
# 6. PARALLEL LOUVAIN
#    Strategy: ThreadPoolExecutor runs n_workers independent Louvain
#    restarts simultaneously (Louvain is non-deterministic due to random
#    tie-breaking). The partition with the highest modularity is selected.
#    This improves solution quality AND reduces wall-clock time vs running
#    restarts sequentially.
# =============================================================================
@timed
def parallel_louvain(adj_csr: sparse.csr_matrix, n_workers: int = 8):
    import networkx as nx

    n = adj_csr.shape[0]
    log.info(f"  Louvain: building NetworkX graph ({n} nodes) ...")
    G = nx.from_scipy_sparse_array(adj_csr)

    try:
        import community as community_louvain

        log.info(f"  Louvain: {n_workers} parallel restarts ...")

        def one_restart(_):
            return community_louvain.best_partition(G)

        candidates = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(one_restart, i) for i in range(n_workers)]
            for f in as_completed(futures):
                candidates.append(f.result())

        # Select partition with highest modularity
        best = max(candidates,
                   key=lambda p: community_louvain.modularity(p, G))
        log.info(f"  Louvain: {len(set(best.values()))} communities "
                 f"(best of {n_workers} restarts)")
        return best

    except ImportError:
        log.warning("python-louvain not found. Using NetworkX greedy_modularity.")
        communities = nx.community.greedy_modularity_communities(G)
        partition   = {node: cid
                       for cid, comm in enumerate(communities)
                       for node in comm}
        log.info(f"  Louvain fallback: {len(communities)} communities")
        return partition


# =============================================================================
# 7. PARALLEL MCL
#    Strategy: two levels of parallelism.
#    Expansion (M^e): BLAS-threaded via OpenBLAS (set OMP_NUM_THREADS).
#    Inflation + Prune: row chunks dispatched to ThreadPoolExecutor.
#      Each thread handles its own rows - disjoint writes, thread-safe.
#    Column normalisation: column chunks dispatched to ThreadPoolExecutor.
# =============================================================================
def _inflate_rows(M_chunk, inflation, prune_threshold):
    """Apply inflation and prune to a row slice. Returns modified chunk."""
    result = np.power(M_chunk.astype(np.float64), inflation).astype(np.float32)
    result[result < prune_threshold] = 0.0
    return result


@timed
def parallel_mcl(adj_csr: sparse.csr_matrix, expansion: int = 2,
                 inflation: float = 2.0, prune_threshold: float = 1e-4,
                 max_iter: int = 100, tol: float = 1e-5,
                 n_workers: int = 8):
    n = adj_csr.shape[0]

    if n > 3000:
        log.warning("MCL: graph > 3000 nodes, reducing to top-2000 by degree")
        degrees = np.array(adj_csr.sum(axis=1)).flatten()
        top_idx = np.sort(np.argsort(degrees)[-2000:])
        adj_csr = adj_csr[top_idx][:, top_idx]
        n       = adj_csr.shape[0]
        log.info(f"  MCL: reduced to {n} nodes")

    M = adj_csr.toarray().astype(np.float32)
    np.fill_diagonal(M, M.diagonal() + 1.0)
    col_sums = M.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    M /= col_sums

    row_chunk = max(1, n // n_workers)
    col_chunk = max(1, n // n_workers)

    def normalise_cols(mat):
        """Thread-parallel column normalisation."""
        cs = mat.sum(axis=0, keepdims=True)
        cs[cs == 0] = 1.0
        def _norm(s, e):
            mat[:, s:e] /= cs[:, s:e]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            fs = [ex.submit(_norm, i, min(i + col_chunk, n))
                  for i in range(0, n, col_chunk)]
            for f in as_completed(fs):
                f.result()
        return mat

    for iteration in range(max_iter):
        M_old = M.copy()

        # Expansion: OpenBLAS-threaded matrix power
        M = np.linalg.matrix_power(
            M.astype(np.float64), expansion).astype(np.float32)

        # Inflation: thread-parallel row chunks
        slices  = [(i, min(i + row_chunk, n)) for i in range(0, n, row_chunk)]
        results = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_idx = {
                ex.submit(_inflate_rows, M[s:e].copy(), inflation,
                          prune_threshold): idx
                for idx, (s, e) in enumerate(slices)
            }
            for f in as_completed(future_to_idx):
                results[future_to_idx[f]] = f.result()

        M = np.vstack([results[i] for i in range(len(slices))])

        # Column normalisation: thread-parallel
        M = normalise_cols(M)

        diff = np.abs(M - M_old).max()
        if diff < tol:
            log.info(f"  MCL converged at iter {iteration+1}, diff={diff:.2e}")
            break

    clusters   = {}
    attractors = np.where(M.diagonal() > 0.01)[0]
    for node in range(n):
        best = (attractors[np.argmax(M[attractors, node])]
                if len(attractors) > 0 else node)
        clusters[node] = int(best)

    log.info(f"  MCL: {len(set(clusters.values()))} clusters")
    return clusters, M


# =============================================================================
# Output helpers (same CSV format as step3 for direct comparison)
# =============================================================================
def save_scores(scores: np.ndarray, gene_ids: np.ndarray,
                path: str, score_name: str = "score"):
    df = pd.DataFrame({"gene_id": gene_ids, score_name: scores}
                      ).sort_values(score_name, ascending=False)
    df.to_csv(path, index=False)
    top = df.iloc[0]
    log.info(f"  Saved {path}  (top: {top['gene_id']} = {top[score_name]:.6f})")


def save_partition(partition: dict, gene_ids: np.ndarray, path: str):
    rows = [{"gene_id": (gene_ids[k] if k < len(gene_ids) else str(k)),
             "community": v}
            for k, v in partition.items()]
    pd.DataFrame(rows).sort_values("community").to_csv(path, index=False)
    log.info(f"  Saved {path}")


def save_bfs(all_dists: np.ndarray, sources: list,
             gene_ids: np.ndarray, path: str):
    df = pd.DataFrame({"gene_id": gene_ids})
    for i, src in enumerate(sources):
        df[f"dist_from_{src}"] = all_dists[i]
    df.to_csv(path, index=False)
    log.info(f"  Saved {path}  ({len(sources)} sources)")


# =============================================================================
# Benchmark report generator
# =============================================================================
def build_benchmark_report(par_timing: dict, single_timing_csv: str,
                            outdir: str, n_workers: int, n_nodes: int,
                            n_edges: int):
    """
    Merge single-thread times (from step3 cpu_timing.csv) with parallel
    times, compute speedup and parallel efficiency, write CSV + text report.
    """
    par_df = pd.DataFrame(
        sorted(par_timing.items()),
        columns=["algorithm", "parallel_time_s"]
    )

    lines = []
    sep   = "=" * 68
    lines += [sep,
              "FYP Benchmark Report: Single-Thread vs Parallel CPU",
              f"Graph : {n_nodes:,} nodes  |  {n_edges:,} edges",
              f"Workers: {n_workers}  |  CPUs available: {cpu_count()}",
              sep]

    if single_timing_csv and os.path.exists(single_timing_csv):
        single_df = pd.read_csv(single_timing_csv)
        # step3 writes columns: algorithm, cpu_time_seconds
        single_df.columns = ["algorithm", "single_time_s"]

        merged = pd.merge(single_df, par_df, on="algorithm", how="outer")
        merged["speedup_x"]     = (merged["single_time_s"]
                                   / merged["parallel_time_s"]).round(3)
        merged["efficiency_pct"] = (merged["speedup_x"] / n_workers * 100
                                    ).round(1)

        # Save CSV
        merged.to_csv(os.path.join(outdir, "benchmark_comparison.csv"),
                      index=False)

        # Text table
        hdr = (f"\n{'Algorithm':<18} {'Single(s)':>10} {'Parallel(s)':>12}"
               f" {'Speedup':>9} {'Efficiency':>11}")
        lines.append(hdr)
        lines.append("-" * 68)
        for _, row in merged.sort_values("algorithm").iterrows():
            s  = f"{row['single_time_s']:.4f}"    if pd.notna(row.get("single_time_s"))   else "  N/A  "
            p  = f"{row['parallel_time_s']:.4f}"  if pd.notna(row.get("parallel_time_s")) else "  N/A  "
            sp = f"{row['speedup_x']:.2f}x"       if pd.notna(row.get("speedup_x"))        else "  N/A  "
            ef = f"{row['efficiency_pct']:.1f}%"  if pd.notna(row.get("efficiency_pct"))   else "  N/A  "
            lines.append(f"{row['algorithm']:<18} {s:>10} {p:>12} {sp:>9} {ef:>11}")

        lines.append("-" * 68)
        valid = merged.dropna(subset=["speedup_x"])
        if not valid.empty:
            avg_sp  = valid["speedup_x"].mean()
            avg_eff = valid["efficiency_pct"].mean()
            total_s = valid["single_time_s"].sum()
            total_p = valid["parallel_time_s"].sum()
            lines += [
                f"\nTotal single-thread time : {total_s:.2f}s",
                f"Total parallel time      : {total_p:.2f}s",
                f"Overall speedup          : {total_s/total_p:.2f}x",
                f"Average per-algo speedup : {avg_sp:.2f}x",
                f"Average parallel efficiency: {avg_eff:.1f}%",
                f"Theoretical max speedup  : {n_workers}x",
                "",
                "Notes:",
                "  Efficiency < 100% is expected due to:",
                "  - Process spawn overhead (multiprocessing)",
                "  - GIL contention (threading)",
                "  - Memory bandwidth saturation on shared L3 cache",
                "  - Algorithms with low arithmetic intensity (BFS, RWR)",
            ]
    else:
        log.warning("Single-thread timing CSV not found. "
                    "Run step3 first to get cpu_timing.csv for comparison.")
        lines.append("\nParallel-only times (no single-thread baseline):")
        lines.append(f"\n{'Algorithm':<18} {'Parallel(s)':>12}")
        lines.append("-" * 32)
        for _, row in par_df.iterrows():
            lines.append(f"{row['algorithm']:<18} {row['parallel_time_s']:>12.4f}")
        par_df.to_csv(os.path.join(outdir, "benchmark_comparison.csv"),
                      index=False)

    lines.append("\n" + sep)
    report = "\n".join(lines)
    print("\n" + report)

    rpt_path = os.path.join(outdir, "benchmark_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report saved: {rpt_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Parallel CPU Baselines + Benchmark vs Single-Thread")
    parser.add_argument("--adj", required=True,
                        help="Path to adj_csr.npz (output of step1)")
    parser.add_argument("--expr", default=None,
                        help="Path to expr_mmap.npy (needed for MCE only)")
    parser.add_argument("--genes", default=None,
                        help="Path to genes_filtered.npy")
    parser.add_argument("--single_timing", default=None,
                        help="Path to cpu_timing.csv from step3 "
                             "(default: output/cpu_results/cpu_timing.csv)")
    parser.add_argument("--workers", type=int,
                        default=min(8, cpu_count()),
                        help="Parallel workers (default: min(8, cpu_count()))")
    parser.add_argument("--algo", default="all",
                        choices=["all", "bfs", "pagerank", "rwr",
                                 "hits", "mce", "louvain", "mcl"],
                        help="Which algorithm to run (default: all)")
    parser.add_argument("--bfs_sources", type=int, default=10,
                        help="Number of BFS source nodes (default: 10)")
    parser.add_argument("--rwr_seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4],
                        help="Seed node indices for RWR (default: 0 1 2 3 4)")
    parser.add_argument("--mce_regulators", type=int, default=50,
                        help="Number of regulator genes for MCE (default: 50)")
    parser.add_argument("--outdir", default="output/parallel_results",
                        help="Output directory")
    args = parser.parse_args()

    # Default single_timing path if not given
    if args.single_timing is None:
        args.single_timing = os.path.join(
            "output", "cpu_results", "cpu_timing.csv")

    os.makedirs(args.outdir, exist_ok=True)

    log.info("=" * 68)
    log.info("FYP Step 4 - Parallel CPU Baselines")
    log.info(f"Workers : {args.workers}  |  CPUs available: {cpu_count()}")
    log.info(f"Outdir  : {os.path.abspath(args.outdir)}")
    log.info("=" * 68)

    # Set BLAS threads before any heavy computation
    configure_blas_threads(args.workers)

    # Load graph
    log.info(f"Loading graph: {args.adj}")
    adj  = load_npz(args.adj)
    n    = adj.shape[0]
    nnz  = adj.nnz
    log.info(f"Graph: {n:,} nodes  |  {nnz:,} edges")

    # Load gene IDs
    if args.genes and os.path.exists(args.genes):
        gene_ids = np.load(args.genes, allow_pickle=True)[:n]
        log.info(f"Gene IDs loaded: {len(gene_ids):,}")
    else:
        gene_ids = np.array([f"gene_{i}" for i in range(n)])
        log.warning("--genes not provided. Using generic gene_0, gene_1, ...")

    # Validate expression mmap path
    expr_path = None
    if args.expr and os.path.exists(args.expr):
        expr_path = args.expr
        log.info(f"Expression mmap: {expr_path}")
    elif args.algo in ["all", "mce"]:
        log.warning("--expr not provided or not found. MCE will be skipped.")

    timing_log = {}

    # ------------------------------------------------------------------
    # 1. BFS
    # ------------------------------------------------------------------
    if args.algo in ["all", "bfs"]:
        log.info("\n--- Parallel BFS ---")
        (all_dists, sources), t = parallel_bfs(
            adj, n_sources=args.bfs_sources, n_workers=args.workers)
        timing_log["bfs"] = t
        save_bfs(all_dists, sources, gene_ids,
                 os.path.join(args.outdir, "par_bfs_distances.csv"))
        np.save(os.path.join(args.outdir, "par_bfs_distances.npy"), all_dists)

    # ------------------------------------------------------------------
    # 2. PageRank
    # ------------------------------------------------------------------
    if args.algo in ["all", "pagerank"]:
        log.info("\n--- Parallel PageRank ---")
        pr, t = parallel_pagerank(adj, n_workers=args.workers)
        timing_log["pagerank"] = t
        save_scores(pr, gene_ids,
                    os.path.join(args.outdir, "par_pagerank_scores.csv"),
                    score_name="pagerank")
        np.save(os.path.join(args.outdir, "par_pagerank_scores.npy"), pr)

    # ------------------------------------------------------------------
    # 3. RWR
    # ------------------------------------------------------------------
    if args.algo in ["all", "rwr"]:
        log.info("\n--- Parallel RWR ---")
        seeds = [s for s in args.rwr_seeds if s < n]
        rwr_p, t = parallel_rwr(adj, seed_nodes=seeds, n_workers=args.workers)
        timing_log["rwr"] = t
        save_scores(rwr_p, gene_ids,
                    os.path.join(args.outdir, "par_rwr_scores.csv"),
                    score_name="rwr_relevance")
        np.save(os.path.join(args.outdir, "par_rwr_scores.npy"), rwr_p)

    # ------------------------------------------------------------------
    # 4. HITS
    # ------------------------------------------------------------------
    if args.algo in ["all", "hits"]:
        log.info("\n--- Parallel HITS ---")
        (hub, auth), t = parallel_hits(adj, n_workers=args.workers)
        timing_log["hits"] = t
        save_scores(hub,  gene_ids,
                    os.path.join(args.outdir, "par_hits_hub_scores.csv"),
                    score_name="hub_score")
        save_scores(auth, gene_ids,
                    os.path.join(args.outdir, "par_hits_authority_scores.csv"),
                    score_name="authority_score")

    # ------------------------------------------------------------------
    # 5. MCE
    # ------------------------------------------------------------------
    if args.algo in ["all", "mce"]:
        if expr_path is not None:
            log.info("\n--- Parallel MCE ---")
            regs = list(range(min(args.mce_regulators, n)))
            mce_scores, t = parallel_mce(
                expr_path, gene_ids,
                regulator_indices=regs,
                n_workers=args.workers)
            timing_log["mce"] = t
            mce_df = (pd.DataFrame(list(mce_scores.items()),
                                   columns=["gene_id", "mce_score"])
                      .sort_values("mce_score"))
            mce_df.to_csv(
                os.path.join(args.outdir, "par_mce_scores.csv"), index=False)
            log.info(f"  MCE top regulator: {mce_df.iloc[0]['gene_id']} "
                     f"(MCE={mce_df.iloc[0]['mce_score']:.4f})")
        else:
            log.warning("MCE skipped: --expr not provided or file not found")

    # ------------------------------------------------------------------
    # 6. Louvain
    # ------------------------------------------------------------------
    if args.algo in ["all", "louvain"]:
        log.info("\n--- Parallel Louvain ---")
        partition, t = parallel_louvain(adj, n_workers=args.workers)
        timing_log["louvain"] = t
        save_partition(partition, gene_ids,
                       os.path.join(args.outdir, "par_louvain_communities.csv"))

    # ------------------------------------------------------------------
    # 7. MCL
    # ------------------------------------------------------------------
    if args.algo in ["all", "mcl"]:
        log.info("\n--- Parallel MCL ---")
        (clusters, _), t = parallel_mcl(adj, n_workers=args.workers)
        timing_log["mcl"] = t
        n_mcl    = len(clusters)
        mcl_ids  = (gene_ids[:n_mcl] if len(gene_ids) >= n_mcl
                    else np.array([f"gene_{i}" for i in range(n_mcl)]))
        save_partition(clusters, mcl_ids,
                       os.path.join(args.outdir, "par_mcl_clusters.csv"))

    # ------------------------------------------------------------------
    # Save parallel timing CSV
    # ------------------------------------------------------------------
    par_timing_path = os.path.join(args.outdir, "parallel_timing.csv")
    pd.DataFrame(sorted(timing_log.items()),
                 columns=["algorithm", "parallel_time_seconds"]
                 ).to_csv(par_timing_path, index=False)
    log.info(f"\nParallel timing saved: {par_timing_path}")

    # ------------------------------------------------------------------
    # Build and print benchmark comparison report
    # ------------------------------------------------------------------
    build_benchmark_report(
        timing_log, args.single_timing,
        args.outdir, args.workers, n, nnz
    )


if __name__ == "__main__":
    # REQUIRED on Windows: multiprocessing must be guarded by this
    from multiprocessing import freeze_support
    freeze_support()
    main()
