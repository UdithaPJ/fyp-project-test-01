"""
=============================================================================
STEP 1 (v2): GTEx 21 GB Parquet Preprocessing  →  Co-expression GRN
=============================================================================
Handles: 21 GB parquet file, ~19,789 samples, any number of genes
RAM:     10-12 GB available
CPU:     8 cores
GPU:     RTX 2060 6 GB VRAM (used in step2)

STRATEGY FOR 21 GB / 19,789 SAMPLES:
  The parquet file cannot be loaded into RAM at once.
  We use a three-pass streaming approach:

  PASS 1 – Stream row-groups to collect gene IDs and compute per-sample
           library sizes (column sums). Never hold full matrix in RAM.
           Output: lib_sizes.npy  (19,789 floats)

  PASS 2 – Stream row-groups again. For each batch of genes:
           - CPM-normalise using lib_sizes
           - log2(CPM+1) transform
           - Apply CPM filter
           - Accumulate kept rows into a memory-mapped file on disk
           Output: expr_mmap.npy  (memory-mapped, G_kept × S float32)

  PASS 3 – Chunked correlation on the memory-mapped file.
           Each chunk of genes is z-scored and correlated against all others
           using 8-core multiprocessing. The mmap file is read-only shared
           across workers without copying.
           Output: edge_list.csv + adj_csr.npz

DISK SPACE NEEDED:
  expr_mmap.npy: G_kept × 19789 × 4 bytes
  For 10,000 genes: ~800 MB  ✓
  For 50,000 genes: ~4 GB    ✓ (still fine)
  For 200,000 genes: ~16 GB  (set --max_genes to reduce)

HOW TO RUN:
  pip install numpy scipy pandas pyarrow tqdm psutil

  python step1_preprocess_gtex.py \
      --input  GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_exon_reads.parquet \
      --max_genes       10000  \
      --cpm_threshold   1.0    \
      --min_samples_frac 0.2   \
      --corr_threshold  0.7    \
      --chunk_size      300    \
      --workers         8      \
      --outdir          output

  For safety first run (quick test):
      --max_genes 2000 --workers 4

OUTPUTS in --outdir:
  genes_filtered.npy       gene ID strings (G_kept,)
  samples.npy              sample ID strings (S,)
  lib_sizes.npy            per-sample library sizes (S,)
  expr_mmap.npy            memory-mapped expression matrix (G_kept × S) float32
  edge_list.csv            src, dst, weight  (Pearson r ≥ threshold)
  adj_csr.npz              scipy CSR sparse adjacency
  graph_stats.txt          network statistics
  preprocess.log           full log
=============================================================================
"""

import os
import sys
import gc
import time
import argparse
import warnings
import logging
import psutil
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
from multiprocessing import Pool
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

import io
# Force UTF-8 on Windows stdout so Unicode characters never crash the logger
if hasattr(sys.stdout, 'buffer'):
    _utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                    errors='replace', line_buffering=True)
else:
    _utf8_stdout = sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(_utf8_stdout),
        logging.FileHandler("output/preprocess.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def mem_gb():
    return psutil.virtual_memory().available / 1e9


def report_mem(label=""):
    vm = psutil.virtual_memory()
    log.info(f"[MEM]  {label:35s}  avail={mem_gb():.1f}GB  used={vm.percent:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Parquet introspection
# ─────────────────────────────────────────────────────────────────────────────
def inspect_parquet(path: str):
    """
    Returns metadata without loading any data:
      total_rows, total_cols, n_row_groups, schema column names
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    schema = pf.schema_arrow

    total_rows = meta.num_rows
    total_cols = meta.num_columns
    n_rg = meta.num_row_groups

    # Estimate uncompressed size
    uncomp_bytes = sum(
        meta.row_group(i).total_byte_size for i in range(n_rg)
    )

    log.info(f"Parquet file: {os.path.getsize(path)/1e9:.2f} GB on disk")
    log.info(f"Rows (genes): {total_rows:,}")
    log.info(f"Columns:      {total_cols:,}  (including gene ID column)")
    log.info(f"Row groups:   {n_rg}")
    log.info(f"Uncompressed: ~{uncomp_bytes/1e9:.1f} GB")
    log.info(f"Column names (first 5): {[schema.field(i).name for i in range(min(5, total_cols))]}")

    return pf, total_rows, total_cols, n_rg, schema


def detect_gene_id_column(schema):
    """Find the non-sample column (gene ID column) by name heuristic."""
    import pyarrow as pa
    for i in range(len(schema)):
        field = schema.field(i)
        name = field.name.lower()
        # String / categorical type OR named like a gene ID
        if (pa.types.is_string(field.type) or
                pa.types.is_large_string(field.type) or
                pa.types.is_dictionary(field.type) or
                any(kw in name for kw in ["gene", "id", "name", "feature",
                                          "transcript", "exon", "symbol"])):
            log.info(f"Gene ID column detected: '{schema.field(i).name}' (index {i})")
            return schema.field(i).name
    log.warning("No gene ID column found by heuristic – using row index as gene ID")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1: Stream to get sample IDs + library sizes
# ─────────────────────────────────────────────────────────────────────────────
def pass1_library_sizes(pf, gene_id_col: str, max_genes: int, out_dir: str):
    """
    Streams through row groups to:
      - Collect sample IDs (column names)
      - Compute per-sample library sizes (sum of all raw counts per sample)
      - Collect gene IDs (up to max_genes)

    Memory: holds only one row-group at a time.
    """
    log.info("PASS 1: Computing library sizes (streaming row groups) ...")
    import pyarrow as pa

    sample_ids = None
    gene_ids_all = []
    lib_sizes = None
    genes_seen = 0
    t0 = time.time()

    n_rg = pf.metadata.num_row_groups
    for rg_idx in tqdm(range(n_rg), desc="Pass1 row-groups"):
        if genes_seen >= max_genes:
            break

        tbl = pf.read_row_group(rg_idx)
        df = tbl.to_pandas()
        del tbl

        # First row group: establish sample IDs
        if sample_ids is None:
            if gene_id_col and gene_id_col in df.columns:
                sample_cols = [c for c in df.columns if c != gene_id_col]
            else:
                # Assume first column is gene ID
                sample_cols = df.columns[1:].tolist()
                if gene_id_col is None:
                    gene_id_col = df.columns[0]
            sample_ids = np.array(sample_cols)
            n_samples = len(sample_ids)
            lib_sizes = np.zeros(n_samples, dtype=np.float64)
            log.info(f"  Samples detected: {n_samples:,}")

        # Collect gene IDs
        remaining = max_genes - genes_seen
        if gene_id_col in df.columns:
            batch_gene_ids = df[gene_id_col].astype(str).values[:remaining]
            expr_chunk = df[sample_cols].values[:remaining].astype(np.float64)
        else:
            n_chunk = min(len(df), remaining)
            batch_gene_ids = np.array([f"gene_{genes_seen + i}" for i in range(n_chunk)])
            expr_chunk = df.iloc[:n_chunk].values.astype(np.float64)

        gene_ids_all.extend(batch_gene_ids.tolist())
        lib_sizes += expr_chunk.sum(axis=0)
        genes_seen += len(batch_gene_ids)

        del df, expr_chunk
        gc.collect()

    gene_ids = np.array(gene_ids_all)
    lib_sizes[lib_sizes == 0] = 1.0  # avoid divide-by-zero

    # Save
    np.save(os.path.join(out_dir, "lib_sizes.npy"), lib_sizes)
    np.save(os.path.join(out_dir, "samples.npy"), sample_ids)

    log.info(f"Pass 1 done in {time.time()-t0:.1f}s")
    log.info(f"  Genes collected: {len(gene_ids):,}")
    log.info(f"  Samples:         {len(sample_ids):,}")
    log.info(f"  Library size range: [{lib_sizes.min():.0f}, {lib_sizes.max():.0f}]")
    report_mem("after pass1")

    return gene_ids, sample_ids, lib_sizes, sample_cols, gene_id_col


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2: Stream to normalise + filter + write memory-mapped output
# ─────────────────────────────────────────────────────────────────────────────
def pass2_normalise_filter(pf, gene_ids_raw: np.ndarray, sample_cols: list,
                           gene_id_col: str, lib_sizes: np.ndarray,
                           cpm_threshold: float, min_samples_frac: float,
                           max_genes: int, out_dir: str):
    """
    Streams row groups, normalises to log2(CPM+1), applies CPM filter,
    and writes passing genes into a memory-mapped numpy file.

    The mmap file is pre-allocated at max possible size, then trimmed.
    """
    log.info("PASS 2: Normalising + filtering -> memory-mapped file ...")
    n_samples = len(sample_cols)
    n_genes_raw = len(gene_ids_raw)
    min_samples = int(min_samples_frac * n_samples)

    # Pre-allocate mmap: worst case all genes pass
    mmap_path = os.path.join(out_dir, "_expr_mmap_tmp.npy")
    mmap_final_path = os.path.join(out_dir, "expr_mmap.npy")

    # Write numpy header manually so we can mmap the data region
    mmap_shape = (n_genes_raw, n_samples)
    mmap_size_gb = n_genes_raw * n_samples * 4 / 1e9
    log.info(f"  Pre-allocating mmap: {n_genes_raw:,} × {n_samples:,} = {mmap_size_gb:.2f} GB")

    if mmap_size_gb > 30:
        log.warning(f"  Mmap > 30 GB! Reduce --max_genes. Proceeding anyway...")

    # Use np.memmap directly
    mmap_arr = np.memmap(mmap_path, dtype=np.float32, mode='w+',
                         shape=mmap_shape)

    kept_indices_raw = []   # which raw row indices passed the filter
    kept_gene_ids = []
    write_row = 0
    genes_processed = 0
    t0 = time.time()

    n_rg = pf.metadata.num_row_groups
    for rg_idx in tqdm(range(n_rg), desc="Pass2 row-groups"):
        if genes_processed >= max_genes:
            break

        tbl = pf.read_row_group(rg_idx)
        df = tbl.to_pandas()
        del tbl

        remaining = max_genes - genes_processed
        if gene_id_col in df.columns:
            expr_raw = df[sample_cols].values[:remaining].astype(np.float32)
            g_ids = df[gene_id_col].astype(str).values[:remaining]
        else:
            expr_raw = df.values[:remaining].astype(np.float32)
            g_ids = np.array([f"gene_{genes_processed + i}" for i in range(len(expr_raw))])

        del df
        gc.collect()

        # CPM normalise: divide by lib_sizes (broadcast), multiply 1e6
        cpm = (expr_raw / lib_sizes.astype(np.float32)) * 1e6

        # CPM filter
        pass_mask = (cpm >= cpm_threshold).sum(axis=1) >= min_samples

        # Log2 transform passing genes only
        cpm_kept = cpm[pass_mask]
        log2_cpm = np.log2(cpm_kept + 1.0)

        n_kept_chunk = log2_cpm.shape[0]
        if n_kept_chunk > 0:
            mmap_arr[write_row: write_row + n_kept_chunk] = log2_cpm
            kept_gene_ids.extend(g_ids[pass_mask].tolist())
            write_row += n_kept_chunk

        genes_processed += len(g_ids)

        del expr_raw, cpm, cpm_kept, log2_cpm
        gc.collect()

    # Flush and trim the mmap to actual kept rows
    mmap_arr.flush()
    del mmap_arr

    n_kept = write_row
    log.info(f"Pass 2 done in {time.time()-t0:.1f}s")
    log.info(f"  Kept {n_kept:,} / {genes_processed:,} genes after CPM filter")

    # Create final correctly-shaped mmap by copying trimmed slice
    log.info(f"  Trimming mmap to ({n_kept}, {n_samples}) ...")
    src = np.memmap(mmap_path, dtype=np.float32, mode='r',
                    shape=(n_genes_raw, n_samples))
    final = np.memmap(mmap_final_path, dtype=np.float32, mode='w+',
                      shape=(n_kept, n_samples))
    # Copy in chunks to avoid RAM spike
    # IMPORTANT: clamp end index so last chunk never overruns n_kept
    chunk = 500
    for i in range(0, n_kept, chunk):
        end = min(i + chunk, n_kept)          # actual slice end, never > n_kept
        final[i:end] = src[i:end]             # both slices are the same size
    final.flush()
    del src, final

    # Remove oversized temp file
    os.remove(mmap_path)

    kept_gene_ids = np.array(kept_gene_ids[:n_kept])
    np.save(os.path.join(out_dir, "genes_filtered.npy"), kept_gene_ids)

    report_mem("after pass2")
    log.info(f"  Mmap saved: {mmap_final_path}  "
             f"({n_kept*n_samples*4/1e9:.2f} GB on disk)")

    return kept_gene_ids, mmap_final_path, n_kept, n_samples


# ─────────────────────────────────────────────────────────────────────────────
# PASS 3: Chunked correlation via memory-mapped reads
# ─────────────────────────────────────────────────────────────────────────────
def _zscore_chunk(arr):
    """Z-score rows of a 2D array."""
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def _corr_worker(args):
    """
    Worker function: computes Pearson r between rows [i_start:i_end]
    of z_mmap and ALL rows, then returns edges above threshold.

    Reads from memory-mapped files — no large array copying between processes.
    """
    i_start, i_end, n_genes, n_samples, mmap_path, z_mmap_path, threshold = args

    # Open mmap read-only (shared, no copy)
    z_all = np.memmap(z_mmap_path, dtype=np.float32, mode='r',
                      shape=(n_genes, n_samples))
    z_chunk = z_all[i_start:i_end].copy()   # small local copy

    # (chunk × S) @ (S × G) = (chunk × G)
    corr = (z_chunk @ z_all.T) / (n_samples - 1)

    rows, cols, vals = [], [], []
    for local_i in range(i_end - i_start):
        gi = i_start + local_i
        row_r = corr[local_i]
        # upper triangle only (avoid duplicates) + threshold
        j_idx = np.where((row_r >= threshold) &
                         (np.arange(n_genes) > gi))[0]
        rows.extend([gi] * len(j_idx))
        cols.extend(j_idx.tolist())
        vals.extend(row_r[j_idx].tolist())

    del z_all, z_chunk, corr
    return rows, cols, vals


def pass3_correlate(mmap_path: str, n_genes: int, n_samples: int,
                    out_dir: str, chunk_size: int = 300,
                    n_workers: int = 8, corr_threshold: float = 0.7):
    """
    Reads the memory-mapped normalised expression, z-scores it into a
    second mmap, then distributes chunked correlation jobs to workers.

    Two mmaps on disk during this step:
      expr_mmap.npy     – log2 CPM  (kept from pass2)
      z_mmap.npy        – z-scored  (temporary, deleted at end)
    """
    log.info("PASS 3: Chunked correlation on memory-mapped data ...")

    # ── Z-score the full mmap → write z_mmap ──────────────────────────────
    z_mmap_path = os.path.join(out_dir, "_z_mmap.npy")
    log.info(f"  Z-scoring {n_genes:,} genes × {n_samples:,} samples ...")
    log.info(f"  Z-mmap size: {n_genes*n_samples*4/1e9:.2f} GB on disk")

    expr_mm = np.memmap(mmap_path, dtype=np.float32, mode='r',
                        shape=(n_genes, n_samples))
    z_mm = np.memmap(z_mmap_path, dtype=np.float32, mode='w+',
                     shape=(n_genes, n_samples))

    batch = 500
    for i in tqdm(range(0, n_genes, batch), desc="Z-scoring"):
        end = min(i + batch, n_genes)
        chunk = expr_mm[i:end].copy().astype(np.float64)
        mu = chunk.mean(axis=1, keepdims=True)
        sd = chunk.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        z_mm[i:end] = ((chunk - mu) / sd).astype(np.float32)

    z_mm.flush()
    del expr_mm, z_mm
    gc.collect()
    log.info("  Z-scoring complete.")

    # ── Build chunk arguments for workers ─────────────────────────────────
    chunks = [
        (i, min(i + chunk_size, n_genes),
         n_genes, n_samples, mmap_path, z_mmap_path, corr_threshold)
        for i in range(0, n_genes, chunk_size)
    ]
    log.info(f"  {len(chunks)} correlation chunks  ×  {n_workers} workers")

    all_rows, all_cols, all_vals = [], [], []
    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        for rows, cols, vals in tqdm(
                pool.imap_unordered(_corr_worker, chunks),
                total=len(chunks), desc="Correlation"):
            all_rows.extend(rows)
            all_cols.extend(cols)
            all_vals.extend(vals)

    elapsed = time.time() - t0
    os.remove(z_mmap_path)

    log.info(f"  Correlation done in {elapsed/60:.1f} min")
    log.info(f"  Edges above threshold: {len(all_vals):,}")
    report_mem("after pass3")

    return all_rows, all_cols, all_vals


# ─────────────────────────────────────────────────────────────────────────────
# Build + save CSR
# ─────────────────────────────────────────────────────────────────────────────
def build_save_graph(gene_ids, rows, cols, vals, n, out_dir):
    log.info("Building CSR adjacency matrix ...")

    # Symmetrise
    r = np.array(rows + cols, dtype=np.int32)
    c = np.array(cols + rows, dtype=np.int32)
    v = np.array(vals + vals, dtype=np.float32)

    csr = sparse.csr_matrix((v, (r, c)), shape=(n, n))
    csr.eliminate_zeros()

    save_npz(os.path.join(out_dir, "adj_csr.npz"), csr)

    edge_df = pd.DataFrame({"src": rows, "dst": cols, "weight": vals})
    edge_df.to_csv(os.path.join(out_dir, "edge_list.csv"), index=False)

    degrees = np.array(csr.sum(axis=1)).flatten()
    with open(os.path.join(out_dir, "graph_stats.txt"), "w") as f:
        f.write(f"Nodes (genes)    : {n:,}\n")
        f.write(f"Edges (symmetric): {csr.nnz:,}\n")
        f.write(f"Edges (unique)   : {len(rows):,}\n")
        f.write(f"Density          : {csr.nnz/(n*n):.8f}\n")
        f.write(f"Avg degree       : {degrees.mean():.2f}\n")
        f.write(f"Max degree       : {degrees.max():.0f}\n")
        f.write(f"Isolated nodes   : {(degrees==0).sum():,}\n")

    log.info(f"CSR saved: {n:,} nodes, {csr.nnz:,} edges")
    return csr


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Memory-safe GTEx 21GB Parquet Preprocessing Pipeline")
    parser.add_argument("--input", required=True,
                        help="Path to GTEx .parquet file")
    parser.add_argument("--max_genes", type=int, default=10000,
                        help="Max genes to process. Each gene costs 19789×4=~77KB RAM. "
                             "10000 genes ≈ 800MB expression matrix. "
                             "Default: 10000. Use 50000 if you have 6+ GB free disk.")
    parser.add_argument("--cpm_threshold", type=float, default=1.0,
                        help="CPM threshold for keeping a gene (default: 1.0)")
    parser.add_argument("--min_samples_frac", type=float, default=0.2,
                        help="Min fraction of samples with CPM >= threshold (default: 0.2)")
    parser.add_argument("--corr_threshold", type=float, default=0.7,
                        help="Pearson r threshold for edges (default: 0.7, "
                             "use 0.8+ for sparser graph, 0.6 for denser)")
    parser.add_argument("--chunk_size", type=int, default=300,
                        help="Genes per correlation chunk (default: 300). "
                             "Higher = faster but more RAM per worker.")
    parser.add_argument("--workers", type=int, default=8,
                        help="CPU workers for correlation (default: 8)")
    parser.add_argument("--outdir", default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--skip_pass1", action="store_true",
                        help="Skip Pass 1 if lib_sizes.npy already exists")
    parser.add_argument("--skip_pass2", action="store_true",
                        help="Skip Pass 2 if expr_mmap.npy already exists")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_total = time.time()

    log.info("=" * 65)
    log.info("FYP Milestone 1.2 – GTEx 21 GB Preprocessing (v2, streaming)")
    log.info("=" * 65)
    report_mem("startup")

    # ── Inspect ──────────────────────────────────────────────────────────
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.input)
    pf, total_rows, total_cols, n_rg, schema = inspect_parquet(args.input)
    gene_id_col = detect_gene_id_column(schema)

    # Warn if max_genes > available rows
    if args.max_genes > total_rows:
        log.warning(f"--max_genes {args.max_genes} > file rows {total_rows}. "
                    f"Setting max_genes = {total_rows}")
        args.max_genes = total_rows

    # Estimate disk space needed for mmap
    # (conservative: assume 80% genes pass filter)
    est_kept = int(args.max_genes * 0.8)
    n_samples_est = total_cols - 1  # minus gene ID column
    mmap_gb = est_kept * n_samples_est * 4 / 1e9
    log.info(f"Estimated mmap size: ~{mmap_gb:.1f} GB  "
             f"({est_kept:,} genes × {n_samples_est:,} samples)")

    disk = psutil.disk_usage(args.outdir)
    if mmap_gb * 2 > disk.free / 1e9:
        log.warning(f"Low disk space! Free: {disk.free/1e9:.1f} GB, "
                    f"need ~{mmap_gb*2:.1f} GB. Consider reducing --max_genes.")

    # ── PASS 1 ────────────────────────────────────────────────────────────
    lib_path  = os.path.join(args.outdir, "lib_sizes.npy")
    samp_path = os.path.join(args.outdir, "samples.npy")

    if args.skip_pass1 and os.path.exists(lib_path) and os.path.exists(samp_path):
        log.info("Skipping Pass 1 (--skip_pass1 set and files exist)")
        lib_sizes  = np.load(lib_path)
        sample_ids = np.load(samp_path, allow_pickle=True)
        # Re-derive sample_cols from schema
        all_cols = [schema.field(i).name for i in range(len(schema))]
        sample_cols = [c for c in all_cols if c != gene_id_col]
        gene_ids_raw = None  # will be loaded from pass2 output
    else:
        pf2 = pq.ParquetFile(args.input)  # fresh file handle for pass1
        gene_ids_raw, sample_ids, lib_sizes, sample_cols, gene_id_col = \
            pass1_library_sizes(pf2, gene_id_col, args.max_genes, args.outdir)

    n_samples = len(sample_ids)

    # ── PASS 2 ────────────────────────────────────────────────────────────
    mmap_final = os.path.join(args.outdir, "expr_mmap.npy")
    gene_ids_path = os.path.join(args.outdir, "genes_filtered.npy")

    if args.skip_pass2 and os.path.exists(mmap_final) and os.path.exists(gene_ids_path):
        log.info("Skipping Pass 2 (--skip_pass2 set and files exist)")
        gene_ids = np.load(gene_ids_path, allow_pickle=True)
        mm = np.memmap(mmap_final, dtype=np.float32, mode='r')
        n_kept = mm.shape[0] if mm.ndim > 1 else len(mm) // n_samples
        del mm
    else:
        if gene_ids_raw is None:
            gene_ids_raw_arr = np.array([f"gene_{i}" for i in range(args.max_genes)])
        else:
            gene_ids_raw_arr = gene_ids_raw

        pf3 = pq.ParquetFile(args.input)
        gene_ids, mmap_final, n_kept, n_samples = pass2_normalise_filter(
            pf3, gene_ids_raw_arr, sample_cols, gene_id_col,
            lib_sizes, args.cpm_threshold, args.min_samples_frac,
            args.max_genes, args.outdir
        )

    log.info(f"\nExpression matrix ready: {n_kept:,} genes × {n_samples:,} samples")

    # ── PASS 3 ────────────────────────────────────────────────────────────
    gene_ids = np.load(os.path.join(args.outdir, "genes_filtered.npy"),
                       allow_pickle=True)

    # Warn about very large correlation jobs
    n_pairs = n_kept * (n_kept - 1) // 2
    log.info(f"Correlation pairs: {n_pairs:,}")
    if n_kept > 20000:
        log.warning(f"{n_kept:,} genes → {n_pairs:,} pairs. "
                    f"This may take several hours. "
                    f"Consider --corr_threshold 0.8 or reducing --max_genes.")

    rows, cols, vals = pass3_correlate(
        mmap_final, n_kept, n_samples,
        args.outdir,
        chunk_size=args.chunk_size,
        n_workers=args.workers,
        corr_threshold=args.corr_threshold
    )

    # ── Build graph ───────────────────────────────────────────────────────
    build_save_graph(gene_ids, rows, cols, vals, n_kept, args.outdir)

    total_elapsed = time.time() - t_total
    log.info(f"\n{'='*65}")
    log.info(f"Pipeline complete in {total_elapsed/60:.1f} min")
    log.info(f"Outputs in: {os.path.abspath(args.outdir)}/")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()