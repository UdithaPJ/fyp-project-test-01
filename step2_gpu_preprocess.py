r"""
=============================================================================
STEP 2 (v3): GPU-Accelerated Correlation  -  GTX 1650 / 4.3 GB VRAM
             NVRTC-free implementation
=============================================================================
ROOT CAUSE OF PREVIOUS ERROR:
  CuPy operations like cp.arange(), cp.where(), and boolean array indexing
  require NVRTC (the NVIDIA Runtime Compiler) to JIT-compile elementwise
  GPU kernels. If your installed CuPy version does not match the exact CUDA
  toolkit DLL version on Windows, you get:
    "Could not find module 'nvrtc64_112_0.dll'"

THIS VERSION AVOIDS NVRTC ENTIRELY:
  - The only GPU operation used is matrix multiplication (z_tile @ z_gpu.T)
    which calls cuBLAS directly, with no JIT compilation needed.
  - All masking, thresholding, and indexing is done in NumPy on CPU after
    transferring the small correlation tile back from GPU.
  - This trades a tiny amount of speed (GPU masking vs CPU masking) for
    complete NVRTC independence.

YOUR HARDWARE:
  GTX 1650 Max-Q  |  4.3 GB total  |  ~3.5 GB free
  3,997 genes x 19,788 samples  ->  Z-matrix = 316 MB  -> fits easily

HOW TO FIX YOUR CUPY INSTALLATION (optional, for future):
  1. Check your CUDA version:  nvidia-smi  (top right shows CUDA version)
  2. Uninstall current cupy:   pip uninstall cupy-cuda11x cupy-cuda12x cupy
  3. Install matching version:
       CUDA 11.x -> pip install cupy-cuda11x
       CUDA 12.x -> pip install cupy-cuda12x
  4. Set CUDA_PATH environment variable to your CUDA toolkit folder, e.g.:
       $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

HOW TO RUN:
  python step2_gpu_preprocess.py ^
      --expr_mmap output/expr_mmap.npy ^
      --gene_ids  output/genes_filtered.npy ^
      --outdir    output

VRAM guidance for GTX 1650 (4.3 GB / 3.5 GB free):
  genes   | Z-matrix  | mode
  --------|-----------|------------------
   3,997  |  316 MB   | full (your case)
  10,000  |  800 MB   | full
  30,000  |  2.4 GB   | full (near limit)
  >35,000 |  > 2.8 GB | streaming (auto)
=============================================================================
"""

import os
import sys
import io
import time
import argparse
import logging
import numpy as np
from scipy import sparse
from scipy.sparse import save_npz

os.makedirs("output", exist_ok=True)

# Windows-safe UTF-8 logging
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
        logging.FileHandler("output/gpu_preprocess.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# GPU check - probes only the CUDA runtime (no NVRTC needed)
# =============================================================================
def check_gpu():
    try:
        import cupy as cp
    except ImportError:
        log.error("CuPy not installed.")
        log.error("  pip install cupy-cuda11x   (if CUDA 11.x)")
        log.error("  pip install cupy-cuda12x   (if CUDA 12.x)")
        log.error("  Check version: nvidia-smi  (top-right shows CUDA version)")
        sys.exit(1)

    # Test CUDA runtime (not NVRTC)
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props['name']
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        log.info(f"GPU : {gpu_name}")
        log.info(f"VRAM: {total/1e9:.1f} GB total  |  {free/1e9:.1f} GB free")
    except Exception as e:
        log.error(f"CUDA runtime error: {e}")
        sys.exit(1)

    # Test cuBLAS matmul (this is ALL we use - no NVRTC)
    try:
        a = cp.ones((4, 4), dtype=cp.float32)
        b = (a @ a).get()
        log.info("cuBLAS matmul test: PASSED")
    except Exception as e:
        log.error(f"cuBLAS test failed: {e}")
        log.error("Cannot perform GPU matrix multiplication.")
        sys.exit(1)

    # Check NVRTC availability but do not exit if missing
    try:
        _ = cp.arange(4)
        log.info("NVRTC test: PASSED (full CuPy available)")
    except Exception:
        log.warning("NVRTC unavailable (DLL version mismatch) - this is OK.")
        log.warning("This script uses only cuBLAS which does NOT need NVRTC.")

    return cp, free, total


# =============================================================================
# Z-score on CPU (avoids NVRTC-dependent elementwise GPU ops)
# =============================================================================
def zscore_cpu(expr: np.ndarray) -> np.ndarray:
    log.info("Z-scoring on CPU ...")
    t0 = time.time()
    expr64 = expr.astype(np.float64)
    mu = expr64.mean(axis=1, keepdims=True)
    sd = expr64.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = ((expr64 - mu) / sd).astype(np.float32)
    del expr64, mu, sd
    log.info(f"Z-score done: {time.time()-t0:.2f}s")
    return z


# =============================================================================
# Full mode: upload entire Z-matrix to GPU, tile matmul, threshold on CPU
# =============================================================================
def _full_mode(cp, z_cpu: np.ndarray, n_genes: int, n_samples: int,
               tile_rows: int, threshold: float):
    """
    Upload Z-matrix once. For each tile of rows:
      GPU: (tile x S) @ (S x G) = (tile x G) correlation matrix  [cuBLAS]
      CPU: threshold and collect edges                             [NumPy]

    No NVRTC used anywhere in this function.
    """
    log.info(f"Uploading {z_cpu.nbytes/1e6:.0f} MB to GPU ...")
    t_upload = time.time()
    z_gpu = cp.asarray(z_cpu)
    cp.cuda.Stream.null.synchronize()
    log.info(f"Upload done: {time.time()-t_upload:.2f}s")

    rows_out, cols_out, vals_out = [], [], []
    n_tiles = (n_genes + tile_rows - 1) // tile_rows
    j_all = np.arange(n_genes, dtype=np.int32)   # used for upper-tri mask on CPU

    t0 = time.time()
    for tile_i, i_start in enumerate(range(0, n_genes, tile_rows)):
        i_end = min(i_start + tile_rows, n_genes)
        tile_len = i_end - i_start

        # --- GPU: cuBLAS matmul only ---
        corr_gpu = (z_gpu[i_start:i_end] @ z_gpu.T) / (n_samples - 1)

        # --- CPU: transfer + threshold + collect ---
        corr_cpu = corr_gpu.get()          # device -> host  (tile x G) float32
        del corr_gpu
        cp.get_default_memory_pool().free_all_blocks()

        for li in range(tile_len):
            gi = i_start + li
            row_r = corr_cpu[li]
            mask = (row_r >= threshold) & (j_all > gi)
            j_idx = np.where(mask)[0]
            if len(j_idx) > 0:
                rows_out.extend([gi] * len(j_idx))
                cols_out.extend(j_idx.tolist())
                vals_out.extend(row_r[j_idx].tolist())

        if (tile_i + 1) % 10 == 0 or tile_i == n_tiles - 1:
            elapsed = time.time() - t0
            rate = (tile_i + 1) / elapsed if elapsed > 0 else 1
            eta = (n_tiles - tile_i - 1) / rate
            log.info(f"  Tile {tile_i+1:4d}/{n_tiles}  |  "
                     f"edges: {len(vals_out):,}  |  "
                     f"elapsed: {elapsed/60:.1f}min  |  "
                     f"ETA: {eta/60:.1f}min")

    del z_gpu
    cp.get_default_memory_pool().free_all_blocks()
    log.info(f"Correlation complete: {(time.time()-t0)/60:.2f} min  |  "
             f"Edges: {len(vals_out):,}")
    return rows_out, cols_out, vals_out


# =============================================================================
# Streaming mode: one tile-pair at a time (for large gene counts)
# =============================================================================
def _streaming_mode(cp, z_cpu: np.ndarray, n_genes: int, n_samples: int,
                    tile_rows: int, threshold: float):
    log.info("Streaming mode: uploading tile pairs one at a time ...")
    rows_out, cols_out, vals_out = [], [], []
    n_tiles = (n_genes + tile_rows - 1) // tile_rows
    t0 = time.time()

    for i_idx, i_start in enumerate(range(0, n_genes, tile_rows)):
        i_end = min(i_start + tile_rows, n_genes)
        z_i_gpu = cp.asarray(z_cpu[i_start:i_end])

        for j_start in range(i_start, n_genes, tile_rows):
            j_end = min(j_start + tile_rows, n_genes)
            z_j_gpu = cp.asarray(z_cpu[j_start:j_end])

            corr_gpu = (z_i_gpu @ z_j_gpu.T) / (n_samples - 1)
            corr_cpu = corr_gpu.get()
            del corr_gpu, z_j_gpu
            cp.get_default_memory_pool().free_all_blocks()

            tile_i_len = i_end - i_start
            tile_j_len = j_end - j_start
            for li in range(tile_i_len):
                gi = i_start + li
                for lj in range(tile_j_len):
                    gj = j_start + lj
                    if gj <= gi:
                        continue
                    r = corr_cpu[li, lj]
                    if r >= threshold:
                        rows_out.append(gi)
                        cols_out.append(gj)
                        vals_out.append(float(r))

        del z_i_gpu
        cp.get_default_memory_pool().free_all_blocks()

        if (i_idx + 1) % 5 == 0 or i_idx == n_tiles - 1:
            elapsed = time.time() - t0
            rate = (i_idx + 1) / elapsed if elapsed > 0 else 1
            eta = (n_tiles - i_idx - 1) / rate
            log.info(f"  Outer tile {i_idx+1}/{n_tiles}  |  "
                     f"edges: {len(vals_out):,}  |  ETA: {eta/60:.1f}min")

    log.info(f"Streaming done: {(time.time()-t0)/60:.2f} min  |  "
             f"Edges: {len(vals_out):,}")
    return rows_out, cols_out, vals_out


# =============================================================================
# Main dispatcher
# =============================================================================
def gpu_correlate(cp, expr_cpu: np.ndarray, vram_free: int,
                  tile_rows: int = 500, corr_threshold: float = 0.7):
    n_genes, n_samples = expr_cpu.shape
    z_bytes = n_genes * n_samples * 4
    # Keep 700 MB headroom for the correlation tile + cuBLAS workspace
    vram_budget = max(vram_free - 700 * 1024**2, vram_free // 2)

    log.info(f"Genes: {n_genes:,}  |  Samples: {n_samples:,}")
    log.info(f"Z-matrix: {z_bytes/1e6:.0f} MB  |  "
             f"VRAM budget: {vram_budget/1e9:.2f} GB")

    z_cpu = zscore_cpu(expr_cpu)

    if z_bytes <= vram_budget:
        log.info("Mode: FULL (entire Z-matrix fits in VRAM)")
        return _full_mode(cp, z_cpu, n_genes, n_samples, tile_rows, corr_threshold)
    else:
        log.info("Mode: STREAMING (Z-matrix larger than VRAM budget)")
        return _streaming_mode(cp, z_cpu, n_genes, n_samples, tile_rows, corr_threshold)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NVRTC-free GPU Correlation (GTX 1650 / any CUDA GPU)")
    parser.add_argument("--expr_mmap", required=True,
                        help="Path to expr_mmap.npy from step1")
    parser.add_argument("--gene_ids", required=True,
                        help="Path to genes_filtered.npy from step1")
    parser.add_argument("--n_genes", type=int, default=None,
                        help="Process only first N genes (for quick testing)")
    parser.add_argument("--corr_threshold", type=float, default=0.7,
                        help="Pearson r cutoff (default: 0.7)")
    parser.add_argument("--tile_rows", type=int, default=500,
                        help="Rows per GPU matmul tile. "
                             "GTX 1650 safe values: 500-2000. "
                             "Higher = faster, but needs more VRAM per tile.")
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    log.info("=" * 60)
    log.info("FYP Step 2 - GPU Correlation (NVRTC-free, cuBLAS only)")
    log.info("=" * 60)

    cp, vram_free, vram_total = check_gpu()

    gene_ids = np.load(args.gene_ids, allow_pickle=True)
    n_genes_total = len(gene_ids)

    # Infer n_samples from file size
    file_bytes = os.path.getsize(args.expr_mmap)
    n_samples = file_bytes // (n_genes_total * 4)
    log.info(f"Expression mmap: {n_genes_total:,} genes x {n_samples:,} samples")
    log.info(f"File size: {file_bytes/1e9:.3f} GB")

    n_genes = min(args.n_genes or n_genes_total, n_genes_total)
    if n_genes < n_genes_total:
        log.info(f"Processing first {n_genes:,} of {n_genes_total:,} genes")

    log.info(f"Loading {n_genes:,} genes from mmap into RAM ...")
    t_load = time.time()
    expr = np.memmap(args.expr_mmap, dtype=np.float32, mode='r',
                     shape=(n_genes_total, n_samples))[:n_genes].copy()
    log.info(f"Loaded: {expr.nbytes/1e6:.0f} MB  ({time.time()-t_load:.2f}s)")

    t0 = time.time()
    rows, cols, vals = gpu_correlate(
        cp, expr, vram_free,
        tile_rows=args.tile_rows,
        corr_threshold=args.corr_threshold
    )
    elapsed = time.time() - t0

    # Build CSR
    n = n_genes
    r_sym = np.array(rows + cols, dtype=np.int32)
    c_sym = np.array(cols + rows, dtype=np.int32)
    v_sym = np.array(vals + vals, dtype=np.float32)
    csr = sparse.csr_matrix((v_sym, (r_sym, c_sym)), shape=(n, n))
    csr.eliminate_zeros()

    out_path = os.path.join(args.outdir, "adj_csr_gpu.npz")
    save_npz(out_path, csr)
    log.info(f"CSR saved: {out_path}  (nnz={csr.nnz:,})")

    with open(os.path.join(args.outdir, "gpu_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"GPU              : {vram_total/1e9:.1f} GB VRAM\n")
        f.write(f"Genes            : {n_genes:,}\n")
        f.write(f"Samples          : {n_samples:,}\n")
        f.write(f"Threshold (r)    : {args.corr_threshold}\n")
        f.write(f"Edges (unique)   : {len(vals):,}\n")
        f.write(f"CSR nnz          : {csr.nnz:,}\n")
        f.write(f"Total time (s)   : {elapsed:.1f}\n")
        f.write(f"Total time (min) : {elapsed/60:.2f}\n")

    log.info("=" * 60)
    log.info(f"GPU pipeline complete: {elapsed/60:.2f} min")
    log.info(f"Outputs: {os.path.abspath(args.outdir)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()