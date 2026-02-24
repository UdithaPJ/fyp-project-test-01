# FYP Pipeline — Complete Test Guide

## Mid-Range GPU-Accelerated Framework for Multi-Scale Biological Network Analysis

### GTEx 21 GB Parquet | 19,788 Samples | GTX 1650 4.3 GB VRAM | 8 CPU Cores

---

## File Overview

All five scripts live in the same folder (e.g. `D:\FYP\test\test-01\`):

| Script                     | Purpose                                         | Milestone       |
| -------------------------- | ----------------------------------------------- | --------------- |
| `step0_verify_env.py`      | Check Python packages and GPU                   | Pre-run         |
| `step1_preprocess_gtex.py` | Stream-parse parquet to co-expression GRN       | 1.2             |
| `step2_gpu_preprocess.py`  | GPU-accelerated correlation (optional speedup)  | 1.2             |
| `step3_cpu_baselines.py`   | Single-thread algorithm baselines               | 1.3             |
| `step4_parallel_cpu.py`    | Parallel algorithm implementations + benchmark  | 2.1 / 3.2       |
| `step5_gpu_algorithms.py`  | GPU algorithm implementations + 3-way benchmark | 2.1 / 2.2 / 2.3 |

Expected output folder structure after a full run:

```
output\
  lib_sizes.npy
  samples.npy
  genes_filtered.npy
  expr_mmap.npy
  edge_list.csv
  adj_csr.npz
  adj_csr_gpu.npz       (step2 only)
  graph_stats.txt
  gpu_stats.txt
  preprocess.log
  gpu_preprocess.log
  parallel_cpu.log
  cpu_results\
    bfs_distances.csv
    pagerank_scores.csv
    rwr_scores.csv
    hits_hub_scores.csv
    hits_authority_scores.csv
    mce_scores.csv
    louvain_communities.csv
    mcl_clusters.csv
    cpu_timing.csv          <- baseline for step4 comparison
  parallel_results\
    par_bfs_distances.csv
    par_pagerank_scores.csv
    par_rwr_scores.csv
    par_hits_hub_scores.csv
    par_hits_authority_scores.csv
    par_mce_scores.csv
    par_louvain_communities.csv
    par_mcl_clusters.csv
    parallel_timing.csv
    benchmark_comparison.csv
    benchmark_report.txt
  gpu_results\
    gpu_bfs_distances.csv
    gpu_pagerank_scores.csv
    gpu_mce_scores.csv
    gpu_rwr_scores.csv
    gpu_hits_hub_scores.csv
    gpu_hits_authority_scores.csv
    gpu_louvain_communities.csv
    gpu_mcl_clusters.csv
    gpu_motif_scores.csv
    gpu_gnn_embeddings.npy
    gpu_gnn_embeddings.csv
    gpu_timing.csv
    three_way_comparison.csv
    three_way_report.txt     <- single vs parallel vs GPU comparison
```

---

## Prerequisites

### 1. Python packages

```powershell
pip install numpy scipy pandas pyarrow tqdm psutil networkx python-louvain
```

### 2. GPU setup (required for step2 only)

Check your CUDA version:

```powershell
nvidia-smi
```

Look at the top-right corner: `CUDA Version: XX.X`

Install the matching CuPy:

```powershell
# CUDA 12.x (requires driver >= 527.41, recommended after driver update):
pip install cupy-cuda12x

# CUDA 11.x (older driver):
pip install cupy-cuda11x
```

Add the CUDA toolkit bin folder to PATH (replace version number as needed):

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:PATH = "$env:CUDA_PATH\bin\x64;" + $env:PATH
```

To make PATH permanent across all future sessions:

```powershell
[System.Environment]::SetEnvironmentVariable(
    "PATH",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64;" +
    [System.Environment]::GetEnvironmentVariable("PATH", "User"),
    "User"
)
```

---

## Step 0 - Verify Environment

Run this before anything else. It checks all packages, tests each algorithm
on a 100-node synthetic graph, and detects your GPU.

```powershell
python step0_verify_env.py
```

Expected: all checks pass, GPU detected, smoke test completes without error.
Fix any missing packages before proceeding.

---

## Step 1 - Preprocess GTEx Parquet to Co-Expression Graph

Streams the 21 GB parquet file in three passes without loading it into RAM.
Peak RAM usage is ~500 MB regardless of file size.

### Quick test first (2,000 genes, ~15 minutes)

```powershell
python step1_preprocess_gtex.py --input "D:\FYP\test\test-02\data\GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_reads.parquet" --max_genes 2000 --workers 4 --outdir output_test
```

Verify the test outputs before running the full dataset:

```powershell
dir output_test
type output_test\graph_stats.txt
```

### Full run (10,000 genes, ~40-60 minutes on 8 cores)

```powershell
python step1_preprocess_gtex.py --input "D:\FYP\test\test-02\data\GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_reads.parquet" --max_genes 10000 --workers 8 --corr_threshold 0.7 --outdir output
```

### Resume after a crash

Step 1 saves each pass before starting the next. If it crashes:

```powershell
# If pass1 + pass2 already completed (expr_mmap.npy exists), skip to correlation:
python step1_preprocess_gtex.py --input "D:\FYP\test\test-02\data\GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_reads.parquet" --max_genes 10000 --workers 8 --skip_pass1 --skip_pass2 --outdir output
```

### Parameter reference

| Parameter            | Default | Notes                                          |
| -------------------- | ------- | ---------------------------------------------- |
| `--max_genes`        | 10000   | Genes to process. 10k gives ~800 MB expr_mmap. |
| `--workers`          | 8       | CPU cores for correlation.                     |
| `--corr_threshold`   | 0.7     | Pearson r cutoff. Lower = more edges.          |
| `--cpm_threshold`    | 1.0     | Min CPM to keep a gene.                        |
| `--min_samples_frac` | 0.2     | Gene must pass CPM filter in 20%+ samples.     |
| `--chunk_size`       | 300     | Genes per worker per correlation chunk.        |

### Disk space needed for expr_mmap.npy

| --max_genes | mmap size | z_mmap temp | Total disk |
| ----------- | --------- | ----------- | ---------- |
| 2,000       | ~160 MB   | ~160 MB     | ~400 MB    |
| 5,000       | ~400 MB   | ~400 MB     | ~1 GB      |
| 10,000      | ~800 MB   | ~800 MB     | ~2 GB      |
| 50,000      | ~4 GB     | ~4 GB       | ~10 GB     |

---

## Step 2 - GPU Correlation (Optional)

Run after step1 completes Pass 1 and Pass 2 (expr_mmap.npy exists).
Replaces the CPU correlation step with GPU acceleration.

For 10,000 genes on GTX 1650: ~8 minutes vs ~40 minutes on CPU.

```powershell
python step2_gpu_preprocess.py --expr_mmap output\expr_mmap.npy --gene_ids output\genes_filtered.npy --outdir output
```

Quick test (first 2,000 genes only):

```powershell
python step2_gpu_preprocess.py --expr_mmap output\expr_mmap.npy --gene_ids output\genes_filtered.npy --n_genes 2000 --outdir output
```

Output: `output\adj_csr_gpu.npz`

To use the GPU-built graph in step3/step4, replace `--adj output\adj_csr.npz`
with `--adj output\adj_csr_gpu.npz` in the commands below.

---

## Step 3 - Single-Thread CPU Baselines

Runs all 7 algorithms in single-thread mode and produces `cpu_timing.csv`.
**Must run before step4** so that step4 has baseline timings to compare against.

### Full run (all algorithms)

```powershell
python step3_cpu_baselines.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --outdir output\cpu_results
```

### Run individual algorithms

```powershell
python step3_cpu_baselines.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo bfs      --outdir output\cpu_results
python step3_cpu_baselines.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo pagerank --outdir output\cpu_results
python step3_cpu_baselines.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo rwr      --outdir output\cpu_results
python step3_cpu_baselines.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo hits     --outdir output\cpu_results
python step3_cpu_baselines.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo louvain  --outdir output\cpu_results
python step3_cpu_baselines.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo mcl      --outdir output\cpu_results
python step3_cpu_baselines.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --algo mce --outdir output\cpu_results
```

---

## Step 4 - Parallel CPU Implementations + Benchmark

Runs parallel versions of all 7 algorithms. Automatically reads
`cpu_timing.csv` from step3 and prints a speedup comparison table.

### Full benchmark run (all algorithms, 8 workers)

```powershell
python step4_parallel_cpu.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --single_timing output\cpu_results\cpu_timing.csv --workers 8 --outdir output\parallel_results
```

### Quick test (2 workers, verify it runs without errors)

```powershell
python step4_parallel_cpu.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --workers 2 --outdir output\parallel_results
```

### Run individual parallel algorithms

```powershell
python step4_parallel_cpu.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo bfs      --workers 8 --outdir output\parallel_results
python step4_parallel_cpu.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo pagerank --workers 8 --outdir output\parallel_results
python step4_parallel_cpu.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo rwr      --workers 8 --outdir output\parallel_results
python step4_parallel_cpu.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo hits     --workers 8 --outdir output\parallel_results
python step4_parallel_cpu.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo louvain  --workers 8 --outdir output\parallel_results
python step4_parallel_cpu.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo mcl      --workers 8 --outdir output\parallel_results
python step4_parallel_cpu.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --algo mce --workers 8 --outdir output\parallel_results
```

### Read the benchmark report

```powershell
type output\parallel_results\benchmark_report.txt
```

### Expected benchmark report format

```
====================================================================
FYP Benchmark Report: Single-Thread vs Parallel CPU
Graph : 3,997 nodes  |  46,844 edges
Workers: 8  |  CPUs available: 8
====================================================================

Algorithm          Single(s)  Parallel(s)   Speedup  Efficiency
--------------------------------------------------------------------
bfs                   0.0160       0.0080     2.00x       25.0%
hits                  1.2300       0.2100     5.86x       73.2%
louvain              12.4500       2.1800     5.71x       71.4%
mce                  45.3200       6.8900     6.58x       82.2%
mcl                  18.7600       3.4200     5.49x       68.6%
pagerank              0.3400       0.1100     3.09x       38.6%
rwr                   1.8200       0.3100     5.87x       73.4%
--------------------------------------------------------------------

Total single-thread time  : 80.00s
Total parallel time       : 13.21s
Overall speedup           : 6.06x
Average per-algo speedup  : 4.94x
Average parallel efficiency: 61.8%
Theoretical max speedup   : 8x
```

BFS and PageRank show lower speedup because the graph is small (4k nodes)
and process/thread spawn overhead dominates a 16ms workload. Speedup for
these algorithms becomes meaningful at 50k+ nodes. MCE shows the highest
speedup because each regulator's entropy computation is large and fully
independent.

---

## Step 5 - GPU Algorithm Implementations + Three-Way Benchmark

Covers Milestones 2.1, 2.2, and 2.3. Runs GPU-accelerated versions of all 9
algorithms and compares them against single-thread (step3) and parallel CPU
(step4) timings in a single three-way report.

**Must run steps 3 and 4 before step 5** so that `cpu_timing.csv` and
`parallel_timing.csv` exist for the comparison.

### Full run (all 9 algorithms, all milestones)

```powershell
python step5_gpu_algorithms.py `
    --adj              output\adj_csr.npz `
    --expr             output\expr_mmap.npy `
    --genes            output\genes_filtered.npy `
    --single_timing    output\cpu_results\cpu_timing.csv `
    --parallel_timing  output\parallel_results\parallel_timing.csv `
    --outdir           output\gpu_results
```

### Run by milestone

```powershell
# Milestone 2.1: BFS, PageRank, MCE
python step5_gpu_algorithms.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --milestone 2.1 --outdir output\gpu_results

# Milestone 2.2: RWR, HITS, Louvain
python step5_gpu_algorithms.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --milestone 2.2 --outdir output\gpu_results

# Milestone 2.3: MCL, Motif Discovery, GNN
python step5_gpu_algorithms.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --milestone 2.3 --outdir output\gpu_results
```

### Run individual GPU algorithms

```powershell
python step5_gpu_algorithms.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo bfs      --outdir output\gpu_results
python step5_gpu_algorithms.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo pagerank --outdir output\gpu_results
python step5_gpu_algorithms.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo rwr      --outdir output\gpu_results
python step5_gpu_algorithms.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo hits     --outdir output\gpu_results
python step5_gpu_algorithms.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo louvain  --outdir output\gpu_results
python step5_gpu_algorithms.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo mcl      --outdir output\gpu_results
python step5_gpu_algorithms.py --adj output\adj_csr.npz --genes output\genes_filtered.npy --algo motif    --outdir output\gpu_results
python step5_gpu_algorithms.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --algo mce --outdir output\gpu_results
python step5_gpu_algorithms.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --algo gnn --outdir output\gpu_results
```

### Read the three-way report

```powershell
type output\gpu_results\three_way_report.txt
```

### Expected three-way report format

```
========================================================================
FYP Three-Way Benchmark: Single-Thread vs Parallel CPU vs GPU
Graph : 3,997 nodes  |  46,844 edges
========================================================================

Algorithm            Single(s)  Parallel(s)    GPU(s)   vs 1T   vs Par
------------------------------------------------------------------------
bfs                     0.0160       0.0080    0.0120   1.33x    0.67x
gnn                        N/A          N/A    1.2400     N/A      N/A
hits                    1.2300       0.2100    0.0480  25.63x    4.38x
louvain                12.4500       2.1800    2.8200   4.41x    0.77x
mce                    45.3200       6.8900    0.8100  55.95x    8.51x
mcl                    18.7600       3.4200    0.3900  48.10x    8.77x
motif                      N/A          N/A    0.0620     N/A      N/A
pagerank                0.3400       0.1100    0.0210  16.19x    5.24x
rwr                     1.8200       0.3100    0.0350  52.00x    8.86x
------------------------------------------------------------------------

Average GPU vs Single-Thread speedup  : 33.77x
Average GPU vs Parallel CPU speedup   : 6.07x

Total single-thread time  : 80.00s
Total GPU time            : 5.45s
Overall GPU speedup       : 14.68x
```

Notes on results:

- BFS shows lower GPU speedup (sometimes < 1x on small 4k-node graphs) because
  the overhead of launching GPU kernels dominates a 16ms serial workload.
  On 100k+ node graphs, GPU BFS shows 10-50x speedup.
- MCE shows the highest GPU speedup because the vectorised histogram approach
  eliminates all Python loops — every target gene is processed simultaneously.
- Louvain GPU may be slower than parallel CPU because it falls back to CPU
  Louvain with GPU-computed initialisation (cuGraph not available on Windows).

---

## GPU Implementation Details (Milestone 2.1 / 2.2 / 2.3)

| Algorithm | Milestone | GPU Method                            | Key GPU Operations                        |
| --------- | --------- | ------------------------------------- | ----------------------------------------- |
| BFS       | 2.1       | Level-synchronous frontier expansion  | cuSPARSE SpMV per level                   |
| PageRank  | 2.1       | Full power iteration on GPU           | cuSPARSE SpMV (T.T @ pr)                  |
| MCE       | 2.1       | Vectorised histogram over all targets | CuPy elementwise + batched ops            |
| RWR       | 2.2       | Batched multi-seed SpMM               | cuSPARSE SpMM (W @ P, all seeds)          |
| HITS      | 2.2       | Both authority + hub SpMV on GPU      | cuSPARSE SpMV x2 per iteration            |
| Louvain   | 2.2       | GPU degree init + CPU Louvain         | cuGraph on Linux; CPU fallback on Windows |
| MCL       | 2.3       | Full dense MCL on GPU, zero transfers | cuBLAS GEMM + CuPy elementwise            |
| Motif     | 2.3       | Triangle counting via SpGEMM          | cuSPARSE SpGEMM (A @ A)                   |
| GNN       | 2.3       | 2-layer GCN on GPU                    | cuSPARSE SpMM + cuBLAS GEMM               |

---

## Complete Sequence (all steps, copy-paste ready)

```powershell
# 0. Verify environment
python step0_verify_env.py

# 1a. Quick test
python step1_preprocess_gtex.py --input "D:\FYP\test\test-02\data\GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_reads.parquet" --max_genes 2000 --workers 4 --outdir output_test

# 1b. Full preprocessing
python step1_preprocess_gtex.py --input "D:\FYP\test\test-02\data\GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_reads.parquet" --max_genes 10000 --workers 8 --outdir output

# 2. GPU correlation (optional)
python step2_gpu_preprocess.py --expr_mmap output\expr_mmap.npy --gene_ids output\genes_filtered.npy --outdir output

# 3. Single-thread baselines (required before steps 4 and 5)
python step3_cpu_baselines.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --outdir output\cpu_results

# 4. Parallel CPU (required before step 5 for full three-way report)
python step4_parallel_cpu.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --single_timing output\cpu_results\cpu_timing.csv --workers 8 --outdir output\parallel_results

# 5. GPU implementations + three-way benchmark
python step5_gpu_algorithms.py --adj output\adj_csr.npz --expr output\expr_mmap.npy --genes output\genes_filtered.npy --single_timing output\cpu_results\cpu_timing.csv --parallel_timing output\parallel_results\parallel_timing.csv --outdir output\gpu_results

# 6. Read the final three-way report
type output\gpu_results\three_way_report.txt
```
