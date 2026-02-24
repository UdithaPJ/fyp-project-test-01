"""
=============================================================================
STEP 0: Environment Setup & Verification
=============================================================================
FYP: Mid-Range GPU-Accelerated Framework for Multi-Scale Biological Network Analysis
Milestone 1.3 – Verify GPU development environment

HOW TO RUN:
  python step0_verify_env.py

This script checks:
  1. Python version
  2. All required CPU packages
  3. CUDA availability and GPU specs
  4. CuPy (GPU array library) availability
  5. File system and output directories
  6. Memory availability (RAM + VRAM)
  7. A small end-to-end smoke test (synthetic 100-node graph)
=============================================================================
"""

import sys
import os
import platform
import subprocess
import time

REQUIRED_PACKAGES = {
    "numpy": "1.24",
    "scipy": "1.10",
    "pandas": "1.5",
    "pyarrow": "10.0",
    "networkx": "3.0",
    "tqdm": "4.0",
    "psutil": "5.0",
}

OPTIONAL_PACKAGES = {
    "cupy": "GPU acceleration (install: pip install cupy-cuda12x)",
    "community": "Louvain (install: pip install python-louvain)",
    "matplotlib": "Visualisation (install: pip install matplotlib)",
}

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [INFO]"


def header(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check_python():
    header("Python")
    v = sys.version_info
    print(f"{INFO} Python {v.major}.{v.minor}.{v.micro} on {platform.system()}")
    if v.major == 3 and v.minor >= 9:
        print(f"{PASS} Python >= 3.9")
        return True
    else:
        print(f"{FAIL} Python 3.9+ required")
        return False


def check_packages():
    header("Required Python Packages")
    all_ok = True
    for pkg, min_ver in REQUIRED_PACKAGES.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"{PASS} {pkg:20s}  version={ver}")
        except ImportError:
            print(f"{FAIL} {pkg:20s}  NOT INSTALLED")
            print(f"        → pip install {pkg}")
            all_ok = False
    return all_ok


def check_optional_packages():
    header("Optional Python Packages")
    for pkg, note in OPTIONAL_PACKAGES.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"{PASS} {pkg:20s}  version={ver}")
        except ImportError:
            print(f"{WARN} {pkg:20s}  not installed  ({note})")


def check_cuda():
    header("CUDA / GPU")
    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"{PASS} nvidia-smi found:")
            for line in result.stdout.strip().split("\n"):
                print(f"      {line}")
        else:
            print(f"{FAIL} nvidia-smi returned error: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print(f"{FAIL} nvidia-smi not found. NVIDIA driver may not be installed.")
        return False
    except subprocess.TimeoutExpired:
        print(f"{FAIL} nvidia-smi timed out.")
        return False

    # Check CUDA toolkit version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            cuda_ver = [l for l in result.stdout.split("\n") if "release" in l]
            print(f"{PASS} nvcc: {cuda_ver[0].strip() if cuda_ver else 'found'}")
        else:
            print(f"{WARN} nvcc not found (CUDA toolkit not in PATH)")
    except FileNotFoundError:
        print(f"{WARN} nvcc not found. Install CUDA Toolkit for custom kernels.")

    return True


def check_cupy():
    header("CuPy (GPU Array Library)")
    try:
        import cupy as cp
        print(f"{PASS} CuPy version: {cp.__version__}")
        device = cp.cuda.Device(0)
        free, total = cp.cuda.runtime.memGetInfo()
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"{INFO} GPU:  {props['name'].decode()}")
        print(f"{INFO} VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
        print(f"{INFO} Compute: {props['major']}.{props['minor']}")

        # Quick smoke test
        a = cp.random.rand(1000, 1000, dtype=cp.float32)
        b = cp.random.rand(1000, 1000, dtype=cp.float32)
        t0 = time.perf_counter()
        c = a @ b
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter() - t0
        print(f"{PASS} GPU matmul (1000×1000) in {t*1000:.1f}ms")
        return True
    except ImportError:
        print(f"{WARN} CuPy not installed.")
        print(f"       Install with: pip install cupy-cuda12x  (for CUDA 12.x)")
        print(f"                 or: pip install cupy-cuda11x  (for CUDA 11.x)")
        print(f"       Check CUDA version: nvcc --version")
        return False
    except Exception as e:
        print(f"{FAIL} CuPy error: {e}")
        return False


def check_memory():
    header("System Memory")
    import psutil
    vm = psutil.virtual_memory()
    print(f"{INFO} RAM total:     {vm.total/1e9:.1f} GB")
    print(f"{INFO} RAM available: {vm.available/1e9:.1f} GB")
    print(f"{INFO} RAM used:      {vm.percent:.0f}%")

    if vm.available > 8e9:
        print(f"{PASS} Sufficient RAM for full pipeline")
    elif vm.available > 4e9:
        print(f"{WARN} Moderate RAM. Use --max_genes <= 3000 in step1")
    else:
        print(f"{FAIL} Low RAM. Use --max_genes <= 1000 and close other apps")


def check_directories():
    header("File System")
    dirs_to_create = ["output", "output/cpu_results"]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"{PASS} Directory ready: {os.path.abspath(d)}")


def smoke_test():
    header("End-to-End Smoke Test (100-node synthetic graph)")
    import numpy as np
    from scipy import sparse

    # Create random sparse graph
    n = 100
    rng = np.random.default_rng(42)
    rows = rng.integers(0, n, size=300)
    cols = rng.integers(0, n, size=300)
    vals = rng.uniform(0.5, 1.0, size=300).astype(np.float32)
    adj = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    adj = adj + adj.T  # symmetrise

    from step3_cpu_baselines import bfs, pagerank, rwr, hits

    print("  Running BFS...", end="")
    (dist, _), t = bfs(adj, source=0)
    print(f"  {t*1000:.1f}ms  reachable={np.sum(dist >= 0)}/{n}")

    print("  Running PageRank...", end="")
    (pr, _), t = pagerank(adj)
    print(f"  {t*1000:.1f}ms  top_score={pr.max():.4f}")

    print("  Running RWR...", end="")
    (rwr_s, _), t = rwr(adj, seed_nodes=[0, 1])
    print(f"  {t*1000:.1f}ms  sum={rwr_s.sum():.4f}")

    print("  Running HITS...", end="")
    (hub, auth), t = hits(adj)
    print(f"  {t*1000:.1f}ms  max_hub={hub.max():.4f}")

    print(f"\n{PASS} Smoke test passed! All CPU baselines functional.")


def main():
    print("=" * 60)
    print("  FYP Environment Verification Script")
    print("  Mid-Range GPU Framework for Biological Network Analysis")
    print("=" * 60)

    results = {}
    results["python"]    = check_python()
    results["packages"]  = check_packages()
    check_optional_packages()
    results["cuda"]      = check_cuda()
    results["cupy"]      = check_cupy()
    check_memory()
    check_directories()

    try:
        smoke_test()
        results["smoke_test"] = True
    except Exception as e:
        print(f"{FAIL} Smoke test failed: {e}")
        results["smoke_test"] = False

    header("Summary")
    all_critical = all([results["python"], results["packages"]])
    for k, v in results.items():
        status = PASS if v else FAIL
        print(f"{status} {k}")

    print()
    if all_critical:
        print("  ✅  Environment ready for CPU baseline pipeline (Step 1 + Step 3)")
        if results["cupy"]:
            print("  ✅  GPU pipeline also ready (Step 2)")
        else:
            print("  ⚠️  GPU pipeline requires CuPy installation")
    else:
        print("  ❌  Fix failed checks before running the pipeline")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
