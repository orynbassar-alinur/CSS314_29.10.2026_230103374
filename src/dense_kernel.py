import json
import time
import numpy as np
from numba import njit


@njit
def naive_multiply(A, B):
    N = A.shape[0]
    C = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]

    return C


@njit
def optimized_multiply(A, B):
    N = A.shape[0]
    C = np.zeros((N, N))

    for i in range(N):
        for k in range(N):
            for j in range(N):
                C[i, j] += A[i, k] * B[k, j]

    return C


with open("results/salted_params.json") as f:
    params = json.load(f)

N = params["N"] // 2
seed = params["seed"]

print(f"Matrix size: {N}x{N}")

rng = np.random.default_rng(seed)

A = rng.random((N, N))
B = rng.random((N, N))

naive_multiply(A[:2, :2], B[:2, :2])
optimized_multiply(A[:2, :2], B[:2, :2])

naive_times = []
optimized_times = []

C_naive = None
C_optimized = None

for run in range(3):

    start = time.perf_counter()
    C_naive = naive_multiply(A, B)
    naive_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    C_optimized = optimized_multiply(A, B)
    optimized_times.append(time.perf_counter() - start)

    print(
        f"Run {run + 1}: "
        f"Naive = {naive_times[-1]:.3f}s, "
        f"Optimized = {optimized_times[-1]:.3f}s"
    )

T_naive = float(np.median(naive_times))
T_optimized = float(np.median(optimized_times))

speedup = T_naive / T_optimized

max_error = float(np.max(np.abs(C_naive - C_optimized)))

results = {
    "matrix_size": N,

    "naive_time_s": T_naive,
    "interchanged_time_s": T_optimized,

    "naive_times": naive_times,
    "optimized_times": optimized_times,

    "median_naive_seconds": T_naive,
    "median_optimized_seconds": T_optimized,

    "speedup": speedup,
    "max_absolute_error": max_error,
    "correct": bool(np.allclose(
        C_naive,
        C_optimized,
        rtol=1e-12,
        atol=1e-12
    ))
}

with open("results/loop_interchange.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nRESULTS")
print("-" * 40)
print(f"Naive:     {T_naive:.4f} seconds")
print(f"Optimized: {T_optimized:.4f} seconds")
print(f"Speedup:   {speedup:.2f}x")
print(f"Max error: {max_error:.2e}")
print(f"Correct:   {results['correct']}")

print("\nSaved to results/loop_interchange.json")