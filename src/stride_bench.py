import time
import csv
import os
import statistics
import numpy as np

SIZE = 16_777_216

data = np.ones(SIZE, dtype=np.float64)

strides = [1, 2, 4, 8, 16, 32, 64, 128]

results = []

print("CS482 - Memory Stride Benchmark")
print("-" * 50)

for stride in strides:

    times = []
    total = 0

    for _ in range(5):
        total = np.sum(data[::stride])

    for _ in range(5):

        start = time.perf_counter_ns()

        total = np.sum(data[::stride])

        end = time.perf_counter_ns()

        elapsed_ns = end - start

        elements_touched = len(data[::stride])

        latency = elapsed_ns / elements_touched

        times.append(latency)

    median_latency = statistics.median(times)

    results.append([stride, median_latency])

    print(
        f"Stride {stride:3d}: "
        f"{median_latency:.3f} ns/element"
    )

os.makedirs("results", exist_ok=True)

with open(
    "results/stride_results.tsv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file, delimiter="\t")

    writer.writerow(["stride", "latency_ns"])

    writer.writerows(results)

print("\nResults saved to results/stride_results.tsv")