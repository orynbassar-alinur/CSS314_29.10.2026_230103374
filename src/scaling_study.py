import json
import time
import csv
import os
import math
import statistics
import multiprocessing as mp

import numpy as np
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor


PROCESS_COUNTS = [1, 2, 4, 8]
RUNS = 3


def stencil_worker(args):
    input_name, output_name, shape, start, end = args

    input_shm = shared_memory.SharedMemory(name=input_name)
    output_shm = shared_memory.SharedMemory(name=output_name)

    try:
        A = np.ndarray(shape, dtype=np.float64, buffer=input_shm.buf)
        B = np.ndarray(shape, dtype=np.float64, buffer=output_shm.buf)

        N = shape[0]

        # 5-point stencil
        B[start:end, 1:-1] = (
            A[start - 1:end - 1, 1:-1]
            + A[start + 1:end + 1, 1:-1]
            + A[start:end, :-2]
            + A[start:end, 2:]
            + A[start:end, 1:-1]
        ) / 5.0

        return float(np.sum(B[start:end, 1:-1]))

    finally:
        input_shm.close()
        output_shm.close()


def run_stencil(N, P, seed):

    shape = (N, N)
    size = N * N * 8

    input_shm = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    output_shm = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    try:
        A = np.ndarray(
            shape,
            dtype=np.float64,
            buffer=input_shm.buf
        )

        B = np.ndarray(
            shape,
            dtype=np.float64,
            buffer=output_shm.buf
        )

        rng = np.random.default_rng(seed)

        A[:] = rng.random(shape)
        B.fill(0)

        rows = N - 2

        chunk = math.ceil(rows / P)

        tasks = []

        for p in range(P):

            start = 1 + p * chunk
            end = min(start + chunk, N - 1)

            if start >= end:
                continue

            tasks.append((
                input_shm.name,
                output_shm.name,
                shape,
                start,
                end
            ))

        start_time = time.perf_counter()

        with ProcessPoolExecutor(max_workers=P) as executor:
            checksums = list(
                executor.map(stencil_worker, tasks)
            )

        elapsed = time.perf_counter() - start_time

        total_checksum = sum(checksums)

        reference = (
            A[:-2, 1:-1]
            + A[2:, 1:-1]
            + A[1:-1, :-2]
            + A[1:-1, 2:]
            + A[1:-1, 1:-1]
        ) / 5.0

        correct = np.allclose(
            B[1:-1, 1:-1],
            reference,
            rtol=1e-12,
            atol=1e-12
        )

        if not correct:
            raise ValueError("Stencil verification failed!")

        return elapsed, total_checksum

    finally:
        input_shm.close()
        output_shm.close()

        input_shm.unlink()
        output_shm.unlink()


def main():

    os.makedirs("results", exist_ok=True)

    with open("results/salted_params.json") as f:
        params = json.load(f)

    N = params["N"]
    seed = params["seed"]

    results = []

    print("=" * 55)
    print("CS482 - SCALING STUDY")
    print("=" * 55)

    print("\nSTRONG SCALING")
    print("-" * 55)

    strong_times = {}

    for P in PROCESS_COUNTS:

        times = []

        for run in range(RUNS):

            elapsed, checksum = run_stencil(N, P, seed)
            times.append(elapsed)

        median_time = statistics.median(times)
        strong_times[P] = median_time

        print(
            f"P={P}: {median_time:.4f}s "
            f"Checksum={checksum:.6f}"
        )

    T1 = strong_times[1]

    for P in PROCESS_COUNTS:

        TP = strong_times[P]

        speedup = T1 / TP
        efficiency = speedup / P * 100

        if P == 1:
            serial_fraction = None
        else:
            serial_fraction = (
                (1 / speedup) - (1 / P)
            ) / (1 - (1 / P))

        results.append({
            "scaling": "strong",
            "processes": P,
            "matrix_size": N,
            "runtime_seconds": TP,
            "speedup": speedup,
            "efficiency_percent": efficiency,
            "serial_fraction": serial_fraction
        })

        print(
            f"P={P}: Speedup={speedup:.3f}x, "
            f"Efficiency={efficiency:.2f}%"
        )

    print("\nWEAK SCALING")
    print("-" * 55)

    weak_times = {}

    for P in PROCESS_COUNTS:

        scaled_N = math.ceil(N * math.sqrt(P))

        times = []

        for run in range(RUNS):

            elapsed, checksum = run_stencil(
                scaled_N,
                P,
                seed
            )

            times.append(elapsed)

        median_time = statistics.median(times)
        weak_times[P] = median_time

        print(
            f"P={P}: N={scaled_N}, "
            f"Time={median_time:.4f}s"
        )

    weak_T1 = weak_times[1]

    for P in PROCESS_COUNTS:

        TP = weak_times[P]

        scaled_speedup = (
            P * weak_T1 / TP
        )

        results.append({
            "scaling": "weak",
            "processes": P,
            "matrix_size": math.ceil(N * math.sqrt(P)),
            "runtime_seconds": TP,
            "speedup": scaled_speedup,
            "efficiency_percent": scaled_speedup / P * 100,
            "serial_fraction": None
        })

        print(
            f"P={P}: Scaled Speedup={scaled_speedup:.3f}x"
        )

    with open(
        "results/scaling_metrics.tsv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scaling",
                "processes",
                "matrix_size",
                "runtime_seconds",
                "speedup",
                "efficiency_percent",
                "serial_fraction"
            ],
            delimiter="\t"
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nROOFLINE ANALYSIS")
    print("-" * 55)

    flops_per_element = 5

    bytes_per_element = 6 * 8

    arithmetic_intensity = (
        flops_per_element / bytes_per_element
    )

    roofline = {
        "kernel": "2D 5-point stencil",
        "matrix_size": N,
        "flops_per_element": flops_per_element,
        "bytes_per_element": bytes_per_element,
        "arithmetic_intensity": arithmetic_intensity,
        "arithmetic_intensity_unit": "FLOP/byte",
        "classification": "Likely Memory-Bandwidth Bound",
        "note": (
            "Low arithmetic intensity suggests "
            "memory bandwidth limitation. "
            "Actual classification requires "
            "measured or documented hardware "
            "bandwidth and peak FLOPS."
        )
    }

    with open(
        "results/roofline_summary.json",
        "w"
    ) as f:

        json.dump(roofline, f, indent=4)

    print(f"Arithmetic Intensity: {arithmetic_intensity:.4f}")
    print(f"Classification: {roofline['classification']}")

    print("\nGenerated files:")
    print("results/scaling_metrics.tsv")
    print("results/roofline_summary.json")

    print("\nMILESTONE 3 COMPLETED!")


if __name__ == "__main__":
    mp.freeze_support()
    main()