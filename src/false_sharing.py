import multiprocessing as mp
import ctypes
import time
import csv
import os
import statistics

ITERATIONS = 5_000_000
PROCESS_COUNTS = [1, 2, 4, 8]
RUNS = 3


def worker_unpadded(array, worker_id, iterations):
    for _ in range(iterations):
        array[worker_id] += 1


def worker_padded(array, worker_id, iterations):
    index = worker_id * 8

    for _ in range(iterations):
        array[index] += 1


def worker_privatized(array, worker_id, iterations):
    local_counter = 0

    for _ in range(iterations):
        local_counter += 1

    array[worker_id * 8] = local_counter


def run_benchmark(mode, processes):

    if mode == "unpadded":
        size = processes
    else:
        size = processes * 8

    shared_array = mp.RawArray(ctypes.c_int64, size)

    workers = []

    start = time.perf_counter()

    for worker_id in range(processes):

        if mode == "unpadded":
            target = worker_unpadded

        elif mode == "padded":
            target = worker_padded

        else:
            target = worker_privatized

        process = mp.Process(
            target=target,
            args=(shared_array, worker_id, ITERATIONS)
        )

        workers.append(process)
        process.start()

    for process in workers:
        process.join()

        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker failed with exit code {process.exitcode}"
            )

    elapsed = time.perf_counter() - start

    if mode == "unpadded":
        total = sum(shared_array[i] for i in range(processes))
    else:
        total = sum(shared_array[i * 8] for i in range(processes))

    expected = processes * ITERATIONS

    if total != expected:
        raise ValueError(
            f"Incorrect result: {total} != {expected}"
        )

    return elapsed


def main():

    os.makedirs("results", exist_ok=True)

    results = []

    print("CS482 - False Sharing Benchmark")
    print("=" * 55)

    for mode in ["unpadded", "padded", "privatized"]:

        print(f"\nMode: {mode.upper()}")

        for processes in PROCESS_COUNTS:

            times = []

            for run in range(RUNS):

                elapsed = run_benchmark(mode, processes)
                times.append(elapsed)

            median_time = statistics.median(times)

            results.append({
                "mode": mode,
                "processes": processes,
                "runtime_seconds": median_time
            })

            print(
                f"P={processes}: "
                f"{median_time:.4f} seconds"
            )

    # Calculate speedup and Karp-Flatt
    baseline = {}

    for result in results:
        if result["processes"] == 1:
            baseline[result["mode"]] = result["runtime_seconds"]

    for result in results:

        P = result["processes"]
        mode = result["mode"]
        runtime = result["runtime_seconds"]

        if P == 1:
            result["speedup"] = 1.0
            result["karp_flatt"] = 0.0
            continue

        speedup = baseline[mode] / runtime

        e = (
            (1 / speedup) - (1 / P)
        ) / (
            1 - (1 / P)
        )

        result["speedup"] = speedup
        result["karp_flatt"] = e

    # Format results for official verifier
    formatted_results = []

    for P in PROCESS_COUNTS:

        row = {"p": P}

        for result in results:

            if result["processes"] == P:

                mode = result["mode"]

                row[f"time_{mode}"] = result["runtime_seconds"]
                row[f"speedup_{mode}"] = result["speedup"]
                row[f"e_{mode}"] = result["karp_flatt"]

        formatted_results.append(row)

    # Save TSV
    filename = "results/false_sharing_karp_flatt.tsv"

    with open(filename, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "p",

                "time_unpadded",
                "time_padded",
                "time_privatized",

                "speedup_unpadded",
                "speedup_padded",
                "speedup_privatized",

                "e_unpadded",
                "e_padded",
                "e_privatized"
            ],
            delimiter="\t"
        )

        writer.writeheader()
        writer.writerows(formatted_results)

    print("\nResults saved to:", filename)


if __name__ == "__main__":
    mp.freeze_support()
    main()