import os
import sys
import json
import platform
import argparse
import subprocess
import psutil

from salt import get_student_params

def get_windows_cpu_info():

    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | "
            "Select-Object Name, Manufacturer, "
            "L2CacheSize, L3CacheSize, "
            "NumberOfCores, NumberOfLogicalProcessors | "
            "ConvertTo-Json"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)

        if isinstance(data, list):
            data = data[0]

        return data

    except Exception as e:
        print("CPU detection error:", e)
        return {}

def get_cache_info():

    cache_info = {
        "l1_cache_kb": None,
        "l2_cache_kb": None,
        "l3_cache_kb": None,
        "cache_line_size_bytes": None
    }

    windows_cpu = get_windows_cpu_info()

    cache_info["l2_cache_kb"] = windows_cpu.get(
        "L2CacheSize"
    )

    cache_info["l3_cache_kb"] = windows_cpu.get(
        "L3CacheSize"
    )

    try:

        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_CacheMemory | "
            "Select-Object Level, InstalledSize | "
            "ConvertTo-Json"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout.strip():

            cache_data = json.loads(result.stdout)

            if isinstance(cache_data, dict):
                cache_data = [cache_data]

            for cache in cache_data:

                level = cache.get("Level")
                size = cache.get("InstalledSize")

                if level == 3:
                    cache_info["l1_cache_kb"] = size

    except Exception as e:
        print("L1 cache detection error:", e)

    cache_info["cache_line_size_bytes"] = 64

    return cache_info

def get_cpu_info():

    windows_cpu = get_windows_cpu_info()

    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    cpu_info = {
        "processor": windows_cpu.get(
            "Name",
            platform.processor()
        ),

        "manufacturer": windows_cpu.get(
            "Manufacturer",
            "Unknown"
        ),

        "architecture": platform.machine(),

        "physical_cores": physical_cores,

        "logical_cores": logical_cores,

        "smt_enabled": (
            logical_cores > physical_cores
            if physical_cores and logical_cores
            else None
        ),

        "cpu_frequency_mhz": None,

        "microarchitecture": "Unknown"
    }

    frequency = psutil.cpu_freq()

    if frequency:
        cpu_info["cpu_frequency_mhz"] = frequency.max

    processor_name = cpu_info["processor"].lower()

    if "ryzen" in processor_name:

        if "7000" in processor_name:
            cpu_info["microarchitecture"] = "Zen 4"

        elif "5000" in processor_name:
            cpu_info["microarchitecture"] = "Zen 3"

        elif "3000" in processor_name:
            cpu_info["microarchitecture"] = "Zen 2"

    cpu_info.update(get_cache_info())

    return cpu_info

def get_system_info():

    system_info = {
        "os": platform.system(),

        "os_version": platform.version(),

        "kernel": platform.release(),

        "python_version": platform.python_version(),

        "python_implementation":
            platform.python_implementation(),

        "python_executable": sys.executable,

        "total_ram_gb": round(
            psutil.virtual_memory().total / (1024 ** 3),
            2
        )
    }

    return system_info

def save_json(filename, data):

    os.makedirs("results", exist_ok=True)

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

def main():

    parser = argparse.ArgumentParser(
        description="CS482 Milestone 0 - Host Profiling"
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Your Student ID"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("CS482 - PARALLEL COMPUTING")
    print("MILESTONE 0: HOST TOPOLOGY PROFILING")
    print("=" * 50)

    print("\n[1] Detecting CPU information...")

    cpu_info = get_cpu_info()

    print("[2] Detecting operating system...")

    system_info = get_system_info()

    print("[3] Generating salted parameters...")

    student_params = get_student_params(
        args.student_id
    )

    host_profile = {
        "student_id": args.student_id,

        "cpu_model": cpu_info["processor"],
        "physical_cores": cpu_info["physical_cores"],
        "logical_threads": cpu_info["logical_cores"],
        "cache_line_bytes": cpu_info["cache_line_size_bytes"],

        "cpu": cpu_info,
        "system": system_info
    }

    save_json(
        "results/host_profile.json",
        host_profile
    )

    save_json(
        "results/salted_params.json",
        student_params
    )

    print("\n" + "=" * 50)
    print("HOST PROFILE")
    print("=" * 50)

    print(
        json.dumps(
            host_profile,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\n" + "=" * 50)
    print("SALTED PARAMETERS")
    print("=" * 50)

    print(
        json.dumps(
            student_params,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\n" + "=" * 50)
    print("MILESTONE 0 COMPLETED!")
    print("=" * 50)

    print("\nGenerated files:")

    print("results/host_profile.json")
    print("results/salted_params.json")


if __name__ == "__main__":
    main()
