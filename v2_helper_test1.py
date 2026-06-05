#!/usr/bin/env python3
import asyncio
import os
import re
import sys
import time
import argparse
from pathlib import Path

# Force line buffering for stdout
sys.stdout.reconfigure(line_buffering=True)

# Auto-detect the Hindi test dir relative to this script's location,
# so it works on any machine regardless of username/OS.
REPO_ROOT = Path(__file__).resolve().parent
HINDI_TEST_DIR = str(REPO_ROOT / "tests" / "nemo_text_processing" / "hi") + os.sep

# Use the current interpreter's pytest so the right conda env is picked up.
PYTEST = [sys.executable, "-m", "pytest"]


async def run_subprocess(cmd):
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=os.environ.copy(),
    )
    stdout, _ = await process.communicate()
    output = stdout.decode(errors="replace") if stdout else ""
    return process.returncode, output


async def collect_test_units():
    """Collect pytest nodeids and group into runnable units (classes or functions)."""
    candidate_cmds = [
        PYTEST + [HINDI_TEST_DIR, "--collect-only", "-q"],
        PYTEST + [HINDI_TEST_DIR, "--collect-only"],
    ]
    output = ""
    for cmd in candidate_cmds:
        returncode, out = await run_subprocess(cmd)
        if returncode == 0 and out.strip():
            output = out
            break
    if not output:
        return []

    units = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<") or line.startswith("collected "):
            continue
        parts = line.split("::")
        if len(parts) >= 3:
            unit = f"{parts[0]}::{parts[1]}"
        elif len(parts) == 2:
            unit = line
        else:
            continue
        units.add(unit)

    return sorted(units)


async def run_test_unit(nodeid, sem, basetemp_dir, timeout=None):
    """Run a class or function nodeid and return results."""
    async with sem:
        cmd = PYTEST + [
            nodeid,
            "--cpu",
            "--disable-warnings",
            "--tb=line",
            "-q",
            "--cache-clear",
            f"--basetemp={basetemp_dir}",
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        start_t = time.time()
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )

            try:
                if timeout:
                    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
                else:
                    stdout, _ = await process.communicate()
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                duration = time.time() - start_t
                return {
                    "unit": nodeid,
                    "returncode": -1,
                    "output": f"\n\n[X] Timeout after {timeout}s (killed)",
                    "duration": duration,
                }

            duration = time.time() - start_t
            output = stdout.decode(errors="replace") if stdout else ""
            return {
                "unit": nodeid,
                "returncode": process.returncode,
                "output": output,
                "duration": duration,
            }
        except Exception as e:
            return {
                "unit": nodeid,
                "returncode": -1,
                "output": str(e),
                "duration": time.time() - start_t,
            }


def parse_summary_line(output: str):
    for line in output.splitlines():
        s = line.strip()
        if ("passed" in s or "failed" in s or "skipped" in s) and s.startswith("===") and s.endswith("==="):
            return s.strip("=").strip()
    return None


def parse_test_counts(summary_line):
    """Parse counts from pytest summary line like '62 passed, 3 failed in 84.85s'."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    if not summary_line:
        return counts
    for key in counts:
        match = re.search(rf"(\d+)\s+{key}", summary_line)
        if match:
            counts[key] = int(match.group(1))
    return counts


async def run_all_tests(limit=None, timeout=None):
    """Collect tests and run classes/functions in parallel."""
    units = await collect_test_units()

    if not units:
        test_dir = Path(HINDI_TEST_DIR)
        units = [str(p) for p in sorted(test_dir.glob("test_*.py"))]
        if not units:
            print(f"[X] No tests found in {HINDI_TEST_DIR}")
            return 1
        print(f"[!] Falling back to per-file parallelization ({len(units)} files)")

    if limit:
        print(f"[!] Limiting to first {limit} units for verification.")
        units = units[:limit]

    max_parallel_env = os.environ.get("TEST_PARALLELISM")
    try:
        default_parallel = 2
        max_parallel = int(max_parallel_env) if max_parallel_env else default_parallel
    except ValueError:
        max_parallel = 4

    max_parallel = max(1, min(max_parallel, len(units)))

    print(f"[>] Running {len(units)} units in parallel (concurrency={max_parallel})\n")
    start_time = time.time()

    sem = asyncio.Semaphore(max_parallel)
    base_root = (REPO_ROOT / ".pytest_basetemp").absolute()
    base_root.mkdir(parents=True, exist_ok=True)

    def safe_name(s: str) -> str:
        return s.replace("/", "_").replace("\\", "_").replace(":", "_")

    unit_to_temp = {u: str(base_root / f"bt_{safe_name(u)}") for u in units}

    tasks = [run_test_unit(unit, sem, unit_to_temp[unit], timeout=timeout) for unit in units]

    print("\n" + "=" * 70)
    print("Test Results by Unit (Streaming)\n")

    completed_results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        completed_results.append(result)

        unit = result["unit"]
        output = result["output"]
        returncode = result["returncode"]
        duration = result["duration"]

        summary_line = parse_summary_line(output)
        status = "[OK]" if returncode == 0 else "[X]"

        line_info = f"{status} {unit} ({duration:.2f}s)"
        if summary_line:
            print(f"{line_info}\n    {summary_line}", flush=True)
        else:
            tail = "\n".join([l for l in output.splitlines()[-20:] if l.strip()])
            if returncode != 0 and tail:
                print(f"{line_info}\n    Last lines:\n    {tail}", flush=True)
            else:
                print(f"{line_info}", flush=True)

    elapsed_time = time.time() - start_time

    failed_units = []
    total_passed = total_failed = total_skipped = total_error = 0

    for result in completed_results:
        if result["returncode"] != 0:
            failed_units.append(result["unit"])
        counts = parse_test_counts(parse_summary_line(result["output"]))
        total_passed += counts["passed"]
        total_failed += counts["failed"]
        total_skipped += counts["skipped"]
        total_error += counts["error"]

    print("=" * 70)
    print(f"\nTotal time: {elapsed_time:.2f} seconds")

    print(f"\nGrand Total: {total_passed} passed, {total_failed} failed", end="")
    if total_skipped > 0:
        print(f", {total_skipped} skipped", end="")
    if total_error > 0:
        print(f", {total_error} error", end="")
    print()

    if failed_units:
        print(f"\n[X] {len(failed_units)} unit(s) failed:")
        for u in failed_units:
            print(f"   - {u}")
        return 1
    else:
        print("\n[OK] All tests passed!")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Pytest Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of units to run")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds per unit")
    args = parser.parse_args()

    exit_code = asyncio.run(run_all_tests(limit=args.limit, timeout=args.timeout))
    sys.exit(exit_code)
