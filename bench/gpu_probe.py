#!/usr/bin/env python
"""Run bench/gpu_probe.mojo only where a CUDA device is actually present.

`DeviceContext()` does not raise on a host with no driver: on this toolchain it
aborts the process inside the driver bindings, so a `try:`/`except:` inside Mojo
cannot turn a missing device into a clean skip. This wrapper checks the driver
first and prints the skip line instead of launching the probe, which is what
keeps `bench/gpu_probe.mojo` safe to run on a CPU-only machine.

    pixi run python bench/gpu_probe.py

Exit code is 0 either way: "no device" is a skip, not a failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "bench", "gpu_probe.mojo")


def device_present() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def main() -> int:
    if not device_present():
        print("no CUDA device; probe skipped")
        return 0
    return subprocess.call(["mojo", "run", PROBE], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
