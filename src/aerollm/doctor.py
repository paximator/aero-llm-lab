"""Report whether the local AeroLLM environment satisfies a requested workflow."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformers",
        action="store_true",
        help="also require the pinned NVIDIA Transformers/CUDA environment",
    )
    args = parser.parse_args(argv)

    print(f"python={platform.python_version()} platform={platform.platform()}")
    print(f"aerollm-lab={_version('aerollm-lab')} uv-lock=present")
    if not args.transformers:
        print("status=ok profile=core")
        return 0

    failures: list[str] = []
    try:
        import torch
    except ImportError:
        failures.append("Torch is missing; run: uv sync --extra transformers")
    else:
        print(f"torch={torch.__version__} torch-cuda={torch.version.cuda}")
        if torch.__version__.split("+")[0] != "2.7.1":
            failures.append("Torch must resolve to 2.7.1 from the uv CUDA index")
        if not torch.cuda.is_available():
            failures.append("CUDA is unavailable; verify the NVIDIA driver and uv environment")
        else:
            print(f"gpu={torch.cuda.get_device_name(0)}")
        if not hasattr(torch, "float8_e8m0fnu"):
            failures.append("Torch lacks UE8M0 FP8 support required by Ministral 3")

    for package in ("transformers", "mistral-common", "kernels"):
        version = _version(package)
        print(f"{package}={version}")
        if version == "missing":
            failures.append(f"{package} is missing; run: uv sync --extra transformers")
    if sys.platform == "win32":
        triton_version = _version("triton-windows")
        print(f"triton-windows={triton_version}")
        if triton_version != "3.7.1.post27":
            failures.append("Windows requires triton-windows==3.7.1.post27")

    if failures:
        for failure in failures:
            print(f"error={failure}", file=sys.stderr)
        return 1
    print("status=ok profile=transformers-cuda")
    return 0


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


if __name__ == "__main__":
    raise SystemExit(main())
