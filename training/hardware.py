from __future__ import annotations

import argparse
import json
import subprocess

import torch


GPU_GUIDANCE = {
    "H100": "fastest, but overkill until the network and batched search are much larger",
    "RTX PRO 6000": "very strong and memory-rich; use for large experiments if Colab offers it",
    "G4": "very strong and memory-rich; use for large experiments if Colab offers it",
    "A100": "best serious-training choice for this repo once runs are hours long",
    "L4": "best default choice: fast enough, efficient, and usually easier to get",
    "T4": "good for free/cheap smoke runs and small training",
}


def detect() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "recommendation": "CPU is only recommended for debugging small smoke runs.",
    }

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        info["device"] = name
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        info["recommendation"] = recommendation_for_gpu(name)
        return info

    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        info["device"] = str(xm.xla_device())
        info["recommendation"] = (
            "TPU detected, but lc0mini is PyTorch/CUDA-oriented today. "
            "Use GPU unless you plan to port training to torch_xla/JAX and batch the search."
        )
    except Exception:
        pass
    return info


def recommendation_for_gpu(name: str) -> str:
    upper_name = name.upper()
    for key, guidance in GPU_GUIDANCE.items():
        if key.upper() in upper_name:
            return guidance
    return "Unknown GPU. If it has CUDA support, use it like L4 for now and benchmark."


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    info = detect()
    smi = nvidia_smi()
    if smi:
        info["nvidia_smi"] = smi
        if not info["cuda_available"]:
            info["recommendation"] = (
                "NVIDIA GPU is visible, but this PyTorch install cannot use CUDA. "
                "Use a CUDA-enabled Colab GPU runtime or install a CUDA PyTorch build."
            )

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for key, value in info.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
