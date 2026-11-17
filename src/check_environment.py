"""Collect the Gate-0 environment inventory into JSON and stdout.

This script is read-only with respect to the environment: it does not install,
upgrade, or download anything.

Usage:
    python -m src.check_environment
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KEY_PACKAGES = (
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
    "PyYAML",
    "psutil",
)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def windows_memory() -> dict[str, Any] | None:
    if os.name != "nt":
        return None
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    mib = 1024 * 1024
    return {
        "total_physical_mib": round(status.ullTotalPhys / mib, 1),
        "available_physical_mib": round(status.ullAvailPhys / mib, 1),
        "total_pagefile_mib": round(status.ullTotalPageFile / mib, 1),
        "available_pagefile_mib": round(status.ullAvailPageFile / mib, 1),
        "memory_load_percent": status.dwMemoryLoad,
    }


def venv_uses_system_site_packages() -> bool | None:
    """Read the venv's pyvenv.cfg instead of guessing from sys.flags."""
    config_path = Path(sys.prefix) / "pyvenv.cfg"
    if not config_path.exists():
        return None
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().lower() == "include-system-site-packages":
                return value.strip().lower() == "true"
    except OSError:
        return None
    return None


def cpu_name() -> str | None:
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return str(value).strip()
        except Exception:  # noqa: BLE001 - best effort inventory
            return None
    return platform.processor() or None


def disk_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for letter in ("C:", "D:", "E:"):
        path = Path(letter + "\\")
        if not path.exists():
            continue
        usage = shutil.disk_usage(path)
        result[letter] = {
            "total_gib": round(usage.total / (1024**3), 2),
            "used_gib": round(usage.used / (1024**3), 2),
            "free_gib": round(usage.free / (1024**3), 2),
        }
    return result


def package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_inventory() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - report, do not hide
        return {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}

    info: dict[str, Any] = {
        "import_ok": True,
        "version": torch.__version__,
        "cuda_built_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        info["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(index)),
                "total_vram_mib": round(
                    torch.cuda.get_device_properties(index).total_memory / (1024**2), 1
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
        info["current_device"] = {
            "index": device,
            "name": props.name,
            "free_vram_mib_at_check": round(free_bytes / (1024**2), 1),
            "total_vram_mib": round(total_bytes / (1024**2), 1),
        }
    return info


def nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - best effort
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_inventory() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - best effort
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": completed.returncode == 0,
        "project_git_head": completed.stdout.strip() or None,
        "stderr": completed.stderr.strip() or None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifest" / "environment.json",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    info = {
        "schema_version": "gate0-environment-v1",
        "generated_at_utc": utc_now(),
        "system": {
            "platform": platform.platform(),
            "os_name": os.name,
            "windows_version": platform.version(),
            "machine": platform.machine(),
            "processor": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python_base_prefix": sys.base_prefix,
            "is_venv": sys.prefix != sys.base_prefix,
            "venv_uses_system_site_packages": venv_uses_system_site_packages(),
            "cwd": str(Path.cwd()),
            "project_root": str(PROJECT_ROOT),
        },
        "memory": windows_memory(),
        "disks": disk_inventory(),
        "cuda": nvidia_smi(),
        "torch": torch_inventory(),
        "packages": package_versions(),
        "environment_variables": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "PIP_INDEX_URL": os.environ.get("PIP_INDEX_URL"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "git": git_inventory(),
        "notes": [
            "Torch and several numerical packages are inherited from the base Python environment via a venv created with --system-site-packages.",
            "This was deliberate: the verified base already contains torch 2.11.0+cu128 for the RTX 5070 Laptop (sm_120); duplicating a CUDA torch install would consume several GB.",
            "Qwen weights were not downloaded or loaded for this inventory.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    system = info["system"]
    print(f"Generated : {info['generated_at_utc']}")
    print(f"Platform  : {system['platform']}")
    print(f"CPU       : {system['processor']} ({system['logical_cpus']} logical)")
    print(
        "Python    : "
        f"{system['python_version']} at {system['python_executable']} "
        f"(venv={system['is_venv']}, system_site={system['venv_uses_system_site_packages']})"
    )
    if info["memory"]:
        print(
            "RAM MiB   : "
            f"total={info['memory']['total_physical_mib']}, "
            f"available={info['memory']['available_physical_mib']}, "
            f"load={info['memory']['memory_load_percent']}%"
        )
    for letter, usage in info["disks"].items():
        print(
            f"Disk {letter}  : total={usage['total_gib']} GiB, "
            f"free={usage['free_gib']} GiB"
        )
    if info["cuda"]["ok"]:
        print(f"nvidia-smi: {info['cuda']['stdout']}")
    else:
        print(f"nvidia-smi: FAILED ({info['cuda'].get('error') or info['cuda'].get('stderr')})")
    if info["torch"].get("import_ok"):
        print(
            "Torch     : "
            f"{info['torch']['version']}, CUDA available={info['torch']['cuda_available']}, "
            f"built CUDA={info['torch']['cuda_built_version']}"
        )
    else:
        print(f"Torch     : FAILED ({info['torch'].get('error')})")
    print("Packages  : " + ", ".join(f"{k}={v}" for k, v in info["packages"].items()))
    print(f"Wrote {args.out}")
    print("ENVIRONMENT_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
