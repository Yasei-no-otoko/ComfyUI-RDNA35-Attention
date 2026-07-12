from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import re
from typing import Any


RDNA35_TARGETS = {"gfx1150", "gfx1151", "gfx1152"}
RDNA3_TARGETS = {"gfx1100", "gfx1101", "gfx1102", "gfx1103"}


def _import_torch():
    try:
        return importlib.import_module("torch"), None
    except Exception as exc:  # pragma: no cover - depends on host install
        return None, f"{type(exc).__name__}: {exc}"


def is_rocm_pytorch() -> bool:
    torch, _ = _import_torch()
    if torch is None:
        return False
    return bool(getattr(getattr(torch, "version", None), "hip", None))


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_device_name(index: int = 0) -> str:
    torch, error = _import_torch()
    if torch is None:
        return f"torch unavailable: {error}"
    if not _cuda_available(torch):
        return "cuda/hip device unavailable"
    try:
        return str(torch.cuda.get_device_name(index))
    except Exception as exc:
        return f"device name unavailable: {type(exc).__name__}: {exc}"


def _extract_gfx(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"gfx[0-9a-zA-Z]+", str(value))
    if match:
        return match.group(0)
    return None


def best_effort_gfx_target(index: int = 0) -> str | None:
    for env_name in ("PYTORCH_ROCM_ARCH", "GPU_ARCHS", "AMDGPU_TARGETS", "ROCM_ARCH"):
        target = _extract_gfx(os.environ.get(env_name))
        if target:
            return target

    torch, _ = _import_torch()
    if torch is None or not _cuda_available(torch):
        return None

    try:
        props = torch.cuda.get_device_properties(index)
    except Exception:
        return None

    for attr in ("gcnArchName", "gfx_version", "name"):
        target = _extract_gfx(getattr(props, attr, None))
        if target:
            return target
    return None


def has_triton(import_module: bool = True) -> tuple[bool, str]:
    if importlib.util.find_spec("triton") is None:
        return False, "triton module not found"

    if not import_module:
        try:
            version = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            try:
                version = importlib.metadata.version("triton-windows")
            except importlib.metadata.PackageNotFoundError:
                version = "installed"
        return True, str(version)

    try:
        triton = importlib.import_module("triton")
    except Exception as exc:
        return False, f"triton import failed: {type(exc).__name__}: {exc}"
    return True, str(getattr(triton, "__version__", "unknown"))


def detect_runtime() -> dict[str, Any]:
    torch, torch_error = _import_torch()
    triton_ok, triton_info = has_triton(import_module=False)
    info: dict[str, Any] = {
        "torch_imported": torch is not None,
        "torch_error": torch_error,
        "torch_version": None,
        "torch_version_hip": None,
        "torch_version_cuda": None,
        "torch_cuda_is_available": False,
        "device": "unavailable",
        "gfx_target": None,
        "is_rocm_pytorch": False,
        "triton_available": triton_ok,
        "triton_info": triton_info,
        "is_rdna35": False,
    }

    if torch is None:
        return info

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    gfx_target = best_effort_gfx_target()

    info.update(
        {
            "torch_version": getattr(torch, "__version__", None),
            "torch_version_hip": hip_version,
            "torch_version_cuda": cuda_version,
            "torch_cuda_is_available": _cuda_available(torch),
            "device": get_device_name(),
            "gfx_target": gfx_target,
            "is_rocm_pytorch": bool(hip_version),
            "is_rdna35": gfx_target in RDNA35_TARGETS,
        }
    )
    return info


def explain_dispatch(info: dict[str, Any] | None = None) -> str:
    runtime = detect_runtime()
    lines = [
        "RDNA35 Fixed Block Attention dispatch diagnostics",
        f"torch: {runtime['torch_version']} imported={runtime['torch_imported']}",
        f"torch.version.hip: {runtime['torch_version_hip']}",
        f"torch.version.cuda: {runtime['torch_version_cuda']}",
        f"torch.cuda.is_available: {runtime['torch_cuda_is_available']}",
        f"device: {runtime['device']}",
        f"gfx target: {runtime['gfx_target']}",
        f"RDNA3.5 target: {runtime['is_rdna35']}",
        f"Triton: {runtime['triton_available']} ({runtime['triton_info']})",
    ]
    if info:
        lines.append("last dispatch:")
        for key in sorted(info):
            lines.append(f"  {key}: {info[key]}")
    return "\n".join(lines)
