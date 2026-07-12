from __future__ import annotations

import os
import pathlib
import subprocess

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = pathlib.Path(__file__).resolve().parent
EXPECTED_CK_COMMIT = "4975bd0c8e17a54bdc27c746527a385e7383bb07"


def find_ck_dir() -> pathlib.Path:
    candidates = []
    if value := os.environ.get("CK_DIR"):
        candidates.append(pathlib.Path(value))
    candidates.append(pathlib.Path.home() / "composable_kernel")

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "include" / "ck_tile" / "core.hpp").is_file():
            return candidate
    raise RuntimeError("Set CK_DIR to a Composable Kernel source tree containing include/ck_tile/core.hpp.")


def ck_commit(ck_dir: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ck_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


CK_DIR = find_ck_dir()
CK_COMMIT = ck_commit(CK_DIR)
if CK_COMMIT != EXPECTED_CK_COMMIT:
    raise RuntimeError(f"rdna35-pisa-ck requires CK commit {EXPECTED_CK_COMMIT}, found {CK_COMMIT}.")
os.environ["MAX_JOBS"] = str(max(32, int(os.environ.get("MAX_JOBS", "32"))))
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1151"
if os.name == "nt":
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")

INCLUDE_DIRS = [str(CK_DIR / "include")]
site_packages = pathlib.Path(torch.__file__).resolve().parent.parent
rocm_devel_include = site_packages / "_rocm_sdk_devel" / "include"
if rocm_devel_include.is_dir():
    INCLUDE_DIRS.append(str(rocm_devel_include))

HIP_ARGS = [
    "-O3",
    "-fbracket-depth=1024",
    "-Wno-unknown-warning-option",
]
rocm_device_libs = site_packages / "_rocm_sdk_core" / "lib" / "llvm" / "amdgcn" / "bitcode"
if os.name == "nt" and rocm_device_libs.is_dir():
    HIP_ARGS.append(f"--rocm-device-lib-path={rocm_device_libs}")


setup(
    ext_modules=[
        CUDAExtension(
            name="rdna35_pisa_ck._C",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "pisa_kernels.cu"),
            ],
            include_dirs=INCLUDE_DIRS,
            define_macros=[
                ("RDNA35_PISA_CK_API", "5"),
            ],
            extra_compile_args={
                "cxx": [],
                "nvcc": HIP_ARGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    packages=["rdna35_pisa_ck"],
)
