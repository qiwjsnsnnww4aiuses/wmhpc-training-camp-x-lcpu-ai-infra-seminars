import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = os.path.dirname(os.path.abspath(__file__))


def arch_flags():
    assert CUDA_HOME is not None, "PyTorch must have CUDA support"
    import torch

    requested = os.getenv("FLASH_KDA_CUDA_ARCHS", "auto").lower()
    if requested == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A visible GPU is required, or set FLASH_KDA_CUDA_ARCHS=103a"
            )
        major, minor = torch.cuda.get_device_capability()
        archs = [f"{major}{minor}a"]
    else:
        archs = [x.strip() for x in requested.split(",") if x.strip()]
    flags = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags


required = [
    "cutlass/include/cutlass/cutlass.h",
    "cutlass/include/cute/tensor.hpp",
    "csrc/smxx/fwd_kernel1.cuh",
    "csrc/smxx/fwd_kernel2.cuh",
]
missing = [path for path in required if not os.path.exists(os.path.join(ROOT, path))]
if missing:
    raise RuntimeError("R source tree is incomplete: " + ", ".join(missing))


extension = CUDAExtension(
    name="flash_kda_r_C",
    sources=[
        "csrc/flash_kda.cpp",
        "csrc/reduce_bindings.cpp",
        "csrc/smxx/fwd_launch.cu",
        "csrc/smxx/reduce_scan.cu",
    ],
    include_dirs=[
        os.path.join(ROOT, "cutlass", "include"),
        os.path.join(ROOT, "csrc"),
        os.path.join(ROOT, "csrc", "smxx"),
    ],
    extra_compile_args={
        "cxx": ["-O3", "-Wno-psabi"],
        "nvcc": [
            "-O3",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
            *arch_flags(),
        ],
    },
)


setup(
    name="flash_kda_r",
    version="0.1.0",
    description="Standalone FlashKDA K2 carry-lookahead core fork",
    packages=["flash_kda_r"],
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
