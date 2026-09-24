import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))

required_cutlass_paths = [
    os.path.join(this_dir, 'cutlass', 'include', 'cutlass', 'cutlass.h'),
    os.path.join(this_dir, 'cutlass', 'include', 'cute', 'tensor.hpp'),
    os.path.join(this_dir, 'cutlass', 'tools', 'util', 'include'),
    os.path.join(this_dir, 'cutlass', 'examples', 'common'),
]
missing_cutlass_paths = [path for path in required_cutlass_paths if not os.path.exists(path)]
if missing_cutlass_paths:
    raise RuntimeError(
        "This standalone tree is missing its vendored CUTLASS v4.3.2 files: "
        + ", ".join(missing_cutlass_paths)
    )


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]


def detect_cuda_arch():
    import torch

    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"{major}{minor}a"


def get_arch_flags():
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"

    requested = os.getenv("FLASH_KDA_CUDA_ARCHS", "auto").lower()
    if requested == "auto":
        arch = detect_cuda_arch()
        if arch is None:
            raise RuntimeError(
                "FLASH_KDA_CUDA_ARCHS=auto requires a visible CUDA device. "
                "Set FLASH_KDA_CUDA_ARCHS=all to build all supported archs."
            )
        archs = [arch]
    elif requested == "all":
        archs = SUPPORTED_CUDA_ARCHS
    else:
        archs = [arch.strip() for arch in requested.split(",") if arch.strip()]

    flags = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags


def make_extension(name, tcgen05):
    nvcc_flags = [
        '-O3',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '--use_fast_math',
        '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills',
        '-lineinfo',
        *get_nvcc_thread_args(),
        *get_arch_flags(),
    ]
    if tcgen05:
        nvcc_flags.append('-DFLASH_KDA_TCGEN05_K2=1')
    return CUDAExtension(
        name=name,
        sources=[
            'csrc/flash_kda.cpp',
            'csrc/smxx/fwd_launch.cu',
        ],
        include_dirs=[
            os.path.join(this_dir, 'cutlass', 'include'),
            os.path.join(this_dir, 'cutlass', 'examples', 'common'),
            os.path.join(this_dir, 'cutlass', 'tools', 'util', 'include'),
            os.path.join(this_dir, 'csrc'),
        ],
        extra_compile_args={
            'cxx': ['-O3', '-Wno-psabi'],
            'nvcc': nvcc_flags,
        },
    )


ext_modules = [
    make_extension('flash_kda_C', tcgen05=False),
    make_extension('flash_kda_tcgen05_C', tcgen05=True),
]
cmdclass = {"build_ext": BuildExtension}

rev = os.getenv("FLASH_KDA_VERSION_SUFFIX", "")
if not rev:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
    except Exception:
        rev = ""

setup(
    name='flash_kda_tcgen05',
    version='0.0.1' + rev,
    description='FlashKDA K2 tcgen05 instruction-replacement experiment',
    ext_modules=ext_modules,
    packages=['flash_kda', 'flash_kda_tcgen05'],
    cmdclass=cmdclass,
    zip_safe=False,
)
