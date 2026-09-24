# FlashKDA-R：K2 carry-lookahead 独立 fork

`R/` 是以课程提供的 `FlashKDA/` 为基线建立的独立源码树。它拥有自己的
Python 包、PyTorch CUDA 扩展、K1/K2 源码、CUTLASS 头文件、测试和 benchmark；
编译与运行不导入兄弟目录中的 `FlashKDA` 或 `FlashKDA-upstream`。

## 目录

```text
R/
├── setup.py
├── flash_kda_r/
│   └── __init__.py              # R 的公开 fwd/serial_fwd
├── csrc/
│   ├── flash_kda.cpp            # 单一 flash_kda_r_C pybind 模块
│   ├── reduce_bindings.cpp      # reduce/scan 的 C++ 接口
│   ├── reduce.h
│   └── smxx/
│       ├── fwd_kernel1.cuh      # fork 内 K1
│       ├── fwd_kernel2.cuh      # fork 内 K2
│       ├── fwd_launch.cu        # 原版与分段 launch 同一编译单元
│       ├── reduce_launch_impl.cuh
│       ├── reduce_scan.cu       # SM80 WMMA 上扫/下扫 GEMM
│       └── utils.cuh
├── cutlass/                      # 仅保留独立编译所需头文件
├── tests/
│   └── test_reduce.py
├── benchmarks/
│   └── bench_reduce.py
└── run_b300.sh
```

## 核心改动

原版 `serial_fwd`：

```text
K1 prepare once -> K2 one (sequence,head) chain
```

R 的默认 `fwd`：

```text
K1 prepare once, retain workspace
  -> K2 segment A summary
  -> K2 segment B summary
  -> SM80 affine-map up-sweep/down-sweep
  -> K2 segment-parallel replay
```

`T=8192,G=32` 时，replay 从每头一条 512-chunk 长链改为每头 16 条
32-chunk 短链。当前版本仍付出两遍 summary K2 的额外工作；这是 core fork
的第一版，不承诺正收益。

## B300 独立构建与测试

```bash
cd ~/lcpu2026/assignment02/team/c1_flashkda/R
source ../FlashKDA-upstream/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export FLASH_KDA_CUDA_ARCHS=103a
python -m pip install -v -e . --no-build-isolation
python tests/test_reduce.py --T 1024 --H 2 --G 8
python benchmarks/bench_reduce.py --T 8192 --H 96 --G 32 --warmup 5 --iters 20
```

这里借用的只有 Python/Torch 运行环境。源码、动态库和测试入口全部属于 `R`。

`cutlass/` 不是旧实现：`fwd_kernel1.cuh`、`fwd_kernel2.cuh` 和 `utils.cuh`
通过它获得 CUTE、TMA layout、pipeline 与 SM80 MMA 模板。目录已经裁剪为
仅含编译所需的 `include/` 头文件，没有 CUTLASS examples、tools 或 tests。
