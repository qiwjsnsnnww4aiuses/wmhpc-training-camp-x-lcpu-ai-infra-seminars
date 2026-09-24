# FInal standalone 边界

该目录编译或运行时不读取 `../FlashKDA`、`../FlashKDA-upstream` 或外部
`fla_kda_ref`。

目录内同时包含：

- 原版 `flash_kda` Python 包；
- 改版 `flash_kda_tcgen05` Python 包；
- 同一份 K1/K2 源码；
- CUTLASS v4.3.2 稀疏头文件副本；
- `fla_kda_ref/`；
- fixed/varlen 正确性、benchmark、SASS 导出脚本。

`setup.py` 从本目录分别编译：

```text
flash_kda_C            原版 SM80 baseline
flash_kda_tcgen05_C    定义 FLASH_KDA_TCGEN05_K2 的实验版
```

外部只要求 CUDA PyTorch、CUDA toolkit，以及参照测试所需的 FLA/Triton Python
依赖；不要求任何兄弟源码目录或预编译 FlashKDA 动态库。
