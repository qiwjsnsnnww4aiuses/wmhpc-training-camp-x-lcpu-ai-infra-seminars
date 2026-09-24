# R fork 相对 FlashKDA 的代码改动

## 保持原样的主体

- `csrc/smxx/fwd_kernel1.cuh`：K1 数学与 MMA 路径不变；
- `csrc/smxx/fwd_kernel2.cuh`：单个 segment 内的 K2 chunk body 不变；
- `csrc/smxx/utils.cuh`：layout、TMA pipeline、数值近似不变；
- `flash_kda_r.serial_fwd`：可独立运行的原版 launch baseline。

## R 新增的核心路径

- `csrc/smxx/fwd_launch.cu`：同一编译单元加入拆分的 K1/K2 launch；
- `csrc/smxx/reduce_launch_impl.cuh`：K1 workspace 保留，K2 按 segment 网格直接启动；
- `csrc/smxx/reduce_scan.cu`：SM80 WMMA 实现仿射 map reduce/down-sweep；
- `csrc/reduce_bindings.cpp`：把上述核心入口绑定进同一个 `flash_kda_r_C.so`；
- `flash_kda_r/__init__.py`：组织一次 K1、两次 summary K2、scan、一次 replay K2；
- `tests/test_reduce.py`：只引用 R 自己的 serial/reduce 两条路径；
- `benchmarks/bench_reduce.py`：只比较同一个 R 动态库内的两条路径。

`T=8192,G=32` 时，原版 K2 的 `grid.x=1`、每 CTA 512 chunks；R replay
的 `grid.x=16`、每 CTA 32 chunks。真正改变的是 K2 的任务切分和状态 carry
生成方式，不是矩阵指令。
