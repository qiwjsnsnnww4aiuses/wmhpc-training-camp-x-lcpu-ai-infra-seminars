下面按 Module 整理 **1–7 的所有 CONCEPT 题**，只保留题干与选项/判断点，便于集中复习。

---

## Module 1：为什么要用 GPU

**prob 1.1（CONCEPT）**
判断对错，可补理由。

- (a) 一块标称 100 TFLOPS 的 GPU，执行单条指令的延迟一定低于 5GHz 的 CPU。
错，GPU用高并发掩盖数据传输的延迟
- (b) HBM 的“高带宽”指大块连续访问时的吞吐，零散的随机访问达不到标称值。
对，零散随机访问会导致传输大量无效数据 Dram RowBuffer连续数据访问更快
- (c) 严格串行的迭代算法（每步依赖上一步的结果），即使换一块算力更强的 GPU 也快不了多少。
对，考虑高延迟时间
- (d) “算力 1000 TFLOPS”意味着每次运算的延迟是 10⁻¹⁵ 秒。
错，是指1s完成约1000TFLOPS

**prob 1.2（CONCEPT）**
“N 方过百万”：总计算量 10¹² FLOP 在当代 GPU 上约毫秒级，为什么严格在线的串行算法仍做不到几秒内跑完？从“延迟”和“吞吐”角度考虑。
GPU 10^12FLOPS毫秒级完成 靠的是计算单元并行和warp并发 但是串行只能等上一步完成-时间=串行步数*单步延迟
而GPU本身单步延迟就打，串行无法利用吞吐优势

**prob 1.3（CONCEPT）**
补全下表（thread 已填好）：

| 执行层次 | 软件含义 | 对应硬件 | 直接可用的存储 | 同步与通信手段 |
| :--- | :--- | :--- | :--- | :--- |
| thread | kernel 的最小执行单位 | 计算单元上的一个 lane | 自己的寄存器（自身天然有序） | |
| warp | 32threads组成的调度单位 | 一个warp scheduler/32lanes |  每个lane的寄存器 |  __syncwarp |
| block/CTA | 一组warp在一个SM上 | SM | sharedmem块内共享 | __syncthreads() |
| grid | 一次kernellaunch所有block|  所有SM | global/constantmem | - |

**prob 1.4（CONCEPT）**
SIMD 与 SIMT 的区别？另判断正误：Nvidia GPU 在 Volta 之后每个线程有独立的 program counter，所以 branch divergence 不再有性能代价。
SIMD单指令多数据 如单指令进行向量加法 SIMT单指令多线程 如launchkernel 通过BlockIdx/BlockDim/ThreadIdx操作 不同线程可以不同分支
每个线程都有独立PC 但是divergence依旧要求串行 还是有性能代价

---

## Module 2：第一个 CUDA 程序

**prob 2.2（CONCEPT）**
为下列五个场景选择正确的修饰符（如 `__global__` 等）：

- (a) 在 GPU 上执行、由 CPU 侧启动的 kernel 函数。
__global__
- (b) 只会被 kernel 调用的辅助函数。
__device__
- (c) host 和 device 代码都要调用的小工具函数。
__host__ __device__
- (d) 整个 kernel 运行期间不变、所有线程都要读的系数表。
__constant__
- (e) block 内线程共享的暂存数组。
__shared__

**prob 2.4（CONCEPT）**
判断对错，可补理由。

- (a) `vectorAdd<<<...>>>(...)` 这条语句返回时，kernel 一定已经执行完毕。
错，只能说明cpu测启动了在GPU上执行的kernel
- (b) 同一个 stream 里，`cudaMemcpy`（device 到 host）会等它前面的 kernel 全部完成后才开始拷贝。
对。同一个stream是FIFO队列。
- (c) kernel 内部的非法访存，会在启动语句处同步地报出来。
错。如果是在GPU执行kernel中非法访存，语句启动时不会有感知报告。

---

## Module 3：SIMT 执行

**prob 3.1（CONCEPT）**
设 `blockDim = (8, 8, 1)`。

- (a) `threadIdx = (3, 5, 0)` 的线性编号是多少？在第几个 warp、warp 内第几个 lane？
5*8+3=43 43/32=1 43%32=11
- (b) 这个 block 一共占多少个 warp？
2
- (c) 若 `blockDim = (33, 1, 1)`，占几个 warp？这样配置浪费在哪里？
2。最后一个warp只有一个lane。

**prob 3.4（CONCEPT）**
`__syncthreads` 只能同步本 block 内的 threads，那需要全 grid 同步时，标准做法是什么？
直接拆成两个或者多个kernel. 比如向量加法在kernel1，加法后点乘在kernel2。
---

## Module 4：存储空间

**prob 4.1（CONCEPT）**
补全下表：

| 空间 | 谁可见 | 生命周期 | 片上/片外 | 谁管理 |
| :--- | :--- | :--- | :--- | :--- |
| register | 单个线程 | 线程存在期间 | 片上 | 编译器自动分配 |
| local | 单个线程 | 线程存在期间 | 片外Dram/GlobalMem虚拟出来的 | 编译器寄存器溢出时 |
| shared | 同block所有线程 | block存在期间 | 片上 | 程序员__shared__显式管理 |
| global | 整个grid所有线程+host | cudaMelloc->cudaFree | 片外Dram| 程序员（显式 cudaMalloc / cudaFree） |
| constant | 整个grid所有线程 | 整个程序(kernel运行期间不变) | 片外Dram但有常量缓存 | 程序员（__constant__ + cudaMemcpyToSymbol） |
| L1 / L2 cache | | | | |

**prob 4.4（CONCEPT）**
判断对错，可补理由。

- (a) local memory 的“local”指作用域私有，它实际上在外显存里。
对。和Global Memory一样存在Dram
- (b) 对数组用运行期才知道的下标做索引，可能迫使它被放进 local memory。
对。运行时不知道变量多少，寄存器溢出就需要放进local memory

---

## Module 5：计时与异步初步

**prob 5.2（CONCEPT）**
判断对错，可补理由。

- (a) 同一个 stream 里的操作按提交顺序执行。
对。遵循FIFO。
- (b) kernel 启动后，host 代码立刻继续往下执行。
对。kernel是在GPU上执行的。启动只是把kernel放进GPU执行队列就返回cpu host
- (c) unified memory 下，CPU 访问一页正被 GPU 占用的内存，会触发缺页与页迁移。
对。UM 下，CPU 和 GPU 共享同一份虚拟地址空间。数据实际存储位置（CPU 内存 or GPU 显存）由系统动态决定。迁移以页为单位，触发方式是缺页。

---

## Module 6：Tile 视角

**prob 6.1（CONCEPT）**
判断对错，可补理由。

- (a) tile 是显存里的一块可变区域，kernel 通过指针直接改写它。
对。 tile 是一块数据区域，可以是 global memory、shared memory 或寄存器里的一个多维子块
- (b) 对 tile 的一次运算（如两个 tile 相加）由编译器映射到 block 内的多个线程上执行。
对。程序员以 tile 为单位写运算，编译器负责把 tile 运算拆解到 block 内的多个线程上
- (c) tile 模型与 SIMT 模型互斥，一个 CUDA 程序只能选一种。
错，可以大尺度上tile，但最小层级thread依旧是SIMT单指令多线程。tile层指令会被编译成SIMT代码操作threads

**prob 6.2（CONCEPT）**
下面是 Guide 2.4.6 的 cuTile Python 向量加法：

```python
import cuda.tile as ct

@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    a_view = a.tiled_view(TILE,)
    b_view = b.tiled_view(TILE,)
    c_view = c.tiled_view(TILE,)
    bid = ct.bid(0)
    a_tile = a_view.load(bid,)
    b_tile = b_view.load(bid,)
    c_view.store(bid, a_tile + b_tile)
```

据此补全下表（Triton 一列可做完 Module 7 再回来填）：

| | CUDA SIMT | cuTile | Triton |
| :--- | :--- | :--- | :--- |
| 并行单位 | block 里的 thread | block | program|
| 编号 | blockIdx / threadIdx | ct.bid(0) | tl.program_id(0) |
| 数据分工 | 线程用全局下标来划分数据 | tiled_view声明| 按program+tl.arrange取出一块|
| 边界处理 | if 判断 | 编译器自动处理| mask|

**prob 6.3（CONCEPT）**
仍看上面这段代码。

- (a) “每个线程对应哪个/些元素”由谁决定？
由编译器决定 没有出现threadIdx
- (b) 列出一些在 CUDA SIMT 版向量加法里一定会出现、这里完全没体现出的概念。
BlockDim threadIdx sharedmem声明 i<=n-1边界判断等

---

## Module 7：TileLang 与 Triton

**prob 7.5（CONCEPT）**
补全下表，每个空填“用户”或“编译器”（二者都涉及的要写清楚各自的范围）：

| 谁负责 | CUDA SIMT | cuTile | Triton | TileLang |
| :--- | :--- | :--- | :--- | :--- |
| 线程到数据的映射 | 用户 | 编译器 | 编译器 | 编译器 |
| 边界处理 | 用户 | 编译器 | 用户mask | 编译器或用户mask |
| tile / block 尺寸的选择 | 用户 | 用户 | 用户 | 用户 |
| block 内同步 | 用户 | 编译器 | 编译器 | 编译器 |