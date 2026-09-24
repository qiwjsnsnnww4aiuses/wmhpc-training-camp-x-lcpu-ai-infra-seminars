# C1 FlashKDA Challenge：tcgen05 指令替换傻瓜式完整讲解

> 目标：让第一次接触这个实验的人，也能从“为什么做”一路理解到“代码怎么跑、数据怎么算、结果说明什么”。
>
> 本文只讨论我们已经完成的 challenge 路线：**不改变数学算法，只比较 SM80 `mma.sync` 与 SM100 `tcgen05.mma` 两条 Tensor Core 实现路径。**

---

## 0. 先给最终结论

我们在 NVIDIA B300 上完成了一轮 microbenchmark：

- GPU：NVIDIA B300 SXM6 AC，compute capability 10.3；
- CUDA Toolkit：13.0，`nvcc V13.0.88`；
- 输入：BF16；
- 累加和输出：FP32；
- 两条路径计算完全相同的矩阵乘；
- 两个 shape 的 SM80 和 tcgen05 结果均与 CPU FP32 reference 严格相等；
- 最终 SASS 中分别确认了 `HMMA.16816.F32.BF16` 和 `UTCHMMA`。

单轮性能结果如下：

| Case | 有用 shape | SM80 MMA | tcgen05 | 加速因子 `SM80时间/tcgen05时间` | 结果 |
|---|---|---:|---:|---:|---|
| thin | M16N128K16 | 6.155 μs | 12.300 μs | 0.500× | tcgen05 约慢 2.00 倍 |
| matched | M128N128K16 | 14.542 μs | 50.765 μs | 0.286× | tcgen05 约慢 3.49 倍 |

所以本次 challenge 的结论是：

> 在保持 CHUNK、数学运算和整体算法不变时，机械地将 SM80 `mma.sync` 路径替换成 SM100 `tcgen05.mma` 路径没有性能收益。M16 场景存在 8 倍 issued/useful FLOP 浪费；即使 M128 与 tcgen05 物理 tile 自然匹配，TMEM 分配、描述符准备、同步、提交、读取和释放等完整路径开销仍然使它更慢。

这个负结果不是失败。题目明确允许“没有正收益但把官方继续采用 SM80 MMA 的理由论证扎实”。

---

## 1. 这个 challenge 在整个 C1 任务中的位置

C1 的核心问题是：

> FlashKDA 在 GB200/B300 上运行，为什么计算主路径依然使用 SM80 世代的 `mma.sync`，而没有换成 SM100 的 `tcgen05`？

完整任务有三层：

1. 复现 FlashKDA，跑通测试和 benchmark；
2. 用源码、SASS、NCU 和纸面分析回答讨论问题；
3. 选择一条 SM100 路线实际动手挑战。

我们选择的 challenge 是最小、最容易控制变量的一条：

```text
不改算法
不改输入和输出数据类型
不改数学矩阵形状
只替换 Tensor Core 指令路径
```

它要回答讨论问题 2：

> tcgen05 的最小 tile 与 CHUNK=16 匹配吗？不动 CHUNK 只换指令有没有收益？先纸上算，再用 microbenchmark 验证。

实验逻辑可以压缩成：

```text
同一批 A、B
   │
   ├── SM80 mma.sync ──► D_sm80
   │
   ├── SM100 tcgen05 ──► D_tcgen05
   │
   └── CPU FP32 ───────► D_ref

先验证：D_sm80 == D_ref，D_tcgen05 == D_ref
再计时：T_sm80 与 T_tcgen05 谁更短
最后看 SASS：二进制里是否真的出现 HMMA 和 UTCHMMA
```

---

## 2. 它不是完整 FlashKDA 内核

这是理解 challenge 的第一道坎。

`instruction_only_mma.cu` 不是把整个 FlashKDA K1/K2 重写成 tcgen05。它抽取了 K1/K2 中最关键的矩阵乘切面，构造一个独立 CUDA microbenchmark。

它保留的是：

- FlashKDA 关心的 BF16 输入、FP32 累加；
- `K=16`，对应 `CHUNK=16`；
- thin GEMM 的 M16 特征；
- K2 state-delta 中可能出现的 M128 匹配特征；
- SM80 `mma.sync` 与 SM100 `tcgen05` 的真实使用路径。

它没有包含完整 FlashKDA 的：

- gate 和 decay 计算；
- Neumann 逆；
- 完整 K1、K2 数据流；
- chunk 间递推状态依赖；
- 最终 output/state 接口；
- 与 Python/Torch extension 的绑定；
- 完整 kernel 中其他 load、store 和融合操作。

所以它能回答：

> 对这个代表性 GEMM 切面，只换 Tensor Core 路径有没有潜在收益？

它不能单独证明：

> 完整 FlashKDA 换成 tcgen05 后一定会得到完全相同的速度比。

---

## 3. 两条路径计算的数学问题完全相同

程序中的矩阵乘是：

$$
D_{m,n} = \sum_{k=0}^{15} A_{m,k} B_{n,k}
$$

换成矩阵写法，相当于：

$$
D = A B^T
$$

张量形状是：

```text
A: [M, K]
B: [N, K]
D: [M, N]
K = 16
N = 128
```

为什么代码写 `B[n,k]` 而不是 `B[k,n]`？

因为 `B` 在内存中保存为 `[N,K]`，数学上使用的是它的转置，所以：

```text
D[m,n] = A 的第 m 行 · B 的第 n 行
```

两条 GPU 路径和 CPU reference 都使用：

- `A`：BF16；
- `B`：BF16；
- 乘加累积：FP32；
- `D`：FP32。

这就是“不改算法”的核心：只改变怎样让 Tensor Core 执行矩阵乘，不改变矩阵乘本身。

---

## 4. 为什么设计两个 case

程序分别实例化 `UsefulM=16` 和 `UsefulM=128`。

### 4.1 Case A：M16 thin GEMM

有用的数学 shape 是：

```text
M16 × N128 × K16
```

它代表 FlashKDA 里由 `CHUNK=16` 形成的瘦矩阵乘特征。

SM80 指令的 shape 是：

```text
mma.sync.m16n8k16
```

其中 M16 和 K16 与问题天然匹配。为了覆盖 N128，需要沿 N 方向分解多个 N8 tile。

tcgen05 路径使用的物理 shape 是：

```text
tcgen05.mma.m128n128k16
```

但我们真正需要的 M 只有 16，所以程序把 A 的第 16～127 行补成精确的 0：

```text
有用行：0～15
补零行：16～127
```

计算结束后，只保存前 16 行结果。

因为：

$$
0 \times B = 0
$$

所以补零不会改变前 16 行有用结果，但 GPU 仍然真实计算了完整 M128 tile。

物理工作量与有用工作量之比：

$$
\frac{128}{16}=8
$$

所以日志显示：

```text
issued/useful FLOP: 8.0x
```

这个 case 用来验证纸面判断：

> tcgen05 的 M128 物理 tile 与 CHUNK=16 的 M16 问题不匹配。

### 4.2 Case B：M128 matched GEMM

第二个 case 的有用 shape 就是：

```text
M128 × N128 × K16
```

它与 tcgen05 的物理 tile：

```text
M128 × N128 × K16
```

完全匹配，因此：

```text
issued/useful FLOP: 1.0x
```

为什么必须增加这个 case？

如果只测 M16，tcgen05 变慢后，我们只能说：

> 它可能只是被 8 倍补零拖慢。

加入 M128 后，就能进一步问：

> 当形状完全匹配、没有补零浪费时，tcgen05 会不会变快？

实际结果仍然变慢，因此说明形状不匹配不是唯一原因，tcgen05 完整使用路径的固定成本也很重要。

---

## 5. 源码目录和每个文件的作用

开发目录：

```text
assignment02/team/c1_flashkda/experiments/
├── instruction_only_mma.cu
├── Makefile
├── README.md
└── run_b300.sh
```

### 5.1 `instruction_only_mma.cu`

核心 CUDA/C++ 源码，里面包含：

- CPU FP32 reference；
- SM80 `mma.sync` kernel；
- SM100 `tcgen05` kernel；
- 正确性比较；
- CUDA event 计时；
- TFLOP/s 和 speedup 计算；
- M16、M128 两个测试 case。

### 5.2 `Makefile`

负责把 `.cu` 源码编译成 GPU 可执行程序，并支持导出 SASS。

主要目标：

```text
make              编译 bin/instruction_only_mma
make run          编译后运行默认参数
make sass         导出 bin/instruction_only_mma.sass
make clean        删除生成的二进制和 SASS
```

### 5.3 `run_b300.sh`

一键运行脚本，顺序完成：

1. 强制重新编译；
2. 创建结果目录；
3. 运行 benchmark；
4. 把程序标准输出保存成带时间戳的文本；
5. 用 `cuobjdump` 导出 SASS；
6. 从 SASS 中初步查找 Tensor Core 指令。

### 5.4 `README.md`

原始实验说明，包含运行方式、参数、NCU 可选命令和结果解释。

---

## 6. `instruction_only_mma.cu` 的整体执行流程

程序入口 `main()` 做了以下事情：

```text
读取 jobs / iterations / warmup / seed
            │
            ▼
读取 GPU 型号和 compute capability
            │
            ▼
检查 GPU major >= 10
            │
            ├── 否：退出，提示 tcgen05 需要 SM100 family
            │
            └── 是
                 │
                 ▼
          run_case<16>()
                 │
                 ▼
          run_case<128>()
                 │
                 ▼
      两个 case 是否都正确？
                 │
                 ▼
      overall correctness: PASS/FAIL
```

每一个 `run_case<UsefulM>()` 又分成六步：

1. 构造随机 BF16 输入；
2. CPU 计算 FP32 reference；
3. 分配显存并复制输入；
4. 各运行一次 SM80 和 tcgen05，用于正确性检查；
5. warmup 后分别正式计时；
6. 打印误差、延迟、TFLOP/s 和加速因子。

---

## 7. 输入数据为什么使用 -2 到 2 的小整数

源码使用：

```cpp
std::uniform_int_distribution<int> distribution(-2, 2);
```

所以每个输入元素来自：

```text
{-2, -1, 0, 1, 2}
```

然后转换成 BF16。

这样设计有两个优点：

1. 这些小整数都能被 BF16 精确表示；
2. 16 项乘积之和通常也能被 FP32 精确表示。

因此，如果两个 kernel 的数据布局、tile 映射、转置方向或结果索引有错误，很容易出现非零 mismatch；如果全部正确，就可以得到：

```text
mismatches=0
max_abs=0
```

但必须注意：

> 这个严格相等测试主要验证指令替换、数据布局和索引是否正确，不是广泛的数值稳定性测试。

FlashKDA 完整算子的 BF16 精度已经在第五问中通过 FP64 reference、不同 gate 和 fixed/varlen 场景另行验证。

---

## 8. CPU reference 到底算了什么

CPU 使用三层循环：

```cpp
for m
    for n
        for k
            reference[m,n] += A[m,k] * B[n,k]
```

这就是最朴素的：

$$
D_{m,n}=\sum_k A_{m,k}B_{n,k}
$$

当前实现的一个边界需要诚实说明：

- GPU 一次 launch 会运行 `jobs=512` 个独立 block/job；
- CPU reference 当前只构造一个 `[UsefulM,N]` 结果；
- 从 GPU 拷回检查的也是输出开头，即第 0 个 job；
- 因而 `mismatches=0` 是第一个代表性 job 的逐元素验证，不是 512 个 job 的全量逐元素验证。

因为每个 CUDA block 使用相同代码处理各自 job，这足以验证主要 tile 映射和公式；但如果将来加强实验，可以为每个 job 都生成 CPU reference，并检查全部 `jobs × M × N` 输出。

报告中推荐写：

> 正确性部分对第一个代表性 job 的全部输出元素与 CPU FP32 reference 进行严格对拍，两条路径均无 mismatch。

不要写成“512 个 job 的每一个元素均已对拍”，因为当前代码没有做这个全量检查。

---

## 9. SM80 `mma.sync` kernel 怎么工作

kernel 名称：

```cpp
sm80_mma_kernel<UsefulM>
```

启动配置：

```text
grid  = jobs 个 block
block = 128 threads = 4 warps
```

一块 CTA 处理一个 job。

主要步骤：

### 9.1 Global Memory → Shared Memory

所有线程合作把 A 和 B 从 global memory 搬进 shared memory：

```text
shared_a[UsefulM × 16]
shared_b[128 × 16]
```

随后调用：

```cpp
__syncthreads();
```

确保所有线程都看到了完整输入。

### 9.2 四个 warp 分 N 方向

每个 warp 负责 32 个输出列：

```text
warp 0 → N 0～31
warp 1 → N 32～63
warp 2 → N 64～95
warp 3 → N 96～127
```

### 9.3 用 `m16n8k16` 分块

核心内联 PTX 是：

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

它表示：

- M tile：16；
- N tile：8；
- K tile：16；
- A/B：BF16；
- accumulator/output：FP32；
- 一个 warp 合作执行。

对于 M16N128K16：

```text
M 方向 tile 数 = 16 / 16 = 1
N 方向 tile 数 = 128 / 8 = 16
总 warp-level tile = 16
```

四个 warp 各负责四个 N8 tile。

对于 M128N128K16：

```text
M 方向 tile 数 = 128 / 16 = 8
N 方向 tile 数 = 128 / 8 = 16
总 warp-level tile = 128
```

注意：源码和 SASS 中某条指令只出现一次或几次，不等于运行时只发射一次。循环、warp 数量和模板展开都会影响动态发射次数。

---

## 10. tcgen05 kernel 怎么工作

kernel 名称：

```cpp
tcgen05_mma_kernel<UsefulM>
```

同样使用：

```text
grid  = jobs
block = 128 threads = 4 warps
```

但 tcgen05 不能像旧 MMA 那样只准备寄存器 fragment 然后直接使用结果，它还需要 TMEM、descriptor 和异步完成同步。

完整流程是：

```text
初始化 mbarrier
      │
      ▼
分配 TMEM
      │
      ▼
把 A/B 按 SM100 swizzle 写入 shared memory
      │
      ▼
生成 A/B descriptor
      │
      ▼
elect 一个线程发出 tcgen05.mma
      │
      ▼
commit 到 mbarrier
      │
      ▼
所有线程等待完成
      │
      ▼
从 TMEM 读取 FP32 结果
      │
      ▼
只保存 UsefulM 行
      │
      ▼
释放 TMEM
```

### 10.1 TMEM 是什么

TMEM 可以理解成 SM100 Tensor Core 路径使用的一种专用 Tensor Memory。tcgen05 的输出先进入 TMEM，然后再由线程使用 `tcgen05.ld` 读出来。

所以 tcgen05 路径额外包含：

- `tcgen05.alloc`；
- `tcgen05.mma`；
- `tcgen05.commit`；
- `mbarrier` 等待；
- `tcgen05.ld`；
- `tcgen05.wait::ld`；
- `tcgen05.dealloc`。

这些都属于真实使用 tcgen05 所必须付出的路径成本。

### 10.2 为什么 shared memory 看起来比逻辑矩阵大

tcgen05 路径声明：

```text
shared_a[128 × 128 bytes]
shared_b[128 × 128 bytes]
```

逻辑上每一行只有 K16 个 BF16，即：

```text
16 × 2 bytes = 32 bytes
```

但为了满足 SM100 128-byte swizzle/descriptor 布局，物理上为每行保留 128-byte stride。

这也是“新指令的更大 tile/布局可能带来固定成本”的组成部分。

### 10.3 `swizzle_128b()` 做什么

它把逻辑位置：

```text
(row, col_byte)
```

映射到 SM100 期望的 shared-memory 物理地址。

它不是改变数学矩阵，而是改变数据在 shared memory 中的排列，让 Tensor Core 能按照 descriptor 正确读取。

### 10.4 为什么只由一个 elected thread 发出 MMA

源码调用：

```text
elect.sync
```

从 warp 中选出一个线程，由它发出 `tcgen05.mma` 和 commit，避免同一个 CTA 内重复发出相同操作。

### 10.5 M16 时为什么四个 warp 仍读取完整物理结果

每个 warp 从 TMEM 读取 32 行：

```text
warp 0 → row 0～31
warp 1 → row 32～63
warp 2 → row 64～95
warp 3 → row 96～127
```

M16 case 最终只写回 row 0～15，但 row 16～127 对应的 TMEM 读取成本仍然被计入。

这是有意设计的：实验要测真实的形状不匹配成本，而不是在计时之外偷偷省掉它。

---

## 11. 正确性是怎样判断的

两条 kernel 首先各运行一次：

```text
SM80 kernel    → device_sm80
tcgen05 kernel → device_tcgen05
```

然后把第一个 job 的结果复制回 CPU，与 reference 比较。

指标有两个：

### 11.1 `mismatches`

只要某元素满足：

```text
got[i] != reference[i]
```

就算一个 mismatch。

当前比较不是 `allclose`，而是严格相等。

### 11.2 `max_abs`

$$
\text{max\_abs}=\max_i |\text{got}_i-\text{reference}_i|
$$

本次结果：

```text
mismatches=0
max_abs=0
```

说明被检查的输出逐元素完全一致。

最终：

```text
overall correctness: PASS
```

只有 M16 和 M128 的 SM80/tcgen05 四项比较全部通过，程序才以成功状态退出。

---

## 12. 性能计时到底测的是什么

这是最容易误解的部分。

运行参数是：

```text
jobs=512
iterations=200
warmup=30
seed=42
```

### 12.1 `jobs=512` 是什么意思

`grid(jobs)`，所以一次 kernel launch 有 512 个 CTA：

```text
CTA 0   → job 0 的矩阵乘
CTA 1   → job 1 的矩阵乘
...
CTA 511 → job 511 的矩阵乘
```

每个 job 都是独立的小矩阵乘。

因此日志中的：

```text
0.006155 ms
```

是“一次 kernel launch 并行处理 512 个 job”的平均时间，不是一个小矩阵在 GPU 上串行执行 6.155 μs 后再乘 512。

使用 512 个 job 的目的，是给 B300 足够多 CTA，避免只发一个极小 CTA 时测到的主要是启动开销和 GPU 未占满状态。

### 12.2 `warmup=30` 是什么意思

正式计时前先运行 30 次 kernel。

目的包括：

- 让 CUDA context、cache 和 GPU 状态稳定；
- 避免第一次运行的初始化成本污染正式计时；
- 让两条路径都经过相同 warmup。

warmup 完成后调用 `cudaDeviceSynchronize()`，确保正式计时从干净的边界开始。

### 12.3 `iterations=200` 是什么意思

CUDA event 记录开始时间，然后连续 launch 200 次同一个 kernel，再记录结束时间。

最终：

$$
T_{reported}=\frac{T_{200\ launches}}{200}
$$

所以报告的是平均每次 kernel launch 的时间。

### 12.4 计时包含什么

计时包含每次 kernel 内的：

- global memory → shared memory；
- CTA 同步；
- Tensor Core 计算；
- 结果写回 global memory；
- tcgen05 路径的 TMEM alloc/commit/wait/load/dealloc。

计时不包含：

- CPU 随机数生成；
- `cudaMalloc`；
- CPU reference；
- Host→Device 输入复制；
- Device→Host 正确性结果复制；
- 编译时间；
- SASS 导出时间。

因此，这是两条“实际可使用 kernel 路径”的公平对比，而不是单条机器指令的裸 latency 测量。

---

## 13. FLOP 和 TFLOP/s 是怎样算出来的

矩阵乘中，每一个输出元素包含 K 次乘法和 K 次加法，通常按 `2K` FLOP 计算。

单个 job 的有用 FLOP：

$$
\text{FLOP}_{job}=2MNK
$$

一次 launch 有 `jobs` 个独立矩阵乘：

$$
\text{FLOP}_{launch}=2\times jobs\times M\times N\times K
$$

### 13.1 M16 case

$$
2\times512\times16\times128\times16
=33,554,432\ \text{FLOP}
$$

这是有用工作量。

tcgen05 实际发出 M128 物理 tile：

$$
2\times512\times128\times128\times16
=268,435,456\ \text{FLOP}
$$

因此：

$$
\frac{268,435,456}{33,554,432}=8
$$

### 13.2 M128 case

有用和物理 shape 相同：

$$
2\times512\times128\times128\times16
=268,435,456\ \text{FLOP}
$$

因此 issued/useful 为 1。

### 13.3 useful TFLOP/s

$$
\text{useful TFLOP/s}
=\frac{\text{useful FLOP}}{T_{seconds}\times10^{12}}
$$

它表示真正为所需数学结果完成了多少有效工作。

### 13.4 issued TFLOP/s

$$
\text{issued TFLOP/s}
=\frac{\text{physical issued FLOP}}{T_{seconds}\times10^{12}}
$$

M16 的 tcgen05 日志显示：

```text
useful 2.728 TFLOP/s
issued 21.824 TFLOP/s
```

`21.824` 看起来很高，但：

$$
21.824 / 2.728 = 8
$$

七八成物理计算对应的是补零行，不能转化为有用输出。

---

## 14. `speedup tcgen05/SM80` 应怎样读

程序代码实际计算：

$$
\text{speedup}=\frac{T_{SM80}}{T_{tcgen05}}
$$

所以：

```text
speedup > 1：tcgen05 更快
speedup = 1：两者相同
speedup < 1：tcgen05 更慢
```

日志标签写作：

```text
speedup tcgen05/SM80
```

它的含义是“tcgen05 相对 SM80 的加速因子”，不是直接做 `T_tcgen05/T_SM80`。

例如 M16：

$$
\frac{0.006155}{0.012300}=0.5004
$$

所以打印 `0.500x`。

它也可以换一个更直观的说法：

$$
\frac{T_{tcgen05}}{T_{SM80}}
=\frac{0.012300}{0.006155}
\approx1.998
$$

即 tcgen05 耗时约为 SM80 的 2.00 倍。

---

## 15. Makefile 逐项解释

Makefile 的关键内容：

```make
NVCC ?= nvcc
ARCH ?= 103a
FLAGS ?= -O3 -std=c++17 -lineinfo --expt-relaxed-constexpr
GENCODE = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
```

### 15.1 `NVCC ?= nvcc`

默认使用 PATH 中的 `nvcc`。`?=` 表示如果外部没有指定，才使用这个默认值。

例如也可以外部覆盖：

```bash
make NVCC=/usr/local/cuda/bin/nvcc
```

### 15.2 `ARCH ?= 103a`

B300 报告 compute capability 10.3，因此本次使用 `103a`。

末尾的 `a` 表示生成依赖该架构特定功能的代码。tcgen05 属于需要明确目标架构支持的新指令路径。

### 15.3 `-O3`

启用较高级别编译优化，用于性能代码。

### 15.4 `-std=c++17`

使用 C++17 标准编译 host C++ 部分。

### 15.5 `-lineinfo`

保留源码行号相关信息，方便 profiler 将机器代码与源码对应。它不是完整调试模式，通常比 `-G` 更适合性能构建。

### 15.6 `--expt-relaxed-constexpr`

放宽 CUDA host/device `constexpr` 使用规则，允许部分模板和辅助函数更顺利编译。

### 15.7 `-gencode arch=compute_103a,code=sm_103a`

告诉 `nvcc`：

```text
前端按 compute_103a 能力编译
后端生成 B300/SM103a 对应机器代码
```

这一步非常关键。目标架构不支持时，tcgen05 内联 PTX 无法正确汇编。

### 15.8 输出文件

```make
TARGET = bin/instruction_only_mma
SOURCE = instruction_only_mma.cu
```

即：

```text
输入源码：instruction_only_mma.cu
输出程序：bin/instruction_only_mma
```

---

## 16. 本次编译命令逐字解释

本次真实编译命令是：

```bash
nvcc -O3 -std=c++17 -lineinfo --expt-relaxed-constexpr \
  -gencode arch=compute_103a,code=sm_103a \
  -o bin/instruction_only_mma \
  instruction_only_mma.cu
```

按从左到右理解：

```text
nvcc
```

调用 NVIDIA CUDA 编译器。

```text
-O3
```

启用性能优化。

```text
-std=c++17
```

使用 C++17。

```text
-lineinfo
```

保留 profiler 所需行号信息。

```text
--expt-relaxed-constexpr
```

放宽 CUDA constexpr 限制。

```text
-gencode arch=compute_103a,code=sm_103a
```

面向 B300 的 SM103a 编译并生成对应机器代码。

```text
-o bin/instruction_only_mma
```

指定输出二进制文件名。

```text
instruction_only_mma.cu
```

要编译的源文件。

真实编译记录保存在：

```text
remote_b300_results/challenge/logs/build_103a.log
```

---

## 17. 一键测试命令逐字解释

本次使用：

```bash
chmod +x run_b300.sh
set -o pipefail
ARCH=103a bash run_b300.sh 512 200 30 42 \
  2>&1 | tee ../repro_results/challenge/full_run_seed42.log
```

### 17.1 `chmod +x run_b300.sh`

给脚本增加“可以直接执行”的权限。

本次使用的是 `bash run_b300.sh`，严格来说即使没有 executable bit，显式交给 `bash` 也能执行；设置权限是为了之后也可以使用：

```bash
./run_b300.sh
```

### 17.2 `set -o pipefail`

一条管道中，只要前面的运行脚本失败，整条命令就返回失败。

如果没有它：

```text
程序失败 | tee 正常写文件
```

shell 可能只看到 `tee` 成功，从而误以为整次实验成功。

### 17.3 `ARCH=103a`

只为这一条命令临时设置环境变量 `ARCH`。

脚本读取：

```bash
C1_ARCH="${ARCH:-103a}"
```

如果外部指定了 `ARCH` 就使用外部值，否则默认也是 `103a`。

### 17.4 `bash run_b300.sh`

明确让 Bash 解释并执行脚本。

### 17.5 参数 `512 200 30 42`

依次对应：

| 参数 | 名称 | 本次值 | 含义 |
|---:|---|---:|---|
| 1 | jobs | 512 | 一次 launch 中 512 个独立 CTA/job |
| 2 | iterations | 200 | 正式计时连续运行 200 次，再除以 200 |
| 3 | warmup | 30 | 正式计时前预热 30 次 |
| 4 | seed | 42 | 随机输入种子 |

### 17.6 末尾的反斜杠 `\`

表示命令还没有结束，下一行继续。

如果写在一行，可以省略：

```bash
ARCH=103a bash run_b300.sh 512 200 30 42 2>&1 | tee ../repro_results/challenge/full_run_seed42.log
```

### 17.7 `2>&1`

Linux 中：

```text
1 = 标准输出 stdout
2 = 标准错误 stderr
```

`2>&1` 表示把错误输出也合并到标准输出。

这样编译错误、CUDA 错误和正常 benchmark 输出都会进入同一个日志。

### 17.8 `| tee ...`

管道符 `|` 把前一条命令输出交给 `tee`。

`tee` 同时做两件事：

1. 仍然把内容显示在终端；
2. 把相同内容写入日志文件。

所以既能现场看，也能永久保存。

---

## 18. `run_b300.sh` 内部具体做了什么

脚本开头：

```bash
set -euo pipefail
```

含义：

- `-e`：某条命令失败就停止；
- `-u`：使用未定义变量时报错；
- `-o pipefail`：管道中任一环节失败都算失败。

然后读取参数和默认值：

```text
ARCH 默认 103a
jobs 默认 256
iterations 默认 100
warmup 默认 20
seed 默认 20260910
```

本次外部传入了：

```text
103a / 512 / 200 / 30 / 42
```

### 18.1 `make -B ARCH="$C1_ARCH"`

`-B` 表示无论文件时间戳如何，都强制重新构建。

这样从 `100a` 切到 `103a` 时，不会误用此前另一架构编译的旧二进制。

### 18.2 生成带时间戳的结果文件

脚本构造：

```text
results/instruction_only_<arch>_<timestamp>.txt
```

本次生成：

```text
results/instruction_only_103a_20260911_024535.txt
```

这个文件只保存程序的标准输出，即：

- GPU 型号；
- 运行参数；
- 正确性；
- 延迟；
- TFLOP/s；
- speedup。

外层的 `full_run_seed42.log` 更完整，因为它还包含：

- `nvcc` 编译命令；
- 脚本生成文件提示；
- SASS grep 摘要；
- 标准错误。

### 18.3 导出 SASS

```bash
cuobjdump --dump-sass bin/instruction_only_mma \
  > bin/instruction_only_mma.sass
```

其中：

- `cuobjdump`：CUDA 二进制检查工具；
- `--dump-sass`：反汇编最终 GPU 机器代码；
- `>`：把输出写进文件，而不是显示在终端。

---

## 19. 源码、PTX、SASS 三层不要混淆

可以把 CUDA 编译过程理解成：

```text
CUDA C++ 源码
     │
     ▼
内联 PTX 指令
     │
     ▼
目标 GPU SASS 机器指令
```

### 19.1 源码/CUDA C++ 层

你能看到 kernel、循环、shared memory、模板等逻辑。

### 19.2 PTX 层

源码内联写了：

```text
mma.sync.aligned.m16n8k16...
tcgen05.mma.cta_group::1.kind::f16...
```

PTX 更像 NVIDIA GPU 的虚拟指令集接口。

### 19.3 SASS 层

最终 B300 真正执行的机器指令由 `cuobjdump --dump-sass` 展示。

本次看到：

```text
SM80 PTX mma.sync → SASS HMMA.16816.F32.BF16
SM100 PTX tcgen05.mma → SASS UTCHMMA
```

为什么源码叫 tcgen05，SASS 却是 `UTCHMMA`？

因为 PTX 名称与硬件机器指令 mnemonic 不要求完全相同。关键证据是：

```text
Function : tcgen05_mma_kernel<...>
    UTCHMMA ...
```

它表明该模板 kernel 的最终机器代码中确实存在 SM100 Tensor Core 指令。

---

## 20. 本次 SASS 证据

M128 tcgen05 kernel：

```text
Function : _Z18tcgen05_mma_kernelILi128EEv...
UTCHMMA ...
```

M128 SM80 kernel：

```text
Function : _Z15sm80_mma_kernelILi128EEv...
HMMA.16816.F32.BF16 ...
```

M16 tcgen05 kernel：

```text
Function : _Z18tcgen05_mma_kernelILi16EEv...
UTCHMMA ...
```

M16 SM80 kernel：

```text
Function : _Z15sm80_mma_kernelILi16EEv...
HMMA.16816.F32.BF16 ...
```

`ILi128E` 和 `ILi16E` 是 C++ 模板实例化后的 name mangling，分别对应：

```text
UsefulM=128
UsefulM=16
```

证据文件：

```text
remote_b300_results/challenge/logs/sass_functions_and_instructions.txt
remote_b300_results/challenge/logs/sass_tcgen05_evidence.txt
```

完整 SASS：

```text
remote_b300_results/challenge/experiments_snapshot/bin/instruction_only_mma.sass
```

---

## 21. 本次生成的目录结构

### 21.1 远端实验运行后

```text
~/lcpu2026/assignment02/team/c1_flashkda/
├── FlashKDA-upstream/
├── experiments/
│   ├── instruction_only_mma.cu
│   ├── Makefile
│   ├── README.md
│   ├── run_b300.sh
│   ├── bin/
│   │   ├── instruction_only_mma
│   │   └── instruction_only_mma.sass
│   └── results/
│       └── instruction_only_103a_20260911_024535.txt
└── repro_results/
    └── challenge/
        ├── build_103a.log
        ├── environment.txt
        ├── export_file_list.txt
        ├── full_run_seed42.log
        ├── sass_functions_and_instructions.txt
        └── sass_tcgen05_evidence.txt
```

### 21.2 拉回本机后的证据快照

```text
assignment02/team/c1_flashkda/remote_b300_results/challenge/
├── experiments_snapshot/
│   ├── instruction_only_mma.cu
│   ├── Makefile
│   ├── README.md
│   ├── run_b300.sh
│   ├── bin/
│   │   ├── instruction_only_mma
│   │   └── instruction_only_mma.sass
│   └── results/
│       └── instruction_only_103a_20260911_024535.txt
└── logs/
    ├── build_103a.log
    ├── environment.txt
    ├── export_file_list.txt
    ├── full_run_seed42.log
    ├── sass_functions_and_instructions.txt
    └── sass_tcgen05_evidence.txt
```

各类文件的区别：

| 类型 | 主要文件 | 用途 |
|---|---|---|
| 源码 | `instruction_only_mma.cu` | 说明实验究竟实现了什么 |
| 构建配置 | `Makefile` | 说明如何编译及目标架构 |
| 自动化 | `run_b300.sh` | 说明整次实验执行顺序 |
| 可执行程序 | `bin/instruction_only_mma` | 实际在 B300 上运行的二进制 |
| 完整机器代码 | `instruction_only_mma.sass` | 检查最终 HMMA/UTCHMMA |
| 纯程序结果 | `results/*.txt` | 正确性和性能数据 |
| 完整终端日志 | `full_run_seed42.log` | 编译、运行、导出全链条 |
| 环境信息 | `environment.txt` | GPU、CUDA、哈希和复现条件 |
| 提取证据 | `sass_*.txt` | 答辩时快速展示关键 SASS |

---

## 22. 为什么要记录 SHA-256

`environment.txt` 保存了源码、脚本、二进制和 SASS 的哈希，例如：

```text
e42f...  instruction_only_mma.cu
c8ef...  bin/instruction_only_mma
3d5b...  bin/instruction_only_mma.sass
```

SHA-256 可以理解成文件指纹：

- 文件内容完全相同，哈希应相同；
- 哪怕只改一个字符，哈希通常也会改变；
- 可以证明某份数据对应哪份源代码和二进制。

本机原始 `instruction_only_mma.cu` 与远端导回快照的 SHA-256 相同：

```text
e42f459978154a12a51e2aa70193959ab77e786ab9410471b37554befdcfeb41
```

Makefile 也一致：

```text
a522c1004980c1d6865d34786498c7fcfa0837033524b65718d4049bade626dd
```

因此可以确认：远端运行的是我们本机准备的这份实验源码，而不是另一份同名文件。

---

## 23. 本次真实测试环境

环境日志记录：

```text
GPU: NVIDIA B300 SXM6 AC
compute capability: 10.3
GPU memory: 275040 MiB
driver: 580.126.09
CUDA Toolkit: 13.0
nvcc: V13.0.88
target arch: sm_103a
```

为什么环境信息重要？

因为 tcgen05 是架构相关指令。以下变化都可能影响结果：

- GPU 型号；
- compute capability；
- CUDA Toolkit 版本；
- driver 版本；
- 编译目标 `sm_103a`/`sm_100a`；
- 编译器优化；
- GPU 时钟和其他负载。

所以报告性能数字时必须同时报告环境，不能只写“tcgen05 是 12.3 μs”。

---

## 24. 本次正确性结果

### 24.1 M16 thin

```text
correctness SM80:    PASS, mismatches=0, max_abs=0
correctness tcgen05: PASS, mismatches=0, max_abs=0
```

说明对于被检查的第一个 M16N128 输出：

- SM80 结果与 CPU reference 严格一致；
- tcgen05 结果与 CPU reference 严格一致；
- 因此两条路径也彼此一致。

### 24.2 M128 matched

```text
correctness SM80:    PASS, mismatches=0, max_abs=0
correctness tcgen05: PASS, mismatches=0, max_abs=0
```

说明对于被检查的第一个 M128N128 输出，两条 GPU 路径同样严格正确。

### 24.3 总结果

```text
overall correctness: PASS
```

因此可以进入性能比较。正确性如果失败，性能再快也没有意义。

---

## 25. 本次性能结果详细计算

## 25.1 M16 thin

原始日志：

```text
SM80 mma.sync: 0.006155 ms, useful 5.452 TFLOP/s
SM100 tcgen05: 0.012300 ms, useful 2.728 TFLOP/s, issued 21.824 TFLOP/s
speedup tcgen05/SM80: 0.500x; issued/useful FLOP: 8.0x
```

毫秒换微秒：

```text
SM80    = 0.006155 ms = 6.155 μs
tcgen05 = 0.012300 ms = 12.300 μs
```

加速因子：

$$
6.155 / 12.300 = 0.5004
$$

即约 `0.500×`。

tcgen05 耗时倍数：

$$
12.300 / 6.155 = 1.998
$$

即约为 SM80 的 `2.00 倍`。

延迟增加百分比：

$$
\frac{12.300-6.155}{6.155}\times100\%
=99.84\%
$$

所以也可以说：

> M16 case 中，tcgen05 延迟比 SM80 增加约 99.8%。

## 25.2 M128 matched

原始日志：

```text
SM80 mma.sync: 0.014542 ms, useful 18.459 TFLOP/s
SM100 tcgen05: 0.050765 ms, useful 5.288 TFLOP/s, issued 5.288 TFLOP/s
speedup tcgen05/SM80: 0.286x; issued/useful FLOP: 1.0x
```

换成微秒：

```text
SM80    = 14.542 μs
tcgen05 = 50.765 μs
```

加速因子：

$$
14.542 / 50.765 = 0.2865
$$

即约 `0.286×`。

tcgen05 耗时倍数：

$$
50.765 / 14.542 = 3.491
$$

即约为 SM80 的 `3.49 倍`。

延迟增加百分比：

$$
\frac{50.765-14.542}{14.542}\times100\%
=249.09\%
$$

所以：

> 即使 M128 物理形状自然匹配，当前 tcgen05 完整路径的延迟仍比 SM80 增加约 249%。

---

## 26. 为什么 M128 匹配了反而比 M16 的相对结果更差

不能简单理解成“shape 匹配后一定更快”。

匹配只消除了这一项：

```text
补零导致的无用 Tensor Core 计算
```

它没有消除：

- TMEM 分配和释放；
- mbarrier 初始化和等待；
- shared-memory descriptor 构造；
- A/B swizzle 布局；
- `tcgen05.commit`；
- 从 TMEM 读出全部结果；
- CTA 同步；
- 当前 microbenchmark 实现可能尚未充分流水化这些步骤。

M128 输出量也从 M16 的 2048 个元素增加到 16384 个元素，结果搬运和 TMEM read 路径更重。

因此目前能够下的结论是：

> 在这份“算法不变、完整路径可实际运行”的实现中，M128 shape 匹配仍不足以覆盖 tcgen05 相关固定开销。

不能武断地说：

> 单条 UTCHMMA 硬件指令本身一定比 HMMA 慢 3.49 倍。

因为我们测量的不是脱离所有准备/读回步骤的单指令 latency。

---

## 27. 这个对比哪些地方是公平的

两条路径保持一致的主要变量：

- 同一 GPU；
- 同一个可执行程序；
- 相同 `jobs=512`；
- 相同 warmup 和 iteration；
- 相同随机种子；
- 相同 A/B 输入；
- 相同 BF16 输入；
- 相同 FP32 累加和输出；
- 相同有用数学结果；
- 相同 CUDA event 计时方法；
- 都包含 global→shared 和结果写回。

因此可以公平回答：

> 要把该路径真正替换成 tcgen05，包含它所需 TMEM/同步机制后，端到端 kernel 路径是否更快？

---

## 28. 这个实验不能证明什么

为了答辩严谨，以下边界必须主动说明。

### 28.1 只有一轮性能测量

本次选择只跑一轮，因此数据可以描述为：

```text
single-run measurement
初步实测
```

不能声称：

- 已经测得稳定中位数；
- 已知方差或置信区间；
- 所有时间、所有 B300 都精确是这个数字。

不过 0.500× 和 0.286× 距离 1 很远，足以支持“当前实现没有看到正收益”这个方向性结论。

### 28.2 不是单指令 latency 实验

它测的是完整可用 kernel 路径，不是只发一条 HMMA/UTCHMMA 的裸周期数。

### 28.3 不是完整 FlashKDA 性能

完整 K1/K2 中还有其他计算和访存，因此不能把 `0.286×` 直接乘到整个 FlashKDA forward 上。

### 28.4 CPU 只验证第一个 job

当前对拍覆盖代表性第 0 个 job，不是 512 个 job 的全量对拍。

### 28.5 输入只使用小整数

严格相等非常适合验证布局和公式，但没有覆盖随机大范围 BF16、特殊值或数值误差分布。

### 28.6 tcgen05 实现仍有优化空间

当前代码目标是最小可复现和公平展示必要路径成本，不代表所有可能的 tcgen05 调度、流水线和融合优化都已经穷尽。

---

## 29. 讨论问题 2 的完整回答模板

> FlashKDA 使用 `CHUNK=16`，其 thin GEMM 的有用 M 维只有 16。SM80 `mma.sync.m16n8k16` 在 M 和 K 维上与之自然匹配，而本实验采用的 tcgen05 物理 tile 为 `m128n128k16`。在不改变 CHUNK 和算法的情况下，M16 输入必须补零到 M128，因此 tcgen05 的 physical issued FLOP 是 useful FLOP 的 8 倍。
>
> 我们在 NVIDIA B300（compute capability 10.3）、CUDA 13.0 上实现了两条 BF16 输入、FP32 累加路径，并与 CPU FP32 reference 对拍。M16 和 M128 两个 case 中，SM80 和 tcgen05 路径均得到 `mismatches=0, max_abs=0`。最终 SASS 中，SM80 路径出现 `HMMA.16816.F32.BF16`，tcgen05 路径出现 `UTCHMMA`，证明目标机器指令实际生成。
>
> 单轮 microbenchmark 中，M16N128K16 的 SM80 延迟为 6.155 μs，tcgen05 延迟为 12.300 μs，加速因子仅为 0.500×，tcgen05 约慢 2.00 倍。为排除结论完全由补零造成，我们又测试自然匹配的 M128N128K16；其 issued/useful FLOP 为 1，但 SM80 延迟为 14.542 μs，tcgen05 延迟为 50.765 μs，加速因子为 0.286×，tcgen05 约慢 3.49 倍。
>
> 因此，纸面上 M16 与 M128 的 tile 不匹配确实造成 8 倍无用工作，但它不是唯一限制。即使 shape 匹配，TMEM alloc、descriptor、commit/wait、TMEM load/dealloc 和同步等完整使用路径成本仍然超过当前小 GEMM 可能获得的 Tensor Core 收益。在保持算法与 CHUNK 不变时，机械替换为 tcgen05 没有正收益。

---

## 30. 讨论问题 6 可以怎样利用这个实验

如果我们是 FlashKDA 作者，当前不建议发布一个“只替换指令”的 SM100a v2。

### 支持继续保留 SM80 路径的证据

- SM80 `mma.sync` 能在 B300 正确运行；
- M16 与 SM80 指令 tile 天然匹配；
- M16 tcgen05 产生 8 倍 physical/useful FLOP；
- M128 无补零时仍没有收益；
- 维护 SM100a 专用分支会增加代码、构建、测试和部署成本；
- 当前实测不支持用这些维护成本换性能。

### 为什么不能永久否定 SM100 专版

本次否定的是：

```text
只换指令，不改算法和执行结构
```

以下路线仍可能改变结果：

- 增大 CHUNK 并配套 rescale；
- 多 head 合并进一个 CTA；
- persistent kernel；
- 多 CTA 协作；
- 重新设计 TMEM 生命周期；
- 将相邻操作融合，以摊薄 alloc/commit/load/dealloc；
- 提高每次 tcgen05 发射所覆盖的有用工作比例。

推荐结论：

> 当前不发布机械指令替换版；保留 SM80 通用主路径。只有在更大 CHUNK、并行度重构或跨阶段融合能显著提高 tcgen05 有效利用率并摊薄 TMEM 固定成本时，再考虑独立的 SM100a dispatch。

---

## 31. 从零复现的最短命令清单

下面仅作为将来复现备忘。

### 31.1 本机同步源码到服务器

```bash
ssh b300-vscode \
  'mkdir -p ~/lcpu2026/assignment02/team/c1_flashkda/experiments'

rsync -avh --progress \
  --exclude 'bin/' \
  --exclude '*.sass' \
  /home/wpy/documents/lcpu2026/assignment02/team/c1_flashkda/experiments/ \
  b300-vscode:~/lcpu2026/assignment02/team/c1_flashkda/experiments/
```

最后两个路径末尾的 `/` 表示同步目录里的内容。

### 31.2 登录并申请 GPU

```bash
ssh b300-vscode
srun -G 1 --time 00:15:00 --pty bash
```

### 31.3 检查 GPU

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvcc --version
```

### 31.4 编译、测试、保存日志

```bash
cd ~/lcpu2026/assignment02/team/c1_flashkda/experiments
mkdir -p ../repro_results/challenge
set -o pipefail

ARCH=103a bash run_b300.sh 512 200 30 42 \
  2>&1 | tee ../repro_results/challenge/full_run_seed42.log
```

### 31.5 提取完整 SASS 证据

```bash
grep -nE 'Function :|UTCHMMA|HMMA' \
  bin/instruction_only_mma.sass \
  | tee ../repro_results/challenge/sass_functions_and_instructions.txt
```

### 31.6 拉回本机

```bash
rsync -avh --progress \
  b300-vscode:~/lcpu2026/assignment02/team/c1_flashkda/experiments/ \
  /home/wpy/documents/lcpu2026/assignment02/team/c1_flashkda/remote_b300_results/challenge/experiments_snapshot/

rsync -avh --progress \
  b300-vscode:~/lcpu2026/assignment02/team/c1_flashkda/repro_results/challenge/ \
  /home/wpy/documents/lcpu2026/assignment02/team/c1_flashkda/remote_b300_results/challenge/logs/
```

---

## 32. 答辩时的傻瓜式讲述顺序

不要一上来陷入 PTX 细节。按以下顺序讲最清楚。

### 第一句：提出问题

> FlashKDA 的 CHUNK 是 16，而 SM100 tcgen05 的物理 M tile 更大。我们想验证保持算法不变、只换指令是否有收益。

### 第二句：说明控制变量

> 两条路径使用相同 BF16 输入、FP32 累加，计算完全相同的 `D=A×B^T`。

### 第三句：说明两个 shape

> M16 用来暴露 CHUNK=16 与 M128 tile 的 8 倍浪费；M128 用作无补零的 matched control，排除“所有损失都来自 padding”的单一解释。

### 第四句：先报正确性

> 两个 shape 的两条路径都与 CPU FP32 reference 严格一致，mismatch 和 max absolute error 都为 0。

### 第五句：证明指令真的生成

> SASS 中 baseline 出现 HMMA，tcgen05 kernel 出现 UTCHMMA，因此不是只改了源码名字。

### 第六句：报性能

> M16 中 tcgen05 约慢 2 倍；M128 中约慢 3.49 倍。两者都没有正收益。

### 第七句：解释而不过度推断

> M16 受 8 倍无用计算影响；M128 没有 padding 仍慢，说明 TMEM 和同步等完整路径固定成本也很关键。这是完整 kernel 路径对比，不是单条指令裸 latency。

### 第八句：给工程结论

> 所以当前不值得发布机械指令替换版 SM100 kernel；如果未来做大 CHUNK、persistent、多 head CTA 或融合来摊薄 TMEM 成本，再重新评估。

---

## 33. 老师可能追问的问题

### Q1：为什么不用完整 FlashKDA 直接替换？

因为 challenge 选择的是题目允许的“只换指令不动算法”切面。先用 microbenchmark 控制变量，能把 tile 匹配和指令路径成本单独量化。完整内核重构会同时改变调度、寄存器、访存和融合，难以判断收益来自哪里。

### Q2：为什么需要 M128？

M16 tcgen05 有 8 倍 padding。如果没有 M128 control，就不能区分“tile mismatch”和“tcgen05 完整路径固定成本”。M128 消除了 padding，但仍然更慢，因此结论更扎实。

### Q3：为什么正确性是 exact match？BF16 不应该有误差吗？

输入采用 -2 到 2 的可精确表示小整数，两条 GPU 路径和 CPU 都做 BF16 输入、FP32 累加，因此这个小范围测试可以得到严格相等。它主要验证数据映射和算法未改变，而完整数值精度由另外的 FP64 reference 测试回答。

### Q4：21.824 TFLOP/s 不是比 5.452 更高吗？

21.824 是 physical issued TFLOP/s，其中包含 M16 补到 M128 后的 8 倍工作。真正有用吞吐只有 2.728 TFLOP/s，低于 SM80 的 5.452 TFLOP/s。

### Q5：为什么最终 SASS 不是 `TCGEN05`？

`tcgen05.mma` 是 PTX 名称，B300 SASS 反汇编显示对应硬件指令为 `UTCHMMA`。它位于 `tcgen05_mma_kernel` 函数内，所以构成有效证据。

### Q6：是不是证明 tcgen05 本身很差？

不是。这里只证明对当前小 GEMM 和当前完整使用路径，机械替换没有收益。tcgen05 可能在更大 tile、更高复用、更多融合和更好流水化的场景中有优势。

### Q7：单轮数据够吗？

它足以完成方向性 microbenchmark 和 challenge 原型，但不够描述稳定中位数、方差和置信区间。报告必须注明 single-run measurement。如果未来有时间，应补至少五轮并报告 median/min/max。

### Q8：6.155 μs 是一个 GEMM 的时间吗？

不是。它是一次 kernel launch 并行处理 512 个独立 job 的平均 launch 时间。不能简单把它理解成单个 GEMM 的串行 latency。

### Q9：为什么不直接拿 issued TFLOP/s 比？

因为算法只需要 useful M 行。补零计算不会改善 FlashKDA 输出，工程决策应该看有用吞吐和最终延迟。

### Q10：这个结果怎样支持官方停在 SM80？

SM80 tile 与 CHUNK=16 更匹配，并且在 B300 上仍然能够执行。本次直接替换 tcgen05 不仅没有加速，在两个 case 都更慢；因此在没有算法级重构前，保留成熟 SM80 路径具有更好的性能、可移植性和维护成本平衡。

---

## 34. 后续有时间可以怎样扩展

### 34.1 最低成本增强

- 重复五轮，报告 median/min/max；
- 对多个 seed 做全量 job 正确性；
- 增加随机 BF16 浮点和不同数值尺度；
- 用 NCU 比较 duration、occupancy、stall、Tensor pipe 和 shared-memory 指标。

### 34.2 大 CHUNK + rescale

目标是让逻辑 M 更接近 tcgen05 的物理 M128，提高 useful/issued 比例。

但必须重新处理：

- BF16 数值范围；
- decay 连乘下溢；
- Neumann inverse 的阶数和代价；
- chunk 间 rescale；
- shared memory、register 和 occupancy。

### 34.3 并行度重构

可能方向：

- 多 head 放进同一个 CTA；
- persistent kernel；
- 2-CTA 协作；
- 将多个小 GEMM 聚合后再使用 tcgen05；
- 让一次 TMEM 分配服务更多工作。

这些路线的共同目标是：

> 不再让 tcgen05 为一次很小的有用计算单独支付全部固定成本。

---

## 35. 最后用一句话记住整个 challenge

> 我们没有假设“新指令一定更快”，而是用相同数学问题、CPU 正确性对拍、M16/M128 控制实验、CUDA event 计时和最终 SASS 证据进行验证；结果表明，在 B300 上保持 FlashKDA 的 CHUNK 和算法不变时，直接把 SM80 MMA 换成 tcgen05 不但没有收益，反而在 M16 和 M128 case 中分别约慢 2.00 倍和 3.49 倍。

