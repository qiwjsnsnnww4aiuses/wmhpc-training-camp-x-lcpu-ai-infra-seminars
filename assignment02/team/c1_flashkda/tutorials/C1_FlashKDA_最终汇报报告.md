# C1 FlashKDA：官方 kernel 为什么停在 SM80 MMA

> 汇报版最终报告：复现、六个讨论问题、SM100 tcgen05 挑战与结论
>
> 实验平台：NVIDIA B300 SXM6 AC，CUDA 13.0，PyTorch 2.14.0+cu130

---

## 阅读导航

- 第 2～3 节：先建立 KDA、K1/K2、Python 包、动态库和 CUDA 源码的完整思路链；
- 第 4 节：B300 复现和官方 benchmark；
- 第 5～10 节：逐题回答讨论问题 1～6；
- 第 11 节：真实 K2 tcgen05 挑战设计、正确性、性能和 SASS；
- 第 13～14 节：可直接用于 presentation 的页序与答辩追问。

---

## 0. 一分钟结论

本项目要回答的问题不是“tcgen05 是否比 `mma.sync` 新”，而是：

> 对 FlashKDA 这个 `CHUNK=16`、小矩阵密集、chunk 间带状态依赖的具体负载，直接把 SM80 `mma.sync` 换成 SM100 `tcgen05` 是否划算？

我们的结论是：**不划算。官方保留 SM80 MMA 是合理的。**

证据分为四层：

1. **源码和 SASS**：原版在 B300 上仍执行 `HMMA.16816.F32.BF16`，没有因为运行在 B300 上就自动变成 tcgen05。
2. **形状推算**：KDA 的核心逻辑行数大量是 16，而本实验 tcgen05 tile 的物理 M 为 128。逻辑 M=16 时要补 112 行零，单次矩阵乘只有 1/8 的行有用。
3. **独立 microbench**：M=16 时，tcgen05 为 0.012300 ms，SM80 MMA 为 0.006155 ms；tcgen05 只有原版的 0.500 倍速度。
4. **完整 FlashKDA K2 实验**：输出与状态均通过对拍，但端到端延迟从 1.755060 ms 增加到 17.552482 ms，改版约慢 10.001 倍。

这不是 tcgen05 峰值能力不足，而是**计算粒度与算法粒度不匹配**：

- KDA 为了数值稳定和低代价求逆选择 `CHUNK=16`；
- SM80 `m16n8k16` 恰好能从 16 行开始工作；
- tcgen05 路径为了做 16 行工作，需要准备 128 行 tile、shared-memory descriptor、TMEM、mbarrier、commit/wait；
- K2 的不同 chunk 又必须按顺序更新状态，不能用无限并行隐藏这些开销。

因此，**v2 不应该发布“直接替换指令”的 sm100a 专版**。若继续研究 SM100，必须同时改变算法或并行组织，例如“大 CHUNK + 分段 rescale”或多序列/多 head 的调度重构，而不是机械换指令。

---

## 1. 任务与验收范围

题目要求三层工作：

| 层次 | 要求 | 本项目完成情况 |
|---|---|---|
| 复现 | B300 安装、官方 benchmark、NCU/SASS 确认主路径 | 已完成 fixed/varlen benchmark、NCU 和 SASS 检查 |
| 分析 | 六个讨论点逐个给出结论和证据 | 本报告第 5～10 节逐题回答 |
| 挑战 | 选择一个 SM100 切面修改；正确性对参考，性能对原版 | 已完成“只换指令不动算法”的 K2 tcgen05 实验 |
| 交付 | 代码、报告、答辩 | 代码在 `FInal/`，数据在 `FInal/results/`，本文件为汇报报告 |

必须先说明挑战范围：

- 我们替换的是 **K2 recurrence kernel 中的矩阵乘路径**；
- KDA 算法、`CHUNK=16`、状态递推和 bf16 落盘点都没有改变；
- K1 prepare kernel 仍保留原来的 SM80 MMA；
- 因此不能声称“整个 FlashKDA 已完全移除 HMMA”，只能声称“挑战切面中的 K2 GEMM 已由 tcgen05 执行”。

题目允许 SM100 路线的“任一切面”，所以该范围符合挑战要求；同时它又足够接近真实 kernel，不是只做孤立矩阵乘的玩具实验。

---

## 2. KDA 与 FlashKDA 到底在算什么

### 2.1 KDA 的状态递推

可以把每个 head 的历史压缩成一个 `D × D` 状态矩阵 `S`。对第 t 个 token，概念上的递推为：

```text
S_t = decay_t * S_(t-1) + update_t
o_t = q_t * S_t
```

其中：

- `q`、`k` 用于读取和修改状态；
- `v` 提供写入的值；
- `g` 控制旧状态的衰减；
- `beta` 控制 delta update 强度；
- 输出和未来 token 都依赖更新后的状态。

因此 8192 个 token **不是彼此完全独立**。FlashKDA 的办法不是忽略依赖，而是把序列切成长度 16 的 chunk：

```text
chunk 0 --> S_1 --> chunk 1 --> S_2 --> chunk 2 --> ...
```

同一个序列、同一个 head 的 chunk 状态链必须按顺序执行。但以下维度仍可并行：

- 不同序列；
- 不同 head；
- K1 中不同 chunk 的局部预处理；
- chunk 内的矩阵和 token 计算。

### 2.2 为什么有 K1 和 K2

FlashKDA 把 forward 分为两个 CUDA kernel：

```text
输入 q/k/v/g/beta
        |
        v
K1: _flash_kda_fwd_prepare
    每个 chunk 的局部量可并行计算
        |
        v
workspace
        |
        v
K2: _flash_kda_fwd_recurrence
    读取旧 state，逐 chunk 更新 state，并生成 output
        |
        +--> output
        +--> final_state
```

K1 的 grid 近似为“序列 × head × chunk”，并行块非常多。K2 的状态链则近似为“序列 × head”，每个 CTA 内按 chunk 循环。

这解释了后面的 NCU 现象：K1 很容易把 GPU 填满，fixed 场景的 K2 只有 96 个 CTA，B300 明显吃不饱。

---

## 3. 项目目录和每个部件的职责

```text
c1_flashkda/
├── TASK.md                         # 题目原文和六个讨论点
├── FlashKDA/                       # 官方/题目提供的原版快照
│   ├── flash_kda/__init__.py       # 原版 Python API
│   ├── csrc/
│   │   ├── flash_kda.cpp           # PyTorch C++/Python 绑定和参数检查
│   │   ├── fwd.h                   # C++ forward 接口声明
│   │   └── smxx/
│   │       ├── fwd_launch.cu       # 选择模板并启动 K1/K2
│   │       ├── fwd_kernel1.cuh     # K1 prepare kernel
│   │       ├── fwd_kernel2.cuh     # K2 recurrence kernel
│   │       └── utils.cuh           # TMA/MMA/布局等辅助代码
│   ├── setup.py                    # 把 CUDA/C++ 编译成 Python 扩展
│   ├── tests/                      # 官方正确性测试
│   └── benchmarks/                 # benchmark 与 NCU 模板
├── fla_kda_ref/
│   ├── naive.py                    # 纯 PyTorch 朴素语义参考
│   └── chunk.py                    # Triton chunk_kda 参考入口
├── experiments/                    # 独立 MMA 指令 microbench
│   ├── instruction_only_mma.cu
│   └── run_b300.sh
└── FInal/
    ├── FlashKDA-tcgen05/           # 在完整 FlashKDA 源码树中修改的版本
    │   ├── flash_kda/              # 原版包入口，用于同进程基线
    │   ├── flash_kda_tcgen05/      # 改版 Python API
    │   ├── csrc/smxx/
    │   │   ├── fwd_kernel2.cuh     # 接入 tcgen05 的真实 K2
    │   │   └── tcgen05.cuh         # PTX、descriptor、TMEM、mbarrier 封装
    │   ├── validation/
    │   │   ├── validate.py         # 原版/改版/naive/chunk 正确性对拍
    │   │   └── benchmark.py        # 原版与改版完整 fwd 性能对比
    │   ├── setup.py                # 同时构建两个独立扩展
    │   └── run_b300.sh             # 构建、验证、计时、SASS 一键执行
    └── results/                    # B300 实测产物
```

### 3.1 `flash_kda`、`flash_kda_C` 是什么

它们处于不同层：

```text
flash_kda                  Python 包，给用户调用
flash_kda_C                编译生成的原版 C++/CUDA 动态链接库
```

`flash_kda/__init__.py` 会导入 `flash_kda_C` 中由 pybind11 导出的 `fwd`。所以运行 Python 时不是 Python 自己执行 CUDA 代码，而是：

```text
Python 调用 flash_kda.fwd(...)
        |
        v
import flash_kda_C（.so 动态库）
        |
        v
csrc/flash_kda.cpp 中的 C++ fwd
        |
        v
csrc/smxx/fwd_launch.cu 选择并 launch K1、K2
        |
        v
GPU 执行 fwd_kernel1.cuh / fwd_kernel2.cuh
```

`setup.py` 是构建说明书：它告诉 PyTorch extension 系统，要编译哪些 `.cpp/.cu` 文件、使用什么架构参数、最终动态库叫什么。

### 3.2 改版的四个相似名字

| 名字 | 类型 | 作用 |
|---|---|---|
| `flash_kda` | Python 包 | 原版用户接口 |
| `flash_kda_C` | `.so` 扩展 | 原版编译后的 C++/CUDA 实现 |
| `flash_kda_tcgen05` | Python 包 | tcgen05 改版用户接口 |
| `flash_kda_tcgen05_C` | `.so` 扩展 | 改版编译后的 C++/CUDA 实现 |

改两个名字的意义是让原版和改版在同一个 Python 进程中共存。这样 `validation/benchmark.py` 能对完全相同的输入先后调用两个库，避免跨进程、不同输入和不同环境造成不公平比较。

### 3.3 源码、PTX、SASS 的关系

```text
C++/CUDA 源码
    |
    | nvcc
    v
PTX（虚拟 ISA，较接近汇编但还不是 GPU 最终执行码）
    |
    | ptxas / driver JIT
    v
cubin 中的 SASS（该 GPU 架构实际执行的机器指令）
```

源码搜索能说明“程序员写了什么路径”；SASS 能说明“编译器最后生成了什么指令”。

- 原版 SASS 中的 `HMMA.16816.F32.BF16` 是 SM80 风格 MMA 的机器指令证据；
- 改版 SASS 中的 `UTCHMMA` 是 tcgen05 被编码到二进制中的证据；
- 还必须把指令所在的 `Function` 对应到 K2 recurrence，并实际运行正确性与 benchmark，才能排除“死代码虽然编译了但没有执行”的疑问。

---

## 4. B300 复现结果

### 4.1 环境

```text
GPU:        NVIDIA B300 SXM6 AC
显存:       275040 MiB
Compute:    10.3
Driver:     580.126.09
CUDA:       13.0
nvcc:       13.0.88
PyTorch:    2.14.0+cu130
```

### 4.2 官方形状 benchmark

形状均为总 token 数 `T=8192`、`H=96`、`D=128`。

| 场景 | FlashKDA bf16 state | `chunk_kda` | gated delta rule | 对 `chunk_kda` 加速 | 对 gated delta rule 加速 |
|---|---:|---:|---:|---:|---:|
| fixed：1×8192 | 1.0304 ms | 2.5793 ms | 1.2935 ms | 2.503× | 1.255× |
| varlen：6 条不等长 | 0.8598 ms | 2.6187 ms | 1.3533 ms | 3.046× | 1.574× |
| varlen：8×1024 | 0.6998 ms | 2.5774 ms | 1.3229 ms | 3.683× | 1.890× |

这组结果说明两件事：

1. FlashKDA 在 B300 上确实能运行，而且相对 Triton `chunk_kda` 保持明显优势；
2. 总 token 相同，8 条序列反而比 1 条序列快约 `1.0304 / 0.6998 = 1.472×`。原因不是少算了 token，而是 8 条独立状态链为 K2 提供了更多 CTA 并行度。

### 4.3 三种 state 参数为什么时间接近

fixed 实测：

| 模式 | 时间 |
|---|---:|
| bf16 state | 1.0304 ms |
| no state | 1.0284 ms |
| fp32 boundary state | 0.9991 ms |

这里的区别主要是 API 边界：

- `bf16 state`：读取/返回 bf16 状态；
- `no state`：不传初始状态或不要求返回状态；
- `fp32 state`：API 接收/返回 fp32 状态。

必须注意：原版 K2 仍会把状态转为 bf16 的内部表示参与递推，再在边界转换回 fp32。它不是“整个递推始终用 fp32 保存状态”。因此这三项相近不能证明 bf16 与真正 fp32 recurrence 精度或性能相同。

### 4.4 与官方 GB200 数字对照

官方 `BENCHMARK_GB200.md` 的 H=96、D=128 数据为：

| 场景 | GB200 FlashKDA | B300 本次 FlashKDA |
|---|---:|---:|
| fixed | 1.0087 ms | 1.0304 ms |
| varlen 6 | 0.8597 ms | 0.8598 ms |
| varlen 8 | 0.7064 ms | 0.6998 ms |

数值非常接近，说明复现结果处于合理范围。

---

## 5. 讨论问题 1：为什么是 CHUNK=16？32/64 时哪个先破？

### 5.1 结论

三个理由不是彼此独立的巧合，而是共同把选择推向 16：

1. **数值范围先给出硬约束**：不引入 chunk 内 rescale 时，32 已可能让指数因子越出 FP32/BF16 可用范围；
2. **Neumann 求逆代价快速上升**：32 的每 token 求逆计算约为 16 的 5.33 倍，64 约为 26.67 倍；
3. **MMA 形状匹配**：16 正好匹配 SM80 `m16n8k16` 的 M/K 基本粒度，32/64 虽可拆 tile，但没有新指令级优势。

所以 `CHUNK=32/64` 时，**首先破的是最坏情况下的数值范围**；即使加入 rescale 修好数值，求逆和临时存储成本也会显著上升。

### 5.2 bf16/FP32 数值范围推算

门控处理后可按最坏情况近似：

```text
g_t in [-5, 0]
一个 C 长度 chunk 的累计值最坏为 G_C = -5C
```

KDA 在 chunk 内会使用由累计 gate 构造的指数缩放。FP32 和 BF16 具有相同数量级的指数范围：

```text
最小正常数约为 exp(-87.34)
最大有限数约为 exp(+88.72)
```

代入最坏情况：

| CHUNK | 最坏累计 gate | `exp(G_C)` | `exp(-G_C)` | 结论 |
|---:|---:|---:|---:|---|
| 16 | -80 | 1.80e-35 | 5.54e34 | 仍在正常有限范围附近 |
| 32 | -160 | 3.26e-70 | 3.07e69 | 明显下溢/上溢 |
| 64 | -320 | 1.06e-139 | 9.42e138 | 更严重 |

如果每个 token 都取极端 `g=-5`，第 18 个 token 累计为 -90，已经越过约 -87.34 的正常范围。因此 16 是常见 2 的幂中仍能保留安全余量的最大选择。

这不是说 CHUNK=32 数学上永远不能做，而是说必须加新的数值算法，例如每 16 个 token 设置 anchor、分段 rescale，或保存额外的 scale。那已经不再是“只把常量 16 改成 32”。

### 5.3 16×16 Neumann 级数求逆代价

chunk 内需要处理一个严格下三角结构，可通过有限 Neumann 级数求逆：

```text
(I - L)^(-1) = I + L + L^2 + ... + L^(C-1)
```

利用幂次合并，C 为 2 的幂时，矩阵乘次数可近似写成：

```text
GEMM 次数 = 2 * (log2(C) - 1)
每个 C×C GEMM 约 2*C^3 FLOP
```

| C | GEMM 次数 | 每 chunk FLOP | 每 token FLOP | 相对 C=16 |
|---:|---:|---:|---:|---:|
| 16 | 6 | 49,152 | 3,072 | 1.00× |
| 32 | 8 | 524,288 | 16,384 | 5.33× |
| 64 | 10 | 5,242,880 | 81,920 | 26.67× |

临时 `C×C` 矩阵的元素数也分别扩大为：

```text
C=16 -> 256
C=32 -> 1024，4×
C=64 -> 4096，16×
```

### 5.4 总计算和 workspace 的量级

在 `D=128` 时，可用下面的粗略式比较每 token、每 head 的 tensor-core 工作量：

```text
F(C) ≈ 6D^2 + 8CD + 4(log2(C)-1)C^2
```

| CHUNK | 估算 FLOP/token/head | 相对 C=16 |
|---:|---:|---:|
| 16 | 117,760 | 1.00× |
| 32 | 147,456 | 1.25× |
| 64 | 245,760 | 2.09× |

workspace 粗略量级：

```text
bytes/tile ≈ 3*C*D*2 + D*4 + 2*C^2*2
```

| C | bytes/tile | bytes/token |
|---:|---:|---:|
| 16 | 13,824 | 864 |
| 32 | 29,184 | 912 |
| 64 | 66,048 | 1,032 |

每 token workspace 增长不如求逆 FLOP 剧烈，但每 CTA 的 tile/共享内存压力会增加，可能降低 occupancy。

### 5.5 答辩用一句话

> CHUNK=16 首先是无需额外 rescale 时的数值安全上限，其次让 16×16 求逆便宜，最后又恰好落在 SM80 m16 MMA 的甜点形状；32/64 不是不能做，而是要同时改变数值方案、计算量和资源占用。

---

## 6. 讨论问题 2：tcgen05 最小 tile 匹配吗？直接换有没有收益？

### 6.1 结论

**不匹配，直接换没有收益。**

本实验采用的 tcgen05 物理 tile 为 `M=128, N=128, K=16`。FlashKDA K1/K2 中大部分逻辑矩阵乘的 M 是 16。为了不改变算法，只能把 16 行补零到 128 行：

```text
有效行比例 = 16 / 128 = 12.5%
无效行比例 = 87.5%
issued/useful FLOP = 128 / 16 = 8×
```

### 6.2 FlashKDA 每 chunk 的矩阵乘形状

以 `CHUNK=16, D=128` 为例：

| 阶段 | 逻辑矩阵乘 | FLOP/chunk/head | 占逻辑 tensor FLOP | 与 M=128 tcgen05 |
|---|---|---:|---:|---|
| K1 | 两个 `16×128 @ 128×16` | 131,072 | 6.96% | M=16，不匹配 |
| K1 | 6 个 `16×16 @ 16×16` | 49,152 | 2.61% | M=16，不匹配 |
| K2 | k/q 两个 `16×128 @ 128×128` | 1,048,576 | 55.65% | M=16，不匹配 |
| K2 | 两个 `16×16 @ 16×128` | 131,072 | 6.96% | M=16，不匹配 |
| K2 | `128×16 @ 16×128` state delta | 524,288 | 27.83% | M=128，自然匹配 |
| 合计 |  | 1,884,160 | 100% | 约 72.2% 逻辑工作是 M=16 |

只有最后的状态增量 GEMM 天然适配 M=128。其余大部分工作必须 padding 或重组。

### 6.3 microbench：先纸算，再实测

独立指令实验使用完全相同输入比较 SM80 MMA 和 tcgen05，并检查结果逐元素一致。

| 形状 | SM80 MMA | tcgen05 | tcgen05/SM80 速度 | issued/useful |
|---|---:|---:|---:|---:|
| 逻辑 M=16，补到物理 M=128 | 0.006155 ms | 0.012300 ms | 0.500× | 8.0× |
| 自然 M=128 | 0.014542 ms | 0.050765 ms | 0.286× | 1.0× |

两条路径均为：

```text
correctness PASS
mismatches = 0
max_abs = 0
```

该 microbench 不等价于完整 FlashKDA，但证实“只换指令”不会凭新架构名字自然获益。即使自然 M=128，单个很小、强同步的操作也可能被 tcgen05 的准备和等待开销主导。

### 6.4 整合到真实 K2 后的 issued/useful

改版 K2 每 chunk 动态发出 19 次 tcgen05：

```text
k @ state       : K=128 拆成 8 次 K=16
q @ state       : K=128 拆成 8 次 K=16
INV @ U0        : 1 次
Mqk @ U         : 1 次
state delta     : 1 次
合计            : 19 次
```

前 18 次的逻辑 M 都是 16，最后 1 次是 M=128。物理/有效 tensor 工作比例为：

```text
(19 * 128) / (18 * 16 + 128) = 5.846
```

即约只有：

```text
1 / 5.846 = 17.1%
```

的物理行计算对应有效逻辑工作，约 82.9% 是 padding 带来的无效行。

此外还有原版 `mma.sync` 路径没有的开销：

- 把寄存器/原共享内存数据重排到 128B-swizzle shared tile；
- 构造 A/B shared-memory descriptor；
- 分配和释放 TMEM；
- proxy fence；
- mbarrier、commit、wait；
- 从 TMEM 读回结果；
- 破坏原版对两个小 GEMM 的融合和寄存器复用。

### 6.5 答辩用一句话

> tcgen05 的峰值吞吐只在足够大的、能摊薄异步流水固定成本的 tile 上有意义；FlashKDA 的主导逻辑 M=16，而我们的物理 M=128，所以直接替换首先得到的是 8 倍 padding 和同步开销，不是峰值收益。

---

## 7. 讨论问题 3：chunk 间有依赖，并行度还能从哪里来？

### 7.1 结论

状态依赖只禁止“同一序列、同一 head 的相邻 chunk 同时更新同一状态”，并不禁止序列间、head 间和 chunk 内并行。最现实的方向是：

1. 优先增加/调度独立的序列 × head 状态链；
2. 用 persistent CTA 处理任务队列，改善 varlen 长短不均；
3. 谨慎考虑多 head 合 CTA；
4. 2-CTA/cluster 只有在单链内部工作足够大时才可能抵消每 chunk 同步；
5. 真正打破 chunk 串行需要 affine scan 等算法重写，已超出只换指令。

### 7.2 并行度地图

```text
可直接并行
├── 不同 sequence
├── 不同 head
├── K1 的不同 chunk
└── 一个 chunk 内的矩阵 tile / warp 工作

不能直接并行
└── 同 sequence、同 head 的 chunk 0 -> 1 -> 2 状态更新
```

fixed `T=8192, H=96` 时：

```text
每条状态链的 chunk 数 = 8192 / 16 = 512
K2 独立链数量约 = N * H = 1 * 96 = 96
```

varlen 8 条时，K2 的独立任务大约增加到 8×96=768 个 CTA，所以即使 token 总数不变，速度也明显更快。

### 7.3 候选方案和反例

#### 方案 A：一个 CTA 处理多个 head

想法：若单个 head 的工作不足以利用 tcgen05 大 tile，把多个 head 拼到一个物理 tile。

优点：

- 让 M=128 中更多行有用；
- 可减少 padding；
- 可能共享一部分门控或调度开销。

反例/风险：

- 每个 head 的 `128×128` bf16 state 约 32 KiB；多个 head 会快速增加寄存器、共享内存和状态流量；
- CTA 数减少，可能进一步降低 occupancy；
- tensor parallel 后本卡常只有较少 local heads，例如 H=12，更不能随意牺牲 CTA 数；
- head 间数据独立，打包和布局转换本身有成本。

结论：只有能证明 tile 利用率收益大于资源和并行度损失时才值得。

#### 方案 B：persistent kernel + work queue

想法：固定少量常驻 CTA，完成一条状态链后从队列领取下一条，避免 varlen 中短序列 CTA 先退出、长序列拖尾。

优点：

- 对长度差异大的 varlen 更有价值；
- 减少调度尾部效应；
- 可按长度排序或动态领取任务。

反例/风险：

- fixed `N=1` 时只有 H 条链，队列不能凭空创造更多独立依赖链；
- 原版 K2 本身就在 CTA 内循环 chunk，已有 persistent 的一部分特征；
- 原子队列和状态管理有额外开销。

结论：适合 varlen 调度优化，不是 fixed 单序列状态依赖的根本解法。

#### 方案 C：2-CTA / cluster 协作

想法：两个 CTA 协作处理一个 head，例如切分 D 维、一个搬运一个计算，或使用 cluster/TMEM 协作。

优点：

- 单链可使用更多 warp；
- 可能构造更适合 tcgen05 的大 tile；
- 有机会把 TMA、矩阵乘和 epilogue 做流水。

反例/风险：

- 每个 chunk 都处在状态递推关键路径上，cluster barrier 会重复 512 次；
- 两个 CTA 消耗双倍 CTA/SM 资源；
- 若只是把两份独立小 M 拼起来，仍可能没有解决 tile 形状；
- 负载本身不是纯大 GEMM，复杂协作可能被同步压倒。

结论：需要精确 pipeline 设计和 microbench，不应仅凭“2 CTA 并行更多”判断。

#### 方案 D：affine prefix scan

若把每个 chunk 的状态变换写成：

```text
S_out = A_chunk * S_in + B_chunk
```

理论上可以组合 `(A, B)` 并做前缀扫描，从串行 O(number_of_chunks) 深度变成树形深度。

反例/风险：

- 中间 A/B 的尺寸、内存量和组合计算可能巨大；
- 数值误差顺序改变；
- 需要重写算法和前后处理，已不是简单 kernel tuning；
- 对 512 个 chunk 是否能回收 scan 和 materialization 成本需要完整测量。

结论：研究价值最高，但工程风险和改动范围也最大。

### 7.4 用实测验证并行度判断

在 token 总数相同的情况下：

```text
fixed 1×8192 : 1.0304 ms
varlen 8×1024: 0.6998 ms
```

8 条状态链使完整 FlashKDA 快约 1.472×。NCU 中 K2 grid 从 96 增至约 768，SM 利用率、DRAM/L2 利用率和 compute throughput 同时提高。这是“并行度来自独立状态链”最直接的数据证据。

---

## 8. 讨论问题 4：这个负载是 compute-bound 还是 memory-bound？

### 8.1 结论

不能给整个 FlashKDA 贴一个统一标签：

- **K1 是高吞吐的混合型阶段**，compute、L1/L2/DRAM 都较高；
- **fixed K2 既没有打满计算，也没有打满 HBM**，主要是低并行度、依赖延迟、同步和 shared-memory 资源限制，可称为 latency/occupancy-bound；
- varlen 增加独立状态链后，K2 的 compute 和 memory 指标一起上升，进一步证明 fixed K2 的首要问题是 GPU 没被填满。

因此仅用 roofline 的“compute-bound 或 memory-bound”二选一会遗漏真正瓶颈。

### 8.2 应看哪些 NCU metric

至少组合以下指标：

| 类别 | metric/报告项 | 回答的问题 |
|---|---|---|
| 计算 | Compute (SM) Throughput | tensor/SM 管线是否接近饱和 |
| HBM | DRAM Throughput | 是否接近显存带宽上限 |
| Cache | L1/TEX、L2 Throughput | 流量是否主要在片上 cache |
| 并行 | Achieved Occupancy | 实际有多少活跃 warp |
| 调度 | Eligible Warps per Scheduler | 每周期有多少 warp 可发射 |
| 停顿 | No Eligible / warp stall | 是否因依赖、barrier 等无工作可发 |
| 资源 | registers/thread、shared memory/CTA | occupancy 被什么静态资源限制 |
| 规模 | grid size、waves per SM | CTA 数是否足够填满设备 |

不能只看 FLOP/s；也不能只看 DRAM 百分比。一个 kernel 的 Compute 20%、DRAM 19% 并不表示“很均衡”，更可能表示两边都没吃满。

### 8.3 fixed 的 K1 数据

| 指标 | K1 prepare |
|---|---:|
| Compute (SM) throughput | 70.21% |
| Memory throughput | 72.32% |
| DRAM throughput | 58.99% |
| L1/TEX throughput | 68.80% |
| L2 throughput | 72.32% |
| Achieved occupancy | 96.60% |
| Eligible warps/scheduler | 2.27 |
| No eligible | 28.55% |
| Grid | 49,152 CTAs |
| Registers/thread | 32 |
| Dynamic shared/CTA | 21.25 KiB |

解释：K1 有大量 chunk 独立 CTA，occupancy 约 96.6%，compute 和多级 memory 指标都在 60%～72% 左右。它不是明显单一的纯 compute-bound 或纯 HBM-bound，而是一个 GPU 利用较充分的混合阶段。

### 8.4 fixed 的 K2 数据

不同 state 变体的 K2 大体相同。以其中一条为例：

| 指标 | K2 recurrence |
|---|---:|
| Duration | 约 0.72～0.75 ms |
| Compute (SM) throughput | 约 20.6%～21.9% |
| DRAM throughput | 约 18.6%～19.6% |
| L2 throughput | 约 27%～28% |
| Achieved occupancy | 约 9.37% |
| Theoretical occupancy | 18.75% |
| Eligible warps/scheduler | 约 0.34～0.36 |
| No eligible | 约 67.5% |
| Grid | 96 CTAs |
| Registers/thread | 66～74 |
| Dynamic shared/CTA | 98.43 KiB |

解释链：

```text
grid 只有 96
    + 每 CTA 共享内存约 98.43 KiB
    + 同一链有 512 个顺序 chunk
    -> occupancy 只有约 9.4%
    -> 每 scheduler 可发射 warp 只有约 0.35
    -> 约 2/3 周期没有 eligible warp
    -> compute 与 DRAM 都只有约 20%
```

所以 fixed K2 的主要瓶颈是**并行度和依赖延迟**，不是“算力达到 100%”或“HBM 达到 100%”。

### 8.5 varlen 的 K2 数据

| 场景 | Grid | Compute | DRAM | L2 | Achieved occupancy |
|---|---:|---:|---:|---:|---:|
| 6 条序列 | 576 | 约 27.8%～28.9% | 约 25%～26% | 约 36%～38% | 约 16.7%～16.9% |
| 8 条序列 | 768 | 约 38.6%～40.3% | 约 35%～36% | 约 51%～53% | 约 16.8%～16.9% |

独立链变多后，Compute、DRAM、L2 一起增加，duration 由 fixed 的约 0.72～0.75 ms 降到 8 序列时约 0.40 ms。若 fixed 本来是纯 compute-bound 或纯 memory-bound，仅增加 CTA 数不应呈现这种“所有资源利用一起改善”的特征。

### 8.6 与 assignment 4.5 瘦 GEMM 的联系

`in_proj_qkvgfab` 是 KDA 前的输入投影，形状更像规则 GEMM，适合用 roofline/瘦 GEMM 表判断计算或带宽上限。FlashKDA recurrence 则混合了：

- 小 GEMM；
- gate/elementwise；
- TMA/shared memory；
- 读写 state；
- chunk 级同步；
- 跨 chunk 依赖。

因此投影层达到高 tensor-core 利用率，并不意味着 recurrence 也应达到同样吞吐。二者的并行结构不同。

### 8.7 答辩用一句话

> K1 是 compute/cache/memory 都较高的混合型 kernel；fixed K2 的 compute 和 DRAM 都只有约 20%，occupancy 约 9.4%、no-eligible 约 67.5%，所以主要是状态链造成的并行度与延迟瓶颈，不是传统意义上的纯算力或纯 HBM 瓶颈。

---

## 9. 讨论问题 5：bf16 状态的精度怎么验证？现有数据说明什么？

### 9.1 结论

正确方法不是只看一次 `torch.allclose`，而是把误差来源分开：

1. 建立 FP64 或至少真正 FP32 state recurrence 作为 gold；
2. 比较“FP32 state + FP32 update”和“BF16 state + FP32 update”；
3. 覆盖长序列、极端 gate、beta、状态尺度、fixed/varlen 和非 16 倍数尾块；
4. 同时报告 output 与 final state 的相对误差分布，并画误差随 token 长度的增长曲线；
5. 使用多随机种子，检查 NaN/Inf 和尾部百分位。

当前数据已经证明短序列下 bf16 状态误差处于约 0.5% rel-RMSE 量级，但它只是**短序列初步证据**，不能替代 8K/32K 长序列压力测试。

### 9.2 为什么只比较 API 的 bf16/fp32 选项不够

FlashKDA 的 `fp32 state` 选项主要控制输入/输出边界 dtype。内部 K2 仍使用 bf16 状态表示。因此下面两者不能被它直接区分：

```text
A: 每个 chunk 都以 FP32 保存和更新 state
B: 每个 chunk 用 BF16 保存 state，但乘加时 FP32 accumulate
```

题目真正问的是 A 与 B 的误差，而不是“最后返回的 tensor dtype 是 fp32 还是 bf16”。

### 9.3 建议的实验矩阵

#### 参考实现层级

| 版本 | state 存储 | 更新/累加 | 用途 |
|---|---|---|---|
| Gold | FP64 | FP64 | 小尺寸最高精度参考 |
| A | FP32 | FP32 | 可扩展的主要工程参考 |
| B | BF16 | FP32 | 官方策略 |
| C | BF16 | BF16 | 负面对照，观察低精度累加损失 |

核心报告是 B 相对 A/Gold 的误差；C 用来证明 FP32 accumulate 的必要性。

#### 覆盖维度

```text
长度 T:       16, 256, 1024, 8192, 32768
gate:         随机、接近 0、接近 -5、常数、交替极值
beta:         接近 0、接近 1、随机、极端分布
初始 state:   0、小尺度、正常尺度、大尺度
布局:         fixed、varlen、含 1/15/17 等尾块
随机种子:     至少 20 个
```

#### 误差指标

对 output 和 final state 分别报告：

```text
max_abs
mean_abs
relative RMSE = ||test-ref||_2 / ||ref||_2
cosine similarity
P50 / P90 / P99 / P99.9 相对误差
NaN / Inf 数量
误差随 token/chunk 下标的曲线
```

相对 RMSE 比单独 `max_abs` 更稳健，但必须与尾部百分位、NaN/Inf 一起使用，避免少数灾难性点被平均值掩盖。

### 9.4 本次已有实际数据

`validation/validate.py` 使用相同随机输入同时比较：

- 原版 `flash_kda`；
- 改版 `flash_kda_tcgen05`；
- `naive_recurrent_kda`，其状态计算保留 FP32；
- Triton `chunk_kda`。

#### fixed：长度 64，H=2

| 比较 | output rel-RMSE | final state rel-RMSE |
|---|---:|---:|
| 原版 vs naive FP32 | 0.00553685 | 0.00476515 |
| 改版 vs naive FP32 | 0.00553685 | 0.00476515 |
| 原版 vs Triton chunk | 0.00541033 | 0.00497233 |
| 改版 vs 原版 | 0 | 0 |

#### varlen：长度 `[17, 31, 16]`，H=2

| 比较 | output rel-RMSE | final state rel-RMSE |
|---|---:|---:|
| 原版 vs naive FP32 | 0.00588561 | 0.00462527 |
| 改版 vs naive FP32 | 0.00588561 | 0.00462527 |
| 原版 vs Triton chunk | 0.00623464 | 0.00485505 |
| 改版 vs 原版 | 0 | 0 |

这些数据可支持以下有限结论：

- 短序列 fixed/varlen 下，官方 bf16 状态策略相对 FP32 naive 的 output/state rel-RMSE 约 0.46%～0.62%；
- 非 16 倍数尾块 `[17,31,16]` 没有产生明显异常；
- tcgen05 改版没有在原版 bf16 误差之外引入可观测额外误差，本次输入上与原版结果相同。

不能从这些数据推出：

- 8192/32768 token 长度仍稳定；
- 所有 gate/beta 极端值都稳定；
- 所有模型层叠加后精度无影响。

正式发表“bf16 state 经过完整验证”前，仍应补齐长序列、多种子和极端分布实验。答辩时主动说明这个边界，比笼统写“内部测试通过”更可信。

---

## 10. 讨论问题 6：如果我们是作者，v2 出不出 sm100a 专版？

### 10.1 支持发布的论据

1. KDA 在 Kimi K3 中占大量层，若单层只有小幅收益，模型整体也可能受益。
2. B300/GB200 是重要部署平台，利用新架构专属 TMEM、TMA 和 tcgen05 有潜在价值。
3. K2 的 `128×16 @ 16×128` 状态增量天然匹配 M=128 tile，占逻辑 tensor FLOP 约 27.83%。
4. 若能通过多 head/多序列打包让 M=128 的行真正有用，当前 82.9% padding 并非不可改变。
5. 专版可以保留原版 fallback，以 runtime dispatch 方式按架构和形状选择。

### 10.2 反对发布的论据

1. 约 72.2% 的逻辑 tensor FLOP 来自 M=16 路径，与当前 tcgen05 M=128 粒度严重不匹配。
2. 原版 SM80 MMA 在 B300 上已经取得对 Triton 2.50×～3.68× 的加速，没有明显“旧指令导致完全落后”的证据。
3. K2 的 fixed 瓶颈主要是独立 CTA 少和状态依赖，不是 tensor core 峰值不足。只换更强指令解决不了调度问题。
4. tcgen05 需要 TMEM、descriptor、fence、mbarrier、commit/wait 和 swizzle staging，代码复杂度与维护成本明显上升。
5. sm100a 是架构专用路径，需要持续维护编译器、CUDA、CUTLASS 和不同 shape 的兼容性；原版 SM80 路径则能跨 Ampere 以后多代设备。
6. 本次真实 K2 直接替换慢 10.001×，已证明“机械迁移”没有产品价值。

### 10.3 Amdahl 上限

若只优化天然匹配的 state-delta GEMM，它占本节估算 tensor FLOP 的约 27.83%。即使这部分时间被理想地降到 0，tensor 部分的理论上限也只有：

```text
speedup_max = 1 / (1 - 0.2783) = 1.386×
```

而完整 kernel 还有 TMA、elementwise、同步和状态访问，因此端到端上限会更低。这个推算不是说该方向毫无价值，而是提醒：若目标是端到端 20% 以上收益，不能只盯一个自然匹配 GEMM。

### 10.4 发布门槛

只有同时满足以下条件，才建议发布 v2 sm100a 专版：

```text
1. fixed、varlen、尾块、全部支持 dtype/shape 正确性通过；
2. 8K/32K 长序列、多种子、极端 gate/beta 精度通过；
3. H=96 与 tensor-parallel 后 H=12 等真实部署形状都测量；
4. 完整 forward 稳定获得至少 10%～15% 收益，而非单个指令峰值；
5. NCU 证明提升来自 useful work，不是只提高 issued FLOP；
6. 保留原版 SM80 fallback 和可靠 runtime dispatch；
7. 维护、编译和二进制体积成本可接受。
```

### 10.5 最终产品决策

**当前不发布直接替换 K2 的 sm100a 专版。**

保留原版 SM80 路径作为默认实现；若继续投入，立项研究下列二选一：

- “大 CHUNK + 分段 rescale”，从算法上制造更适合 tcgen05 的 tile；
- “多序列/多 head 打包 + persistent 调度”，提高 M 利用率和独立 CTA 数。

只有新方案跨过上述发布门槛，才作为可选 sm100a specialization 合入。这样既保留可移植性，也不给峰值优化关门。

### 10.6 答辩用一句话

> 我们不是因为一次负结果就否定 SM100，而是否定“只换指令”的产品方案：当前瓶颈是形状和并行结构，若不先解决这两点，tcgen05 的峰值能力无法转化为 useful throughput。

---

## 11. 挑战：在真实 K2 中用 tcgen05 替换 SM80 MMA

### 11.1 设计目标

选择题目允许的第一条路线：

```text
只换指令，不动算法
```

保持不变：

- KDA 数学语义；
- `CHUNK=16`；
- K1/K2 两阶段结构；
- K2 的 chunk 顺序；
- bf16 状态落盘和 FP32 accumulate 语义；
- Python API 的输入输出。

改变：

- K2 原来使用 `SM80_16x8x16_F32BF16BF16F32_TN` 的矩阵乘改由 tcgen05 路径完成；
- 逻辑 M=16 的输入补零为物理 M=128；
- 操作数写入符合 tcgen05 要求的 swizzled shared memory；
- 累加结果存入 TMEM，再读回原有数据流。

### 11.2 K2 数据流

```text
K1 生成 workspace（不变）
          |
          v
K2 读取一个 chunk 与旧 state
          |
          +-- 8× tcgen05: k @ state  --> KS
          +-- 8× tcgen05: q @ state  --> QS
          |
          +-- U0 = (v - KS) * sigmoid(beta)
          +-- 1× tcgen05: INV @ U0   --> U
          +-- 1× tcgen05: Mqk @ U    --> output correction
          +-- 1× tcgen05: k^T @ U    --> state delta
          |
          v
state = decay * old_state + delta
          |
          v
下一个 chunk
```

### 11.3 构建与运行链

`run_b300.sh` 的逻辑是：

```text
1. 记录 GPU/CUDA/Python/PyTorch 环境
2. 构建原版 flash_kda_C
3. 构建改版 flash_kda_tcgen05_C
4. 运行 validation/validate.py
5. 运行 validation/benchmark.py
6. 找到改版 .so
7. cuobjdump/nvdisasm 导出 SASS
8. 提取 Function、UTCHMMA、HMMA 关键行
9. 将所有记录写入 FInal/results/
```

完整命令曾在服务器 GPU allocation 中执行：

```bash
source ~/lcpu2026/assignment02/.venv/bin/activate
cd ~/lcpu2026/assignment02/team/c1_flashkda/FInal
bash prepare_on_server.sh
cd FlashKDA-tcgen05
bash run_b300.sh ../../fla_kda_ref 2>&1 | tee ../results/full_run.log
```

### 11.4 正确性为什么要四方对拍

```text
原版 FlashKDA
    vs 改版 FlashKDA tcgen05
        -> 隔离“换指令”是否改变结果

原版/改版
    vs naive.py
        -> 检查是否符合 KDA 数学语义

原版/改版
    vs Triton chunk_kda
        -> 检查是否与题目指定工程参考一致
```

只做“改版 vs 原版”不够，因为二者可能共享同一个算法 bug；只做“改版 vs naive”也不够，因为无法分离原版既有 bf16 误差和改版新增误差。

测试覆盖：

- fixed：长度 64；
- varlen：`[17,31,16]`；
- output 与 final state；
- 非 16 倍数尾块。

结果：所有 case PASS，改版与原版本次为 0 差异；相对 naive/chunk 的 rel-RMSE 约 0.46%～0.62%。

### 11.5 完整 forward 性能结果

形状：`T=8192, H=96, D=128`，同进程、同输入、同 B300、成对计时。

| 实现 | 平均延迟 | 最小 | 最大 |
|---|---:|---:|---:|
| 原版 SM80 K2 | 1.755060 ms | 1.749056 ms | 1.767328 ms |
| tcgen05 K2 | 17.552482 ms | 17.512993 ms | 17.694336 ms |

```text
原版 / 改版 = 0.1000×
改版 / 原版 = 10.001×
延迟增加约 900.1%
```

这里的 1.755 ms 不应直接与第 4 节官方 harness 的约 1.03 ms 比较。两个 harness 的同步和计时方式不同。挑战的公平结论来自**同一个 paired benchmark 内**的原版与改版比值。

### 11.6 SASS 证据

改版二进制中：

```text
K2 recurrence 的 UTCHMMA 静态指令：70
    = 14 个模板实例 × 5 个静态 tcgen05 call site

K1 prepare 的 HMMA 静态指令：88
```

为什么静态只有 5 个 call site，而动态每 chunk 是 19 次？

- 源码中 k@state 和 q@state 各有一个静态 call site；
- 运行时它们分别在 K=128 循环中执行 8 次；
- 其余三个静态 call site各执行 1 次；
- 因此动态为 `8 + 8 + 1 + 1 + 1 = 19`。

K1 仍出现 HMMA 是预期行为，不是挑战失败，因为本次只替换 K2。

证据链为：

```text
源码 fwd_kernel2.cuh 调用 tcgen05.cuh
    -> setup.py 编译为 flash_kda_tcgen05_C.so
    -> SASS 在 K2 Function 中出现 UTCHMMA
    -> validation 实际调用改版包并通过
    -> paired benchmark 实际计时改版完整 fwd
```

### 11.7 为什么慢 10 倍，而不只是纸面的 5.846 倍

5.846 只计算物理行与有效行的 FLOP 放大。真实延迟还叠加：

1. 每个小 GEMM 的 shared-memory packing；
2. 128B swizzle 和 descriptor 构造；
3. generic/async proxy fence；
4. TMEM allocation/deallocation；
5. 每次 tcgen05 的 mbarrier commit/wait；
6. TMEM load 回普通寄存器/共享内存；
7. 原版小 tile 寄存器复用与融合被打断；
8. 每个 head 有 512 个串行 chunk，固定开销重复且难以隐藏；
9. 编译日志中部分模板实例存在少量 spill，也可能增加额外访存。

所以结果不是简单的“无效 FLOP 5.846×，延迟也恰好 5.846×”，而是约 10×。

---

## 12. 证据索引：每个结论去哪找

| 结论 | 主要证据文件 |
|---|---|
| 题目要求 | [`TASK.md`](./TASK.md) |
| 官方算法设计 | [`20260420-flashkda-v1-deep-dive.md`](./FlashKDA/docs/20260420-flashkda-v1-deep-dive.md) |
| GB200 官方基线 | [`BENCHMARK_GB200.md`](./FlashKDA/BENCHMARK_GB200.md) |
| B300 fixed NCU | [`b300_fixed_h96_run1_details.txt`](../../../b300_fixed_h96_run1_details.txt) |
| B300 varlen NCU | [`b300_varlen_h96_run1_details.txt`](../../../b300_varlen_h96_run1_details.txt) |
| 独立 MMA microbench | [`instruction_only_mma.cu`](./experiments/instruction_only_mma.cu) 和服务器生成结果 |
| tcgen05 K2 源码 | [`fwd_kernel2.cuh`](./FInal/FlashKDA-tcgen05/csrc/smxx/fwd_kernel2.cuh) |
| tcgen05 PTX helper | [`tcgen05.cuh`](./FInal/FlashKDA-tcgen05/csrc/smxx/tcgen05.cuh) |
| 构建记录 | [`build.log`](./FInal/results/build.log) |
| 正确性数据 | [`correctness.log`](./FInal/results/correctness.log) |
| 成对性能数据 | [`benchmark.log`](./FInal/results/benchmark.log) |
| 完整 SASS | [`full.sass`](./FInal/results/sass/full.sass) |
| SASS 关键指令 | [`key_instructions.txt`](./FInal/results/sass/key_instructions.txt) |
| 挑战完整思路链 | [`FInal_结果分析与完整思路链.md`](./FInal/FInal_结果分析与完整思路链.md) |

注意：表中的相对路径均以 `assignment02/team/c1_flashkda/` 为参照。根目录 NCU 文件实际位于 `lcpu2026/`。

---

## 13. 建议汇报结构（10～12 分钟）

### 第 1 页：问题

```text
新 GPU 上为什么仍用 SM80 mma.sync？
tcgen05 更先进，是否直接替换就更快？
```

### 第 2 页：KDA 数据依赖

画出：

```text
K1 chunk-parallel -> workspace -> K2 state recurrence
```

强调同一 head 的 chunk 有依赖，但 sequence/head 之间可并行。

### 第 3 页：CHUNK=16 的三重约束

- 数值：16 最坏累计 -80 尚可，32 的 -160 越界；
- 求逆：32 每 token 代价约 5.33×；
- 指令：16 匹配 SM80 m16。

### 第 4 页：B300 复现

给出 benchmark 表，突出 FlashKDA 对 `chunk_kda` 2.50×～3.68×。

### 第 5 页：NCU

对比 K1 与 fixed K2：

```text
K1 occupancy 96.6%，compute 70.2%，memory 72.3%
K2 occupancy 9.4%，compute ~21%，DRAM ~19%，no eligible ~67.5%
```

结论：K2 不是峰值算力不够，而是独立工作和 latency hiding 不够。

### 第 6 页：纸面 tile 分析

展示：

```text
逻辑 M=16，物理 M=128
单次只有 12.5% 行有效
K2 整体 physical/useful = 5.846×
```

### 第 7 页：microbench

展示 M16 时 tcgen05 0.500×，自然 M128 时 0.286×，正确性均为 PASS。

### 第 8 页：真实 K2 挑战

展示源码调用链和四方对拍，强调不是纯 GEMM 玩具。

### 第 9 页：最终性能

```text
原版 1.755 ms
改版 17.552 ms
改版慢 10.001×
```

### 第 10 页：决策

```text
不发布机械替换版；
保留 SM80 fallback；
后续只研究“大 CHUNK + rescale”或并行度重构。
```

---

## 14. 答辩可能追问

### Q：你们是不是只模拟了矩阵乘，没有改真实 FlashKDA？

不是。`experiments/` 是先做纸面判断的独立 microbench；最终挑战在完整源码树 `FInal/FlashKDA-tcgen05/` 的真实 K2 recurrence 中替换了五类 GEMM，并运行完整 forward、四方正确性对拍和端到端计时。

### Q：为什么 SASS 里还有 HMMA？

因为挑战只替换 K2，K1 按设计保持原版。K2 Function 中出现 UTCHMMA，K1 Function 中仍出现 HMMA，恰好符合修改边界。

### Q：UTCHMMA 存在就能证明执行了吗？

单独不能。我们还把它定位到 K2 Function，并用 `flash_kda_tcgen05` 包实际通过正确性和成对 benchmark，形成“源码—二进制—运行”的完整证据链。

### Q：为什么 tcgen05 自然 M=128 的 microbench 也更慢？

单次问题很小，tcgen05 的 descriptor、TMEM、commit/wait 等固定成本无法被长流水摊薄。它的优势需要大 tile、多个连续 stage 和足够并行，而不是一次同步后立刻读回的小操作。

### Q：变慢是否说明 tcgen05 本身差？

不说明。只说明它不适合当前映射方式。tcgen05 面向不同的数据搬运、累加和流水模型；要发挥它，需要让更多物理 tile 成为 useful work，并减少每 chunk 的同步边界。

### Q：fixed 和 varlen 怎么证明都测了？

benchmark 输出分别有 fixed shape，以及两组明确的 `seq_lens`；正确性脚本也分别运行 `T=64` fixed 和 `[17,31,16]` varlen。varlen 的 `cu_seqlens` 将拼接 token 划分为独立序列，K2 grid 和 NCU 也随序列数增加。

### Q：状态是前后依赖的，为什么 8192 token 还能并行？

同一序列/同一 head 的 chunk 状态必须顺序更新；并行来自不同 head、不同序列、K1 的 chunk 局部预处理和 chunk 内矩阵计算。并不是把同一状态链的 8192 token 全部打散。

### Q：bf16 精度真的验证完了吗？

短序列 fixed/varlen 已有实际数据，rel-RMSE 约 0.46%～0.62%，且无 tcgen05 额外误差。但长序列、多种子、极端 gate/beta 仍是待补的完整压力测试。报告明确区分了“已有证据”和“尚未覆盖”，没有把短测试夸大成全场景保证。

### Q：做不出正收益，挑战算完成吗？

算。题目明确允许负收益，前提是正确性、真实性和量化论证完整。本项目不仅得到负结果，还解释了 padding、动态指令数、同步、TMEM 和状态依赖如何共同导致负收益，从而补全了“官方为什么停在 SM80”的论证。

---

## 15. 最终结论

本项目完成了从猜想到证据的完整闭环：

```text
观察：B300 上原版仍使用 SM80 MMA
  -> 分析：CHUNK=16 由数值、求逆和 m16 tile 共同决定
  -> 推算：tcgen05 M128 会造成大面积 padding
  -> microbench：直接替换没有收益
  -> NCU：真实 K2 的首要问题是并行度/延迟，不是峰值算力
  -> 真实改版：K2 tcgen05 正确，但完整 forward 慢约 10×
  -> 产品决策：保留 SM80；除非同时重构 chunk/rescale 或并行组织
```

因此对题目“为什么官方 kernel 停在 SM80 MMA”的最终回答是：

> SM80 MMA 虽然名字属于较早架构，但它的 `m16n8k16` 粒度与 FlashKDA 的 `CHUNK=16` 天然匹配，寄存器级小 GEMM 的固定开销也低。SM100 tcgen05 的峰值能力建立在更大 tile、TMEM 和异步流水上；在当前算法中，大部分逻辑 M=16，K2 又受状态依赖和低 CTA 数限制。直接替换既产生约 5.846 倍物理/有效工作比，又重复支付 staging 和同步成本，最终 B300 实测慢 10.001 倍。官方当前保留 SM80 路径，是针对该负载形状和依赖结构的合理工程选择，而不是遗漏了新指令。
