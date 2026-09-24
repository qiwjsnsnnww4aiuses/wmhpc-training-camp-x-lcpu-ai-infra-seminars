# M4：完整 GEMM、TMA 与多级 Pipeline——知识点与题解

> 参考材料：Session 03 的 SM100/tcgen05 部分、Session 04《Pipeline Ordering》、`handout/src/assignment02.md` 和 `cuda/m4_gemm/` 中的题目文件。  
> 覆盖范围：4.1、4.2、4.3、4.5；按要求不展开、不完成 4.4。  
> 文中会给出纸面问题的结论与理由。TFLOPS、耗时、达成率和 NCU 指标必须来自你实际占满的 B300，本文不会虚构实验数字。

## 0. 一句话理解 M4

M3 教你“让一个固定 tile 在 Tensor Core 上算对”；M4 教你“让成千上万个 tile 持续不断地喂满 Tensor Core”。

```text
M3：一辆车能发动、能跑、能刹车
M4：修一条高速公路，让装货、运输、卸货形成稳定流水
```

M4 的三层优化梯子：

```text
4.1 tiled
  手工 GMEM→寄存器→SMEM，算完一批再搬下一批
        │ 减少 CUDA Core 的搬运和地址计算
        ▼
4.2 TMA
  一个线程发起 GMEM→SMEM bulk tensor copy，但仍是单缓冲串行
        │ 增加多个 SMEM stage，把下一批搬运与当前批计算重叠
        ▼
4.3 multistage pipeline
  TMA 与 tcgen05 并行，full/empty barrier 管理每个 stage 的所有权
```

4.5 则故意离开“漂亮的方阵”，回答另一个现实问题：

> 当 M 很小或 K 很短时，Tensor Core 的峰值再高，也可能因为形状、延迟和固定开销而帮不上忙。

---

## 1. M4 与 M0–M3 的联系

### 1.1 M0：峰值和 Roofline 是性能判断的尺子

M0 给出两个关键硬件上限：

- Tensor Core 峰值计算吞吐 (P_{peak})，单位 FLOP/s；
- HBM 峰值带宽 (B_{peak})，单位 byte/s。

机器平衡点：

\[
\beta = \frac{P_{peak}}{B_{peak}}\quad \text{FLOP/byte}
\]

对一个 kernel：

- 若算术强度 (AI < \beta)，Roofline 预测更容易受内存带宽限制；
- 若 (AI > \beta)，Roofline 允许它进入计算受限区；
- 但“允许”不等于“实际达到”。指令开销、同步、tile 形状、并行度不足都能让两个 roof 的达成率同时很低。

M4.1、M4.5 都要用这把尺子。

### 1.2 M1：Tensor Core 不是自动变快

M1 说明：

- MMA 有固定 shape；
- 操作数布局必须匹配硬件；
- 单条 MMA 计算强度不代表整个 kernel 的计算强度；
- 数据供给跟不上时，Tensor Core 会饿住。

M4 的重点正是为 Tensor Core 建立持续的数据供给。

### 1.3 M2：descriptor 与 swizzle 成为 M4 的输入契约

M2 的两个组件在 M4 继续使用：

- 64-bit SMEM matrix descriptor：告诉 `tcgen05.mma` A/B 在 SMEM 如何解释；
- 128B swizzle：改变 SMEM bank placement，减少 Tensor Core 取数冲突。

4.1 由 CUDA 线程手工按照 swizzle 公式写 SMEM；4.2/4.3 改由 TMA 自动落成同样的 swizzled 布局。消费者 descriptor 不需要改变。

### 1.4 M3：M4 沿用完整的 tcgen05 生命周期

M3.2 已经具备：

```text
TMEM alloc
→ A/B 进入 SMEM
→ tcgen05.mma
→ commit 到 mbarrier
→ wait
→ tcgen05.ld
→ wait::ld
→ TMEM dealloc
```

M4 新增的并不是另一套 Tensor Core 指令，而是：

1. grid 中每个 CTA 认领不同输出 tile；
2. K 远大于 BK，需要循环加载和累加；
3. 用 TMA 替换手工 staging；
4. 用多级 SMEM 缓冲把搬运与计算重叠。

### 1.5 M3.3 的 phase/parity 在 M4 中升级

M3.3 只有一个反复复用的 barrier。M4.3 中每个 stage 都有自己的 full/empty barrier，而且每个 barrier 都跨多个 generation 复用。

```text
M3.3：一个地址，多轮 phase
M4.3：多个 stage 地址 × 每个地址多轮 phase
```

如果 M3.3 的 parity 没真正理解，M4.3 会非常容易死锁。

### 1.6 M3.4 的 2-CTA 为什么在 M4 被提及

M3.4 说明 CTA pair 可以减少重复的 B staging。M4.3 中每增加一个 pipeline stage，都要再存一份 A/B tile，因此 B 的 SMEM 节省可以换成：

- 更多 stages；
- 更大的 tile；
- 更好的资源余量。

这就是 4.3(c) 要你联系 3.4 的原因，并不要求你完成 4.4 的 2-CTA pipeline。

---

## 2. Session 4 的物理直觉：为什么需要 Pipeline

### 2.1 Little's Law：高吞吐需要足够多的在途工作

Session 4 用：

\[
L = \lambda W
\]

解释延迟隐藏：

- (W)：一次 HBM/TMA 操作的长延迟；
- ​\(\lambda\)：希望维持的吞吐率；
- (L)：系统中必须同时处于 in-flight 状态的工作量。

如果只允许一份 copy 在飞：

```text
copy k → 等待 → compute k → copy k+1 → 等待 → compute k+1
```

Tensor Core 计算得越快，反而越容易在下一次数据尚未到达时闲置。要维持高吞吐，必须让更多未来 tile 提前进入流水。

### 2.2 GEMM 的两个核心手段：reuse + overlap

GEMM 快的根本不只是使用 Tensor Core，而是：

- **reuse**：A/B tile 搬到片上后被多次乘加；
- **overlap**：下一 tile 的数据移动与当前 tile 的计算并行。

```text
只复用、不重叠：少搬数据，但 Tensor Core 仍会等
只重叠、不复用：在搬很多数据，HBM 压力仍然巨大
复用 + 重叠：片上 tile 多次使用，同时提前准备下一 tile
```

### 2.3 Active、Eligible、Issued

Session 4 强调：一个 warp 驻留在 SM 上，只能说明它是 active；它的下一条指令没有依赖阻塞时才是 eligible；调度器本周期选中它，才是 issued。

```text
active resident warp
      │ 下一条指令是否 ready？
      ▼
eligible warp
      │ 本周期是否被 scheduler 选中？
      ▼
issued instruction
```

所以：

- stalled warp 是正常现象；
- 真正损失是某周期没有任何 eligible warp，出现 empty issue slot；
- occupancy 高不保证性能高，它只增加潜在的 TLP；
- 更深 pipeline 增加 MLP，但会消耗 SMEM，可能降低 occupancy/TLP。

### 2.4 ILP、MLP、TLP

延迟隐藏有三种来源：

| 来源 | 含义 | M4 中的例子 |
|---|---|---|
| ILP | 同一个 warp 内有独立指令链 | 发射控制、地址准备与其他工作交错 |
| MLP | 同时存在更多内存请求 | 多个 TMA stage 在飞 |
| TLP | 其他 resident warp/block 可运行 | 同一 SM 上切换到别的 CTA |

M4.3 的多级 stage 主要增加 MLP；大 grid 还能靠不同 CTA 提供 TLP。两个实验形状对 stages 的敏感度不同，核心就在这里。

### 2.5 Pipeline ordering 的四个问题

每看见一个同步指令，都问它究竟回答哪一题：

| 问题 | 含义 | M4 对应机制 |
|---|---|---|
| Completion | 异步操作完成了吗？ | TMA full mbarrier、MMA empty mbarrier |
| Visibility | 消费者能看见数据了吗？ | barrier acquire、tcgen05 fence |
| Ordering | 不同 agent/proxy 的访问排好序了吗？ | proxy 规则、phase/token |
| Ownership | 这个存储槽能被下一生产者覆盖了吗？ | empty barrier / stage release |

很多错误来自“看见一个 wait 就以为四件事全解决了”。实际上，一个 wait 通常只承担其中一部分语义。

---

## 3. M4 固定形状与资源账本

M4.1–M4.3 固定：

```text
BM = 128
BN = 64
BK = 64
输入 = bf16
累加 = f32
每 CTA = 128 threads = 4 warps
```

### 3.1 一个 K-stage 的 SMEM

A stage：

\[
128\times64\times2=16384\text{ B}=16\text{ KiB}
\]

B stage：

\[
64\times64\times2=8192\text{ B}=8\text{ KiB}
\]

合计：

\[
(BM+BN)\times BK\times2=24576\text{ B}=24\text{ KiB/stage}
\]

### 3.2 TMEM accumulator

\[
BM\times BN\times4
=128\times64\times4
=32768\text{ B}=32\text{ KiB}
\]

等价于使用 64 个 TMEM column，覆盖全部 128 lane。

### 3.3 每个 BK=64 内有几条 MMA

bf16 属于 `kind::f16`，dense K 步长为16：

\[
BK/16=64/16=4
\]

所以每个 K tile 发四条 k16 MMA。

### 3.4 4096³ 的 grid 和 K 循环

```text
grid.x = M/BM = 4096/128 = 32
grid.y = N/BN = 4096/64  = 64
CTA 数 = 32×64 = 2048
K iterations = K/BK = 4096/64 = 64
```

每个 CTA 负责唯一的 `128×64` 输出 tile，并在同一块 TMEM accumulator 上累加64轮。

### 3.5 动态 SMEM 为什么多给1024 B

程序获得 `smem_raw` 后，把实际起点向上对齐到1024-byte swizzle atom 边界。最坏会浪费接近1023 B，因此 host 分配：

```text
实际数据容量 + 1024 B 对齐余量
```

不是说每个 stage 额外需要一个完整的1 KiB 数据区。

---

## 4. 4.1：从单 tile 扩展到完整 Tiled GEMM

### 4.1.1 数学和存储约定

程序中的 A、B 分别按：

```text
A[M, K]
B[N, K]
```

保存，输出：

\[
D[M,N]=A[M,K]\,B[N,K]^T
\]

某个 CTA 的坐标：

```text
tileM = blockIdx.x × BM
tileN = blockIdx.y × BN
```

它计算：

```text
D[tileM:tileM+BM, tileN:tileN+BN]
```

### 4.1.2 K 维分块

完整 K 不会一次放进 SMEM。每轮 `it` 搬：

```text
A[tileM : tileM+BM, it×BK : (it+1)×BK]
B[tileN : tileN+BN, it×BK : (it+1)×BK]
```

每轮内部再拆成四个 k16 MMA：

```text
global K offset = it×BK + kk
kk ∈ {0,16,32,48}
```

### 4.1.3 全局地址到 swizzled SMEM

A 的逻辑元素：

```text
gA[(tileM + row) × K + (it×BK + k)]
```

B 的逻辑元素：

```text
gB[(tileN + row) × K + (it×BK + k)]
```

进入 SMEM 时物理地址统一为：

```text
swz128(row, k × 2)
```

注意：全局坐标包含 tile 偏移，SMEM 坐标从本地 row=0、k=0 重新开始。

### 4.1.4 最容易错的累加谓词

M3.2 只有一个 BK=64 tile，所以 `kk>0` 可以判断是不是第一条 MMA。M4 多了一层 `it`，整个 GEMM 只有：

```text
it == 0 且 kk == 0
```

这一条不能读取旧 D。正确的抽象条件：

```text
accumulate = (it != 0) || (kk != 0)
```

若每个 `it` 的 `kk=0` 都关闭累加，前一轮 BK 的结果会被覆盖，最后只剩当前或最后一个 K tile 的贡献。

### 4.1.5 为什么每轮必须等 MMA 消费完 SMEM

4.1 只有一份 A/B SMEM buffer：

```text
iteration it：Tensor Core 正在读 stage
iteration it+1：CUDA threads 想覆盖同一 stage
```

如果不等待本轮 MMA 完成，下一轮的 `st.shared` 会与 Tensor Core 读取发生读写竞争。

所以每轮顺序是：

```text
staging it
→ proxy fence
→ CTA sync
→ MMA it
→ commit
→ mbarrier wait 当前 phase
→ CTA sync/进入下一轮覆盖
```

barrier phase 随 `it` 翻转。单 barrier 的等待 token 可以按迭代 parity 管理。

### 4.1.6 epilogue 如何加入 tile 偏移

M3 的本地行列：

```text
localRow = warp×32 + lane
localCol = 0,8,...,56
```

写回完整矩阵时：

```text
globalRow = tileM + localRow
globalCol = tileN + localCol + i
gD[globalRow × N + globalCol]
```

漏掉 `tileN` 会让不同 y-block 写到同一列范围；漏掉 `tileM` 会让不同 x-block 写到同一行范围。

### 4.1.7 为什么可以与 cuBLAS 逐位严格比较

测试输入是小整数；题面保证累加绝对值不超过 (2^{24})。在该范围内，相关整数可以被 f32 精确表示，乘积和部分和也不发生普通随机浮点数据那种舍入差异。

因此顺序不同仍可严格相等，便于把布局、累加和同步错误直接暴露出来。

### 4.1.8 4.1 题目答案：瓶颈在哪里

结论应写成：

> 4.1 已使用 tcgen05 Tensor Core，但主要瓶颈通常仍在数据供给路径，而不是 Tensor Core 算术能力本身。每个 BK tile 都由128个 CUDA线程执行大量全局 load、整数地址计算、swizzle 计算和 shared store；copy、等待、MMA 又串行进行，Tensor Core 在 staging 阶段没有工作。

手工 staging 的具体成本包括：

1. GMEM load 指令；
2. 数据经普通寄存器中转；
3. global/local 索引计算；
4. swizzle 地址计算；
5. SMEM store 指令；
6. proxy fence 与 CTA barrier；
7. 无 copy/compute overlap 带来的空窗。

如何用实验支撑：

- 将 4.1 TFLOPS 与 M0 的 Tensor Core 峰值比较；
- 看 NCU 中 Tensor Core 活跃度是否远低于可用周期；
- 看 LSU/地址计算/shared-store 指令和 warp stall；
- 比较 4.2：数学与 MMA 不变，只替换 staging；若性能明显提升，差值就是强证据。

不要只凭完整 GEMM 的理论 AI 宣称 compute-bound。Roofline 是上限；手工 staging 的指令瓶颈和序列化也可能使实际性能远离两个 roof。

---

## 5. 4.2：用 TMA 替换手工 Staging

### 5.1 TMA 解决什么

4.1 的路径：

```text
GMEM → 普通寄存器 → SMEM
       每个线程做多次 load/store 和地址计算
```

TMA 路径：

```text
一个 elected thread 发起 bulk tensor transfer
GMEM ─────────────── TMA hardware ───────────────→ SMEM
                       地址生成 + 搬运 + swizzle
```

TMA 的价值不只是“异步”两个字，还包括：

- 大块传输由硬件地址生成；
- payload 不经过大量普通寄存器；
- 大量线程级 load/store 指令被一条 bulk copy 取代；
- SMEM swizzle 可由 tensor map 描述并自动完成；
- 完成事件直接接入 mbarrier。

### 5.2 Tensor map 是传输契约

Tensor map 同时描述四类信息：

| 类别 | 内容 |
|---|---|
| Source geometry | base pointer、rank、global dimensions、byte strides |
| Tile geometry | box dimensions、element strides |
| SMEM placement | interleave、swizzle |
| Boundary/cache policy | OOB fill、L2 promotion |

它不是单纯的二维指针，而是“源 tensor + 每次 tile + 目标 SMEM 布局”的完整契约。

### 5.3 A/B tensor map 参数

CUDA tensor map 的 dim0 是最内层维。当前 A/B 都是 K 连续，因此：

| 参数 | A map | B map |
|---|---|---|
| dtype | bf16 | bf16 |
| rank | 2 | 2 |
| base | `dA` | `dB` |
| global dimensions | `{K, M}` | `{K, N}` |
| 外维 byte stride | `{K×2}` | `{K×2}` |
| box dimensions | `{BK, BM}` | `{BK, BN}` |
| element strides | `{1,1}` | `{1,1}` |
| interleave | NONE | NONE |
| swizzle | 128B | 128B |
| L2 promotion | NONE | NONE |
| OOB fill | NONE | NONE |

TMA 坐标也按 tensor map 的维度顺序：

```text
A coordinate = {it×BK, tileM}
B coordinate = {it×BK, tileN}
```

最常见错误是把习惯中的 `(row, col)` 直接传进去，忘记 map 的 dim0 是 K。

### 5.4 为什么 kernel 参数用 `__grid_constant__`

`CUtensorMap` 作为 launch 参数传入。`__grid_constant__` 表明该参数在整个 grid 内保持只读一致，避免编译器为每个线程生成不必要的本地副本，并满足这类 descriptor 参数的高效传递方式。

### 5.5 `expect_tx` 为什么报告 A+B 总字节数

一轮有两条 TMA copy：

```text
A：BM×BK×2 = 16 KiB
B：BN×BK×2 =  8 KiB
总计             = 24 KiB
```

两条 copy 共用同一个 full mbarrier，因此本轮通过 `arrive.expect_tx` 一次承诺总字节数：

\[
(BM+BN)\times BK\times2=24576\text{ bytes}
\]

full phase 要同时满足：

- thread arrival 条件完成；
- 两条 TMA 承诺的总 transaction bytes 全部完成。

### 5.6 full 和 empty 各自回答什么

单缓冲也需要两个概念不同的 barrier：

| barrier | 谁使它完成 | wait 返回说明什么 |
|---|---|---|
| `full` | TMA transaction completion | A/B 已经装满，MMA 可以读 |
| `empty` | tcgen05 commit completion | MMA 已读完该 SMEM，下一轮可以覆盖 |

stage 的所有权循环：

```text
EMPTY
  │ producer 获得所有权，发 TMA
  ▼
FILLING
  │ full barrier 完成
  ▼
FULL
  │ Tensor Core 消费
  ▼
CONSUMING
  │ empty barrier 完成
  ▼
EMPTY
```

### 5.7 单缓冲 TMA 的正确时间线

```text
第0轮：
  issue TMA0 → wait full0 → MMA0 → commit empty0

第1轮：
  wait empty0 → issue TMA1 → wait full1 → MMA1 → commit empty1

第2轮：
  wait empty1 → issue TMA2 → wait full2 → MMA2 → commit empty2

循环结束：
  wait 最后一轮 empty → epilogue
```

为什么最后还要 drain？最后一轮之后没有“下一轮覆盖”来触发顶部的 empty wait，但 epilogue 仍必须等最终 TMEM accumulator 完成。

### 5.8 full/empty 的 parity

单个 full barrier 每轮使用一次：

```text
full wait token = it & 1
```

empty 表示本轮 MMA 完成；下一轮覆盖前等待前一轮：

```text
iteration it>0 覆盖前，等待上一轮 empty token = (it-1) & 1
```

循环结束等待最后一轮：

```text
empty token = (iters-1) & 1
```

这里 token 表示“当前 phase 开始时我看到的 parity”，等待其翻转，而不是“希望最终变成哪个 parity”。

### 5.9 为什么 4.2 不再需要 `fence.proxy.async`

4.1：

```text
普通 st.shared = generic proxy
tcgen05 descriptor read = async proxy
```

需要 generic→async proxy fence。

4.2：

```text
TMA write SMEM = async proxy
tcgen05 descriptor read = async proxy
```

生产者与消费者都在 async proxy 路径，TMA full mbarrier 已报告完成，因此不再需要 4.1 的 generic→async fence。

但下面仍需要：

- full mbarrier wait：确认 copy 完成；
- `tcgen05.fence::after_thread_sync`：跨线程把完成关系接到各自 tcgen05 操作；
- empty mbarrier：确认 MMA 不再读取 stage；
- `tcgen05.wait::ld`：确认 TMEM→register 完成。

删掉一个 fence 不代表所有同步都消失。

### 5.10 4.2 题目答案：TMA 消除了哪些普通 CUDA 工作

4.1 staging 包含：

```text
逐元素 GMEM load
→ 普通寄存器中转
→ 每线程计算 global index
→ 每线程计算 swizzle address
→ 逐元素 SMEM store
```

TMA 后，上述大部分地址生成、数据移动和 swizzle placement 都由 TMA 硬件完成；CUDA Core 主要负责：

- 构造/传入 tensor map；
- elected thread 发起两条 bulk transfer；
- 维护 mbarrier 和循环控制。

仍然存在的成本：

- HBM/L2 到 SMEM 的真实字节传输；
- TMA 发射和 tensor map setup；
- full/empty 同步；
- 单缓冲导致的 copy→wait→compute 串行。

因此 4.2 通常比 4.1 快，但还没有解决 overlap。

---

## 6. 4.3：多级循环缓冲 Pipeline

### 6.1 从“传输机制”到“调度机制”

Session 4 的一句重要区分：

```text
TMA = transfer mechanism
multistage buffering = schedule
```

仅仅使用异步指令，不代表程序自动发生重叠。若发出 TMA 后立刻等待，再发 MMA，仍然是串行。

多级缓冲的目的，是在计算当前 tile 时，提前发出未来 tile 的 TMA：

```text
issue copy k+1
→ compute k
→ 在真正使用 k+1 前才 wait
```

### 6.2 每个 stage 的内存布局

设 stage 数为 S，每个 stage 24 KiB：

```text
stage 0：[A0 16 KiB][B0 8 KiB]
stage 1：[A1 16 KiB][B1 8 KiB]
...
stage S-1：[A 16 KiB][B 8 KiB]
```

stage 下标循环复用：

```text
s = it % S
generation = it / S
```

例如 S=3：

| it | stage | generation | full parity |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 2 | 0 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 1 | 1 |
| 5 | 2 | 1 | 1 |
| 6 | 0 | 2 | 0 |

所以同一 stage 的 full wait parity：

\[
full\_phase=(it/S)\mathbin{\&}1
\]

### 6.3 为什么每个 stage 都要两个 mbarrier

对 stage s：

- `full[s]`：TMA 已经把本 generation 数据写完；
- `empty[s]`：MMA 已经消费完本 generation，可以被下一 generation 覆盖。

不能把所有 stages 共用一个 empty barrier，因为 parity 只能区分同一个 barrier 地址的相邻 generation，无法表达“stage0 已空，但 stage1 仍满”这种独立状态。

状态空间实际上是：

```text
stage0：FULL generation 3
stage1：CONSUMING generation 3
stage2：FILLING generation 4
```

一个全局二值 parity 无法表示这些并行状态。

### 6.4 初次使用与复用时的 empty parity

stage 第一次使用时天然为空，不需要等待一个历史 empty 事件。

当 `generation>=1` 时，覆盖 stage 前必须等待上一 generation 的 MMA 完成：

\[
empty\_token=((generation-1)\mathbin{\&}1)
\]

例如 stage0：

```text
it=0，gen0：首次使用，不等 empty
MMA0 完成，使 empty[0] 的 phase0完成

it=3，gen1：覆盖前 wait empty[0] token0
MMA3 完成，使 empty[0] 的 phase1完成

it=6，gen2：覆盖前 wait empty[0] token1
```

### 6.5 Pipeline 三段：prime、steady、drain

#### Prime：预热

先发：

```text
min(S, K_iterations)
```

轮 TMA，把尽可能多的 stage 填起来。此时 Tensor Core 还没有历史 tile 可算，无法避免启动气泡。

#### Steady state：稳态

每轮做三类事：

1. 保证当前要消费的 tile 已经发出；
2. 有空槽时机会式向更远未来预取；
3. 等当前 full，发 MMA，commit 到当前 empty。

TMA 和 Tensor Core 属于不同异步引擎，因而未来 tile 的 TMA 可以与当前 tile 的 MMA 重叠。

#### Drain：排空

最后一个 K tile 发出 MMA 后，必须等它对应的 empty 完成，再进入 epilogue。末尾没有更多 tile 可以遮住这个等待，因此会有 drain 气泡。

### 6.6 S=3 的形象时空图

下图只表达所有权和重叠，不按真实周期比例：

```text
时间 ─────────────────────────────────────────────────────▶

stage 0: [TMA0]──FULL0──[MMA0]──EMPTY0──[TMA3]──FULL3──[MMA3]
stage 1:   [TMA1]──FULL1──────[MMA1]──EMPTY1──[TMA4]──FULL4
stage 2:     [TMA2]──FULL2──────────[MMA2]──EMPTY2──[TMA5]

TMA engine:  TMA0 TMA1 TMA2       TMA3       TMA4       TMA5
TC engine :             MMA0 MMA1 MMA2 MMA3 MMA4 MMA5
```

理想稳态中：

```text
TMA 正在填未来 stage
Tensor Core 正在消费当前 stage
CUDA threads 正在维护控制状态
```

### 6.7 强制发射与机会式预取

这是 4.3 最危险的 hazard。

机会式预取逻辑：

```text
try_wait(empty[next_stage])
如果空了：发未来 TMA
如果没空：立即停止，不阻塞
```

它只能提高预取深度，不能保证正确性。可能出现：

```text
早先检查 tile k：stage 还没空，所以跳过 TMA k
        ↓
主循环走到要消费 tile k
        ↓
直接 wait full[k]
        ↓
但 TMA k 从未发出，full[k] 永远不可能完成
        ↓
死锁
```

因此在等待当前 full 前必须有一条“强制发射”路径：

```text
若当前 tile 尚未 issued：
    阻塞等待它的 stage empty
    发出该 tile 的 TMA
然后才允许 wait full
```

一句口诀：

> try-prefetch 负责快，force-issue 负责活。

### 6.8 为什么小形状可能通过、大形状稳定挂死

hazard 是时序窗口：机会式检查恰好遇到 stage 尚未释放时才触发。循环短、CTA 少时可能碰不到；4096³ 包含大量 CTA 和 K iteration，偶发窗口会被放大为高概率甚至稳定复现。

这说明：

- 小测试通过不证明 pipeline 正确；
- 必须使用题目给出的较大 stress shape；
- deadlock 测试需要 `timeout -k`，否则进程可能持续占卡。

### 6.9 Stage 数的资源成本

每 stage 24 KiB：

| STAGES | A/B SMEM | 加1024 B对齐余量后的申请量约为 |
|---:|---:|---:|
| 2 | 48 KiB | 49 KiB |
| 3 | 72 KiB | 73 KiB |
| 4 | 96 KiB | 97 KiB |
| 6 | 144 KiB | 145 KiB |

另外还有 full/empty mbarrier 和少量控制状态，但主导项是 A/B buffers。

TMEM accumulator 仍是32 KiB，不随 STAGES 增加。

### 6.10 为什么 stages 不是越多越好

增加 stages 的收益：

- 更早预取；
- 更多 memory requests in flight；
- 给长延迟更大的隐藏窗口；
- 当前 tile 等待 full 的概率下降。

代价：

- SMEM/block 线性增长；
- 每 SM 可驻留 CTA 数可能下降；
- active warps/TLP 下降；
- barrier、phase、issued 状态增加；
- prime/drain 和调度复杂度增加。

如果 S=3 已足以遮住 TMA 延迟，S=6 不会凭空提高 Tensor Core 峰值，反而可能因 occupancy 下降变慢。

### 6.11 两个 sweep 形状为什么敏感度不同

#### 4096×4096×4096

```text
grid = (32,64) = 2048 CTAs
K iterations = 64
```

CTA 数很多。某个 CTA 等 TMA 时，调度器可能从同一 SM 的其他 resident CTA/warp 找到 eligible work。它既能依赖 intra-CTA pipeline 的 MLP，也有较强的 block-level TLP。

因此：

- 适中的 S 通常已经够用；
- S 继续增大若降低 resident blocks，可能抵消更深预取；
- 对 stages 的收益可能较快饱和。

#### 256×4096×16384

```text
grid = (2,64) = 128 CTAs
K iterations = 256
```

grid 小、每个 CTA 的 K 链很长。可供调度器切换的其他 CTA 更少，block-level TLP 弱，更依赖单 CTA 内的 TMA/MMA overlap。

因此：

- stage 深度更可能显著影响延迟隐藏；
- 长 K 稳态占比更高，prime/drain 更容易被摊薄；
- 但若 stage 增多导致资源过重，仍可能出现收益饱和或回退。

最终趋势必须用你的 B300 sweep 数据确认，不能预先写死“S=几必胜”。

### 6.12 4.3(a) 答案：瓶颈如何移动

可以按梯子写：

1. **4.1 tiled**：主要受手工 staging 指令、地址/swizzle 计算、寄存器中转和 copy/compute 串行影响；Tensor Core 经常等数据。
2. **4.2 TMA**：线程级搬运与地址计算显著减少，但单缓冲仍是 `TMA→wait→MMA`，主要剩余瓶颈是传输延迟暴露和串行等待。
3. **4.3 pipeline**：TMA 与 MMA 重叠后，显式传输等待减少；瓶颈转向 Tensor Core 实际利用率、SMEM/TMEM资源、occupancy、可用CTA数量、pipeline控制以及prime/drain气泡。

实际报告要把这些判断与 TFLOPS、NCU stall/pipe 指标对应。

### 6.13 4.3(b) 答案：梯子每一级减少什么

| 实现 | 相比上一级减少的主要开销 | 仍未解决 |
|---|---|---|
| assignment01 naive fp32 | 基线，无 tile/Tensor Core 优化 | 数据复用、Tensor Core、异步搬运 |
| 4.1 tiled | 使用bf16 Tensor Core、片上tile复用 | 手工staging、串行copy/compute |
| 4.2 TMA | 普通load/store、寄存器中转、地址和swizzle指令 | 单缓冲暴露TMA延迟 |
| 4.3 pipeline | copy与compute的串行空窗 | 资源/occupancy、调度、剩余气泡 |

注意 naive 使用 fp32，不能把数值当作完全同 dtype 的公平加速比；题面只要求比较性能量级和瓶颈演化。

### 6.14 4.3(c) 答案：SMEM 与 TMEM 谁先顶住

对当前设计：

```text
TMEM = 32 KiB accumulator，基本不随 STAGES 增长
SMEM = STAGES × 24 KiB，随 STAGES 线性增长
```

所以继续增加 stages 时，**SMEM 会先成为容量/occupancy 限制**。

增大 tile 时两者都可能增加：

- SMEM：((BM+BN)\times BK\times2\times STAGES)；
- TMEM：(BM\times BN\times4)，并受 column 数限制。

但当前 S=3 已有72 KiB SMEM，而 TMEM 只使用64/512 columns。沿题目建议的“增加 stages 或适度扩大 tile”路线，SMEM 压力更直接。

联系 3.4：`cta_group::2` 让每 CTA 只保存一半 B。当前每 stage 的 B 从8 KiB降到4 KiB，A仍为16 KiB，因此每 stage 可从24 KiB降到20 KiB。省下的空间可用于：

- 在相同 SMEM 预算下增加 stage；
- 扩大 N/K tile；
- 保留更多 occupancy 余量。

这就是 2-CTA 的间接收益，即使单 tile 时间没有明显变化。

---

## 7. 4.5：瘦 GEMM 与 Roofline

### 7.1 为什么单独研究 M

在投影层 GEMM 中：

- N、K 由模型权重确定；
- M 是本次处理的 token 数。

```text
decode：M≈batch，常见1–16
过渡区：M≈64–256
chunked prefill：M可达几千或几万
```

所以同一份权重在 decode 和 prefill 上是完全不同的计算形状。

### 7.2 七个 Kimi K3 投影形状

| 名称 | N | K | 特点 |
|---|---:|---:|---|
| `f_b_proj` | 1536 | 128 | K极短，固定开销难摊薄 |
| `q_b_proj` | 2304 | 1536 | 中等形状 |
| `o_proj` | 7168 | 1536 | N较大 |
| `fused_qkv_a_proj` | 2112 | 7168 | K较长 |
| `in_proj_qkvgfab` | 6288 | 7168 | KDA输入投影，C1直接参考 |
| `dense_down_proj` | 7168 | 8448 | 大投影 |
| `dense_gate_up_proj` | 16896 | 7168 | N很大 |

### 7.3 Arithmetic Intensity 推导

题面口径：bf16 A、B、D，各2 byte：

```text
FLOP = 2MNK
bytes = 2MK + 2NK + 2MN
```

所以：

\[
AI=\frac{2MNK}{2MK+2NK+2MN}
=\frac{MNK}{MK+NK+MN}
\]

当 M 很小、N/K 很大时，权重项 (NK) 主导：

\[
AI\approx\frac{MNK}{NK}=M
\]

这解释了一个非常重要的规律：

> decode batch为1时，AI约为1 FLOP/byte；batch为8时约为8；batch为16时约为16。小M天然难以利用极高的Tensor Core峰值。

当 M 趋向无穷时：

\[
AI_{\infty}=\frac{NK}{N+K}
\]

这说明 K 很短的 `f_b_proj` 即使 M 很大，AI 也有很低的形状上限。

### 7.4 预计算 AI 表

下表按题面公式计算，可作为运行前的纸面预测：

| shape | M=1 | 8 | 16 | 64 | 256 | 1024 | 4096 | 16384 | 65536 | (AI_\infty) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `f_b_proj` | 1.0 | 7.5 | 14.1 | 41.5 | 80.8 | 105.9 | 114.8 | 117.3 | 117.9 | 118.2 |
| `q_b_proj` | 1.0 | 7.9 | 15.7 | 59.8 | 200.3 | 485.1 | 752.3 | 872.5 | 908.8 | 921.6 |
| `o_proj` | 1.0 | 7.9 | 15.8 | 60.9 | 212.9 | 565.9 | 966.5 | 1174.3 | 1241.0 | 1264.9 |
| `fused_qkv_a_proj` | 1.0 | 8.0 | 15.8 | 61.6 | 221.3 | 629.1 | 1166.7 | 1483.6 | 1591.7 | 1631.3 |
| `in_proj_qkvgfab` | 1.0 | 8.0 | 15.9 | 62.8 | 237.8 | 784.2 | 1842.7 | 2781.0 | 3186.7 | 3349.6 |
| `dense_down_proj` | 1.0 | 8.0 | 15.9 | 63.0 | 240.1 | 810.1 | 1991.9 | 3135.6 | 3661.1 | 3877.8 |
| `dense_gate_up_proj` | 1.0 | 8.0 | 15.9 | 63.2 | 243.6 | 850.9 | 2258.2 | 3850.2 | 4673.9 | 5032.9 |

### 7.5 如何预测 compute-bound / memory-bound

把你的 M0 机器平衡点 ​\(\beta\) 与表中 AI 比较：

```text
AI < β：memory roof = AI × peak_GB/s 更低
AI ≥ β：compute roof = peak_TFLOPS 更低
```

理论性能上限：

\[
P_{roof}=\min(P_{peak},\ AI\times B_{peak})
\]

注意单位：若 AI 用 FLOP/byte，带宽用 GB/s，则 (AI\times GB/s) 数值上得到 GFLOP/s；除1000换成TFLOP/s。

### 7.6 两种达成率分别说明什么

程序打印：

```text
%TCpeak = measured_TFLOPS / peak_TFLOPS
%BW     = measured_GB/s / peak_GB/s
```

解释方式：

- `%TCpeak` 高：接近计算峰值；
- `%BW` 高：接近按题面字节口径估计的带宽峰值；
- `%TCpeak` 低但 `%BW` 高：典型 memory-bound；
- 两者都低：通常有延迟、形状、并行度、tile浪费或固定开销，不能简单归类成纯带宽或纯计算瓶颈。

### 7.7 4.5(a) 答案：性能随 M 的趋势

定性结论：

1. M=1–16 时，输出 tile 很瘦，AI约等于M，权重读取和固定开销占主导，Tensor Core TFLOPS明显偏低。
2. M增大到64–256，AI提高，可用CTA和每次调用工作量增加，性能进入快速上升/过渡区。
3. M达到1024或更大后，多数大K投影逐渐进入平台，Tensor Core tile被充分填充，启动和prologue/epilogue得到摊薄。
4. `f_b_proj` 因K=128，会与其他形状表现不同，不应强行套用相同平台结论。

题目询问“从什么时候明显下降、什么范围进入平台、平台达成率多少”，其中确切断点和百分比必须从你的输出填写。建议定义清楚判断口径，例如：

```text
平台：连续三个较大 M 的 TFLOPS 变化小于10%
明显下降：相对平台 TFLOPS 低于70%
```

然后根据实测写：

```text
在本机 B300 上，______ 形状从 M=______ 以下明显下降；
M=______ 至 ______ 进入平台；平台约 ______ TFLOPS，
为 cuBLAS/理论峰值的 ______%。
```

### 7.8 4.5(b) 答案：哪些形状受显存带宽限制

纸面判断方法：

- 表中 `AI < β` 的点，Roofline 预测为 memory-bound；
- 小 M 时 AI≈M，所以 M≤16 的绝大多数点一定远低于现代高算力GPU的平衡点，理论上属于带宽侧；
- 随 M 增大，大K形状会跨越平衡点，逐渐转向计算侧；
- `f_b_proj` 的 (AI_\infty≈118.2)，是否能跨越机器平衡点要直接与M0的 ​\(\beta\) 比较。

实测归因不能只看 AI：

```text
AI < β 且 %BW 较高 → 带宽限制证据充分
AI < β 但 %BW 也很低 → roof虽在带宽侧，实际还受延迟/形状/并行度影响
```

### 7.9 4.5(c) 答案：`f_b_proj` 为什么两个 roof 都低

核心是 **K=128 太短**：

- K方向只有很少的 Tensor Core/TMA 主循环迭代；
- pipeline 刚完成 prime，很快就 drain，稳定区太短；
- Tensor Core setup、tile调度、epilogue和kernel launch难以摊薄；
- 对固定 tile 而言，K尾部/形状利用可能不理想；
- (AI_\infty≈118.2)，即使M无限增大，算术强度也很快封顶。

因此它可能既没有足够长的计算链逼近 Tensor Core 峰值，也没有形成足够高效、连续的字节流逼近 HBM roof。限制属于 shape/latency/amortization，而不是简单的“算力不够”或“带宽打满”。

### 7.10 4.5(d) 答案：为什么 M≤16 选择 skinny CUDA Core kernel

M≤16 时：

- Tensor Core 的 M tile 大量空置或需要 padding，存在无效计算；
- TMA/tensor map、TMEM、barrier、epilogue 等固定成本相对有效工作过大；
- CTA 数量和并行工作不足，难以形成稳定 pipeline；
- AI≈M，计算量相对权重读取量太少；
- 直接 CUDA Core FMA 的指令吞吐峰值虽低，但启动简单、形状贴合、控制开销小。

所以选择标准不是“CUDA Core 峰值高于 Tensor Core”，而是：

> 对非常瘦的实际工作量，避免 Tensor Core 路径的固定成本与 tile 浪费，比追求理论峰值更重要。

这也解释了上游在 M≤16 时使用 skinny kernel 的工程决策。

---

## 8. 4.1–4.3 的完整思路清单

### 8.1 4.1 实现前检查

```text
[ ] grid=(M/BM,N/BN)
[ ] tileM/tileN 映射正确
[ ] A/B global index 带 tile 与 it 偏移
[ ] SMEM index 使用本地 row/k 和 swz128
[ ] 每轮 generic→async proxy fence
[ ] 每 BK 发4条 k16 MMA
[ ] 仅整个 GEMM 第一条 MMA 不累加
[ ] 每轮 commit/wait 后才覆盖单缓冲
[ ] epilogue 写回带 tileM/tileN
[ ] 全部读完后 dealloc
```

### 8.2 4.2 实现前检查

```text
[ ] A map dimensions={K,M}, box={BK,BM}
[ ] B map dimensions={K,N}, box={BK,BN}
[ ] byte stride=K×2
[ ] TMA coordinates={it×BK,tileM/tileN}
[ ] 128B swizzle 与 tcgen05 descriptor 匹配
[ ] full/empty barrier 各一个
[ ] expect_tx 报 A+B 总字节数24KiB
[ ] 两条 TMA copy 共用 full barrier
[ ] 删除手工 st.shared/swz128/proxy fence
[ ] 保留 full wait、tcgen fence、empty wait
[ ] 最后一轮 MMA 要 drain
```

### 8.3 4.3 实现前检查

```text
[ ] SMEM 划成 S 个独立 stage
[ ] 每 stage 有 full[s] 与 empty[s]
[ ] stage=it%S，generation=it/S
[ ] full parity=generation&1
[ ] 首次使用 stage 不等 empty
[ ] 复用前等待上一 generation 的 empty
[ ] prime min(S,iters) 个 TMA
[ ] 维护 next_to_issue/issued 状态
[ ] 当前 tile 未发时必须 force-issue
[ ] opportunistic try-prefetch 失败不能丢掉该 tile
[ ] MMA descriptor 指向当前 stage
[ ] 全局累加谓词仍只有第一条为false
[ ] 最后一轮 empty drain 后才能epilogue
[ ] 大形状和多STAGES都进行超时判测
```

---

## 9. 实验与报告模板

### 9.1 运行前后检查占卡

```bash
nvidia-smi --query-compute-apps=pid,name --format=csv
```

性能数字必须来自独占 GPU，否则 stage sweep 的小差异没有解释价值。

### 9.2 梯子判测

在 `assignment02/cuda/m4_gemm/`：

```bash
./judge_ladder.sh
```

它会从上一级 `cuda/Makefile` 构建并依次运行 4.1、4.2、4.3。

### 9.3 Stage sweep

```bash
./sweep_stages.sh
```

脚本会使用 `make -B` 重新编译 S=2/3/4/6。`-B` 不能省，因为只改变编译宏时，make 看不到源文件时间戳变化。

### 9.4 梯子表

| 实现 | TFLOPS | 对 cuBLAS 达成率 | 一句话瓶颈 |
|---|---:|---:|---|
| naive fp32 | 实测 | 仅比较量级 | 无Tensor Core、低复用、同步数据路径 |
| 4.1 tiled | 实测 | 实测 | 手工staging与copy/compute串行 |
| 4.2 TMA | 实测 | 实测 | 单缓冲暴露TMA延迟 |
| 4.3 pipeline S=3 | 实测 | 实测 | 资源/occupancy/剩余流水气泡 |
| cuBLAS | 实测 | 100% | 优化基准 |

### 9.5 Stage sweep 表

| 形状 | S=2 | S=3 | S=4 | S=6 | 最佳S与解释 |
|---|---:|---:|---:|---:|---|
| 4096³ |  |  |  |  | 大grid，TLP较强；关注SMEM导致的驻留变化 |
| 256×4096×16384 |  |  |  |  | 小grid、长K，更依赖intra-CTA MLP |

### 9.6 NCU 归因框架

具体 metric 名称可能随 NCU/架构版本变化，先用 `ncu --query-metrics` 确认。报告至少覆盖这些问题：

| 要回答的问题 | 观察方向 |
|---|---|
| Tensor Core 是否饿住 | tensor pipe active/utilization |
| 4.1 是否被 staging 指令淹没 | LSU、shared store、整数地址指令、指令数 |
| 是否在等内存 | long scoreboard、memory dependency stall |
| 4.3 是否改善 eligible work | eligible warps/scheduler、issue slot |
| stages 是否压低 occupancy | resident blocks/warps、SMEM per block |
| HBM 是否接近 roof | DRAM throughput/bytes |

不要只贴一张 NCU 截图。每个指标后写一句因果：

```text
观察 → 它排除了什么 → 它支持哪个瓶颈结论
```

---

## 10. 与 C1 FlashKDA 的联系

### 10.1 `in_proj_qkvgfab` 是直接参照

4.5 的：

```text
N=6288, K=7168, name=in_proj_qkvgfab
```

对应 KDA 输入投影。它告诉你在不同 token 数 M 下，这个真实 KDA 相关 GEMM 的 shape、AI 和 cuBLAS/Tensor Core 表现。

### 10.2 M4 给 C1 的四个判断工具

1. **形状匹配**：理论峰值必须乘上 tile 利用率和实际并行度。
2. **数据供给**：新 MMA 指令若需要更复杂 staging，峰值优势可能被供数吞掉。
3. **流水重叠**：异步指令本身不够，必须有足够 stages 和正确 ownership。
4. **资源代价**：更深 stage、更大 tile、2-CTA 会占用 SMEM/TMEM并影响 occupancy。

### 10.3 对 FlashKDA 的提醒

FlashKDA 的核心递推不是规则的4096³ GEMM。即使 M4.3 的 tcgen05 pipeline 在大方阵上很好，也不能直接推出 SM100 FlashKDA 一定更快。仍要回答：

- `CHUNK=16` 是否能填满 tcgen05 的 tile；
- 是否需要合并多个 head 才能构造大 M；
- chunk 间状态依赖是否限制 in-flight 工作；
- 多 stage 的 SMEM 是否与 recurrent state 争资源；
- 额外 TMEM/descriptor/barrier setup 能否摊薄。

M4 提供机制和参照系，C1 需要对具体 KDA shape 再做一次量化。

---

## 11. 最终速记

### 11.1 优化梯子

```text
4.1：数据复用 + Tensor Core，但手工搬运、完全串行
4.2：TMA减少指令和寄存器中转，但仍单缓冲
4.3：多stage让TMA与MMA重叠，代价是SMEM和控制复杂度
4.5：形状太瘦时，所有高级机制的固定成本可能不划算
```

### 11.2 Stage 所有权口诀

```text
等 EMPTY → 发 TMA → 等 FULL → 发 MMA → commit EMPTY → 允许复用
```

### 11.3 四类问题

```text
Completion：工作结束了吗？
Visibility：消费者看得见吗？
Ordering：不同代理排好序了吗？
Ownership：存储槽能覆盖了吗？
```

### 11.4 4.3 最重要的正确性规则

```text
机会式预取可以失败；当前必需 tile 的发射绝不能丢。
```

### 11.5 4.5 最重要的近似

```text
小M、大N/K时：AI≈M
```

这就是 decode M≤16 难以利用 Tensor Core 峰值的最短解释。

---

## 12. 自测题

1. 为什么 4.1 中 `accumulate = kk>0` 不再正确？
2. 4.1 的 proxy fence 和 `__syncthreads()` 分别保证什么？
3. TMA tensor map 为什么把 K 放在 dim0？
4. 两条 TMA copy 为什么只向一个 full barrier 报一次总 expected bytes？
5. 4.2 使用 TMA 后为什么可以删除 generic→async proxy fence？
6. full barrier 与 empty barrier 分别代表哪一方拥有 stage？
7. 为什么 TMA 是异步的仍不自动意味着 copy/compute overlap？
8. S=3 时，iteration 7 使用哪个 stage、哪个 generation 和哪个 full parity？
9. 为什么每个 stage 必须有独立 empty barrier？
10. 机会式预取失败后，为什么当前 tile 仍必须走强制发射路径？
11. 为什么4096³与256×4096×16384对STAGES的敏感度不同？
12. 当前设计增加STAGES时，为什么SMEM先于TMEM成为限制？
13. 当M很小时，如何从AI公式推出 `AI≈M`？
14. `%TCpeak`和`%BW`都低时，为什么不能说“既compute-bound又memory-bound”？
15. `f_b_proj` 的K=128为什么使pipeline难以进入长稳态？
16. 为什么skinny CUDA Core kernel可能击败理论峰值更高的Tensor Core路径？

能脱稿回答这些问题，就已经掌握 M4 的主线。

---

## 13. 材料索引

- `session03.pdf`：SM100 TMEM、tcgen05、mbarrier、2-CTA。
- `session04.pdf`：latency hiding、async copy、TMA、pipeline ordering、stage ownership。
- `handout/src/assignment02.md`：M4正式题面。
- `cuda/m4_gemm/01_tiled.cu`：完整 tiled GEMM 骨架。
- `cuda/m4_gemm/02_tma.cu`：TMA tensor map 与单缓冲协议。
- `cuda/m4_gemm/03_pipeline.cu`：多级 full/empty pipeline 与已知 hazard。
- `cuda/m4_gemm/05_thin_gemm.cu`：Kimi K3真实形状与Roofline实验。
- `cuda/m4_gemm/judge_ladder.sh`：4.1–4.3梯子判测。
- `cuda/m4_gemm/sweep_stages.sh`：S=2/3/4/6性能扫描。
- `M3_tcgen05_知识点详解.md`：M3前置知识与3.2/3.3逐段解释。
