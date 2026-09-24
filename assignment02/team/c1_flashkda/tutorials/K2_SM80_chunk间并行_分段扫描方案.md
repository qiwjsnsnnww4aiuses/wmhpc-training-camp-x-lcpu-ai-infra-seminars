# FlashKDA K2：SM80 chunk 间并行方案

> 方案名称：Segment Summary + Affine Scan + Parallel Replay
>
> 目标：打破原版 K2 中“一个 CTA 顺序处理 512 个 chunk”的执行结构，在不使用 tcgen05/WGMMA 的前提下，通过 SM80 MMA 和多 kernel 全局同步创造 chunk 组之间的并行。
>
> 本文件是架构与实施方案，不表示代码已经实现。

> **修订说明：**本文最初给出的 Hillis--Steele ping/pong scan 是正确性基线，但不是最终推荐。用户要求的是“超前进位 + reduce”思路，最终应采用 work-efficient 的上扫 reduce + 下扫 carry。详见
> [`K2_SM80_超前进位Reduce最终方案.md`](./K2_SM80_超前进位Reduce最终方案.md)。

---

## 0. 先纠正上一版方案

上一版 P/W 因子化：

```text
P = R@Kd
W = R@V
U = W-P@S
```

只减少了每个 chunk 内的一次小 GEMM。状态链仍然是：

```text
chunk 0 -> chunk 1 -> chunk 2 -> ... -> chunk 511
```

因此它没有产生 chunk 间并行。

如果目标是让不同 chunk 组同时执行，必须利用 chunk 状态转移的仿射性质并进行 prefix scan。

推荐结构：

```text
原版 K1
   |
   v
K2a: 每个 segment 并行生成状态摘要 (A_seg, B_seg)
   |
   v
K2b: 对 segment 摘要做并行 affine prefix scan
   |
   v
K2c: 所有 segment 从各自 S_start 并行 replay，生成 output
```

同一个 segment 内仍顺序执行 G 个 chunk；不同 segment 在 K2c 中可以同时执行。

---

## 1. 数学基础

### 1.1 单 chunk 是仿射状态变换

原版 K2 的状态部分可以写成：

```text
KS = Kd @ S_in
U  = INV @ [diag(beta) @ (V-KS)]
S_out = G*S_in + Kr^T@U
```

整理得到：

```text
S_out = A_i*S_in + B_i
```

其中：

```text
A_i = G_i - Kr_i^T @ INV_i @ diag(beta_i) @ Kd_i
B_i = Kr_i^T @ INV_i @ diag(beta_i) @ V_i
```

`A_i、B_i` 都是数学上的 `D×D` 矩阵。

### 1.2 两个连续变换如何组合

若：

```text
T_left(S)  = A_left*S + B_left
T_right(S) = A_right*S + B_right
```

先执行 left，再执行 right：

```text
T_right(T_left(S))
= A_right*A_left*S + A_right*B_left + B_right
```

组合结果为：

```text
A_combined = A_right @ A_left
B_combined = A_right @ B_left + B_right
```

这个组合操作是结合的，因此可以做并行 scan。

顺序不能写反。KDA 的序列方向是：

```text
earlier chunk -> later chunk
```

所以 scan 中必须计算：

```text
later o earlier
```

### 1.3 输出也依赖 segment 起始状态

每个 chunk 的 output 可写成：

```text
O_i = C_i*S_in + D_i
```

但不建议为所有 chunk 显式保存 `C_i,D_i`。更稳妥的做法是 scan 得到每个 segment 的起始状态，然后在 K2c 中重放该 segment 的原版 recurrence，同时生成 output。

---

## 2. 为什么不做“每个 chunk 一个完整 A/B”

fixed `T=8192,H=96,CHUNK=16`：

```text
chunks/head = 8192/16 = 512
chunk-head tiles = 512*96 = 49152
```

每个 bf16 A/B：

```text
A[D,D] = 128*128*2 = 32768 bytes
B[D,D] = 128*128*2 = 32768 bytes
A+B                    65536 bytes
```

如果每个 chunk 都保存 A/B：

```text
49152*65536 bytes = 3.00 GiB
```

Hillis--Steele scan 为避免读写别名还需要 ping/pong 两套 buffer：

```text
约 6.00 GiB
```

同时每次组合需要：

```text
A_right@A_left
A_right@B_left
```

即两次 `128×128 @ 128×128`。直接对 512 个 chunk 扫描的计算量和内存量都过大。

因此采用 blocked/segmented scan：只给每 G 个 chunk 保存一个 A/B。

---

## 3. 推荐算法：三阶段 K2

令：

```text
G = 每个 segment 包含的 chunk 数
```

例如 `G=32`：

```text
每 segment token 数 = 32*16 = 512
每 head segment 数 = 512/32 = 16
```

### 3.1 K2a：Segment Summary

每个 CTA 负责：

```text
(sequence, head, segment)
```

并为该 segment 生成：

```text
A_seg[D,D]
B_seg[D,D]
```

不需要先物化其中每个 chunk 的 A/B。
#### 如何直接计算 A_seg

对状态递推关闭 V 项：

```text
V = 0
S_initial = I
```

因为此时没有常数项，运行 G 个 chunk 后：

```text
S_final = A_seg*I = A_seg
```

#### 如何直接计算 B_seg

使用真实 V，并令：

```text
S_initial = 0
```

运行 G 个 chunk 后：

```text
S_final = A_seg*0+B_seg = B_seg
```

因此 summary CTA 内维护两个虚拟状态：

```text
state_A: 初始为 I，使用 V=0
state_B: 初始为 0，使用真实 V
```

两条轨迹读取相同的 Kd/Kr/G/INV/beta；只有 V 项不同。

### 3.2 K2b：Affine Prefix Scan

对每个 sequence/head 的 segment maps 做 inclusive scan。

以第 `level` 层 stride 为：

```text
stride = 1, 2, 4, 8, ...
```

若当前 local segment `j >= stride`：

```text
A_new[j] = A_old[j] @ A_old[j-stride]
B_new[j] = A_old[j] @ B_old[j-stride] + B_old[j]
```

否则直接复制：

```text
A_new[j] = A_old[j]
B_new[j] = B_old[j]
```

每层读一个 buffer、写另一个 buffer，然后交换 ping/pong。

最后第 j 个位置保存：

```text
从该 sequence 起点到 segment j 末尾的完整仿射变换
```

### 3.3 K2c：Parallel Replay

每个 CTA 再负责一个 segment，但这次生成真实 output。

segment j 的输入状态：

```text
j == 0:
    S_start = initial_state 或 0

j > 0:
    读取 scan 后 j-1 的 prefix map
    S_start = A_prefix[j-1] @ initial_state + B_prefix[j-1]
```

如果没有 initial state：

```text
S_start = B_prefix[j-1]
```

因此常见 no-state 场景完全不需要额外的 `A@initial_state` GEMM。

拿到 `S_start` 后，CTA 运行原版 K2 的 G 个 chunk：

```text
加载 chunk factors
计算 Kd@S、Qd@S
计算 U、output
更新 state
进入 segment 内下一个 chunk
```

不同 segment 已拥有自己的正确起始状态，所以可以并行 replay。

只有每条 sequence 的最后一个 segment 写 `final_state`；所有 segment 都写自己不重叠的 output token 区间。

---

## 4. 并行度发生了什么变化

原版 fixed H=96：

```text
grid = 1 sequence * 96 heads = 96 CTAs
每 CTA 顺序处理 512 chunks
```

G=32：

```text
segments/head = 16
summary grid = 16*96 = 1536 CTAs
replay  grid = 16*96 = 1536 CTAs
每 CTA 只顺序处理 32 chunks
scan levels = log2(16) = 4
```

因此：

```text
独立 CTA 数增加 16x
单 CTA chunk 链长度缩短 16x
```

粗略关键路径由：

```text
512 chunks
```

变为：

```text
32-chunk summary + 4 scan levels + 32-chunk replay
```

这不代表必然 8 倍加速，因为总计算量增加，但它确实创造了 chunk 组之间的并行。

---

## 5. G 如何选择

对 fixed H=96：

| G | segments/head | summary/replay CTAs | scan levels | 双 buffer A/B |
|---:|---:|---:|---:|---:|
| 8 | 64 | 6144 | 6 | 768 MiB |
| 16 | 32 | 3072 | 5 | 384 MiB |
| 32 | 16 | 1536 | 4 | 192 MiB |
| 64 | 8 | 768 | 3 | 96 MiB |

权衡：

```text
G 小：
    更多 segment CTA
    每 CTA 串行链更短
    scan map 更多，scan 工作和内存更大

G 大：
    map/scan 开销更低
    每 CTA 串行链更长
    并行度下降
```

第一组推荐实验：

```text
H=96: G = 16, 32, 64
H=12: G = 8, 16, 32
```

默认从：

```text
G=32
```

开始。它把 fixed H96 的 grid 从 96 提升到 1536，同时只需约 192 MiB 的 scan ping/pong buffer。

真实实现应根据：

```text
H
sequence length
number of sequences
available workspace
```

做 runtime dispatch/autotune。

---

## 6. Global memory 布局

### 6.1 保留原 K1 workspace

第一版不要修改 K1。继续保留：

```text
Kd, Qd, Kr, G, INV, Mqk
```

原因：

- summary 需要 Kd/Kr/G/INV；
- replay 需要全部原始因子；
- 保留 K1 可以把问题集中在 K2 重构；
- 原版 K1 已达到约 96.6% occupancy，不应在第一版同时重写。

### 6.2 Segment metadata

新增：

```text
segment_prefix[N+1] int32
```

对 sequence n：

```text
chunks_n   = ceil(seq_len_n / CHUNK)
segments_n = ceil(chunks_n / G)
```

`segment_prefix` 是每条 sequence 的 segment 数前缀和，用于将 global segment id 映射到：

```text
seq_idx
local_segment_idx
first_chunk
num_chunks_in_segment
```

### 6.3 Map buffer

建议 head-major：

```text
A_ping [H, total_segments, D, D] bf16
B_ping [H, total_segments, D, D] bf16
A_pong [H, total_segments, D, D] bf16
B_pong [H, total_segments, D, D] bf16
```

实际线性 index：

```text
map_idx = head_idx*total_segments + global_segment_idx
```

好处：同一 head 中相邻 segment 连续，scan 的 `j` 与 `j-stride` 访问容易计算。

### 6.4 fixed G=32 的容量

```text
total_segments = 16*96 = 1536
单个 A 或 B = 1536*128*128*2 = 48 MiB
一套 A+B   = 96 MiB
ping+pong   = 192 MiB
```

加上原版约 648 MiB workspace，主 workspace 总量约 840 MiB，不包括输出和输入。

### 6.5 为什么需要 ping/pong

计算：

```text
A_new = A_cur @ A_prev
```

不能安全地一边读取 `A_cur`，一边把结果覆盖回同一矩阵，因为后续 output tile 仍可能需要被覆盖的输入元素。

第一版使用 ping/pong，换取简单且可靠的读写隔离。若结果有性能价值，再实现 work-efficient/in-place tree scan 降低内存。

---

## 7. Shared memory 与 warp 设计

### 7.1 K2a summary shared memory

建议：

```text
state_A[D,D] bf16 = 32 KiB
state_B[D,D] bf16 = 32 KiB
2-stage input pipeline，每 stage 约 13 KiB
barrier/alignment
```

每 input stage 只需：

```text
Kd [16,128]   4 KiB
Kr [16,128]   4 KiB
V  [16,128]   4 KiB
G  [128]      0.5 KiB
INV [16,16]   0.5 KiB
beta          64 B
```

summary 不计算 output，所以不需要加载 Qd/Mqk，也不需要 output stages。

预计 shared 总量约：

```text
64 KiB + 2*13 KiB + metadata
约 90--94 KiB
```

与原版 K2 的 98.43 KiB 同量级。

### 7.2 summary warp roles

建议 block：

```text
1 load warp
4 MMA warps -> state_A
4 MMA warps -> state_B
合计 9 warps = 288 threads
```

两组 MMA warp 使用同一个 input stage：

- A 组令 V=0；
- B 组使用真实 V；
- 两组分别更新自己的 state shared buffer；
- load warp 预取下一 chunk 的 Kd/Kr/V/G/INV/beta。

这样 summary 的依赖深度接近 G，而不是先完整跑 A 再完整跑 B 的 2G。

需要检查：

- 两组 warp 是否竞争 shared bandwidth；
- registers/thread 和 spill；
- 288-thread block 的 theoretical occupancy；
- 两个 state buffer 的 bank-conflict/layout。

### 7.3 K2b scan shared memory

每个 scan CTA 只计算一个 D×D GEMM：

```text
blockIdx.z == 0: A_cur @ A_prev
blockIdx.z == 1: A_cur @ B_prev + B_cur
```

grid：

```text
(total_segments, H, 2)
```

每个 CTA 使用 SM80：

```text
SM80_16x8x16_F32BF16BF16F32_TN
```

建议 8 warp，每个 warp 轮流计算若干 `16×16` output tiles。A/B 输入可采用：

```text
方案 1：整张两个 128×128 bf16 输入放 shared
        约 64 KiB，复用率高，实现简单

方案 2：双缓冲 128×16 / 16×128 panels
        shared 更小，但 TMA/pipeline 更复杂
```

第一版建议方案 1：每个 CTA 64 KiB shared，输出 fragment 直接 vector store 到 pong，避免再放 32 KiB output shared。

边界 `local_segment < stride` 时不做 GEMM，只用线程协作把当前 A/B 从 ping 复制到 pong。

### 7.4 K2c replay shared memory

第一版直接复用原版 `SharedStorageK2` 和 SM80 MMA 主循环：

- state buffer 仍为 32 KiB bf16；
- 3-stage input pipeline；
- 2-stage output pipeline；
- dynamic shared 约 98.43 KiB；
- 每 CTA 的 t-loop 从整条序列改为本 segment 的 G 个 chunk。

虽然每 CTA occupancy 可能仍低，但 grid 从 96 增至数百或数千 CTA，可以提供足够 waves。

---

## 8. SM80 scan GEMM 的实现要点

### 8.1 128×128 输出分块

```text
D=128
基本 output tile=16×16
总共 8×8=64 个 output tiles
```

使用 8 个 warp：

```text
每 warp 负责 8 个 16×16 tiles
每 tile 的 K 方向为 8 个 K=16 blocks
```

SM80 atom 的 N 基本粒度为 8；一个逻辑 16×16 tile需要两个 n8 MMA。

### 8.2 两类 scan product

```text
Product A:
    C = A_cur @ A_prev

Product B:
    C = A_cur @ B_prev
    C += B_cur
```

两类用不同 CTA，通过 `blockIdx.z` 区分，避免一个 CTA 顺序做两张输出而降低并行度。

### 8.3 累加与存储

第一版：

```text
input maps    bf16
MMA accumulate fp32
output maps   bf16
```

每个 level 都会发生一次 bf16 map 舍入。必须把 scan level 数纳入精度分析。

若误差过大，可增加 accurate 模式：

- map global storage 改 fp32；
- scan 使用 SM80 TF32 MMA，仍属于 SM80 tensor core；
- 或加载 fp32 后显式转换到 bf16 panel，但这只改善存储误差，不能消除 MMA 输入量化。

不要在第一版同时实现多种精度。

---

## 9. 计算量模型

### 9.1 原版 K2

每 chunk/head tensor FLOP 约：

```text
kS + qS       1,048,576
INV@U            65,536
Mqk@U            65,536
state delta     524,288
合计           1,703,936
```

fixed H96、512 chunks/head：

```text
约 83.75 GFLOP
```

### 9.2 Summary

只做 state transition，不做 q/output。一个虚拟状态每 chunk：

```text
Kd@S           524,288
INV@U           65,536
Kr^T@U         524,288
合计          1,114,112
```

A/B 两条轨迹：

```text
2,228,224 FLOP/chunk
```

所有 chunk 的 summary 总量约：

```text
109.5 GFLOP
```

它基本不随 G 改变。

### 9.3 Scan

一次 map composition 有两个 D×D GEMM：

```text
2 * (2*D^3)
= 8,388,608 FLOP
```

Hillis--Steele 对 m 个 segment 的 combine 数：

```text
sum over stride (m-stride)
```

fixed H96：

| G | m | combine/head | scan GFLOP（H96） |
|---:|---:|---:|---:|
| 8 | 64 | 321 | 258.5 |
| 16 | 32 | 129 | 103.9 |
| 32 | 16 | 49 | 39.5 |
| 64 | 8 | 17 | 13.7 |

### 9.4 Replay

Replay 与原 K2 基本相同，总 tensor FLOP 仍约：

```text
83.75 GFLOP
```

### 9.5 总工作量

| G | summary + scan + replay | 相对原 K2 |
|---:|---:|---:|
| 8 | 451.8 GFLOP | 5.39× |
| 16 | 297.2 GFLOP | 3.55× |
| 32 | 232.7 GFLOP | 2.78× |
| 64 | 207.0 GFLOP | 2.47× |

这张表说明：

> chunk 间并行不是免费午餐。我们用额外计算与 workspace，换取更短的依赖链和更高的 GPU 占用率。

由于原 fixed K2 Compute throughput 只有约 21%，而 B300 大矩阵 tensor core 能力很强，这种“多算但更并行”的方案仍值得实验，但不能预先保证正收益。

---

## 10. 数值问题：bf16 递推并非严格仿射

数学上的：

```text
S_out = A*S_in+B
```

是严格仿射。

但原版 K2 每个 chunk 都会把状态保存在 bf16 shared memory，实际执行更接近：

```text
S_out = round_bf16(A*S_in+B)
```

`round_bf16` 不是线性操作，所以：

```text
F(I)-F(0)
```

以及 scan 重结合顺序不能逐 bit 复现原串行路径。

本方案的主要精度风险：

1. A/B summary 使用两条虚拟状态产生近似仿射 map；
2. 每个 scan level 把 A/B 再舍入为 bf16；
3. scan 改变浮点结合顺序；
4. replay 的 segment 起始状态来自近似 prefix；
5. 误差随后在 G 个串行 chunk 内传播。

### 10.1 G 的精度权衡

```text
G 小：
    单个 summary map 跨越较少 bf16 round
    但 scan levels/map compositions 更多

G 大：
    scan levels 更少
    但 segment map 对任意输入状态的仿射近似跨越更多 round
```

因此 G 不仅是性能参数，也是精度参数。

### 10.2 必测精度

至少比较：

```text
原版串行 FlashKDA
scan-replay 改版
naive FP32 reference
Triton chunk_kda
```

覆盖：

```text
T = 16, 256, 1024, 8192, 32768
G = 8,16,32,64
fixed / varlen / tail
无 initial state / 随机 bf16 state / fp32 boundary state
gate 接近 0、接近 -5、随机、交替极值
beta 接近 0、接近 1、随机
至少 20 seeds
```

指标：

```text
output/state max_abs
mean_abs
rel_rmse
cosine
P99/P99.9
NaN/Inf
误差随 segment/token 的曲线
```

如果长序列误差无法接受，说明原版 bf16 chunk-rounding 破坏了可扫描性。此时可能需要 FP32 map/TF32 scan，或者停止该路线。

---

## 11. Varlen 设计

不能把不同 sequence 的 map 合并。

为每条 sequence 计算：

```text
chunk_count[n]   = ceil(seq_len[n]/16)
segment_count[n] = ceil(chunk_count[n]/G)
segment_prefix[n+1]
```

scan step 对 global segment `s` 先通过 `segment_prefix` 找到：

```text
seq_idx
local_seg
```

只有：

```text
local_seg >= stride
```

时才与同一 sequence 内的 `local_seg-stride` 组合。

最后一个 segment 可能少于 G 个 chunk；summary 和 replay 都使用：

```text
segment_chunks = min(G, chunks_in_seq-first_chunk)
```

最后一个 chunk 又可能少于 16 token，继续沿用原版 mask/tail 逻辑。

---

## 12. Kernel 与文件结构

建议不要直接把所有代码塞进 `fwd_kernel2.cuh`，而是拆分：

```text
csrc/smxx/
├── fwd_kernel1.cuh                 # 第一版保持不变
├── fwd_kernel2.cuh                 # 保留原版串行 fallback
├── fwd_kernel2_segment_summary.cuh # K2a
├── affine_scan_sm80.cuh            # K2b
├── fwd_kernel2_segment_replay.cuh  # K2c
├── fwd_launch.cu                   # runtime dispatch 和多 kernel launch
└── utils.cuh                       # map sizes/layout/SM80 GEMM helper
```

### 12.1 `utils.cuh`

新增：

```text
SegmentMapSizes<D>
SegmentMapLayouts<D>
compose_affine_map_sm80()
```

### 12.2 `flash_kda.cpp`

workspace size需要加入：

```text
原 K1 workspace
ping A/B
pong A/B
segment_prefix
```

接口增加可选参数或编译时宏：

```text
use_segment_scan
segment_chunks G
```

第一版最好单独构建扩展名，避免覆盖原版：

```text
flash_kda_scan_C
flash_kda_scan
```

### 12.3 `fwd_launch.cu`

新 launch 链：

```text
K1 prepare
build_segment_prefix
K2a summary
for stride in 1,2,4,...:
    K2b scan_step ping -> pong
    swap(ping,pong)
K2c replay
```

所有 kernel 位于同一个 CUDA stream。每次 kernel launch天然构成 device-wide 顺序边界，因此不需要在单个 kernel 内实现跨 CTA 全局 barrier。

保留原版路径：

```text
if chunks_per_head 较少：原版 K2
if scan workspace 不足：原版 K2
if 架构/shape 不支持：原版 K2
```

---

## 13. 分阶段实现

### 阶段 1：CPU/PyTorch 仿射 scan 原型

先不写 CUDA。使用 `naive.py` 的小 D 版本验证：

1. 单 chunk A/B；
2. map composition 顺序；
3. inclusive prefix；
4. 从 prefix 得到每 segment S_start；
5. replay output 与串行 reference 对比。

这一步主要防止 `A_right@A_left` 顺序写反。

### 阶段 2：只实现 summary，暂不 scan

CUDA K2a 输出 A_seg/B_seg。用 Python 验证：

```text
A_seg@random_S+B_seg
```

是否接近将 `random_S` 送入原版 recurrence 跑 G chunks 的结果。

分别测 G=1/2/4/8/16/32，先观察 bf16 非线性误差随 G 的增长。

这是 go/no-go gate：如果 G=8 已经误差不可接受，不值得继续写 scan。

### 阶段 3：SM80 map composition microbench

只测试：

```text
A_new=A_cur@A_prev
B_new=A_cur@B_prev+B_cur
```

验证：

- 对 PyTorch reference；
- SASS 有 HMMA.16816；
- 一个 composition 的 latency；
- 64 KiB shared 与 8 warp 的 occupancy；
- bf16 map 多层组合误差。

### 阶段 4：scan fixed only

先实现 `N=1,T` 为 16G 的整数倍。

测试：

```text
H=1/2
segments=1/2/3/4/8/16
```

对拍每个 inclusive prefix A/B，而不是只看最后一个。

### 阶段 5：replay fixed

让每个 segment CTA 从 prefix state 开始运行原 K2 主循环。测试 output/final_state。

### 阶段 6：varlen/tail/state variants

加入：

```text
segment_prefix
非 16 token 尾块
非 G chunks 尾 segment
initial/final state
```

### 阶段 7：端到端 benchmark 与 NCU

扫 G=8/16/32/64，输出：

```text
K1
summary
每个 scan level
replay
完整 fwd
```

不能只报告 replay 比原 K2 快；summary 和 scan 必须计入端到端。

---

## 14. Benchmark 和 NCU 验收

### 14.1 性能形状

```text
T=8192,H=96,D=128 fixed
T=8192,H=12,D=128 fixed
varlen 6 条不等长
varlen 8×1024
```

### 14.2 分阶段计时

```text
original K2
summary
scan total
scan per level
replay
new K2 total = summary+scan+replay
full original fwd
full scan fwd
```

### 14.3 NCU 关键问题

Summary/replay：

```text
grid 是否从 96 提升到预期数量？
achieved occupancy 是否提升？
eligible warps/scheduler 是否提升？
no eligible 是否从约 67.5% 下降？
compute throughput 是否明显超过原 K2 的约 21%？
```

Scan：

```text
HMMA throughput
DRAM/L2 throughput
64 KiB shared 对 occupancy 的限制
map load/store 是否成为瓶颈
每个 level 尾部是否 underfill
```

完整 forward：

```text
总延迟是否低于原版？
额外 2.5--5.4 倍 K2 FLOP 是否被更高利用率抵消？
workspace 是否可接受？
H=12 是否仍有收益？
```

---

## 15. 风险排序

### 风险 1：bf16 rounding 破坏严格仿射

最高风险。必须在写完整 scan 前先做 summary 的随机状态映射测试。

### 风险 2：额外计算量大于并行收益

G=32 仍约为原 K2 的 2.78 倍 tensor FLOP。只有原版低利用率足够严重时才可能回收。

### 风险 3：map global traffic

scan 每 level 都读写 D×D maps。即使 tensor core 很快，也可能变成 L2/DRAM bound。

### 风险 4：initial state

有 initial state 时，每个非首 segment 需要：

```text
A_prefix@initial_state+B_prefix
```

额外 D×D GEMM。可先以 no-state 推理场景作为第一版范围。

### 风险 5：短序列/多序列不值得 scan

varlen 中每条 sequence segment 很少时，scan launch 和 summary 开销可能超过收益。必须 runtime fallback。

---

## 16. Runtime dispatch 建议

第一版规则可以保守设置：

```text
if chunks_per_sequence < 8*G:
    使用原版串行 K2

if workspace 不足：
    使用原版串行 K2

if 有 initial_state 且第一版未优化：
    使用原版串行 K2

else:
    使用 summary + scan + replay
```

之后根据 benchmark 建立选择表：

```text
(H, chunks_per_seq, N) -> G
```

不要强制所有 shape 都使用 scan。

---

## 17. 最小可行版本（MVP）

建议把第一版范围严格限制为：

```text
fixed length
N=1
D=128
CHUNK=16
H=96
T=8192
no initial state
bf16 map
G=32
SM80 BF16 MMA + FP32 accumulator
```

实现：

1. K1 完全不改；
2. K2a 双状态 summary；
3. 4 层 Hillis--Steele ping/pong scan；
4. K2c 16 个 segment/head 并行 replay；
5. 最后 segment 写 final state；
6. 与原版/naive 对拍；
7. 分阶段计时和 NCU。

MVP 通过后再加：

```text
G autotune
H=12
varlen
initial state
FP32/TF32 map 精度模式
work-efficient scan
```

---

## 18. 最终判断

真正创造 K2 chunk 间并行，推荐：

```text
K2a Segment Summary
    每 segment 生成 A_seg/B_seg

K2b Affine Prefix Scan
    SM80 D×D GEMM 组合 segment maps

K2c Parallel Replay
    每 segment 从正确 S_start 独立生成 output
```

它比“每 chunk 保存 A/B”节省大量 global memory，也比只做 P/W 真正产生了 chunk 组并行。

但要明确研究假设：

> 用 2.5--5.4 倍的 K2 总 tensor 工作量和额外 map workspace，换取 8--32 倍 CTA 数与显著缩短的串行关键路径。

这条路线是否能赢，取决于 B300 上原 K2 的低利用率能否覆盖 scan/replay 的额外工作。第一道 gate 不是性能，而是验证 bf16 chunk-rounding 下 segment map 是否仍有足够精度；通过之后才值得完成整条 CUDA scan。
