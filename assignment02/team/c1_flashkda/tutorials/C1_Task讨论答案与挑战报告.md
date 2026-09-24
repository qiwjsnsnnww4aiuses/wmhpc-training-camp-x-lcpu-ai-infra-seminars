# C1 FlashKDA：Task 六问与 SM100 指令替换挑战报告

> 状态：六个讨论问题已完成纸面分析；挑战代码已完成。所有标成
> **[待 B300 实测]** 的数字必须在 `dev-slurm` 上运行后填写，不能用纸面值
> 冒充测量结果。
>
> 本文档是“最终结论与报告”；每个讨论题要做什么、如何留证据，
> 以及如何把本地 challenge 同步到 B300，见
> [完整操作指南](C1_第一阶段_复现操作指南.md#21-从复现成功到完整-c1现在处于哪里)。

## 0. 结论摘要

1. `CHUNK=16` 首先由数值范围约束。`lower_bound=-5` 时，最坏累计门控到
   第 18 个 token 已使一个指数低于 FP32 FTZ normal 范围、另一个超过
   BF16/FP32 最大有限值。`CHUNK=32/64` 必须增加 subchunk rescale。
2. 把当前 dense Neumann 路径直接从 16 扩到 32/64，inverse 的每 token
   FLOP 分别变成 5.33 倍和 26.67 倍；这已不是改模板常数。
3. `CHUNK=16` 与 `mma.sync.m16n8k16` 完全匹配，却不能无浪费地填满课程中
   已验证的 `tcgen05.m128n128k16`。M=16 路径会产生 8 倍 issued FLOP。
4. 唯一自然匹配的主要切面是 K2 状态增量
   `k_restored.T @ U`，shape 为 `m128n128k16`，占纸面 Tensor FLOP 的
   约 27.8%。
5. K2 的根本限制不只是一条 MMA 的速度，还包括 chunk 依赖与 `N×H`
   CTA 数。官方同 token 数的 8 序列 case 比单序列 case 快约 30%，已说明
   并行度的重要性。
6. 建议 v2 保留 portable SM80 主路径，只在完整 B300 实验稳定提升
   10–15%、精度与 TP8 shape 均通过时，增加 SM100a dispatch。

挑战选择“换指令、不改算法”。代码同时测 M=16 补零路径和 M=128 自然
匹配路径，见 [实验说明](experiments/README.md) 与
[`instruction_only_mma.cu`](experiments/instruction_only_mma.cu)。

---

## 1. `CHUNK=16` 的三个理由：32/64 时谁先破、代价多大？

### 1.1 结论

最先破的是当前无重标定的指数表示。随后是 dense Neumann inverse 的
立方代价；SM80 tile 匹配是让 C=16 更划算的第三个原因。

### 1.2 BF16/FP32-FTZ 数值范围

激活后门控满足：

$$
g_t \in [-5,0].
$$

一个 chunk 内的累计量最坏为：

$$
G_C = \sum_{t=1}^{C} g_t = -5C.
$$

kernel 同时会形成正向和反向尺度：

$$
e^{G_i},\qquad e^{-G_i}.
$$

源码使用 `ex2.approx.ftz.f32`。FP32 最小 normal 约为 $2^{-126}$，其
自然对数约为 -87.34；BF16 与 FP32 的最大有限值都约为
$3.39\times10^{38}$，自然对数约为 88.72。

| CHUNK | 最坏累计值 | 正向指数 | 反向指数 | 判断 |
|---:|---:|---:|---:|---|
| 16 | -80 | $1.80\times10^{-35}$ | $5.54\times10^{34}$ | 两边仍有限且未被 FTZ 清零 |
| 32 | -160 | $3.26\times10^{-70}$ | $3.07\times10^{69}$ | 下溢/上溢 |
| 64 | -320 | $1.06\times10^{-139}$ | $9.42\times10^{138}$ | 更严重失效 |

最坏输入下，第 17 个 token 累计为 -85，仍在边界内；第 18 个 token
累计为 -90，已经越界。因此 16 是小于约 17.5 的最大常用 2 的幂。

平均 gate 未必接近 -5，但 kernel 的正确性不能依赖平均输入。一旦反向
指数为 inf，后续还可能由 `0 * inf` 传播 NaN。

### 1.3 C=32/64 的数值修复为什么算算法改动

可把每 16 token 设一个 anchor $a_b$：

$$
\widehat G_i = G_i-a_b,
$$

使 subchunk 内指数保持安全，再显式传播不同 subchunk 的尺度。它需要
额外 anchor、rescale、off-diagonal block 合并、同步与尾块逻辑。
所以大 CHUNK 仍以内层 16-token 数值块为基础，不是把 `16` 改成 `64`。

### 1.4 Neumann inverse 的代价

`L` 是严格下三角矩阵，因此 $L^C=0$，有限 Neumann 级数为：

$$
(I-L)^{-1}=I+L+L^2+\cdots+L^{C-1}.
$$

当前乘积展开对 2 的幂 C 需要 $2(\log_2 C-1)$ 个 C³ GEMM。按一次 GEMM
为 $2C^3$ FLOP：

| C | GEMM 数 | inverse FLOP/chunk | inverse FLOP/token | 相对 C16/token |
|---:|---:|---:|---:|---:|
| 16 | 6 | 49,152 | 3,072 | 1.00 倍 |
| 32 | 8 | 524,288 | 16,384 | 5.33 倍 |
| 64 | 10 | 5,242,880 | 81,920 | 26.67 倍 |

中间矩阵元素数也从 256 增至 1024/4096，即 4 倍/16 倍。

### 1.5 整个 Tensor Core 工作量与 workspace

令 head dimension $D=128$。按现有 dense 公式，每 token、每 head 的主
Tensor FLOP 近似为：

$$
F=6D^2+8CD+4(\log_2C-1)C^2.
$$

| C | FLOP/token/head | 相对 C16 |
|---:|---:|---:|
| 16 | 117,760 | 1.00 倍 |
| 32 | 147,456 | 1.25 倍 |
| 64 | 245,760 | 2.09 倍 |

K1→K2 workspace 每 tile 为：

$$
W=3CD\times2+D\times4+2C^2\times2.
$$

| C | workspace/tile | workspace/token/head |
|---:|---:|---:|
| 16 | 13,824 B | 864 B |
| 32 | 29,184 B | 912 B |
| 64 | 66,048 B | 1,032 B |

大 C 虽减少循环轮数，C² 中间量却使每 token workspace 反而增长。

### 1.6 MMA 形状

`mma.sync.m16n8k16` 对 C=16 的 M、K 均无需 padding；一个 16×16 输出只需
沿 N 发两条 atom。C=32/64 可以继续由 atom 拼接，但前述数值和 inverse
成本已经先要求算法重构。因此“SM80 shape 匹配”是加分项，不是唯一约束。

---

## 2. tcgen05 最小 tile 与 C=16 匹配吗？只换指令有收益吗？

### 2.1 结论

不直接匹配。课程中已经在 B300 验证的 dense 单 CTA 路径为 M=128；对
M=16 有效行利用率仅 12.5%，issued/useful FLOP 为 8 倍。即使工具链支持
M=64 变体，也仍只有 25% 有效行和 4 倍 FLOP。

### 2.2 按真实 phase 分类

C=16、D=128 时，每 chunk 的纸面 Tensor Core 工作如下：

| phase | 数学 shape | FLOP/chunk | 总占比 | 直接 tcgen 判断 |
|---|---|---:|---:|---|
| K1：L 与 Mqk | 两个 `16×128 @ 128×16` | 131,072 | 6.96% | M16 不匹配 |
| K1：inverse | 六个 `16×16 @ 16×16` | 49,152 | 2.61% | M16 且过小 |
| K2：k/q @ state | 两个 `16×128 @ 128×128` | 1,048,576 | 55.65% | 最大热点但 M16 |
| K2：INV/Mqk @ U | 两个 `16×16 @ 16×128` | 131,072 | 6.96% | M16 不匹配 |
| K2：state delta | `128×16 @ 16×128` | 524,288 | 27.83% | 自然匹配 |

合计为 1,884,160 FLOP/chunk/head，即 460 条等价的
`m16n8k16` atom；其中一条 atom 为 4096 FLOP。

约 72.2% FLOP 的数学 M=16，不适合机械替换。状态增量的 27.8% 是最值得
尝试的切面，但它之后还需完成：

$$
S_{next}=S\odot g_{total}+\Delta S.
$$

当前旧状态在 BF16 SMEM，tcgen 输出在 TMEM；读回、逐元素 FMA 和再写回
的成本不能省略。

### 2.3 为什么不同 head 不能直接拼成 M=64/128

不同 head 计算 $K_hS_h$，每个 head 的 $S_h$ 不同。普通 dense GEMM 的
所有 M 行共享一个右操作数 B，所以把四个 head 的 K 沿 M 拼起来，并不
会得到一个合法的共享 B。block-diagonal B 会引入大量零计算；连续 chunk
也不能拼，因为后一个 chunk 依赖前一个状态。

### 2.4 固定成本

相较寄存器 accumulator 的 SM80 warp MMA，tcgen05 路径新增：

- SMEM descriptor 与 swizzle；
- TMEM alloc/dealloc；
- MMA commit 与 mbarrier wait；
- `tcgen05.ld` 把结果从 TMEM 取回；
- M padding；
- 更复杂的状态 epilogue 和潜在 occupancy 损失。

所以“指令数少”不能推出“延迟低”。

### 2.5 本项目的 microbench

挑战代码公平比较：

| Case | SM80 | SM100 | issued/useful |
|---|---|---|---:|
| thin | 精确 `m16n128k16` | 零填充 `m128n128k16` | 8 倍 |
| matched | 精确 `m128n128k16` | 精确 `m128n128k16` | 1 倍 |

计时包含 staging、同步、完整 TMEM 生命周期与输出写回。结果表：

| Case | SM80 ms | tcgen05 ms | tcgen/SM80 speedup | useful TF/s | issued TF/s | 结论 |
|---|---:|---:|---:|---:|---:|---|
| M16 padded | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** |
| M128 matched | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** |

预注册判断：M16 以 useful 延迟/吞吐判输赢，不能用含 87.5% padding 的
issued TFLOP/s 自我欺骗；M128 若更快，只证明状态更新 slice 有潜力。

---

## 3. chunk 间有依赖，并行度还能从哪里来？

### 3.1 当前并行性

K1 网格为 `N×H×num_chunks`，chunk 可并行；K2 网格只有 `N×H`，一个 CTA
按顺序推进本 `(sequence, head)` 的所有 chunk。可用独立性来自 sequence、
head、D/value tiles，而不是同一状态链上的相邻 chunk。

官方 GB200、`T=8192,H=96,D=128` 数据：

| 序列组织 | K2 CTA 数 | chunk/CTA | FlashKDA 延迟 |
|---|---:|---:|---:|
| `[8192]` | 96 | 512 | 1.0087 ms |
| 6 条不等长序列 | 576 | 不均匀 | 0.8597 ms |
| `[1024]×8` | 768 | 64 | 0.7064 ms |

总 token 与 FLOP 基本相同，8 序列却快约 30%。它把 CTA 数增为 8 倍并把
每条依赖链缩至 1/8，是并行度/延迟隐藏的直接证据。

### 3.2 候选 A：一个 CTA 交错多个 head/sequence

收益：摊薄 pipeline/barrier 固定成本，并在一条链等待时执行另一条链。

反例：每 head 的 BF16 state 为 $128^2\times2=32$ KiB，输入和状态均不
共享；多 head 会迅速增加 SMEM/TMEM/register 压力。TP8 下每卡 H=12，
若 4 head/CTA，全卡只剩 3 个 CTA，反而更差。因此目标应是增加 ILP，不能
简单减少 CTA 数。

### 3.3 候选 B：persistent worker + 动态任务队列

收益：varlen 时动态领取 `(sequence, head)`，减少长短序列不均衡；可复用
descriptor/pipeline 生命周期。

反例：fixed `N=1,H=12` 仍只有 12 条独立状态链，persistent 不会创造更多
并行任务；队列原子、上下文切换也有成本。现有 K2 已在一个 CTA 内持续
处理完整链，本身已有部分 persistent 特征。

### 3.4 候选 C：2-CTA/cluster

可把 D=128 列分给两个 CTA，或用 `cta_group::2` 扩大 tile。

反例：两 CTA 每 chunk 都需同步，cluster barrier 进入递推临界路径；
`cta_group::2` 主要扩大 M，而这里最大矛盾是 M=16。cluster 还限制驻留，
可能用双倍资源完成同一条依赖链。除非 NCU 明确显示单 CTA 算力不足且
cluster 复用能覆盖同步，否则优先级低。

### 3.5 候选 D：把 recurrence 改写成 prefix scan

若每 chunk 能表达为可结合的仿射映射 $S'=A_cS+B_c$，可并行 scan。
但 KDA residual 本身依赖当前 S，中间表示和读写可能大幅膨胀。这是研究级
算法重构，不属于本次“只换指令”挑战。

### 3.6 推荐次序

先 sweep `N=1/2/4/8` 和每卡 `H=12/24/48/96`，用 active SM、waves、
eligible warps 定位 underfill。若服务负载以 varlen 为主，再试 persistent
队列；多链/CTA 只有在不减少全卡 worker 数且资源放得下时才值得做；
2-CTA 作为最后候选。

---

## 4. B300 上是 compute-bound 还是 memory-bound？

### 4.1 纸面 Roofline

C=16、D=128 时，每 `(chunk,head)` 的 workspace：

| 张量 | 字节 |
|---|---:|
| k_decayed、q_decayed、k_restored | 3 × 4096 |
| g_total、INV、Mqk | 3 × 512 |
| 合计 | 13,824 B |

K1 写一次、K2 读一次，仅 workspace 即 27,648 B/chunk/head。包含 q/k/g、
v/beta、output 的最低逻辑流量约 48,772 B/chunk/head。

官方 fixed shape 有 $8192/16\times96=49,152$ 个 chunk-head：

$$
\text{Tensor FLOP}=1,884,160\times49,152=92.61\text{ GFLOP},
$$

$$
\text{logical bytes}\approx48,772\times49,152=2.397\text{ GB},
$$

$$
\text{arithmetic intensity}\approx38.6\text{ FLOP/B}.
$$

workspace 往返约 1.359 GB，占最低逻辑流量约 56.7%。用官方 1.0087 ms
只作一致性估算，可得约 91.8 useful TFLOP/s 与 2.38 TB/s logical rate；
后者不是实测 DRAM 带宽，因为 cache line、命中与写回会改变物理流量。

纸面结论：端到端不像纯 Tensor Core compute-bound，但不能只靠 AI 给 K1、
K2 贴同一个标签。

### 4.2 必测指标

先用 NCU `--set full`，再从版本支持的 metric 中抽取：

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
smsp__inst_executed_pipe_tensor.sum
dram__bytes_read.sum
dram__bytes_write.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
lts__throughput.avg.pct_of_peak_sustained_elapsed
lts__t_sector_hit_rate.pct
l1tex__throughput.avg.pct_of_peak_sustained_elapsed
```

还要记录 K1/K2 duration、grid、waves/SM、achieved occupancy、active/eligible
warps、register/CTA、SMEM/CTA，以及 long scoreboard、short scoreboard、
barrier、wait 等 stall。

### 4.3 判定规则

| 观察 | 结论倾向 |
|---|---|
| Tensor pipe 高、DRAM 低、eligible warp 足 | compute-bound |
| DRAM 接近峰值且 long scoreboard 高 | HBM-bound |
| L2 throughput 高、DRAM 不高 | L2/cache-bound |
| Tensor 与 DRAM 都低、active SM 少 | 网格/并行度不足 |
| active SM 足、eligible warp 低、barrier 高 | pipeline/同步限制 |
| occupancy 被 register 或 SMEM 封顶 | resource-bound |

最终应分别写：

| 部分 | duration | Tensor % | DRAM % | L2 % | active SM | 主 stall | 分类 |
|---|---:|---:|---:|---:|---:|---|---|
| K1 | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** |
| K2 | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** |

与 M4.5 的 `in_proj_qkvgfab` 联系在于二者都存在瘦 M/shape utilization
问题，且在真实模型中前后相邻；但它们是不同 kernel，不能直接复用同一个
Roofline 数字。

---

## 5. BF16 recurrent state 的精度验证怎么设计？

### 5.1 先澄清源码事实

API 的 FP32 initial/final state 不等于内部 FP32 recurrence。源码流程是：

```text
FP32 initial state
  -> load FP32 buffer
  -> cast to BF16 state_acc
  -> chunks between updates remain BF16
  -> cast back to FP32
  -> FP32 final state
```

因此 `flash_kda (fp32 state)` 只改变边界 dtype，不能作为内部 FP32 状态
消融对照。

### 5.2 三层验证

1. 实现一致性：官方 `tests/torch_ref.py` 复刻近似 sigmoid/exp、BF16 cast、
   FP16 inverse 与 chunk 间 BF16 round-trip。exact match 只说明实现符合
   设计，不说明该设计相对高精度数学足够准确。
2. 算子精度：以 FP64 `fused_recurrent_kda` 为 gold，对比 FlashKDA、
   `fla_kda_ref` chunk 实现、真正 FP32-state 实验版。
3. 模型质量：有 K3 权重时，再测 logits cosine/KL、token 一致率、生成
   分叉位置、perplexity/任务准确率。没有权重时只下算子级结论。

### 5.3 压力测试矩阵

- 长度：16、256、1024、8192、32768；长序列可用 H=1 降成本。
- gate：接近 0、接近 -5、常数 -1/-3/-5、随机、强弱交替。
- beta：接近 0、0.5、接近 1、随机和交替极值。
- initial state：零、正态、尺度 1e-2/1/1e2、低秩、单 outlier。
- shape：fixed、官方 varlen、H=1/12/96、尾块 1/15/16 token。
- 随机种子：至少 20 个；失败 case 固定 seed 复现。

输出与 final state 都记录：relative RMSE、mean/max absolute error、
P50/P90/P99/P99.9 error、safe relative error、cosine、NaN/Inf count；并画
误差随 token/chunk 位置的曲线，判断递推误差是否累积。

### 5.4 最重要的消融

| 版本 | state 保存 | update 累加 | 作用 |
|---|---|---|---|
| Gold | FP64 | FP64 | 数学参考 |
| A | FP32 | FP32 | 隔离 chunk/approx 误差 |
| B | BF16 | FP32 | 官方设计，隔离 state round-off |
| C | BF16 | BF16 | 说明 FP32 update 的必要性 |

真正回答“BF16 state 是否安全”的核心是 A 与 B 的差，而不是比较 API 的
BF16/FP32 final-state dtype。

BF16 unit roundoff 量级为 $2^{-7}=0.0078125$。先复现官方约 0.005/0.006
容差，再报告完整误差分布，并要求 FlashKDA 不显著劣于 FLA baseline。

结果表：

| T/gate/state scale | output rel-RMSE | state rel-RMSE | max error | cosine | NaN/Inf | vs FLA |
|---|---:|---:|---:|---:|---:|---|
| **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** | **[待实测]** |

---

## 6. 如果是作者，v2 是否发布 SM100a 专版？

### 6.1 支持发布

- K3 的 93 层中 69 层使用 KDA，小的单层收益可能被大量调用放大。
- B300/GB200 是重要部署平台，专用优化有价值。
- state-delta `m128n128k16` 天然匹配 tcgen05。
- TMEM 可能降低 register pressure，或让 accumulator 更长时间保留 FP32。
- tcgen05 异步数据流可能与 gate/epilogue 的 CUDA Core 工作重叠。
- runtime dispatch 可保留旧路径，不必让所有 GPU 承担新实现。

### 6.2 反对发布

- 约 72.2% 的主 Tensor FLOP 位于 M=16 GEMM，机械替换会 padding。
- 当前 SM80 路径已经在 GB200 比 FLA 快 1.7–3.3 倍。
- TMEM、descriptor、mbarrier、layout 与新 pipeline 增加维护成本。
- K2 可能先受 `N×H` underfill、依赖链和 workspace 流量限制。
- 单独加快 27.8% 的 state-delta，即使假设无限快，按 Amdahl 定律整个
  Tensor 部分上限也只有约 $1/(1-0.278)=1.385$ 倍；完整 kernel 上限更低。
- 新模型若改变 C、D 或 state layout，专版可能快速失效。

### 6.3 预先定义 ship gate

我会先保留实验分支，只有同时满足以下条件才正式发布：

1. 官方 fixed/两组 varlen 全部正确，完整算子对 `fla_kda_ref` 通过；
2. BF16/FP32 边界 state、尾块与长序列压力精度通过；
3. K3 H=96 和 TP8 每卡 H=12 都不明显退化；
4. 核心 shape 多次测量的端到端收益稳定达到约 10–15%；
5. NCU 证明收益来自 useful work，而不是 padding 后虚高的 issued FLOP；
6. 非 SM100 可靠 fallback 到现有 SM80 kernel。

10–15% 是工程门槛，不是硬件定理，但应在看结果前固定，避免事后移动标准。

### 6.4 最终决定

当前决定是：**不替换 portable 主路径；保留 SM100a 实验 dispatch。**

- 若 B300 实测显示 M16 padded 路径输、M128 matched 路径赢，下一步只把
  state-delta 与旧状态 epilogue 集成后测完整 K2。
- 若完整 K2/端到端稳定超过 ship gate，再发布专版。
- 若 TMEM 往返、padding 或 underfill 抵消收益，则负结果恰好证明官方停在
  SM80 是合理的。

---

## 7. 挑战实现：算法不动，指令替换

### 7.1 数学不变量

SM80 与 SM100 两条路径对外都计算：

$$
D_{m,n}=\sum_{k=0}^{15}A_{m,k}B_{n,k},
$$

输入均为 BF16、accumulator/output 均为 FP32。未改变 CHUNK、公式、dtype
或输出元素。M16 的适配仅把新增 A 行设为零并丢弃无用输出。

### 7.2 两条实现

SM80 baseline：

- 4 warp/CTA，每 warp 负责 32 个 N 列；
- 用 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`；
- M128 case 沿 M 拼 8 个 16-row tile；
- accumulator 直接在寄存器并写 global。

SM100 replacement：

- 128B swizzled shared-memory staging；
- SM100 descriptor；
- warp-collective TMEM alloc/dealloc；
- `tcgen05.mma.cta_group::1.kind::f16`；
- commit 到 mbarrier，等待后用 `tcgen05.ld` 读回；
- M16 case 真实执行完整 M128 physical tile，因此测量没有隐藏 padding。

### 7.3 正确性

输入限制为可由 BF16 精确表示的小整数，K=16，FP32 和的范围很小，因此
两条 Tensor Core 路径应与 CPU 参考 bit-exact。程序对两个 shape、两条
路径分别报告 mismatch 与 max absolute error；任一失败返回非零状态。

这是 slice correctness。完整 FlashKDA correctness 仍按以下层次验证：

1. 官方 kernel 对 `tests/torch_ref.py`；
2. 完整 output/final state 对 `fla_kda_ref`；
3. 本次被替换 GEMM slice 对 CPU exact reference。

### 7.4 运行

当前 `experiments/` 源码位于本地 WSL，而已完成的 FlashKDA 复现日志在
`dev-slurm`。必须先按操作指南第 29 节同步四个 challenge 源文件，
不需要重跑已完成的 FlashKDA benchmark。

```bash
cd ~/lcpu2026/assignment02/team/c1_flashkda/experiments
bash run_b300.sh 512 200 30 42
```

如果课程工具链使用 `compute_100a`：

```bash
ARCH=100a bash run_b300.sh 512 200 30 42
```

输出粘贴处：

```text
[待 B300 实测后粘贴完整 stdout]
```

### 7.5 挑战结论填写规则

- M16 `speedup > 1` 且 useful TFLOP/s 更高：才可说机械替换对小 M 有收益。
- M16 慢、M128 快：支持“只专门改 state-delta，不改其余 phase”。
- 两者都慢：说明 TMEM 固定成本在此粒度无法摊薄，支持保留 SM80。
- 两者都快：仍需集成 epilogue 并测完整 K2，不能直接宣称端到端胜出。

---

## 8. B300 最终待填清单

- [ ] FlashKDA 官方 correctness 与 `fla_kda_ref` 对拍。
- [ ] 官方三个 GB200 对照 shape 在 B300 的均值、P50、P90。
- [ ] SASS 中 K1/K2 的 SM80 MMA 证据。
- [ ] K1/K2 分离 NCU 指标与 bound 分类。
- [ ] 本挑战两个 shape 的 PASS、latency、useful/issued TFLOP/s。
- [ ] 挑战 cubin 同时包含 SM80 MMA 和 TCGEN05 的 SASS 证据。
- [ ] BF16 state 长序列与极端 gate 消融数据。
- [ ] 根据预先定义的 ship gate 写出最终 v2 go/no-go。

更完整的算法背景、K1/K2 逐阶段解释和答辩问答见
[任务拆解与完整方案](C1_FlashKDA_任务拆解与完整方案.md)。
