# C1 FlashKDA：任务拆解、原理、六问答案与执行方案

> 目标：让没有完整做完 M4.3、但已经理解 M3 核心代码的同学，可以直接开始 C1，并最终能解释“FlashKDA 为什么在 B300/GB200 上仍使用 SM80 `mma.sync`，迁移到 SM100 `tcgen05` 是否值得”。
>
> 本文基于仓库中的 `TASK.md`、FlashKDA 源码、官方 deep-dive、GB200/H20 benchmark、FLA 参考实现，以及前面的 M3/M4 知识文档。
>
> 状态标记：
>
> - **[源码可证]**：能直接从当前源码确认；
> - **[纸面推导]**：由形状、FLOP、byte 或数值范围推导；
> - **[官方数据]**：来自仓库已有 benchmark；
> - **[待 B300 实测]**：报告中必须补上自己的实验，本文绝不虚构。

---

## 0. 先读结论：这道题真正要回答什么

这道题不是“把所有 `mma.sync` 文本替换成 `tcgen05.mma`”。

真正的问题是：

> FlashKDA 的数学分块天然是 `CHUNK=16`，而 SM100 的标准 dense `tcgen05` 更喜欢较大的 M tile。新指令的峰值更高，但如果形状不匹配、并行度不够、数据搬运占主导，端到端反而不一定更快。

目前最有希望站住脚的候选结论是：

1. `CHUNK=16` 首先是**数值约束**，然后才是性能选择。`lower_bound=-5` 时，16 个 token 的最坏累计门控仍在 BF16/FP32-FTZ 的安全范围；32 个 token 会在第 18 个最坏门控附近同时出现正向指数下溢、反向指数上溢。
2. 当前代码中的主要小矩阵是 `16×128`、`16×16`。它们与课程 M3/M4 使用的 `m128...` tcgen05 路径明显不匹配；标准单 CTA dense tcgen05 的较小 M 形状也仍大于 16。
3. 不是所有运算都不匹配。K2 的状态更新包含一个天然的 `128×128×16` GEMM，它与 tcgen05 很匹配，但它只占纸面总 FLOP 的约 28%，而把结果引入 TMEM 后还要解决状态缩放、下一阶段取数和 TMEM→寄存器/SMEM 的数据路径。
4. `T=8192,H=96,D=128` 下，当前两 kernel 的最低逻辑流量约为 2.40 GB，纸面 Tensor Core 工作量约为 92.61 GFLOP，算术强度只有约 38.6 FLOP/B。仅 workspace 的“一写一读”就约为 1.36 GB，占最低逻辑流量约 56.7%。因此不能只看 Tensor Core 峰值。
5. K2 的 chunk 之间存在真正的递推依赖。官方固定长度 benchmark 中 K2 只有 `N×H=96` 个 CTA；K3 TP8 部署按每卡 12 头算时，固定 batch=1 甚至只有 12 个 CTA。它更可能先受到并行度、延迟和流水线约束，而不是纯 Tensor Core 吞吐约束。
6. 在截止时间紧的情况下，最稳妥的挑战路线是：
   - 复现和 profile 官方版本；
   - 写一个针对真实子形状的 SM80-vs-tcgen05 microbench；
   - 至少覆盖 `m16n16k16`、`m16n128k128` 和天然匹配的 `m128n128k16`；
   - 如果小 M 的 tcgen 路径无收益，把负结论做扎实即可完成挑战。
7. 产品决策建议不是“永远不做 SM100”，而是：**保留 portable SM80 主路径；只有当 SM100 专版在真实 K3/TP 形状上取得稳定的端到端收益并通过长序列精度验证时，才作为 dispatch 分支发布。**

这些只是候选结论。最终报告必须把 `[待 B300 实测]` 的格子填上，才能把“推理”升级成“证据”。

---

## 1. C1 的验收标准

题目分三层：

### 1.1 复现

你需要在 B300 上完成：

1. 安装 pin 到指定 commit 的 FlashKDA；
2. 通过 correctness test；
3. 跑官方三个序列形状，至少覆盖 `H=96,D=128`；
4. 最好再跑 `H=64` 和 K3 TP8 对应的 `H=12`；
5. 用 SASS/NCU 证明主路径确实是 SM80 `mma.sync`，而不是 tcgen05；
6. 分开观察 K1 prepare 和 K2 recurrence。

### 1.2 分析

六个讨论点不能只写观点。每问至少需要：

```text
一句结论
  + 源码证据
  + 纸面数字
  + B300 数据
  + 对反例的解释
```

### 1.3 挑战

三条路线任选其一：

- 只换指令，不改算法；
- 增大 CHUNK，并加入 rescale；
- 重构并行度。

挑战不要求正加速。负结果也成立，但要回答：

- 为什么没有收益？
- 是 shape、数据路径、同步、并行度还是数值稳定性导致？
- 实验是否公平？
- 什么条件变化后，它可能重新有收益？

### 1.4 交付物

- 代码：复现脚本、microbench 或修改后的 kernel；
- 报告：六问逐项“结论 + 证据”；
- 答辩：10 分钟陈述，5 分钟提问。

---

## 2. 开始 C1 前，M0–M4 到底需要会多少

你不必先把 M4.3 每一行都写熟，才能开始 C1。但下面这些概念必须会说。

| 前置模块 | C1 中对应的用途 | 最低掌握要求 |
|---|---|---|
| M0 | 峰值、Roofline、机器平衡点 | 会区分 compute roof、memory roof 和“两个都低的延迟/并行度问题” |
| M1 | SM80 `mma.sync` | 知道一个 warp 协作执行 `m16n8k16`，结果 fragment 在寄存器 |
| M2 | descriptor、swizzle、SMEM | 知道 Tensor Core 对布局有契约，不能只替换指令名 |
| M3 | tcgen05、TMEM、mbarrier | 会解释单线程发射、异步完成、TMEM 分配、commit/wait、读取与释放 |
| M4.2 | TMA | 知道一个线程可发起 GMEM→SMEM bulk copy，并通过 barrier 通知完成 |
| M4.3 | 多级 pipeline | 知道 full/empty stage 的所有权，以及 prime/steady/drain |
| M4.5 | skinny GEMM | 知道形状太瘦时，Tensor Core 峰值不等于有效性能 |

最重要的一条联系是：

```text
M3：一条 tcgen05 指令怎样算对
M4：怎样持续给 tcgen05 喂数据
C1：这个真实算法的形状，究竟值不值得使用 tcgen05
```

M5 的 FP4/block scaling 不是 C1 的前置要求。C1 讨论的 `rescale` 是指数动态范围的重标定，不等于 M5 的低精度 block scaling。

---

## 3. 从直觉理解 KDA：一个会遗忘、会纠错的键值记忆

### 3.1 状态矩阵在干什么

设 head dimension 为 `D=128`，每个 head 维护一个状态矩阵：

$$
S_t\in\mathbb{R}^{D\times D}.
$$

可以把它想象成一个在线更新的小型线性模型：

- key `k_t` 是输入特征；
- value `v_t` 是希望记住的目标；
- `S_t` 是当前“记忆”；
- 用 `k_t` 查询状态，得到对 `v_t` 的预测；
- 新 value 与旧预测的差就是 delta；
- 用 delta 修正状态。

### 3.2 逐 token 的递推公式

源码 `fla_kda_ref/naive.py` 的核心逻辑可写为：

$$
\widetilde S_t = D_t S_{t-1},
$$

$$
r_t = v_t-k_t^T\widetilde S_t,
$$

$$
S_t = \widetilde S_t+\beta_t k_t r_t^T,
$$

$$
o_t = q_t^T S_t.
$$

其中：

- $D_t=\operatorname{diag}(e^{g_t})$ 是逐维衰减；
- 激活后的 $g_t\in[-5,0]$，所以 $e^{g_t}\in[e^{-5},1]$；
- $\beta_t=\sigma(\text{beta\_logit}_t)\in(0,1)$ 控制这次纠错写入多强；
- q、k 在 kernel 内做 L2 normalization；
- q 还乘 `scale=1/sqrt(D)`。

一句话理解：

> 先让旧记忆按维度遗忘，再用“真实 value − 旧记忆预测”做一次 delta-rule 更新，最后用 q 读取新状态。

### 3.3 为什么不能把所有 token 完全并行

第 t 个 token 使用 $S_{t-1}$，而 $S_{t-1}$ 又由前一个 token 更新得到：

```text
S0 → S1 → S2 → S3 → ... → ST
```

这是一条真实的数据依赖链。不能简单地让 8192 个 CTA 各算一个 token，因为后面的 CTA 不知道前面的最终状态。

chunk 算法所做的是：

- 在一个 chunk 内，把 16 次更新代数展开成矩阵运算；
- chunk 内部获得 Tensor Core 友好的并行计算；
- chunk 与 chunk 之间仍按顺序传递状态。

所以 chunking 是在“数学并行性”和“递推依赖”之间折中。

---

## 4. 从逐 token 递推到 CHUNK=16 的矩阵形式

令一个 chunk 含 C 个 token，当前实现固定 `C=16`。

对 chunk 内第 i 个 token，定义累计门控：

$$
G_i=\sum_{j=0}^{i}g_j.
$$

源码为了使用 `ex2.approx.ftz.f32`，实际保存的是 base-2 指数：

$$
G_i^{(2)}=\frac{G_i}{\ln 2},
\qquad
2^{G_i^{(2)}}=e^{G_i}.
$$

这就是 `flash_kda.cpp` 中：

```cpp
gate_scale = lower_bound * 1.4426950408889634;
```

以及 `fwd_kernel1.cuh` 中直接调用 `ex2_approx_ftz_f32(g)` 的关系。`1.442695...=1/ln(2)`。

### 4.1 K1 构造的五类衰减张量

忽略 layout 后，可把 K1 的核心中间量理解为：

$$
K_d[i]=k_i e^{G_i},
$$

$$
Q_d[i]=q_i e^{G_i}/\sqrt D,
$$

$$
K_{inv}[i]=k_i e^{-G_i},
$$

$$
K_r[i]=k_i e^{G_C-G_i},
$$

以及 chunk 总衰减 $e^{G_C}$。

形象地说：

- `k_decayed`、`q_decayed`：把 token 放到 chunk 起点的坐标系；
- `k_inv`：撤销累计衰减，用于构造 token 间关系；
- `k_restored`：把 token 的写入搬到 chunk 终点的坐标系；
- `g_total`：旧状态跨过整个 chunk 后应该剩多少。

### 4.2 L 与 Neumann 逆

chunk 内第 i 次 delta 更新会受到更早 token 的影响。这些因果关系形成严格下三角矩阵 L：

$$
L=\operatorname{tril}(K_dK_{inv}^T,-1),
$$

再按每一行乘激活后的 beta。

由于 L 严格下三角，C×C 情况下必有：

$$
L^C=0.
$$

所以：

$$
(I+L)^{-1}=I-L+L^2-L^3+\cdots+(-L)^{C-1}.
$$

这不是“无穷级数近似”，而是因为 $L^C=0$ 后**有限项精确终止**。

当前 C=16 的源码没有顺序做 15 次乘法，而是使用乘积分解：

$$
(I+L)^{-1}
=(I-L)(I+L^2)(I+L^4)(I+L^8).
$$

`utils.cuh::neumann_inv_fused_1warp` 对每个幂做两次矩阵乘：

1. `L²=L×L`，然后 `INV=INV+INV×L²`；
2. `L⁴=L²×L²`，然后 `INV=INV+INV×L⁴`；
3. `L⁸=L⁴×L⁴`，然后 `INV=INV+INV×L⁸`。

一共 6 个 `16×16×16` GEMM。

### 4.3 chunk 级更新

K2 对一个 chunk 大致执行：

$$
R=\beta\odot(V-K_dS),
$$

$$
U=(I+L)^{-1}R,
$$

$$
O=Q_dS+M_{qk}U,
$$

$$
S_{next}=\operatorname{diag}(e^{G_C})S+K_r^TU.
$$

这里的 `Mqk` 是带因果 mask 的 `Q_d K_inv^T`。

不必死记每个转置。答辩时只要能说明：

1. K1 把 chunk 内部的 token-token 依赖预计算成 `INV` 和 `Mqk`；
2. K2 用当前状态算 residual 和 output；
3. K2 在 chunk 末更新状态，供下一个 chunk 使用。

---

## 5. FlashKDA 两个 kernel 的真实分工

### 5.1 总数据流

```text
q, k, g, beta
      │
      ▼
K1: _flash_kda_fwd_prepare
每个 (sequence, head, chunk) 独立
      │
      ├─ k_decayed  [16,128] bf16
      ├─ q_decayed  [16,128] bf16
      ├─ k_restored [16,128] bf16
      ├─ g_total    [128]    fp32
      ├─ INV        [16,16]  bf16
      └─ Mqk        [16,16]  bf16
              写入 global workspace
      │
      ▼
K2: _flash_kda_fwd_recurrence
每个 (sequence, head) 一个 CTA，按顺序扫 chunk
      │
      ├─ 读 v、beta 和 K1 workspace
      ├─ 维护 [128,128] bf16 on-chip state
      ├─ 计算 residual、U、output
      └─ 更新 state，进入下一个 chunk
```

### 5.2 K1：高并行 prepare

启动网格在源码中是：

```cpp
dim3 grid_k1(total_tiles, H);
```

逻辑上等价于 `N × H × num_chunks`。

以固定长度 `T=8192,H=96,C=16` 为例：

$$
\text{chunks/head}=8192/16=512,
$$

$$
\text{K1 CTAs}=512\times96=49152.
$$

因此 K1 几乎不缺 CTA 级并行性。

每个 K1 CTA 完成：

1. TMA 一次性载入 q、k、g、beta、dt_bias；
2. q/k L2 normalization；
3. gate activation、chunk 内 cumsum；
4. 计算正向/反向指数与四种衰减 q/k；
5. 两个 `16×128 × 128×16` GEMM，得到 L 与 Mqk；
6. 6 个 `16×16×16` FP16 MMA，得到 INV；
7. TMA 把六个中间量写到 workspace。

K1 使用 `__launch_bounds__(256,8)`，通过 shared-memory union 复用生命周期不重叠的缓冲区，目标是提高每 SM 的驻留 CTA 数。

### 5.3 K2：低并行、长递推

启动网格是：

```cpp
dim3 grid_k2(N, H);
```

固定 `B=N=1,H=96` 时只有 96 个 CTA。每个 CTA 内部串行处理 512 个 chunk。

K3 TP8 时，应按每卡头数：

$$
H_{gpu}=96/8=12.
$$

若每卡 `N=1`，K2 只有 12 个 CTA。它可能远远不能填满整张 B300。

K2 每 CTA 共 192 个线程：

- 4 个 MMA warp，共 128 个计算线程；
- 1 个 TMA load warp；
- 1 个 TMA store warp。

每个 chunk 的 MMA 阶段是：

1. `k_decayed @ state` 与 `q_decayed @ state`；
2. `R=(v-k@state)*beta`；
3. `U=INV@R`；
4. `out=q@state+Mqk@U`；
5. `state=state*g_total+k_restored.T@U`。

### 5.4 K2 已经包含你还没完全学完的 M4.3

K2 不是“先全搬完，再全计算”。它已经是一个多级 producer-consumer pipeline：

- input stages = 3；
- output stages = 2；
- load warp 是 producer；
- 4 个 MMA warp 是 input consumer、output producer；
- store warp 是 output consumer。

一个 input stage 包含：

| 数据 | 字节 |
|---|---:|
| v `[16,128]` | 4096 |
| beta TMA 区 `[32]` | 64 |
| k_decayed、q_decayed、k_restored | 12288 |
| g_total `[128]` fp32 | 512 |
| INV、Mqk | 1024 |
| 合计 | **17984** |

三个 input stage 约 53,952 B；两个 output stage 约 8,192 B；BF16 state 是 32,768 B。

因为 pipeline buffer 与 FP32 state 边界转换 buffer 使用 union，K2 的主要共享内存约为：

$$
32768+\max(53952+8192,65536)\approx98304\text{ B},
$$

再加少量 pipeline/barrier 元数据。

这就是 M4.3 的核心：

```text
load warp:   stage 0 写满 → stage 1 写满 → stage 2 写满 → 等 empty
MMA warps:   等 full 0 → 计算 0 → 释放 0 → 等 full 1 → ...
store warp:  等 output full → TMA store → 释放 output stage
```

你暂时不需要能默写 CUTLASS pipeline API，但必须理解：

- `producer_acquire`：确认这个槽空了，生产者取得所有权；
- TMA barrier：确认数据真正到达 SMEM；
- `consumer_wait`：消费者等这个槽变 full；
- `consumer_release`：消费者用完，槽重新变 empty；
- phase/parity：循环复用同一 barrier 时区分“上一轮”与“下一轮”。

这与 M3.3 的 mbarrier phase bug 是同一类问题。

### 5.5 为什么拆成 K1/K2

早期全融合版本把 K1 和 K2 绑在同一个低并行网格上。结果是本来可以按 chunk 并行的 K1，也被迫只剩 `N×H` 个 CTA。

官方 deep-dive 给出的结果是：拆成两个 kernel 后端到端至少提升 15%。

代价是 K1 与 K2 之间必须经过一个很大的 global workspace。这个代价将在 Roofline 分析中非常重要。

---

## 6. 当前 SM80 MMA 主路径的指令与 FLOP 清点

### 6.1 源码证据

`fwd_kernel1.cuh` 和 `fwd_kernel2.cuh` 都显式构造：

```cpp
SM80_16x8x16_F32BF16BF16F32_TN
```

Neumann inverse 使用：

```cpp
SM80_16x8x16_F16F16F16F16_TN
```

所以源码层面已经可以证明算法路径以 SM80 MMA atom 为核心。最终报告仍应补 SASS，因为题目明确要求在 B300 二进制上确认。

### 6.2 每 chunk 的 FLOP

令 `C=CHUNK`、`D=128`。

K1：

- L、Mqk 两个 GEMM：$4C^2D$ FLOP；
- Neumann inverse：C=16 时 6 个 C³ GEMM，即 $12C^3$ FLOP。

K2：

- `k@state` 与 `q@state`：$4CD^2$ FLOP；
- `INV@R` 与 `Mqk@U`：$4C^2D$ FLOP；
- `k_restored.T@U`：$2CD^2$ FLOP。

代入 C=16、D=128：

| 部分 | FLOP/chunk |
|---|---:|
| K1：L + Mqk | 131,072 |
| K1：Neumann inverse | 49,152 |
| K1 合计 | **180,224** |
| K2：双 state GEMM | 1,048,576 |
| K2：INV/Mqk 两个小 GEMM | 131,072 |
| K2：state delta GEMM | 524,288 |
| K2 合计 | **1,703,936** |
| 两 kernel 合计 | **1,884,160** |

每 token、每 head：

$$
1,884,160/16=117,760\ \text{FLOP}.
$$

这里没有计入 normalization、sigmoid、exp、逐元素 FMA 等 CUDA Core 工作，所以是 Tensor Core 主计算的近似值，不是完整指令级 FLOP。

### 6.3 每 chunk 的 `mma.sync` atom 数

一个 `m16n8k16` atom 计算：

$$
2\times16\times8\times16=4096\ \text{FLOP}.
$$

因此：

| 部分 | `m16n8k16` atom/chunk |
|---|---:|
| K1 两个主 GEMM | 32 |
| K1 inverse | 12 |
| K1 合计 | 44 |
| K2 合计 | 416 |
| 总计 | **460** |

检查：

$$
460\times4096=1,884,160.
$$

官方 `T8192,H96` 一次 forward 含：

$$
512\times96\times460=22,609,920
$$

个 SM80 MMA atom。

这个数量看起来很大，但不能直接推出 compute-bound；数据流量和并行度同样重要。

---

## 7. 讨论点 1：为什么 CHUNK=16？32/64 谁先坏、代价多大？

## 7.1 数值范围：最先出现的是指数失效

激活后的自然对数门控范围是：

$$
g_t\in[-5,0].
$$

最坏情况下，C 个 token 的累计和为：

$$
G_C=-5C.
$$

内核同时需要：

$$
e^{G_i}\quad\text{和}\quad e^{-G_i}.
$$

由于使用 `ex2.approx.ftz.f32`：

- FP32 最小 normal 约为 $2^{-126}$，自然对数约 `-87.34`；
- `.ftz` 会把更小的结果 flush to zero；
- BF16 最大有限值约 `3.39e38`，自然对数约 `88.72`。

数值表：

| CHUNK | 最坏 $G_C$ | $e^{G_C}$ | $e^{-G_C}$ | 结论 |
|---:|---:|---:|---:|---|
| 16 | -80 | `1.80e-35` | `5.54e34` | 两边仍可表示 |
| 32 | -160 | `3.26e-70` | `3.07e69` | 前者归零，后者溢出 |
| 64 | -320 | `1.06e-139` | `9.42e138` | 更严重地失效 |

按每 token 都取 -5 的离散最坏情况：

- 第 17 个 token：累计 -85，仍安全；
- 第 18 个 token：累计 -90，`e^-90` 已低于 FTZ 阈值，`e^90` 也超过有限上界。

所以 16 不是一个随意的小整数，而是“小于约 17.5 个最坏门控”的最大常用 2 的幂。

**候选结论：** CHUNK 从 16 直接改到 32 时，首先破坏的是当前无重标定指数表示，而不是 shared memory 容量，也不是 Neumann 公式本身。

### 7.2 为什么实际数据没有每次都等于 -5，也不能忽略这个问题

门控由：

$$
g=\text{lower\_bound}\cdot\sigma(e^{A_{log}}(g_{raw}+dt\_bias))
$$

得到。

很多 token 可能远离 -5，因此平均情况不一定溢出。但 kernel 正确性不能建立在“模型大概不会遇到极值”上，尤其是：

- `g_raw+dt_bias` 很大时 sigmoid 会饱和到 1；
- 长序列中极值更容易出现；
- BF16 输入会让接近饱和区的分布更离散；
- 反向指数 `e^{-G}` 一旦变 inf，后续乘 0 还可能生成 NaN。

### 7.3 CHUNK=32/64 怎样修：subchunk anchor/rescale

可行方法是每 16 个 token 选一个 anchor：

$$
\widehat G_i=G_i-a_b,
$$

让每个 subchunk 内的差值保持在安全范围，再把 subchunk 之间的尺度单独传播。

FLA 的 C=64 路径也按 `sub_chunk_size=16` 计算局部块和块间关系。这说明：

> 大 CHUNK 没有消灭 16 的数值尺度，它只是把多个安全的 16-token 子块组织成一个更大的块。

代价包括：

- 额外保存/加载 anchor 或 block-end gate；
- 更多 `exp2` 和逐元素缩放；
- off-diagonal subchunk 组合；
- 更复杂的 inverse/triangular solve；
- 更多同步和寄存器压力；
- 尾块处理更复杂。

因此“大 CHUNK”已经不是简单改一个模板参数。

## 7.4 Neumann inverse 的定量代价

对 2 的幂 C，当前乘积分解需要：

$$
p(C)=2(\log_2 C-1)
$$

个 C×C×C 矩阵乘。

| C | 矩阵乘个数 | inverse FLOP/chunk | 相对 C16/chunk | inverse FLOP/token | 相对 C16/token |
|---:|---:|---:|---:|---:|---:|
| 16 | 6 | 49,152 | 1× | 3,072 | 1× |
| 32 | 8 | 524,288 | 10.67× | 16,384 | 5.33× |
| 64 | 10 | 5,242,880 | 106.67× | 81,920 | 26.67× |

矩阵元素数也按 C² 增长：

- 16×16：256；
- 32×32：1024，4×；
- 64×64：4096，16×。

所以即使数值问题已经通过 rescale 修好，直接对整个 64×64 L 使用同样的 dense Neumann 路径也会非常昂贵。FLA 选择 16×16 子块和块级合并，正是为了避免这个代价。

## 7.5 总 FLOP 与 workspace 代价

若暂时假设 C32/C64 仍使用同类 dense 公式，则每 token 的主 Tensor Core FLOP 为：

$$
F/token=6D^2+8CD+4(\log_2C-1)C^2.
$$

代入 D=128：

| CHUNK | 主 FLOP/token/head | 相对 C16 |
|---:|---:|---:|
| 16 | 117,760 | 1.00× |
| 32 | 147,456 | 1.25× |
| 64 | 245,760 | 2.09× |

workspace 每 tile：

$$
W(C)=3\times C\times D\times2+D\times4+2\times C^2\times2.
$$

| CHUNK | workspace/tile | workspace/token/head | 相对每 token |
|---:|---:|---:|---:|
| 16 | 13,824 B | 864 B | 1.00× |
| 32 | 29,184 B | 912 B | 1.056× |
| 64 | 66,048 B | 1,032 B | 1.194× |

大 CHUNK 减少 chunk 数和固定循环开销，但：

- inverse FLOP 急剧上升；
- C² 中间量让 workspace/token 反而增加；
- 数值上必须引入 subchunk rescale；
- chunk 越大，单个序列的 K2 循环次数减少，但单轮临界路径更长。

### 7.6 讨论点 1 的最终回答模板

> CHUNK=16 的第一性原因是数值范围：`lower_bound=-5` 时最坏累计门控为 -80，正反指数仍落在 BF16/FP32-FTZ 可表示范围；CHUNK=32 在第 18 个极端 token 左右就会发生下溢和上溢。第二个原因是严格下三角 16×16 L 的 Neumann 逆只需 6 个小 GEMM；若直接扩到 32/64，同类逆的每 token FLOP 分别约为 5.33×/26.67×。第三个原因是 16×16 与 SM80 `m16n8k16` atom 无 padding 对齐。因而 32/64 不是改模板参数，而是需要 16-token subchunk、rescale 和分块三角求解的算法改写。

---

## 8. 讨论点 2：tcgen05 最小 tile 与 CHUNK=16 匹配吗？

## 8.1 先把“更强指令”拆成三个问题

1. 数学矩阵的形状能否无 padding 表达？
2. 数据当前在哪里：寄存器、SMEM 还是 TMEM？
3. prologue、commit/wait、TMEM load/store/dealloc 的固定成本能否被足够多的计算摊薄？

只看第 1 条都不够，更不能只比较理论峰值。

## 8.2 M 维不匹配

课程 M3/M4 验证过的标准 `cta_group::1 kind::f16` 路径是 M=128；标准 dense tcgen05 的较小单 CTA M 形状仍大于 16。即使用可用的 M=64 变体，CHUNK=16 也只有 25% 的 M 行是有效计算；按 M=128 路径则只有 12.5%。

因此：

| 数学 M | tcgen M | 有效行利用率 | issued/useful FLOP 膨胀 |
|---:|---:|---:|---:|
| 16 | 64 | 25% | 4× |
| 16 | 128 | 12.5% | 8× |

最终应以 B300 所用 PTX/CUTLASS 版本支持的具体 shape 列表为准；但“16 不能无浪费映射到课程标准 dense tcgen 路径”的结论不变。

## 8.3 FlashKDA 的各个 GEMM 哪些匹配

| 运算 | 数学形状 | 占总 FLOP | tcgen05 形状判断 |
|---|---|---:|---|
| K1：L、Mqk | `16×128 @ 128×16` | 6.96% | M=16，不匹配 |
| K1：inverse | `16×16 @ 16×16` | 2.61% | M=16，不匹配，且工作太小 |
| K2：k/q @ state | 两次 `16×128 @ 128×128` | 55.65% | 最大热点，但 M=16，不匹配 |
| K2：INV/Mqk @ U | 两次 `16×16 @ 16×128` | 6.96% | M=16，不匹配 |
| K2：state delta | `128×16 @ 16×128` | 27.83% | **天然 `m128n128k16`，匹配** |

这张表给出一个重要的、更细的结论：

> “FlashKDA 不适合 tcgen05”过于绝对。更准确的说法是：大多数以 chunk 为 M 的 GEMM 不适合直接替换；状态更新 GEMM 则是一个值得单独 microbench 的自然 SM100 切入点。

## 8.4 为什么不能简单把 4 个 head 拼成 M=64

表面看，4 个 head × 16 token = 64 行，似乎可以填满 M64。

但普通 GEMM：

$$
C=AB
$$

的所有 M 行共享同一个 B。不同 head 的状态矩阵 $S_h$ 不相同：

$$
K_{d,h}S_h.
$$

把四个 head 的 K 沿 M 拼起来后，没有一个共同 B 能同时代表四个不同的状态。除非：

- 构造巨大的 block-diagonal B，产生大量零计算；或
- 仍发四次独立 MMA，只是塞进同一 CTA；或
- 硬件/库提供真正的 batched/grouped MMA 机制。

同样，不能把同一 head 的连续四个 chunk 拼起来，因为后一个 chunk 所用状态依赖前一个 chunk 的更新结果。

所以“多 head 进一个 CTA”可能减少启动和 load warp 开销，却不能自动把四个独立小 GEMM 合成一个高利用率 dense GEMM。

## 8.5 只替换指令为什么可能更慢

SM80 路径的优势：

- `m16n8k16` 与 C=16 完全匹配；
- accumulator 在寄存器，可直接接下一段标量/矩阵逻辑；
- `MOVM_T` 可以在寄存器中重排 U；
- 不需要 TMEM alloc/dealloc；
- 不需要额外 commit→mbarrier→tcgen05.ld 生命周期；
- 一个 CTA 中四个 warp 可以分别处理不同的 16×16 输出块。

tcgen05 路径会新增：

- 描述符与 TMEM 地址管理；
- 异步提交与等待；
- TMEM 结果读回寄存器或转存；
- padding 的无效 FLOP；
- 可能更高的 SMEM/TMEM 占用和更低 occupancy；
- 重新组织 K2 的 register-resident 中间值。

所以“tcgen 指令条数更少”不代表“端到端周期更少”。

## 8.6 唯一自然匹配的 phase 也不是免费替换

`k_restored.T @ U` 是 `m128n128k16`，tcgen05 可以很好地表达。

但当前状态更新还包含：

$$
S_{next}=S\odot g_{total}+\Delta S.
$$

tcgen 的 `D` 在 TMEM，而旧状态当前在 BF16 SMEM。你需要选择：

1. 把 tcgen 结果读回寄存器，和旧状态逐元素 FMA，再写回 SMEM；
2. 尝试让状态驻留 TMEM，但下一轮 `k@state` 的取数方向、逐列门控缩放和输出转置都要重构；
3. 用额外 GEMM 表示门控缩放，但可能让计算量和数据流更差。

因此它是“值得实验的局部改写”，不是一行替换。

## 8.7 推荐的 microbench

至少做三组：

| 组 | 真实含义 | SM80 baseline | tcgen 版本 |
|---|---|---|---|
| A | L/inverse atom | `m16n16k16` | pad 到最小合法 M |
| B | K2 最大热点 | `m16n128k128` | M padding，K 分 8 个 k16 |
| C | state delta | `m128n128k16` | 自然 tcgen05 tile |

每组做两种口径：

1. **instruction-only steady state**：输入已在 SMEM/TMEM，循环很多轮，观察纯指令吞吐；
2. **honest end-to-end**：包含 SMEM layout、TMEM alloc、commit/wait、TMEM load、结果写回。

记录：

- 总周期/总时间；
- useful TFLOPS：只按真实数学矩阵计 FLOP；
- issued TFLOPS：包含 padding 计 FLOP；
- active CTA/SM；
- Tensor pipe 利用率；
- barrier/scoreboard stall；
- SMEM、TMEM、register 使用；
- 随机输入正确性误差。

如果只报 issued TFLOPS，M16→M128 padding 可能看起来“峰值很高”，但 87.5% 都是无用计算，结论会误导。

### 8.8 讨论点 2 的最终回答模板

> CHUNK=16 与标准 tcgen05 dense M tile 不直接匹配。按 M64 至少产生 4× issued/useful FLOP，按课程 M128 路径产生 8×；同时需要 TMEM 分配、异步完成等待和结果读回。FlashKDA 约 72% 的主 FLOP 都位于 M=16 的 GEMM 中，因此不改算法只换指令大概率没有端到端收益。例外是占总 FLOP 约 27.8% 的 `m128n128k16` 状态更新，它无 padding 匹配 tcgen05，应该单独 microbench；是否值得集成仍由包含 TMEM 数据路径的 B300 实测决定。

---

## 9. 讨论点 3：chunk 间有状态依赖，并行度还能从哪里来？

## 9.1 当前的并行性地图

| 维度 | K1 | K2 | 原因 |
|---|---|---|---|
| sequence N | 可并行 | 可并行 | 每个序列状态独立 |
| head H | 可并行 | 可并行 | 每个 head 状态独立 |
| chunk | 可并行 | **不可直接并行** | K2 后一 chunk 依赖前一状态 |
| token within chunk | 矩阵化并行 | 矩阵化并行 | 通过 chunk 代数展开 |
| D/K/V tile | warp 间并行 | warp 间并行 | D=128 可切输出列块 |

### 9.2 官方 benchmark 已经暴露 K2 并行度问题

GB200、`T_total=8192,H=96,D=128`：

| 序列组织 | K2 CTA 数 | 每 CTA chunk 数 | FlashKDA 延迟 |
|---|---:|---:|---:|
| Fixed `[8192]` | 96 | 512 | 1.0087 ms |
| Varlen 6 条 | 576 | 不均匀 | 0.8597 ms |
| Varlen `[1024]×8` | 768 | 64 | 0.7064 ms |

总 token 和 head 相同，但 8 条序列版本快约 30%。这不是数学 FLOP 变少，而是：

- K2 CTA 数增加 8×；
- 每条依赖链缩短为原来的 1/8；
- 调度和延迟隐藏明显改善。

H=64 时 fixed 0.9247 ms、`1024×8` 0.4811 ms，差距更大。这是讨论点 3 最现成的官方证据。

### 9.3 候选 A：多个 head 放进一个 CTA

可能收益：

- 一个 load/store warp 服务多个 head；
- amortize CTA、pipeline、barrier 固定开销；
- 若某些 operand 可共享，可能提高复用；
- 可以让同一 CTA 交错推进多个独立状态链，增加 ILP。

反例与风险：

- 不同 head 的 state、q/k/v/g 都不相同，核心 B operand 不能共享；
- state 每 head 32 KiB BF16，两个 head 就 64 KiB；
- 现有 K2 pipeline 已约 98 KiB SMEM，再放多 state 很容易压低 occupancy；
- CTA 数变少：TP8 下 H=12，2 head/CTA 只剩 6 个 CTA，4 head/CTA 只剩 3 个；
- dense MMA 不能直接把不同 B 的独立 head 拼成一个 M64/M128 GEMM。

什么时候可能成立：

- 不是单纯减少 CTA，而是在一个 persistent CTA 内交错多条独立链；
- TMEM/SMEM 能放下多个轻量 state 表示；
- N 足够大，使全卡仍有足够 worker；
- 实测表明单 head CTA 的 stall 主要是依赖延迟，而不是资源带宽。

### 9.4 候选 B：persistent kernel

设计想法：启动固定数量 worker CTA，每个 worker 不退出，而是从任务队列中领取 `(sequence,head)` 工作。

可能收益：

- 减少反复启动和初始化开销；
- varlen 时长序列/短序列可以动态负载均衡；
- 可让 worker 在多条独立状态链间交错；
- 可以把某些 descriptor/pipeline/TMEM 生命周期跨任务复用。

反例与风险：

- fixed `N=1,H=12` 总共仍只有 12 条独立链，persistent 不会凭空创造第 13 条；
- 单个 head 内 chunk 依赖仍存在；
- 长任务会让少数 CTA 长期占用 SM；
- work queue、原子操作和状态切换有额外成本；
- 当前单次 K2 已经一个 CTA 处理完整序列，本身就具有“沿 chunk 持续运行”的 persistent 特征。

因此 persistent 的真正价值更可能是**跨多个 sequence/head 的动态调度**，不是消除同一序列的递推。

### 9.5 候选 C：2-CTA / cluster

可能想法：两个 CTA 共同处理一个 head，把 D=128 的 value/output 维切成两半，或共同使用某个 B tile。

可能收益：

- 增加一个 head 内的线程级/CTA 级并行；
- 某个操作数可通过 cluster shared memory 复用；
- 对天然的大 M tile 可使用 `cta_group::2`。

反例与风险：

- 两个 CTA 必须同步推进同一 chunk，最慢的一方决定速度；
- state 更新后下一 chunk 才能开始，cluster barrier 进入临界路径；
- KDA 的每 head state 并不是 M3.4 中两个 CTA 可自然共享的同一个 B tile 情形；
- cluster 会限制调度位置和驻留，数据中心 GPU 上也不保证更快；
- `cta_group::2` 常用于扩大 M，而 FlashKDA 的主要问题恰恰是 M=16 太小。

### 9.6 候选 D：对 chunk recurrence 做 scan

理论上，若每个 chunk 都能表示成一个仿射状态变换：

$$
S_{next}=A_cS+B_c,
$$

且组合满足结合律，就可以用 prefix scan 并行合并 chunk。

这是最根本的并行化方向，但对 KDA 并不免费：

- $A_c$ 是逐维衰减，比较简单；
- $B_c$ 又依赖当前 state 产生的 residual；
- 需要重新推导能否把所有 state 依赖封装成可结合的变换；
- 中间表示可能从 O(D²) 膨胀到更大；
- scan 的额外读写可能抵消收益。

这是研究路线，不适合作为几小时内的首个挑战实现。

### 9.7 推荐结论

短期：

- 不要用“多 head 拼 M”作为默认答案；不同 head 的 B/state 不同。
- 先把 `N=1/2/4/8`、`H=12/24/48/96` 做二维 sweep，证明并行度拐点。
- 用 NCU 看固定 H 下 active SM、waves per SM、eligible warps 和 Tensor pipe。

中期最有价值的是：

- varlen/paged serving 下的 persistent work queue；或
- 一个 CTA 交错推进多条独立 sequence/head chain，但必须保持全卡 CTA 数。

2-CTA 更像反例：它扩大单任务资源占用，却没有解决 chunk 依赖和 M16 mismatch。

---

## 10. 讨论点 4：compute-bound 还是 memory-bound？

## 10.1 先做纸面流量模型

C=16、D=128 时，每个 `(chunk,head)` 的 K1→K2 workspace 为：

| 中间量 | 字节 |
|---|---:|
| k_decayed | 4096 |
| q_decayed | 4096 |
| k_restored | 4096 |
| g_total | 512 |
| INV | 512 |
| Mqk | 512 |
| 合计 | **13,824 B** |

K1 写一次、K2 读一次，所以仅 workspace 就是：

$$
27,648\text{ B/chunk/head}.
$$

按源码 TMA tile 计算，最低逻辑 global 流量近似为：

| 部分 | B/chunk/head |
|---|---:|
| K1 读 q/k/g | 12,288 |
| K1 读 beta tile、dt_bias、A_log | 约 580 |
| K1 写 workspace | 13,824 |
| K2 读 v、beta、workspace | 17,984 |
| K2 写 output | 4,096 |
| 合计 | **约 48,772** |

这还没有加入每个 sequence/head 的 initial/final state：

- BF16 state load：32,768 B；
- BF16 state store：32,768 B。

固定 T8192 时它们均摊很小；短序列 serving 时会更明显。

### 10.2 官方形状的纸面 Roofline 点

`T=8192,H=96,C=16`：

$$
\text{chunks}=512\times96=49152.
$$

主 Tensor Core 工作量：

$$
1,884,160\times49152=92.61\text{ GFLOP}.
$$

最低逻辑流量：

$$
48,772\times49152\approx2.397\text{ GB}.
$$

所以：

$$
AI\approx\frac{92.61\text{ GFLOP}}{2.397\text{ GB}}
\approx38.6\text{ FLOP/B}.
$$

其中 workspace 一写一读：

$$
13,824\times2\times49,152
\approx1.359\text{ GB},
$$

约占最低逻辑流量的 56.7%。

官方 GB200 固定形状延迟 1.0087 ms。若用纸面量除以该延迟：

- useful Tensor work rate 约 91.8 TFLOP/s；
- logical byte rate 约 2.38 TB/s。

第二个数不是“实测 HBM 带宽”，因为 dt_bias 等可能命中 cache，写回策略和 cache line 也会改变真实 DRAM byte；它只是说明这个 kernel 的全局数据通路非常重。

### 10.3 为什么不能给整个 FlashKDA 贴一个简单标签

K1：

- 49,152 个 CTA，并行度充足；
- 做 normalization、sigmoid、exp、L/Mqk、inverse；
- 每 CTA 工作较小；
- 写大量 workspace；
- 可能是 instruction/compute、SMEM 或 workspace store 混合限制。

K2：

- CTA 少、每 CTA 循环长；
- 3-stage TMA pipeline；
- 每 chunk 做较多 MMA；
- chunk 之间有 state 依赖；
- 可能是 HBM/L2、Tensor pipe、同步、依赖延迟或全卡 underfill。

因此正确回答形式应是：

```text
K1 是什么 bound？证据是什么？
K2 是什么 bound？证据是什么？
端到端谁占时间？改 tcgen 后 Amdahl 上限是多少？
```

## 10.4 必看的 NCU 项目

不同 NCU/CUDA 版本的 metric 全名可能变化，先使用 `--set full`，再从报告中提取以下概念。

### 时间和占比

- 每次 K1/K2 duration；
- 两者在端到端中的比例；
- 每个 shape 的调用次数；
- 若只加速 K1 或 K2，Amdahl 上限是多少。

### 计算管线

- SM throughput / SOL；
- Tensor pipe active/throughput；
- FP32 pipe：state FMA、normalization、exp/sigmoid；
- 实际 MMA 指令数；
- issue slot utilization。

常见 metric 候选：

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
smsp__inst_executed_pipe_tensor.sum
```

### 内存系统

- DRAM bytes 与 throughput；
- L2 bytes、hit rate、throughput；
- L1/SMEM throughput；
- TMA load/store 是否有效重叠。

常见候选：

```text
dram__bytes_read.sum
dram__bytes_write.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
lts__throughput.avg.pct_of_peak_sustained_elapsed
lts__t_sector_hit_rate.pct
l1tex__throughput.avg.pct_of_peak_sustained_elapsed
```

### 并行度和 occupancy

- grid size；
- waves per SM；
- theoretical/achieved occupancy；
- active warps；
- eligible warps/cycle；
- issued warps/cycle；
- register/CTA、SMEM/CTA。

### stall 原因

- long scoreboard：等远端内存；
- short scoreboard：常与 shared/TMA 依赖相关；
- barrier/membar：同步；
- not selected：有可运行 warp，但没被选中，通常说明 TLP 足；
- wait：指令依赖/固定延迟；
- no instruction：前端或控制流。

不要看到某一个 stall 百分比就下结论，要和 throughput、eligible warp、active SM 一起解释。

## 10.5 判定逻辑

| 观察 | 更支持的结论 |
|---|---|
| Tensor pipe 高，DRAM 低，eligible warp 足 | compute-bound |
| DRAM 接近峰值，long scoreboard 高 | HBM-bound |
| L2 高、DRAM不高，L2 hit/traffic 主导 | L2/cache-bandwidth-bound |
| Tensor 与 DRAM 都低，active SM 很少 | grid/并行度不足 |
| active SM 足但 eligible warp 很低，barrier 高 | pipeline/synchronization limited |
| occupancy 低且 register/SMEM 达上限 | resource limited |
| K2 fixed 慢、N×8 明显快 | 状态链长度/CTA 并行度是主要证据 |

## 10.6 与 M4.5 `in_proj_qkvgfab` 的联系

`in_proj_qkvgfab` 是生成 KDA 输入张量的投影 GEMM。它和 FlashKDA 本体不是同一个 kernel，但在真实模型中前后相邻：

```text
hidden states
  → in_proj_qkvgfab GEMM
  → q/k/v/g/beta 等张量
  → FlashKDA
```

M4.5 给你的启示是：

- 小 M/decode 时，投影 GEMM 也可能 shape-underutilized；
- KDA kernel 的 C=16 同样是“小 M”问题；
- 端到端优化不能只提高单个 kernel 的 issued TFLOPS；
- 若投影写出、FlashKDA 再读入大量中间张量，未来更激进的方向可能是 producer-consumer 融合或 layout 契约优化。

但本次报告不要把 `in_proj_qkvgfab` 的 GEMM Roofline 数字直接当作 FlashKDA 的 Roofline；两者 FLOP/byte 公式不同。

### 10.7 讨论点 4 的候选结论

> 纸面上 FlashKDA 官方形状的主计算约 92.61 GFLOP、最低逻辑流量约 2.40 GB，AI 约 38.6 FLOP/B；workspace 一写一读就占约 1.36 GB。因此其端到端很难被视为纯 Tensor Core compute-bound。更可能的细分是：K1 具有高 CTA 并行度但受小矩阵、标量指令和 workspace 写出影响；K2 受 workspace 读取、状态依赖和 `N×H` 网格限制。最终分类必须用 B300 上 K1/K2 分离的 Tensor/DRAM/L2 throughput、active SM、eligible warp 和 stall 数据确认。

---

## 11. 讨论点 5：怎样验证 BF16 recurrent state 的精度

## 11.1 一个非常容易误解的源码事实

FlashKDA API 接受 FP32 initial/final state，但这不等于内部递推使用 FP32 state。

源码路径是：

```text
FP32 initial state
  → TMA load 到 FP32 SMEM buffer
  → 转为 BF16 state_acc
  → 所有 chunk 之间仍以 BF16 保存 state
  → 结束后转回 FP32
  → FP32 final state
```

所以 benchmark 中的 `flash_kda (fp32 state)` 只测试**边界存储 dtype**，不是“真正的 FP32 on-chip recurrence”。

答辩时指出这一点，会比简单复述官方“内部测试通过”更有含金量。

## 11.2 验证要区分三层误差

### 层 A：实现一致性

当前 `tests/torch_ref.py` 逐步复刻：

- `tanh.approx` sigmoid；
- `ex2.approx.ftz`；
- BF16 cast 位置；
- FP16 accumulation 的 Neumann inverse；
- 每 chunk BF16 state round-trip。

kernel 与它 exact match，能证明“实现符合预定算法路径”，但不能证明预定的 BF16/approx 路径足够准确。

### 层 B：算子数值精度

使用 `fused_recurrent_kda` FP64 作为 gold，分别比较：

- FlashKDA BF16 state；
- FLA `chunk_kda`；
- 如果实现了实验版本，再比较真正 FP32 state；
- 可选：精确 sigmoid/exp 与 approx 版本。

### 层 C：模型质量

若有 K3 权重和推理环境，再比较：

- 最终 logits cosine/KL；
- top-1 token 一致率；
- 固定 prompt 的生成分叉位置；
- perplexity 或任务准确率；
- 多层 KDA 误差是否累积。

只有层 C 才能支撑“无可测模型质量损失”；没有权重时，应诚实写成“算子级验证通过”。

## 11.3 测试矩阵

### 序列长度

```text
16, 256, 1024, 8192, 32768
```

如果 FP64 gold 太慢，长序列只使用 H=1，并减少 case 数；不能因此只测 T=16。

### gate 分布

覆盖：

- 激活后接近 0：几乎不衰减，旧误差最容易长期保留；
- 激活后接近 -5：指数动态范围最危险；
- 常数 -1/-3/-5；
- 随机 uniform/normal；
- 每维不同的 `dt_bias`；
- 交替强衰减/弱衰减；
- A_log 的真实模型范围，如果能取得。

现有测试已包含 raw g 的 `-8,-4,0,8,U(-8,8),N(0,8)` 和多种 dt_bias，可直接扩展。

### beta 分布

- beta sigmoid 接近 0：几乎不写状态；
- beta=0.5；
- beta 接近 1：持续强纠错；
- 随机和交替极值。

### initial state

- 全零；
- 标准正态；
- 不同尺度 `1e-2,1,1e2`；
- 低秩、有结构、单一大 outlier；
- BF16 可精确表示与不可精确表示的值。

### shape

- fixed B=1；
- varlen 官方两组；
- H=1 做精度压力测试；
- H=12/96 做真实路径确认；
- 尾 chunk 长度 1、15、16。

## 11.4 误差指标

输出和 final state 都要报：

$$
\text{RMSE ratio}=\frac{\|x-\hat x\|_2}{\|x\|_2+\epsilon},
$$

- mean absolute error；
- max absolute error；
- P50/P90/P99/P99.9 absolute error；
- safe relative error：分母加阈值，避免参考值接近 0 时爆炸；
- cosine similarity；
- NaN/Inf count。

更重要的是按 token window 画误差随位置变化：

```text
token position → output RMSE ratio
chunk index    → state RMSE ratio
```

这样才能看出误差是稳定、有界，还是随递推长度积累。

## 11.5 阈值怎样定

现有测试对 FP64 reference 使用约 `0.005/0.006` 的容差，这是可复现起点，但不要无条件把它当作最终产品阈值。

BF16 unit roundoff 量级约为：

$$
2^{-7}=0.0078125.
$$

合理做法是：

1. 先复现官方容差；
2. 对所有压力 case 报分布，不只报 PASS；
3. 将 FlashKDA 与 FLA chunk 的误差放在同一张图；
4. 若 FlashKDA 不显著差于现有生产 baseline，可支持“算子级可接受”；
5. 若要声称模型质量无损，必须补模型级指标。

## 11.6 最关键的消融实验

理想情况下做四个版本：

| 版本 | state 保存 | state update | 意义 |
|---|---|---|---|
| Gold | FP64 | FP64 | 数学参考 |
| A | FP32 | FP32 FMA | 隔离 chunk 算法/approx 误差 |
| B | BF16 | FP32 FMA | 官方设计，隔离 state round-off |
| C | BF16 | BF16 accumulate | 证明 FP32 FMA 为什么必要 |

当前官方代码相当于 B。要真正回答“BF16 state 是否安全”，A 与 B 的差最重要。

### 11.7 讨论点 5 的回答模板

> 我们把实现一致性、算子精度和模型质量分开验证。现有 torch_ref exact match 只证明 kernel 正确复刻了 BF16/FP16/approx 路径；真正的精度结论需要对 FP64 fused recurrence，按序列长度、gate、beta、initial-state scale 和 varlen 形状做 sweep，同时记录 output 与 final state 的窗口化 RMSE、max error、cosine、NaN/Inf。API 的 FP32 state 仍会在 K2 开始时转成 BF16，因此不能作为内部 FP32 recurrence 对照；应另做真正 FP32 state 消融。没有模型权重时，只声称算子级误差不劣于 FLA baseline，不声称模型质量绝对无损。

---

## 12. 讨论点 6：如果你是作者，v2 出不出 SM100a 专版？

## 12.1 支持专版的论据

- K3 是核心模型，69/93 层使用 KDA，微小单层收益可能被大量调用放大；
- B300/GB200 是重要部署平台，架构专用优化有商业价值；
- `m128n128k16` state update 天然适配 tcgen05；
- TMEM 有机会降低 register pressure 或保存更高精度 accumulator；
- tcgen05 异步执行可能与 CUDA Core 的 gate/epilogue 重叠；
- K2 的 3-stage pipeline 已经具备架构专用数据流的基础；
- 可通过 runtime dispatch 保留旧路径，并不要求所有 GPU 都走 SM100。

## 12.2 反对专版的论据

- 主要热点是 M16 GEMM，与标准 tcgen05 tile 不匹配；
- padding 可能浪费 4×/8× issued FLOP；
- 当前 SM80 路径已在 GB200 上比 FLA 快 1.7–3.3×；
- SM80 `mma.sync` 路径可覆盖 Hopper、Blackwell 等多代 GPU；
- SM100 版会新增 TMEM、mbarrier、descriptor、layout 和 pipeline 维护成本；
- K2 的更大问题可能是 `N×H` 并行度，而不是单 CTA 算力；
- K1/K2 workspace 造成的全局流量不会因替换一条 MMA 自动消失；
- 需要维护更多 correctness/precision/shape 分支；
- 未来模型参数、CHUNK 或 head_dim 变化可能使专版迅速失去价值。

## 12.3 推荐的 ship gate

不是凭感觉决定，而是预先定义发布门槛：

1. 官方三个 shape 全部正确；
2. H=12 的 TP8 形状不能明显退化；
3. fixed、varlen、BF16/FP32 boundary state 均通过；
4. 长序列精度不劣于既定 FLA/FlashKDA tolerance；
5. 至少在核心 K3 形状上获得稳定的端到端收益；
6. 收益应明显超过噪声和维护成本，例如稳定 10–15% 以上，而不是一次测到 2%；
7. 没命中 SM100 条件时可靠 fallback 到现有路径。

阈值 10–15% 是工程决策建议，不是题目给定真理。最终可以根据团队的维护成本修改，但要在实验前写明，避免看完结果后移动标准。

## 12.4 推荐结论

> v2 可以保留一个实验性 SM100a dispatch 分支，但不应在没有证据时替换 portable SM80 主路径。第一阶段只针对天然匹配的 state-delta phase 或做 shape microbench；若完整 KDA 在 B300 的真实 K3/TP 形状上稳定提升至少约 10–15%，且长序列精度通过，再发布 SM100a 专版。若小 M padding、TMEM 往返和 K2 underfill 抵消收益，则官方停在 SM80 是合理决策，报告负结果即可。

---

## 13. 推荐挑战路线：先做“真实子形状指令替换”

这是当前时间压力下风险最低、证据密度最高的路线。

### 13.1 为什么不先做 CHUNK=32/64

它同时要求：

- rescale；
- subchunk/block triangular solve；
- workspace/layout 修改；
- K1/K2 两边模板修改；
- 重新做完整精度验证。

这更像数天到数周的工作，不适合第一个几小时。

### 13.2 为什么不先重写整个 K2 为 tcgen05

当前 K2 把中间结果留在寄存器，并用 `MOVM_T` 避免 SMEM round-trip。整体改成 TMEM 后，每个 phase 的 producer/consumer 契约都会变化，很容易出现：

- mbarrier phase 错误；
- TMEM 结果未完成就读取；
- SMEM/TMEM proxy 可见性错误；
- state layout 转置错误；
- pipeline stage 被过早复用。

先用 microbench 确认“硬件上值得”，再承担集成复杂度。

### 13.3 建议的代码目录

```text
team/c1_flashkda/experiments/
  mma_shape_microbench.cu
  run_microbench.sh
  parse_ncu.py              # 可选
  results/
    b300_env.txt
    baseline.csv
    microbench.csv
```

### 13.4 microbench 的公平性要求

- 相同输入 dtype：BF16；
- 相同数学输出：FP32 accumulation 后按需要 cast；
- 相同真实矩阵 shape；
- 分别报告 padded issued FLOP 和 useful FLOP；
- 相同 CTA 数 sweep，避免一个版本 grid 更大；
- warmup 后多轮测试，报告 median/P10/P90 或 mean/min/max；
- 锁不住频率时记录 `--clock-control none`；
- 每次 benchmark 前后同步；
- 随机多 seed 对拍；
- 纯指令口径与完整数据路径口径分开。

### 13.5 成功、失败分别怎样写

若自然 `m128n128k16` tcgen 明显更快，但 M16 路径更慢：

> SM100 有局部机会，但 instruction-only 全替换不成立；值得探索只替换 state-delta phase。

若所有 tcgen 端到端版本都更慢：

> 小工作量无法摊薄 TMEM 生命周期；SM80 register path 更适合 C16。官方决策成立。

若纯指令 tcgen 快、完整路径慢：

> 瓶颈不是 Tensor Core 计算，而是 TMEM/SMEM/同步数据路径。后续优化必须重构融合，而不是继续追求指令峰值。

若 H=96 加速、H=12 退化：

> benchmark shape 的局部胜利不能转化为 K3 TP8 部署收益；dispatch 必须包含 H/N 条件。

---

## 14. B300 复现与测量步骤

## 14.1 环境记录

在日志中保存：

```bash
nvidia-smi -L
nvidia-smi
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())'
```

报告同时写：

- FlashKDA commit `1ce47ea`；
- CUTLASS commit `5c149f5`；
- CUDA driver/toolkit；
- PyTorch 与 FLA 版本；
- GPU 型号；
- 是否共享节点；
- clock-control 口径。

### 14.2 准备完整上游仓库

题目快照没有包含 CUTLASS 子模块。建议在 C1 目录内另建完整 clone，避免覆盖快照：

```bash
cd ~/lcpu2026/assignment02/team/c1_flashkda
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA FlashKDA-upstream
cd FlashKDA-upstream
git checkout 1ce47ea
git submodule update --init --recursive
git submodule status
```

确认 CUTLASS 显示预期 pin。不要在报告中只写“最新 main”。

### 14.3 构建

```bash
cd ~/lcpu2026/assignment02/team/c1_flashkda/FlashKDA-upstream
FLASH_KDA_CUDA_ARCHS=100a NVCC_THREADS=8 pip install -v --no-build-isolation -e .
```

若 B300 的 capability/工具链采用不同 arch 标记，以节点实际环境和 setup.py 支持项为准。

保留 ptxas 输出，特别关注：

- registers；
- spill stores/loads；
- shared memory；
- 编译的 `sm_100a` code object。

### 14.4 correctness

```bash
bash tests/test.sh
```

记录：

- exact torch_ref 是否通过；
- FLA comparison 是否通过；
- 哪些 case 只 warning；
- output 与 final state 的误差；
- fixed/varlen 是否都覆盖。

### 14.5 benchmark

```bash
python benchmarks/bench_fwd.py --mode all --H 96 --D 128 --warmup 30 --iters 200 --repeats 5
python benchmarks/bench_fwd.py --mode all --H 64 --D 128 --warmup 30 --iters 200 --repeats 5
python benchmarks/bench_fwd.py --mode all --H 12 --D 128 --warmup 30 --iters 200 --repeats 5
```

建议增加 N sweep，但保持总 token 为 8192：

```text
[8192]
[4096,4096]
[2048]×4
[1024]×8
[512]×16
```

这能把“总工作量”和“状态链并行度”分离。

### 14.6 NCU

仓库已有模板：

```bash
bash benchmarks/ncu.sh
```

它会 profile：

```text
_flash_kda_fwd_prepare
_flash_kda_fwd_recurrence
```

第一次使用 `--set full` 得到完整报告；之后为减少开销，选取第 10 节的关键 metrics。

一定要分别导出：

- fixed H96；
- fixed H12；
- varlen 1024×8 H12/H96；
- BF16 boundary state；
- no state；
- 若时间允许，FP32 boundary state。

### 14.7 SASS

找到 extension：

```bash
python -c 'import flash_kda_C; print(flash_kda_C.__file__)'
```

然后：

```bash
cuobjdump --dump-sass /path/to/flash_kda_C.so > flash_kda_b300.sass
rg -n 'HMMA|MMA|TCGEN|WGMMA' flash_kda_b300.sass
```

报告中保留少量代表性片段即可：

- K1/K2 函数名；
- SM80 BF16/FP16 MMA 指令；
- 没有 tcgen05 主路径；
- code object 的目标架构。

不要贴几千行 SASS。

---

## 15. 实验表格模板

### 15.1 端到端 benchmark

| GPU | T/seq_lens | H | state | FlashKDA ms | FLA ms | speedup | mean/min/max | 备注 |
|---|---|---:|---|---:|---:|---:|---|---|
| B300 | `[8192]` | 96 | BF16 | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | official K3 shape |
| B300 | `1024×8` | 96 | BF16 | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | parallelism |
| B300 | `[8192]` | 12 | BF16 | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | TP8 |
| B300 | `1024×8` | 12 | BF16 | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | TP8 + N8 |

### 15.2 K1/K2 profile

| Shape | Kernel | duration | grid | active SM | Tensor % | DRAM % | L2 % | eligible warp/cycle | top stall | reg | smem |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| H96 fixed | K1 | `[待]` | 49152 | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` |
| H96 fixed | K2 | `[待]` | 96 | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` |
| H12 fixed | K2 | `[待]` | 12 | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` |
| H12 N8 | K2 | `[待]` | 96 | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` |

### 15.3 microbench

| Shape | 实现 | padded shape | useful FLOP | issued FLOP | latency | useful TF/s | issued TF/s | active CTA/SM | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 16×16×16 | SM80 | exact | `[算]` | `[算]` | `[待]` | `[待]` | `[待]` | `[待]` | |
| 16×16×16 | tcgen | M64/M128 | `[算]` | `[算]` | `[待]` | `[待]` | `[待]` | `[待]` | |
| 16×128×128 | SM80 | exact | `[算]` | `[算]` | `[待]` | `[待]` | `[待]` | `[待]` | |
| 16×128×128 | tcgen | M64/M128 | `[算]` | `[算]` | `[待]` | `[待]` | `[待]` | `[待]` | |
| 128×128×16 | SM80 | exact | `[算]` | `[算]` | `[待]` | `[待]` | `[待]` | `[待]` | |
| 128×128×16 | tcgen | exact | `[算]` | `[算]` | `[待]` | `[待]` | `[待]` | `[待]` | |

### 15.4 精度

| T | gate case | beta case | state scale | implementation | output RMSE ratio | state RMSE ratio | max abs | cosine | NaN/Inf |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 8192 | near 0 | random | 1 | FlashKDA BF16 | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` |
| 8192 | near -5 | near 1 | 1 | FlashKDA BF16 | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` |
| 32768 | alternating | near 1 | 100 | FlashKDA BF16 | `[待]` | `[待]` | `[待]` | `[待]` | `[待]` |

---

## 16. 你几小时后开始时，按这个顺序做

### 第 0–30 分钟：只建立全局图

读本文：

- 第 0 节；
- 第 3–5 节；
- 第 7、8 节的结论；
- 第 16 节执行表。

目标不是记代码，而是能说：

```text
KDA 是什么
为什么 chunk
K1/K2 各做什么
为什么 tcgen 不一定快
```

### 第 30–90 分钟：环境与正确性

1. clone 完整 pin 仓库；
2. 确认 CUTLASS submodule；
3. 编译 sm100a；
4. 跑 tests；
5. 保存环境和编译日志。

如果构建失败，优先查：

- 是否在 `FlashKDA` 根目录；
- CUTLASS 是否真的存在；
- CUDA 是否 ≥12.9；
- PyTorch CUDA 是否匹配；
- `FLASH_KDA_CUDA_ARCHS=100a` 是否正确；
- 节点是否真的分配了 B300。

### 第 90–150 分钟：baseline

先跑：

- H96 fixed；
- H96 `1024×8`；
- H12 fixed；
- H12 `1024×8`。

这四个点已经能回答“官方形状”和“TP8 并行度”。

### 第 150–240 分钟：SASS + NCU

1. SASS 证明 SM80 MMA；
2. NCU 分开记录 K1/K2 duration；
3. 优先检查 K2 H12 的 active SM；
4. 再看 Tensor/DRAM/L2 throughput；
5. 最后看 stall，避免被单一 stall 指标带偏。

### 第 4–6 小时：microbench

按难度排序：

1. 复用 M1 的 SM80 m16 atom；
2. 复用 M3.2 的 tcgen05 生命周期；
3. 先做天然匹配 `m128n128k16`；
4. 再做 padded `m16n16k16`；
5. 时间够再做 `m16n128k128`。

先有一个正确、可测的 A/B 对照，再做优化。

### 当天结束前

至少得到：

- 一个 B300 baseline 表；
- 一段 SASS 证据；
- K1/K2 各一组 NCU；
- 一个 microbench 的正确性与延迟；
- 六问每问一句临时结论；
- 所有未验证项明确标 `[待实测]`。

---

## 17. 10 分钟 presentation 结构

### 0:00–1:00：问题

> FlashKDA 在 GB200 已比 FLA 快 1.7–3.3×，却仍用 SM80 MMA。我们要判断 SM100 tcgen05 是否还能带来真实收益。

### 1:00–2:20：算法和两 kernel

- 一张 KDA 状态递推图；
- 一张 K1/K2 数据流图；
- 强调 K1 chunk-parallel、K2 head-parallel serial recurrence。

### 2:20–3:40：CHUNK=16

- 数值表：C16 安全、C32 第 18 个最坏 token 破；
- inverse 表：C32/C64 每 token 5.33×/26.67×；
- SM80 atom 恰好 M16。

### 3:40–5:00：tcgen shape

- 一张五类 GEMM 形状表；
- 72% 左右 FLOP 位于 M16 运算；
- state delta 的 128×128×16 是唯一自然切入口。

### 5:00–6:30：B300 profile

- SASS 证据；
- K1/K2 latency；
- Tensor/DRAM/L2；
- H96 与 TP8 H12 的 active SM；
- fixed vs N8。

### 6:30–8:00：挑战实验

- microbench 设计；
- useful vs issued TFLOPS；
- 自然 tile 与 padded tile 的结果；
- 若失败，明确失败发生在哪个数据路径。

### 8:00–9:20：精度与产品决策

- FP32 API state 实际仍转 BF16；
- 长序列验证矩阵；
- portable 主路径 + conditional SM100 dispatch 的结论。

### 9:20–10:00：一句话收尾

建议使用这种句式：

> 我们发现新架构指令是否值得，取决于算法 tile、数据驻留和并行度，而不是峰值代际。对 FlashKDA，M16 主路径支持保留 SM80；SM100 的合理入口是自然匹配的状态更新或更深的数据流重构，而不是机械替换。

---

## 18. 可能被问的问题与回答

### Q1：B300 为什么还能运行 SM80 MMA？

`SM80_...` 是 CUTLASS atom/指令族名称，不意味着二进制只能运行在 A100。构建目标仍是 `sm_100a`，而该架构保留相应 warp-level MMA 能力。最终以 sm100a code object 的 SASS 为证。

### Q2：tcgen05 峰值更高，为什么不一定快？

峰值假设 tile 被填满且数据持续供给。FlashKDA 主要 M=16，tcgen 需要 padding；还要付 TMEM alloc、commit/wait、load/store，并且 K2 可能只有 12 个 CTA。有效工作率可能下降。

### Q3：为什么不用 wgmma？

wgmma 是 SM90a 专属路线；课程材料明确指出 B300 的 SM100 路径使用 tcgen05，不能把 Hopper wgmma 当作 B300 的替代方案。

### Q4：把 4 个 head 拼成 M64 不就行了吗？

不同 head 的状态矩阵 B 不同。普通 dense GEMM 的所有 M 行共享同一个 B，直接拼 M 在代数上不成立；block diagonal 会引入更大浪费。

### Q5：为什么官方 `[1024]×8` 比 `[8192]` 快？

总 token 相同，但 K2 从 H 个 CTA 变成 8H 个 CTA，每条状态依赖链从 512 chunks 缩短为 64 chunks，增加并行度并改善延迟隐藏。

### Q6：你们说 memory-bound，为什么还做 MMA microbench？

先确认局部计算是否存在可利用的 SM100收益。如果纯指令有收益而端到端没有，恰好能证明瓶颈在数据路径；若纯指令也没有收益，则可更早停止高风险集成。

### Q7：FP32 state benchmark 不能证明 FP32 recurrence 吗？

不能。源码在 K2 入口把 FP32 state 转 BF16，chunk 循环中仍用 BF16 保存；出口才转回 FP32。它只改变边界格式。

### Q8：C32 用 rescale 后是否一定更快？

不一定。它减少 chunk 循环数，但增加 subchunk 组合、inverse、C² workspace、缩放和同步。必须比较端到端，而不能只看 chunk 数减半。

### Q9：为什么负结果也算贡献？

题目明确允许负结论。只要实验公平，并定量说明 shape padding、TMEM 往返、workspace 或并行度抵消了收益，就完成了官方设计决策中缺失的论证。

### Q10：最大的局限是什么？

如果没有完整 K3 权重，只能给出算子级精度结论；如果只测 H96 而不测 TP8 H12，部署结论不完整；如果 microbench 不包含完整 TMEM 数据路径，只能说明指令潜力，不能说明端到端收益。

---

## 19. 最终报告的推荐目录

```text
1. 问题与结论摘要
2. KDA 数学与 FlashKDA 架构
3. B300 复现环境与官方 baseline
4. SASS：SM80 MMA 证据
5. CHUNK=16 的数值、inverse、shape 定量分析
6. K1/K2 Roofline 与 NCU 性能归因
7. 状态依赖与并行度候选/反例
8. BF16 state 精度实验
9. SM100 挑战实现与 microbench
10. 是否发布 SM100a v2
11. 局限与后续工作
附录：命令、完整表格、额外 SASS/NCU
```

每节开头先写一句结论，再放图表和证据。不要让老师从十页数据里猜你的观点。

---

## 20. 最终检查清单

### 源码理解

- [ ] 能写出逐 token KDA 四个公式；
- [ ] 能解释 g、beta、q/k normalization；
- [ ] 能解释 cumulative gate 和 base-2 换底；
- [ ] 能解释 L 严格下三角，因此 Neumann 有限终止；
- [ ] 能说清 K1 六个 workspace 输出；
- [ ] 能说清 K2 的五个 phase；
- [ ] 能解释 3-stage input / 2-stage output pipeline；
- [ ] 知道 FP32 API state 内部仍转 BF16。

### 复现

- [ ] B300 环境与版本记录完整；
- [ ] commit 和 CUTLASS pin 正确；
- [ ] correctness 通过；
- [ ] H96/H64/H12 baseline；
- [ ] fixed/varlen；
- [ ] SASS 确认 SM80 MMA；
- [ ] K1/K2 分别 profile。

### 六问

- [ ] C16/C32/C64 指数表；
- [ ] inverse FLOP 表；
- [ ] workspace/FLOP 表；
- [ ] tcgen padding useful/issued 区分；
- [ ] 多 head/persistent/2-CTA 各有反例；
- [ ] compute/memory/parallelism 用多指标交叉验证；
- [ ] BF16 精度含长序列和 final state；
- [ ] v2 决策有预先定义的 ship gate。

### 挑战

- [ ] SM80 baseline 与 tcgen 使用相同数学 shape；
- [ ] 纯指令与完整数据路径分开；
- [ ] 至少一个天然匹配 tile；
- [ ] 至少一个 M16 padded tile；
- [ ] 正确性多 seed；
- [ ] 结果注明 GPU；
- [ ] 无收益时也有归因。

---

## 21. 一页速记

```text
KDA：
  旧状态先衰减
  residual = value - key 对旧状态的预测
  state += beta * key outer residual
  output = query 读状态

FlashKDA：
  C=16, D=128
  K1 grid = chunks × heads：高并行，准备 INV/Mqk/decayed qk
  K2 grid = sequences × heads：低并行，按 chunk 递推 state

C16：
  worst cumsum = -80
  exp(-80) 与 exp(80) 仍安全
  C32 在极端第18 token 左右下溢/上溢
  inverse 只需 6 个 16³ GEMM

SM100：
  大多数热点 M=16，不匹配标准 dense tcgen tile
  M16→M64：4× issued/useful
  M16→M128：8× issued/useful
  state delta = m128n128k16，天然匹配，值得 microbench

纸面官方形状：
  92.61 GFLOP
  2.397 GB 最低逻辑流量
  AI ≈ 38.6 FLOP/B
  workspace 写+读 ≈ 1.359 GB

并行度：
  H96 fixed：K2 96 CTA
  TP8 H12 fixed：K2 12 CTA
  N8 可把 CTA 数乘 8、依赖链除 8

产品结论：
  保留 SM80 portable 主路径
  SM100 先做真实子形状 microbench
  只有真实 K3/TP shape 端到端稳定提升且精度通过才 dispatch
```

---

## 22. 材料索引

- 题目：`assignment02/team/c1_flashkda/TASK.md`
- 官方设计说明：`assignment02/team/c1_flashkda/FlashKDA/docs/20260420-flashkda-v1-deep-dive.md`
- API/构建：`assignment02/team/c1_flashkda/FlashKDA/README.md`
- K1：`assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_kernel1.cuh`
- K2：`assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_kernel2.cuh`
- MMA/Neumann helper：`assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/utils.cuh`
- launch/grid/workspace：`assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_launch.cu`
- API state 转换：`assignment02/team/c1_flashkda/FlashKDA/csrc/flash_kda.cpp`
- 朴素数学参考：`assignment02/team/c1_flashkda/fla_kda_ref/naive.py`
- FLA chunk 路径：`assignment02/team/c1_flashkda/fla_kda_ref/chunk_fwd.py`
- 精确复刻参考：`assignment02/team/c1_flashkda/FlashKDA/tests/torch_ref.py`
- 精度测试：`assignment02/team/c1_flashkda/FlashKDA/tests/test_fwd.py`
- benchmark：`assignment02/team/c1_flashkda/FlashKDA/benchmarks/bench_fwd.py`
- NCU 模板：`assignment02/team/c1_flashkda/FlashKDA/benchmarks/ncu.sh`
- 官方数据：`BENCHMARK_GB200.md`、`BENCHMARK_H20.md`
- M3 衔接文档：`assignment02/M3_tcgen05_知识点详解.md`
- M4 衔接文档：`assignment02/M4_GEMM_TMA_Pipeline_知识点与题解.md`

你开始时先完成“正确性 + baseline + SASS + 一组 K2 NCU”。这四项一旦拿到，后面的分析就不再是空谈；microbench 即使得出负结果，也能形成一条完整、可信的 C1 故事。
