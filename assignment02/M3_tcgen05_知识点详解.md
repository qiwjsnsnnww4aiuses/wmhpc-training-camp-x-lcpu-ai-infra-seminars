# M3：SM100 `tcgen05` 知识点详解

> 适用范围：Assignment 02 Module 3，配合 Session 03 讲义 S071–S093 阅读。  
> 本文讲清概念、数据流、同步关系、资源计算和调试方法，但不直接填写 3.1 的判断题答案、不指出 3.3 的具体修改行，也不提供 3.2 的可提交完整实现。

## 0. 先建立一张总地图

M3 表面上出现了很多新名词，实际上只是在回答四个问题：

1. **输入在哪里？** A、B 先放在 shared memory，用 matrix descriptor 描述。
2. **谁发起计算？** 一个被选中的线程发射 `tcgen05.mma`。
3. **累加器在哪里？** 不再长期占用普通寄存器，而是放在 TMEM。
4. **其他线程怎么知道计算完成？** 发射线程用 `tcgen05.commit` 把完成事件报告给 mbarrier；消费者等待后，再用 `tcgen05.ld` 把结果读进寄存器。

把完整数据通路先背下来：

```text
GMEM
  │  普通 load/store（M3 手搓版本）或 TMA（生产/M4）
  ▼
SMEM：A、B tile，K-major + swizzle
  │  64-bit matrix descriptor
  ▼
Tensor Core：tcgen05.mma
  │
  ▼
TMEM：C/D accumulator
  │  tcgen05.ld + tcgen05.wait::ld
  ▼
RMEM：每个线程拿到自己负责的结果
  │  普通 global store / epilogue
  ▼
GMEM：输出 D
```

M3 的难点不是矩阵乘法，而是这条通路中存在三套不同的“完成”概念：

- 线程是否都到达了某处；
- shared memory 的写是否对 async proxy 可见；
- Tensor Core 的异步工作是否真正结束。

`__syncthreads()`、`fence.proxy.async`、mbarrier 分别解决不同问题，不能相互替代。

---

## 1. 为什么 Blackwell 要从 `wgmma` 继续演化到 `tcgen05`

### 1.1 三代 Tensor Core 编程模型

| 世代 | 主要指令 | 谁共同发射 | A/B 来源 | 累加器 C/D | 完成同步 |
|---|---|---:|---|---|---|
| SM80 / Ampere | `mma.sync` | 1 warp，32 线程 | register fragment | RMEM | 隐式 warp 同步 |
| SM90 / Hopper | `wgmma` | 1 warpgroup，128 线程 | SMEM descriptor | RMEM | `commit_group` / `wait_group` |
| SM100 / Blackwell | `tcgen05.mma` | 1 个线程 | A：SMEM/TMEM；B：SMEM | **TMEM** | **mbarrier** |

这里有两条连续的演化路线。

第一条是“**操作数离开寄存器**”：

```text
SM80：A/B/C/D 大量依赖普通寄存器
       ↓
SM90：A/B 留在 SMEM，Tensor Core 通过 descriptor 自己取
       ↓
SM100：A/B 在 SMEM，累加器也从 RMEM 搬到 TMEM
```

第二条是“**发射者越来越少**”：

```text
SM80：32 个线程共同执行
SM90：128 个线程共同执行
SM100：1 个线程发射，Tensor Core 异步工作
```

### 1.2 `wgmma` 留下的两个问题

Hopper 的 `wgmma` 已经能让 Tensor Core 从 SMEM 直接读取 A/B，也允许 CUDA Core 与 Tensor Core 重叠工作，但还有两个压力：

1. 累加器仍放在 RMEM，会吃掉大量寄存器；对这些寄存器的普通操作还可能迫使 `wgmma` 串行化。
2. 发射一条 `wgmma` 仍需要整个 warpgroup 的 128 个线程共同参与，但真正做矩阵乘的是 Tensor Core，并不需要 128 个 CUDA 线程“亲手计算”。

Blackwell 的回答是：

- 用 **TMEM** 专门保存 Tensor Core 数据；
- 让 **单线程发射** `tcgen05.mma`；
- 用 **mbarrier** 把“谁发射”和“谁消费”解耦。

形象地说：

```text
mma.sync：32 个人一起抬货
wgmma：  128 个人一起下单，货放在他们的仓位里
tcgen05：1 个人下单，机器自动加工，成品先放进 TMEM 仓库
```

注意：只有 `mma` 的发射缩成了一个线程。TMEM 分配和 TMEM→寄存器读取仍有 warp 级协作要求。

---

## 2. TMEM：给 Tensor Core 单独准备的“累加器仓库”

### 2.1 TMEM 是什么，不是什么

TMEM（Tensor Memory）是 Blackwell 每个 SM 上的专用存储空间：

- 容量：**256 KiB / SM**；
- 逻辑组织：**128 lane × 512 column**；
- 每个格子：**32 bit = 4 byte**；
- 总量：`128 × 512 × 4 B = 262144 B = 256 KiB`。

它不是：

- 普通 CUDA 寄存器；
- shared memory 的一个别名；
- 可以由普通 `ld`/`st` 任意访问的地址空间。

CUDA Core 不能直接读写 TMEM，只能通过 `tcgen05.ld`、`tcgen05.st`、`tcgen05.cp` 等专门指令访问。

### 2.2 为什么是 128 lane × 512 column

可以把 TMEM 想成一个巨大的表格：

```text
                    column 0                     column 511
lane   0     [ 32 bit ][ 32 bit ] ... [ 32 bit ]
lane   1     [ 32 bit ][ 32 bit ] ... [ 32 bit ]
 ...
lane  31     ← warp 0 可访问的 lane 范围 →
lane  32     ← warp 1 可访问的 lane 范围 →
 ...
lane  63
lane  64     ← warp 2 →
 ...
lane  95
lane  96     ← warp 3 →
 ...
lane 127
```

一个 128 行输出 tile 正好覆盖 128 条 TMEM lane。四个 warp 各负责 32 条 lane，因此读取完整 128 行累加器时，四个 warp 都要参与。

### 2.3 TMEM 地址怎么理解

课程采用的 32 位 TMEM 地址可以按两部分理解：

```text
31                         16 15                           0
+----------------------------+-----------------------------+
|        lane offset         |        column offset        |
+----------------------------+-----------------------------+
```

- 高 16 位选择 lane 偏移；
- 低 16 位选择 column 偏移。

因此，把某个 warp 的起始 lane 加入地址，本质上是在高 16 位加 `warp_id × 32`；在某个输出列块间移动，则改变低 16 位的 column。

不要把 TMEM address 当成字节指针。它描述的是 TMEM 的二维逻辑坐标，不是普通线性内存地址。

### 2.4 每个 warp 能看到哪一部分

`warp i` 只能访问 lane `32i` 到 `32i+31`。于是：

- warp 0：输出第 0–31 行；
- warp 1：输出第 32–63 行；
- warp 2：输出第 64–95 行；
- warp 3：输出第 96–127 行。

一个 warp 不能单独读完整个 m128 累加器。这条访问规则解释了为什么：

- `tcgen05.mma` 可以由一个线程发射；
- epilogue 却仍需要四个 warp 分工读出结果。

### 2.5 容量计算

对于 f32 累加器，粗略容量可以用：

```text
TMEM bytes = M × N × 4 B
TMEM columns = N（当逻辑映射是一列对应一个 32-bit 累加槽时）
```

例如 M3 的 `m128n64` 累加器：

```text
128 × 64 × 4 B = 32768 B = 32 KiB
```

它占 64 个 column、全部 128 个 lane。

讲义中的 `m128n256` 则覆盖 256 个 column，即 TMEM 的一半。剩下的另一半可以用来做 double buffer：Tensor Core 往一半写下一块时，CUDA 线程从另一半做 epilogue。

### 2.6 TMEM 的分配与释放

TMEM 不是进入 kernel 就自动属于 CTA，必须显式经历生命周期：

```text
申请 alloc → 得到 taddr → 使用 → 所有消费者完成 → dealloc
```

关键约束：

- 分配粒度按 **column** 计，课程路径中列数要求为不小于 32 的 2 的幂；
- `tcgen05.alloc...sync.aligned` 是 **warp 级协作指令**，需要一个完整 warp 执行；
- 分配结果不是直接返回到某个普通寄存器，而是写到一个 shared-memory 地址；
- 其他线程先同步，再从 shared memory 读取这个 TMEM base address；
- 分配后要 relinquish allocation permit，让分配机制继续服务其他请求；
- 用完必须显式 `dealloc`；释放前必须保证所有 warp 已经完成读取。

这里很容易产生误解：

> “单线程发射 MMA”不等于“所有 tcgen05 指令都是单线程指令”。

MMA 发射、TMEM 分配、TMEM 读取是三种不同的协作粒度。

---

## 3. `tcgen05.mma`：一个线程给 Tensor Core 下订单

### 3.1 指令的角色分解

把 `tcgen05.mma` 的主要操作数按角色记忆：

| 信息 | 意义 |
|---|---|
| `taddr_d` | 输出/累加器 D 在 TMEM 中的地址 |
| `a_desc` | A 的来源；可以是 SMEM descriptor，也可以来自 TMEM |
| `b_desc` | B 的 SMEM descriptor；B 必须从 descriptor 路径提供 |
| `idesc` | 32 位 instruction descriptor，描述精确 shape、dtype、转置等 |
| `p` | `enable_input_d`，决定本次是否读取旧 D 并累加 |

`kind` 和 `idesc` 分工如下：

- `kind`：粗粒度类别，例如 f16、tf32、整数、低精度类别；
- `idesc`：运行时精确参数，例如 f16 类别中究竟是 fp16 还是 bf16、M/N 尺寸、是否转置等。

### 3.2 为什么 `mma` 不再需要 `.sync.aligned`

`mma.sync` 需要一个 warp 一致执行；`wgmma` 需要一个 warpgroup 一致执行。`tcgen05.mma` 改为一个线程发射，所以 MMA 指令本身不再要求整组线程共同执行。

典型组织方式是：

```text
128-thread CTA
├─ warp 0：其中 elect.sync 选出一个线程作为“发射员”
├─ warp 1：之后参与结果读取
├─ warp 2：之后参与结果读取
└─ warp 3：之后参与结果读取
```

使用 `elect.sync` 的意义是，在一个 warp 的活动线程中确定唯一发射者，又保持 warp 内控制流语义明确。

### 3.3 `kind` 与 K 方向步长

Session 03 给出的常见类别如下：

| kind | 操作数类别 | dense MMA 的 K 步长 |
|---|---|---:|
| `f16` | fp16 / bf16 | 16 |
| `tf32` | tf32 | 8 |
| `i8` | s8 / u8 | 32 |
| `f8f6f4` | e4m3 / e5m2 / e3m2 / e2m3 / e2m1 | 32 |
| `mxf8f6f4` | 上述低精度 + 硬件 block scaling | 32 |
| `mxf4` / `mxf4nvf4` | fp4 + 硬件 block scaling | 64 |

M3 的核心路径是 bf16 输入、f32 累加，属于 `kind::f16`，所以一次 MMA 消耗 K 方向的 16 个元素。对于 K=64 的 tile，逻辑上会切成四个 k16 片段。

### 3.4 `enable_input_d`：首片覆盖，后片累加

矩阵乘沿 K 分块时：

```text
D = A[:, 0:16] × B[:, 0:16]^T
  + A[:,16:32] × B[:,16:32]^T
  + A[:,32:48] × B[:,32:48]^T
  + A[:,48:64] × B[:,48:64]^T
```

第一次计算时，TMEM 中旧 D 没有有效意义；后续片段才应读取旧 D 并累加。因此谓词的抽象行为是：

```text
第一片：D ← partial_product
后续片：D ← D + partial_product
```

这比提前清零整块 TMEM 更自然，也避免额外的初始化工作。

### 3.5 idesc：运行时的“订单详情”

课程里的 32 位 idesc 包括这些信息：

- D 的数据类型；
- A、B 的精确数据类型；
- A/B 的符号或取负控制；
- A/B 是否转置；
- N，以 8 为单位编码；
- M，以 16 为单位编码。

记忆方式：

```text
kind 说“我要一类什么计算”
idesc 说“这次具体多大、什么类型、怎么解释输入”
descriptor 说“A/B 数据在 shared memory 里怎么摆”
```

这三个 descriptor/类别不要混为一谈。

### 3.6 `.ws` 变体

`tcgen05.mma` 还有 weight-stationary（`.ws`）变体，可让 B 驻留复用，适合某些 M 较小且 B 被反复使用的形状。它不是 M3 单 tile 主线要求，但理解它有助于后续分析小 M、skinny GEMM 或 FlashKDA 的特殊形状。

是否适合 `.ws` 不能只看“能否发出指令”，还要看：

- B 的复用次数是否足够；
- shape 是否落在支持范围；
- 数据重排与启动开销是否抵消收益；
- CTA 数量是否足以填满 GPU。

---

## 4. SMEM descriptor 与 swizzle：M2 如何接到 M3

### 4.1 descriptor 的本质

从 SM90 起，Tensor Core 不需要先把 A/B 搬成每个线程的 register fragment，而是读取一个 64 位 matrix descriptor，自行按照 canonical layout 从 SMEM 取数。

descriptor 回答的是：

- tile 从哪个 shared-memory 地址开始；
- core matrix 沿两个方向的步距；
- 使用哪种 swizzle 布局；
- 当前架构要求的版本字段。

它不是数据本身，而是一张“仓库货架地图”。

### 4.2 为什么 M3 仍需要 M2 的 K-major 与 swizzle

M3 的输入数据流是：

```text
逻辑矩阵坐标
   │
   ├─ 软件 staging：按 swizzle 公式写进物理 SMEM 地址
   │
   └─ descriptor：告诉 Tensor Core 用同一布局解释这些地址
```

二者必须完全一致。如果软件按普通行主序写，descriptor 却声明 128B swizzle，Tensor Core 会稳定地读错数据。

### 4.3 swizzle 解决什么

shared memory 有 32 个 bank。Tensor Core 按列或特定块状模式高并发取数时，如果同一批请求落在相同 bank，就会形成 bank conflict。

128B swizzle 的直观效果是：

```text
逻辑上相同的 16B chunk 列
        ↓ 根据行号做地址位 XOR
物理上分散到不同 bank group
```

它保持一一映射，不丢数据，只改变物理位置。M2 的 host 判测验证“数学上是一一映射且无冲突”，M3 的真实 Tensor Core GEMM 则验证“软件写法和硬件 descriptor 对布局的理解确实一致”。

---

## 5. 异步执行：发射完成不等于计算完成

### 5.1 `tcgen05.mma` 的时间线

一个线程执行 MMA 指令后，只是把任务交给 Tensor Core：

```text
时间 ──────────────────────────────────────────────▶

发射线程： issue mma ─ issue mma ─ commit ─ 做别的工作
Tensor Core：   [------ 异步计算 ------][-- 完成 --]
mbarrier：      未满足                     arrive，phase 完成
消费者：        try_wait ... try_wait ... 通过后读取
```

因此，“程序计数器已经越过 `tcgen05.mma`”不能证明结果可读。

### 5.2 `tcgen05.commit` 的语义

`commit` 不是阻塞等待，它更像给硬件留下一张回执单：

> 在该线程此前发射的相关 tcgen05 工作全部完成时，对指定 mbarrier 执行一次 arrive。

这使发射者可以继续执行，消费者则通过 mbarrier 独立等待。与 `wgmma.wait_group` 相比，生产者与消费者不必是同一批线程。

---

## 6. mbarrier：M3 最关键的状态机

### 6.1 mbarrier 维护什么状态

mbarrier 是 shared memory 中的一个 64 位对象，概念上维护：

| 状态 | 含义 |
|---|---|
| phase | 第几轮 barrier；硬件等待通常只比较奇偶 parity |
| arrival count | 当前 phase 还差多少次 arrive |
| tx-count | 当前 phase 还差多少字节的异步传输完成 |

一个 phase 只有在下面两项都归零时才完成：

```text
arrival count == 0  且  tx-count == 0
```

M3 的 MMA 完成通知主要使用 arrival count；M4 的 TMA pipeline 还会用 tx-count 表示尚未完成的异步搬运字节数。

### 6.2 phase 与 parity

可以把复用 barrier 想成红绿灯：

```text
phase 0（parity 0）完成
        ↓ 翻转
phase 1（parity 1）完成
        ↓ 翻转
phase 2（parity 0）完成
        ↓ 翻转
phase 3（parity 1）完成
```

硬件等待通常只关心 parity，因此软件必须明确自己等的是哪一轮。如果多轮异步工作重复使用同一 barrier，却一直等待同一个 parity，旧一轮的完成状态可能被误认为新一轮完成；反过来，也可能等待一个不会以预期方式到来的状态。

调试时不要只盯着“有没有 arrive”，而要画出：

```text
轮次 | 进入时 parity | expected arrivals | 谁触发 arrive | 何时归零 | 下一轮 parity
```

### 6.3 初始化

初始化要明确 expected arrival count。若当前 phase 预期一个完成事件，初始 arrival count 就对应一个事件。

初始化后还需要考虑两件事：

- mbarrier 对其他线程何时可见；
- 其他线程是否已经等到初始化完成再使用它。

课程代码中出现的 init fence 与 CTA 同步，就是为了解决“对象已经写好且所有参与者都能安全使用”的问题。

### 6.4 `try_wait.parity`

`try_wait` 是尝试等待：

- 若目标 parity 已完成，返回成功；
- 若未完成，硬件可以挂起该线程，让执行资源去做其他工作；
- 常见写法是在循环中重试，而不是用 CUDA Core 做有意义的繁忙计算。

带 acquire 语义的等待不仅说明“事件发生了”，还保证事件之前建立的内存效果在等待返回后对消费者可见。

### 6.5 mbarrier 与 `wgmma.wait_group` 的区别

```text
wgmma：发射者和等待者通常是同一个 warpgroup
tcgen05：一个线程发射，任意需要结果的线程都可等待同一个 mbarrier
```

这正是 mbarrier 的价值：它把执行关系从“同一批线程的内部队列”提升为“一个可共享的完成事件”。

### 6.6 多轮 MMA 最稳妥的纸面调试法

在碰代码前，先画表：

| round | 本轮发射的 MMA | commit 关联的工作 | 等待的 parity | wait 返回时允许做什么 |
|---:|---|---|---:|---|
| 0 | 第 0 个 K 片段 | round 0 前已发射工作 | 自己推导 | 读本轮 TMEM |
| 1 | 第 1 个 K 片段 | round 1 前已发射工作 | 自己推导 | 读本轮 TMEM |
| 2 | 第 2 个 K 片段 | round 2 前已发射工作 | 自己推导 | 读本轮 TMEM |

然后逐项检查：

1. 本轮 commit 最终是否恰好 arrive 一次；
2. wait 是否等待本轮，而不是上一轮；
3. 所有 warp 是否读完 TMEM 后才允许下一轮覆盖；
4. 释放 TMEM 前是否所有读取都结束。

这套方法正是 3.3 想训练的能力，比“反复加同步直到不挂”可靠得多。

---

## 7. `tcgen05.ld`：把 TMEM 结果交还给 CUDA 线程

### 7.1 为什么 MMA 是单线程发射，读取却是 warp 协作

MMA 发射只需要描述整个矩阵运算；但结果最终要进入每个线程的普通寄存器，供 epilogue 或 global store 使用。寄存器属于线程，因此消费者必须分工搬运。

典型分工：

```text
warp 0：TMEM lane  0–31 → 各线程寄存器
warp 1：TMEM lane 32–63 → 各线程寄存器
warp 2：TMEM lane 64–95 → 各线程寄存器
warp 3：TMEM lane 96–127 → 各线程寄存器
```

### 7.2 `.sync.aligned` 为什么又出现了

`tcgen05.mma` 是单线程发射，所以不需要 warp 协作标记；`tcgen05.ld` 要把数据分发到整个 warp 的寄存器中，因此重新成为 warp 级协作指令，需要参与线程一致执行。

### 7.3 load shape 和数量

讲义列出的 load shape 包括：

- `.32x32b`
- `.16x64b`
- `.16x128b`
- `.16x256b`
- `.16x32bx2`

数量修饰从 `.x1` 到更大的 2 的幂。它们决定一次从 TMEM 的多少 lane/column 取多少份数据，并映射到每个线程的目标寄存器集合。

阅读具体指令时，先把名字拆成三层：

```text
每次覆盖多少 lane × 每个 lane 取多少 bit × 重复多少组
```

### 7.4 `tcgen05.ld` 本身也是异步的

`tcgen05.ld` 出现在程序里，不代表目标寄存器立刻可用。使用这些寄存器做 global store 或其他运算前，必须完成对应的 `tcgen05.wait::ld`。

```text
tcgen05.ld 发起 TMEM→RMEM 搬运
        ↓
tcgen05.wait::ld 确认寄存器数据就绪
        ↓
epilogue / global store
```

这和 MMA 的 mbarrier 等待是两层不同等待：

1. mbarrier：等 Tensor Core 把结果写完 TMEM；
2. `wait::ld`：等 TMEM 内容搬进普通寄存器。

漏掉任何一层，都可能读到未完成的数据。

---

## 8. 三种同步不要混：线程、proxy、异步完成

这是 M3 最值得反复看的部分。

### 8.1 `__syncthreads()`：线程集合到齐

它解决：

- CTA 内所有线程是否都到达；
- 普通线程内存操作之间的 CTA 级同步。

它不能单独保证：

- generic proxy 写入一定已对 async proxy 可见；
- Tensor Core 异步 MMA 已经完成；
- `tcgen05.ld` 目标寄存器已经就绪。

### 8.2 `fence.proxy.async.shared::cta`：跨 proxy 可见

普通 `st.shared` 走 generic proxy，而 Tensor Core 通过 descriptor 读 SMEM 走 async proxy。两个代理访问同一块物理 shared memory，但没有显式 proxy fence 时，不能仅凭地址相同就假定异步消费者看到新值。

```text
CUDA threads --st.shared--> generic proxy
                                  │
                     fence.proxy.async
                                  │
Tensor Core <---- descriptor ---- async proxy
```

因此 M3 用普通线程 staging A/B 时，需要让每个生产线程的 generic 写对 async proxy 可见，再建立线程间同步，之后才能安全发射 MMA。

### 8.3 mbarrier：异步引擎真的完成

mbarrier 解决的是：

- Tensor Core 的异步工作何时完成；
- TMA 的异步传输何时完成；
- 生产者和消费者可能不是同一批线程时，如何传递完成事件。

它不是普通 CTA rendezvous 的同义词。

### 8.4 `tcgen05.fence::after_thread_sync`：跨线程移交 tcgen05 操作

典型场景是：

- 线程 A 发射 MMA；
- 其他线程等待 mbarrier；
- 线程 B/C/D 随后执行自己的 `tcgen05.ld`。

线程同步或 mbarrier 建立了“先后关系”后，每个将执行后续 tcgen05 操作的线程还要遵守 tcgen05 自身的跨线程可见/排序要求。`tcgen05.fence::after_thread_sync` 就用于这类移交。

可以把几种机制记成四句：

```text
__syncthreads：人到齐了吗？
proxy fence：另一个窗口看得到货了吗？
mbarrier：机器真的加工完了吗？
tcgen05 fence：换了操作线程后，tcgen05 的先后关系接上了吗？
```

---

## 9. 单 tile GEMM 的七步生命周期

M3.2 的核心不是背 PTX，而是把资源生命周期和数据依赖放对位置。

```text
┌──────────────────────────────────────────────────────────────┐
│ 1. 初始化完成事件 + 分配 TMEM                               │
│    一个线程初始化 mbarrier；一个完整 warp 执行 TMEM alloc   │
├──────────────────────────────────────────────────────────────┤
│ 2. A/B staging 到 SMEM                                      │
│    全体线程并行写入；布局必须与 descriptor/swizzle 一致      │
├──────────────────────────────────────────────────────────────┤
│ 3. 建立 SMEM 可见性与线程同步                               │
│    generic→async proxy fence；所有线程确认 staging 完成       │
├──────────────────────────────────────────────────────────────┤
│ 4. 发射 MMA + commit                                        │
│    elect 一个线程；沿 K 发射若干 MMA；最后登记完成回执        │
├──────────────────────────────────────────────────────────────┤
│ 5. 等待 Tensor Core 完成                                    │
│    消费者等待正确 mbarrier parity；返回后结果才在 TMEM 可用   │
├──────────────────────────────────────────────────────────────┤
│ 6. epilogue：TMEM→RMEM→GMEM                                 │
│    四个 warp 各读自己的 32 条 lane；等待 ld；再写 global      │
├──────────────────────────────────────────────────────────────┤
│ 7. 释放 TMEM                                                │
│    所有读者结束后同步；完整 warp 执行 dealloc                 │
└──────────────────────────────────────────────────────────────┘
```

### 9.1 对 M3.2 形状做资源体检

题目形状是 `m128n64k64`、bf16 输入、f32 累加：

| 项目 | 计算 | 规模 |
|---|---:|---:|
| A tile | `128 × 64 × 2 B` | 16 KiB |
| B tile | `64 × 64 × 2 B` | 8 KiB |
| D accumulator | `128 × 64 × 4 B` | 32 KiB TMEM |
| K 片段数 | `64 / 16` | 4 个 MMA 片段 |
| 输出行分工 | `128 / 4 warps` | 每 warp 32 行 |

做题前先算这些数，能提前发现很多错误：

- shared-memory 数组容量是否足够；
- TMEM column 数是否匹配 N；
- K 循环次数是否正确；
- 四个 warp 是否覆盖且只覆盖全部 128 行；
- descriptor 的 K 起点是否随片段推进。

### 9.2 为什么判测用小整数并严格相等

bf16 只能精确表示一部分实数，但小整数在一定范围内可以精确表示；小整数乘积与不太大的累加结果也可以在 f32 中精确表达。这样，若 GPU 与 CPU 结果不严格相等，通常更可能是：

- fragment/tile 坐标错；
- descriptor 起点或布局错；
- K 片段遗漏/重复；
- 第一次覆盖与后续累加控制错；
- 同步不完整导致读取在飞结果。

它把“布局 bug”和“正常浮点舍入误差”尽量分开。

---

## 10. Pipeline：mbarrier 如何把 TMA、MMA、epilogue 串起来

M3 先做单 tile，M4 才会真正多级流水，但 M3 已经给出完整骨架：

```text
阶段 A：TMA/线程把 A、B 搬进 SMEM
           │ 完成后报告 tma_mbar
           ▼
阶段 B：MMA 消费 SMEM，写 TMEM
           │ commit 后报告 mma_mbar
           ▼
阶段 C：消费者从 TMEM 读到寄存器并做 epilogue
```

稳态时希望出现重叠：

```text
时间 →       t0        t1        t2        t3

SMEM stage0  load A0/B0 ──────── reuse for A2/B2
SMEM stage1            load A1/B1 ──────── reuse for A3/B3
Tensor Core            MMA tile0  MMA tile1  MMA tile2
CUDA Core                         epi tile0  epi tile1
```

每次复用缓冲区前，都必须证明上一个使用者已结束：

- 重写 SMEM stage 前，确认上一轮 MMA 已不再读它；
- 覆盖 TMEM buffer 前，确认上一轮 epilogue 已读完；
- 等待 barrier 时，使用与当前逻辑轮次匹配的 parity。

pipeline bug 往往不是“少一个同步”这么简单，而是生产者/消费者对“这一格当前属于第几轮”理解不一致。

---

## 11. 2-CTA MMA：两个 SM 合作完成一个更大的 M tile

### 11.1 基本组织

`cta_group::2` 使用一个包含两个 CTA 的 cluster：

```text
cluster_dims(2, 1, 1)

CTA 0（leader，cluster rank 0）       CTA 1（peer，rank 1）
┌───────────────────────────┐       ┌───────────────────────────┐
│ A 的上 128 行             │       │ A 的下 128 行             │
│ B 的一部分                │       │ B 的另一部分              │
│ SM 0 Tensor Core          │       │ SM 1 Tensor Core          │
│ 自己的 128 行 TMEM 结果   │       │ 自己的 128 行 TMEM 结果   │
└───────────────────────────┘       └───────────────────────────┘
              ▲                                  ▲
              └──── leader 发一条 cta_group::2 MMA ────┘
```

只有 leader 发射 MMA，但一条指令驱动两个 SM 上的 Tensor Core，使 M 可扩展到 256。

### 11.2 为什么 B 可以减少重复 staging

若两个独立 CTA 分别计算输出 C 的上下两块，它们的 A 不同，但 B 完全相同：

```text
C_top    = A_top    × Bᵀ
C_bottom = A_bottom × Bᵀ
```

两个独立 CTA 会各自完整搬一份 B。CTA pair 可以让两个 CTA 分别保存 B 的一部分，硬件通过 cluster/peer 访问把两份拼成逻辑完整的 B。

资源比较要分清“每 CTA”和“两个 CTA 合计”：

```text
cta_group::1：
  每 CTA 的输入 = 128×K 的 A + N×K 的 B

cta_group::2：
  每 CTA 的输入 = 128×K 的 A + (N/2)×K 的 B
```

请自己把 M3.4 的 N、K 和 bf16 的 2 byte 代入，分别算：

- 每 CTA 的 B shared-memory 容量；
- 两个 CTA 合计的 B 容量；
- A+B 总 staging 流量，而不是只看 B；
- 每个 CTA 自己要保存的 TMEM 输出范围。

### 11.3 descriptor 为什么仍可写本地 SMEM 地址

CTA pair 两侧把各自 B 片段放在相同的相对 shared-memory offset，并保持相同布局。descriptor 仍描述本地 shared-memory 地址；硬件根据 CTA pair 语义访问 peer CTA 对应 offset。

所以两侧必须满足：

- shared-memory 布局一致；
- descriptor 相对地址约定一致；
- cluster 同步后再允许 leader 发射。

### 11.4 完成通知为什么要 multicast

计算完成后，两个 CTA 都要读取自己所在 SM 的 TMEM。若只有 leader 的 barrier 收到完成事件，peer CTA 可能永远等不到。因此 CTA pair 的 commit 需要把完成到达事件 multicast 到 pair 两侧对应的 mbarrier。

### 11.5 alloc/dealloc 也必须成对协作

`cta_group::2` 的 TMEM allocation 与 deallocation 是 CTA pair 协作操作：

- 两侧同号 warp 都要执行；
- 两侧结果保持对称；
- 释放前要确认两侧都已结束读取。

只让 leader 管理整个生命周期是不完整的。

### 11.6 为什么不要用这个单 tile 的耗时判断 2-CTA 优劣

单 tile 时：

- kernel launch、cluster 调度等固定开销占比大；
- 两种路径本身都太短；
- 测量噪声可能大于数据搬运差异。

M3.4 应重点比较：

- 每 block shared-memory 容量；
- staging store 流量；
- Tensor Core 从 shared memory 取数的 wavefront/流量；
- 节省的 SMEM 是否能在后续换来更深 pipeline 或更大 tile。

这体现一个重要性能原则：

> 一个优化的直接收益可能不是“当前 kernel 立刻更快”，而是释放资源，让下一层优化成为可能。

### 11.7 为什么这类能力更常见于数据中心 GPU

2-CTA MMA 依赖 CTA cluster 以及 cluster 内跨 CTA/跨 SM 的硬件协作。它需要：

- 稳定的 cluster 调度与驻留保证；
- peer shared-memory 访问与硬件互联；
- 两个 SM 上 Tensor Core 的协同执行；
- 更复杂的 multicast 完成通知；
- 足够大的片上资源，让大 tile/多 CTA 协作真正有收益。

数据中心 GPU 更强调大模型吞吐、规则大矩阵和可控调度，愿意为这些机制付出面积与复杂度；消费卡更重视成本、通用图形负载和不同的产品权衡，因此不能仅凭“架构编号更新”推断它一定拥有同样的 2-CTA 能力。

---

## 12. M3 各题到底在训练什么

### 12.1 3.1：建立正确的语义边界

五个判断分别在检查：

1. TMEM lane 的 warp 可见范围；
2. `tcgen05.mma` 的发射粒度；
3. TMEM 到 GMEM 的合法数据通路；
4. TMEM 容量计算；
5. `commit` 与真正等待完成的区别。

做判断题时，不要只写“对/错”，建议用固定句式：

```text
结论 → 指令/地址空间语义 → 对程序数据流的直接后果
```

### 12.2 3.2：把七种语义拼成一条正确数据通路

它不是单纯考 inline PTX，而是在同时验证：

- M2 的 descriptor 与 swizzle；
- TMEM allocation 生命周期；
- 单线程 MMA 发射；
- K 方向多片累加；
- generic/async proxy 可见性；
- mbarrier 完成通知；
- 四个 warp 的 TMEM 读取与输出映射。

调试建议按层推进：

1. 先纸算资源、K 片段、输出覆盖；
2. 再检查 SMEM 物理布局与 descriptor 是否一致；
3. 再检查 first/accumulate 语义；
4. 最后检查异步完成与 epilogue 顺序。

若一开始就把所有问题混在一起看，错误会非常难定位。

### 12.3 3.3：理解“可复用 barrier 是状态机”

3.3 让同一 barrier 连续服务多轮 MMA，并逐轮回读 TMEM。它要求你观察不同 rounds 的现象，再用 phase、arrival count 解释。

记录实验时建议保留：

| seed | rounds | 是否返回 | PASS/FAIL | 首批错误形态 | 推测卡在哪个事件 |
|---:|---:|---|---|---|---|
|  | 1 |  |  |  |  |
|  | 2 |  |  |  |  |
|  | 4 |  |  |  |  |

然后单独画正确/错误状态机。不要把“程序挂死”只写成现象，它通常意味着某一 phase 的 expected arrival 永远无法满足；数值错误则常意味着等待过早返回，`tcgen05.ld` 读取了在飞结果。

### 12.4 3.4：用资源与流量评价 CTA pair

3.4 不要求改程序，重点是先预测后测量：

- 每 CTA 的 B SMEM 怎么变；
- 总 B 流量怎么变；
- A+B 总量怎么变；
- TMEM 仍如何分布在两个 SM；
- cluster/peer access 与 multicast completion 如何配合；
- 节省的 SMEM 能为 M4 pipeline 换来什么。

结论必须以“资源或流量证据”为主，而不是以噪声范围内的单 tile 时间为主。

---

## 13. 常见混淆与排错索引

### 13.1 症状：输出像是固定规律的行/列置换

优先检查：

- SMEM staging 的 swizzle 与 descriptor 是否一致；
- K-major/MN-major 的逻辑坐标解释；
- TMEM address 的 lane/column 是否写反；
- 每个 warp 的 row 区间是否正确。

这种规律性错位通常比随机浮点误差更像布局问题。

### 13.2 症状：第一轮正确，多轮开始失败或挂死

优先画：

- 每轮等待的 parity；
- 每轮 commit 对哪个 phase arrive；
- arrival count 是否能恰好归零；
- 下一轮是否在所有 warp 读完前覆盖 TMEM。

不要首先怀疑 MMA 数学本身。

### 13.3 症状：偶发错误，删/加同步后现象变化

区分三类问题：

| 可能问题 | 应检查的机制 |
|---|---|
| 线程还没都写完 SMEM | CTA/cluster thread synchronization |
| 普通 SMEM 写未对 Tensor Core 可见 | generic→async proxy fence |
| Tensor Core 尚未写完 TMEM | commit + mbarrier wait |
| TMEM→寄存器还在传输 | `tcgen05.wait::ld` |

“多加一个 `__syncthreads()`”不一定能修跨 proxy 或异步引擎完成问题。

### 13.4 症状：结果正确但性能差

依次考虑：

- A/B staging 是否存在 shared-memory bank conflict；
- Tensor Core 是否在等供数；
- 发射/等待之间是否有可重叠工作；
- TMEM 和 SMEM 用量是否限制 occupancy；
- tile 太小导致指令与启动开销占比高；
- CTA 数量不足，无法填满所有 SM；
- 2-CTA 节省的资源是否真的被转化成更深 pipeline/更大 tile。

---

## 14. M3 与 C1 FlashKDA 的直接联系

C1 的问题是：FlashKDA 当前使用 SM80 `mma.sync`，是否值得做 SM100 专版。M3 提供的不是“tcgen05 一定更快”这个结论，而是一套分析工具。

### 14.1 指令形状匹配

需要比较：

- FlashKDA 的 `CHUNK=16` 产生什么矩阵形状；
- SM80 m16 tile 为什么自然；
- tcgen05 的合法 M/N/K 形状是否需要 padding、合并多个 head 或扩大 chunk；
- 无效计算和布局改造成本有多大。

### 14.2 发射开销与并行度

tcgen05 单线程发射减少了参与线程，但 FlashKDA 的瓶颈未必是“发射需要多少线程”。还要看：

- 每卡 head 数是否足够提供 CTA；
- chunk 间递推是否限制并行；
- 是否可将多个 head 合并进一个 CTA；
- persistent kernel 或 CTA pair 是否改善利用率。

### 14.3 累加器从 RMEM 移到 TMEM

潜在收益：

- 减少普通寄存器压力；
- 允许 Tensor Core 与 CUDA Core 更好重叠；
- 为更大 tile 或 double buffer 提供空间。

潜在代价：

- 需要 TMEM alloc/dealloc；
- epilogue 必须经过 `tcgen05.ld`；
- 小 tile 时固定成本可能过重；
- 特殊布局和同步增加实现复杂度。

### 14.4 2-CTA 是否适合 FlashKDA

不能只说“两个 SM 一起算更强”。要问：

- 两个相邻输出 tile 是否共享同一份 B 类输入；
- 每 CTA 保存半份 B 后，SMEM 是否真的成为下一步优化的瓶颈；
- cluster 调度是否减少了可并行 CTA 数；
- recurrent dependency 是否让更大 M tile 难以构造；
- 节省的资源能否换来可测的 pipeline/occupancy 收益。

因此 M3 的最佳学习成果是：你能列出 tcgen05 的收益条件和反例，而不是默认新指令必胜旧指令。

---

## 15. 一页速记版

### 15.1 五个核心对象

```text
SMEM descriptor：告诉 Tensor Core A/B 在 SMEM 怎么摆
idesc：告诉 Tensor Core 这次 MMA 的精确 shape/dtype
TMEM：保存累加器
mbarrier：传递异步完成事件
tcgen05.ld：把 TMEM 结果搬进线程寄存器
```

### 15.2 七步口诀

```text
分配 → 写入 → 可见 → 发射 → 等待 → 读回 → 释放
```

展开：

```text
mbarrier/TMEM alloc
→ A/B 写 SMEM
→ proxy fence + threads sync
→ elected thread MMA + commit
→ consumers wait mbarrier
→ each warp tcgen05.ld + wait::ld + store
→ all readers done + TMEM dealloc
```

### 15.3 三种“等”

```text
等线程：__syncthreads / cluster sync
等异步引擎：mbarrier
等 TMEM load：tcgen05.wait::ld
```

### 15.4 三个最危险的误区

1. `commit` 是登记完成回执，不是阻塞等待。
2. `__syncthreads()` 不能替代 proxy fence 或 mbarrier。
3. 单线程发射 MMA，不代表 alloc、ld、dealloc 都能单线程执行。

---

## 16. 学完后的自测题

先不看前文，尝试用自己的话回答：

1. 为什么 `tcgen05.mma` 能由一个线程发射，而 `tcgen05.ld` 仍需 warp 协作？
2. TMEM 地址的高、低 16 位分别表示什么？
3. m128n64 f32 累加器使用多少 TMEM 字节和多少 column？
4. `kind`、`idesc`、SMEM matrix descriptor 各描述什么？
5. 为什么 K=64 的 bf16 GEMM 在 `kind::f16` 下要分成多个 MMA 片段？
6. `commit`、mbarrier wait、`wait::ld` 分别等待什么？
7. 为什么 `__syncthreads()` 不能保证普通 `st.shared` 的结果已被 Tensor Core descriptor 路径看到？
8. 多轮复用 mbarrier 时，为什么必须跟踪 phase/parity？
9. `cta_group::2` 为什么能减少 B 的重复 staging？
10. 为什么 2-CTA 单 tile 实验不应主要用运行时间下结论？
11. TMEM 资源减少了什么压力，又引入了什么固定开销？
12. 对 FlashKDA 的 `CHUNK=16`，为什么“tcgen05 峰值更高”不足以证明迁移值得？

如果能脱稿讲清这 12 个问题，M3 的知识主线就已经建立完成。

---

## 17. 对照材料

- `session03.pdf`：2.4 “sm100: tcgen05”，S071–S093。
- `handout/src/assignment02.md`：M3 题面和阅读范围。
- `cuda/m3_tcgen05/02_single_tile.cu`：七步流程与题目形状约束。
- `cuda/m3_tcgen05/03_bug_mbarrier.cu`：多轮 mbarrier 状态机实验。
- `cuda/m3_tcgen05/04_cta_pair.cu`：`cta_group::1/2` 的完整对照材料。
- `cuda/m2_smem/02_descriptor.cu`、`03_swizzle.cu`：M3 输入布局的前置知识。
- PTX ISA：`tcgen05`、Tensor Memory、mbarrier、async proxy 相关章节；遇到具体语法时以课程指定 CUDA/PTX 版本为准。

---

## 18. 3.2 完成版逐段讲解

本节对应已经补全的 `cuda/m3_tcgen05/02_single_tile.cu`。先不要逐行陷入 PTX 语法，先抓住它在做的一件事：

```text
D[128, 64] = A[128, 64] × B[64, 64]ᵀ
```

代码中的 B 按 `[N, K]` 保存，所以数学表达式里要看成 `Bᵀ`。

### 18.1 为什么是 128 个线程

一个 CTA 有 128 个线程，也就是四个 warp：

```text
warp 0：thread  0–31
warp 1：thread 32–63
warp 2：thread 64–95
warp 3：thread 96–127
```

MMA 只需要一个 elected thread 发射，但输出是 m128，对应 TMEM 的 128 条 lane。读回时四个 warp 各负责 32 行，所以整个 CTA 仍然安排 128 个线程。

### 18.2 Shared memory 对象

kernel 首先声明：

```cpp
__shared__ __align__(1024) uint8_t sA[M * K * 2];
__shared__ __align__(1024) uint8_t sB[N * K * 2];
__shared__ __align__(8) uint64_t mbar;
__shared__ uint32_t s_taddr[1];
```

它们分别承担：

| 对象 | 作用 | 大小 |
|---|---|---:|
| `sA` | swizzled A tile | `128×64×2 = 16 KiB` |
| `sB` | swizzled B tile | `64×64×2 = 8 KiB` |
| `mbar` | MMA 完成事件 | 8 B |
| `s_taddr` | TMEM allocator 返回的 base address | 4 B |

为什么 `sA/sB` 使用 1024-byte alignment？128B swizzle 的一个完整 atom 是 `8 行 × 128 B = 1024 B`。对齐后，descriptor 的 base 与软件 swizzle atom 边界一致，地址解释更直接。

### 18.3 第一步：初始化 mbarrier，分配 TMEM

分成两种协作粒度：

```text
lane 0：初始化 mbarrier，expected arrival count = 1
warp 0 全体：执行 tcgen05.alloc，申请 64 个 TMEM column
```

为什么申请 64 个 column？输出 N=64，f32 累加器的每个输出列对应一个 32-bit TMEM column。M=128 则自然铺满 128 条 lane。

allocator 把得到的 TMEM base address 写入 `s_taddr`，而不是返回给单个线程的寄存器。后面的 CTA 同步让所有线程都能读取同一个 `taddr`。

`relinquish_alloc_permit` 表示当前 CTA 已完成本次分配操作，可以把 allocator permit 交还给硬件。它不是释放刚申请的 TMEM；真正释放发生在最后的 `dealloc`。

### 18.4 第二步：把 A/B 写成 swizzled SMEM 布局

线程以 grid-stride 方式覆盖全部元素：

```text
thread tid 处理：tid, tid+128, tid+256, ...
```

逻辑坐标由线性下标恢复：

```text
row = i / K
k   = i % K
```

bf16 每个元素 2 byte，所以 K 坐标转成行内字节偏移时使用 `k * 2`。最终物理位置不是 `row*K+k`，而是：

```text
swz128(row, k * 2)
```

这一步的关键不在于“把数据放进 shared memory”，而在于软件写出的物理布局必须与后面的 descriptor 完全一致。

### 18.5 第三步：为什么既要 proxy fence，又要 `__syncthreads()`

这里连续出现：

```text
fence.proxy.async.shared::cta
__syncthreads()
```

两者回答不同问题：

- proxy fence：当前线程通过普通 `st.shared` 写的数据，能否被 Tensor Core 的 async-proxy descriptor 路径看见；
- `__syncthreads()`：CTA 中其他生产线程是否也已经完成 staging 和各自的 proxy fence。

只有 fence：发射线程不知道其他线程是否写完。  
只有 barrier：线程虽然到齐，但没有建立 generic→async proxy 可见性。

### 18.6 第四步：选一个发射线程

`elect.sync` 在 warp 0 中选出一个活动线程。条件：

```cpp
if (warp == 0 && elected)
```

最终只会让一个线程进入 MMA 发射区。其他 127 个线程不需要重复发射相同的矩阵运算。

在该线程开始自己的 tcgen05 操作前，执行 `tcgen05.fence::after_thread_sync`，把前面的跨线程同步关系接入 tcgen05 的操作顺序。

### 18.7 idesc 每一部分在表达什么

完成版构造的 idesc 表达：

```text
D：f32
A：bf16
B：bf16
N：64，以 8 为编码单位，所以字段值是 8
M：128，以 16 为编码单位，所以字段值是 8
```

要分清：

- idesc 描述本次 MMA 的 shape 和 dtype；
- `a_desc/b_desc` 描述本次 K 片段在 SMEM 的起点和布局；
- `kind::f16` 选择 fp16/bf16 这一大类 Tensor Core 运算。

### 18.8 为什么发射四条 MMA

`kind::f16` 的 dense K 步长为 16，而题目 K=64：

```text
kk =  0：处理 K[ 0:16]
kk = 16：处理 K[16:32]
kk = 32：处理 K[32:48]
kk = 48：处理 K[48:64]
```

每个片段通过调整 SMEM descriptor 的 start address 选择：

```text
base + kk × sizeof(bf16)
```

因为 `kk` 每次增加 16，所以字节地址每次增加 32 B，仍满足 descriptor 的 16B 地址编码粒度。

### 18.9 为什么第一条 MMA 不累加

谓词控制 `enable_input_d`：

```text
kk = 0：false，D ← A0×B0ᵀ
kk > 0：true， D ← D + Ak×Bkᵀ
```

第一条不读取 TMEM 中未初始化的旧 D，后面三条才累加到已有部分和。最终 TMEM 中得到完整 K=64 的结果。

### 18.10 commit 为什么只发一次

四条 MMA 由同一个发射线程连续发出，最后执行一次 commit。它的含义是：

> 当前线程在 commit 之前发射的这组 tcgen05 工作全部完成后，对 mbarrier arrive 一次。

因此 mbarrier 的 expected arrival count 初始化为 1，而不是 4。这里计数的是一次 commit 产生的完成事件，不是 MMA 指令条数。

### 18.11 第五步：等待 phase 0

3.2 只使用 barrier 一轮，所以等待初始 phase 的 parity 0 即可。等待成功意味着 commit 覆盖的四条 MMA 都已完成，TMEM accumulator 可以交给消费者。

这里再次强调：commit 返回时不能读取；mbarrier wait 返回后才进入读取阶段。

### 18.12 第六步：四个 warp 怎样读出 m128n64

每个 warp 的 TMEM 起始地址是：

```text
taddr + ((warp × 32) << 16) + col
```

- `warp × 32` 放进高 16 位，选择该 warp 对应的 32 条 lane；
- `col` 放进低 16 位，选择输出列起点。

每次 `.32x32b.x8` 为每条 lane 取 8 个 32-bit 值。因此每个线程得到 8 个 f32 寄存器，列循环为：

```text
col = 0, 8, 16, 24, 32, 40, 48, 56
```

八轮正好覆盖 N=64。行号：

```text
row = warp × 32 + lane
```

四个 warp 联合覆盖 row 0–127，没有重叠也没有遗漏。

`tcgen05.ld` 后立即跟 `tcgen05.wait::ld`，因为 ld 也不是目标寄存器立即就绪的普通同步 load。等寄存器可用后，线程才能写 `gD[row, col:col+8]`。

### 18.13 第七步：为什么释放前还要一次 CTA 同步

四个 warp 的 epilogue 进度可能不同。若 warp 0 读完后立刻 dealloc，其他 warp 可能仍在读取相同 TMEM allocation。

因此：

```text
四个 warp 全部完成读回
        ↓ __syncthreads
warp 0 全体执行 dealloc
```

这次同步保护的是资源生命周期，而不是 MMA 完成。

### 18.14 3.2 一句话复述

```text
全 CTA 把 A/B 按 descriptor 认可的 swizzle 放入 SMEM，
一个线程沿 K 发四条 tcgen05.mma 并把完成通知交给 mbarrier，
四个 warp 等完成后各读 32 行 TMEM，最后统一释放 TMEM。
```

---

## 19. 3.3 的问题：为什么多轮 mbarrier 会出错

### 19.1 3.3 和 3.2 的计算策略不同

3.2 的做法是：

```text
四个 k16 MMA 全部在 TMEM 内累加
→ commit 一次
→ wait 一次
→ 最后读取完整结果
```

3.3 故意改成：

```text
每轮只算一个 k16 partial product，且不读取旧 TMEM D
→ 每轮 commit
→ 每轮 wait
→ 每轮把 TMEM 读进 RMEM
→ 在普通 float acc 寄存器里累加四轮
```

所以 3.3 会重复使用同一个 mbarrier，用来检验你是否真正理解 phase/parity。

### 19.2 问题所在

循环内当前写法每轮都执行：

```cpp
mbar_wait(mbar_u32, 0);
```

也就是始终拿 parity 0 的 token 去等待。但是 mbarrier 每完成一个 phase，parity 会翻转：

```text
phase 0 完成：parity 0 → 1
phase 1 完成：parity 1 → 0
phase 2 完成：parity 0 → 1
...
```

等待参数表示“我看到本轮开始时是这个 parity，请等到它发生变化”。因此每轮必须跟踪当前 phase token，不能永远传 0。

### 19.3 第一轮为什么常常正常

初始 barrier 位于 phase 0：

```text
round 0 开始：当前 parity = 0
发 MMA0 + commit0
wait(token=0) 尚未看到翻转，所以阻塞
MMA0 完成 → commit0 arrive → parity 变成 1
wait(token=0) 观察到变化，正确返回
```

因此 `rounds=1` 可能完全通过，把 bug 隐藏起来。

### 19.4 第二轮为什么危险

第二轮开始时当前 parity 已经是 1，但代码仍传 token 0：

```text
round 1 开始：当前 parity = 1
发 MMA1 + commit1
wait(token=0)：发现当前 parity 已经不是 0
               → 可能立即认为“等待条件已满足”
```

这个“变化”其实来自上一轮，而不是本轮 MMA1。于是消费者可能在 MMA1 尚未完成时执行 `tcgen05.ld`。

后果并不保证只有一种：

- 读到上一轮残留结果；
- 读到尚未完整更新的结果；
- TMEM 上的 in-flight MMA 与 ld 产生非法/未定义的时序；
- 程序错误退出或挂死。

所以题目要求同时记录 FAIL 和 timeout；不要假设同步 bug 必定表现为稳定的数值差异。

### 19.5 为什么 `__syncthreads()` 救不了这个问题

循环末尾确实有 `__syncthreads()`，但它只能保证四个 warp 都完成了自己当前走到的代码位置。

若 mbarrier wait 已经过早返回，那么所有线程都可能一致地、整齐地去读取一个仍在被 Tensor Core 写入的 TMEM。线程全部到齐，并不等于异步 MMA 已完成。

### 19.6 正确的 phase 跟踪方式

需要在循环外保存当前 phase token，并在每轮成功等待后翻转：

```cpp
uint32_t phase = 0;
for (int round = 0; round < rounds; round++) {
    // 发射本轮 MMA，并 commit 到同一个 mbarrier
    mbar_wait(mbar_u32, phase);
    phase ^= 1;

    // 现在才可读取本轮 TMEM
    // tcgen05.ld + tcgen05.wait::ld
    // 所有 warp 读完后再进入下一轮
}
```

在这个题目的严格结构中，每轮恰好完成一个 barrier phase，所以也可以从轮次推导 parity；显式保存并翻转 phase 更能表达状态机含义，也更容易推广到复杂 pipeline。

### 19.7 修正后的逐轮状态图

```text
初始 token = 0

round 0
  issue MMA0 → commit0 → wait(0)
  hardware complete：parity 0→1
  wait 返回，软件 token 0→1
  ld0 → wait::ld → acc += partial0
  四个 warp 同步

round 1
  issue MMA1 → commit1 → wait(1)
  hardware complete：parity 1→0
  wait 返回，软件 token 1→0
  ld1 → wait::ld → acc += partial1
  四个 warp 同步

round 2
  issue MMA2 → commit2 → wait(0)
  hardware complete：parity 0→1
  wait 返回，软件 token 0→1
  ld2 → wait::ld → acc += partial2
  四个 warp 同步

round 3
  issue MMA3 → commit3 → wait(1)
  hardware complete：parity 1→0
  wait 返回，软件 token 1→0
  ld3 → wait::ld → acc += partial3
```

### 19.8 arrival count 在每轮如何变化

mbarrier 初始化 expected arrival count 为 1。每个 phase 中：

```text
phase 开始：arrival count = 1
本轮 commit 对应的 MMA 完成
        ↓
commit 触发 arrive::one
        ↓
arrival count = 0
        ↓
本 phase 完成，parity 翻转
        ↓
下一 phase 重新等待一次完成事件
```

这里不需要每轮重新执行 `mbarrier.init`。mbarrier 本身就是可跨 phase 复用的状态机，软件只需要正确携带 token/parity。

### 19.9 为什么每轮 MMA 都设置“不累加 TMEM”

3.3 想逐轮读取一个 k16 partial product，再在普通寄存器 `acc` 中累计。因此每轮的 TMEM D 都应该表示：

```text
D_round = A[:, kk:kk+16] × B[:, kk:kk+16]ᵀ
```

而不是在 TMEM 内继续叠加历史结果。否则 RMEM 再执行 `acc += D_round` 时，会重复计算前面片段：

```text
错误示意：acc += partial0
          acc += (partial0 + partial1)
```

所以 3.3 中“每轮覆盖 TMEM、RMEM 负责总累加”是刻意设计的实验结构，不是另一个 bug。

### 19.10 3.3 的完整依赖链

每轮必须严格满足：

```text
上一轮四个 warp 已读完 TMEM
        ↓
发射本轮 MMA，覆盖 TMEM
        ↓
commit 登记本轮完成事件
        ↓
等待与本轮匹配的 parity
        ↓
tcgen05.fence::after_thread_sync
        ↓
四个 warp tcgen05.ld
        ↓
tcgen05.wait::ld
        ↓
普通寄存器 acc 累加
        ↓
__syncthreads，允许下一轮覆盖
```

其中每条边都有独立意义：

- phase/parity：防止拿上一轮完成事件冒充本轮；
- mbarrier wait：防止读取仍在飞的 MMA；
- `wait::ld`：防止使用尚未到达寄存器的数据；
- 循环末 CTA 同步：防止下一轮覆盖时仍有 warp 在读上一轮。

### 19.11 实验报告应该怎样写

建议报告分三块：

1. **现象表**：记录 `rounds=1/2/4`、seed、PASS/FAIL/timeout；数据必须来自 B300 实测。
2. **错误状态机**：画出第二轮仍等待 token 0，导致旧 parity 过早满足。
3. **正确状态机**：画出软件 phase token 每轮与硬件 parity 同步翻转。

一句核心归因可以写成：

> 错误版本重复使用 phase-0 token；第一轮完成后 barrier parity 已翻转，后续某轮的 wait 可能被历史 phase 状态提前满足，使 `tcgen05.ld` 在本轮 MMA 尚未完成时访问 TMEM。修正后，软件为每次 barrier 复用维护并翻转 phase token，使每轮读取只在对应 commit 完成后发生。
