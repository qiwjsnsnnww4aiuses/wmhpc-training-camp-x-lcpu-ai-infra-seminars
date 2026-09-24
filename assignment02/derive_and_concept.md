以下是作业中所有 **DERIVE** 和 **CONCEPT** 题目的提取汇总：

---

## DERIVE 题

---

### 2.2 {.prob type=DERIVE file=cuda/m2_smem/02_descriptor.cu}

根据文件头给出的位域，实现 SM100 的 64 位 smem matrix descriptor 编码函数，并分别对下面三种情况推导 LBO、SBO 和 layout：

- K-major，无 swizzle；
- K-major，128B swizzle；
- MN-major，128B swizzle。

判测为纯 host，无需 GPU。测试使用的描述符真值已经在 B300 上通过实际 tcgen05 指令验证，3.2 中也会使用这三组描述符：

```
cd assignment02/cuda
make run/m2_smem/02_descriptor
```

场景 2 和场景 3 最终得到的 descriptor 相同。在报告中回答：MN-major 与 K-major 的区别体现在哪里？

---


## CONCEPT 题

### 0.3 {.prob type=CONCEPT}

判断下列说法是否正确，并给出一句理由。

(a) 一条 mma 的计算强度，分子是 $2MNK$，分母按 A、B 读入与 D 写回的字节总和计（S016 的口径）。
对。2表示乘法和加法两次操作，分母是读入写回的总字节数

(b) mma.sync 是 warp 级协作指令：32 个 lane 各持 fragment 的一部分，要求全 warp 一致地执行这条指令；有 lane 发散时行为未定义。
对。mma.sync表示矩阵乘法且warp内部lane必须同步

(c) 增大 mma 的形状 M/N/K 能提高单条指令的计算强度，而且没有代价，所以指令形状越大越好。
错误。M太小的确会退化到被M限制计算强度，但是太大不利于数据搬运如fragment受限

(d) 只要单条 mma 的计算强度低于机器平衡点，GEMM kernel 就不可能逼近计算峰值。
错。多条mma可以隐藏延迟等多种手段提升计算强度

---

### 2.1 {.prob type=CONCEPT}

(a) 一个 warpgroup 使用 wgmma 读取刚写入 shared memory 的数据。将下面六个操作排成正确顺序，并说明每一步用于避免哪两个参与者之间的哪种乱序：

`wgmma.mma_async` / `st.shared` / `wgmma.commit_group` / `fence.proxy.async` / `wgmma.fence` / `wgmma.wait_group`
st.shared -> fence.proxy.async -> wgmma.fence -> wgmma.mma_async -> wgmma.commit_group -> wgmma.wait_group

(b) 判断下列说法是否正确，并给出一句理由。

1. `fence.proxy.async` 是 wgmma 专属的指令，TMA 与 tcgen05 的场景不需要它。
错。fence.proxy.async是通用的group_proxy-async_proxy间的同步指令，TMA tcgen05需要
2. `wgmma.commit_group` 会阻塞，直到它之前发射的 wgmma 全部完成。
错。wgmma.commit_group是把一个warpgroup发出的指令打包，交给wgmma.wait_group处理
3. 不加 `fence.proxy.async` 时，wgmma 可能读到 shared memory 中的旧值，因为 `st.shared` 的写经过 generic proxy，而 wgmma 的读经过 async proxy。
对。generic proxy 和 async proxy 是两个不同的域，fence.proxy.async保证之前的修改互相可见
另外wgmma.fence是保证取数据的顺序性-smem读完数据再发射
---

### 3.1 {.prob type=CONCEPT}

判断下列说法是否正确，并给出一句理由。其中 (d) 需要写出计算过程。

(a) `tcgen05.ld` 读取 TMEM 时，每个 warp 只能读取自己对应的 32 条 lane，warp 之间不能互相读取。
对。tcgen05.ld是warp协作指令，每个warp只能读取自己的lane（对应的TMEM rows）

(b) 与 `mma.sync` 由 warp 协作、wgmma 由 warpgroup 协作不同，`tcgen05.mma` 由单个线程发射，随后由硬件异步执行。
对。mma.sync是warp协作，而wgmma是warpgroup协作，发射颗粒度不同

(c) TMEM 中的累加结果可以直接通过 TMA 搬回 global memory，不需要经过寄存器。
错。需要tcgen05.ld先从寄存器把数据搬回TMEM

(d) TMEM 每个 SM 包含 128 lane × 512 column × 4 B；一个 m128n256 的 f32 accumulator 恰好占用其中一半。
对。s256*4Bytes/4Bytes = 256 columns 恰好一半

(e) `tcgen05.commit` 会阻塞直到之前发射的 mma 全部完成，因此 commit 返回后即可安全读取 TMEM。
错。tcgen05.commit是把之前所有的异步操作mma copy等交给mbarrier管
