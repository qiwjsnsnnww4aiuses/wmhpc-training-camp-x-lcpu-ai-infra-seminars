# FlashKDA K2：SM80 超前进位 Reduce 最终方案

> 核心：把 K2 的 chunk 状态递推映射为“传播 A + 生成 B”，使用类似 carry-lookahead / Brent--Kung 的上扫 reduce 和下扫 carry，在 `O(log m)` 深度内得到所有 segment 的输入状态。

---

## 1. 与超前进位的对应关系

普通进位递推：

```text
carry_i = propagate_i * carry_(i-1) + generate_i
```

KDA chunk 状态递推：

```text
S_i = A_i * S_(i-1) + B_i
```

对应关系：

| 超前进位 | KDA |
|---|---|
| carry | 状态矩阵 S |
| propagate P | 状态传播矩阵 A |
| generate G | 状态生成矩阵 B |
| group P/G | 一段 chunks 的 A_group/B_group |

两个相邻区间的 reduce 运算：

```text
left  = (A_L, B_L)
right = (A_R, B_R)

right o left = (
    A_R @ A_L,
    A_R @ B_L + B_R
)
```

它具有结合性，所以可以构造树形 reduce。

---

## 2. 为什么不是普通 Hillis--Steele scan

若有 m 个 segment：

```text
Hillis--Steele:
    depth = O(log m)
    work  = O(m log m)

超前进位 reduce/down-sweep:
    depth = O(log m)
    work  = O(m)
```

Hillis--Steele 会在每一层重新更新大量已经计算过的 prefix。超前进位树只计算：

```text
上扫：m-1 个区间组合
下扫：m-1 个节点的 carry 传播
```

这才是应该使用的正式方案。

---

## 3. 为什么仍然需要 segment

如果直接对 512 个 chunk 建树，每个 leaf 都保存完整 bf16 A/B：

```text
512 chunks/head * 96 heads * 64 KiB
= 3.00 GiB leaf maps
```

内部树节点还需要接近同样大小的存储，最终约 6 GiB。

而且树中每次区间组合都包含两个 `128×128` GEMM。因此先把 G 个 chunk 压缩为一个 segment：

```text
512 chunks
    -> G=32
16 segment maps
    -> carry-lookahead tree
16 segment start states
    -> parallel replay
```

这样既落实超前进位，又控制 A/B 数量。

---

## 4. 最终四阶段执行链

```text
K1：原版 chunk-parallel prepare
  |
  v
K2a：Segment Reduce Input
     每个 segment 独立生成 A_seg/B_seg
  |
  v
K2b：Up-sweep Reduce
     自底向上生成 group A/B
  |
  v
K2c：Down-sweep Carry
     自顶向下传播真实 S_start
  |
  v
K2d：Parallel Replay
     所有 segment 同时运行原版 chunk 计算并生成 output
```

K2b/K2c 对应超前进位网络；K2a/K2d 用于避免给每一个 chunk 物化完整 A/B。

---

## 5. K2a：生成 segment 的传播/生成项

每个 CTA 对应：

```text
(sequence, head, segment)
```

CTA 内同时维护：

```text
state_A[D,D]，初始为单位矩阵 I，计算时令 V=0
state_B[D,D]，初始为 0，使用真实 V
```

顺序执行该 segment 内的 G 个 chunk 后：

```text
state_A = A_seg
state_B = B_seg
```

因为：

```text
F_seg(I, V=0) = A_seg
F_seg(0, real V) = B_seg
```

第一版保持 K1 不变。Summary 从原 workspace 读取：

```text
Kd, Kr, g_total, INV
```

并从原始输入读取：

```text
V, beta
```

不需要 Qd/Mqk，因为 summary 不生成 output。

---

## 6. K2b：Up-sweep Reduce

以 8 个 segment 为例：

```text
level 0:
T0  T1  T2  T3  T4  T5  T6  T7
 \ /     \ /     \ /     \ /
 T01     T23     T45     T67

level 1:
    \     /           \     /
     T03               T47

level 2:
          \           /
              T07
```

每个父节点：

```text
A_parent = A_right @ A_left
B_parent = A_right @ B_left + B_right
```

每层是一个 CUDA kernel launch，天然提供跨 CTA 的全局同步。

树深度：

```text
ceil(log2(m))
```

非 2 的幂 segment 数可以：

- 为缺失 leaf 使用 identity map `(I,0)`；或
- 在 kernel 中检查右子节点是否存在。

第一版建议 padding 到 2 的幂，逻辑更简单。

---

## 7. K2c：Down-sweep Carry

这是与普通 full-prefix scan 最大的区别。

根节点获得真实 initial state：

```text
S_root = initial_state 或 0
```

对每个内部节点：

```text
左子树起始状态：
S_left = S_parent

右子树起始状态：
S_right = A_left @ S_parent + B_left
```

也就是说：

- 状态直接传给左子树；
- 左区间的 group propagate/generate 一次性算出右子树 carry；
- 不需要为每个节点生成完整 exclusive-prefix A/B。

一直传播到 leaf 后：

```text
S_leaf[j] = segment j 的真实输入状态
```

这就是超前进位：不是等前一个 segment 真正执行完再得到 state，而是通过区间 A/B 在树中提前计算 carry。

---

## 8. K2d：Parallel Replay

得到所有：

```text
S_leaf[0 ... m-1]
```

后，每个 segment CTA 独立运行：

```text
state = S_leaf[segment]

for chunk in this segment:
    运行原版 K2 chunk 主循环
    生成 output
    更新 state
```

所有 segment 同时 replay。只有最后一个 segment 写 final state。

同 segment 内仍有 G 个 chunk 的串行依赖，但原版的 512 长链变为 G 长链。

---

## 9. SM80 实现

所有矩阵乘继续使用：

```text
SM80_16x8x16_F32BF16BF16F32_TN
```

架构组合仍与原版一致：

```text
global <-> shared：SM90 TMA
矩阵计算：SM80 mma.sync / HMMA.16816
```

### Up-sweep 节点

一个 node 需要两个 D×D GEMM：

```text
A_R @ A_L
A_R @ B_L
```

建议拆成两个 CTA，通过 `blockIdx.z` 区分：

```text
z=0 -> 输出 A_parent
z=1 -> 输出 B_parent，并加 B_R
```

每个 CTA 使用 8 个 warp；`128×128` 输出拆成 64 个 `16×16` tiles，每 warp 负责 8 个。

### Down-sweep 节点

只需要对右子树做一个 GEMM：

```text
S_right = A_left @ S_parent + B_left
```

左子状态是 copy/alias：

```text
S_left = S_parent
```

所以一次内部节点下扫只需一个 D×D GEMM，而不是重新组合两张 prefix map。

---

## 10. Global memory 树布局

推荐按 level 单独分区：

```text
map_A_levels:
level 0: m leaf maps
level 1: ceil(m/2)
level 2: ceil(m/4)
...

map_B_levels: 同样布局
```

每条 sequence/head 独立一棵树。

所有 level 节点总数小于：

```text
2*m
```

bf16 A/B map tree 上限：

```text
2*m nodes * 64 KiB/node
```

G=32、m=16、H=96：

```text
leaf + internal map tree < 192 MiB
```

另外需要 carry state buffers。每个状态为：

```text
D*D*2 = 32 KiB
```

如果为整棵树保存状态，上限约 96 MiB。更好的实现是 down-sweep 只保留相邻两层，最大约 72 MiB。

总体新增 workspace 约：

```text
map tree 约 192 MiB
carry levels 约 48--72 MiB
合计约 240--264 MiB
```

加原版 K1 workspace 约 648 MiB，总计约 0.87--0.89 GiB。

---

## 11. Shared memory

### Segment summary

```text
state_A 32 KiB
state_B 32 KiB
2-stage inputs 约 26 KiB
barrier/alignment
总计约 90--94 KiB
```

建议：

```text
1 load warp
4 MMA warps 更新 A
4 MMA warps 更新 B
共 288 threads
```

### Reduce/down-sweep GEMM

第一版可以把两张输入 D×D bf16 map 放 shared：

```text
2*32 KiB = 64 KiB
```

输出用 register fragment 直接 vector store 到 global，避免增加 32 KiB output shared。

### Replay

复用原 K2 shared layout，约 98.43 KiB。区别只是每 CTA 的 chunk loop 从 512 缩短为 G。

---

## 12. Work-efficient 计算量

### 原版 K2

fixed `T=8192,H=96`：

```text
约 83.75 GFLOP
```

### Summary

A/B 两条状态轨迹：

```text
约 109.52 GFLOP
```

基本不随 G 改变。

### Carry-lookahead tree

一次 up-sweep combine：

```text
2 个 D×D GEMM
= 8,388,608 FLOP
```

一次 down-sweep 右 carry：

```text
1 个 D×D GEMM
= 4,194,304 FLOP
```

m 个 leaf 总共：

```text
up-sweep   m-1 个 combine
down-sweep m-1 个 carry GEMM
```

H=96：

| G | m | Tree GFLOP | Summary+Tree+Replay | 相对原 K2 |
|---:|---:|---:|---:|---:|
| 8 | 64 | 76.1 | 269.4 | 3.22× |
| 16 | 32 | 37.4 | 230.7 | 2.75× |
| 32 | 16 | 18.1 | 211.4 | 2.52× |
| 64 | 8 | 8.5 | 201.7 | 2.41× |

说明：表中 tree 包含上扫和下扫；replay 约为原 K2 的 83.75 GFLOP。

相比 Hillis--Steele，G=32 的 scan/tree 工作从约 39.5 GFLOP 降到约 18.1 GFLOP，并且直接输出 leaf 起始状态。若下扫状态 buffer 和 copy 进一步融合，收益还会更好。

真正最大的额外工作仍然是双状态 summary，而不是 scan 本身。

---

## 13. 对性能预期的修正

原版 K2：

```text
约 0.75 ms
Compute throughput 约 21%
grid=96
```

G=32：

```text
summary/replay grid=1536
树深度=4 层 up + 4 层 down
总 tensor 工作约原版 2.52×
```

忽略 global traffic，计算回本要求：

```text
新平均利用率 > 2.52*21% = 52.9%
```

加上 tree map/carry global traffic和 8 次 level launch，实际需要约 58%--63% 的有效利用率才可能跑赢。

因此预测：

```text
第一版：完整 forward 大概率慢 5%--20%
调优成功：可能快 5%--10%
非常理想：约 10%--15%
```

超前进位 reduce 改善了普通 scan 的工作效率，但没有消除 segment summary/replay 的额外工作，所以不能承诺大幅提升。

---

## 14. 最关键的 go/no-go gate

原版每 chunk 将 state 落为 bf16：

```text
S_out = round_bf16(A*S_in+B)
```

round 破坏严格线性，因此 segment A/B 是对数学递推的近似，而不保证逐 bit 复现原版。

在写完整树之前，先完成：

```text
对随机 S_test：

prediction = A_seg@S_test+B_seg
reference  = 原版 K2 从 S_test 顺序跑 G chunks
```

测试：

```text
G=1,2,4,8,16,32,64
T up to 8192/32768
随机和极端 gate/beta
output/state rel_rmse
```

如果 G=16/32 时 segment-map 误差已经不可接受，超前进位方向应停止或切换到 FP32/TF32 map，而不是继续优化 CUDA 树。

---

## 15. MVP 实施顺序

1. PyTorch 实现 `(A,B)` combine 与 reduce/down-sweep，确认顺序；
2. CUDA 实现 G=32 segment summary，验证 `A_seg@S+B_seg`；
3. SM80 microbench 实现一个 up-sweep node；
4. 实现 4 层 up-sweep reduce；
5. 实现 4 层 down-sweep carry，导出全部 leaf `S_start`；
6. 对拍 leaf 状态；
7. 将原 K2 改为 segment replay；
8. 端到端 fixed H96 correctness；
9. 分阶段 benchmark/NCU；
10. 再加入 G autotune、H12、varlen 和 initial state。

第一版范围：

```text
fixed N=1
T=8192,H=96,D=128
CHUNK=16,G=32
no initial state
bf16 maps
SM80 BF16 MMA / FP32 accumulator
```

---

## 16. 最终结构

```text
                    上扫 reduce
          [0..7]                      [8..15]
         /      \                    /       \
     [0..3]    [4..7]            [8..11]   [12..15]
       ...       ...                ...        ...

                    下扫 carry
                       S0
                 /            \
              S[0]           S[8]
             /   \           /   \
          S[0]   S[4]     S[8]   S[12]
           ...    ...      ...     ...

leaf S[j] -> segment j 并行 replay
```

一句话总结：

> K1 继续生成 chunk 局部因子；K2a 把 G 个 chunk reduce 成 group propagate/generate；K2b 上扫生成区间 A/B；K2c 像超前进位一样把真实状态 carry 下传到每个 leaf；K2d 从所有 leaf 状态并行 replay。这样才是真正落实“超前进位 + reduce”并创造 K2 chunk 间并行。
