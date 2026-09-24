# C1 FlashKDA FInal：结果分析与完整思路链

本文只使用 `assignment02/team/c1_flashkda/TASK.md` 的要求、当前源码和本次
B300 实测日志作结论。没有测到的内容会明确写成“尚未验证”。

## 0. 一句话结论

本次选择了 `TASK.md` 挑战层允许的“只换指令不动算法”切面：保留 FlashKDA
的 K1、CHUNK=16 和 chunk 间递推关系，把 K2 recurrence 中的五类 SM80 MMA
矩阵乘改成 tcgen05。

结果是：

- 改版在 B300、CUDA 13.0 上成功编译、安装、运行；
- fixed 和 varlen 的 output、final state 与原版 FlashKDA 逐元素完全相等；
- 对 `naive.py` 和 FLA `chunk_kda` 的相对 RMSE 约为 0.46%～0.62%；
- SASS 证明 K2 recurrence 中生成了 UTCHMMA；
- 完整 `T=8192,H=96,D=128` forward 从 1.755060 ms 变成 17.552482 ms；
- tcgen05 改版慢 10.001 倍，即所谓 speedup 只有 0.1000x。

因此挑战层已经得到一个合法的负结果：对当前 CHUNK=16 和当前并行组织，直接把
K2 的 SM80 MMA 换成 tcgen05 不值得；这正好补强讨论问题 6 中“官方停在 SM80
是合理的”一侧。

注意结论边界：本次替换范围是 **K2 recurrence**。K1 prepare 仍使用 HMMA。
`TASK.md` 允许“任一切面”，所以这满足挑战选题；不能把它描述成“FlashKDA
所有 kernel 都不再含 SM80 指令”。

---

## 1. 按 TASK.md 三层要求核对

### 1.1 第一层：复现

要求：B300 上安装原版、跑官方 benchmark、用 NCU/SASS 确认原版主路径是 SM80
MMA。

当前状态：

- 原版测试和官方形状 benchmark 已经在此前流程中跑通；
- 本地保存了 fixed/varlen 的 NCU key metrics；
- 本次 `FInal/results` 主要保存的是挑战改版证据，不是原版复现证据全集。

因此从“工作是否做过”看，复现已完成；从“最终交付是否自包含”看，还应该把原版的
test、benchmark、原版 SASS 和 NCU 文件统一归档到最终报告目录。改版 SASS 不能替代
原版 SASS。

### 1.2 第二层：六个讨论问题

已有 `C1_Task讨论答案与挑战报告.md` 覆盖了六问。与本次新结果的关系是：

- 问题 1：仍由 CHUNK=16 的数值范围、Neumann inverse 和 MMA 形状分析回答；
- 问题 2：此前 instruction-only microbench 是“局部指令层”证据，本次完整 K2
  实验是“真实 kernel 层”证据；两层证据方向一致；
- 问题 3：本次 K2 仍是一条 sequence/head 一个 CTA，CTA 内顺序遍历 chunk，实际
  证明 chunk 依赖没有因为换指令消失；
- 问题 4：此前原版 NCU 指标回答 compute/memory/并行度瓶颈；本次没有对 tcgen05
  改版再跑 NCU，所以不能用当前日志进一步拆分 10 倍慢的硬件指标；
- 问题 5：当前 correctness 数据能证明本改版没有额外精度损失，但它不是完整的
  “BF16 state vs FP32 state 长序列压力测试”。问题 5 若要扎实，仍需专门的多 seed、
  多门控、长序列 BF16/FP32 state 消融数据；
- 问题 6：本次 0.1000x 的负收益，是“不发布这种朴素 K2 tcgen05 专版”的直接证据。

### 1.3 第三层：挑战

`TASK.md` 对挑战的硬要求是：

1. 选择一个 SM100 切面；
2. 有代码；
3. 正确性对 `naive.py` 和 `chunk.py`；
4. 性能对 FlashKDA 本体；
5. 允许负收益。

本次均已覆盖。当前挑战可以判定为“完成”，但报告中要写清是 K2 切面和负结果。

---

## 2. 实测环境与构建结果

环境日志：

```text
host: dev-slurm
GPU: NVIDIA B300 SXM6 AC
compute capability: 10.3
CUDA toolkit: 13.0, nvcc 13.0.88
Python: 3.12.13
PyTorch: 2.14.0+cu130
```

`build.log` 最终显示：

```text
Successfully built flash_kda_tcgen05
Successfully installed flash_kda_tcgen05-0.0.1
```

生成的动态库为：

```text
.../FInal/FlashKDA-tcgen05/
flash_kda_tcgen05_C.cpython-312-x86_64-linux-gnu.so
```

### 2.1 两类编译警告

第一类是 CUTLASS 对 `long4/ulong4/...` 的 deprecated 警告，与挑战逻辑无关。

第二类值得记录：

```text
specified alignment (1024) is different from alignment (128)
specified on a previous declaration
```

原因是 K1 对同名动态 shared-memory 符号声明了 128B 对齐，K2 tcgen05 分支声明
1024B 对齐。本次运行正确，说明没有造成可见运行错误，但最终清理代码时最好统一动态
shared-memory 声明，避免依赖实现给出的实际地址对齐。

少数 K2 模板实例还有：

```text
4 bytes spill stores, 8 bytes spill loads
```

绝大多数实例使用 64 个寄存器且没有 spill。这一点不是 10 倍慢的主要解释，但应在
报告中如实记录。

---

## 3. 正确性结果怎么读

### 3.1 tcgen05 对原版 FlashKDA

| case | 比较对象 | max_abs | mean_abs | rel_rmse |
|---|---|---:|---:|---:|
| fixed `[64]` | output | 0 | 0 | 0 |
| fixed `[64]` | final_state | 0 | 0 | 0 |
| varlen `[17,31,16]` | output | 0 | 0 | 0 |
| varlen `[17,31,16]` | final_state | 0 | 0 | 0 |

这表明在当前 seed、H=2 和 T=64 的测试里，改版与原版最终 BF16 output/state 逐元素
完全一致。

“完全一致”本身不能证明 tcgen05 被执行，因为也可能错误地调用了同一个扩展。因此
必须与后面的三条证据合起来看：

1. 改版动态库路径独立，名称是 `flash_kda_tcgen05_C`；
2. 改版端到端时间比原版慢 10 倍，不像调用了同一实现；
3. 改版 K2 recurrence 的 SASS 明确包含 UTCHMMA。

这三点与逐元素相等一起构成完整证据链。

### 3.2 对 `fla_kda_ref/naive.py`

| case | 实现 | output rel_rmse | state rel_rmse |
|---|---|---:|---:|
| fixed | 原版/改版 | 0.00553685 | 0.00476515 |
| varlen | 原版/改版 | 0.00588561 | 0.00462527 |

原版和改版误差完全相同，说明误差主要来自 FlashKDA 与朴素参考之间已有的 BF16
落盘、近似 sigmoid/exp 和 reduction 顺序差异，而不是 tcgen05 改版引入的新误差。

### 3.3 对 FLA `chunk_kda`

| case | 实现 | output rel_rmse | state rel_rmse |
|---|---|---:|---:|
| fixed | 原版/改版 | 0.00541033 | 0.00497233 |
| varlen | 原版/改版 | 0.00623464 | 0.00485505 |

所有相对 RMSE 都低于 0.7%，并且改版没有比原版更差。

一个需要在最终归档中补清的小点：`naive.py` 是按 `--ref-root` 从课程快照直接加载；
Triton 路径当前由 `from fla.ops.kda import chunk_kda` 加载已安装的 FLA。最好补记已安装
FLA 的 commit/version，证明它与题目 pin 的 `a3edffc` 相同；或者让验证脚本显式记录
模块的 `__file__`。

### 3.4 fixed 与 varlen 各自验证了什么

fixed `[64]`：

- 一条连续序列；
- 64/16=4 个 chunk；
- 同一个 K2 CTA 连续做四次状态递推。

varlen `[17,31,16]`：

- 三条独立序列被拼进一个 `[1,64,H,D]` 张量；
- `cu_seqlens=[0,17,48,64]` 指出每条序列边界；
- 17 和 31 都不是 16 的倍数，所以同时覆盖 tail chunk；
- 每条序列使用独立 initial/final state，不能跨序列继承状态。

因此 varlen 不是“换了一个 g”，而是在验证序列寻址、tile prefix、尾块写回和状态隔离。

---

## 4. 性能结果与负收益解释

实测同一进程、同一输入、同一 B300：

| 实现 | mean | min | max |
|---|---:|---:|---:|
| FlashKDA SM80 | 1.755060 ms | 1.749056 ms | 1.767328 ms |
| FlashKDA K2 tcgen05 | 17.552482 ms | 17.512993 ms | 17.694336 ms |

```text
tcgen05 / SM80 latency = 10.0011
latency increase       = 900.1%
reported speedup       = SM80/tcgen05 = 0.1000x
```

两组 min-max 展宽都约为各自 mean 的 1.03%～1.04%，单轮数据虽然不足以做严格统计，
但 10 倍差距远大于测量抖动，不可能由普通噪声解释。

### 4.1 每个 chunk 实际发出多少 tcgen05

当前 K2 每 chunk 的运行时 tcgen05 次数：

```text
k_decayed @ state^T：K=128，拆成 8 个 K=16 tile  -> 8
q_decayed @ state^T：K=128，拆成 8 个 K=16 tile  -> 8
INV @ U0                                           -> 1
Mqk @ U                                            -> 1
k_restored^T @ U                                   -> 1
总计                                               -> 19
```

其中前 18 次的逻辑 M 都是 16，但物理 M 是 128；只有最后一次状态增量天然是
128x128x16。

按 Tensor Core 乘加工作量计算：

```text
physical/useful
= (19 * 128) / (18 * 16 + 1 * 128)
= 2432 / 416
= 5.846
```

即只有约 17.1% 的 tcgen05 物理工作对应有用输出，约 82.9% 是 tile 形状不匹配导致
的补零工作。

### 4.2 除了 5.846 倍无效计算，还有什么

当前实现为保证正确性复用了同一组 shared-memory A/B buffer，因此每个 tcgen05 tile
都要经历：

```text
写 swizzled shared memory
→ fence
→ compute barrier
→ tcgen05.mma
→ tcgen05.commit
→ mbarrier wait
→ tcgen05.ld
```

对两个 K=128 GEMM，还要反复 8 次 staging + commit + wait。

原版 SM80 路径则有两个关键优势：

- `k@s` 和 `q@s` 是 dual GEMM，可以共享 state 的 B operand 搬运；
- U 尽可能留在寄存器中，通过 MOVM_T 在后续 phase 复用。

改版为了适配 tcgen05，串行化了原来融合的工作，并增加 shared memory、TMEM 读回和
同步开销。因此端到端慢 10 倍比单纯的 5.846 倍物理工作膨胀更严重，是合理现象。

### 4.3 为什么 T=8192 会放大问题

```text
8192 tokens / 16 tokens per chunk = 512 chunks
```

一个 sequence/head 的 K2 CTA 必须按顺序执行这 512 个 chunk。每个 chunk 增加的
19 次 tcgen 路径开销不能通过把相邻 chunk 同时算来隐藏，因为下一个 chunk 需要上一个
chunk 更新后的 state。

K2 的 grid 是 `(N,H)`。当前 fixed case 是 `(1,96)`，只有 96 个 recurrence CTA；
每个 CTA 内部都背着很长的 512-chunk 依赖链。换指令没有增加 head/sequence 维度的
并行度，反而把单 CTA 的串行关键路径拉长了。

### 4.4 与 instruction-only microbench 的关系

此前 microbench 回答的是“单个指令形状是否划算”；本次回答的是“把它放进真实
FlashKDA K2 是否划算”。

两者不是重复：

```text
microbench：隔离 tile 形状、指令本体、零填充
full K2：再加入真实数据布局、shared/TMEM staging、barrier、chunk 递推
```

microbench 已经没有显示收益，完整 K2 又进一步恶化到 0.1000x，证据方向一致。

### 4.5 Benchmark 的结论边界

当前 `validation/benchmark.py` 每次用 CUDA event 测完整 `fwd()`，并逐 iteration
同步。它对原版与改版是公平的 paired comparison，但计时组织与此前官方
`bench_fwd.py` 的批量 event 方式不同。因此：

- 1.755060 ms 与 17.552482 ms 可以直接相除；
- 不应把这里的 1.755060 ms 与另一轮官方 benchmark 的约 1.03 ms 直接比较并推断
  代码退化，因为运行时间、输入、同步方式和 GPU 状态不同。

---

## 5. SASS 证据怎么解释

对改版动态库 `cuobjdump --dump-sass` 后统计得到：

| kernel 类别 | 指令 | 静态出现次数 |
|---|---|---:|
| K2 `_flash_kda_fwd_recurrence` | UTCHMMA | 70 |
| K1 `_flash_kda_fwd_prepare` | HMMA | 88 |

为什么 K2 是 70，而源码只有五个逻辑调用点？

- K2 有 14 个 state/fixed-varlen 模板实例；
- 每个模板实例有 5 个静态 `tcgen05.mma` 调用点；
- 14x5=70。

注意“静态调用点”不等于“每 chunk 只执行 5 次”。前两个调用点各在 K=128 的 8 次
循环中，因此每 chunk 动态执行次数仍是 19。

为什么 SASS 仍有 HMMA？因为 K1 prepare 没有改。SASS 分类结果显示 HMMA 属于
`_flash_kda_fwd_prepare`，UTCHMMA 属于 `_flash_kda_fwd_recurrence`。这不是 tcgen05
分支失效，而是挑战切面的边界。

正确说法：

> 本实验把 K2 recurrence 的矩阵乘路径换成了 tcgen05；K1 prepare 保持原版
> HMMA。SASS 中 K2 只看到 UTCHMMA，而 K1 仍看到 HMMA。

错误说法：

> 整个 FlashKDA 二进制已经完全没有 HMMA。

---

## 6. 从命令到 GPU 的完整思路链

### Step 0：依据 TASK.md 选择挑战切面

选择：

```text
只换指令，不改 CHUNK，不做大 CHUNK+rescale，不重构跨 CTA 并行
```

控制变量是算法和输入；实验变量是 K2 的 MMA 指令路径。

### Step 1：本机源码目录

```text
c1_flashkda/
├── TASK.md                 # 题目和验收标准
├── FlashKDA/               # 课程提供的原版快照
├── fla_kda_ref/            # naive.py、chunk.py 参照实现
├── experiments/            # 早期 instruction-only microbench
└── FInal/                  # 真实 kernel 挑战
```

### Step 2：rsync 上传

本机 `FInal/` 同步到服务器同名目录。`--exclude results/` 防止重新上传代码时删掉
服务器实验结果。

### Step 3：独立依赖检查

文件：`FInal/prepare_on_server.sh`。

当前交付目录已经物理包含：

```text
FInal/FlashKDA-tcgen05/cutlass/       精简 CUTLASS v4.3.2
FInal/FlashKDA-tcgen05/fla_kda_ref/   正确性参考
FInal/FlashKDA-tcgen05/flash_kda/     原版 Python 包
FInal/FlashKDA-tcgen05/flash_kda_tcgen05/  改版 Python 包
```

`prepare_on_server.sh` 只检查这些文件，不再建立指向 `FlashKDA-upstream` 的
软链接。`setup.py` 同时从本目录编译原版 `flash_kda_C` 与改版
`flash_kda_tcgen05_C`，因此源码、编译和验证均不依赖兄弟目录。

### Step 4：`run_b300.sh` 总控

文件：`FInal/FlashKDA-tcgen05/run_b300.sh`。

顺序是：

1. 保存 hostname、GPU、nvcc、Python、PyTorch；
2. grep 保存源码指令路径；
3. `pip install -e . --no-build-isolation --no-deps` 编译；
4. 运行 correctness；
5. 运行完整 forward benchmark；
6. 找到改版 `.so`；
7. `cuobjdump --dump-sass`；
8. 提取 Function/HMMA/UTCHMMA 对应关系。

脚本使用 `set -euo pipefail`。任何一步失败都会停止，避免前面失败而后面仍打印
“成功”。本次出现 `ALL STEPS COMPLETED`，说明整条链实际走到 SASS 结束。

### Step 5：`setup.py` 决定编译哪个分支

文件：`FlashKDA-tcgen05/setup.py`。

关键设置：

```text
extension name = flash_kda_tcgen05_C
CUDA arch       = sm_103a
macro           = FLASH_KDA_TCGEN05_K2=1
Python package  = flash_kda_tcgen05
```

宏使 `fwd_kernel2.cuh` 的 `#ifdef FLASH_KDA_TCGEN05_K2` 生效，原 SM80 K2 代码在
`#else`，不会进入该改版扩展。

### Step 6：Python 入口

文件：`flash_kda_tcgen05/__init__.py`。

调用链：

```text
flash_kda_tcgen05.fwd(...)
→ 根据 T_total、H、N 分配 uint8 workspace
→ flash_kda_tcgen05_C.fwd(...)
```

原版走：

```text
flash_kda.fwd(...)
→ flash_kda_C.fwd(...)
```

两个 Python 包名和两个 `.so` 名不同，所以可以在一个进程里使用完全相同的输入对拍。

### Step 7：C++ binding 的 fixed/varlen 分支

文件：`csrc/flash_kda.cpp`。

先检查 CUDA、contiguous、BF16/FP32、D=128 等约束。然后：

```text
cu_seqlens is None     → fixed/batched → IsVarlen=false
cu_seqlens is provided → varlen        → IsVarlen=true，且 B 必须为 1
```

fixed 时 `N=B`；varlen 时 `N=cu_seqlens.numel()-1`。

### Step 8：C++ binding 的 state 分支

同一文件根据三个条件分派模板：

```text
HasStateIn
HasStateOut
StateFP32
```

支持的典型路径：

| 输入 state | 输出 state | 模板行为 |
|---|---|---|
| 无 | 无 | shared state 从零开始，不写 final state |
| BF16 | BF16 | TMA 直接载入/写回 BF16 state |
| FP32 | FP32 | TMA 载入 FP32，转 BF16 计算，再转 FP32 写回 |
| 无 | BF16/FP32 | 零初态，但要求输出 final state |
| BF16/FP32 | 无 | 使用 initial state，但不导出 final state |

本次 correctness/benchmark 使用的是 `HasStateIn=true`、`HasStateOut=true`、
`StateFP32=false`。其他模板被编译进 `.so`，但不等于本次都运行验证过。

### Step 9：`fwd_launch.cu` 启动 K1

K1 grid：

```text
(total_tiles, H)
```

每个 CTA 独立负责“一个 chunk、一个 head”的预处理，所以不同 chunk 的 K1 可以并行。
varlen 时先构造 tile prefix，K1 用二分查找把全局 tile 映射回 sequence 和 local tile；
fixed 时用整除直接映射。

### Step 10：K1 prepare 做什么

文件：`csrc/smxx/fwd_kernel1.cuh`。

单个 16-token chunk 内：

1. TMA 载入 q、k、g、beta、dt_bias；
2. q/k L2 normalize；
3. gate activation 和 chunk 内 cumsum；
4. 生成 `q_decayed/k_decayed/k_restored/g_total`；
5. 计算 16x16 的 L 和 Mqk；
6. 用 Neumann series 计算 INV；
7. 把这些中间量写入全局 workspace。

K1 只准备每个 chunk 自身的数据，不读取跨 chunk recurrent state，因此可以为全部 tile
并行展开。

### Step 11：`fwd_launch.cu` 启动 K2

K2 grid：

```text
(N,H)
```

一个 CTA 对应“一条 sequence 的一个 head”。该 CTA 内部：

```text
for t in sequence chunks:
    读取当前 chunk 的 K1 workspace
    使用当前 state 计算 output
    更新 state
```

这是依赖链所在位置，也是为什么不能把 8192 token 简单拆成 512 个完全独立 K2 CTA。

### Step 12：K2 warp specialization

文件：`csrc/smxx/fwd_kernel2.cuh`。

192 threads = 6 warps：

```text
warp 0～3：MMA/compute，共 128 threads
warp 4：TMA load
warp 5：TMA store
```

load pipeline 有 3 stages，store pipeline 有 2 stages。它们让下一个 chunk 的数据搬运
与当前 chunk 的部分计算重叠，但数学上的 state 更新仍严格有序。

### Step 13：K2 初始状态分支

K2 开始时：

```text
HasStateIn && BF16  → TMA 直接载入 state_acc
HasStateIn && FP32  → TMA 到 FP32 buffer，再转换到 BF16 state_acc
无 initial state   → state_acc 清零
```

也就是说，即使 API 接受 FP32 state，当前 kernel 内部 recurrence state 仍转成 BF16
计算；FP32 是输入/输出保存格式，不是把每个 chunk 的内部 state 更新都改成 FP32。

### Step 14：SM80 与 tcgen05 编译分支

```cpp
#ifdef FLASH_KDA_TCGEN05_K2
    // 本实验路径
#else
    // 官方 SM80 mma.sync 路径
#endif
```

二者共享 K1、输入、workspace、TMA 管线、数学 phase 和最终输出接口。这样性能比较尽量
只改变 K2 指令实现。

### Step 15：tcgen05 shared memory 与 TMEM

改版额外分配：

```text
tcgen_a：128x128 bytes，128B-swizzle A descriptor
tcgen_b：128x128 bytes，128B-swizzle B descriptor
temp0：逻辑 16x128 BF16
temp1：逻辑 16x128 BF16
mbarrier
tmem address
```

大约增加 40 KiB dynamic shared storage。TMEM 在 CTA 开始分配一次，在所有 chunk
结束后释放，不是每个 chunk 重新 alloc/dealloc。

### Step 16：每个 tcgen05 tile

每个 tile 的控制流：

1. 128 compute threads把逻辑 operand 写成 128B-swizzle shared layout；
2. `fence.proxy.async.shared::cta`；
3. named barrier 确保 staging 完成；
4. warp 0 elected thread 构造 A/B descriptor；
5. 发出 `tcgen05.mma.cta_group::1.kind::f16`；
6. `tcgen05.commit` 通知 mbarrier；
7. compute threads 等待对应 parity；
8. `tcgen05.ld` 从 TMEM 读 FP32 accumulator；
9. 按原版边界转回 BF16。

K=128 的两个 GEMM通过 accumulate predicate 在 TMEM 中累积 8 个 K=16 tile。

### Step 17：K2 数学 phase

```text
KS = k_decayed @ state^T
QS = q_decayed @ state^T
U0 = (v - KS) * sigmoid(beta)
U  = INV @ U0
out = QS + Mqk @ U
state = state * g_total + k_restored^T @ U
```

phase 间 BF16 materialization 点被保留，因此本次得到与原版逐元素相同的 BF16
output 和 state。

### Step 18：tail 与 final-state 分支

完整 16-token tile 用 TMA store output；尾块 `actual_len<16` 时手动只写真实 token，
避免覆盖下一条 varlen sequence。

final state：

```text
BF16 → TMA 直接写回
FP32 → 全 CTA 同步，BF16 转 FP32，再 TMA 写回
```

### Step 19：correctness

文件：`validation/validate.py`。

同一批输入依次运行原版、改版、naive、chunk，然后比较 output 和 state。因为原版和
改版扩展名不同，不是“先算一个结果再复制给另一个名字”。

### Step 20：benchmark

文件：`validation/benchmark.py`。

使用 `T=8192,H=96,D=128`，10 次 warmup，50 次计时。CUDA event 记录的是 GPU
从 `fwd()` 开始到完整 K1+K2 完成的时间。

### Step 21：SASS

`cuobjdump` 对改版 `.so` 反汇编。源码 grep 说明“写了什么”，SASS 说明“ptxas 最终
生成了什么”。Function 名再说明该指令属于 K1 还是 K2。三层不能互相替代。

---

## 7. 目录中每个部件的职责

```text
FInal/
├── README.md
│   └── 上传、编译、验证操作指南
├── prepare_on_server.sh
│   └── 检查并连接 upstream CUTLASS
├── FInal_结果分析与完整思路链.md
│   └── 本文，最终实验解释
├── results/
│   ├── full_run.log
│   ├── build.log
│   ├── correctness.log
│   ├── benchmark.log
│   ├── env/
│   ├── source/
│   └── sass/
└── FlashKDA-tcgen05/
    ├── setup.py
    │   └── 构建独立 CUDA extension、打开 tcgen05 宏
    ├── run_b300.sh
    │   └── 一键执行完整实验链
    ├── flash_kda_tcgen05/__init__.py
    │   └── 改版 Python API 和 workspace 分配
    ├── flash_kda/__init__.py
    │   └── 从原版树继承的原版包装层；用于同进程 baseline
    ├── csrc/
    │   ├── flash_kda.cpp
    │   │   └── PyTorch binding、参数检查、fixed/varlen/state dispatch
    │   ├── fwd.h
    │   │   └── C++ launch 接口声明
    │   └── smxx/
    │       ├── fwd_launch.cu
    │       │   └── TMA descriptor、K1/K2 模板实例和 kernel launch
    │       ├── fwd_kernel1.cuh
    │       │   └── 每 chunk 独立 prepare
    │       ├── fwd_kernel2.cuh
    │       │   └── 跨 chunk recurrence、SM80/tcgen05 分支
    │       ├── tcgen05.cuh
    │       │   └── descriptor、swizzle、TMEM、mbarrier、inline PTX
    │       └── utils.cuh
    │           └── 公共 layout、pipeline、转换和原版 MMA helper
    ├── validation/
    │   ├── validate.py
    │   │   └── 四实现 fixed/varlen output/state 对拍
    │   └── benchmark.py
    │       └── 原版与改版完整 fwd paired benchmark
    ├── tests/
    │   └── 上游官方测试快照；不是本挑战改版的主验证入口
    ├── benchmarks/
    │   └── 上游官方 benchmark/NCU 快照
    └── docs/
        └── FlashKDA 上游设计说明
```

---

## 8. 答辩时最短的故事线

可以按下面顺序讲：

1. **问题**：B300 支持 tcgen05，为什么 FlashKDA 仍用 SM80 MMA？
2. **形状假设**：CHUNK=16 与 tcgen05 的物理 M=128 不匹配，直接替换可能浪费。
3. **局部证据**：instruction-only microbench 没有收益。
4. **真实改动**：在 FlashKDA K2 recurrence 内保留算法和递推，仅替换矩阵乘路径。
5. **正确性**：fixed/varlen，output/state 对原版逐元素相等；对 naive/chunk 的误差
   与原版一致。
6. **机器码**：K2 SASS 是 UTCHMMA；K1 保持 HMMA。
7. **性能**：1.755060 ms → 17.552482 ms，慢 10.001 倍。
8. **原因**：每 chunk 19 个 tcgen05，18 个 M=16 tile 要补到 M=128；只有约
   17.1% 物理计算有用，且 staging/barrier/TMEM 破坏了原版寄存器复用与 dual GEMM。
9. **结论**：不应发布这种“CHUNK=16、直接替换 K2”的 SM100 专版。未来只有结合
   大 CHUNK+数值 rescale 或跨 head/CTA 并行重构，才值得重新评估。

---

## 9. 当前还值得补的证据（按优先级）

如果时间很紧，当前挑战已经满足 `TASK.md`，不必立刻做大 CHUNK。为了让最终交付更
难被追问，优先补这些低成本证据：

1. 保存原版 `flash_kda_C.__file__` 和改版路径，彻底排除包名碰撞疑问；
2. 保存 FLA `chunk_kda` 模块路径和 commit/version；
3. 把原版 test、官方 benchmark、原版 SASS、NCU 和 microbench 日志归到报告目录；
4. correctness 再补 2～3 个 seed，并补一组较长 T 的“改版 vs 原版”对拍；
5. 问题 5 单独跑 BF16 state vs FP32 state 的长序列、多 gate 压力测试；
6. 若要解释 10 倍慢的硬件细分，再对改版 K2 跑一次 NCU，而不是仅凭推理。

这些是“提高证据强度”，不是否定当前挑战结果。
