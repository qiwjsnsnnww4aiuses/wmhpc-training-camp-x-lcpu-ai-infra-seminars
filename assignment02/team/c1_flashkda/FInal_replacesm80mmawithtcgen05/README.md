# FInal：在 FlashKDA K2 kernel 内把 SM80 MMA 路径替换为 tcgen05

本次 B300 实测结论、`TASK.md` 验收核对和从 Python 到 GPU 的完整分支链，见
[`FInal_结果分析与完整思路链.md`](./FInal_结果分析与完整思路链.md)。

## 1. 这份代码现在是什么

这不是独立矩阵乘 microbenchmark，也不是用 Python 重写 KDA。

`FlashKDA-tcgen05/` 是以题目给的 FlashKDA 快照为底稿修改的完整源码树。主要改动位于：

- `csrc/smxx/fwd_kernel2.cuh`：FlashKDA 的 K2 recurrence kernel 本体；
- `csrc/smxx/tcgen05.cuh`：SM100 `tcgen05` 的 PTX、描述符、TMEM 和 mbarrier 辅助代码；
- `setup.py`：打开 `FLASH_KDA_TCGEN05_K2`，并把实验扩展命名为 `flash_kda_tcgen05_C`；
- `flash_kda_tcgen05/__init__.py`：独立 Python 包名，使原版与改版能在同一进程中直接对拍；
- `validation/validate.py`：正确性对拍；
- `validation/benchmark.py`：端到端性能对 FlashKDA 本体；
- `run_b300.sh`：把构建、正确性、性能、SASS 证据一次跑完。

原版 K1 prepare kernel 不变。改版接管 K2 中原先由同一个
`SM80_16x8x16_F32BF16BF16F32_TN` atom 完成的所有矩阵乘：

1. `k_decayed @ state^T`；
2. `q_decayed @ state^T`；
3. `INV @ (v - k@s) * beta`；
4. `Mqk @ U`；
5. `k_restored^T @ U`（状态增量）。

KDA 的 chunk 大小仍为 16，状态仍逐 chunk 递推，门控、残差和 bf16
落盘点也不改。因此这是题目所说的“只换指令不动算法”。

## 2. 为什么代码看起来比独立 GEMM 长很多

SM80 `mma.sync.m16n8k16` 正好能服务 FlashKDA 的 16 行小矩阵。
tcgen05 此处选用的物理 tile 是 `m128n128k16`。所以：

- 逻辑 M=16 的四类矩阵乘必须把 112 行补零，物理计算量放大 8 倍；
- `k@s`、`q@s` 的 K=128 还要拆成 8 个 K=16 tile，在 TMEM 中累加；
- 只有状态增量 `128x16 @ 16x128` 与物理 tile 天然匹配；
- 每次都要准备 128B-swizzle 的 A/B shared-memory descriptor；
- 结果先留在 TMEM，再由 4 个 compute warp 用 `tcgen05.ld` 读回。

这正是挑战要量化的代价，而不是为了跑赢而偷偷更换算法或形状。

## 3. 数据流

```text
原版 K1（不变，TMA 写 workspace）
          |
          v
改版 K2：TMA 把本 chunk 数据搬进 shared memory
          |
          +-- tcgen05: k @ state ----> KS(bf16)
          +-- tcgen05: q @ state ----> QS(bf16)
          |
          +-- U0 = (v - KS) * sigmoid(beta)
          +-- tcgen05: INV @ U0 -----> U(bf16)
          +-- tcgen05: Mqk @ U ------> QS += result，写 output
          +-- tcgen05: k_restored^T @ U
                                      |
                                      v
                     state = state * g_total + delta（bf16）
                                      |
                             下一 chunk（严格依赖）
```

## 4. 正确性比较了什么

`validation/validate.py` 每个 case 使用完全相同的随机输入，同时运行：

- `flash_kda`：未经修改的原版；
- `flash_kda_tcgen05`：本目录的改版；
- `fla_kda_ref/naive.py::naive_recurrent_kda`：小规模纯 PyTorch 语义参考；
- `fla.ops.kda.chunk_kda`：题目指定的 Triton 参照。

它同时打印 output 和 final state 的 `max_abs`、`mean_abs`、`rel_rmse`，并测试：

- fixed：一条长度 64 的序列；
- varlen：长度 `[17, 31, 16]` 的三条拼接序列，包含非 16 倍数尾块。

tcgen05 与 SM80 的 FP32 累加顺序不同，因此不把逐 bit 相等当作必要条件；脚本用
`rel_rmse < 0.08` 先拦截转置、布局、同步等大错误。最终报告必须如实给出全部三方
误差，不能只写 `PASS`。

## 5. 性能比较了什么

`validation/benchmark.py` 测的是完整 `fwd()`：K1 + K2 + TMA + 状态读写，输入是题目
指定的 `T=8192, H=96, D=128`。同一进程、同一组输入、同一张 B300 上分别计时原版
和改版，输出：

- 平均、最小、最大 kernel 调用延迟；
- `原版时间 / 改版时间`。大于 1 才是改版加速，小于 1 是变慢。

负收益并不等于挑战失败。若正确性通过且 SASS 证明真正执行了 tcgen05，负收益恰好
支持“CHUNK=16 与 tcgen05 最小 tile 不匹配，因此官方继续使用 SM80 MMA 是合理的”。

## 6. 服务器操作（完整命令）

本机 PowerShell：

```powershell
rsync -av --delete --exclude results/ \\wsl.localhost\Ubuntu\home\wpy\documents\lcpu2026\assignment02\team\c1_flashkda\FInal\ b300-vscode:~/lcpu2026/assignment02/team/c1_flashkda/FInal/
```

如果 Windows 版 rsync 不接受 UNC 路径，就在 WSL 终端执行：

```bash
rsync -av --delete --exclude results/ \
  /home/wpy/documents/lcpu2026/assignment02/team/c1_flashkda/FInal/ \
  b300-vscode:~/lcpu2026/assignment02/team/c1_flashkda/FInal/
```

登录并申请 GPU（编译加测试建议申请 30 分钟）：

```bash
ssh b300-vscode
srun -G 1 --time 00:30:00 --pty bash
# 激活任意已经安装 CUDA PyTorch、FLA/Triton 的 Python 环境
# source /path/to/your/venv/bin/activate
cd ~/lcpu2026/assignment02/team/c1_flashkda/FInal
bash prepare_on_server.sh
cd FlashKDA-tcgen05
bash run_b300.sh 2>&1 | tee ../results/full_run.log
```

`FInal/FlashKDA-tcgen05/` 现在自带精简 CUTLASS v4.3.2、原版/改版源码以及
`fla_kda_ref`。`setup.py` 会从同一目录同时编译 `flash_kda_C` 和
`flash_kda_tcgen05_C`，不读取兄弟目录中的源码、动态库或 CUTLASS。
`prepare_on_server.sh` 只检查独立目录是否完整，不再创建软链接。

## 7. 结果目录

```text
FInal/results/
├── build.log                 # nvcc/ptxas 编译记录、寄存器和 spill 信息
├── correctness.log           # fixed/varlen，三方 output/state 误差
├── benchmark.log             # 原版 vs 改版的完整 fwd 延迟
├── full_run.log              # 全流程总日志
├── env/                      # 主机、GPU、CUDA、Python、PyTorch 环境
├── source/instruction_paths.txt # 源码级 SM80/tcgen05 路径
└── sass/
    ├── extension_path.txt    # 改版 .so 的绝对路径
    ├── full.sass             # 改版扩展的完整反汇编
    └── key_instructions.txt  # Function、UTCHMMA、HMMA 行
```

看到 `UTCHMMA` 只能证明二进制包含 tcgen05。还要把它所在的 `Function` 对应到
`_flash_kda_fwd_recurrence`，再结合改版包实际通过正确性和计时，证据链才完整。

## 8. 当前状态与必须遵守的结论边界

本机没有 B300，当前只完成源码构造和静态检查；尚未宣称编译通过、正确性通过或性能
结论。第一次服务器运行很可能暴露 PTX 约束、shared-memory 资源或同步问题。把
`results/full_run.log` 原样发回，才能依据真实错误继续修正，不能预先编造数据。
