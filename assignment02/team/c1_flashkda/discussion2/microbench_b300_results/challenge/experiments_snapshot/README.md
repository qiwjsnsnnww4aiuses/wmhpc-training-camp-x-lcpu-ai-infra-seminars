# C1 Challenge：不改算法，只替换 Tensor Core 指令

这个目录实现 `TASK.md` 挑战部分中的第一条路线：数学运算不变，把
SM80 世代的 `mma.sync` 路径换成 SM100 的 `tcgen05.mma` 路径。

## 实验回答什么

两个实现都计算同一个 BF16 输入、FP32 累加的矩阵乘：

$$
D_{m,n}=\sum_{k=0}^{15} A_{m,k}B_{n,k}.
$$

程序包含两组 shape：

| Case | useful shape | `tcgen05` physical shape | 用途 |
|---|---|---|---|
| thin | `m16n128k16` | `m128n128k16` | 代表 FlashKDA 中以 `CHUNK=16` 为 M 的热点；补零 8 倍 |
| matched | `m128n128k16` | `m128n128k16` | 代表 K2 的 `k_restored.T @ U`；无补零 |

thin case 中，A 的 16–127 行被置为精确零，输出只保留前 16 行。因此
数学结果没有变化，但 `tcgen05` 的 shape 浪费被真实计入时间。

程序会：

1. 用小整数 BF16 输入分别运行两条路径；
2. 与 CPU FP32 参考逐元素严格相等对拍；
3. warmup 后用 CUDA event 计时；
4. 同时打印 useful TFLOP/s 和包含 padding 的 issued TFLOP/s；
5. 打印 `tcgen05` 相对 `mma.sync` 的延迟加速比。

## 在 B300 上运行

```bash
cd ~/lcpu2026/assignment02/team/c1_flashkda/experiments
bash run_b300.sh
```

参数依次是 jobs、计时迭代数、warmup 次数、随机种子：

```bash
bash run_b300.sh 512 200 30 42
```

默认使用 `ARCH=103a`。如果集群课程环境要求 `100a`，执行：

```bash
ARCH=100a bash run_b300.sh 512 200 30 42
```

脚本使用 `make -B`，因此切换 `ARCH` 时不会误用上一种架构编译出的 cubin。
stdout 会同时保存到 `results/instruction_only_<arch>_<时间>.txt`，可以直接
复制进报告的结果表。

如果不确定 GPU 和工具链：

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvcc --version
```

## 检查 SASS

`run_b300.sh` 已生成：

```text
bin/instruction_only_mma.sass
```

也可手工执行：

```bash
make ARCH=103a sass
grep -E 'HMMA|TCGEN05|tcgen05' bin/instruction_only_mma.sass | head -n 80
```

要证明的不是源文件里出现了指令名，而是最终 cubin 中两条路径分别存在
SM80 MMA 与 TCGEN05 指令。

## 用 NCU 测量

先用 `--set full` 捕获每个模板实例的一次调用：

```bash
ncu --set full \
  --kernel-name 'regex:.*(sm80_mma_kernel|tcgen05_mma_kernel).*' \
  --launch-count 8 \
  --export c1_instruction_only_full \
  ./bin/instruction_only_mma 1 1 0 42
```

`jobs=1,iters=1,warmup=0` 时，每个 shape 会各执行一次 correctness SM80、
一次 correctness tcgen05、一次计时 SM80、一次计时 tcgen05，共 8 次 launch。
如果当前 NCU 的 kernel regex 语法不接受括号，去掉 `--kernel-name`，用这个
最小参数运行一次；NCU 报告中会显示实际的四个模板 kernel 名，再分别过滤。

```bash
ncu --set full --launch-count 8 \
  --export c1_instruction_only_unfiltered \
  ./bin/instruction_only_mma 1 1 0 42
```

然后分别指定完整名称。报告至少记录：

- kernel duration；
- SM/Tensor pipe throughput；
- achieved occupancy、active warps、eligible warps；
- barrier、short/long scoreboard stall；
- shared-memory throughput；
- 每 CTA 的 register/shared-memory 使用。

## 如何读输出

- `correctness` 必须两条路径、两个 shape 都为 `PASS`。
- `speedup tcgen05/SM80` 大于 1 才是 tcgen05 更快。
- thin case 的 `issued/useful FLOP` 固定为 8 倍；matched case 为 1 倍。
- thin case 即使 issued TFLOP/s 很高，若 useful TFLOP/s 或延迟不如
  baseline，结论仍是“仅换指令无收益”。
- matched case 若快，只证明状态更新这个切面有潜力；完整 K2 的最大理论
  加速还受该 phase 占比和 TMEM/状态融合成本限制。

## 公平性与局限

- 两条路径使用相同 input/output dtype、数学 shape、jobs 和计时方法。
- 计时包含 global→shared staging、同步、Tensor Core 和结果写回；tcgen05
  还真实包含 TMEM alloc、commit/wait、load、dealloc。
- 这是题目允许的“任一切面”挑战，不是完整 FlashKDA SM100 kernel。
- slice 的 CPU 精确对拍验证指令替换没有改数学；完整算子仍需运行
  FlashKDA 自带测试并对 `fla_kda_ref` 验证。
- 当前开发机没有 B300，报告中的实测格必须在 `dev-slurm` 填写。
