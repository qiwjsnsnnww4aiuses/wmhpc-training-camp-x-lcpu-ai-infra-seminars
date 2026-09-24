# C1 FlashKDA 工作区索引

本目录包含 C1 任务原始代码、参考实现、B300 复现证据、SM100 tcgen05
挑战实现、报告和汇报材料。课程题目以 [`TASK.md`](TASK.md) 为准。

## 最常用入口

| 目标 | 文件或目录 |
|---|---|
| 查看课程任务 | [`TASK.md`](TASK.md) |
| 阅读最终汇报报告 | [`C1_FlashKDA_最终汇报报告.md`](C1_FlashKDA_最终汇报报告.md) |
| 阅读六个讨论问题与挑战结论 | [`C1_Task讨论答案与挑战报告.md`](C1_Task讨论答案与挑战报告.md) |
| 查看汇报 PPT | [`presentation/C1_FlashKDA_汇报.pdf`](presentation/C1_FlashKDA_汇报.pdf) |
| 查看 tcgen05 最终挑战 | [`FInal/README.md`](FInal/README.md) |
| 查看 B300 原始证据 | [`remote_b300_results/`](remote_b300_results/) |

## 代码和结果目录

```text
c1_flashkda/
├── TASK.md                 # 课程原题
├── FlashKDA/               # 课程提供的官方 FlashKDA 快照
├── fla_kda_ref/            # naive PyTorch 与 Triton 参考实现
├── experiments/            # 指令级 SM80/tcgen05 microbenchmark 源码
├── remote_b300_results/    # 从 B300 导回的 microbenchmark 原始证据
├── FInal/                  # tcgen05“只换指令”挑战实现与验证结果
├── presentation/           # LaTeX Beamer 源码与最终 PDF
└── *.md                    # 学习笔记、操作指南、方案与报告
```

### 基线与参考

- `FlashKDA/`：官方 kernel 基线。其 CUTLASS 依赖声明为 Git 子模块，本地快照
  没有初始化 `cutlass/` 时不能单独编译。
- `fla_kda_ref/`：正确性对拍参考；`naive.py` 是纯 PyTorch 参考，
  `chunk.py` 及相关文件是 Triton 路径。

### 实验与挑战

- `experiments/microbench/`：独立比较 SM80 MMA 和 SM100 tcgen05 的小实验。
- `remote_b300_results/challenge/`：microbenchmark 可执行文件、SASS 和 B300
  终端日志快照，作为报告证据保留。
- `FInal/FlashKDA-tcgen05/`：基于 FlashKDA kernel 的 tcgen05 挑战源码。
- `FInal/results/`：该挑战的环境、编译、正确性、性能和 SASS 结果。

## 文档分工

| 文档 | 用途 |
|---|---|
| `C1_FlashKDA_任务拆解与完整方案.md` | 从零理解任务和整体路线 |
| `C1_第一阶段_复现操作指南.md` | B300 环境、编译、测试、benchmark、NCU/SASS 操作 |
| `C1_FlashKDA_K1_逐变量详解.md` | K1 数据流与中间变量 |
| `C1_Challenge_tcgen05_傻瓜式完整讲解.md` | 指令替换挑战的逐步解释 |
| `C1_Task讨论答案与挑战报告.md` | TASK 六问的结论与证据 |
| `C1_FlashKDA_最终汇报报告.md` | 汇报用完整报告 |
| `K2_SM80_仿射因子化改造方案.md` | K2 仿射递推推导 |
| `K2_SM80_chunk间并行_分段扫描方案.md` | chunk 间分段并行初步方案 |
| `K2_SM80_超前进位Reduce最终方案.md` | carry-lookahead/reduce 方案总结 |

## 当前状态说明

- 官方复现、B300 benchmark、NCU/SASS 证据和 tcgen05 指令挑战均已有本地材料。
- `FInal/` 是 tcgen05 挑战目录，保留原拼写以避免破坏已有命令和文档路径。
- 截至 2026-09-14，本机目录中没有 `R/` 分段扫描实验源码或其新下载结果；
  如仍需保留，应从 B300 的 `team/c1_flashkda/R/` 单独同步回来。

## 保留与清理规则

- 保留：源码、Markdown、TeX、PDF、测试数据、benchmark、NCU、SASS 和终端日志。
- 可清理：`__pycache__/`、`*.pyc`、LaTeX 的 `aux/log/nav/out/snm/toc` 中间文件。
- 不应混入源码快照：虚拟环境、临时 build 目录和重复的完整 CUTLASS 测试树。
