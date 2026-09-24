"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T

# 用于按形状缓存编译后的 kernel
_kernel_cache = {}

def get_next_power_of_2(n: int) -> int:
    """计算大于等于 n 的最小的 2 的幂次"""
    return 1 if n == 0 else 2 ** (n - 1).bit_length()

def make_softmax_kernel(M: int, N: int):
    # fragment 的宽度取不小于 N 的 2 的幂
    BLOCK_N = get_next_power_of_2(N)
    
    @T.prim_func
    def softmax_kernel(
        x: T.Buffer((M, N), "float32"),
        y: T.Buffer((M, N), "float32"),
    ):
        # 一个 block 处理一行
        with T.Kernel(M, threads=128) as (bx,):
            # 分配片上 fragment 空间，必须为归约结果显式分配空间
            x_frag = T.alloc_fragment((BLOCK_N,), "float32")
            exp_frag = T.alloc_fragment((BLOCK_N,), "float32")
            max_val = T.alloc_fragment((1,), "float32")
            sum_val = T.alloc_fragment((1,), "float32")
            
            # 1. 读入数据，越界位置补 -inf
            for i in T.Parallel(BLOCK_N):
                x_frag[i] = T.if_then_else(i < N, x[bx, i], -T.infinity("float32"))
                
            # 2. 全 Fragment 归约求最大值
            # 修正点：将 max_val 作为 out 参数传入
            T.reduce_max(x_frag, max_val, dim=0)
            
            # 3. 减去最大值后取 exp
            for i in T.Parallel(BLOCK_N):
                # 越界部分补 0
                exp_frag[i] = T.if_then_else(i < N, T.exp(x_frag[i] - max_val[0]), 0.0)
                
            # 4. 全 Fragment 归约求和
            # 修正点：将 sum_val 作为 out 参数传入
            T.reduce_sum(exp_frag, sum_val, dim=0)
                
            # 5. 归一化并写回全局内存
            for i in T.Parallel(BLOCK_N):
                if i < N:
                    y[bx, i] = exp_frag[i] / sum_val[0]
                    
    return softmax_kernel

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
    对每一行独立做 softmax。
    """
    M, N = x.shape
    key = (M, N)
    
    # 按照输入矩阵的形状生成并编译对应的 kernel
    if key not in _kernel_cache:
        kernel_func = make_softmax_kernel(M, N)
        _kernel_cache[key] = tilelang.compile(kernel_func, target="cuda")
        
    y = torch.empty_like(x)
    _kernel_cache[key](x, y)
    
    return y
