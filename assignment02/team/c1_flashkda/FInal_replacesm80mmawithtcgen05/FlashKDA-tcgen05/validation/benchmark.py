#!/usr/bin/env python3
"""End-to-end original FlashKDA vs in-kernel tcgen05 variant benchmark."""

import argparse
import math
import statistics

import torch
import flash_kda
import flash_kda_tcgen05


def measure(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return statistics.mean(values), min(values), max(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8192)
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    shape = (1, args.T, args.H, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a_log = torch.rand(args.H, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(args.H, 128, device="cuda", dtype=torch.float32)
    initial = torch.randn(
        1, args.H, 128, 128, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    out_base = torch.empty_like(q)
    out_tc = torch.empty_like(q)
    state_base = torch.empty_like(initial)
    state_tc = torch.empty_like(initial)
    scale = 1.0 / math.sqrt(128)

    def base():
        flash_kda.fwd(
            q, k, v, g, beta, scale, out_base, a_log, dt_bias, -5.0,
            initial_state=initial, final_state=state_base,
        )

    def tcgen():
        flash_kda_tcgen05.fwd(
            q, k, v, g, beta, scale, out_tc, a_log, dt_bias, -5.0,
            initial_state=initial, final_state=state_tc,
        )

    base_ms = measure(base, args.warmup, args.iters)
    tc_ms = measure(tcgen, args.warmup, args.iters)
    print(
        f"shape=[{args.T},{args.H},128] warmup={args.warmup} "
        f"iters={args.iters} seed={args.seed}"
    )
    print(f"FlashKDA SM80 : mean={base_ms[0]:.6f} min={base_ms[1]:.6f} max={base_ms[2]:.6f} ms")
    print(f"FlashKDA tcgen05: mean={tc_ms[0]:.6f} min={tc_ms[1]:.6f} max={tc_ms[2]:.6f} ms")
    print(f"speedup SM80/tcgen05: {base_ms[0] / tc_ms[0]:.4f}x")


if __name__ == "__main__":
    main()
