import argparse
import math
import statistics

import torch
import flash_kda_r


def measure(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.mean(samples), min(samples), max(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=96)
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shape = (1, args.T, args.H, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a_log = torch.rand(args.H, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(args.H, 128, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(128)
    out_serial, out_reduce = torch.empty_like(q), torch.empty_like(q)

    def serial():
        flash_kda_r.serial_fwd(
            q, k, v, g, beta, scale, out_serial, a_log, dt_bias, -5.0
        )

    def reduce():
        flash_kda_r.fwd(
            q, k, v, g, beta, scale, out_reduce, a_log, dt_bias, -5.0,
            segment_chunks=args.G,
        )

    serial_ms = measure(serial, args.warmup, args.iters)
    reduce_ms = measure(reduce, args.warmup, args.iters)
    print(f"shape=[1,{args.T},{args.H},128] G={args.G}")
    print(
        f"R serial : mean={serial_ms[0]:.6f} min={serial_ms[1]:.6f} "
        f"max={serial_ms[2]:.6f} ms"
    )
    print(
        f"R reduce : mean={reduce_ms[0]:.6f} min={reduce_ms[1]:.6f} "
        f"max={reduce_ms[2]:.6f} ms"
    )
    print(f"speedup serial/reduce: {serial_ms[0] / reduce_ms[0]:.4f}x")


if __name__ == "__main__":
    main()
