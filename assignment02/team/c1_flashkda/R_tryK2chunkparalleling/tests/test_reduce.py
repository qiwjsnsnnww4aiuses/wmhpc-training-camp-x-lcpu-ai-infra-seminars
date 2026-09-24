import argparse
import math

import torch
import flash_kda_r


def error_stats(actual, expected):
    diff = actual.float() - expected.float()
    denom = expected.float().square().mean().sqrt().item() + 1e-8
    return (
        diff.abs().max().item(),
        diff.abs().mean().item(),
        diff.square().mean().sqrt().item() / denom,
    )


def show(label, actual, expected):
    max_abs, mean_abs, rel_rmse = error_stats(actual, expected)
    print(
        f"{label:34s} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} rel_rmse={rel_rmse:.6g}"
    )
    return rel_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rel-rmse", type=float, default=0.15)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
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

    out_serial = torch.empty_like(q)
    state_serial = torch.empty(
        1, args.H, 128, 128, device="cuda", dtype=torch.bfloat16
    )
    flash_kda_r.serial_fwd(
        q, k, v, g, beta, scale, out_serial,
        a_log, dt_bias, -5.0, final_state=state_serial,
    )

    out_reduce = torch.empty_like(q)
    state_reduce = torch.empty_like(state_serial)
    debug = flash_kda_r.fwd(
        q, k, v, g, beta, scale, out_reduce,
        a_log, dt_bias, -5.0, final_state=state_reduce,
        segment_chunks=args.G, return_debug=True,
    )
    torch.cuda.synchronize()

    print(
        f"shape=[1,{args.T},{args.H},128] G={args.G} "
        f"segments={debug.num_segments} levels={debug.tree_levels}"
    )
    out_error = show("R reduce vs R serial output", out_reduce, out_serial)
    state_error = show("R reduce vs R serial state", state_reduce, state_serial)
    assert torch.isfinite(out_reduce).all()
    assert torch.isfinite(state_reduce).all()
    assert out_error < args.max_rel_rmse
    assert state_error < args.max_rel_rmse
    print("overall correctness: PASS")


if __name__ == "__main__":
    main()
