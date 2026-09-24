#!/usr/bin/env python3
"""Correctness checks for the in-kernel tcgen05 K2 replacement.

Three-way evidence:
  1. experimental FlashKDA vs unmodified FlashKDA (fixed + varlen),
  2. both kernels vs fla_kda_ref/naive.py (small semantic oracle),
  3. both kernels vs FLA chunk_kda (Triton implementation).
"""

import argparse
import importlib.util
import math
import os
from pathlib import Path

os.environ.setdefault("FLA_FLASH_KDA", "0")

import torch
import torch.nn.functional as F
import flash_kda
import flash_kda_tcgen05


def stats(actual, expected):
    a = actual.float()
    e = expected.float()
    diff = a - e
    rmse = diff.square().mean().sqrt().item()
    base = e.square().mean().sqrt().item()
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rel_rmse": rmse / (base + 1e-8),
    }


def show(label, actual, expected):
    s = stats(actual, expected)
    print(
        f"{label:42s} max_abs={s['max_abs']:.6g} "
        f"mean_abs={s['mean_abs']:.6g} rel_rmse={s['rel_rmse']:.6g}"
    )
    return s


def make_inputs(total_t, heads, sequences, seed):
    torch.manual_seed(seed)
    device = "cuda"
    d = 128
    q = torch.randn(1, total_t, heads, d, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(1, total_t, heads, device=device, dtype=torch.bfloat16)
    a_log = torch.rand(heads, device=device, dtype=torch.float32)
    dt_bias = torch.rand(heads, d, device=device, dtype=torch.float32)
    initial = torch.randn(
        sequences, heads, d, d, device=device, dtype=torch.bfloat16
    ) * 0.1
    return q, k, v, g, beta, a_log, dt_bias, initial


def run_flash(module, tensors, cu_seqlens=None):
    q, k, v, g, beta, a_log, dt_bias, initial = tensors
    out = torch.empty_like(q)
    final = torch.empty_like(initial)
    module.fwd(
        q, k, v, g, beta, 1.0 / math.sqrt(128), out,
        A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
        initial_state=initial, final_state=final,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    return out, final


def load_naive(ref_root):
    path = Path(ref_root).expanduser().resolve() / "naive.py"
    spec = importlib.util.spec_from_file_location("assignment_fla_kda_naive", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_recurrent_kda


def activate_for_naive(q, k, g, beta, a_log, dt_bias):
    # Same mathematical activations fused by FlashKDA.  Minor differences from
    # the kernel's approximation/reduction order are expected and measured.
    qn = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(torch.bfloat16)
    kn = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(torch.bfloat16)
    gate = -5.0 * torch.sigmoid(
        torch.exp(a_log)[None, None, :, None] *
        (g.float() + dt_bias[None, None, :, :])
    )
    return qn, kn, gate, torch.sigmoid(beta.float())


def run_naive(naive, tensors, seq_lens):
    q, k, v, g, beta, a_log, dt_bias, initial = tensors
    qn, kn, gate, beta_activated = activate_for_naive(
        q, k, g, beta, a_log, dt_bias
    )
    outs, states = [], []
    start = 0
    for seq, length in enumerate(seq_lens):
        stop = start + length
        out, state_kv = naive(
            qn[:, start:stop], kn[:, start:stop], v[:, start:stop],
            gate[:, start:stop], beta_activated[:, start:stop],
            scale=1.0 / math.sqrt(128),
            initial_state=initial[seq:seq + 1].float().transpose(-1, -2),
            output_final_state=True,
        )
        outs.append(out)
        states.append(state_kv.transpose(-1, -2))
        start = stop
    return torch.cat(outs, dim=1), torch.cat(states, dim=0)


def run_chunk(tensors, cu_seqlens=None):
    from fla.ops.kda import chunk_kda

    q, k, v, g, beta, a_log, dt_bias, initial = tensors
    out, state = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=1.0 / math.sqrt(128),
        initial_state=initial.float(), output_final_state=True,
        use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, safe_gate=True,
        A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
        state_v_first=True, cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    return out, state


def check_case(name, seq_lens, heads, seed, naive):
    total = sum(seq_lens)
    tensors = make_inputs(total, heads, len(seq_lens), seed)
    cu = None
    if len(seq_lens) > 1:
        prefix = [0]
        for length in seq_lens:
            prefix.append(prefix[-1] + length)
        cu = torch.tensor(prefix, device="cuda", dtype=torch.long)

    base_o, base_s = run_flash(flash_kda, tensors, cu)
    tc_o, tc_s = run_flash(flash_kda_tcgen05, tensors, cu)
    naive_o, naive_s = run_naive(naive, tensors, seq_lens)
    chunk_o, chunk_s = run_chunk(tensors, cu)

    print(f"\n=== {name}: seq_lens={seq_lens}, H={heads} ===")
    direct_o = show("tcgen05 vs FlashKDA output", tc_o, base_o)
    direct_s = show("tcgen05 vs FlashKDA final_state", tc_s, base_s)
    show("FlashKDA vs fla_kda_ref/naive output", base_o, naive_o)
    show("tcgen05 vs fla_kda_ref/naive output", tc_o, naive_o)
    show("FlashKDA vs fla_kda_ref/naive state", base_s, naive_s)
    show("tcgen05 vs fla_kda_ref/naive state", tc_s, naive_s)
    show("FlashKDA vs FLA chunk.py output", base_o, chunk_o)
    show("tcgen05 vs FLA chunk.py output", tc_o, chunk_o)
    show("FlashKDA vs FLA chunk.py state", base_s, chunk_s)
    show("tcgen05 vs FLA chunk.py state", tc_s, chunk_s)

    assert torch.isfinite(tc_o).all() and torch.isfinite(tc_s).all()
    # Instruction order differs, so exact bf16 equality is not required.  This
    # threshold catches layout/transpose/synchronization bugs by a wide margin.
    assert direct_o["rel_rmse"] < 0.08, direct_o
    assert direct_s["rel_rmse"] < 0.08, direct_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-root",
        default=str(Path(__file__).resolve().parents[1] / "fla_kda_ref"),
        help="path to assignment02/team/c1_flashkda/fla_kda_ref",
    )
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    naive = load_naive(args.ref_root)
    check_case("fixed", [64], args.heads, args.seed, naive)
    check_case("varlen", [17, 31, 16], args.heads, args.seed + 1, naive)
    print("\noverall correctness: PASS")


if __name__ == "__main__":
    main()
