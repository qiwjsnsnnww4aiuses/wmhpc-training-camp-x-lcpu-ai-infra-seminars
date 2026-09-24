"""Standalone FlashKDA fork with a K2 carry-lookahead execution path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import flash_kda_r_C as _C

CHUNK = 16
D = 128


@dataclass
class DebugInfo:
    segment_tokens: int
    num_segments: int
    tree_levels: int
    segment_a: torch.Tensor | None = None
    segment_b: torch.Tensor | None = None
    segment_starts: torch.Tensor | None = None


def serial_fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
               initial_state=None, final_state=None, cu_seqlens=None):
    """The unmodified FlashKDA path compiled inside this same R extension."""
    total_t = q.shape[0] * q.shape[1]
    heads = q.shape[2]
    nseq = cu_seqlens.numel() - 1 if cu_seqlens is not None else q.shape[0]
    workspace = torch.empty(
        _C.get_workspace_size(total_t, heads, nseq),
        dtype=torch.uint8, device=q.device,
    )
    _C.fwd(
        q, k, v, g, beta, float(scale), out, workspace,
        A_log, dt_bias, float(lower_bound),
        initial_state=initial_state, final_state=final_state,
        cu_seqlens=cu_seqlens,
    )


def _power_of_two(value, name):
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def _scan_bmm(a, b, add=None):
    shape = a.shape
    if shape != b.shape or shape[-2:] != (D, D):
        raise ValueError(f"bad scan GEMM shapes: {a.shape=} {b.shape=}")
    flat_a = a.contiguous().view(-1, D, D)
    flat_b = b.contiguous().view(-1, D, D)
    flat_add = None if add is None else add.contiguous().view(-1, D, D)
    return _C.reduce_scan_bmm(flat_a, flat_b, flat_add).view(shape)


def carry_lookahead_starts(segment_a, segment_b, initial_state=None):
    """Work-efficient affine-map up-sweep and carry down-sweep."""
    if segment_a.shape != segment_b.shape:
        raise ValueError("segment_a and segment_b must have identical shapes")
    if segment_a.ndim != 4 or segment_a.shape[-2:] != (D, D):
        raise ValueError("segment maps must be [segments,H,128,128]")
    nseg, heads = segment_a.shape[:2]
    _power_of_two(nseg, "num_segments")

    levels = [(segment_a.contiguous(), segment_b.contiguous())]
    while levels[-1][0].shape[0] > 1:
        child_a, child_b = levels[-1]
        left_a, right_a = child_a[0::2].contiguous(), child_a[1::2].contiguous()
        left_b, right_b = child_b[0::2].contiguous(), child_b[1::2].contiguous()
        levels.append((
            _scan_bmm(right_a, left_a),
            _scan_bmm(right_a, left_b, right_b),
        ))

    if initial_state is None:
        parent_starts = torch.zeros(
            (1, heads, D, D), device=segment_a.device, dtype=torch.bfloat16
        )
    else:
        if initial_state.shape != (1, heads, D, D):
            raise ValueError("initial_state must be [1,H,128,128]")
        parent_starts = initial_state.to(torch.bfloat16).contiguous()

    for depth in range(len(levels) - 2, -1, -1):
        left_a = levels[depth][0][0::2].contiguous()
        left_b = levels[depth][1][0::2].contiguous()
        right_starts = _scan_bmm(left_a, parent_starts, left_b)
        child_starts = torch.empty(
            (parent_starts.shape[0] * 2, heads, D, D),
            device=segment_a.device, dtype=torch.bfloat16,
        )
        child_starts[0::2].copy_(parent_starts)
        child_starts[1::2].copy_(right_starts)
        parent_starts = child_starts
    return parent_starts, levels


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=None, cu_seqlens=None, *,
        segment_chunks=32, return_debug=False):
    """One K1 prepare, two segment summaries, scan, and parallel K2 replay."""
    if cu_seqlens is not None:
        raise NotImplementedError("R carry-lookahead v1 supports fixed B=1 only")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != D:
        raise ValueError("q must be [1,T,H,128]")
    tensors = (q, k, v, g, beta, out)
    if not all(x.is_cuda and x.is_contiguous() for x in tensors):
        raise ValueError("q/k/v/g/beta/out must be contiguous CUDA tensors")
    if not all(x.dtype == torch.bfloat16 for x in tensors):
        raise ValueError("q/k/v/g/beta/out must be bfloat16")
    if initial_state is not None and initial_state.dtype != torch.bfloat16:
        raise NotImplementedError("R v1 supports BF16 state only")
    if final_state is not None and final_state.dtype != torch.bfloat16:
        raise NotImplementedError("R v1 supports BF16 state only")

    _power_of_two(segment_chunks, "segment_chunks")
    total_t, heads = int(q.shape[1]), int(q.shape[2])
    segment_tokens = CHUNK * segment_chunks
    if total_t % segment_tokens:
        raise ValueError(f"T={total_t} must be divisible by {segment_tokens}")
    nseg = total_t // segment_tokens
    _power_of_two(nseg, "num_segments")

    boundaries = torch.arange(
        0, total_t + 1, segment_tokens, device=q.device, dtype=torch.long
    )
    state_shape = (nseg, heads, D, D)
    beta_t = beta.reshape(total_t, heads).t().contiguous()
    workspace = torch.empty(
        _C.reduce_workspace_size(total_t, heads, nseg),
        device=q.device, dtype=torch.uint8,
    )
    _C.reduce_prepare(
        q, k, g, beta_t, workspace, A_log, dt_bias,
        float(scale), float(lower_bound), boundaries,
    )

    dummy_out = torch.empty_like(out)
    eye = torch.eye(D, device=q.device, dtype=torch.bfloat16)
    identity = eye.view(1, 1, D, D).expand(state_shape).clone()
    zeros = torch.zeros(state_shape, device=q.device, dtype=torch.bfloat16)
    segment_a = torch.empty_like(identity)
    segment_b = torch.empty_like(identity)
    _C.reduce_recurrence(
        torch.zeros_like(v), beta_t, workspace, dummy_out,
        identity, segment_a, boundaries,
    )
    _C.reduce_recurrence(
        v, beta_t, workspace, dummy_out,
        zeros, segment_b, boundaries,
    )

    starts, levels = carry_lookahead_starts(segment_a, segment_b, initial_state)
    segment_finals = torch.empty_like(segment_a)
    _C.reduce_recurrence(
        v, beta_t, workspace, out, starts, segment_finals, boundaries,
    )
    if final_state is not None:
        final_state[0].copy_(segment_finals[-1])

    if return_debug:
        return DebugInfo(
            segment_tokens, nseg, int(math.log2(nseg)),
            segment_a, segment_b, starts,
        )
    return None


__all__ = ["fwd", "serial_fwd", "carry_lookahead_starts", "DebugInfo"]
