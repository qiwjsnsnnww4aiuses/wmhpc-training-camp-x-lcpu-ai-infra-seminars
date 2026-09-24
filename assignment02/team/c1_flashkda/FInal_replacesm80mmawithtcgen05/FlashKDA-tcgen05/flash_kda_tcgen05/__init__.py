import torch
from flash_kda_tcgen05_C import fwd as _fwd_raw, get_workspace_size


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=None, cu_seqlens=None):
    """Experimental FlashKDA forward whose K2 GEMMs use tcgen05."""
    B, T_seq, H = q.shape[0], q.shape[1], q.shape[2]
    T_total = B * T_seq
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
    workspace = torch.empty(
        get_workspace_size(T_total, H, N), dtype=torch.uint8, device=q.device)
    _fwd_raw(
        q, k, v, g, beta, float(scale), out, workspace,
        A_log, dt_bias, lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["fwd", "get_workspace_size"]
