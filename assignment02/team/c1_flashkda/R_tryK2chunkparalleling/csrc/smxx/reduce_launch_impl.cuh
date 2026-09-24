#include "../reduce.h"

#include "fwd_kernel1.cuh"
#include "fwd_kernel2.cuh"

using BF16 = cutlass::bfloat16_t;

namespace {

constexpr int kD = 128;
constexpr int kChunk = 16;
constexpr int kInputStages = 3;
constexpr int kOutputStages = 2;

struct WorkspaceViews {
    BF16* kd;
    BF16* qd;
    BF16* kr;
    float* gt;
    BF16* inv;
    BF16* mqk;
    int* tile_prefix;
};

WorkspaceViews split_workspace(void* workspace, int total_tiles, int heads) {
    using WS = WorkspaceSizes<kChunk, kD>;
    const int64_t n_ht = int64_t(heads) * total_tiles;
    char* ws = reinterpret_cast<char*>(workspace);
    return {
        reinterpret_cast<BF16*>(ws),
        reinterpret_cast<BF16*>(ws + n_ht * WS::kKDecayed),
        reinterpret_cast<BF16*>(
            ws + n_ht * (WS::kKDecayed + WS::kQDecayed)),
        reinterpret_cast<float*>(
            ws + n_ht *
                (WS::kKDecayed + WS::kQDecayed + WS::kKRestored)),
        reinterpret_cast<BF16*>(
            ws + n_ht *
                (WS::kKDecayed + WS::kQDecayed + WS::kKRestored +
                 WS::kGTotal)),
        reinterpret_cast<BF16*>(
            ws + n_ht *
                (WS::kKDecayed + WS::kQDecayed + WS::kKRestored +
                 WS::kGTotal + WS::kINV)),
        reinterpret_cast<int*>(ws + n_ht * WS::kPerTile),
    };
}

}  // namespace

void flash_kda_prepare_launch(
    const void* q_raw,
    const void* k_raw,
    const void* g_raw,
    const void* beta_raw,
    void* workspace,
    int total_tiles,
    int total_t,
    int heads,
    int nseq,
    const int64_t* cu_seqlens,
    const float* a_log,
    const float* dt_bias,
    float scale,
    float gate_scale,
    cudaStream_t stream) {
    using L = K1Layouts<kD, kChunk>;
    using TMAQKLayout = typename L::TMAQKLayout;
    using TMABetaLayout = typename L::TMABetaSmemLayout;
    using TMAVOLayout = typename L::TMAVOLayout;
    using TMALMLayout = typename L::TMALMLayout;
    using TMAGTotalLayout = typename L::TMAGTotalSmemLayout;

    const auto* q = static_cast<const BF16*>(q_raw);
    const auto* k = static_cast<const BF16*>(k_raw);
    const auto* g = static_cast<const BF16*>(g_raw);
    const auto* beta = static_cast<const BF16*>(beta_raw);
    WorkspaceViews ws = split_workspace(workspace, total_tiles, heads);

    auto input_layout = make_layout(
        make_shape(heads, total_t, kD),
        make_stride(kD, kD * heads, 1));
    auto beta_layout = make_layout(make_shape(heads * total_t));
    auto q_tensor = make_tensor(make_gmem_ptr(q), input_layout);
    auto k_tensor = make_tensor(make_gmem_ptr(k), input_layout);
    auto g_tensor = make_tensor(make_gmem_ptr(g), input_layout);
    auto beta_tensor = make_tensor(make_gmem_ptr(beta), beta_layout);

    const int n_ht = heads * total_tiles;
    auto ws_vec_layout = make_layout(make_shape(n_ht, kChunk, kD), LayoutRight{});
    auto ws_gt_layout = make_layout(make_shape(n_ht, kD), LayoutRight{});
    auto ws_lm_layout = make_layout(
        make_shape(n_ht, kChunk, kChunk), LayoutRight{});
    auto kd_tensor = make_tensor(make_gmem_ptr(ws.kd), ws_vec_layout);
    auto qd_tensor = make_tensor(make_gmem_ptr(ws.qd), ws_vec_layout);
    auto kr_tensor = make_tensor(make_gmem_ptr(ws.kr), ws_vec_layout);
    auto gt_tensor = make_tensor(make_gmem_ptr(ws.gt), ws_gt_layout);
    auto inv_tensor = make_tensor(make_gmem_ptr(ws.inv), ws_lm_layout);
    auto mqk_tensor = make_tensor(make_gmem_ptr(ws.mqk), ws_lm_layout);

    auto dt_layout = make_layout(make_shape(heads, kD), LayoutRight{});
    auto dt_tensor = make_tensor(make_gmem_ptr(dt_bias), dt_layout);

    auto load_q = make_tma_copy(SM90_TMA_LOAD{}, q_tensor, TMAQKLayout{});
    auto load_k = make_tma_copy(SM90_TMA_LOAD{}, k_tensor, TMAQKLayout{});
    auto load_g = make_tma_copy(SM90_TMA_LOAD{}, g_tensor, TMAQKLayout{});
    auto load_beta = make_tma_copy(
        SM90_TMA_LOAD{}, beta_tensor, TMABetaLayout{});
    auto load_dt = make_tma_copy(SM90_TMA_LOAD{}, dt_tensor, TMAGTotalLayout{});
    auto store_kd = make_tma_copy(SM90_TMA_STORE{}, kd_tensor, TMAVOLayout{});
    auto store_qd = make_tma_copy(SM90_TMA_STORE{}, qd_tensor, TMAVOLayout{});
    auto store_kr = make_tma_copy(SM90_TMA_STORE{}, kr_tensor, TMAVOLayout{});
    auto store_gt = make_tma_copy(SM90_TMA_STORE{}, gt_tensor, TMAGTotalLayout{});
    auto store_inv = make_tma_copy(SM90_TMA_STORE{}, inv_tensor, TMALMLayout{});
    auto store_mqk = make_tma_copy(SM90_TMA_STORE{}, mqk_tensor, TMALMLayout{});

    constexpr int kThreads = 256;
    using Shared = SharedStorageK1<L>;
    const int smem = sizeof(Shared);
    auto kernel = _flash_kda_fwd_prepare<
        decltype(load_q), decltype(load_k), decltype(load_beta),
        decltype(load_g), decltype(load_dt), decltype(store_kd),
        decltype(store_qd), decltype(store_kr), decltype(store_gt),
        decltype(store_inv), decltype(store_mqk), kChunk, kD, kThreads, true>;
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    _flash_kda_build_tile_prefix<<<1, 32, 0, stream>>>(
        cu_seqlens, nseq, kChunk, ws.tile_prefix);
    kernel<<<dim3(total_tiles, heads), kThreads, smem, stream>>>(
        load_q, load_k, load_beta, load_g, load_dt,
        store_kd, store_qd, store_kr, store_gt, store_inv, store_mqk,
        scale, total_t, heads, nseq, cu_seqlens, total_tiles,
        a_log, gate_scale, ws.tile_prefix);
}

void flash_kda_recurrence_launch(
    const void* v_raw,
    const void* beta_raw,
    const void* workspace_raw,
    void* out_raw,
    const void* initial_state_raw,
    void* final_state_raw,
    int total_tiles,
    int total_t,
    int heads,
    int nseq,
    const int64_t* cu_seqlens,
    cudaStream_t stream) {
    using L = K2Layouts<kD, kChunk>;
    using TMAVOLayout = typename L::TMAVOLayout;
    using TMABetaLayout = typename L::TMABetaSmemLayout;
    using TMAStateLayout = typename L::TMAStateSmemLayout;
    using TMALMLayout = typename L::TMALMLayout;
    using TMAGTotalLayout = typename L::TMAGTotalSmemLayout;

    const auto* v = static_cast<const BF16*>(v_raw);
    const auto* beta = static_cast<const BF16*>(beta_raw);
    auto* out = static_cast<BF16*>(out_raw);
    const auto* initial_state = static_cast<const BF16*>(initial_state_raw);
    auto* final_state = static_cast<BF16*>(final_state_raw);
    WorkspaceViews ws = split_workspace(
        const_cast<void*>(workspace_raw), total_tiles, heads);

    auto input_layout = make_layout(
        make_shape(heads, total_t, kD),
        make_stride(kD, kD * heads, 1));
    auto beta_layout = make_layout(make_shape(heads * total_t));
    auto state_layout = make_layout(
        make_shape(nseq * heads, kD, kD), LayoutRight{});
    auto v_tensor = make_tensor(make_gmem_ptr(v), input_layout);
    auto out_tensor = make_tensor(make_gmem_ptr(out), input_layout);
    auto beta_tensor = make_tensor(make_gmem_ptr(beta), beta_layout);
    auto initial_tensor = make_tensor(make_gmem_ptr(initial_state), state_layout);
    auto final_tensor = make_tensor(make_gmem_ptr(final_state), state_layout);

    const int n_ht = heads * total_tiles;
    auto ws_vec_layout = make_layout(make_shape(n_ht, kChunk, kD), LayoutRight{});
    auto ws_gt_layout = make_layout(make_shape(n_ht, kD), LayoutRight{});
    auto ws_lm_layout = make_layout(
        make_shape(n_ht, kChunk, kChunk), LayoutRight{});
    auto kd_tensor = make_tensor(make_gmem_ptr(ws.kd), ws_vec_layout);
    auto qd_tensor = make_tensor(make_gmem_ptr(ws.qd), ws_vec_layout);
    auto kr_tensor = make_tensor(make_gmem_ptr(ws.kr), ws_vec_layout);
    auto gt_tensor = make_tensor(make_gmem_ptr(ws.gt), ws_gt_layout);
    auto inv_tensor = make_tensor(make_gmem_ptr(ws.inv), ws_lm_layout);
    auto mqk_tensor = make_tensor(make_gmem_ptr(ws.mqk), ws_lm_layout);

    auto load_v = make_tma_copy(SM90_TMA_LOAD{}, v_tensor, TMAVOLayout{});
    auto load_beta = make_tma_copy(
        SM90_TMA_LOAD{}, beta_tensor, TMABetaLayout{});
    auto load_kd = make_tma_copy(SM90_TMA_LOAD{}, kd_tensor, TMAVOLayout{});
    auto load_qd = make_tma_copy(SM90_TMA_LOAD{}, qd_tensor, TMAVOLayout{});
    auto load_kr = make_tma_copy(SM90_TMA_LOAD{}, kr_tensor, TMAVOLayout{});
    auto load_gt = make_tma_copy(SM90_TMA_LOAD{}, gt_tensor, TMAGTotalLayout{});
    auto load_inv = make_tma_copy(SM90_TMA_LOAD{}, inv_tensor, TMALMLayout{});
    auto load_mqk = make_tma_copy(SM90_TMA_LOAD{}, mqk_tensor, TMALMLayout{});
    auto load_state = make_tma_copy(
        SM90_TMA_LOAD{}, initial_tensor, TMAStateLayout{});
    auto store_state = make_tma_copy(
        SM90_TMA_STORE{}, final_tensor, TMAStateLayout{});
    auto store_out = make_tma_copy(SM90_TMA_STORE{}, out_tensor, TMAVOLayout{});

    constexpr int kThreads = 32 * 2 + 128;
    using Shared = SharedStorageK2<L, kInputStages, kOutputStages>;
    const int smem = sizeof(Shared);
    auto kernel = _flash_kda_fwd_recurrence<
        decltype(load_v), decltype(load_beta), decltype(load_kd),
        decltype(load_qd), decltype(load_kr), decltype(load_gt),
        decltype(load_inv), decltype(load_mqk), decltype(load_state),
        decltype(store_state), decltype(store_out), kChunk, kD,
        kInputStages, kOutputStages, kThreads, true, true, false, true>;
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(nseq, heads), kThreads, smem, stream>>>(
        load_v, load_beta, load_kd, load_qd, load_kr, load_gt,
        load_inv, load_mqk, load_state, store_state, store_out,
        out, total_t, heads, nseq, cu_seqlens, total_tiles);
}
