#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>
#include <climits>
#include <cstdint>
#include <optional>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int kD = 128;
constexpr int kTile = 16;
constexpr int kTilesPerDim = kD / kTile;
constexpr int kOutputTiles = kTilesPerDim * kTilesPerDim;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = 32 * kWarpsPerBlock;

// One warp computes one 16x16 output tile.  Eight warps share a block only to
// amortize launch/block scheduling; each warp owns a separate 1 KiB FP32
// scratch tile used to convert the WMMA accumulator back to BF16.
__global__ void bmm_bf16_sm80_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ add,
    __nv_bfloat16* __restrict__ out,
    int batch) {
    const int matrix = int(blockIdx.x);
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int output_tile = int(blockIdx.y) * kWarpsPerBlock + warp;
    if (matrix >= batch || output_tile >= kOutputTiles) return;

    const int tile_m = output_tile / kTilesPerDim;
    const int tile_n = output_tile % kTilesPerDim;
    const int64_t matrix_offset = int64_t(matrix) * kD * kD;

    // BF16 WMMA operands use CUDA's storage type directly.  The
    // wmma::precision namespace is for precision tags such as TF32 and does
    // not define a bfloat16 member in CUDA 13.
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                   __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                   __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

#pragma unroll
    for (int kb = 0; kb < kTilesPerDim; ++kb) {
        const __nv_bfloat16* a_ptr =
            a + matrix_offset + (tile_m * kTile) * kD + kb * kTile;
        const __nv_bfloat16* b_ptr =
            b + matrix_offset + (kb * kTile) * kD + tile_n * kTile;
        wmma::load_matrix_sync(a_frag, a_ptr, kD);
        wmma::load_matrix_sync(b_frag, b_ptr, kD);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    __shared__ float scratch[kWarpsPerBlock][kTile * kTile];
    wmma::store_matrix_sync(
        scratch[warp], acc_frag, kTile, wmma::mem_row_major);
    __syncwarp();

    const int row0 = tile_m * kTile;
    const int col0 = tile_n * kTile;
    for (int linear = lane; linear < kTile * kTile; linear += 32) {
        const int row = linear / kTile;
        const int col = linear % kTile;
        const int64_t idx = matrix_offset + int64_t(row0 + row) * kD + col0 + col;
        float value = scratch[warp][linear];
        if (add != nullptr) value += __bfloat162float(add[idx]);
        out[idx] = __float2bfloat16_rn(value);
    }
}

}  // namespace

torch::Tensor bmm_bf16_sm80_cuda(
    torch::Tensor a,
    torch::Tensor b,
    std::optional<torch::Tensor> add) {
    c10::cuda::CUDAGuard device_guard(a.device());
    auto out = torch::empty_like(a);
    const int64_t matrices64 = a.numel() / (kD * kD);
    TORCH_CHECK(matrices64 > 0 && matrices64 <= INT_MAX,
                "unsupported batch size: ", matrices64);
    const int matrices = int(matrices64);

    const auto* a_ptr = reinterpret_cast<const __nv_bfloat16*>(
        a.data_ptr<at::BFloat16>());
    const auto* b_ptr = reinterpret_cast<const __nv_bfloat16*>(
        b.data_ptr<at::BFloat16>());
    const __nv_bfloat16* add_ptr = nullptr;
    if (add.has_value()) {
        add_ptr = reinterpret_cast<const __nv_bfloat16*>(
            add->data_ptr<at::BFloat16>());
    }
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(
        out.data_ptr<at::BFloat16>());

    dim3 grid(
        static_cast<unsigned>(matrices),
        static_cast<unsigned>(kOutputTiles / kWarpsPerBlock));
    bmm_bf16_sm80_kernel<<<
        grid, kThreads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        a_ptr, b_ptr, add_ptr, out_ptr, matrices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
