#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <optional>

#include "reduce.h"

torch::Tensor bmm_bf16_sm80_cuda(
    torch::Tensor a,
    torch::Tensor b,
    std::optional<torch::Tensor> add);

namespace {

constexpr int kChunk = 16;
constexpr int kD = 128;

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, name, " must be bfloat16");
}

void check_scan_matrix(const torch::Tensor& x, const char* name) {
    check_bf16_cuda(x, name);
    TORCH_CHECK(x.dim() >= 3 && x.size(-2) == kD && x.size(-1) == kD,
                name, " must have shape [...,128,128]");
}

void check_boundaries(const torch::Tensor& cu, int64_t total_t) {
    TORCH_CHECK(cu.is_cuda() && cu.is_contiguous(),
                "cu_seqlens must be contiguous CUDA");
    TORCH_CHECK(cu.scalar_type() == at::kLong && cu.dim() == 1,
                "cu_seqlens must be int64 [N+1]");
    TORCH_CHECK(cu.numel() >= 2, "cu_seqlens must contain at least [0,T]");
    // Endpoints and monotonicity are established by the Python wrapper that
    // creates this tensor.  Avoid a host synchronization in the hot path.
    (void)total_t;
}

int64_t workspace_size_impl(int64_t total_t, int64_t heads, int64_t nseq) {
    // Match the original FlashKDA varlen upper bound and byte layout.
    const int64_t total_tiles = (total_t + kChunk - 1) / kChunk + nseq;
    const int64_t per_tile =
        3 * kChunk * kD * 2 + kD * 4 + 2 * kChunk * kChunk * 2;
    const int64_t tile_prefix = ((nseq + 1) * 4 + 127) / 128 * 128;
    return heads * total_tiles * per_tile + tile_prefix;
}

}  // namespace

torch::Tensor scan_bmm(
    torch::Tensor a,
    torch::Tensor b,
    std::optional<torch::Tensor> add = std::nullopt) {
    check_scan_matrix(a, "a");
    check_scan_matrix(b, "b");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have identical shapes");
    if (add.has_value()) {
        check_scan_matrix(*add, "add");
        TORCH_CHECK(a.sizes() == add->sizes(), "add must match a/b shape");
    }
    return bmm_bf16_sm80_cuda(a, b, add);
}

int64_t core_workspace_size(int64_t total_t, int64_t heads, int64_t nseq) {
    TORCH_CHECK(total_t > 0 && heads > 0 && nseq > 0,
                "T, H and N must be positive");
    return workspace_size_impl(total_t, heads, nseq);
}

void core_prepare(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor g,
    torch::Tensor beta_t,
    torch::Tensor workspace,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    double scale,
    double lower_bound,
    torch::Tensor cu_seqlens) {
    check_bf16_cuda(q, "q");
    check_bf16_cuda(k, "k");
    check_bf16_cuda(g, "g");
    check_bf16_cuda(beta_t, "beta_t");
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == kD,
                "q must be [1,T,H,128]");
    TORCH_CHECK(k.sizes() == q.sizes() && g.sizes() == q.sizes(),
                "k/g must match q");
    const int64_t total_t = q.size(1);
    const int64_t heads = q.size(2);
    const int64_t nseq = cu_seqlens.numel() - 1;
    TORCH_CHECK(beta_t.dim() == 2 && beta_t.size(0) == heads &&
                    beta_t.size(1) == total_t,
                "beta_t must be [H,T]");
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() &&
                    workspace.scalar_type() == at::kByte,
                "workspace must be contiguous CUDA uint8");
    TORCH_CHECK(workspace.numel() >= workspace_size_impl(total_t, heads, nseq),
                "workspace is too small");
    TORCH_CHECK(a_log.is_cuda() && a_log.is_contiguous() &&
                    a_log.scalar_type() == at::kFloat &&
                    a_log.dim() == 1 && a_log.size(0) == heads,
                "A_log must be CUDA fp32 [H]");
    TORCH_CHECK(dt_bias.is_cuda() && dt_bias.is_contiguous() &&
                    dt_bias.scalar_type() == at::kFloat &&
                    dt_bias.dim() == 2 && dt_bias.size(0) == heads &&
                    dt_bias.size(1) == kD,
                "dt_bias must be CUDA fp32 [H,128]");
    check_boundaries(cu_seqlens, total_t);

    c10::cuda::CUDAGuard guard(q.device());
    const int total_tiles = int((total_t + kChunk - 1) / kChunk + nseq);
    const float gate_scale = float(lower_bound * 1.4426950408889634);
    flash_kda_prepare_launch(
        q.data_ptr(), k.data_ptr(), g.data_ptr(), beta_t.data_ptr(),
        workspace.data_ptr(), total_tiles, int(total_t), int(heads), int(nseq),
        cu_seqlens.data_ptr<int64_t>(), a_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(), float(scale), gate_scale,
        at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void core_recurrence(
    torch::Tensor v,
    torch::Tensor beta_t,
    torch::Tensor workspace,
    torch::Tensor out,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    torch::Tensor cu_seqlens) {
    check_bf16_cuda(v, "v");
    check_bf16_cuda(beta_t, "beta_t");
    check_bf16_cuda(out, "out");
    check_bf16_cuda(initial_state, "initial_state");
    check_bf16_cuda(final_state, "final_state");
    TORCH_CHECK(v.dim() == 4 && v.size(0) == 1 && v.size(3) == kD,
                "v must be [1,T,H,128]");
    TORCH_CHECK(out.sizes() == v.sizes(), "out must match v");
    const int64_t total_t = v.size(1);
    const int64_t heads = v.size(2);
    const int64_t nseq = cu_seqlens.numel() - 1;
    TORCH_CHECK(beta_t.dim() == 2 && beta_t.size(0) == heads &&
                    beta_t.size(1) == total_t,
                "beta_t must be [H,T]");
    TORCH_CHECK(initial_state.dim() == 4 &&
                    initial_state.size(0) == nseq &&
                    initial_state.size(1) == heads &&
                    initial_state.size(2) == kD &&
                    initial_state.size(3) == kD,
                "initial_state must be [N,H,128,128]");
    TORCH_CHECK(final_state.dim() == 4 &&
                    final_state.size(0) == nseq &&
                    final_state.size(1) == heads &&
                    final_state.size(2) == kD &&
                    final_state.size(3) == kD,
                "final_state must be [N,H,128,128]");
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() &&
                    workspace.scalar_type() == at::kByte &&
                    workspace.numel() >= workspace_size_impl(total_t, heads, nseq),
                "bad workspace");
    check_boundaries(cu_seqlens, total_t);

    c10::cuda::CUDAGuard guard(v.device());
    const int total_tiles = int((total_t + kChunk - 1) / kChunk + nseq);
    flash_kda_recurrence_launch(
        v.data_ptr(), beta_t.data_ptr(), workspace.data_ptr(), out.data_ptr(),
        initial_state.data_ptr(), final_state.data_ptr(), total_tiles,
        int(total_t), int(heads), int(nseq), cu_seqlens.data_ptr<int64_t>(),
        at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bind_reduce(py::module_& m) {
    m.def("reduce_workspace_size", &core_workspace_size,
          "Workspace size for one K1 prepare");
    m.def("reduce_prepare", &core_prepare,
          "Launch the original FlashKDA K1 once");
    m.def("reduce_recurrence", &core_recurrence,
          "Launch the original FlashKDA K2 directly on a prepared workspace");
    m.def("reduce_scan_bmm", &scan_bmm,
          "SM80 BF16 128x128 batched GEMM for carry-lookahead",
          py::arg("a"), py::arg("b"), py::arg("add") = py::none());
}
