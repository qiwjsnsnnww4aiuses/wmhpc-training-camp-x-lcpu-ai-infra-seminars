// C1 FlashKDA challenge: keep the GEMM algorithm unchanged and replace only
// the Tensor Core instruction path.
//
// Both kernels compute exactly
//     D[m,n] = sum_{k=0}^{15} A[m,k] * B[n,k]
// with BF16 inputs and FP32 accumulation.  The SM80 baseline decomposes the
// work into mma.sync.m16n8k16 instructions.  The SM100 path issues one
// tcgen05.mma.m128n128k16 instruction.  For M=16, rows 16..127 of A are zero
// padded and discarded, exposing the shape-mismatch cost without changing the
// useful result.  M=128 is a naturally matched control case.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
                         cudaGetErrorName(err_), __FILE__, __LINE__,           \
                         cudaGetErrorString(err_));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int kPhysicalM = 128;
constexpr int kN = 128;
constexpr int kK = 16;

// SM100 128-byte swizzle.  Logical rows are the M/N dimension and colByte is
// the byte offset along K.  Each row occupies a 128-byte physical stride.
__host__ __device__ __forceinline__ int swizzle_128b(int row, int col_byte) {
    const int atom = row >> 3;
    const int row_in_atom = row & 7;
    const int chunk = col_byte >> 4;
    const int in_16b = col_byte & 15;
    return atom * 1024 + row_in_atom * 128 +
           ((chunk ^ row_in_atom) << 4) + in_16b;
}

__device__ __forceinline__ uint64_t make_desc_sm100(uint32_t shared_addr,
                                                     uint32_t leading_byte,
                                                     uint32_t stride_byte,
                                                     uint32_t layout) {
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((shared_addr >> 4) & 0x3fff);
    desc |= static_cast<uint64_t>((leading_byte >> 4) & 0x3fff) << 16;
    desc |= static_cast<uint64_t>((stride_byte >> 4) & 0x3fff) << 32;
    desc |= static_cast<uint64_t>(1) << 46;  // SM100 descriptor version.
    desc |= static_cast<uint64_t>(layout) << 61;
    return desc;
}

__device__ __forceinline__ void wait_mbarrier(uint32_t mbar_addr,
                                               uint32_t phase) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n"
            "}"
            : "=r"(done)
            : "r"(mbar_addr), "r"(phase));
    }
}

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo,
                                               __nv_bfloat16 hi) {
    uint32_t bits;
    *reinterpret_cast<__nv_bfloat162*>(&bits) = __halves2bfloat162(lo, hi);
    return bits;
}

// Warp-level reference path.  Each warp owns 32 output columns.  It uses the
// same SM80-generation instruction family as FlashKDA's current hot path.
template <int UsefulM>
__global__ void sm80_mma_kernel(const __nv_bfloat16* __restrict__ global_a,
                                const __nv_bfloat16* __restrict__ global_b,
                                float* __restrict__ global_d) {
    static_assert(UsefulM == 16 || UsefulM == 128,
                  "This experiment supports M=16 and M=128");
    __shared__ __nv_bfloat16 shared_a[UsefulM * kK];
    __shared__ __nv_bfloat16 shared_b[kN * kK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int job = blockIdx.x;

    const __nv_bfloat16* job_a = global_a + job * UsefulM * kK;
    const __nv_bfloat16* job_b = global_b + job * kN * kK;
    float* job_d = global_d + job * UsefulM * kN;

    for (int i = tid; i < UsefulM * kK; i += blockDim.x) shared_a[i] = job_a[i];
    for (int i = tid; i < kN * kK; i += blockDim.x) shared_b[i] = job_b[i];
    __syncthreads();

    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int warp_n = warp * 32;

    for (int m0 = 0; m0 < UsefulM; m0 += 16) {
#pragma unroll
        for (int n8 = 0; n8 < 32; n8 += 8) {
            const int k_lo = thread_in_group * 2;
            uint32_t a[4];
            a[0] = pack_bf16(shared_a[(m0 + group) * kK + k_lo],
                             shared_a[(m0 + group) * kK + k_lo + 1]);
            a[1] = pack_bf16(shared_a[(m0 + group + 8) * kK + k_lo],
                             shared_a[(m0 + group + 8) * kK + k_lo + 1]);
            a[2] = pack_bf16(shared_a[(m0 + group) * kK + k_lo + 8],
                             shared_a[(m0 + group) * kK + k_lo + 9]);
            a[3] = pack_bf16(shared_a[(m0 + group + 8) * kK + k_lo + 8],
                             shared_a[(m0 + group + 8) * kK + k_lo + 9]);

            const int n_for_b = warp_n + n8 + group;
            uint32_t b[2];
            b[0] = pack_bf16(shared_b[n_for_b * kK + k_lo],
                             shared_b[n_for_b * kK + k_lo + 1]);
            b[1] = pack_bf16(shared_b[n_for_b * kK + k_lo + 8],
                             shared_b[n_for_b * kK + k_lo + 9]);

            float c[4] = {0.f, 0.f, 0.f, 0.f};
            float d[4];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%10,%11,%12,%13};\n"
                : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                  "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
                  "f"(c[2]), "f"(c[3]));

            const int n0 = warp_n + n8 + thread_in_group * 2;
            job_d[(m0 + group) * kN + n0] = d[0];
            job_d[(m0 + group) * kN + n0 + 1] = d[1];
            job_d[(m0 + group + 8) * kN + n0] = d[2];
            job_d[(m0 + group + 8) * kN + n0 + 1] = d[3];
        }
    }
}

// SM100 instruction-replacement path.  The physical Tensor Core tile is
// m128n128k16.  UsefulM=16 deliberately pads A with zeros but stores only the
// first 16 rows, so its externally visible mathematical operation is identical
// to the SM80 baseline.
template <int UsefulM>
__global__ void tcgen05_mma_kernel(const __nv_bfloat16* __restrict__ global_a,
                                   const __nv_bfloat16* __restrict__ global_b,
                                   float* __restrict__ global_d) {
    static_assert(UsefulM == 16 || UsefulM == 128,
                  "This experiment supports M=16 and M=128");
    __shared__ __align__(1024) uint8_t shared_a[kPhysicalM * 128];
    __shared__ __align__(1024) uint8_t shared_b[kN * 128];
    __shared__ __align__(8) uint64_t mbarrier;
    __shared__ uint32_t shared_tmem_addr[1];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int job = blockIdx.x;
    const __nv_bfloat16* job_a = global_a + job * UsefulM * kK;
    const __nv_bfloat16* job_b = global_b + job * kN * kK;
    float* job_d = global_d + job * UsefulM * kN;

    const uint32_t mbar_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&mbarrier));

    // TMEM allocation is a warp-collective operation.
    if (warp == 0) {
        if (lane == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :
                         : "r"(mbar_addr), "r"(1));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        const uint32_t dst = static_cast<uint32_t>(
            __cvta_generic_to_shared(shared_tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;"
            :
            : "r"(dst), "r"(128));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    // The physical M tile is 128.  Invalid rows become exact zero, which is
    // the only adaptation required for the useful M=16 operation.
    for (int i = tid; i < kPhysicalM * kK; i += blockDim.x) {
        const int row = i / kK;
        const int k = i % kK;
        const __nv_bfloat16 value =
            row < UsefulM ? job_a[row * kK + k] : __float2bfloat16(0.f);
        *reinterpret_cast<__nv_bfloat16*>(
            &shared_a[swizzle_128b(row, k * 2)]) = value;
    }
    for (int i = tid; i < kN * kK; i += blockDim.x) {
        const int row = i / kK;
        const int k = i % kK;
        *reinterpret_cast<__nv_bfloat16*>(
            &shared_b[swizzle_128b(row, k * 2)]) = job_b[row * kK + k];
    }

    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t tmem_addr = shared_tmem_addr[0];

    uint32_t elected;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xffffffff;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}"
        : "=r"(elected));
    if (warp == 0 && elected) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        const uint32_t a_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_a));
        const uint32_t b_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(shared_b));
        const uint64_t a_desc = make_desc_sm100(a_base, 0, 1024, 2);
        const uint64_t b_desc = make_desc_sm100(b_base, 0, 1024, 2);

        // f32 D, bf16 A/B, N=128 and M=128.  The accumulate predicate is
        // false because K=16 is completed by one instruction.
        const uint32_t instruction_desc =
            (1u << 4) | (1u << 7) | (1u << 10) |
            (16u << 17) | (8u << 24);
        asm volatile(
            "{\n"
            ".reg .pred accumulate;\n"
            "setp.ne.b32 accumulate, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, accumulate;\n"
            "}\n"
            :
            : "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
              "r"(instruction_desc), "r"(0));
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
            ".shared::cluster.b64 [%0];"
            :
            : "r"(mbar_addr)
            : "memory");
    }

    wait_mbarrier(mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    // Every warp reads its 32 physical rows.  In the padded case, rows 16..127
    // are intentionally discarded, but their TMEM-read cost remains measured.
    const int row = warp * 32 + lane;
    for (int col = 0; col < kN; col += 8) {
        const uint32_t src =
            tmem_addr + (static_cast<uint32_t>(warp * 32) << 16) + col;
        float result[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(result[0]), "=f"(result[1]), "=f"(result[2]),
              "=f"(result[3]), "=f"(result[4]), "=f"(result[5]),
              "=f"(result[6]), "=f"(result[7])
            : "r"(src));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        if (row < UsefulM) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                job_d[row * kN + col + i] = result[i];
        }
    }

    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_addr), "r"(128));
    }
}

template <typename Launch>
float benchmark_ms(Launch launch, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / iterations;
}

struct Validation {
    long mismatches = 0;
    float max_abs_error = 0.f;
};

Validation compare(const std::vector<float>& got,
                   const std::vector<float>& reference) {
    Validation result;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float error = std::fabs(got[i] - reference[i]);
        result.max_abs_error = std::max(result.max_abs_error, error);
        if (error != 0.f) {
            if (result.mismatches < 5) {
                std::fprintf(stderr,
                             "mismatch [%zu]: got %.7g, expected %.7g\n",
                             i, got[i], reference[i]);
            }
            ++result.mismatches;
        }
    }
    return result;
}

template <int UsefulM>
bool run_case(const char* case_name, int jobs, int warmup, int iterations,
              unsigned seed) {
    std::mt19937 rng(seed + UsefulM);
    std::uniform_int_distribution<int> distribution(-2, 2);
    std::vector<__nv_bfloat16> host_a(
        static_cast<size_t>(jobs) * UsefulM * kK);
    std::vector<__nv_bfloat16> host_b(
        static_cast<size_t>(jobs) * kN * kK);
    for (auto& value : host_a)
        value = __float2bfloat16(static_cast<float>(distribution(rng)));
    for (auto& value : host_b)
        value = __float2bfloat16(static_cast<float>(distribution(rng)));

    std::vector<float> reference(UsefulM * kN, 0.f);
    for (int m = 0; m < UsefulM; ++m) {
        for (int n = 0; n < kN; ++n) {
            for (int k = 0; k < kK; ++k) {
                reference[m * kN + n] +=
                    __bfloat162float(host_a[m * kK + k]) *
                    __bfloat162float(host_b[n * kK + k]);
            }
        }
    }

    __nv_bfloat16* device_a = nullptr;
    __nv_bfloat16* device_b = nullptr;
    float* device_sm80 = nullptr;
    float* device_tcgen05 = nullptr;
    const size_t a_bytes = host_a.size() * sizeof(__nv_bfloat16);
    const size_t b_bytes = host_b.size() * sizeof(__nv_bfloat16);
    const size_t d_elements = static_cast<size_t>(jobs) * UsefulM * kN;
    const size_t d_bytes = d_elements * sizeof(float);
    CUDA_CHECK(cudaMalloc(&device_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&device_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&device_sm80, d_bytes));
    CUDA_CHECK(cudaMalloc(&device_tcgen05, d_bytes));
    CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), a_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), b_bytes,
                          cudaMemcpyHostToDevice));

    const dim3 grid(jobs);
    const dim3 block(128);
    auto launch_sm80 = [&] {
        sm80_mma_kernel<UsefulM><<<grid, block>>>(device_a, device_b,
                                                  device_sm80);
    };
    auto launch_tcgen05 = [&] {
        tcgen05_mma_kernel<UsefulM><<<grid, block>>>(device_a, device_b,
                                                    device_tcgen05);
    };

    launch_sm80();
    launch_tcgen05();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got_sm80(reference.size());
    std::vector<float> got_tcgen05(reference.size());
    CUDA_CHECK(cudaMemcpy(got_sm80.data(), device_sm80,
                          got_sm80.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_tcgen05.data(), device_tcgen05,
                          got_tcgen05.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const Validation sm80_check = compare(got_sm80, reference);
    const Validation tcgen05_check = compare(got_tcgen05, reference);

    const float sm80_ms = benchmark_ms(launch_sm80, warmup, iterations);
    const float tcgen05_ms = benchmark_ms(launch_tcgen05, warmup, iterations);

    const double useful_flops =
        2.0 * jobs * UsefulM * kN * kK;
    const double tcgen05_issued_flops =
        2.0 * jobs * kPhysicalM * kN * kK;
    const double sm80_useful_tflops = useful_flops / (sm80_ms * 1.0e9);
    const double tcgen05_useful_tflops = useful_flops / (tcgen05_ms * 1.0e9);
    const double tcgen05_issued_tflops =
        tcgen05_issued_flops / (tcgen05_ms * 1.0e9);

    std::printf("\n[%s] useful=(M=%d,N=%d,K=%d), tcgen05 physical M=%d\n",
                case_name, UsefulM, kN, kK, kPhysicalM);
    std::printf("  correctness  SM80:    %s, mismatches=%ld, max_abs=%.7g\n",
                sm80_check.mismatches == 0 ? "PASS" : "FAIL",
                sm80_check.mismatches, sm80_check.max_abs_error);
    std::printf("  correctness  tcgen05: %s, mismatches=%ld, max_abs=%.7g\n",
                tcgen05_check.mismatches == 0 ? "PASS" : "FAIL",
                tcgen05_check.mismatches, tcgen05_check.max_abs_error);
    std::printf("  SM80 mma.sync: %.6f ms, useful %.3f TFLOP/s\n",
                sm80_ms, sm80_useful_tflops);
    std::printf("  SM100 tcgen05: %.6f ms, useful %.3f TFLOP/s, "
                "issued %.3f TFLOP/s\n",
                tcgen05_ms, tcgen05_useful_tflops, tcgen05_issued_tflops);
    std::printf("  speedup tcgen05/SM80: %.3fx; issued/useful FLOP: %.1fx\n",
                sm80_ms / tcgen05_ms,
                tcgen05_issued_flops / useful_flops);

    CUDA_CHECK(cudaFree(device_tcgen05));
    CUDA_CHECK(cudaFree(device_sm80));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_a));
    return sm80_check.mismatches == 0 && tcgen05_check.mismatches == 0;
}

int main(int argc, char** argv) {
    const int jobs = argc > 1 ? std::atoi(argv[1]) : 256;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 20;
    const unsigned seed = argc > 4 ? static_cast<unsigned>(std::strtoul(
                                                argv[4], nullptr, 10))
                                   : 20260910u;
    if (jobs <= 0 || iterations <= 0 || warmup < 0) {
        std::fprintf(stderr,
                     "usage: %s [jobs>0] [iterations>0] [warmup>=0] [seed]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp property{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&property, device));
    std::printf("device: %s (compute capability %d.%d)\n", property.name,
                property.major, property.minor);
    std::printf("jobs=%d iterations=%d warmup=%d seed=%u\n",
                jobs, iterations, warmup, seed);
    if (property.major < 10) {
        std::fprintf(stderr,
                     "tcgen05 requires an SM100-family GPU; run this on B300.\n");
        return EXIT_FAILURE;
    }

    const bool small_ok =
        run_case<16>("K1/K2 thin GEMM (zero-padded)", jobs, warmup,
                     iterations, seed);
    const bool matched_ok =
        run_case<128>("K2 state-delta GEMM (naturally matched)", jobs,
                      warmup, iterations, seed);
    std::printf("\noverall correctness: %s\n",
                small_ok && matched_ok ? "PASS" : "FAIL");
    return small_ok && matched_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
