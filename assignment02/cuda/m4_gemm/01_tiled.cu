// 问题 4.1(FROM-SCRATCH):tiling——把 3.2 的单 tile 扩成完整 GEMM。
//
// 形状默认 4096^3,bf16 输入,f32 累加,cta_group::1。tile 固定
// BM=128 BN=64 BK=64,每 block 128 线程,grid = (M/BM, N/BN)。
// 判测(main 已给出)对 cuBLAS 逐位严格相等:数据是小整数,和的
// 绝对值 <= 2^24,f32 累加无舍入,顺序无关,可以逐位判。
//
// 相对 3.2 的新内容只有两件:
//   - 每个 block 按 blockIdx 认领输出 tile(tileM, tileN)
//   - K 维不再一次装完:iters = K/BK 轮,每轮 staging 一段、发一组
//     mma 累加到同一块 TMEM,循环结束再走 epilogue
// TMEM 分配、descriptor、swizzled staging、epilogue 都是 3.2 已有的,
// 原样搬过来改数字即可。
//
// 交付:PASS + 本文件输出的 TFLOPS/达成率(梯子表第一行,后面 4.2/4.3
// 各补一行);回答 handout 4.1 的问题(达成率 vs 机器平衡点,此时瓶颈
// 在哪个环节)。
//
// 运行:make run/m4_gemm/01_tiled;自定形状 ./bin/m4_gemm/01_tiled M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

constexpr int BM = 128, BN = 64, BK = 64;

// 128B swizzle 的物理偏移(2.3/3.2 用过的同一个)。
__host__ __device__ inline int swz128(int row, int colByte) {
    int atom = row >> 3, r = row & 7, chunk = colByte >> 4, in16 = colByte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

// SM100 smem descriptor(2.2 的位域)。
__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

__global__ void gemm_tiled(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                           float* gD, int M, int N, int K) {
    // smem 用动态分配(main 已按 (BM+BN)*BK*2 + 1024 传入),基址对齐
    // 到 1024(swizzle atom 的要求);A 区 [0, BM*BK*2),B 区随后。
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    __shared__ __align__(8) uint64_t empty;
    __shared__ uint32_t s_taddr[1];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tileM = blockIdx.x * BM, tileN = blockIdx.y * BN;
    const uint32_t empty_addr = (uint32_t)__cvta_generic_to_shared(&empty);
    const uint32_t a_base = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t b_base = (uint32_t)__cvta_generic_to_shared(smem + BM * BK * 2);

    if (warp == 0) {
        if (lane == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : :
                             "r"(empty_addr), "r"(1));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(s_taddr);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;" : : "r"(dst), "r"(64));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    const uint32_t taddr = s_taddr[0];
    uint32_t elected;
    asm volatile(
        "{\n.reg .pred p;\nelect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(elected));
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                           (8u << 17) | (8u << 24);

    for (int it = 0; it < K / BK; ++it) {
        // 逻辑 global 坐标带 tile/K 偏移；SMEM swizzle 使用本地坐标。
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int row = i / BK, k = i % BK;
            *reinterpret_cast<__nv_bfloat16*>(&smem[swz128(row, k * 2)]) =
                gA[(tileM + row) * K + it * BK + k];
        }
        for (int i = tid; i < BN * BK; i += blockDim.x) {
            int row = i / BK, k = i % BK;
            *reinterpret_cast<__nv_bfloat16*>(
                &smem[BM * BK * 2 + swz128(row, k * 2)]) =
                gB[(tileN + row) * K + it * BK + k];
        }
        asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
        __syncthreads();

        if (warp == 0 && elected) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (int kk = 0; kk < BK; kk += 16) {
                uint64_t a_desc = make_desc_sm100(a_base + kk * 2, 0, 1024, 2);
                uint64_t b_desc = make_desc_sm100(b_base + kk * 2, 0, 1024, 2);
                uint32_t accumulate = (it != 0) || (kk != 0);
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 "
                    "[%0], %1, %2, %3, p;\n}" : :
                    "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                    "r"(accumulate));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];" : : "r"(empty_addr) : "memory");
        }
        // 单缓冲：全部线程确认 MMA 已不再读取这份 A/B，才覆盖它。
        mbar_wait(empty_addr, it & 1);
        __syncthreads();
    }

    asm volatile("tcgen05.fence::after_thread_sync;");
    const int row = warp * 32 + lane;
    for (int col = 0; col < BN; col += 8) {
        uint32_t src = taddr + ((uint32_t)(warp * 32) << 16) + col;
        float r[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]),
              "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
            : "r"(src));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
        for (int i = 0; i < 8; ++i)
            gD[(tileM + row) * N + tileN + col + i] = r[i];
    }
    __syncthreads();
    if (warp == 0)
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" : :
            "r"(taddr), "r"(64));
    (void)M;
}

int main(int argc, char** argv) {
    int M = argc > 3 ? atoi(argv[1]) : 4096;
    int N = argc > 3 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK) {
        printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
        return 1;
    }
    size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    // 先填 NaN 模式:kernel 没写满/没写对时判测必 FAIL,不受残留数据干扰
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    dim3 grid(M / BM, N / BN);
    size_t smemBytes = (size_t)(BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_tiled,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_tiled<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K);
    };
    launch();
    CUDA_CHECK_KERNEL();

    // cuBLAS 参考(bf16 输入 f32 累加;小整数下与 tensor core 逐位一致)
    // D[M,N] 行主序 = D^T 列主序:C_col[N,M] = B_col[K,N]^T x A_col[K,M]
    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF,
                 K, dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; i++) bad += got[i] != ref[i];

    int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, iters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        },
        iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[4.1 tiled] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f TFLOPS  "
           "(cuBLAS %.1f, 达成率 %.0f%%)\n",
           M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
           100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
