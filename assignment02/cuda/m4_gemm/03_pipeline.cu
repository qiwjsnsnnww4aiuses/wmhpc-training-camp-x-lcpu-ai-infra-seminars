// 问题 4.3(FROM-SCRATCH,模块压轴):多级缓冲流水。
//
// 从你自己的 02_tma.cu 出发,把单缓冲扩成 STAGES 级循环缓冲:TMA 往
// 前预取后续 K 段,mma 消费当前段,装载与计算重叠。STAGES 是编译参数:
//   STAGES=4 make -B run/m4_gemm/03_pipeline
// (-B 不能省:只改 -D 不改文件,make 会认为无需重编。)
//
// 明确不要求:warp specialization、persistent kernel、epilogue 融合。
// 不设达成率门槛,评分看实验与归因质量。
//
// 两个已知事实,直接告知:
//   1. smem 用量 = STAGES*(BM+BN)*BK*2,STAGES>=3 起超过 48KB 静态
//      上限,必须动态 smem + cudaFuncSetAttribute(main 已配好)。
//   2. 一条真实的流水线 hazard(我们开发答案时踩到的,写出来让你避开):
//      "机会式预取"(try_wait 非阻塞,空了就发)不能替代"强制发射"。
//      若本轮要消费的那段 TMA 在早先检查时 stage 未空而被跳过,后面
//      wait full 等的就是一条从未发出的拷贝——死锁。症状签名很典型:
//      1024^3 侥幸全过,4096^3 必挂(13 万次机会必中一次)。正确结构:
//      本轮要消费的 TMA 用阻塞等 empty 保证发出,机会式 try_wait 只
//      用于更深的预取。另外 empty mbarrier 必须每 stage 一个:单个
//      mbar 的 parity 区分不了相隔 2 轮的完成,STAGES>=2 必然歧义。
//
// 交付:
//   - 梯子表第三行(4096^3,默认 STAGES=3)
//   - stages 扫描表:S ∈ {2,3,4,6},在两个形状上各扫一遍——4096^3 与
//     M=256 N=4096 K=16384(小 grid、长 K)。两张表的 S 敏感度不一样,
//     解释差异来自什么(提示方向:每 SM 常驻 block 数怎么随 smem 用量
//     变、块间并发本身能隐藏多少延迟)。./sweep_stages.sh 会跑全表
//   - 流水时空图:任选一个 S,画出稳态下 TMA/mma 在各 stage 上的重叠
//   - handout 4.3 的三问:瓶颈移动;梯子表逐级归因(含 assignment01
//     的 naive matmul 同口径对照);smem 与 TMEM 谁先顶住扩 stage/tile
//
// 运行:make run/m4_gemm/03_pipeline;./bin/m4_gemm/03_pipeline M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

#ifndef STAGES
#define STAGES 3
#endif

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int NSTAGE = STAGES;
constexpr int A_BYTES = BM * BK * 2;
constexpr int STAGE_BYTES = (BM + BN) * BK * 2;
static_assert(NSTAGE >= 2, "pipeline needs at least two stages");

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

// 非阻塞版:成功返回 true。机会式深预取用它。
__device__ inline bool mbar_try(uint32_t mbar, uint32_t phase) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(mbar), "r"(phase));
    return done;
}

// 只负责发射：full 完成才表示 A/B 真正已写完。
__device__ inline void issue_tile(int it, uint32_t a_base, uint32_t b_base,
                                  uint32_t full_addr, uint64_t map_a,
                                  uint64_t map_b, int tileM, int tileN) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        : : "r"(full_addr), "r"(STAGE_BYTES) : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        : : "r"(a_base), "l"(map_a), "r"(full_addr),
            "r"(it * BK), "r"(tileM) : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        : : "r"(b_base), "l"(map_b), "r"(full_addr),
            "r"(it * BK), "r"(tileN) : "memory");
}

__global__ void gemm_pipeline(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                              float* gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    __shared__ __align__(8) uint64_t full[NSTAGE], empty[NSTAGE];
    __shared__ uint32_t s_taddr[1];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tileM = blockIdx.x * BM, tileN = blockIdx.y * BN;
    if (warp == 0) {
        if (lane == 0) {
            for (int s = 0; s < NSTAGE; ++s) {
                uint32_t f = (uint32_t)__cvta_generic_to_shared(&full[s]);
                uint32_t e = (uint32_t)__cvta_generic_to_shared(&empty[s]);
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : :
                                 "r"(f), "r"(1));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : :
                                 "r"(e), "r"(1));
            }
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
    const int iters = K / BK;

    if (warp == 0 && elected) {
        const uint64_t map_a = reinterpret_cast<uint64_t>(&tmapA);
        const uint64_t map_b = reinterpret_cast<uint64_t>(&tmapB);
        int next_issue = 0; // 所有 j < next_issue 的 TMA 已发射且只发一次。

        // Prime：每个 stage 第一次使用天然为空，先把最多 S 块送上路。
        for (; next_issue < NSTAGE && next_issue < iters; ++next_issue) {
            int s = next_issue;
            uint8_t* stage = smem + s * STAGE_BYTES;
            issue_tile(next_issue,
                       (uint32_t)__cvta_generic_to_shared(stage),
                       (uint32_t)__cvta_generic_to_shared(stage + A_BYTES),
                       (uint32_t)__cvta_generic_to_shared(&full[s]),
                       map_a, map_b, tileM, tileN);
        }

        for (int it = 0; it < iters; ++it) {
            // Force issue：早先 try 失败时，当前必需块不能被遗忘。
            while (next_issue <= it) {
                int j = next_issue, s = j % NSTAGE, gen = j / NSTAGE;
                if (gen != 0)
                    mbar_wait((uint32_t)__cvta_generic_to_shared(&empty[s]),
                              (gen - 1) & 1);
                uint8_t* stage = smem + s * STAGE_BYTES;
                issue_tile(j, (uint32_t)__cvta_generic_to_shared(stage),
                           (uint32_t)__cvta_generic_to_shared(stage + A_BYTES),
                           (uint32_t)__cvta_generic_to_shared(&full[s]),
                           map_a, map_b, tileM, tileN);
                ++next_issue;
            }

            // Opportunistic prefetch：只提前处理空出的 stage，不阻塞。
            while (next_issue < iters) {
                int j = next_issue, s = j % NSTAGE, gen = j / NSTAGE;
                if (gen != 0 &&
                    !mbar_try((uint32_t)__cvta_generic_to_shared(&empty[s]),
                              (gen - 1) & 1))
                    break;
                uint8_t* stage = smem + s * STAGE_BYTES;
                issue_tile(j, (uint32_t)__cvta_generic_to_shared(stage),
                           (uint32_t)__cvta_generic_to_shared(stage + A_BYTES),
                           (uint32_t)__cvta_generic_to_shared(&full[s]),
                           map_a, map_b, tileM, tileN);
                ++next_issue;
            }

            int s = it % NSTAGE, gen = it / NSTAGE;
            mbar_wait((uint32_t)__cvta_generic_to_shared(&full[s]), gen & 1);
            // 保守地完成上一条 D 累加链，再发下一组 MMA；TMA 预取
            // 已经发出，仍能与当前/上一轮 Tensor Core 工作重叠。
            if (it != 0) {
                int prev = (it - 1) % NSTAGE;
                int prev_gen = (it - 1) / NSTAGE;
                mbar_wait((uint32_t)__cvta_generic_to_shared(&empty[prev]),
                          prev_gen & 1);
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint8_t* stage = smem + s * STAGE_BYTES;
            uint32_t a_base = (uint32_t)__cvta_generic_to_shared(stage);
            uint32_t b_base = (uint32_t)__cvta_generic_to_shared(stage + A_BYTES);
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
            uint32_t empty_addr = (uint32_t)__cvta_generic_to_shared(&empty[s]);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];" : : "r"(empty_addr) : "memory");
        }
    }

    // Drain：最后一轮没有后继迭代替我们等待它。
    int last = iters - 1;
    mbar_wait((uint32_t)__cvta_generic_to_shared(&empty[last % NSTAGE]),
              (last / NSTAGE) & 1);
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
    (void)gA; (void)gB; (void)M;
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
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    CUtensorMap tmapA = {}, tmapB = {};
    const cuuint64_t dimsA[2] = {(cuuint64_t)K, (cuuint64_t)M};
    const cuuint64_t dimsB[2] = {(cuuint64_t)K, (cuuint64_t)N};
    const cuuint64_t strides[1] = {(cuuint64_t)K * 2};
    const cuuint32_t boxA[2] = {BK, BM}, boxB[2] = {BK, BN};
    const cuuint32_t elemStrides[2] = {1, 1};
    CUresult rc = cuTensorMapEncodeTiled(
        &tmapA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dA, dimsA, strides,
        boxA, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "A tensor map encode failed: CUresult=%d\n", (int)rc);
        return 1;
    }
    rc = cuTensorMapEncodeTiled(
        &tmapB, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dB, dimsB, strides,
        boxB, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "B tensor map encode failed: CUresult=%d\n", (int)rc);
        return 1;
    }

    dim3 grid(M / BM, N / BN);
    // NSTAGE=3 时 72KB+对齐余量,超 48KB 静态上限,动态 smem 必须。
    size_t smemBytes = (size_t)NSTAGE * (BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA,
                                                tmapB);
    };
    launch();
    CUDA_CHECK_KERNEL();

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
    printf("[4.3 pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
           "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
           NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops,
           cub_tflops, 100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
