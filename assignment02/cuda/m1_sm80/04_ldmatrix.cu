// 问题 1.4:把 1.3 的手工装载换成 ldmatrix,两条路径共存、都要 PASS。
//
// 把你 1.3 的 kernel 拆成两个装载函数移植进来:
//   load_manual:1.3 的逐 byte 手工装载(公式来自 1.1)
//   load_ldsm:  用 ldmatrix 装载。A 是 fp8——ldmatrix 的元素是 16 个
//               原始 bit,不关心类型;1.1 附加问的打包方向在这里起作用。
//               变体(.x1/.x2/.x4、是否 .trans)自己从 PTX 文档选。
// 数据已在 smem(main 里先从 global 拷入),两条路径都从 smem 装载。
//
// 都 PASS 之后:make ptx/m1_sm80/04_ldmatrix 或 nvdisasm 反汇编,
// 数两条路径 smem->fragment 段的指令构成(装载条数、地址算术条数),
// 报告里回答:ldmatrix 消掉的是哪部分工作?为什么手工路径绕不开它?
//
// 运行:make run/m1_sm80/04_ldmatrix(内部两条路径各跑多 seed)
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdlib>
#include <random>
#include "../common.h"

// smem 布局:sA 按 [16][32] 行主序;B 备了两种布局——sBk 按 [32][8]
// (k-major,1.3 用的就是它),sBn 按 [8][32](n-major,每个 n 的 32
// 个 k 字节连续)。手工路径用哪种都行;ldmatrix 的每个"行地址"要求
// 16 byte 连续,B 的 fragment 需要 k 方向相邻的字节成对进 b16——
// 想清楚哪种布局能满足它。
//
// 手工路径沿用 1.1 的映射：逐字节读取，再按低位到高位打包成 b32。
__device__ __forceinline__ void load_manual(const uint8_t* sA, const uint8_t* sBk,
                            const uint8_t* sBn, unsigned (&a)[4],
                            unsigned (&b)[2]) {
    (void)sBn;  // 手工路径选用原始的 B[k][n] 布局。
    int lane = threadIdx.x & 31;
    int gid = lane >> 2, tig = lane & 3;

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        a[r] = 0;
        int row = gid + 8 * (r & 1);
        int col = 4 * tig + 16 * (r >> 1);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a[r] |= static_cast<unsigned>(sA[row * 32 + col + j]) << (8 * j);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        b[r] = 0;
        int k = 4 * tig + 16 * r;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            b[r] |= static_cast<unsigned>(sBk[(k + j) * 8 + gid]) << (8 * j);
    }
}

// 下列偏移属于“提供小矩阵行首地址的线程”，不是该线程最终取到的元素。
// A 视作 16x16 个 b16，每个 b16 包含两个相邻的 FP8 编码。
// lane 0..7 / 8..15 / 16..23 / 24..31 分别提供左上/左下/右上/右下
// 四个 8x8 b16 小矩阵的行首，对应 a[0] / a[1] / a[2] / a[3]。
__device__ __forceinline__ unsigned ldsm_a_offset(unsigned lane) {
    return (lane & 15) * 32 + (lane >> 4) * 16;
}

// sBn[n][k] 使同一列 B 的 k 元素连续，视作 8x16 个 b16。
// lane 0..7 给出 k=0..15 的行首，lane 8..15 给出 k=16..31 的行首。
// 高 16 个 lane 重复低 16 个 lane 的有效地址；.x2 不使用这些额外地址。
__device__ __forceinline__ unsigned ldsm_b_offset(unsigned lane) {
    return (lane & 7) * 32 + ((lane >> 3) & 1) * 16;
}

__device__ __forceinline__ void load_ldsm(const uint8_t* sA, const uint8_t* sBk,
                          const uint8_t* sBn, unsigned (&a)[4],
                          unsigned (&b)[2]) {
    (void)sBk;  // 此路径使用已经转置存放的 sBn，不再使用 .trans。
    unsigned lane = threadIdx.x & 31;
    unsigned addr_a = static_cast<unsigned>(
        __cvta_generic_to_shared(sA + ldsm_a_offset(lane)));
    unsigned addr_b = static_cast<unsigned>(
        __cvta_generic_to_shared(sBn + ldsm_b_offset(lane)));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr_a)
        : "memory");
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr_b)
        : "memory");
}

template <bool USE_LDSM>
__global__ void mma_kernel(const uint8_t* A, const uint8_t* B, float* D) {
    // b16 的 8 元素行占 16 字节，显式保证 ldmatrix 行首地址的对齐。
    __shared__ __align__(16) uint8_t sA[16 * 32];
    __shared__ __align__(16) uint8_t sBk[32 * 8];
    __shared__ __align__(16) uint8_t sBn[8 * 32];
    for (int i = threadIdx.x; i < 16 * 32; i += 32) sA[i] = A[i];
    for (int i = threadIdx.x; i < 32 * 8; i += 32) {
        sBk[i] = B[i];
        sBn[(i & 7) * 32 + (i >> 3)] = B[i];  // 转成 n-major
    }
    __syncwarp();
    unsigned a[4], b[2];
    if constexpr (USE_LDSM)
        load_ldsm(sA, sBk, sBn, a, b);
    else
        load_manual(sA, sBk, sBn, a, b);
    float c[4] = {0, 0, 0, 0}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    int group = threadIdx.x >> 2, tig = threadIdx.x & 3;
    D[group * 8 + tig * 2] = d[0];
    D[group * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

static int run_path(bool ldsm, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8], ref[16 * 8] = {};
    for (int i = 0; i < 16 * 32; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hA[i] = *(uint8_t*)&v;
        fA[i] = float(v);
    }
    for (int i = 0; i < 32 * 8; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hB[i] = *(uint8_t*)&v;
        fB[i] = float(v);
    }
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 32; k++)
                ref[r * 8 + n] += fA[r * 32 + k] * fB[k * 8 + n];
    uint8_t *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    if (ldsm)
        mma_kernel<true><<<1, 32>>>(dA, dB, dD);
    else
        mma_kernel<false><<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16 * 8; i++) {
        if (got[i] != ref[i]) {
            if (bad < 5)
                fprintf(stderr, "%s seed=%u D[%d][%d]: got %.0f, want %.0f\n",
                        ldsm ? "ldsm" : "manual", seed, i / 8, i % 8,
                        got[i], ref[i]);
            ++bad;
        }
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad;
}

int main() {
    long total = 0;
    for (unsigned s : {1u, 7u, 42u}) {
        int bm = run_path(false, s), bl = run_path(true, s);
        printf("seed=%-6u manual %s(%d)  ldsm %s(%d)\n", s,
               bm ? "FAIL" : "PASS", bm, bl ? "FAIL" : "PASS", bl);
        total += bm + bl;
    }
    return total != 0;
}
