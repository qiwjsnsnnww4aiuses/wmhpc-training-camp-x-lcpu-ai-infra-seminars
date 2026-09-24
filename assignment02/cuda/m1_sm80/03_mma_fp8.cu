// 问题 1.3：一个 warp 手动装载 FP8 fragment，执行 MMA，并与 CPU 对拍。
// D[16x8] = A[16x32] * B[32x8]，C=0；输入 E4M3，累加和输出 FP32。
// 程序接受一个 seed；省略时默认 seed=1，方便使用 Makefile 的 run 目标。
#include <cuda_fp8.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <random>
#include "../common.h"

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 32;

// 沿用 1.1 的映射：i 是线程内的元素编号，r 是寄存器编号，j 是字节编号。
__host__ __device__ static int a_row_of(int lane, int i) {
    int gid = lane >> 2;
    int r = i >> 2;
    return gid + 8 * (r & 1);
}

__host__ __device__ static int a_col_of(int lane, int i) {
    int tig = lane & 3;
    int r = i >> 2, j = i & 3;
    return (tig << 2) + 16 * (r >> 1) + j;
}

__host__ __device__ static int b_row_of(int lane, int i) {
    int tig = lane & 3;
    int r = i >> 2, j = i & 3;
    return (tig << 2) + 16 * (r & 1) + j;
}

__host__ __device__ static int b_col_of(int lane, int /*i*/) {
    return lane >> 2;
}

// A、B 都按行主序存放。每个 b32 从低到高装 4 个 FP8 的原始编码。
// __x 是 FP8 类型提供的存储字段；不能用数值到整数的转换替代它。
__host__ __device__ static void load_fragments(
    const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, int lane,
    unsigned (&a)[4], unsigned (&b)[2]) {
#pragma unroll
    for (int r = 0; r < 4; ++r) a[r] = 0;
#pragma unroll
    for (int r = 0; r < 2; ++r) b[r] = 0;

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int row = a_row_of(lane, i), col = a_col_of(lane, i);
        unsigned bits = static_cast<unsigned>(A[row * K + col].__x);
        a[i >> 2] |= bits << (8 * (i & 3));
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int row = b_row_of(lane, i), col = b_col_of(lane, i);
        unsigned bits = static_cast<unsigned>(B[row * N + col].__x);
        b[i >> 2] |= bits << (8 * (i & 3));
    }
}

__global__ void mma_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* D) {
    int lane = threadIdx.x;  // 固定启动一个 block、32 个线程。
    unsigned a[4], b[2];
    load_fragments(A, B, lane, a, b);

    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    // 输出仍然是 16x8 的 FP32 矩阵，布局与 M0 相同。
    int gid = lane >> 2, tig = lane & 3;
    D[gid * N + tig * 2] = d[0];
    D[gid * N + tig * 2 + 1] = d[1];
    D[(gid + 8) * N + tig * 2] = d[2];
    D[(gid + 8) * N + tig * 2 + 1] = d[3];
}

static bool parse_seed(const char* text, unsigned& seed) {
    // 限定为十进制非负整数，拒绝空串、负数和尾随字符。
    if (*text == '\0') return false;
    for (const char* p = text; *p; ++p)
        if (*p < '0' || *p > '9') return false;
    errno = 0;
    char* end = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || value > UINT_MAX) return false;
    seed = static_cast<unsigned>(value);
    return true;
}

int main(int argc, char* argv[]) {
    unsigned seed = 1;
    if (argc > 2 || (argc == 2 && !parse_seed(argv[1], seed))) {
        std::printf("FAIL: expected an unsigned integer seed\n");
        std::fprintf(stderr, "Usage: %s [seed]\n", argv[0]);
        return 1;
    }

    __nv_fp8_e4m3 hA[M * K];
    __nv_fp8_e4m3 hB[K * N];

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-2, 2);

    for(int i = 0; i < M * K; ++i) {
        float value = static_cast<float>(dist(rng));
        hA[i] = __nv_fp8_e4m3(value);
    }
    for(int i = 0; i < K * N; ++i) {
        float value = static_cast<float>(dist(rng));
        hB[i] = __nv_fp8_e4m3(value);
    }
    // 从实际 FP8 输入转换回 float，保证 CPU 和 GPU 使用同一份数据。
    // 小整数的乘积与这里的累加结果都可精确表示，因此使用严格相等判测。
    float ref[M * N] = {};
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col)
            for (int k = 0; k < K; ++k)
                ref[row * N + col] += static_cast<float>(hA[row * K + k]) *
                                     static_cast<float>(hB[k * N + col]);

    __nv_fp8_e4m3 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(ref)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));

    mma_fp8<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[M * N];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dD));

    int bad = 0;
    for (int i = 0; i < M * N; ++i) {
        if (got[i] != ref[i]) {
            if (bad < 5)
                std::fprintf(stderr, "D[%d][%d]: got %.0f, want %.0f\n",
                             i / N, i % N, got[i], ref[i]);
            ++bad;
        }
    }

    // judge 要求 stdout 以 PASS 或 FAIL/MISMATCH 开头。
    if (bad) {
        std::printf("MISMATCH seed=%u: %d / %d elements\n", seed, bad, M * N);
        return 1;
    }
    std::printf("PASS seed=%u\n", seed);
    return 0;
}
