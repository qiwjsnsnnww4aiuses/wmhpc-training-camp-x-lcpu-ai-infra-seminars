#pragma once

#include <cstdint>

// Minimal SM100/Blackwell tcgen05 helpers used by the experimental K2 path.
// The code intentionally uses inline PTX so that the generated SASS can be
// audited independently of CUTLASS/CuTe's instruction selection.

__host__ __device__ __forceinline__ int flash_kda_tcgen05_swizzle_128b(
    int row, int col_byte) {
    const int atom = row >> 3;
    const int row_in_atom = row & 7;
    const int chunk = col_byte >> 4;
    const int in_16b = col_byte & 15;
    return atom * 1024 + row_in_atom * 128 +
           ((chunk ^ row_in_atom) << 4) + in_16b;
}

__device__ __forceinline__ uint64_t flash_kda_tcgen05_make_desc(
    uint32_t shared_addr, uint32_t leading_byte, uint32_t stride_byte,
    uint32_t layout) {
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((shared_addr >> 4) & 0x3fff);
    desc |= static_cast<uint64_t>((leading_byte >> 4) & 0x3fff) << 16;
    desc |= static_cast<uint64_t>((stride_byte >> 4) & 0x3fff) << 32;
    desc |= static_cast<uint64_t>(1) << 46;
    desc |= static_cast<uint64_t>(layout) << 61;
    return desc;
}

__device__ __forceinline__ void flash_kda_tcgen05_wait(
    uint32_t mbar_addr, uint32_t phase) {
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

__device__ __forceinline__ uint32_t flash_kda_tcgen05_elect_one() {
    uint32_t elected;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xffffffff;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}"
        : "=r"(elected));
    return elected;
}

__device__ __forceinline__ void flash_kda_tcgen05_issue(
    uint32_t tmem_addr, uint64_t a_desc, uint64_t b_desc,
    uint32_t mbar_addr, bool accumulate) {
    // f32 accumulator, bf16 A/B, M=N=128, K=16.
    constexpr uint32_t instruction_desc =
        (1u << 4) | (1u << 7) | (1u << 10) |
        (16u << 17) | (8u << 24);
    asm volatile(
        "{\n"
        ".reg .pred p_acc;\n"
        "setp.ne.b32 p_acc, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 "
        "[%0], %1, %2, %3, p_acc;\n"
        "}\n"
        :
        : "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
          "r"(instruction_desc), "r"(accumulate ? 1 : 0));
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :
        : "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__ void flash_kda_tcgen05_load_x8(
    uint32_t src, float (&result)[8]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(result[0]), "=f"(result[1]), "=f"(result[2]),
          "=f"(result[3]), "=f"(result[4]), "=f"(result[5]),
          "=f"(result[6]), "=f"(result[7])
        : "r"(src));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}
