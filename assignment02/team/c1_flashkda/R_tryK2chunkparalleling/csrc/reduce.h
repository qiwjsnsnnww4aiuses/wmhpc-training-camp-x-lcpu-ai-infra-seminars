#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void flash_kda_prepare_launch(
    const void* q,
    const void* k,
    const void* g,
    const void* beta_t,
    void* workspace,
    int total_tiles,
    int T_total,
    int H,
    int N,
    const int64_t* cu_seqlens,
    const float* a_log,
    const float* dt_bias,
    float scale,
    float gate_scale,
    cudaStream_t stream);

void flash_kda_recurrence_launch(
    const void* v,
    const void* beta_t,
    const void* workspace,
    void* out,
    const void* initial_state,
    void* final_state,
    int total_tiles,
    int T_total,
    int H,
    int N,
    const int64_t* cu_seqlens,
    cudaStream_t stream);
