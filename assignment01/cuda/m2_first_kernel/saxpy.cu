// saxpy.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============ 1. 自己写错误检查宏 ============
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#define CUDA_CHECK_KERNEL() \
    do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

// ============ 2. 自己写 GpuTimer ============
struct GpuTimer {
    cudaEvent_t start_, stop_;
    
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() { 
        CUDA_CHECK(cudaEventRecord(start_)); 
    }
    
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============ 3. Kernel ============
__global__
void saxpy_kernel(int n, float a, float *x, float *y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

// ============ 4. main ============
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    if (n < 0) {
        fprintf(stderr, "n must be >= 0\n");
        return 1;
    }
    
    // 特殊处理 n = 0
    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }
    
    // ====== 5. 分配 host 内存并生成数据 ======
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512;
    }
    
    // ====== 6. 分配 device 内存并拷贝 ======
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // ====== 7. 启动 kernel 并计时 ======
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    GpuTimer timer;
    timer.start();
    saxpy_kernel<<<blocks, threads>>>(n, 2.0f, d_x, d_y);
    CUDA_CHECK_KERNEL();
    float ms = timer.stop_ms();
    
    // ====== 8. 拷回结果 ======
    CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // ====== 9. 用 double 累加并输出 ======
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += h_y[i];
    }
    printf("SUM=%.0f\n", sum);
    
    // ====== 10. 清理 ======
    free(h_x);
    free(h_y);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    return 0;
}
