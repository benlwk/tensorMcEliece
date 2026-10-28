#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

/* Portability helpers shared by all harnesses. */

#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <cuda_runtime.h>

#ifndef N_TRIALS
#define N_TRIALS 30
#endif

/* CUDA reserves the per-thread stack limit for every thread that can be
 * resident on the device, so a fixed large limit (e.g. 128 KB) exhausts the
 * memory of GPUs with many SMs. Size the limit from what the kernels actually
 * need (cudaFuncAttributes::localSizeBytes) plus a 25% + 4 KB margin. */
static inline void bench_set_stack_limit(std::initializer_list<const void *> kernels)
{
    size_t need = 0;
    for (const void *k : kernels) {
        cudaFuncAttributes a;
        cudaError_t e = cudaFuncGetAttributes(&a, k);
        if (e != cudaSuccess) {
            fprintf(stderr, "[Host] cudaFuncGetAttributes failed: %s\n", cudaGetErrorString(e));
            exit(1);
        }
        if (a.localSizeBytes > need) need = a.localSizeBytes;
    }
    size_t limit = (need + need / 4 + 4096 + 1023) & ~(size_t)1023;
    cudaError_t e = cudaDeviceSetLimit(cudaLimitStackSize, limit);
    size_t got = 0;
    cudaDeviceGetLimit(&got, cudaLimitStackSize);
    printf("[Host] per-thread stack: kernels need %zu B, limit set to %zu B\n", need, got);
    if (e != cudaSuccess || got < need) {
        fprintf(stderr, "[Host] cudaDeviceSetLimit(stack=%zu) failed: %s\n", limit, cudaGetErrorString(e));
        exit(1);
    }
}

/* Abort on any pending launch/execution error, so a kernel that fails to
 * launch can never be timed or reported as a result. */
static inline void bench_check(const char *what)
{
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[Host] CUDA error after %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static inline void bench_print_device(void)
{
    cudaDeviceProp p;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&p, dev);
    printf("[Host] GPU: %s  SMs: %d  CC: %d.%d  N_TRIALS=%d\n",
           p.name, p.multiProcessorCount, p.major, p.minor, N_TRIALS);
}

#endif
