#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void bmma_m8n8k128_and_popc(
    unsigned int &d0, unsigned int &d1,
    unsigned int a,
    unsigned int b,
    unsigned int c0, unsigned int c1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a), "r"(b), "r"(c0), "r"(c1)
    );
}

__global__ void test_bmma_kernel(
    const uint32_t *d_A,
    const uint32_t *d_B,
    uint32_t *d_D
) {
    int lane = threadIdx.x;
    if (lane >= 32) return;

    int r_A = lane / 4;
    int c_A = lane % 4;
    uint32_t a_val = d_A[r_A * 4 + c_A];

    int col_B = lane / 4;
    int r_B   = lane % 4;
    uint32_t b_val = d_B[col_B * 4 + r_B];

    uint32_t d0 = 0, d1 = 0;
    bmma_m8n8k128_and_popc(d0, d1, a_val, b_val, 0, 0);

    int r_out = lane / 4;
    int c_out0 = 2 * (lane % 4);
    int c_out1 = 2 * (lane % 4) + 1;

    d_D[r_out * 8 + c_out0] = d0;
    d_D[r_out * 8 + c_out1] = d1;
}

int main(void) {
    uint32_t h_A[8][4];
    uint32_t h_B[8][4];
    uint32_t h_D_expected[8][8];
    uint32_t h_D_gpu[8][8];

    for (int i = 0; i < 8; i++) {
        for (int k = 0; k < 4; k++) {
            h_A[i][k] = (i + 1) * 0x12345678 + k * 0x9abcdef0;
        }
    }
    for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 4; k++) {
            h_B[j][k] = (j + 1) * 0x0f0f0f0f + k * 0x55555555;
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            uint32_t popc_sum = 0;
            for (int k = 0; k < 4; k++) {
                uint32_t and_val = h_A[i][k] & h_B[j][k];
                popc_sum += __builtin_popcount(and_val);
            }
            h_D_expected[i][j] = popc_sum;
        }
    }

    uint32_t *d_A, *d_B, *d_D;
    cudaMalloc(&d_A, sizeof(h_A));
    cudaMalloc(&d_B, sizeof(h_B));
    cudaMalloc(&d_D, sizeof(h_D_gpu));

    cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice);

    test_bmma_kernel<<<1, 32>>>(d_A, d_B, d_D);
    cudaDeviceSynchronize();

    cudaMemcpy(h_D_gpu, d_D, sizeof(h_D_gpu), cudaMemcpyDeviceToHost);

    int pass = 1;
    printf("BMMA test results (Expected vs GPU):\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("[%d,%d]: CPU=%u GPU=%u (GF2=%u vs %u)\n",
                   i, j, h_D_expected[i][j], h_D_gpu[i][j],
                   h_D_expected[i][j] & 1, h_D_gpu[i][j] & 1);
            if (h_D_expected[i][j] != h_D_gpu[i][j]) {
                pass = 0;
            }
        }
    }

    if (pass) {
        printf("\n>>> SUCCESS: BMMA PTX instruction matches CPU reference perfectly! <<<\n");
    } else {
        printf("\n>>> FAILURE: Mismatch detected! <<<\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    return pass ? 0 : 1;
}
