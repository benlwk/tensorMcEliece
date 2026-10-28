#ifndef INT8_TENSOR_ENC_348864_H
#define INT8_TENSOR_ENC_348864_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "params.h"
#include "shake256.h"
#include "util.h"
#include "encap_device.h"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define INT8_K_PADDED ((PK_NCOLS + 15) / 16 * 16)
#define INT8_M_PADDED ((PK_NROWS + 15) / 16 * 16)

static inline void pack_pk_int8_wmma(int8_t *h_pk_int8, const unsigned char *h_pk) {
    memset(h_pk_int8, 0, (size_t)INT8_M_PADDED * INT8_K_PADDED * sizeof(int8_t));
    for (int r = 0; r < PK_NROWS; r++) {
        const unsigned char *src_row = h_pk + (size_t)r * PK_ROW_BYTES;
        int8_t *dst_row = h_pk_int8 + (size_t)r * INT8_K_PADDED;
        for (int c = 0; c < PK_NCOLS; c++) {
            dst_row[c] = (src_row[c >> 3] >> (c & 7)) & 1;
        }
    }
}

__global__ void kernel_gen_e_and_pack_int8_wmma(
    unsigned char *d_e_all,
    int8_t        *d_E_int8,
    const unsigned char *d_seeds,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    unsigned char e[SYS_N / 8];
    gen_e(e, d_seeds + (size_t)idx * 32);

    unsigned char *e_dst = d_e_all + (size_t)idx * (SYS_N / 8);
    for (int i = 0; i < SYS_N / 8; i++) e_dst[i] = e[i];

    const unsigned char *e_ns = e + SYND_BYTES;
    for (int k = 0; k < INT8_K_PADDED; k++) {
        int8_t bit_val = 0;
        if (k < PK_NCOLS) {
            bit_val = (e_ns[k >> 3] >> (k & 7)) & 1;
        }
        d_E_int8[(size_t)k * N + idx] = bit_val;
    }
}

__global__ void kernel_int8_wmma_syndrome_gemm(
    unsigned char *d_S_bytes,
    const int8_t  *d_T_int8,
    const int8_t  *d_E_int8,
    int N
) {
    int warp_m = blockIdx.y * WMMA_M;
    int warp_n = blockIdx.x * WMMA_N;

    if (warp_m >= PK_NROWS || warp_n >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    wmma::fill_fragment(c_frag, 0);

    int k_tiles = INT8_K_PADDED / WMMA_K;

    for (int k = 0; k < k_tiles; k++) {
        const int8_t *a_ptr = d_T_int8 + (size_t)warp_m * INT8_K_PADDED + k * WMMA_K;
        const int8_t *b_ptr = d_E_int8 + (size_t)(k * WMMA_K) * N + warp_n;

        wmma::load_matrix_sync(a_frag, a_ptr, INT8_K_PADDED);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ int store_tile[WMMA_M][WMMA_N];

    wmma::store_matrix_sync(&store_tile[0][0], c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    int lane_id = threadIdx.x;
    for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
        int r = warp_m + i / WMMA_N;
        int c = warp_n + i % WMMA_N;
        if (r < PK_NROWS && c < N) {
            int sum_val = store_tile[i / WMMA_N][i % WMMA_N];
            d_S_bytes[(size_t)r * N + c] = (unsigned char)(sum_val & 1);
        }
    }
}

#endif
