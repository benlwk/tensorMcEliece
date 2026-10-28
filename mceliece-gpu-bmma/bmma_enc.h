#ifndef BMMA_ENC_H
#define BMMA_ENC_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "params.h"
#include "shake256.h"
#include "util.h"
#include "encap_device.h"

#ifndef PK_ROW_BITS
#define PK_ROW_BITS   2720
#endif

#define K_BITS_PADDED 2816
#define K_TILES       22
#define M_TILES       96

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

static inline void pack_pk_bmma(uint32_t *h_pk_bmma, const unsigned char *h_pk) {
    memset(h_pk_bmma, 0, (size_t)PK_NROWS * (K_BITS_PADDED / 32) * sizeof(uint32_t));
    for (int r = 0; r < PK_NROWS; r++) {
        const unsigned char *src_row = h_pk + r * PK_ROW_BYTES;
        uint32_t *dst_row = h_pk_bmma + r * (K_BITS_PADDED / 32);
        memcpy(dst_row, src_row, PK_ROW_BYTES);
    }
}

__global__ void kernel_gen_e_and_pack(
    unsigned char *d_e_all,
    uint32_t      *d_E_bmma,
    const unsigned char *d_seeds,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    unsigned char *e = d_e_all + (size_t)idx * (SYS_N / 8);
    gen_e(e, d_seeds + (size_t)idx * 32);

    const unsigned char *e_ns = e + SYND_BYTES;

    for (int k = 0; k < K_BITS_PADDED / 32; k++) {
        uint32_t val = 0;
        for (int b = 0; b < 4; b++) {
            int byte_idx = k * 4 + b;
            if (byte_idx < PK_ROW_BYTES) {
                val |= ((uint32_t)e_ns[byte_idx]) << (b * 8);
            }
        }
        d_E_bmma[(size_t)k * N + idx] = val;
    }
}

#define BM_ROWS 32
#define BN_COLS 32

__global__ void kernel_bmma_syndrome_gemm(
    uint32_t *d_S_gemm,
    const uint32_t *d_T_bmma,
    const uint32_t *d_E_bmma,
    int N
) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    int block_m = blockIdx.y * BM_ROWS;
    int block_n = blockIdx.x * BN_COLS;

    if (block_n >= N) return;

    int n_tile_idx = warp_id;
    int col_base = block_n + n_tile_idx * 8;

    uint32_t acc_d0[4] = {0, 0, 0, 0};
    uint32_t acc_d1[4] = {0, 0, 0, 0};

    int col_B = lane_id / 4;
    int k_sub_B = lane_id % 4;
    int actual_col_E = col_base + col_B;

    int r_A_sub = lane_id / 4;
    int k_sub_A = lane_id % 4;

    #pragma unroll
    for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
        int k_word_B = k_tile * 4 + k_sub_B;
        uint32_t b_val = (actual_col_E < N) ? d_E_bmma[(size_t)k_word_B * N + actual_col_E] : 0u;

        #pragma unroll
        for (int m_sub = 0; m_sub < 4; m_sub++) {
            int row_T = block_m + m_sub * 8 + r_A_sub;
            int k_word_A = k_tile * 4 + k_sub_A;
            uint32_t a_val = (row_T < PK_NROWS) ? d_T_bmma[(size_t)row_T * (K_BITS_PADDED / 32) + k_word_A] : 0u;

            bmma_m8n8k128_and_popc(acc_d0[m_sub], acc_d1[m_sub], a_val, b_val, acc_d0[m_sub], acc_d1[m_sub]);
        }
    }

    int r_sub = lane_id / 4;
    int c_offset0 = 2 * (lane_id % 4);
    int c_offset1 = 2 * (lane_id % 4) + 1;

    int c0 = col_base + c_offset0;
    int c1 = col_base + c_offset1;

    for (int m_sub = 0; m_sub < 4; m_sub++) {
        int r = block_m + m_sub * 8 + r_sub;
        if (r < PK_NROWS) {
            uint32_t bit0 = acc_d0[m_sub] & 1u;
            uint32_t bit1 = acc_d1[m_sub] & 1u;

            unsigned char *d_S_bytes = (unsigned char *)d_S_gemm;
            if (c0 < N) d_S_bytes[(size_t)r * N + c0] = (unsigned char)bit0;
            if (c1 < N) d_S_bytes[(size_t)r * N + c1] = (unsigned char)bit1;
        }
    }
}

__global__ void kernel_finalize_encap(
    unsigned char       *d_cs,
    unsigned char       *d_keys,
    const unsigned char *d_S_bytes,
    const unsigned char *d_e_all,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const unsigned char *e = d_e_all + (size_t)idx * (SYS_N / 8);
    unsigned char *c       = d_cs   + (size_t)idx * SYND_BYTES;
    unsigned char *key     = d_keys + (size_t)idx * KEY_SIZE;

    for (int byte_i = 0; byte_i < SYND_BYTES; byte_i++) {
        unsigned char val = 0;
        for (int b = 0; b < 8; b++) {
            int bit_idx = byte_i * 8 + b;
            unsigned char gemm_bit = d_S_bytes[(size_t)bit_idx * N + idx];
            unsigned char e_sys_bit = (e[byte_i] >> b) & 1;
            val |= ((gemm_bit ^ e_sys_bit) & 1) << b;
        }
        c[byte_i] = val;
    }

    unsigned char one_ec[1 + SYS_N / 8 + SYND_BYTES];
    one_ec[0] = 1;
    for (int i = 0; i < SYS_N / 8; i++)  one_ec[1 + i]            = e[i];
    for (int i = 0; i < SYND_BYTES; i++) one_ec[1 + SYS_N / 8 + i] = c[i];
    crypto_hash_32b(key, one_ec, sizeof(one_ec));
}

#endif
