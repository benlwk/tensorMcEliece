#ifndef INT8_TENSOR_BS2_ENC_H
#define INT8_TENSOR_BS2_ENC_H

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

#define BS2_WMMA_M 16
#define BS2_WMMA_N 16
#define BS2_WMMA_K 16

#define BS2_K_PADDED ((PK_NCOLS + 15) / 16 * 16)
#define BS2_M_PADDED ((PK_NROWS + 15) / 16 * 16)

#define BS2_PAIRS_PADDED(_N) ((((_N) / 2) + 15) / 16 * 16)

#define BS2_GROUPS_PADDED(_N) ((((_N) / 4) + 15) / 16 * 16)

__global__ void kernel_gen_e_and_pack_int8_bs2(
    unsigned char *d_e_all,
    int8_t        *d_E_bs2,
    const unsigned char *d_seeds,
    int N,
    int pairs_padded
) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= N / 2) return;

    int j0 = 2 * pair;
    int j1 = 2 * pair + 1;

    const int nb = pair >> 4;
    const int pr = pair & 15;
    const size_t tile_base = (size_t)nb * (BS2_K_PADDED * 16);
    const size_t ldm = 16;

    unsigned char e0[SYS_N / 8];
    unsigned char e1[SYS_N / 8];

    gen_e(e0, d_seeds + (size_t)j0 * 32);
    if (j1 < N) gen_e(e1, d_seeds + (size_t)j1 * 32);

    unsigned char *dst0 = d_e_all + (size_t)j0 * (SYS_N / 8);
    for (int i = 0; i < SYS_N / 8; i++) dst0[i] = e0[i];
    if (j1 < N) {
        unsigned char *dst1 = d_e_all + (size_t)j1 * (SYS_N / 8);
        for (int i = 0; i < SYS_N / 8; i++) dst1[i] = e1[i];
    }

    const unsigned char *e0_ns = e0 + SYND_BYTES;
    const unsigned char *e1_ns = e1 + SYND_BYTES;

    for (int k = 0; k < BS2_K_PADDED; k++) {
        uint8_t b0 = 0, b1 = 0;
        if (k < PK_NCOLS) {
            b0 = (uint8_t)((e0_ns[k >> 3] >> (k & 7)) & 1);
            b1 = (j1 < N) ? (uint8_t)((e1_ns[k >> 3] >> (k & 7)) & 1) : 0;
        }
        d_E_bs2[tile_base + (size_t)k * ldm + pr] = (int8_t)(b0 | (b1 << 4));
    }
}

__global__ void kernel_int8_wmma_bs2_syndrome_gemm(
    unsigned char *d_S_bytes,
    const int8_t  *d_T_int8,
    const int8_t  *d_E_bs2,
    int N,
    int pairs_padded
) {
    int warp_m = blockIdx.y * BS2_WMMA_M;
    int warp_n = blockIdx.x * BS2_WMMA_N;
    if (warp_m >= PK_NROWS || warp_n >= pairs_padded) return;

    wmma::fragment<wmma::matrix_a, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_even, c_odd;

    wmma::fill_fragment(c_even, 0);
    wmma::fill_fragment(c_odd, 0);

    int k_tiles = BS2_K_PADDED / BS2_WMMA_K;

    for (int k = 0; k < k_tiles; k++) {
        const int8_t *a_ptr = d_T_int8 + (size_t)warp_m * BS2_K_PADDED + k * BS2_WMMA_K;

        const int8_t *b_ptr = d_E_bs2 + (size_t)blockIdx.x * (BS2_K_PADDED * 16)
                                     + (size_t)(k * BS2_WMMA_K) * 16;

        wmma::load_matrix_sync(a_frag, a_ptr, BS2_K_PADDED);
        wmma::load_matrix_sync(b_frag, b_ptr, 16);

        wmma::mma_sync(c_even, a_frag, b_frag, c_even);

        #pragma unroll
        for (int e = 0; e < b_frag.num_elements; e++) {
            uint8_t x = (uint8_t)b_frag.x[e];
            b_frag.x[e] = (int8_t)(((x & 0x0F) << 4) | ((x >> 4) & 0x0F));
        }
        wmma::mma_sync(c_odd, a_frag, b_frag, c_odd);
    }

    __shared__ int store_even[BS2_WMMA_M][BS2_WMMA_N];
    __shared__ int store_odd [BS2_WMMA_M][BS2_WMMA_N];

    wmma::store_matrix_sync(&store_even[0][0], c_even, BS2_WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(&store_odd[0][0],  c_odd,  BS2_WMMA_N, wmma::mem_row_major);
    __syncthreads();

    int lane_id = threadIdx.x;
    for (int i = lane_id; i < BS2_WMMA_M * BS2_WMMA_N; i += 32) {
        int r = warp_m + i / BS2_WMMA_N;
        int p = warp_n + i % BS2_WMMA_N;
        int c0 = 2 * p;
        int c1 = 2 * p + 1;
        if (r < PK_NROWS && c1 < N) {
            d_S_bytes[(size_t)r * N + c0] = (unsigned char)(store_even[i / BS2_WMMA_N][i % BS2_WMMA_N] & 1);
            d_S_bytes[(size_t)r * N + c1] = (unsigned char)(store_odd [i / BS2_WMMA_N][i % BS2_WMMA_N] & 1);
        }
    }
}

#define BS2_M_CHUNK 4

__global__ void kernel_int8_wmma_bs2_m4_syndrome_gemm(
    unsigned char *d_S_bytes,
    const int8_t  *d_T_int8,
    const int8_t  *d_E_bs2,
    int N,
    int pairs_padded
) {
    const int warp_m_base = blockIdx.y * (BS2_M_CHUNK * 16);
    const int tile_base   = blockIdx.x * (BS2_K_PADDED * 16);
    if (blockIdx.x * 16 >= pairs_padded) return;

    wmma::fragment<wmma::matrix_a, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> b_frag;
    wmma::fragment<wmma::matrix_b, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> b_frag_swap;
    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_even[BS2_M_CHUNK];
    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_odd [BS2_M_CHUNK];

    #pragma unroll
    for (int ml = 0; ml < BS2_M_CHUNK; ml++) {
        wmma::fill_fragment(c_even[ml], 0);
        wmma::fill_fragment(c_odd [ml], 0);
    }

    int k_tiles = BS2_K_PADDED / BS2_WMMA_K;

    for (int k = 0; k < k_tiles; k++) {

        const int8_t *b_ptr = d_E_bs2 + tile_base + (size_t)(k * BS2_WMMA_K) * 16;
        wmma::load_matrix_sync(b_frag, b_ptr, 16);

        #pragma unroll
        for (int e = 0; e < b_frag.num_elements; e++) {
            uint8_t x = (uint8_t)b_frag.x[e];
            b_frag_swap.x[e] = (int8_t)(((x & 0x0F) << 4) | ((x >> 4) & 0x0F));
        }

        #pragma unroll
        for (int ml = 0; ml < BS2_M_CHUNK; ml++) {
            const int m_row = warp_m_base + ml * BS2_WMMA_M;
            if (m_row >= BS2_M_PADDED) continue;
            const int8_t *a_ptr = d_T_int8 + (size_t)m_row * BS2_K_PADDED + k * BS2_WMMA_K;
            wmma::load_matrix_sync(a_frag, a_ptr, BS2_K_PADDED);
            wmma::mma_sync(c_even[ml], a_frag, b_frag,      c_even[ml]);
            wmma::mma_sync(c_odd [ml], a_frag, b_frag_swap, c_odd [ml]);
        }
    }

    __shared__ int store_even[BS2_WMMA_M][BS2_WMMA_N];
    __shared__ int store_odd [BS2_WMMA_M][BS2_WMMA_N];

    #pragma unroll
    for (int ml = 0; ml < BS2_M_CHUNK; ml++) {
        const int r0 = warp_m_base + ml * BS2_WMMA_M;
        if (r0 >= BS2_M_PADDED) continue;
        wmma::store_matrix_sync(&store_even[0][0], c_even[ml], BS2_WMMA_N, wmma::mem_row_major);
        wmma::store_matrix_sync(&store_odd[0][0],  c_odd [ml], BS2_WMMA_N, wmma::mem_row_major);
        __syncthreads();
        for (int i = threadIdx.x; i < BS2_WMMA_M * BS2_WMMA_N; i += 32) {
            int r = r0 + i / BS2_WMMA_N;
            int p = blockIdx.x * BS2_WMMA_N + i % BS2_WMMA_N;
            int c0 = 2 * p;
            int c1 = 2 * p + 1;
            if (r < PK_NROWS && c1 < N) {
                d_S_bytes[(size_t)r * N + c0] = (unsigned char)(store_even[i / BS2_WMMA_N][i % BS2_WMMA_N] & 1);
                d_S_bytes[(size_t)r * N + c1] = (unsigned char)(store_odd [i / BS2_WMMA_N][i % BS2_WMMA_N] & 1);
            }
        }
        __syncthreads();
    }
}

#define BS2_W4_CHUNK 2

__device__ __forceinline__ int8_t bs2_w4_rot(int8_t v, int sh) {
    uint8_t x = (uint8_t)v;
    return (int8_t)(uint8_t)((x >> sh) | (x << (8 - sh)));
}

__global__ void kernel_gen_e_and_pack_int8_bs2_w4(
    unsigned char *d_e_all,
    int8_t        *d_E_w4,
    const unsigned char *d_seeds,
    int N,
    int groups_padded
) {
    int grp = blockIdx.x * blockDim.x + threadIdx.x;
    if (grp >= N / 4) return;

    int j0 = 4 * grp, j1 = 4 * grp + 1, j2 = 4 * grp + 2, j3 = 4 * grp + 3;
    unsigned char e0[SYS_N / 8], e1[SYS_N / 8], e2[SYS_N / 8], e3[SYS_N / 8];

    gen_e(e0, d_seeds + (size_t)j0 * 32);
    gen_e(e1, d_seeds + (size_t)j1 * 32);
    gen_e(e2, d_seeds + (size_t)j2 * 32);
    gen_e(e3, d_seeds + (size_t)j3 * 32);

    unsigned char *e_dst = d_e_all + (size_t)j0 * (SYS_N / 8);
    for (int i = 0; i < SYS_N / 8; i++) e_dst[i] = e0[i];
    e_dst = d_e_all + (size_t)j1 * (SYS_N / 8);
    for (int i = 0; i < SYS_N / 8; i++) e_dst[i] = e1[i];
    e_dst = d_e_all + (size_t)j2 * (SYS_N / 8);
    for (int i = 0; i < SYS_N / 8; i++) e_dst[i] = e2[i];
    e_dst = d_e_all + (size_t)j3 * (SYS_N / 8);
    for (int i = 0; i < SYS_N / 8; i++) e_dst[i] = e3[i];

    const unsigned char *e0_ns = e0 + SYND_BYTES;
    const unsigned char *e1_ns = e1 + SYND_BYTES;
    const unsigned char *e2_ns = e2 + SYND_BYTES;
    const unsigned char *e3_ns = e3 + SYND_BYTES;

    const int nb = grp >> 4;
    const int pr = grp & 15;
    const size_t tile_base = (size_t)nb * (BS2_K_PADDED * 16);

    for (int k = 0; k < BS2_K_PADDED; k++) {
        uint8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        if (k < PK_NCOLS) {
            b0 = (uint8_t)((e0_ns[k >> 3] >> (k & 7)) & 1);
            b1 = (uint8_t)((e1_ns[k >> 3] >> (k & 7)) & 1);
            b2 = (uint8_t)((e2_ns[k >> 3] >> (k & 7)) & 1);
            b3 = (uint8_t)((e3_ns[k >> 3] >> (k & 7)) & 1);
        }
        d_E_w4[tile_base + (size_t)k * 16 + pr] = (int8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
    }
}

__global__ void kernel_int8_wmma_bs2_w4_syndrome_gemm(
    unsigned char *d_S_bytes,
    const int8_t  *d_T_int8,
    const int8_t  *d_E_w4,
    int N,
    int groups_padded
) {
    const int warp_m_base = blockIdx.y * (BS2_W4_CHUNK * 16);
    const int tile_base   = blockIdx.x * (BS2_K_PADDED * 16);
    if (blockIdx.x * 16 >= groups_padded) return;

    wmma::fragment<wmma::matrix_a, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c[BS2_W4_CHUNK][4];

    #pragma unroll
    for (int ml = 0; ml < BS2_W4_CHUNK; ml++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c[ml][j], 0);

    int k_tiles = BS2_K_PADDED / BS2_WMMA_K;

    for (int k = 0; k < k_tiles; k++) {
        const int8_t *b_ptr = d_E_w4 + tile_base + (size_t)(k * BS2_WMMA_K) * 16;
        wmma::load_matrix_sync(b_frag[0], b_ptr, 16);

        #pragma unroll
        for (int e = 0; e < b_frag[0].num_elements; e++) {
            int8_t x = b_frag[0].x[e];
            b_frag[1].x[e] = bs2_w4_rot(x, 2);
            b_frag[2].x[e] = bs2_w4_rot(x, 4);
            b_frag[3].x[e] = bs2_w4_rot(x, 6);
        }

        #pragma unroll
        for (int ml = 0; ml < BS2_W4_CHUNK; ml++) {
            const int m_row = warp_m_base + ml * BS2_WMMA_M;
            if (m_row >= BS2_M_PADDED) continue;
            const int8_t *a_ptr = d_T_int8 + (size_t)m_row * BS2_K_PADDED + k * BS2_WMMA_K;
            wmma::load_matrix_sync(a_frag, a_ptr, BS2_K_PADDED);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                wmma::mma_sync(c[ml][j], a_frag, b_frag[j], c[ml][j]);
        }
    }

    __shared__ int sbuf[BS2_WMMA_M][BS2_WMMA_N];

    #pragma unroll
    for (int ml = 0; ml < BS2_W4_CHUNK; ml++) {
        const int r0 = warp_m_base + ml * BS2_WMMA_M;
        if (r0 >= BS2_M_PADDED) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&sbuf[0][0], c[ml][j], BS2_WMMA_N, wmma::mem_row_major);
            __syncthreads();
            for (int i = threadIdx.x; i < BS2_WMMA_M * BS2_WMMA_N; i += 32) {
                int r = r0 + i / BS2_WMMA_N;
                int p = blockIdx.x * BS2_WMMA_N + i % BS2_WMMA_N;
                int col = 4 * p + j;
                if (r < PK_NROWS && col < N)
                    d_S_bytes[(size_t)r * N + col] = (unsigned char)(sbuf[i / BS2_WMMA_N][i % BS2_WMMA_N] & 1);
            }
            __syncthreads();
        }
    }
}

#define BS2_BLK_M_TILES 4
#define BS2_BLK_N_TILES 4

__global__ void kernel_int8_wmma_bs2_blk_syndrome_gemm(
    unsigned char *d_S_bytes,
    const int8_t  *d_T_int8,
    const int8_t  *d_E_bs2,
    int N,
    int pairs_padded
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int row_base  = blockIdx.y * (BS2_BLK_M_TILES * 16);
    const int pair_base = blockIdx.x * (BS2_BLK_N_TILES * 16);
    if (pair_base >= pairs_padded) return;

    __shared__ __align__(16) int8_t sA[BS2_BLK_M_TILES * 16][16];
    __shared__ __align__(16) int8_t sB[BS2_BLK_N_TILES][16][16];
    __shared__ __align__(16) int stage[4][2][16][16];

    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_even[2][2];
    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_odd [2][2];
    #pragma unroll
    for (int a = 0; a < 2; a++)
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            wmma::fill_fragment(c_even[a][b], 0);
            wmma::fill_fragment(c_odd [a][b], 0);
        }

    wmma::fragment<wmma::matrix_a, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> b_frag, b_swap;

    const int k_tiles = BS2_K_PADDED / BS2_WMMA_K;
    const int grp0    = pair_base >> 4;

    for (int k = 0; k < k_tiles; k++) {

        {
            int t = threadIdx.x;
            int r = t >> 1;
            int h = t & 1;
            const int8_t *src = d_T_int8 + (size_t)(row_base + r) * BS2_K_PADDED + k * BS2_WMMA_K + h * 8;
            *reinterpret_cast<uint64_t *>(&sA[r][h * 8]) = *reinterpret_cast<const uint64_t *>(src);
        }

        {
            int idx8 = threadIdx.x * 8;
            int g    = idx8 >> 8;
            int off  = idx8 & 255;
            const int8_t *src = d_E_bs2 + (size_t)(grp0 + g) * (BS2_K_PADDED * 16) + k * (BS2_WMMA_K * 16) + off;
            *reinterpret_cast<uint64_t *>(&sB[g][0][0] + off) = *reinterpret_cast<const uint64_t *>(src);
        }
        __syncthreads();

        const int m_grp = warp_id >> 1;
        const int n_grp = warp_id & 1;
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            const int mt = m_grp * 2 + m;
            wmma::load_matrix_sync(a_frag, &sA[mt * 16][0], 16);
            #pragma unroll
            for (int n = 0; n < 2; n++) {
                const int nt = n_grp * 2 + n;
                wmma::load_matrix_sync(b_frag, &sB[nt][0][0], 16);
                #pragma unroll
                for (int e = 0; e < b_frag.num_elements; e++) {
                    uint8_t x = (uint8_t)b_frag.x[e];
                    b_swap.x[e] = (int8_t)(((x & 0x0F) << 4) | ((x >> 4) & 0x0F));
                }
                wmma::mma_sync(c_even[m][n], a_frag, b_frag,  c_even[m][n]);
                wmma::mma_sync(c_odd [m][n], a_frag, b_swap,  c_odd [m][n]);
            }
        }
        __syncthreads();
    }

    const int w_row  = row_base  + (warp_id >> 1) * 32;
    const int w_pair = pair_base + (warp_id & 1) * 32;
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            const int row0  = w_row  + m * 16;
            const int pair0 = w_pair + n * 16;
            wmma::store_matrix_sync(&stage[warp_id][0][0][0], c_even[m][n], 16, wmma::mem_row_major);
            wmma::store_matrix_sync(&stage[warp_id][1][0][0], c_odd [m][n], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                int r  = row0  + i / 16;
                int p  = pair0 + i % 16;
                int c0 = 2 * p;
                int c1 = 2 * p + 1;
                if (r < PK_NROWS && c1 < N) {
                    d_S_bytes[(size_t)r * N + c0] = (unsigned char)(stage[warp_id][0][i / 16][i % 16] & 1);
                    d_S_bytes[(size_t)r * N + c1] = (unsigned char)(stage[warp_id][1][i / 16][i % 16] & 1);
                }
            }
            __syncwarp();
        }
    }
}

__global__ void kernel_int8_wmma_bs2_blk2_syndrome_gemm(
    unsigned char *d_S_bytes,
    const int8_t  *d_T_int8,
    const int8_t  *d_E_bs2,
    int N,
    int pairs_padded
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int row_base  = blockIdx.y * (BS2_BLK_M_TILES * 16);
    const int pair_base = blockIdx.x * (BS2_BLK_N_TILES * 16);
    if (pair_base >= pairs_padded) return;

    __shared__ __align__(16) int8_t sA[2][BS2_BLK_M_TILES * 16][16];
    __shared__ __align__(16) int8_t sB[2][BS2_BLK_N_TILES][16][16];
    __shared__ __align__(16) int stage[4][2][16][16];

    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_even[2][2];
    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_odd [2][2];
    #pragma unroll
    for (int a = 0; a < 2; a++)
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            wmma::fill_fragment(c_even[a][b], 0);
            wmma::fill_fragment(c_odd [a][b], 0);
        }

    wmma::fragment<wmma::matrix_a, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> b_frag, b_swap;

    const int k_tiles = BS2_K_PADDED / BS2_WMMA_K;
    const int grp0    = pair_base >> 4;

    auto issueA = [&](int k, int buf) {
        int t = threadIdx.x;
        int r = t >> 1;
        int h = t & 1;
        const int8_t *src = d_T_int8 + (size_t)(row_base + r) * BS2_K_PADDED + k * BS2_WMMA_K + h * 8;
        unsigned dst = (unsigned)__cvta_generic_to_shared(&sA[buf][r][h * 8]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src));
    };
    auto issueB = [&](int k, int buf) {
        int idx8 = threadIdx.x * 8;
        int g    = idx8 >> 8;
        int off  = idx8 & 255;
        const int8_t *src = d_E_bs2 + (size_t)(grp0 + g) * (BS2_K_PADDED * 16) + k * (BS2_WMMA_K * 16) + off;
        unsigned dst = (unsigned)__cvta_generic_to_shared((int8_t *)&sB[buf][g][0][0] + off);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src));
    };

    issueA(0, 0);
    issueB(0, 0);
    asm volatile("cp.async.commit_group;");

    for (int k = 0; k < k_tiles; k++) {
        if (k + 1 < k_tiles) {
            issueA(k + 1, (k + 1) & 1);
            issueB(k + 1, (k + 1) & 1);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        const int buf = k & 1;
        const int m_grp = warp_id >> 1;
        const int n_grp = warp_id & 1;
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            const int mt = m_grp * 2 + m;
            wmma::load_matrix_sync(a_frag, &sA[buf][mt * 16][0], 16);
            #pragma unroll
            for (int n = 0; n < 2; n++) {
                const int nt = n_grp * 2 + n;
                wmma::load_matrix_sync(b_frag, &sB[buf][nt][0][0], 16);
                #pragma unroll
                for (int e = 0; e < b_frag.num_elements; e++) {
                    uint8_t x = (uint8_t)b_frag.x[e];
                    b_swap.x[e] = (int8_t)(((x & 0x0F) << 4) | ((x >> 4) & 0x0F));
                }
                wmma::mma_sync(c_even[m][n], a_frag, b_frag,  c_even[m][n]);
                wmma::mma_sync(c_odd [m][n], a_frag, b_swap,  c_odd [m][n]);
            }
        }
        __syncthreads();
    }

    const int w_row  = row_base  + (warp_id >> 1) * 32;
    const int w_pair = pair_base + (warp_id & 1) * 32;
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            const int r0 = w_row + m * 16;
            const int p0 = w_pair + n * 16;
            wmma::store_matrix_sync(&stage[warp_id][0][0][0], c_even[m][n], 16, wmma::mem_row_major);
            wmma::store_matrix_sync(&stage[warp_id][1][0][0], c_odd [m][n], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                int r  = r0 + i / 16;
                int p  = p0 + i % 16;
                int c0 = 2 * p, c1 = 2 * p + 1;
                if (r < PK_NROWS && c1 < N) {
                    d_S_bytes[(size_t)r * N + c0] = (unsigned char)(stage[warp_id][0][i / 16][i % 16] & 1);
                    d_S_bytes[(size_t)r * N + c1] = (unsigned char)(stage[warp_id][1][i / 16][i % 16] & 1);
                }
            }
            __syncwarp();
        }
    }
}

__global__ void kernel_int8_wmma_bs2_blk3_syndrome_gemm(
    unsigned char *d_S_bytes,
    const int8_t  *d_T_int8,
    const int8_t  *d_E_bs2,
    int N,
    int pairs_padded
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int row_base  = blockIdx.y * (BS2_BLK_M_TILES * 16);
    const int pair_base = blockIdx.x * (BS2_BLK_N_TILES * 16);
    if (pair_base >= pairs_padded) return;

    __shared__ __align__(16) unsigned char pool[4096];

    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_even[2][2];
    wmma::fragment<wmma::accumulator, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K, int> c_odd [2][2];
    #pragma unroll
    for (int a = 0; a < 2; a++)
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            wmma::fill_fragment(c_even[a][b], 0);
            wmma::fill_fragment(c_odd [a][b], 0);
        }

    wmma::fragment<wmma::matrix_a, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BS2_WMMA_M, BS2_WMMA_N, BS2_WMMA_K,
                   signed char, wmma::row_major> b_frag, b_swap;

    const int k_tiles = BS2_K_PADDED / BS2_WMMA_K;
    const int grp0    = pair_base >> 4;

    auto issueA = [&](int k, int buf) {
        int t = threadIdx.x;
        int r = t >> 1;
        int h = t & 1;
        if ((row_base + r) >= BS2_M_PADDED) return;
        unsigned dst = (unsigned)__cvta_generic_to_shared(pool + buf * 1024 + r * 16 + h * 8);
        const int8_t *src = d_T_int8 + (size_t)(row_base + r) * BS2_K_PADDED + k * BS2_WMMA_K + h * 8;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src));
    };
    auto issueB = [&](int k, int buf) {
        int idx8 = threadIdx.x * 8;
        int g    = idx8 >> 8;
        int off  = idx8 & 255;
        unsigned dst = (unsigned)__cvta_generic_to_shared(pool + 2048 + buf * 1024 + g * 256 + off);
        const int8_t *src = d_E_bs2 + (size_t)(grp0 + g) * (BS2_K_PADDED * 16) + k * (BS2_WMMA_K * 16) + off;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src));
    };

    issueA(0, 0);
    issueB(0, 0);
    asm volatile("cp.async.commit_group;");

    for (int k = 0; k < k_tiles; k++) {
        if (k + 1 < k_tiles) {
            issueA(k + 1, (k + 1) & 1);
            issueB(k + 1, (k + 1) & 1);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        const int buf = k & 1;
        const int m_grp = warp_id >> 1;
        const int n_grp = warp_id & 1;
        const int8_t *sA = (const int8_t *)pool + (buf * 1024);
        const int8_t *sB = (const int8_t *)pool + 2048 + (buf * 1024);
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            const int mt = m_grp * 2 + m;
            wmma::load_matrix_sync(a_frag, sA + mt * 256, 16);
            #pragma unroll
            for (int n = 0; n < 2; n++) {
                const int nt = n_grp * 2 + n;
                wmma::load_matrix_sync(b_frag, sB + nt * 256, 16);
                #pragma unroll
                for (int e = 0; e < b_frag.num_elements; e++) {
                    uint8_t x = (uint8_t)b_frag.x[e];
                    b_swap.x[e] = (int8_t)(((x & 0x0F) << 4) | ((x >> 4) & 0x0F));
                }
                wmma::mma_sync(c_even[m][n], a_frag, b_frag,  c_even[m][n]);
                wmma::mma_sync(c_odd [m][n], a_frag, b_swap,  c_odd [m][n]);
            }
        }
        __syncthreads();
    }

    int (*stage)[16] = reinterpret_cast<int (*)[16]>(pool + warp_id * 1024);
    const int w_row  = row_base  + (warp_id >> 1) * 32;
    const int w_pair = pair_base + (warp_id & 1) * 32;
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            const int r0 = w_row + m * 16;
            const int p0 = w_pair + n * 16;
            wmma::store_matrix_sync(&stage[0][0], c_even[m][n], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                int r = r0 + i / 16, p = p0 + i % 16, c0 = 2 * p;
                if (r < PK_NROWS && c0 < N)
                    d_S_bytes[(size_t)r * N + c0] = (unsigned char)(stage[i / 16][i % 16] & 1);
            }
            wmma::store_matrix_sync(&stage[0][0], c_odd[m][n], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                int r = r0 + i / 16, p = p0 + i % 16, c1 = 2 * p + 1;
                if (r < PK_NROWS && c1 < N)
                    d_S_bytes[(size_t)r * N + c1] = (unsigned char)(stage[i / 16][i % 16] & 1);
            }
        }
    }
}

#endif
