/* Parameter-generic Tensor Core kernels for the multi-parameter harness.
 * Requires a generated ref/<set>.cuh (SYS_N, SYS_T, GFBITS, PK_*, SYND_BYTES, gen_e, ...). */
#ifndef TC_KERNELS_CUH
#define TC_KERNELS_CUH

#include <mma.h>
using namespace nvcuda;

/* ------------------------------------------------------------------------- */
/* sizes                                                                     */
/* ------------------------------------------------------------------------- */
#define E_BYTES      (SYS_N / 8)
#define NVEC         (1 << (GFBITS - 6))          /* bitsliced vectors of 64 field elements */
#define LOC_W        ((GFBITS == 12) ? 1 : 2)     /* locator words per bit slice */
#define SPRIV_W      ((GFBITS == 12) ? 2 : 4)     /* double-size syndrome words per bit slice */

#define F_COLS       (SYND_BYTES * 8)             /* ciphertext bits incl. padding */
#define F_KW         (((F_COLS + 127) / 128) * 4) /* uint32 per packed row */
#define F_ROWS       (SPRIV_W * GFBITS * 64)
#define A_COLS       (LOC_W * GFBITS * 64)
#define A_KW         (A_COLS / 32)
#define A_ROWS       (NVEC * GFBITS * 64)
#define G_COLS       (NVEC * 64)
#define G_KW         (G_COLS / 32)
#define G_ROWS       F_ROWS
#define F_OUTW       (F_ROWS / 64)                /* 64-bit output words per operation */
#define A_OUTW       (A_ROWS / 64)

#if GFBITS == 12
#define LOC_ARG(p)   ((vec *)(p))
#else
#define LOC_ARG(p)   ((vec (*)[GFBITS])(p))
#endif
#define VEC2D(p)     ((vec (*)[GFBITS])(p))

__device__ __forceinline__ void bmma_m8n8k128_and_popc(unsigned &d0, unsigned &d1, unsigned a, unsigned b,
                                                       unsigned c0, unsigned c1)
{
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
                 "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
                 : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}

/* bit i of the non-systematic part e_ns = bit (mt + i) of e */
__device__ __forceinline__ unsigned ens_bit(const unsigned char *e, int i)
{
    int b = PK_NROWS + i;
    return (e[b >> 3] >> (b & 7)) & 1u;
}

/* ------------------------------------------------------------------------- */
/* Encapsulation: 1-bit BMMA syndrome (same tiling as the McEliece348864     */
/* engine; K and M padding derived from the parameter set)                   */
/* ------------------------------------------------------------------------- */
/* One extra zero k-tile when the padded row stride of T would be a multiple of 128 bytes
 * (McEliece6688128: 40 tiles = 640 B): the eight rows read by one fragment load then hit
 * the same cache sets, which cost the BMMA engine about a third of its throughput. */
#define ENC_KT_BASE ((PK_NCOLS + 127) / 128)
#ifndef ENC_KT_EXTRA
#define ENC_KT_EXTRA ((ENC_KT_BASE % 8 == 0) ? 1 : 0)
#endif
#define ENC_KT      (ENC_KT_BASE + ENC_KT_EXTRA)
#define ENC_KW      (ENC_KT * 4)
#define ENC_BM      32
#define ENC_BN      32

static void pack_pk_bmma(uint32_t *dst, const unsigned char *pk)
{
    memset(dst, 0, (size_t)PK_NROWS * ENC_KW * 4);
    for (int r = 0; r < PK_NROWS; r++)
        memcpy(dst + (size_t)r * ENC_KW, pk + (size_t)r * PK_ROW_BYTES, PK_ROW_BYTES);
}

__global__ void kernel_gen_e_pack_bmma(unsigned char *d_e, uint32_t *d_E, const unsigned char *d_seeds, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned char *e = d_e + (size_t)idx * E_BYTES;
    gen_e(e, d_seeds + (size_t)idx * 32);
    for (int w = 0; w < ENC_KW; w++) {
        uint32_t v = 0;
#if PK_NROWS % 8 == 0
        for (int b = 0; b < 4; b++) {
            int byte = PK_NROWS / 8 + 4 * w + b;
            if (4 * w + b < PK_ROW_BYTES) v |= (uint32_t)e[byte] << (8 * b);
        }
#else
        for (int b = 0; b < 32; b++) {
            int i = 32 * w + b;
            if (i < PK_NCOLS) v |= ens_bit(e, i) << b;
        }
#endif
        d_E[(size_t)w * N + idx] = v;
    }
}

__global__ void kernel_bmma_syndrome(unsigned char *d_S, const uint32_t *d_T, const uint32_t *d_E, int N)
{
    int warp = threadIdx.y, lane = threadIdx.x;
    int block_m = blockIdx.y * ENC_BM, block_n = blockIdx.x * ENC_BN;
    if (block_n >= N) return;
    int col_base = block_n + warp * 8;
    unsigned d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0};
    int col = col_base + lane / 4, ks = lane % 4, ra = lane / 4;

    for (int kt = 0; kt < ENC_KT; kt++) {
        unsigned b = (col < N) ? d_E[(size_t)(kt * 4 + ks) * N + col] : 0u;
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            int row = block_m + m * 8 + ra;
            unsigned a = (row < PK_NROWS) ? d_T[(size_t)row * ENC_KW + kt * 4 + ks] : 0u;
            bmma_m8n8k128_and_popc(d0[m], d1[m], a, b, d0[m], d1[m]);
        }
    }
    int c0 = col_base + 2 * (lane % 4), c1 = c0 + 1;
    for (int m = 0; m < 4; m++) {
        int r = block_m + m * 8 + lane / 4;
        if (r < PK_NROWS) {
            if (c0 < N) d_S[(size_t)r * N + c0] = (unsigned char)(d0[m] & 1u);
            if (c1 < N) d_S[(size_t)r * N + c1] = (unsigned char)(d1[m] & 1u);
        }
    }
}

/* c = e_sys + (T e_ns), padding bits zero; K = H(1, e, c) */
__global__ void kernel_finalize_encap(unsigned char *d_cs, unsigned char *d_keys, const unsigned char *d_S,
                                      const unsigned char *d_e, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const unsigned char *e = d_e + (size_t)idx * E_BYTES;
    unsigned char *c = d_cs + (size_t)idx * SYND_BYTES;
    unsigned char one_ec[1 + E_BYTES + SYND_BYTES];
    for (int i = 0; i < SYND_BYTES; i++) {
        unsigned char v = 0;
        for (int b = 0; b < 8; b++) {
            int bit = 8 * i + b;
            if (bit < PK_NROWS)
                v |= (unsigned char)(((d_S[(size_t)bit * N + idx] ^ (e[i] >> b)) & 1) << b);
        }
        c[i] = v;
    }
    one_ec[0] = 1;
    for (int i = 0; i < E_BYTES; i++) one_ec[1 + i] = e[i];
    for (int i = 0; i < SYND_BYTES; i++) one_ec[1 + E_BYTES + i] = c[i];
    crypto_hash_32b(d_keys + (size_t)idx * 32, one_ec, sizeof(one_ec));
}

/* ------------------------------------------------------------------------- */
/* Encapsulation: INT8 WMMA, two-column nibble packing (bs2-m4, bs2-blk3)    */
/* ------------------------------------------------------------------------- */
#ifndef BS2_K_EXTRA
#define BS2_K_EXTRA 0
#endif
#define BS2_K_PADDED ((PK_NCOLS + 15) / 16 * 16 + 16 * BS2_K_EXTRA)
#define BS2_M_PADDED ((PK_NROWS + 15) / 16 * 16)
#define BS2_PAIRS_PADDED(_N) ((((_N) / 2) + 63) / 64 * 64)
#define BS2_M_CHUNK 4

static void pack_pk_int8(int8_t *dst, const unsigned char *pk)
{
    memset(dst, 0, (size_t)BS2_M_PADDED * BS2_K_PADDED);
    for (int r = 0; r < PK_NROWS; r++)
        for (int c = 0; c < PK_NCOLS; c++)
            dst[(size_t)r * BS2_K_PADDED + c] = (pk[(size_t)r * PK_ROW_BYTES + (c >> 3)] >> (c & 7)) & 1;
}

/* one thread per pair of operations, as in the McEliece348864 engine */
__global__ void kernel_gen_e_pack_bs2(unsigned char *d_e, int8_t *d_E, const unsigned char *d_seeds, int N)
{
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= N / 2) return;
    int j0 = 2 * pair, j1 = 2 * pair + 1;
    size_t tile = (size_t)(pair >> 4) * (BS2_K_PADDED * 16);
    int pr = pair & 15;
    unsigned char *e0 = d_e + (size_t)j0 * E_BYTES, *e1 = d_e + (size_t)j1 * E_BYTES;
    gen_e(e0, d_seeds + (size_t)j0 * 32);
    gen_e(e1, d_seeds + (size_t)j1 * 32);
    for (int k = 0; k < BS2_K_PADDED; k++) {
        unsigned b0 = 0, b1 = 0;
        if (k < PK_NCOLS) { b0 = ens_bit(e0, k); b1 = ens_bit(e1, k); }
        d_E[tile + (size_t)k * 16 + pr] = (int8_t)(b0 | (b1 << 4));
    }
}

__device__ __forceinline__ void bs2_swap(wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> &dst,
                                         const wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> &src)
{
    #pragma unroll
    for (int e = 0; e < src.num_elements; e++) {
        uint8_t x = (uint8_t)src.x[e];
        dst.x[e] = (int8_t)(((x & 0x0F) << 4) | ((x >> 4) & 0x0F));
    }
}

__global__ void kernel_bs2_m4(unsigned char *d_S, const int8_t *d_T, const int8_t *d_E, int N, int pairs_padded)
{
    const int m_base = blockIdx.y * (BS2_M_CHUNK * 16);
    const size_t tile = (size_t)blockIdx.x * (BS2_K_PADDED * 16);
    if (blockIdx.x * 16 >= pairs_padded) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b, bsw;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> ce[BS2_M_CHUNK], co[BS2_M_CHUNK];
    #pragma unroll
    for (int m = 0; m < BS2_M_CHUNK; m++) { wmma::fill_fragment(ce[m], 0); wmma::fill_fragment(co[m], 0); }
    for (int k = 0; k < BS2_K_PADDED / 16; k++) {
        wmma::load_matrix_sync(b, d_E + tile + (size_t)k * 256, 16);
        bs2_swap(bsw, b);
        #pragma unroll
        for (int m = 0; m < BS2_M_CHUNK; m++) {
            int row = m_base + m * 16;
            if (row >= BS2_M_PADDED) continue;
            wmma::load_matrix_sync(a, d_T + (size_t)row * BS2_K_PADDED + k * 16, BS2_K_PADDED);
            wmma::mma_sync(ce[m], a, b, ce[m]);
            wmma::mma_sync(co[m], a, bsw, co[m]);
        }
    }
    __shared__ int se[16][16], so[16][16];
    for (int m = 0; m < BS2_M_CHUNK; m++) {
        int r0 = m_base + m * 16;
        if (r0 >= BS2_M_PADDED) continue;
        wmma::store_matrix_sync(&se[0][0], ce[m], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(&so[0][0], co[m], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = threadIdx.x; i < 256; i += 32) {
            int r = r0 + i / 16, p = blockIdx.x * 16 + i % 16, c0 = 2 * p, c1 = c0 + 1;
            if (r < PK_NROWS && c1 < N) {
                d_S[(size_t)r * N + c0] = (unsigned char)(se[i / 16][i % 16] & 1);
                d_S[(size_t)r * N + c1] = (unsigned char)(so[i / 16][i % 16] & 1);
            }
        }
        __syncwarp();
    }
}

/* four warps, 64 rows x 64 pairs, cp.async double-buffered 4 KB pool */
__global__ void kernel_bs2_blk3(unsigned char *d_S, const int8_t *d_T, const int8_t *d_E, int N, int pairs_padded)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row_base = blockIdx.y * 64, pair_base = blockIdx.x * 64;
    if (pair_base >= pairs_padded) return;
    __shared__ __align__(16) unsigned char pool[4096];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> ce[2][2], co[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) { wmma::fill_fragment(ce[i][j], 0); wmma::fill_fragment(co[i][j], 0); }
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b, bsw;
    const int kt = BS2_K_PADDED / 16, grp0 = pair_base >> 4;

    auto issue = [&](int k, int buf) {
        int t = threadIdx.x, r = t >> 1, h = t & 1;
        if (row_base + r < BS2_M_PADDED) {
            unsigned dst = (unsigned)__cvta_generic_to_shared(pool + buf * 1024 + r * 16 + h * 8);
            const int8_t *src = d_T + (size_t)(row_base + r) * BS2_K_PADDED + k * 16 + h * 8;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src));
        }
        int idx8 = t * 8, g = idx8 >> 8, off = idx8 & 255;
        unsigned dstb = (unsigned)__cvta_generic_to_shared(pool + 2048 + buf * 1024 + g * 256 + off);
        const int8_t *srcb = d_E + (size_t)(grp0 + g) * (BS2_K_PADDED * 16) + k * 256 + off;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dstb), "l"(srcb));
    };
    issue(0, 0);
    asm volatile("cp.async.commit_group;");
    for (int k = 0; k < kt; k++) {
        if (k + 1 < kt) issue(k + 1, (k + 1) & 1);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        const int8_t *sA = (const int8_t *)pool + (k & 1) * 1024;
        const int8_t *sB = (const int8_t *)pool + 2048 + (k & 1) * 1024;
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            wmma::load_matrix_sync(a, sA + ((warp >> 1) * 2 + m) * 256, 16);
            #pragma unroll
            for (int n = 0; n < 2; n++) {
                wmma::load_matrix_sync(b, sB + ((warp & 1) * 2 + n) * 256, 16);
                bs2_swap(bsw, b);
                wmma::mma_sync(ce[m][n], a, b, ce[m][n]);
                wmma::mma_sync(co[m][n], a, bsw, co[m][n]);
            }
        }
        __syncthreads();
    }
    int (*stage)[16] = reinterpret_cast<int (*)[16]>(pool + warp * 1024);
    const int w_row = row_base + (warp >> 1) * 32, w_pair = pair_base + (warp & 1) * 32;
    #pragma unroll
    for (int m = 0; m < 2; m++)
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            int r0 = w_row + m * 16, p0 = w_pair + n * 16;
            wmma::store_matrix_sync(&stage[0][0], ce[m][n], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                int r = r0 + i / 16, c0 = 2 * (p0 + i % 16);
                if (r < PK_NROWS && c0 < N) d_S[(size_t)r * N + c0] = (unsigned char)(stage[i / 16][i % 16] & 1);
            }
            __syncwarp();
            wmma::store_matrix_sync(&stage[0][0], co[m][n], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                int r = r0 + i / 16, c1 = 2 * (p0 + i % 16) + 1;
                if (r < PK_NROWS && c1 < N) d_S[(size_t)r * N + c1] = (unsigned char)(stage[i / 16][i % 16] & 1);
            }
            __syncwarp();
        }
}

/* ------------------------------------------------------------------------- */
/* Encapsulation / decapsulation on CUDA cores: the reference code per thread */
/* ------------------------------------------------------------------------- */
__global__ void kernel_gen_e(unsigned char *d_e, const unsigned char *d_seeds, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    gen_e(d_e + (size_t)idx * E_BYTES, d_seeds + (size_t)idx * 32);
}

/* PKE setting: syndrome of a supplied e, ciphertext and session key */
__global__ void kernel_cc_encap_pke(unsigned char *d_cs, unsigned char *d_keys, const unsigned char *d_pk,
                                    const unsigned char *d_e, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned char one_ec[1 + E_BYTES + SYND_BYTES];
    unsigned char *e = one_ec + 1, *c = one_ec + 1 + E_BYTES;
    one_ec[0] = 1;
    for (int i = 0; i < E_BYTES; i++) e[i] = d_e[(size_t)idx * E_BYTES + i];
    syndrome(c, d_pk, e);
    for (int i = 0; i < SYND_BYTES; i++) d_cs[(size_t)idx * SYND_BYTES + i] = c[i];
    crypto_hash_32b(d_keys + (size_t)idx * 32, one_ec, sizeof(one_ec));
}

/* KEM setting: fixed-weight sampling + syndrome + session key (pk padding checked once on the host) */
__global__ void kernel_cc_encap_kem(unsigned char *d_cs, unsigned char *d_keys, const unsigned char *d_pk,
                                    const unsigned char *d_seeds, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned char one_ec[1 + E_BYTES + SYND_BYTES];
    unsigned char *e = one_ec + 1, *c = one_ec + 1 + E_BYTES;
    one_ec[0] = 1;
    pke_encrypt(c, d_pk, e, d_seeds + (size_t)idx * 32);
    for (int i = 0; i < SYND_BYTES; i++) d_cs[(size_t)idx * SYND_BYTES + i] = c[i];
    crypto_hash_32b(d_keys + (size_t)idx * 32, one_ec, sizeof(one_ec));
}

template <bool REENC>
__global__ void kernel_cc_decap(unsigned char *d_keys, const unsigned char *d_cs, const unsigned char *d_sk, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    operation_dec_t<REENC>(d_keys + (size_t)idx * 32, d_cs + (size_t)idx * SYND_BYTES, d_sk);
}

/* ------------------------------------------------------------------------- */
/* Decapsulation on Tensor Cores: generic bit-GEMM with K staged in shared   */
/* memory, 64 output rows per block (one 64-bit word per operation)          */
/* ------------------------------------------------------------------------- */
#define GB_ROWS 64
#define GB_KC   8                       /* k-tiles per shared-memory chunk */
#define GB_SROW (GB_KC * 4 + 4)         /* padded: conflict-free fragment loads */

template <int NT, int KW>
__global__ void kernel_bmma_bits(uint64_t *d_out, int out_words, const uint32_t *d_A, const uint32_t *d_B, int N)
{
    constexpr int KT = KW / 4;                  /* 128-bit k-tiles */
    __shared__ uint32_t sA[GB_ROWS][GB_SROW];
    __shared__ unsigned long long s_word[32 * NT];
    const int warp = threadIdx.y, lane = threadIdx.x, tid = warp * 32 + lane;
    const int block_m = blockIdx.y * GB_ROWS, block_n = blockIdx.x * (32 * NT);
    if (block_n >= N) return;
    const int ra = lane / 4, ks = lane % 4;
    unsigned d0[8][NT], d1[8][NT];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        #pragma unroll
        for (int n = 0; n < NT; n++) d0[m][n] = d1[m][n] = 0;

    for (int kc = 0; kc < KT; kc += GB_KC) {
        const int cw = (KT - kc < GB_KC ? KT - kc : GB_KC) * 4;
        __syncthreads();
        for (int i = tid; i < GB_ROWS * cw; i += 128)
            sA[i / cw][i % cw] = d_A[(size_t)(block_m + i / cw) * KW + kc * 4 + i % cw];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < GB_KC; k++) {
            if (KT % GB_KC != 0 && kc + k >= KT) break;
            unsigned a[8];
            #pragma unroll
            for (int m = 0; m < 8; m++) a[m] = sA[m * 8 + ra][k * 4 + ks];
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                int col = block_n + (warp + 4 * n) * 8 + ra;
                unsigned b = (col < N) ? d_B[(size_t)((kc + k) * 4 + ks) * N + col] : 0u;
                #pragma unroll
                for (int m = 0; m < 8; m++) bmma_m8n8k128_and_popc(d0[m][n], d1[m][n], a[m], b, d0[m][n], d1[m][n]);
            }
        }
    }
    for (int i = tid; i < 32 * NT; i += 128) s_word[i] = 0ULL;
    __syncthreads();
    #pragma unroll
    for (int n = 0; n < NT; n++) {
        unsigned long long v0 = 0, v1 = 0;
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            v0 |= (unsigned long long)(d0[m][n] & 1u) << (ra + 8 * m);
            v1 |= (unsigned long long)(d1[m][n] & 1u) << (ra + 8 * m);
        }
        int c = (warp + 4 * n) * 8 + 2 * ks;
        atomicOr(&s_word[c], v0);
        atomicOr(&s_word[c + 1], v1);
    }
    __syncthreads();
    for (int i = tid; i < 32 * NT; i += 128) {
        int col = block_n + i;
        if (col < N) d_out[(size_t)col * out_words + blockIdx.y] = s_word[i];
    }
}

__global__ void kernel_pack_c(uint32_t *d_c32, const unsigned char *d_cs, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const unsigned char *c = d_cs + (size_t)idx * SYND_BYTES;
    for (int w = 0; w < F_KW; w++) {
        uint32_t v = 0;
        for (int b = 0; b < 4; b++)
            if (4 * w + b < SYND_BYTES) v |= (uint32_t)c[4 * w + b] << (8 * b);
        d_c32[(size_t)w * N + idx] = v;
    }
}

__global__ void kernel_tc_bm(uint64_t *d_loc, const uint64_t *d_spriv, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    vec spriv[SPRIV_W * GFBITS], loc[LOC_W * GFBITS];
    for (int i = 0; i < SPRIV_W * GFBITS; i++) spriv[i] = d_spriv[(size_t)idx * F_OUTW + i];
    bm(LOC_ARG(loc), VEC2D(spriv));
    for (int i = 0; i < LOC_W * GFBITS; i++) d_loc[(size_t)idx * LOC_W * GFBITS + i] = loc[i];
}

__global__ void kernel_pack_loc(uint32_t *d_loc32, const uint64_t *d_loc, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const uint64_t *L = d_loc + (size_t)idx * LOC_W * GFBITS;
    for (int w = 0; w < A_KW; w++) d_loc32[(size_t)w * N + idx] = (uint32_t)(L[w >> 1] >> (32 * (w & 1)));
}

__device__ __forceinline__ void tc_error_from_eval(vec *error, const uint64_t *ev, const uint64_t *cA)
{
    vec allone = vec_setbits(1);
    for (int i = 0; i < NVEC; i++) {
        vec r = 0;
        for (int b = 0; b < GFBITS; b++) r |= ev[i * GFBITS + b] ^ cA[i * GFBITS + b];
        error[i] = r ^ allone;
    }
}

/* PKE setting: error extraction, Benes, weight check, hash (no re-encryption check) */
__global__ void kernel_tc_tail_pke(unsigned char *d_keys, const unsigned char *d_cs, const unsigned char *d_sk,
                                   const uint64_t *d_eval, const uint64_t *d_cA, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    vec error[NVEC];
    tc_error_from_eval(error, d_eval + (size_t)idx * A_OUTW, d_cA);
    operation_dec_from_error(d_keys + (size_t)idx * 32, d_cs + (size_t)idx * SYND_BYTES, d_sk, error, 1);
}

/* KEM setting, before the G GEMM: error vector per operation and as the interleaved B operand */
__global__ void kernel_tc_error_extract(uint64_t *d_err, uint32_t *d_err32, const uint64_t *d_eval,
                                        const uint64_t *d_cA, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    vec error[NVEC];
    tc_error_from_eval(error, d_eval + (size_t)idx * A_OUTW, d_cA);
    for (int i = 0; i < NVEC; i++) {
        d_err[(size_t)idx * NVEC + i] = error[i];
        d_err32[(size_t)(2 * i) * N + idx] = (uint32_t)error[i];
        d_err32[(size_t)(2 * i + 1) * N + idx] = (uint32_t)(error[i] >> 32);
    }
}

/* KEM setting, after the G GEMM: syndrome comparison, Benes, weight check, hash */
__global__ void kernel_tc_tail_kem(unsigned char *d_keys, const unsigned char *d_cs, const unsigned char *d_sk,
                                   const uint64_t *d_err, const uint64_t *d_spriv, const uint64_t *d_scmp, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    vec error[NVEC], diff = 0;
    for (int i = 0; i < NVEC; i++) error[i] = d_err[(size_t)idx * NVEC + i];
    for (int w = 0; w < F_OUTW; w++) diff |= d_spriv[(size_t)idx * F_OUTW + w] ^ d_scmp[(size_t)idx * F_OUTW + w];
    operation_dec_from_error(d_keys + (size_t)idx * 32, d_cs + (size_t)idx * SYND_BYTES, d_sk, error,
                             (uint16_t)vec_testz(diff));
}

/* ------------------------------------------------------------------------- */
/* Matrix extraction: push basis vectors through the reference functions     */
/* ------------------------------------------------------------------------- */
__device__ void ref_front_end(uint64_t *out, const unsigned char *sk, const unsigned char *c)
{
    vec recv[NVEC], scaled[NVEC][GFBITS], inv[NVEC][GFBITS], spriv[SPRIV_W][GFBITS];
    preprocess(recv, c);
    benes(recv, sk + 40 + IRR_BYTES, 1);
    scaling(scaled, inv, sk + 40, recv);
    fft_tr(spriv, scaled);
    for (int i = 0; i < SPRIV_W * GFBITS; i++) out[i] = ((vec *)spriv)[i];
}

__device__ void ref_afft(uint64_t *out, const uint64_t *loc_in)
{
    vec loc[LOC_W * GFBITS], eval[NVEC][GFBITS];
    for (int i = 0; i < LOC_W * GFBITS; i++) loc[i] = loc_in[i];
    fft(eval, LOC_ARG(loc));
    for (int i = 0; i < NVEC * GFBITS; i++) out[i] = ((vec *)eval)[i];
}

__device__ void ref_reenc(uint64_t *out, const unsigned char *sk, const uint64_t *err_in)
{
    vec recv[NVEC], scaled[NVEC][GFBITS], inv[NVEC][GFBITS], s[SPRIV_W][GFBITS], err[NVEC];
    for (int i = 0; i < NVEC; i++) { recv[i] = 0; err[i] = err_in[i]; }
    scaling(scaled, inv, sk + 40, recv);
    scaling_inv(scaled, inv, err);
    fft_tr(s, scaled);
    for (int i = 0; i < SPRIV_W * GFBITS; i++) out[i] = ((vec *)s)[i];
}

/* one thread per column (basis vector); which = 0: F, 1: A, 2: G */
__global__ void kernel_build_cols(uint64_t *d_cols, int which, const unsigned char *d_sk, int ncols)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncols) return;
    if (which == 0) {
        unsigned char c[SYND_BYTES];
        for (int i = 0; i < SYND_BYTES; i++) c[i] = 0;
        c[t >> 3] = (unsigned char)(1u << (t & 7));
        ref_front_end(d_cols + (size_t)t * F_OUTW, d_sk, c);
    } else if (which == 1) {
        uint64_t loc[LOC_W * GFBITS];
        for (int i = 0; i < LOC_W * GFBITS; i++) loc[i] = 0;
        if (t < A_COLS) loc[t >> 6] = 1ULL << (t & 63);
        ref_afft(d_cols + (size_t)t * A_OUTW, loc);   /* t == A_COLS: the constant cA */
    } else {
        uint64_t err[NVEC];
        for (int i = 0; i < NVEC; i++) err[i] = 0;
        err[t >> 6] = 1ULL << (t & 63);
        ref_reenc(d_cols + (size_t)t * F_OUTW, d_sk, err);
    }
}

/* single-thread oracles for the verification of the extracted matrices */
__global__ void kernel_oracle(uint64_t *d_out, int which, const unsigned char *d_sk, const uint64_t *d_x)
{
    if (which == 0) {
        unsigned char c[SYND_BYTES];
        for (int i = 0; i < SYND_BYTES; i++) c[i] = (unsigned char)(d_x[i >> 3] >> (8 * (i & 7)));
        ref_front_end(d_out, d_sk, c);
    } else if (which == 1) {
        ref_afft(d_out, d_x);
    } else {
        ref_reenc(d_out, d_sk, d_x);
    }
}

#endif
