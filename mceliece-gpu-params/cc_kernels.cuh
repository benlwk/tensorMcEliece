/* CUDA-core (warp-cooperative) decapsulation for the multi-parameter harness.
 *
 * One warp per operation, ported from ../mceliece-gpu-lfmass/cc_decap.h:
 *   kernel_cc_front     s_priv = F c by table lookup: c is split into CC_CHUNKS
 *                       4-bit chunks and s_priv is the XOR of the precomputed
 *                       F_OUTW-word entries (method of Four Russians, linear in c);
 *   kernel_tc_bm        reference Berlekamp-Massey, one thread per operation;
 *   kernel_cc_afft      warp-cooperative AFFT of the locator + error extraction;
 *   kernel_cc_tafft     warp-cooperative TAFFT of inv & error (KEM setting only);
 *   kernel_tc_tail_kem  reference tail: syndrome comparison, Benes, weight, hash.
 * The (T)AFFT tables are the fft_* arrays of the generated ref/<set>.cuh.
 *   m = 12: the 64-point network of cc_decap.h; its fft_* tables are byte-identical
 *           to fft_data_cmem.h of the lfmass build (checked symbolically), so that
 *           code is reused verbatim;
 *   m = 13: the 128-point Gao-Mateer network, derived from the reference
 *           radix_conversions{,_tr} / butterflies{,_tr} (see CC_PORT_NOTES.md).
 */
#ifndef CC_KERNELS_CUH
#define CC_KERNELS_CUH

#define CC_WARPS    (GFBITS >= 13 ? 2 : 4)       /* warps per block: smem footprint vs occupancy */
#define CC_NPRE     (GFBITS >= 13 ? 7 : 6)       /* beta values of the forward FFT */
#define CC_PSTRIDE  (NVEC + 1)                   /* plane stride: conflict-free per-plane access */
#define CC_CHUNKS   (F_COLS / 4)                 /* ciphertext bits in 4-bit chunks */
#define CC_TAB_WORDS ((size_t)CC_CHUNKS * 16 * F_OUTW)
#define CC_OUT_SLOTS ((F_OUTW + 31) / 32)        /* output/state words per lane */
#define CC_WARP_OK  1                            /* 64- and 128-point (T)AFFT networks available */

/* Two reference details vary between the generated ref/<set>.cuh (the Makefile
 * passes both from a grep of the header; the defaults cover McEliece460896 and
 * McEliece6960119, whose SYS_T < 128):
 *   HAS_FFT_POWERS         fft adds the x^128 term `fft_powers` (present only
 *                          when SYS_T == 128, i.e. not for 460896/6960119);
 *   HAS_FFT_TR_POSTPROCESS fft_tr ends with `postprocess(out)`, zeroing row 3
 *                          "justifying the length of the output". */
#ifndef PARAM_ID
#define PARAM_ID 0
#endif
#ifndef HAS_FFT_POWERS
#define HAS_FFT_POWERS (PARAM_ID != 460896 && PARAM_ID != 6960119)
#endif
#ifndef HAS_FFT_TR_POSTPROCESS
#define HAS_FFT_TR_POSTPROCESS (!HAS_FFT_POWERS)
#endif

#define CC_CHECK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(_e)); exit(1); } } while (0)

struct CCWarp {
    uint64_t S[GFBITS][CC_PSTRIDE];   /* bitsliced state, plane-major: S[b][vector] */
    uint64_t buf[2 * GFBITS - 1];
    uint64_t L[LOC_W * GFBITS];       /* locator / (T)AFFT input, LOC_W rows */
    uint64_t O[SPRIV_W][GFBITS];      /* TAFFT output (SPRIV_W * GFBITS == F_OUTW) */
    uint64_t PRE[CC_NPRE][GFBITS];
    uint64_t T[GFBITS];
    uint64_t BETA[6][GFBITS];
};

/* global copies of the (T)AFFT tables: lanes index them non-uniformly */
struct CCTables {
    const uint64_t *consts;           /* fft_consts,       [63][GFBITS] (m = 12) [127][GFBITS] (m = 13) */
    const uint64_t *powers;           /* fft_powers,       [64][GFBITS] (m = 12) [128][GFBITS] (m = 13) */
    const uint64_t *scalars;          /* fft_scalars,      [5][GFBITS] (m = 12) */
    const uint64_t *scalars_2x;       /* fft_scalars_2x,   [5][2][GFBITS] */
    const uint64_t *scalars_4x;       /* fft_scalars_4x,   [6][4][GFBITS] (m = 13) */
};

#define CC_SMEM (CC_WARPS * (int)sizeof(CCWarp))

/* inv[NVEC][GFBITS] of the reference scaling() with recv = 0: the key-dependent
 * scaling factors of the re-encryption, precomputed once per key (one thread). */
__global__ void kernel_cc_scaling_inv(uint64_t *d_inv, const unsigned char *d_sk) {
    vec scaled[NVEC][GFBITS], inv[NVEC][GFBITS], recv[NVEC];
    for (int i = 0; i < NVEC; i++) recv[i] = 0;
    scaling(scaled, inv, d_sk + 40, recv);
    for (int i = 0; i < NVEC; i++)
        for (int j = 0; j < GFBITS; j++) d_inv[i * GFBITS + j] = inv[i][j];
}

#if CC_WARP_OK

/* ---------------------------------------------------------------- constants */
/* field constants of ref/<set>.cuh, reachable from a warp-non-uniform index */
static inline CCTables cc_tables(void) {
    CCTables tb;
    void *p = 0;
    CC_CHECK(cudaGetSymbolAddress(&p, fft_consts));     tb.consts = (const uint64_t *)p;
#if HAS_FFT_POWERS
    CC_CHECK(cudaGetSymbolAddress(&p, fft_powers));     tb.powers = (const uint64_t *)p;
#else
    tb.powers = 0;                     /* ref/<set>.cuh omits the x^128 term */
#endif
    CC_CHECK(cudaGetSymbolAddress(&p, fft_scalars_2x)); tb.scalars_2x = (const uint64_t *)p;
#if GFBITS == 12
    CC_CHECK(cudaGetSymbolAddress(&p, fft_scalars));    tb.scalars = (const uint64_t *)p;
    tb.scalars_4x = 0;
#else
    CC_CHECK(cudaGetSymbolAddress(&p, fft_scalars_4x)); tb.scalars_4x = (const uint64_t *)p;
    tb.scalars = 0;
#endif
    return tb;
}

/* ------------------------------------------------------- warp vector multiply */
__device__ __forceinline__ void warp_vec_mul(CCWarp &w, uint64_t *h, const uint64_t *f, const uint64_t *g, int lane) {
    if (lane < 2 * GFBITS - 1) {
        uint64_t acc = 0;
        #pragma unroll
        for (int i = 0; i < GFBITS; i++) {
            int j = lane - i;
            if (j >= 0 && j < GFBITS) acc ^= f[i] & g[j];
        }
        w.buf[lane] = acc;
    }
    __syncwarp();
    if (lane == 0) {
        #pragma unroll
        for (int i = 2 * GFBITS - 2; i >= GFBITS; i--) {
#if GFBITS >= 13
            /* x^13 = x^4 + x^3 + x + 1 (libmceliece vec_mul, m = 13) */
            w.buf[i - GFBITS + 4] ^= w.buf[i];
            w.buf[i - GFBITS + 3] ^= w.buf[i];
            w.buf[i - GFBITS + 1] ^= w.buf[i];
#else
            /* x^12 = x^3 + 1 (libmceliece vec_mul, m = 12) */
            w.buf[i - GFBITS + 3] ^= w.buf[i];
#endif
            w.buf[i - GFBITS + 0] ^= w.buf[i];
        }
    }
    __syncwarp();
    if (lane < GFBITS) h[lane] = w.buf[lane];
    __syncwarp();
}

__device__ __forceinline__ void reg_vec_mul_c(uint64_t *h, const uint64_t *f, const uint64_t *gconst) {
    uint64_t g[GFBITS];
    #pragma unroll
    for (int j = 0; j < GFBITS; j++) g[j] = __ldg(gconst + j);
    vec_mul(h, f, g);
}

/* one layer of TAFFT (TR) or AFFT butterflies at stride d: lane j = butterfly j */
template <bool TR>
__device__ __forceinline__ void warp_bfly_layer(CCWarp &w, const CCTables &tb, int d, int lane) {
    int p = lane % d;
    int k = (lane / d) * 2 * d + p;
    const uint64_t *c = tb.consts + (d - 1 + p) * GFBITS;
    uint64_t a[GFBITS], bb[GFBITS], t[GFBITS];
    #pragma unroll
    for (int b = 0; b < GFBITS; b++) { a[b] = w.S[b][k]; bb[b] = w.S[b][k + d]; }
    if (TR) {
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) a[b] ^= bb[b];
        reg_vec_mul_c(t, a, c);
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) bb[b] ^= t[b];
    } else {
        reg_vec_mul_c(t, bb, c);
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) { a[b] ^= t[b]; bb[b] ^= a[b]; }
    }
    #pragma unroll
    for (int b = 0; b < GFBITS; b++) { w.S[b][k] = a[b]; w.S[b][k + d] = bb[b]; }
    __syncwarp();
}

__device__ __constant__ unsigned char cc_rev[64] = {
    0,32,16,48, 8,40,24,56, 4,36,20,52,12,44,28,60,
    2,34,18,50,10,42,26,58, 6,38,22,54,14,46,30,62,
    1,33,17,49, 9,41,25,57, 5,37,21,53,13,45,29,61,
    3,35,19,51,11,43,27,59, 7,39,23,55,15,47,31,63
};

#if GFBITS == 12
/* fft_tr of the reference on w.S, result in w.O[SPRIV_W][GFBITS] */
__device__ void warp_fft_tr(CCWarp &w, const CCTables &tb, int lane) {
    for (int d = 32; d >= 1; d >>= 1) warp_bfly_layer<true>(w, tb, d, lane);

    if (lane < GFBITS) {
        uint64_t *P = w.S[lane];
        /* buf[rev[j]] = in[j][i]; transpose_64x64(buf, buf); in[j][i] = buf[j] */
        for (int j = 0; j < 64; j++) {
            int r = cc_rev[j];
            if (j < r) { uint64_t x = P[j]; P[j] = P[r]; P[r] = x; }
        }
        const uint64_t masks[6][2] = {
            {0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL}, {0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL},
            {0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL}, {0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL},
            {0x0000FFFF0000FFFFULL, 0xFFFF0000FFFF0000ULL}, {0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL}};
        for (int dd = 5; dd >= 0; dd--) {
            int s = 1 << dd;
            for (int i = 0; i < 64; i += 2 * s)
                for (int j = i; j < i + s; j++) {
                    uint64_t x = (P[j] & masks[dd][0]) | ((P[j + s] & masks[dd][0]) << s);
                    uint64_t y = ((P[j] & masks[dd][1]) >> s) | (P[j + s] & masks[dd][1]);
                    P[j] = x;
                    P[j + s] = y;
                }
        }
        uint64_t pre0, pre1, pre2, pre3, pre4, pre5, o0;
        pre0 = P[32]; P[33] = P[33] ^ P[32]; pre1 = P[33]; P[35] = P[35] ^ P[33]; pre0 = pre0 ^ P[35]; P[34] = P[34] ^ P[35]; pre2 = P[34]; P[38] = P[38] ^ P[34]; pre0 = pre0 ^ P[38]; P[39] = P[39] ^ P[38]; pre1 = pre1 ^ P[39]; P[37] = P[37] ^ P[39]; pre0 = pre0 ^ P[37]; P[36] = P[36] ^ P[37]; pre3 = P[36]; P[44] = P[44] ^ P[36]; pre0 = pre0 ^ P[44]; P[45] = P[45] ^ P[44]; pre1 = pre1 ^ P[45]; P[47] = P[47] ^ P[45]; pre0 = pre0 ^ P[47]; P[46] = P[46] ^ P[47]; pre2 = pre2 ^ P[46]; P[42] = P[42] ^ P[46]; pre0 = pre0 ^ P[42]; P[43] = P[43] ^ P[42]; pre1 = pre1 ^ P[43]; P[41] = P[41] ^ P[43]; pre0 = pre0 ^ P[41]; P[40] = P[40] ^ P[41]; pre4 = P[40]; P[56] = P[56] ^ P[40]; pre0 = pre0 ^ P[56]; P[57] = P[57] ^ P[56]; pre1 = pre1 ^ P[57]; P[59] = P[59] ^ P[57]; pre0 = pre0 ^ P[59]; P[58] = P[58] ^ P[59]; pre2 = pre2 ^ P[58]; P[62] = P[62] ^ P[58]; pre0 = pre0 ^ P[62]; P[63] = P[63] ^ P[62]; pre1 = pre1 ^ P[63]; P[61] = P[61] ^ P[63]; pre0 = pre0 ^ P[61]; P[60] = P[60] ^ P[61]; pre3 = pre3 ^ P[60]; P[52] = P[52] ^ P[60]; pre0 = pre0 ^ P[52]; P[53] = P[53] ^ P[52]; pre1 = pre1 ^ P[53]; P[55] = P[55] ^ P[53]; pre0 = pre0 ^ P[55]; P[54] = P[54] ^ P[55]; pre2 = pre2 ^ P[54]; P[50] = P[50] ^ P[54]; pre0 = pre0 ^ P[50]; P[51] = P[51] ^ P[50]; pre1 = pre1 ^ P[51]; P[49] = P[49] ^ P[51]; pre0 = pre0 ^ P[49]; P[48] = P[48] ^ P[49]; pre5 = P[48]; P[16] = P[16] ^ P[48]; pre0 = pre0 ^ P[16]; P[17] = P[17] ^ P[16]; pre1 = pre1 ^ P[17]; P[19] = P[19] ^ P[17]; pre0 = pre0 ^ P[19]; P[18] = P[18] ^ P[19]; pre2 = pre2 ^ P[18]; P[22] = P[22] ^ P[18]; pre0 = pre0 ^ P[22]; P[23] = P[23] ^ P[22]; pre1 = pre1 ^ P[23]; P[21] = P[21] ^ P[23]; pre0 = pre0 ^ P[21]; P[20] = P[20] ^ P[21]; pre3 = pre3 ^ P[20]; P[28] = P[28] ^ P[20]; pre0 = pre0 ^ P[28]; P[29] = P[29] ^ P[28]; pre1 = pre1 ^ P[29]; P[31] = P[31] ^ P[29]; pre0 = pre0 ^ P[31]; P[30] = P[30] ^ P[31]; pre2 = pre2 ^ P[30]; P[26] = P[26] ^ P[30]; pre0 = pre0 ^ P[26]; P[27] = P[27] ^ P[26]; pre1 = pre1 ^ P[27]; P[25] = P[25] ^ P[27]; pre0 = pre0 ^ P[25]; P[24] = P[24] ^ P[25]; pre4 = pre4 ^ P[24]; P[8] = P[8] ^ P[24]; pre0 = pre0 ^ P[8]; P[9] = P[9] ^ P[8]; pre1 = pre1 ^ P[9]; P[11] = P[11] ^ P[9]; pre0 = pre0 ^ P[11]; P[10] = P[10] ^ P[11]; pre2 = pre2 ^ P[10]; P[14] = P[14] ^ P[10]; pre0 = pre0 ^ P[14]; P[15] = P[15] ^ P[14]; pre1 = pre1 ^ P[15]; P[13] = P[13] ^ P[15]; pre0 = pre0 ^ P[13]; P[12] = P[12] ^ P[13]; pre3 = pre3 ^ P[12]; P[4] = P[4] ^ P[12]; pre0 = pre0 ^ P[4]; P[5] = P[5] ^ P[4]; pre1 = pre1 ^ P[5]; P[7] = P[7] ^ P[5]; pre0 = pre0 ^ P[7]; P[6] = P[6] ^ P[7]; pre2 = pre2 ^ P[6]; P[2] = P[2] ^ P[6]; pre0 = pre0 ^ P[2]; P[3] = P[3] ^ P[2]; pre1 = pre1 ^ P[3]; P[1] = P[1] ^ P[3]; pre0 = pre0 ^ P[1]; o0 = P[0] ^ P[1];
        w.O[0][lane] = o0;
        w.PRE[0][lane] = pre0; w.PRE[1][lane] = pre1; w.PRE[2][lane] = pre2;
        w.PRE[3][lane] = pre3; w.PRE[4][lane] = pre4; w.PRE[5][lane] = pre5;
        const uint16_t beta[6] = {8, 1300, 3408, 1354, 2341, 1154};
        for (int i = 0; i < 6; i++) w.BETA[i][lane] = (uint64_t)(-(int64_t)((beta[i] >> lane) & 1));
    }
    __syncwarp();
    warp_vec_mul(w, w.O[1], w.PRE[0], w.BETA[0], lane);
    for (int i = 1; i < 6; i++) {
        warp_vec_mul(w, w.T, w.PRE[i], w.BETA[i], lane);
        if (lane < GFBITS) w.O[1][lane] ^= w.T[lane];
        __syncwarp();
    }
    /* radix_conversions_tr on O[2][12] */
    const uint64_t mask[6][2] = {
        {0x2222222222222222ULL, 0x4444444444444444ULL}, {0x0C0C0C0C0C0C0C0CULL, 0x3030303030303030ULL},
        {0x00F000F000F000F0ULL, 0x0F000F000F000F00ULL}, {0x0000FF000000FF00ULL, 0x00FF000000FF0000ULL},
        {0x00000000FFFF0000ULL, 0x0000FFFF00000000ULL}, {0xFFFFFFFF00000000ULL, 0x00000000FFFFFFFFULL}};
    for (int j = 5; j >= 0; j--) {
        if (j < 5) {
            warp_vec_mul(w, w.O[0], w.O[0], tb.scalars_2x + (j * 2 + 0) * GFBITS, lane);
            warp_vec_mul(w, w.O[1], w.O[1], tb.scalars_2x + (j * 2 + 1) * GFBITS, lane);
        }
        if (lane < GFBITS) {
            uint64_t x0 = w.O[0][lane], x1 = w.O[1][lane];
            for (int k = j; k <= 4; k++) {
                x0 ^= (x0 & mask[k][0]) << (1 << k);
                x0 ^= (x0 & mask[k][1]) << (1 << k);
                x1 ^= (x1 & mask[k][0]) << (1 << k);
                x1 ^= (x1 & mask[k][1]) << (1 << k);
            }
            x1 ^= (x0 & mask[5][0]) >> 32;
            x1 ^= (x1 & mask[5][1]) << 32;
            w.O[0][lane] = x0;
            w.O[1][lane] = x1;
        }
        __syncwarp();
    }
}

/* fft of the reference on the locator w.L, result in w.S */
__device__ void warp_fft(CCWarp &w, const CCTables &tb, int lane) {
    const uint64_t mask[5][2] = {
        {0x8888888888888888ULL, 0x4444444444444444ULL}, {0xC0C0C0C0C0C0C0C0ULL, 0x3030303030303030ULL},
        {0xF000F000F000F000ULL, 0x0F000F000F000F00ULL}, {0xFF000000FF000000ULL, 0x00FF000000FF0000ULL},
        {0xFFFF000000000000ULL, 0x0000FFFF00000000ULL}};
    for (int j = 0; j <= 4; j++) {
        if (lane < GFBITS) {
            uint64_t x = w.L[lane];
            for (int k = 4; k >= j; k--) {
                x ^= (x & mask[k][0]) >> (1 << k);
                x ^= (x & mask[k][1]) >> (1 << k);
            }
            w.L[lane] = x;
        }
        __syncwarp();
        warp_vec_mul(w, w.L, w.L, tb.scalars + j * GFBITS, lane);
    }
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        int v = lane + 32 * h, r = cc_rev[v];
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) w.S[b][v] = (uint64_t)(-(int64_t)((w.L[b] >> r) & 1));
    }
    __syncwarp();
    for (int d = 1; d <= 32; d <<= 1) warp_bfly_layer<false>(w, tb, d, lane);
#if HAS_FFT_POWERS
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        int v = lane + 32 * h;
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) w.S[b][v] ^= __ldg(tb.powers + v * GFBITS + b);
    }
#endif
    __syncwarp();
}

#else /* GFBITS >= 13: 128-point Gao-Mateer (T)AFFT, locator/syndrome in two rows */

/* reversal table of butterflies / butterflies_tr */
__device__ __constant__ unsigned char cc_rev7[128] = {
     0, 64, 32, 96, 16, 80, 48,112,  8, 72, 40,104, 24, 88, 56,120,
     4, 68, 36,100, 20, 84, 52,116, 12, 76, 44,108, 28, 92, 60,124,
     2, 66, 34, 98, 18, 82, 50,114, 10, 74, 42,106, 26, 90, 58,122,
     6, 70, 38,102, 22, 86, 54,118, 14, 78, 46,110, 30, 94, 62,126,
     1, 65, 33, 97, 17, 81, 49,113,  9, 73, 41,105, 25, 89, 57,121,
     5, 69, 37,101, 21, 85, 53,117, 13, 77, 45,109, 29, 93, 61,125,
     3, 67, 35, 99, 19, 83, 51,115, 11, 75, 43,107, 27, 91, 59,123,
     7, 71, 39,103, 23, 87, 55,119, 15, 79, 47,111, 31, 95, 63,127
};

/* in-place 64 x 64 bit transpose of P[base .. base+63] (transpose_64x64) */
__device__ __forceinline__ void cc_transpose64(uint64_t *P, int base) {
    const uint64_t m[6][2] = {
        {0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL}, {0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL},
        {0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL}, {0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL},
        {0x0000FFFF0000FFFFULL, 0xFFFF0000FFFF0000ULL}, {0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL}};
    for (int d = 5; d >= 0; d--) {
        int s = 1 << d;
        for (int i = 0; i < 64; i += 2 * s)
            for (int j = i; j < i + s; j++) {
                uint64_t a = P[base + j], b = P[base + j + s];
                P[base + j]     = (a & m[d][0]) | ((b & m[d][0]) << s);
                P[base + j + s] = ((a & m[d][1]) >> s) | (b & m[d][1]);
            }
    }
}

/* one layer of the 128-point network at stride s = 2 ... 64; lane handles the
 * butterflies lane and lane+32, the twiddle of (k, k+s) is fft_consts[s + (k-j)] */
template <bool TR>
__device__ __forceinline__ void warp_bfly_layer13(CCWarp &w, const CCTables &tb, int s, int lane) {
    for (int h = 0; h < 2; h++) {
        int q = lane + 32 * h, p = q % s, k = (q / s) * 2 * s + p;
        const uint64_t *c = tb.consts + (s + p) * GFBITS;
        uint64_t a[GFBITS], bb[GFBITS], t[GFBITS];
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) { a[b] = w.S[b][k]; bb[b] = w.S[b][k + s]; }
        if (TR) {
            #pragma unroll
            for (int b = 0; b < GFBITS; b++) a[b] ^= bb[b];
            reg_vec_mul_c(t, a, c);
            #pragma unroll
            for (int b = 0; b < GFBITS; b++) bb[b] ^= t[b];
        } else {
            reg_vec_mul_c(t, bb, c);
            #pragma unroll
            for (int b = 0; b < GFBITS; b++) { a[b] ^= t[b]; bb[b] ^= a[b]; }
        }
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) { w.S[b][k] = a[b]; w.S[b][k + s] = bb[b]; }
    }
    __syncwarp();
}

/* radix_conversions of the reference on the two locator rows w.L */
__device__ void warp_radix_conversions(CCWarp &w, const CCTables &tb, int lane) {
    const uint64_t mask[5][2] = {
        {0x8888888888888888ULL, 0x4444444444444444ULL}, {0xC0C0C0C0C0C0C0C0ULL, 0x3030303030303030ULL},
        {0xF000F000F000F000ULL, 0x0F000F000F000F00ULL}, {0xFF000000FF000000ULL, 0x00FF000000FF0000ULL},
        {0xFFFF000000000000ULL, 0x0000FFFF00000000ULL}};
    for (int j = 0; j <= 5; j++) {
        if (lane < GFBITS) {
            uint64_t x0 = w.L[lane], x1 = w.L[GFBITS + lane];
            x1 ^= x1 >> 32;
            x0 ^= x1 << 32;
            for (int k = 4; k >= j; k--) {
                x0 ^= (x0 & mask[k][0]) >> (1 << k);
                x0 ^= (x0 & mask[k][1]) >> (1 << k);
                x1 ^= (x1 & mask[k][0]) >> (1 << k);
                x1 ^= (x1 & mask[k][1]) >> (1 << k);
            }
            w.L[lane] = x0;
            w.L[GFBITS + lane] = x1;
        }
        __syncwarp();
        if (j < 5) {
            warp_vec_mul(w, w.L, w.L, tb.scalars_2x + (2 * j + 0) * GFBITS, lane);
            warp_vec_mul(w, w.L + GFBITS, w.L + GFBITS, tb.scalars_2x + (2 * j + 1) * GFBITS, lane);
        }
    }
}

/* radix_conversions_tr of the reference on the four output rows w.O */
__device__ void warp_radix_conversions_tr(CCWarp &w, const CCTables &tb, int lane) {
    const uint64_t mask[6][2] = {
        {0x2222222222222222ULL, 0x4444444444444444ULL}, {0x0C0C0C0C0C0C0C0CULL, 0x3030303030303030ULL},
        {0x00F000F000F000F0ULL, 0x0F000F000F000F00ULL}, {0x0000FF000000FF00ULL, 0x00FF000000FF0000ULL},
        {0x00000000FFFF0000ULL, 0x0000FFFF00000000ULL}, {0xFFFFFFFF00000000ULL, 0x00000000FFFFFFFFULL}};
    for (int j = 6; j >= 0; j--) {
        if (j < 6)
            for (int r = 0; r < 4; r++)
                warp_vec_mul(w, w.O[r], w.O[r], tb.scalars_4x + (4 * j + r) * GFBITS, lane);
        if (lane < GFBITS) {
            uint64_t x0 = w.O[0][lane], x1 = w.O[1][lane], x2 = w.O[2][lane], x3 = w.O[3][lane];
            for (int k = j; k <= 4; k++) {
                x0 ^= (x0 & mask[k][0]) << (1 << k); x0 ^= (x0 & mask[k][1]) << (1 << k);
                x1 ^= (x1 & mask[k][0]) << (1 << k); x1 ^= (x1 & mask[k][1]) << (1 << k);
                x2 ^= (x2 & mask[k][0]) << (1 << k); x2 ^= (x2 & mask[k][1]) << (1 << k);
                x3 ^= (x3 & mask[k][0]) << (1 << k); x3 ^= (x3 & mask[k][1]) << (1 << k);
            }
            if (j <= 5) {
                x1 ^= x0 >> 32; x1 ^= x1 << 32;
                x3 ^= x2 >> 32; x3 ^= x3 << 32;
            }
            x2 ^= x1; x3 ^= x2;
            w.O[0][lane] = x0; w.O[1][lane] = x1; w.O[2][lane] = x2; w.O[3][lane] = x3;
        }
        __syncwarp();
    }
}

/* fft of the reference on the locator w.L[2][GFBITS], result in w.S[GFBITS][128] */
__device__ void warp_fft(CCWarp &w, const CCTables &tb, int lane) {
    const uint16_t beta[7] = {2522, 7827, 7801, 8035, 6897, 8167, 3476};
    warp_radix_conversions(w, tb, lane);
    for (int t = 0; t < 7; t++) {
        uint64_t bt[GFBITS];
        #pragma unroll
        for (int j = 0; j < GFBITS; j++) bt[j] = (uint64_t)(-(int64_t)((beta[t] >> j) & 1));
        warp_vec_mul(w, w.PRE[t], w.L + GFBITS, bt, lane);
    }
    if (lane < GFBITS) {
        uint64_t *P = w.S[lane];
        P[0] = w.L[lane];
        for (int t = 0; t < 7; t++) {          /* buf[j] = in[0] ^ XOR_{t in bits(j)} pre[t] */
            int st = 1 << t;
            uint64_t pt = w.PRE[t][lane];
            for (int j = 0; j < st; j++) P[st + j] = P[j] ^ pt;
        }
        cc_transpose64(P, 0);                  /* transpose_64x64(buf +  0) */
        cc_transpose64(P, 64);                 /* transpose_64x64(buf + 64) */
        for (int j = 0; j < 128; j++) {        /* out[reversal[j]] = buf[j] */
            int r = cc_rev7[j];
            if (j < r) { uint64_t x = P[j]; P[j] = P[r]; P[r] = x; }
        }
    }
    __syncwarp();
    for (int s = 2; s <= 64; s <<= 1) warp_bfly_layer13<false>(w, tb, s, lane);
#if HAS_FFT_POWERS
    #pragma unroll
    for (int h = 0; h < 4; h++) {
        int v = lane + 32 * h;
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) w.S[b][v] ^= __ldg(tb.powers + v * GFBITS + b);
    }
#endif
    __syncwarp();
}

/* fft_tr of the reference on w.S[GFBITS][128], result in w.O[SPRIV_W][GFBITS] */
__device__ void warp_fft_tr(CCWarp &w, const CCTables &tb, int lane) {
    const uint16_t bt[6] = {5246, 5306, 6039, 6685, 4905, 6755};
    if (lane < GFBITS)                         /* BETA has GFBITS entries per row */
        #pragma unroll
        for (int i = 0; i < 6; i++) w.BETA[i][lane] = (uint64_t)(-(int64_t)((bt[i] >> lane) & 1));
    for (int s = 64; s >= 1; s >>= 1) warp_bfly_layer13<true>(w, tb, s, lane);

    if (lane < GFBITS) {
        uint64_t *P = w.S[lane];
        for (int j = 0; j < 128; j++) {        /* buf[reversal[j]] = in[j] */
            int r = cc_rev7[j];
            if (j < r) { uint64_t x = P[j]; P[j] = P[r]; P[r] = x; }
        }
        cc_transpose64(P, 0);
        cc_transpose64(P, 64);
    }
    __syncwarp();
    for (int k = 0; k < 2; k++) {
        if (lane < GFBITS) {
            uint64_t *P = w.S[lane];
            w.PRE[0][lane] = P[64 * k + 32]; P[64 * k + 33] ^= P[64 * k + 32];
            w.PRE[1][lane] = P[64 * k + 33]; P[64 * k + 35] ^= P[64 * k + 33];
            w.PRE[0][lane] ^= P[64 * k + 35]; P[64 * k + 34] ^= P[64 * k + 35];
            w.PRE[2][lane] = P[64 * k + 34]; P[64 * k + 38] ^= P[64 * k + 34];
            w.PRE[0][lane] ^= P[64 * k + 38]; P[64 * k + 39] ^= P[64 * k + 38];
            w.PRE[1][lane] ^= P[64 * k + 39]; P[64 * k + 37] ^= P[64 * k + 39];
            w.PRE[0][lane] ^= P[64 * k + 37]; P[64 * k + 36] ^= P[64 * k + 37];
            w.PRE[3][lane] = P[64 * k + 36]; P[64 * k + 44] ^= P[64 * k + 36];
            w.PRE[0][lane] ^= P[64 * k + 44]; P[64 * k + 45] ^= P[64 * k + 44];
            w.PRE[1][lane] ^= P[64 * k + 45]; P[64 * k + 47] ^= P[64 * k + 45];
            w.PRE[0][lane] ^= P[64 * k + 47]; P[64 * k + 46] ^= P[64 * k + 47];
            w.PRE[2][lane] ^= P[64 * k + 46]; P[64 * k + 42] ^= P[64 * k + 46];
            w.PRE[0][lane] ^= P[64 * k + 42]; P[64 * k + 43] ^= P[64 * k + 42];
            w.PRE[1][lane] ^= P[64 * k + 43]; P[64 * k + 41] ^= P[64 * k + 43];
            w.PRE[0][lane] ^= P[64 * k + 41]; P[64 * k + 40] ^= P[64 * k + 41];
            w.PRE[4][lane] = P[64 * k + 40]; P[64 * k + 56] ^= P[64 * k + 40];
            w.PRE[0][lane] ^= P[64 * k + 56]; P[64 * k + 57] ^= P[64 * k + 56];
            w.PRE[1][lane] ^= P[64 * k + 57]; P[64 * k + 59] ^= P[64 * k + 57];
            w.PRE[0][lane] ^= P[64 * k + 59]; P[64 * k + 58] ^= P[64 * k + 59];
            w.PRE[2][lane] ^= P[64 * k + 58]; P[64 * k + 62] ^= P[64 * k + 58];
            w.PRE[0][lane] ^= P[64 * k + 62]; P[64 * k + 63] ^= P[64 * k + 62];
            w.PRE[1][lane] ^= P[64 * k + 63]; P[64 * k + 61] ^= P[64 * k + 63];
            w.PRE[0][lane] ^= P[64 * k + 61]; P[64 * k + 60] ^= P[64 * k + 61];
            w.PRE[3][lane] ^= P[64 * k + 60]; P[64 * k + 52] ^= P[64 * k + 60];
            w.PRE[0][lane] ^= P[64 * k + 52]; P[64 * k + 53] ^= P[64 * k + 52];
            w.PRE[1][lane] ^= P[64 * k + 53]; P[64 * k + 55] ^= P[64 * k + 53];
            w.PRE[0][lane] ^= P[64 * k + 55]; P[64 * k + 54] ^= P[64 * k + 55];
            w.PRE[2][lane] ^= P[64 * k + 54]; P[64 * k + 50] ^= P[64 * k + 54];
            w.PRE[0][lane] ^= P[64 * k + 50]; P[64 * k + 51] ^= P[64 * k + 50];
            w.PRE[1][lane] ^= P[64 * k + 51]; P[64 * k + 49] ^= P[64 * k + 51];
            w.PRE[0][lane] ^= P[64 * k + 49]; P[64 * k + 48] ^= P[64 * k + 49];
            w.PRE[5][lane] = P[64 * k + 48]; P[64 * k + 16] ^= P[64 * k + 48];
            w.PRE[0][lane] ^= P[64 * k + 16]; P[64 * k + 17] ^= P[64 * k + 16];
            w.PRE[1][lane] ^= P[64 * k + 17]; P[64 * k + 19] ^= P[64 * k + 17];
            w.PRE[0][lane] ^= P[64 * k + 19]; P[64 * k + 18] ^= P[64 * k + 19];
            w.PRE[2][lane] ^= P[64 * k + 18]; P[64 * k + 22] ^= P[64 * k + 18];
            w.PRE[0][lane] ^= P[64 * k + 22]; P[64 * k + 23] ^= P[64 * k + 22];
            w.PRE[1][lane] ^= P[64 * k + 23]; P[64 * k + 21] ^= P[64 * k + 23];
            w.PRE[0][lane] ^= P[64 * k + 21]; P[64 * k + 20] ^= P[64 * k + 21];
            w.PRE[3][lane] ^= P[64 * k + 20]; P[64 * k + 28] ^= P[64 * k + 20];
            w.PRE[0][lane] ^= P[64 * k + 28]; P[64 * k + 29] ^= P[64 * k + 28];
            w.PRE[1][lane] ^= P[64 * k + 29]; P[64 * k + 31] ^= P[64 * k + 29];
            w.PRE[0][lane] ^= P[64 * k + 31]; P[64 * k + 30] ^= P[64 * k + 31];
            w.PRE[2][lane] ^= P[64 * k + 30]; P[64 * k + 26] ^= P[64 * k + 30];
            w.PRE[0][lane] ^= P[64 * k + 26]; P[64 * k + 27] ^= P[64 * k + 26];
            w.PRE[1][lane] ^= P[64 * k + 27]; P[64 * k + 25] ^= P[64 * k + 27];
            w.PRE[0][lane] ^= P[64 * k + 25]; P[64 * k + 24] ^= P[64 * k + 25];
            w.PRE[4][lane] ^= P[64 * k + 24]; P[64 * k + 8] ^= P[64 * k + 24];
            w.PRE[0][lane] ^= P[64 * k + 8]; P[64 * k + 9] ^= P[64 * k + 8];
            w.PRE[1][lane] ^= P[64 * k + 9]; P[64 * k + 11] ^= P[64 * k + 9];
            w.PRE[0][lane] ^= P[64 * k + 11]; P[64 * k + 10] ^= P[64 * k + 11];
            w.PRE[2][lane] ^= P[64 * k + 10]; P[64 * k + 14] ^= P[64 * k + 10];
            w.PRE[0][lane] ^= P[64 * k + 14]; P[64 * k + 15] ^= P[64 * k + 14];
            w.PRE[1][lane] ^= P[64 * k + 15]; P[64 * k + 13] ^= P[64 * k + 15];
            w.PRE[0][lane] ^= P[64 * k + 13]; P[64 * k + 12] ^= P[64 * k + 13];
            w.PRE[3][lane] ^= P[64 * k + 12]; P[64 * k + 4] ^= P[64 * k + 12];
            w.PRE[0][lane] ^= P[64 * k + 4]; P[64 * k + 5] ^= P[64 * k + 4];
            w.PRE[1][lane] ^= P[64 * k + 5]; P[64 * k + 7] ^= P[64 * k + 5];
            w.PRE[0][lane] ^= P[64 * k + 7]; P[64 * k + 6] ^= P[64 * k + 7];
            w.PRE[2][lane] ^= P[64 * k + 6]; P[64 * k + 2] ^= P[64 * k + 6];
            w.PRE[0][lane] ^= P[64 * k + 2]; P[64 * k + 3] ^= P[64 * k + 2];
            w.PRE[1][lane] ^= P[64 * k + 3]; P[64 * k + 1] ^= P[64 * k + 3];

            w.PRE[0][lane] ^= P[64 * k + 1]; w.O[k][lane] = P[64 * k + 0] ^ P[64 * k + 1];
        }
        __syncwarp();
        warp_vec_mul(w, w.O[2 + k], w.PRE[0], w.BETA[0], lane);
        for (int i = 1; i < 6; i++) {
            warp_vec_mul(w, w.T, w.PRE[i], w.BETA[i], lane);
            if (lane < GFBITS) w.O[2 + k][lane] ^= w.T[lane];
            __syncwarp();
        }
    }
    __syncwarp();
    warp_radix_conversions_tr(w, tb, lane);
#if HAS_FFT_TR_POSTPROCESS
    postprocess(w.O);                  /* ref/<set>.cuh: zeroes row 3 (SYS_T < 128) */
#endif
}

#endif /* GFBITS == 12 */

/* s_out = fft_tr(inv & in) for in = recv (front end) or error (re-encryption).
 * All F_OUTW words are written: with SPRIV_W == 4 the last two rows are scratch
 * inside F_OUTW, and the final kernel compares all F_OUTW words of both sides. */
__global__ void __launch_bounds__(32 * CC_WARPS) kernel_cc_tafft(
    uint64_t *d_s, const uint64_t *d_in, const uint64_t *d_inv, CCTables tb, int N)
{
    extern __shared__ __align__(16) unsigned char cc_smem[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * CC_WARPS + warp;
    if (idx >= N) return;
    CCWarp &w = reinterpret_cast<CCWarp *>(cc_smem)[warp];
    const uint64_t *in = d_in + (size_t)idx * NVEC;
    #pragma unroll
    for (int h = 0; h < NVEC / 32; h++) {
        int v = lane + 32 * h;
        uint64_t r = in[v];
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) w.S[b][v] = __ldg(d_inv + v * GFBITS + b) & r;
    }
    __syncwarp();
    warp_fft_tr(w, tb, lane);
    for (int o = lane; o < F_OUTW; o += 32)
        d_s[(size_t)idx * F_OUTW + o] = reinterpret_cast<const uint64_t *>(&w.O[0][0])[o];
}

/* error[i] = NOT OR_b fft(locator)[i][b], in the same word order as d_err of the
 * TC pipeline (vec error[NVEC]); one warp per operation */
__global__ void __launch_bounds__(32 * CC_WARPS) kernel_cc_afft(
    uint64_t *d_err, const uint64_t *d_loc, CCTables tb, int N)
{
    extern __shared__ __align__(16) unsigned char cc_smem[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * CC_WARPS + warp;
    if (idx >= N) return;
    CCWarp &w = reinterpret_cast<CCWarp *>(cc_smem)[warp];
    if (lane < LOC_W * GFBITS) w.L[lane] = d_loc[(size_t)idx * LOC_W * GFBITS + lane];
    __syncwarp();
    warp_fft(w, tb, lane);
    #pragma unroll
    for (int h = 0; h < NVEC / 32; h++) {
        int v = lane + 32 * h;
        uint64_t r = 0;
        #pragma unroll
        for (int b = 0; b < GFBITS; b++) r |= w.S[b][v];
        d_err[(size_t)idx * NVEC + v] = r ^ vec_setbits(1);
    }
}

/* s_priv = F c by table lookup: one warp per operation, lane w owns the output
 * words w, w + 32, ... ; the ciphertext address is the same in every lane */
__global__ void __launch_bounds__(32 * CC_WARPS) kernel_cc_front(
    uint64_t *d_spriv, const unsigned char *d_cs, const uint64_t *d_ftab, int N)
{
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * CC_WARPS + warp;
    if (idx >= N) return;
    const unsigned char *c = d_cs + (size_t)idx * SYND_BYTES;
    uint64_t acc[CC_OUT_SLOTS];
    #pragma unroll
    for (int k = 0; k < CC_OUT_SLOTS; k++) acc[k] = 0;
    #pragma unroll 4
    for (int q = 0; q < CC_CHUNKS; q++) {
        int nib = (c[q >> 1] >> (4 * (q & 1))) & 15;   /* same address in all lanes: broadcast */
        #pragma unroll
        for (int k = 0; k < CC_OUT_SLOTS; k++) {
            int o = lane + 32 * k;
            if (o < F_OUTW) acc[k] ^= __ldg(d_ftab + ((size_t)q * 16 + nib) * F_OUTW + o);
        }
    }
    #pragma unroll
    for (int k = 0; k < CC_OUT_SLOTS; k++) {
        int o = lane + 32 * k;
        if (o < F_OUTW) d_spriv[(size_t)idx * F_OUTW + o] = acc[k];
    }
}

/* ---- CC_SENTINEL ---- */

#endif /* CC_WARP_OK */

/* Host: table of the linear front end F, entry (chunk q, nibble v) = XOR of the
 * F columns of the set bits of v; the nibble covers the ciphertext bits 4q ... 4q+3. */
static inline void cc_build_ftab(std::vector<uint64_t> &ftab, const std::vector<uint64_t> &Fc)
{
    ftab.assign(CC_TAB_WORDS, 0);
    for (int q = 0; q < CC_CHUNKS; q++)
        for (int v = 1; v < 16; v++) {
            int t0 = __builtin_ctz(v);
            const uint64_t *lo = ftab.data() + ((size_t)q * 16 + (v & (v - 1))) * F_OUTW;
            const uint64_t *col = Fc.data() + (size_t)(4 * q + t0) * F_OUTW;
            uint64_t *dst = ftab.data() + ((size_t)q * 16 + v) * F_OUTW;
            for (int w = 0; w < F_OUTW; w++) dst[w] = lo[w] ^ col[w];
        }
}

#endif
