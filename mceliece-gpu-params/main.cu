/* Multi-parameter Classic McEliece GPU benchmark.
 * Built once per parameter set:  -DPARAM_ID=<set> -include ref/<set>.cuh
 * Encapsulation: CUDA-core reference, 1-bit BMMA, INT8 bs2-m4 / bs2-blk3.
 * Decapsulation: CUDA-core warp-cooperative (table front end + warp (T)AFFT) and
 * Tensor Core pipeline (F, A, G on 1-bit TCs).
 * Every engine is timed in the PKE and the KEM setting and checked against libmceliece. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <mceliece.h>
#include "bench_common.h"
#include "tc_kernels.cuh"
#include "cc_kernels.cuh"

#define CAT3_(a, b, c) a##b##c
#define CAT3(a, b, c) CAT3_(a, b, c)
#define CPU_KEYPAIR CAT3(mceliece_kem_, PARAM_ID, _keypair)
#define CPU_ENC     CAT3(mceliece_kem_, PARAM_ID, _enc)
#define CPU_DEC     CAT3(mceliece_kem_, PARAM_ID, _dec)

#ifndef NMAX
#define NMAX 65536
#endif
#ifndef N_INVALID
#define N_INVALID 1024
#endif
#define BREAKDOWN_N 8192

#define CHECK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while (0)

static const int BATCHES[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
static int NB = 0;   /* batches <= NMAX, set in main */

static cudaEvent_t ev0, ev1;
template <typename F> static double time_ops(int N, F body)
{
    body();
    bench_check("warm-up");
    cudaEventRecord(ev0);
    for (int t = 0; t < N_TRIALS; t++) body();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    bench_check("timed passes");
    float ms;
    cudaEventElapsedTime(&ms, ev0, ev1);
    return (double)N / (ms / N_TRIALS / 1000.0);
}

/* column-major GF(2) matrix (ncols x outw uint64) -> row-major bit rows (rows x kw uint32) */
static void pack_rows(std::vector<uint32_t> &dst, const std::vector<uint64_t> &cols, int ncols, int outw, int kw)
{
    dst.assign((size_t)outw * 64 * kw, 0);
    for (int j = 0; j < ncols; j++)
        for (int w = 0; w < outw; w++) {
            uint64_t v = cols[(size_t)j * outw + w];
            while (v) {
                int b = __builtin_ctzll(v);
                v &= v - 1;
                dst[(size_t)(64 * w + b) * kw + (j >> 5)] |= 1u << (j & 31);
            }
        }
}

static void matvec(const std::vector<uint64_t> &cols, int ncols, int outw, const uint64_t *x, uint64_t *out)
{
    memset(out, 0, (size_t)outw * 8);
    for (int j = 0; j < ncols; j++)
        if ((x[j >> 6] >> (j & 63)) & 1)
            for (int w = 0; w < outw; w++) out[w] ^= cols[(size_t)j * outw + w];
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng64(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }

/* build one matrix on the device and verify it against the reference oracle on random inputs */
static bool build_matrix(int which, const char *name, int ncols, int outw, int in_words, const unsigned char *d_sk,
                         std::vector<uint64_t> &cols, std::vector<uint64_t> *cconst)
{
    int nthreads = ncols + (which == 1 ? 1 : 0);
    uint64_t *d_cols, *d_x, *d_y;
    CHECK(cudaMalloc((void **)&d_cols, (size_t)nthreads * outw * 8));
    CHECK(cudaMalloc((void **)&d_x, (size_t)in_words * 8));
    CHECK(cudaMalloc((void **)&d_y, (size_t)outw * 8));
    kernel_build_cols<<<(nthreads + 63) / 64, 64>>>(d_cols, which, d_sk, nthreads);
    bench_check("matrix extraction");
    cols.resize((size_t)nthreads * outw);
    CHECK(cudaMemcpy(cols.data(), d_cols, cols.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<uint64_t> c0(outw, 0);
    if (which == 1) {                                  /* affine constant = response to 0 */
        for (int w = 0; w < outw; w++) c0[w] = cols[(size_t)ncols * outw + w];
        for (int j = 0; j < ncols; j++)
            for (int w = 0; w < outw; w++) cols[(size_t)j * outw + w] ^= c0[w];
        cols.resize((size_t)ncols * outw);
        if (cconst) *cconst = c0;
    }
    bool ok = true;
    std::vector<uint64_t> x(in_words), got(outw), want(outw);
    for (int trial = 0; trial < 4 && ok; trial++) {
        for (int w = 0; w < in_words; w++) x[w] = (trial == 0) ? 0 : rng64();
        if (which == 0 && F_COLS % 64)                 /* only SYND_BYTES bytes are inputs */
            x[in_words - 1] &= (F_COLS % 64) ? ((1ULL << (F_COLS % 64)) - 1) : ~0ULL;
        CHECK(cudaMemcpy(d_x, x.data(), (size_t)in_words * 8, cudaMemcpyHostToDevice));
        kernel_oracle<<<1, 1>>>(d_y, which, d_sk, d_x);
        bench_check("matrix oracle");
        CHECK(cudaMemcpy(got.data(), d_y, (size_t)outw * 8, cudaMemcpyDeviceToHost));
        matvec(cols, ncols, outw, x.data(), want.data());
        for (int w = 0; w < outw; w++) want[w] ^= c0[w];
        if (memcmp(got.data(), want.data(), (size_t)outw * 8) != 0) ok = false;
    }
    printf("[matrix] %s: %d x %d, %s\n", name, outw * 64, ncols, ok ? "oracle check OK" : "oracle check FAILED");
    cudaFree(d_cols); cudaFree(d_x); cudaFree(d_y);
    return ok;
}

static int cpu_check_keys(const unsigned char *cs, const unsigned char *keys, const unsigned char *sk, int n)
{
    int bad = 0;
    for (int i = 0; i < n; i++) {
        unsigned char k[32];
        CPU_DEC(k, cs + (size_t)i * SYND_BYTES, sk);
        if (memcmp(k, keys + (size_t)i * 32, 32) != 0) bad++;
    }
    return bad;
}

static int count_mismatch(const unsigned char *a, const unsigned char *b, int n)
{
    int bad = 0;
    for (int i = 0; i < n; i++) if (memcmp(a + (size_t)i * 32, b + (size_t)i * 32, 32) != 0) bad++;
    return bad;
}

int main(void)
{
    printf("== mceliece-gpu-params: %s (m=%d n=%d t=%d mt=%d k=%d) ==\n", PARAM_NAME, GFBITS, SYS_N, SYS_T,
           PK_NROWS, PK_NCOLS);
    bench_print_device();
    bench_set_stack_limit({(const void *)kernel_cc_decap<true>, (const void *)kernel_cc_decap<false>,
                           (const void *)kernel_cc_encap_kem, (const void *)kernel_cc_encap_pke,
#if CC_WARP_OK
                           (const void *)kernel_cc_scaling_inv, (const void *)kernel_cc_front,
                           (const void *)kernel_cc_tafft, (const void *)kernel_cc_afft,
#endif
                           (const void *)kernel_tc_tail_pke, (const void *)kernel_tc_tail_kem,
                           (const void *)kernel_tc_error_extract, (const void *)kernel_tc_bm,
                           (const void *)kernel_build_cols, (const void *)kernel_oracle,
                           (const void *)kernel_gen_e_pack_bmma, (const void *)kernel_gen_e_pack_bs2,
                           (const void *)kernel_finalize_encap, (const void *)kernel_gen_e});
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    bool all_ok = true;
    while (NB < (int)(sizeof(BATCHES) / sizeof(BATCHES[0])) && BATCHES[NB] <= NMAX) NB++;

    /* ---------------- keys and CPU reference data ---------------- */
    std::vector<unsigned char> pk(PK_BYTES), sk(SK_BYTES);
    CPU_KEYPAIR(pk.data(), sk.data());
    std::vector<unsigned char> ct((size_t)NMAX * SYND_BYTES), keys_cpu((size_t)NMAX * 32);
    for (int i = 0; i < NMAX; i++) {
        CPU_ENC(ct.data() + (size_t)i * SYND_BYTES, keys_cpu.data() + (size_t)i * 32, pk.data());
    }
    std::vector<unsigned char> ct_inv((size_t)N_INVALID * SYND_BYTES), keys_inv((size_t)N_INVALID * 32);
    for (size_t i = 0; i < ct_inv.size(); i++) ct_inv[i] = (unsigned char)rng64();
    for (int i = 0; i < N_INVALID; i++)
        CPU_DEC(keys_inv.data() + (size_t)i * 32, ct_inv.data() + (size_t)i * SYND_BYTES, sk.data());
    printf("[host] keypair + %d valid and %d random ciphertexts with CPU keys\n", NMAX, N_INVALID);

    unsigned char *d_pk, *d_sk, *d_ct, *d_ct_inv, *d_seeds, *d_e, *d_cs, *d_keys, *d_S;
    CHECK(cudaMalloc((void **)&d_pk, PK_BYTES));
    CHECK(cudaMalloc((void **)&d_sk, SK_BYTES));
    CHECK(cudaMalloc((void **)&d_ct, (size_t)NMAX * SYND_BYTES));
    CHECK(cudaMalloc((void **)&d_ct_inv, (size_t)N_INVALID * SYND_BYTES));
    CHECK(cudaMalloc((void **)&d_seeds, (size_t)NMAX * 32));
    CHECK(cudaMalloc((void **)&d_e, (size_t)NMAX * E_BYTES));
    CHECK(cudaMalloc((void **)&d_cs, (size_t)NMAX * SYND_BYTES));
    CHECK(cudaMalloc((void **)&d_keys, (size_t)NMAX * 32));
    CHECK(cudaMalloc((void **)&d_S, (size_t)PK_NROWS * NMAX));
    CHECK(cudaMemcpy(d_pk, pk.data(), PK_BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sk, sk.data(), SK_BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ct, ct.data(), ct.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ct_inv, ct_inv.data(), ct_inv.size(), cudaMemcpyHostToDevice));
    {
        std::vector<unsigned char> seeds((size_t)NMAX * 32);
        for (size_t i = 0; i < seeds.size(); i++) seeds[i] = (unsigned char)rng64();
        CHECK(cudaMemcpy(d_seeds, seeds.data(), seeds.size(), cudaMemcpyHostToDevice));
    }

    /* ---------------- encapsulation operands ---------------- */
    std::vector<uint32_t> T_bmma((size_t)PK_NROWS * ENC_KW);
    pack_pk_bmma(T_bmma.data(), pk.data());
    std::vector<int8_t> T_int8((size_t)BS2_M_PADDED * BS2_K_PADDED);
    pack_pk_int8(T_int8.data(), pk.data());
    uint32_t *d_T_bmma, *d_E_bmma;
    int8_t *d_T_int8, *d_E_bs2;
    CHECK(cudaMalloc((void **)&d_T_bmma, T_bmma.size() * 4));
    CHECK(cudaMalloc((void **)&d_T_int8, T_int8.size()));
    CHECK(cudaMalloc((void **)&d_E_bmma, (size_t)ENC_KW * NMAX * 4));
    CHECK(cudaMalloc((void **)&d_E_bs2, (size_t)BS2_K_PADDED * BS2_PAIRS_PADDED(NMAX) * 2));
    CHECK(cudaMemcpy(d_T_bmma, T_bmma.data(), T_bmma.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_T_int8, T_int8.data(), T_int8.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_E_bs2, 0, (size_t)BS2_K_PADDED * BS2_PAIRS_PADDED(NMAX) * 2));

    std::vector<unsigned char> h_cs((size_t)NMAX * SYND_BYTES), h_keys((size_t)NMAX * 32);

    /* engine order: CC, BMMA, BS2-M4, BS2-BLK3 */
    const int NE = 4;
    const char *enc_names[NE] = {"CC", "BMMA", "BS2-M4", "BS2-BLK3"};
    double enc_pke[16][NE], enc_kem[16][NE];
    int enc_ok[16];

    for (int bi = 0; bi < NB; bi++) {
        int N = BATCHES[bi], bad = 0, blk = (N + 31) / 32;
        int pairs = BS2_PAIRS_PADDED(N);
        auto verify = [&](void) {
            int n = N < 64 ? N : 64;
            CHECK(cudaMemcpy(h_cs.data(), d_cs, (size_t)n * SYND_BYTES, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_keys.data(), d_keys, (size_t)n * 32, cudaMemcpyDeviceToHost));
            bad += cpu_check_keys(h_cs.data(), h_keys.data(), sk.data(), n);
        };
        /* CUDA cores: reference syndrome per thread */
        kernel_gen_e<<<blk, 32>>>(d_e, d_seeds, N);
        enc_pke[bi][0] = time_ops(N, [&] { kernel_cc_encap_pke<<<blk, 32>>>(d_cs, d_keys, d_pk, d_e, N); });
        verify();
        enc_kem[bi][0] = time_ops(N, [&] { kernel_cc_encap_kem<<<blk, 32>>>(d_cs, d_keys, d_pk, d_seeds, N); });
        verify();
        /* 1-bit BMMA */
        dim3 g_bmma((N + ENC_BN - 1) / ENC_BN, (PK_NROWS + ENC_BM - 1) / ENC_BM), b4(32, 4);
        auto gen_bmma = [&] { kernel_gen_e_pack_bmma<<<blk, 32>>>(d_e, d_E_bmma, d_seeds, N); };
        auto body_bmma = [&] {
            kernel_bmma_syndrome<<<g_bmma, b4>>>(d_S, d_T_bmma, d_E_bmma, N);
            kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, d_S, d_e, N);
        };
        gen_bmma();
        enc_pke[bi][1] = time_ops(N, body_bmma);
        verify();
        enc_kem[bi][1] = time_ops(N, [&] { gen_bmma(); body_bmma(); });
        verify();
        /* INT8 bs2-m4 and bs2-blk3 */
        auto gen_bs2 = [&] { kernel_gen_e_pack_bs2<<<(N / 2 + 255) / 256, 256>>>(d_e, d_E_bs2, d_seeds, N); };
        dim3 g_m4(pairs / 16, (BS2_M_PADDED + BS2_M_CHUNK * 16 - 1) / (BS2_M_CHUNK * 16));
        auto body_m4 = [&] {
            kernel_bs2_m4<<<g_m4, 32>>>(d_S, d_T_int8, d_E_bs2, N, pairs);
            kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, d_S, d_e, N);
        };
        gen_bs2();
        enc_pke[bi][2] = time_ops(N, body_m4);
        verify();
        enc_kem[bi][2] = time_ops(N, [&] { gen_bs2(); body_m4(); });
        verify();
        dim3 g_blk(pairs / 64, (BS2_M_PADDED + 63) / 64);
        auto body_blk = [&] {
            kernel_bs2_blk3<<<g_blk, 128>>>(d_S, d_T_int8, d_E_bs2, N, pairs);
            kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, d_S, d_e, N);
        };
        gen_bs2();
        enc_pke[bi][3] = time_ops(N, body_blk);
        verify();
        enc_kem[bi][3] = time_ops(N, [&] { gen_bs2(); body_blk(); });
        verify();
        enc_ok[bi] = (bad == 0);
        if (bad) all_ok = false;
        fprintf(stderr, "  encap N=%d done\n", N);
    }
    for (int conv = 0; conv < 2; conv++) {
        printf("\n[ENCAP-%s] %s\n%8s", conv ? "KEM" : "PKE",
               conv ? "fixed-weight sampling and packing timed" : "e generated before timing", "Batch N");
        for (int e = 0; e < NE; e++) printf(" | %12s", enc_names[e]);
        printf(" | Verification\n");
        for (int bi = 0; bi < NB; bi++) {
            printf("%8d", BATCHES[bi]);
            for (int e = 0; e < NE; e++) printf(" | %12.0f", conv ? enc_kem[bi][e] : enc_pke[bi][e]);
            printf(" | %s\n", enc_ok[bi] ? "PASSED" : "FAILED");
        }
    }
    cudaFree(d_E_bmma); cudaFree(d_E_bs2); cudaFree(d_S); cudaFree(d_T_int8); cudaFree(d_T_bmma);

    /* ---------------- Tensor Core decapsulation matrices ---------------- */
    std::vector<uint64_t> Fc, Ac, Gc, cA;
    bool mok = build_matrix(0, "F (front end)", F_COLS, F_OUTW, (F_COLS + 63) / 64, d_sk, Fc, nullptr);
    mok &= build_matrix(1, "A (AFFT)", A_COLS, A_OUTW, A_COLS / 64, d_sk, Ac, &cA);
    mok &= build_matrix(2, "G (re-encryption)", G_COLS, F_OUTW, G_COLS / 64, d_sk, Gc, nullptr);
    if (!mok) all_ok = false;
    std::vector<uint32_t> Fp, Ap, Gp;
    pack_rows(Fp, Fc, F_COLS, F_OUTW, F_KW);
    pack_rows(Ap, Ac, A_COLS, A_OUTW, A_KW);
    pack_rows(Gp, Gc, G_COLS, F_OUTW, G_KW);
    printf("[matrix] packed: F %.1f KB, A %.1f MB, G %.1f KB\n", Fp.size() * 4 / 1024.0, Ap.size() * 4 / 1048576.0,
           Gp.size() * 4 / 1024.0);

    uint32_t *d_F, *d_A, *d_G, *d_c32, *d_loc32, *d_err32;
    uint64_t *d_cA, *d_spriv, *d_scmp, *d_loc, *d_eval, *d_err;
    CHECK(cudaMalloc((void **)&d_F, Fp.size() * 4));
    CHECK(cudaMalloc((void **)&d_A, Ap.size() * 4));
    CHECK(cudaMalloc((void **)&d_G, Gp.size() * 4));
    CHECK(cudaMalloc((void **)&d_cA, (size_t)A_OUTW * 8));
    CHECK(cudaMalloc((void **)&d_c32, (size_t)F_KW * NMAX * 4));
    CHECK(cudaMalloc((void **)&d_loc32, (size_t)A_KW * NMAX * 4));
    CHECK(cudaMalloc((void **)&d_err32, (size_t)G_KW * NMAX * 4));
    CHECK(cudaMalloc((void **)&d_spriv, (size_t)F_OUTW * NMAX * 8));
    CHECK(cudaMalloc((void **)&d_scmp, (size_t)F_OUTW * NMAX * 8));
    CHECK(cudaMalloc((void **)&d_loc, (size_t)LOC_W * GFBITS * NMAX * 8));
    CHECK(cudaMalloc((void **)&d_eval, (size_t)A_OUTW * NMAX * 8));
    CHECK(cudaMalloc((void **)&d_err, (size_t)NVEC * NMAX * 8));
    CHECK(cudaMemcpy(d_F, Fp.data(), Fp.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A, Ap.data(), Ap.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_G, Gp.data(), Gp.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cA, cA.data(), (size_t)A_OUTW * 8, cudaMemcpyHostToDevice));

    /* ---------------- CUDA-core warp-cooperative decapsulation ---------------- */
#if CC_WARP_OK
    CC_CHECK(cudaFuncSetAttribute(kernel_cc_afft, cudaFuncAttributeMaxDynamicSharedMemorySize, CC_SMEM));
    CC_CHECK(cudaFuncSetAttribute(kernel_cc_tafft, cudaFuncAttributeMaxDynamicSharedMemorySize, CC_SMEM));
    std::vector<uint64_t> ftab;
    cc_build_ftab(ftab, Fc);
    uint64_t *d_ftab, *d_inv;
    CHECK(cudaMalloc((void **)&d_ftab, ftab.size() * 8));
    CHECK(cudaMemcpy(d_ftab, ftab.data(), ftab.size() * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void **)&d_inv, (size_t)NVEC * GFBITS * 8));
    kernel_cc_scaling_inv<<<1, 1>>>(d_inv, d_sk);
    bench_check("scaling inversion");
    CCTables cc_tb = cc_tables();
    printf("[cc] front end by table lookup: %.2f MB (%d chunks x 16 nibbles x %d words)\n",
           ftab.size() * 8 / 1048576.0, CC_CHUNKS, F_OUTW);
    auto cc_front = [&](const unsigned char *cts, int N) {
        int wb = (N + CC_WARPS - 1) / CC_WARPS;
        kernel_cc_front<<<wb, 32 * CC_WARPS>>>(d_spriv, cts, d_ftab, N);
        kernel_tc_bm<<<(N + 127) / 128, 128>>>(d_loc, d_spriv, N);
        kernel_cc_afft<<<wb, 32 * CC_WARPS, CC_SMEM>>>(d_err, d_loc, cc_tb, N);
    };
    auto cc_pke = [&](const unsigned char *cts, int N) {
        cc_front(cts, N);
        kernel_tc_tail_kem<<<(N + 127) / 128, 128>>>(d_keys, cts, d_sk, d_err, d_spriv, d_spriv, N);
    };
    auto cc_kem = [&](const unsigned char *cts, int N) {
        int wb = (N + CC_WARPS - 1) / CC_WARPS;
        cc_front(cts, N);
        kernel_cc_tafft<<<wb, 32 * CC_WARPS, CC_SMEM>>>(d_scmp, d_err, d_inv, cc_tb, N);
        kernel_tc_tail_kem<<<(N + 127) / 128, 128>>>(d_keys, cts, d_sk, d_err, d_spriv, d_scmp, N);
    };
#else
    /* m = 13: the warp (T)AFFT is not ported yet, keep the reference CUDA-core code */
    auto cc_pke = [&](const unsigned char *cts, int N) {
        kernel_cc_decap<false><<<(N + 31) / 32, 32>>>(d_keys, cts, d_sk, N);
    };
    auto cc_kem = [&](const unsigned char *cts, int N) {
        kernel_cc_decap<true><<<(N + 31) / 32, 32>>>(d_keys, cts, d_sk, N);
    };
#endif

    auto tc_front = [&](const unsigned char *cts, int N) {
        int blk = (N + 127) / 128;
        kernel_pack_c<<<blk, 128>>>(d_c32, cts, N);
        kernel_bmma_bits<1, F_KW><<<dim3((N + 31) / 32, F_OUTW), dim3(32, 4)>>>(d_spriv, F_OUTW, d_F, d_c32, N);
        kernel_tc_bm<<<blk, 128>>>(d_loc, d_spriv, N);
        kernel_pack_loc<<<blk, 128>>>(d_loc32, d_loc, N);
        kernel_bmma_bits<4, A_KW><<<dim3((N + 127) / 128, A_OUTW), dim3(32, 4)>>>(d_eval, A_OUTW, d_A, d_loc32, N);
    };
    auto tc_pke = [&](const unsigned char *cts, int N) {
        tc_front(cts, N);
        kernel_tc_tail_pke<<<(N + 127) / 128, 128>>>(d_keys, cts, d_sk, d_eval, d_cA, N);
    };
    auto tc_kem = [&](const unsigned char *cts, int N) {
        int blk = (N + 127) / 128;
        tc_front(cts, N);
        kernel_tc_error_extract<<<blk, 128>>>(d_err, d_err32, d_eval, d_cA, N);
        kernel_bmma_bits<4, G_KW><<<dim3((N + 127) / 128, F_OUTW), dim3(32, 4)>>>(d_scmp, F_OUTW, d_G, d_err32, N);
        kernel_tc_tail_kem<<<blk, 128>>>(d_keys, cts, d_sk, d_err, d_spriv, d_scmp, N);
    };

    /* ---------------- decapsulation sweep ---------------- */
    double dec[16][4];          /* CC-PKE, TC-PKE, CC-KEM, TC-KEM */
    int dec_ok[16];
    std::vector<unsigned char> kg((size_t)NMAX * 32);
    for (int bi = 0; bi < NB; bi++) {
        int N = BATCHES[bi], bad = 0;
        auto chk = [&](const char *who) {
            CHECK(cudaMemcpy(kg.data(), d_keys, (size_t)N * 32, cudaMemcpyDeviceToHost));
            int b = count_mismatch(kg.data(), keys_cpu.data(), N);
            if (b) fprintf(stderr, "  [decap N=%d %s] %d of %d keys mismatch the CPU\n", N, who, b, N);
            bad += b;
        };
        dec[bi][0] = time_ops(N, [&] { cc_pke(d_ct, N); }); chk("CC-PKE");
        dec[bi][1] = time_ops(N, [&] { tc_pke(d_ct, N); }); chk("TC-PKE");
        dec[bi][2] = time_ops(N, [&] { cc_kem(d_ct, N); }); chk("CC-KEM");
        dec[bi][3] = time_ops(N, [&] { tc_kem(d_ct, N); }); chk("TC-KEM");
        if (bi == 0) {                                 /* GEMM outputs vs host matrix-vector products */
            int okv = 1;
            std::vector<uint64_t> sp(F_OUTW), sc(F_OUTW), want(A_OUTW), x(A_OUTW), loc(LOC_W * GFBITS), err(NVEC);
            for (int d = 0; d < 8; d++) {
                std::vector<uint64_t> cw((F_COLS + 63) / 64, 0);
                memcpy(cw.data(), ct.data() + (size_t)d * SYND_BYTES, SYND_BYTES);
                CHECK(cudaMemcpy(sp.data(), d_spriv + (size_t)d * F_OUTW, F_OUTW * 8, cudaMemcpyDeviceToHost));
                matvec(Fc, F_COLS, F_OUTW, cw.data(), want.data());
                if (memcmp(sp.data(), want.data(), F_OUTW * 8)) okv = 0;
                CHECK(cudaMemcpy(loc.data(), d_loc + (size_t)d * LOC_W * GFBITS, loc.size() * 8, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(x.data(), d_eval + (size_t)d * A_OUTW, A_OUTW * 8, cudaMemcpyDeviceToHost));
                matvec(Ac, A_COLS, A_OUTW, loc.data(), want.data());
                if (memcmp(x.data(), want.data(), A_OUTW * 8)) okv = 0;
                CHECK(cudaMemcpy(err.data(), d_err + (size_t)d * NVEC, NVEC * 8, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(sc.data(), d_scmp + (size_t)d * F_OUTW, F_OUTW * 8, cudaMemcpyDeviceToHost));
                matvec(Gc, G_COLS, F_OUTW, err.data(), want.data());
                if (memcmp(sc.data(), want.data(), F_OUTW * 8) || memcmp(sc.data(), sp.data(), F_OUTW * 8)) okv = 0;
            }
            printf("[verify] TC GEMM outputs (F*c, A*locator, G*error == s_priv): %s\n", okv ? "OK" : "FAILED");
            if (!okv) all_ok = false;
        }
        dec_ok[bi] = (bad == 0);
        if (bad) all_ok = false;
        fprintf(stderr, "  decap N=%d done\n", N);
    }
    printf("\n[DECAP] PKE: no re-encryption check; KEM: reference Decap\n%8s | %12s | %12s | %12s | %12s | Verification\n",
           "Batch N", "CC-PKE", "TC-PKE", "CC-KEM", "TC-KEM");
    for (int bi = 0; bi < NB; bi++)
        printf("%8d | %12.0f | %12.0f | %12.0f | %12.0f | %s\n", BATCHES[bi], dec[bi][0], dec[bi][1], dec[bi][2],
               dec[bi][3], dec_ok[bi] ? "PASSED" : "FAILED");

    /* ---------------- implicit rejection: random ciphertexts ---------------- */
    {
        int N = N_INVALID, bad_cc, bad_tc, bad_cc_pke, bad_tc_pke;
        cc_kem(d_ct_inv, N); bench_check("invalid CC");
        CHECK(cudaMemcpy(kg.data(), d_keys, (size_t)N * 32, cudaMemcpyDeviceToHost));
        bad_cc = count_mismatch(kg.data(), keys_inv.data(), N);
        tc_kem(d_ct_inv, N); bench_check("invalid TC");
        CHECK(cudaMemcpy(kg.data(), d_keys, (size_t)N * 32, cudaMemcpyDeviceToHost));
        bad_tc = count_mismatch(kg.data(), keys_inv.data(), N);
        cc_pke(d_ct_inv, N); bench_check("invalid CC-PKE");
        CHECK(cudaMemcpy(kg.data(), d_keys, (size_t)N * 32, cudaMemcpyDeviceToHost));
        bad_cc_pke = count_mismatch(kg.data(), keys_inv.data(), N);
        tc_pke(d_ct_inv, N); bench_check("invalid TC-PKE");
        CHECK(cudaMemcpy(kg.data(), d_keys, (size_t)N * 32, cudaMemcpyDeviceToHost));
        bad_tc_pke = count_mismatch(kg.data(), keys_inv.data(), N);
        printf("[verify] %d random ciphertexts, keys mismatching CPU: CC-KEM %d, TC-KEM %d (PKE setting: CC %d, TC %d)\n",
               N, bad_cc, bad_tc, bad_cc_pke, bad_tc_pke);
        if (bad_cc || bad_tc) all_ok = false;
    }

    /* ---------------- per-kernel breakdown of the TC KEM pipeline ---------------- */
    if (BREAKDOWN_N <= NMAX) {
        int N = BREAKDOWN_N, blk = (N + 127) / 128;
        const int NK = 8;
        const char *names[NK] = {"pack_c", "bmma_F", "bm", "pack_loc", "bmma_A", "error_extract", "bmma_G", "final/hash"};
        cudaEvent_t e[NK + 1];
        for (int k = 0; k <= NK; k++) cudaEventCreate(&e[k]);
        float s[NK] = {0}, t;
        for (int r = 0; r < N_TRIALS; r++) {
            cudaEventRecord(e[0]);
            kernel_pack_c<<<blk, 128>>>(d_c32, d_ct, N);                                               cudaEventRecord(e[1]);
            kernel_bmma_bits<1, F_KW><<<dim3((N + 31) / 32, F_OUTW), dim3(32, 4)>>>(d_spriv, F_OUTW, d_F, d_c32, N); cudaEventRecord(e[2]);
            kernel_tc_bm<<<blk, 128>>>(d_loc, d_spriv, N);                                             cudaEventRecord(e[3]);
            kernel_pack_loc<<<blk, 128>>>(d_loc32, d_loc, N);                                          cudaEventRecord(e[4]);
            kernel_bmma_bits<4, A_KW><<<dim3((N + 127) / 128, A_OUTW), dim3(32, 4)>>>(d_eval, A_OUTW, d_A, d_loc32, N); cudaEventRecord(e[5]);
            kernel_tc_error_extract<<<blk, 128>>>(d_err, d_err32, d_eval, d_cA, N);                    cudaEventRecord(e[6]);
            kernel_bmma_bits<4, G_KW><<<dim3((N + 127) / 128, F_OUTW), dim3(32, 4)>>>(d_scmp, F_OUTW, d_G, d_err32, N); cudaEventRecord(e[7]);
            kernel_tc_tail_kem<<<blk, 128>>>(d_keys, d_ct, d_sk, d_err, d_spriv, d_scmp, N);           cudaEventRecord(e[8]);
            cudaEventSynchronize(e[8]);
            for (int k = 0; k < NK; k++) { cudaEventElapsedTime(&t, e[k], e[k + 1]); s[k] += t; }
        }
        bench_check("breakdown");
        float tot = 0;
        printf("\n[BREAKDOWN] TC KEM pipeline at N=%d (mean of %d)\n", N, N_TRIALS);
        for (int k = 0; k < NK; k++) { printf("  %-14s: %8.3f ms\n", names[k], s[k] / N_TRIALS); tot += s[k] / N_TRIALS; }
        printf("  %-14s: %8.3f ms (%.0f ops/s)\n", "TOTAL", tot, N / (tot / 1000.0));
        printf("  mma per decapsulation: F %d, A %d, G %d\n", F_ROWS / 8 * (F_KW / 4) / 8, A_ROWS / 8 * (A_KW / 4) / 8,
               G_ROWS / 8 * (G_KW / 4) / 8);
    }

    /* ---------------- per-kernel breakdown of the CC KEM pipeline ---------------- */
#if CC_WARP_OK
    if (BREAKDOWN_N <= NMAX) {
        int N = BREAKDOWN_N, wb = (N + CC_WARPS - 1) / CC_WARPS, tb32 = (N + 127) / 128;
        const int NK = 5;
        const char *names[NK] = {"cc_front", "cc_bm", "cc_afft", "cc_tafft", "cc_final/hash"};
        cudaEvent_t e[NK + 1];
        for (int k = 0; k <= NK; k++) cudaEventCreate(&e[k]);
        float s[NK] = {0}, t;
        for (int r = 0; r < N_TRIALS; r++) {
            cudaEventRecord(e[0]);
            kernel_cc_front<<<wb, 32 * CC_WARPS>>>(d_spriv, d_ct, d_ftab, N);                        cudaEventRecord(e[1]);
            kernel_tc_bm<<<tb32, 128>>>(d_loc, d_spriv, N);                                          cudaEventRecord(e[2]);
            kernel_cc_afft<<<wb, 32 * CC_WARPS, CC_SMEM>>>(d_err, d_loc, cc_tb, N);                  cudaEventRecord(e[3]);
            kernel_cc_tafft<<<wb, 32 * CC_WARPS, CC_SMEM>>>(d_scmp, d_err, d_inv, cc_tb, N);         cudaEventRecord(e[4]);
            kernel_tc_tail_kem<<<tb32, 128>>>(d_keys, d_ct, d_sk, d_err, d_spriv, d_scmp, N);        cudaEventRecord(e[5]);
            cudaEventSynchronize(e[5]);
            for (int k = 0; k < NK; k++) { cudaEventElapsedTime(&t, e[k], e[k + 1]); s[k] += t; }
        }
        bench_check("cc breakdown");
        float tot = 0;
        printf("\n[BREAKDOWN] CC (warp) KEM pipeline at N=%d (mean of %d)\n", N, N_TRIALS);
        for (int k = 0; k < NK; k++) { printf("  %-14s: %8.3f ms\n", names[k], s[k] / N_TRIALS); tot += s[k] / N_TRIALS; }
        printf("  %-14s: %8.3f ms (%.0f ops/s)\n", "cc_TOTAL", tot, N / (tot / 1000.0));
    }
#endif

    printf("\n[result] %s\n", all_ok ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    return all_ok ? 0 : 1;
}
