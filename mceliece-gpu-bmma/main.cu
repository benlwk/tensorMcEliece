#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <mceliece.h>

#include "params.h"
#include "bmma_enc.h"
#include "int8_tensor_enc.h"
#include "int8_tensor_bs2_enc.h"
#include "bench_common.h"


static float elapsed_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

typedef int (*decap_fn_t)(unsigned char *, const unsigned char *, const unsigned char *);

static int verify_against_cpu(const unsigned char *h_cs, const unsigned char *h_keys,
                              const unsigned char *h_sk, int N, decap_fn_t dec)
{
    int verified = 1;
    int check_count = (N < 64) ? N : 64;
    for (int i = 0; i < check_count; i++) {
        unsigned char key_dec_cpu[KEY_SIZE];
        if (dec(key_dec_cpu, h_cs + (size_t)i * CT_SIZE, h_sk) != 0) { verified = 0; break; }
        if (memcmp(key_dec_cpu, h_keys + (size_t)i * KEY_SIZE, KEY_SIZE) != 0) { verified = 0; break; }
    }
    return verified;
}

int main(void)
{
    printf("\n");
    printf("=========================================================================================\n");
    printf("     McEliece 348864 -- Tensor Core syndrome GEMM benchmark                               \n");
    printf("     Comparing 1-bit BMMA vs. INT8 Tensor Core GEMM family (H100/5070 Path)            \n");
    printf("=========================================================================================\n\n");

    bench_print_device();
    bench_set_stack_limit({(const void *)kernel_gen_e_and_pack, (const void *)kernel_gen_e_and_pack_int8_wmma,
                           (const void *)kernel_gen_e_and_pack_int8_bs2, (const void *)kernel_gen_e_and_pack_int8_bs2_w4,
                           (const void *)kernel_finalize_encap});
    int sm_count = 0;
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        sm_count = prop.multiProcessorCount;
    }

    unsigned char *h_pk = (unsigned char *)malloc(PK_SIZE);
    unsigned char *h_sk = (unsigned char *)malloc(SK_SIZE);
    if (!h_pk || !h_sk) { fprintf(stderr, "malloc failed\n"); return 1; }

    printf("[Host] Generating CPU keypair...\n");
    mceliece_kem_348864_keypair(h_pk, h_sk);
    printf("[Host] Keypair generated successfully.\n\n");

    uint32_t *h_pk_bmma = (uint32_t *)malloc(PK_NROWS * (K_BITS_PADDED / 32) * sizeof(uint32_t));
    pack_pk_bmma(h_pk_bmma, h_pk);

    int8_t *h_pk_int8 = (int8_t *)malloc((size_t)INT8_M_PADDED * INT8_K_PADDED * sizeof(int8_t));
    pack_pk_int8_wmma(h_pk_int8, h_pk);

    uint32_t *d_T_bmma;
    int8_t   *d_T_int8;
    cudaMalloc((void**)&d_T_bmma, PK_NROWS * (K_BITS_PADDED / 32) * sizeof(uint32_t));
    cudaMalloc((void**)&d_T_int8, (size_t)INT8_M_PADDED * INT8_K_PADDED * sizeof(int8_t));

    cudaMemcpy(d_T_bmma, h_pk_bmma, PK_NROWS * (K_BITS_PADDED / 32) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_int8, h_pk_int8, (size_t)INT8_M_PADDED * INT8_K_PADDED * sizeof(int8_t), cudaMemcpyHostToDevice);

    int batch_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    // wklee, engine order: BMMA, INT8 0/1, BS2, BS2-M4, BS2-W4, BS2-BLK */
    enum { NENG = 6 };
    double pke_ops[16][NENG], kem_ops[16][NENG];
    int    verified_row[16];

    for (int b = 0; b < num_batches; b++) {
        int N = batch_sizes[b];

        unsigned char *d_cs, *d_keys, *d_seeds, *d_e_all, *d_S_bytes;
        uint32_t *d_E_bmma, *d_S_gemm;
        int8_t   *d_E_int8, *d_E_bs2, *d_E_w4;

        cudaMalloc((void**)&d_cs,      (size_t)N * CT_SIZE);
        cudaMalloc((void**)&d_keys,    (size_t)N * KEY_SIZE);
        cudaMalloc((void**)&d_seeds,   (size_t)N * 32);
        cudaMalloc((void**)&d_e_all,   (size_t)N * (SYS_N / 8));
        cudaMalloc((void**)&d_E_bmma,  (size_t)(K_BITS_PADDED / 32) * N * sizeof(uint32_t));
        cudaMalloc((void**)&d_E_int8,  (size_t)INT8_K_PADDED * N * sizeof(int8_t));
        cudaMalloc((void**)&d_E_bs2,   (size_t)BS2_K_PADDED * BS2_PAIRS_PADDED(N) * sizeof(int8_t));
        cudaMalloc((void**)&d_E_w4,    (size_t)BS2_K_PADDED * BS2_GROUPS_PADDED(N) * sizeof(int8_t));
        cudaMalloc((void**)&d_S_gemm,  (size_t)PK_NROWS * N);
        cudaMalloc((void**)&d_S_bytes, (size_t)PK_NROWS * N);

        unsigned char *h_seeds = (unsigned char *)malloc(N * 32);
        for (int i = 0; i < N; i++) {
            memset(h_seeds + i * 32, 0, 32);
            memcpy(h_seeds + i * 32, &i, sizeof(i));
        }
        cudaMemcpy(d_seeds, h_seeds, N * 32, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        unsigned char *h_cs   = (unsigned char *)malloc((size_t)N * CT_SIZE);
        unsigned char *h_keys = (unsigned char *)malloc((size_t)N * KEY_SIZE);
        int verified = 1;

        /* Times one engine in both settings and verifies its keys.
         * PKE: e is generated and packed once before timing (Wu et al., Alg. 1).
         * KEM: every timed pass also runs fixed-weight sampling + operand packing. */
        auto bench = [&](int eng, auto gen, auto body) {
            gen(); body();
            bench_check("warm-up");
            cudaEventRecord(start);
            for (int t = 0; t < N_TRIALS; t++) body();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            pke_ops[b][eng] = (double)N / (elapsed_ms(start, stop) / N_TRIALS / 1000.0);

            cudaEventRecord(start);
            for (int t = 0; t < N_TRIALS; t++) { gen(); body(); }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            kem_ops[b][eng] = (double)N / (elapsed_ms(start, stop) / N_TRIALS / 1000.0);
            bench_check("timed passes");

            cudaMemcpy(h_cs,   d_cs,   (size_t)N * CT_SIZE,  cudaMemcpyDeviceToHost);
            cudaMemcpy(h_keys, d_keys, (size_t)N * KEY_SIZE, cudaMemcpyDeviceToHost);
            if (!verify_against_cpu(h_cs, h_keys, h_sk, N, mceliece_kem_348864_dec)) verified = 0;
        };

        dim3 gemm_grid((N + BN_COLS - 1) / BN_COLS, (PK_NROWS + BM_ROWS - 1) / BM_ROWS);
        dim3 gemm_block(32, 4);
        bench(0,
            [&] { kernel_gen_e_and_pack<<<(N + 31) / 32, 32>>>(d_e_all, d_E_bmma, d_seeds, N); },
            [&] {
                kernel_bmma_syndrome_gemm<<<gemm_grid, gemm_block>>>(d_S_gemm, d_T_bmma, d_E_bmma, N);
                kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, (const unsigned char *)d_S_gemm, d_e_all, N);
            });

        dim3 wmma_grid((N + WMMA_N - 1) / WMMA_N, (PK_NROWS + WMMA_M - 1) / WMMA_M);
        dim3 wmma_block(32, 1);
        bench(1,
            [&] { kernel_gen_e_and_pack_int8_wmma<<<(N + 31) / 32, 32>>>(d_e_all, d_E_int8, d_seeds, N); },
            [&] {
                kernel_int8_wmma_syndrome_gemm<<<wmma_grid, wmma_block>>>(d_S_bytes, d_T_int8, d_E_int8, N);
                kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, d_S_bytes, d_e_all, N);
            });

        int bs2_pairs = BS2_PAIRS_PADDED(N);
        auto gen_bs2 = [&] {
            kernel_gen_e_and_pack_int8_bs2<<<(N / 2 + 255) / 256, 256>>>(d_e_all, d_E_bs2, d_seeds, N, bs2_pairs);
        };
        dim3 bs2_grid(bs2_pairs / 16, BS2_M_PADDED / 16);
        bench(2, gen_bs2, [&] {
            kernel_int8_wmma_bs2_syndrome_gemm<<<bs2_grid, 32>>>(d_S_bytes, d_T_int8, d_E_bs2, N, bs2_pairs);
            kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, d_S_bytes, d_e_all, N);
        });

        int m4_y = (BS2_M_PADDED + BS2_M_CHUNK * 16 - 1) / (BS2_M_CHUNK * 16);
        dim3 bs2m4_grid(bs2_pairs / 16, m4_y);
        bench(3, gen_bs2, [&] {
            kernel_int8_wmma_bs2_m4_syndrome_gemm<<<bs2m4_grid, 32>>>(d_S_bytes, d_T_int8, d_E_bs2, N, bs2_pairs);
            kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, d_S_bytes, d_e_all, N);
        });

        int w4_groups = BS2_GROUPS_PADDED(N);
        dim3 w4_grid(w4_groups / 16, BS2_M_PADDED / (BS2_W4_CHUNK * 16));
        bench(4,
            [&] { kernel_gen_e_and_pack_int8_bs2_w4<<<(N / 4 + 255) / 256, 256>>>(d_e_all, d_E_w4, d_seeds, N, w4_groups); },
            [&] {
                kernel_int8_wmma_bs2_w4_syndrome_gemm<<<w4_grid, 32>>>(d_S_bytes, d_T_int8, d_E_w4, N, w4_groups);
                kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, d_S_bytes, d_e_all, N);
            });

        pke_ops[b][5] = kem_ops[b][5] = 0.0;
        if (bs2_pairs % 64 == 0) {
            dim3 blk_grid(bs2_pairs / 64, BS2_M_PADDED / 64);
            bench(5, gen_bs2, [&] {
                kernel_int8_wmma_bs2_blk3_syndrome_gemm<<<blk_grid, 128>>>(d_S_bytes, d_T_int8, d_E_bs2, N, bs2_pairs);
                kernel_finalize_encap<<<(N + 127) / 128, 128>>>(d_cs, d_keys, d_S_bytes, d_e_all, N);
            });
        }
        verified_row[b] = verified;

        free(h_seeds); free(h_cs); free(h_keys);
        cudaFree(d_cs); cudaFree(d_keys); cudaFree(d_seeds); cudaFree(d_e_all);
        cudaFree(d_E_bmma); cudaFree(d_E_int8); cudaFree(d_E_bs2); cudaFree(d_E_w4);
        cudaFree(d_S_gemm); cudaFree(d_S_bytes);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    for (int conv = 0; conv < 2; conv++) {
        double (*ops)[NENG] = conv == 0 ? pke_ops : kem_ops;
        printf("\n[%s] %s\n", conv == 0 ? "PKE" : "KEM",
               conv == 0 ? "PKE setting: e generated and packed before timing"
                         : "KEM setting: fixed-weight sampling + packing timed");
        printf("%-8s | %-14s | %-16s | %-14s | %-14s | %-14s | %-14s | %-12s\n",
               "Batch N", "1-bit BMMA", "INT8 WMMA", "INT8 BS2", "BS2-M4", "BS2-W4", "BS2-BLK", "Verification");
        printf("-------------------------------------------------------------------------------------------------------------\n");
        for (int b = 0; b < num_batches; b++)
            printf("%-8d | %-14.0f | %-16.0f | %-14.0f | %-14.0f | %-14.0f | %-14.0f | %s\n",
                   batch_sizes[b], ops[b][0], ops[b][1], ops[b][2], ops[b][3], ops[b][4], ops[b][5],
                   verified_row[b] ? "PASSED" : "FAILED");
    }
    double last_bmma_ops = pke_ops[num_batches - 1][0], last_bs2m4_ops = pke_ops[num_batches - 1][3];
    double last_w4_ops = pke_ops[num_batches - 1][4], last_blk_ops = pke_ops[num_batches - 1][5];

    printf("-----------------------------------------------------------------------------------------\n");

    printf("Per-SM @ N=%d (%d SMs):  BMMA %.0f ops/s/SM | BS2-M4 %.0f ops/s/SM | BS2-W4 %.0f ops/s/SM | BS2-BLK %.0f ops/s/SM\n",
           batch_sizes[num_batches - 1], sm_count, last_bmma_ops / sm_count, last_bs2m4_ops / sm_count, last_w4_ops / sm_count, last_blk_ops / sm_count);
    printf("(reference paper: 223,658 ops/s/SM on RTX 4090)\n");

    free(h_pk); free(h_sk); free(h_pk_bmma); free(h_pk_int8);
    cudaFree(d_T_bmma); cudaFree(d_T_int8);

    return 0;
}
