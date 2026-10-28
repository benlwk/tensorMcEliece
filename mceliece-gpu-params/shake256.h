#ifndef SHAKE256_H
#define SHAKE256_H

#include <stdint.h>

#define SHAKE_RATEBYTES 136
#define SHAKE_SQUEEZEBYTES 136
#define SHAKE_PADDING 31

typedef unsigned char shake_u8;
typedef uint64_t shake_u64;

__device__ static inline shake_u64 shake_ROL(shake_u64 a, shake_u8 n) {
  return n ? ((a << n) | (a >> (64 - n))) : a;
}

__device__ static inline shake_u64 shake_L64(const shake_u8 *x) {
  shake_u64 r = 0;
  for (int i = 0; i < 8; i++) {
    r |= ((shake_u64)x[i]) << (8 * i);
  }
  return r;
}

__device__ static inline void shake_F(shake_u64 *s) {
  const shake_u8 RHO[25] = {
      0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39,
      41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
  const shake_u8 PI[25] = {
      0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22,
      23, 8, 18, 3, 13, 14, 24, 9, 19, 4};
  const shake_u64 RC[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
      0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
      0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
      0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
      0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
      0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
      0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
      0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
  shake_u64 C[5], D[5], B[25];
  #pragma unroll 1
  for (int round = 0; round < 24; round++) {
    #pragma unroll
    for (int x = 0; x < 5; x++)
      C[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
    #pragma unroll
    for (int x = 0; x < 5; x++) {
      int xm1 = (x + 4) % 5, xp1 = (x + 1) % 5;
      D[x] = C[xm1] ^ shake_ROL(C[xp1], 1);
    }
    #pragma unroll
    for (int x = 0; x < 5; x++)
      #pragma unroll
      for (int y = 0; y < 5; y++) s[x + 5 * y] ^= D[x];

    #pragma unroll
    for (int i = 0; i < 25; i++)
      B[PI[i]] = shake_ROL(s[i], RHO[i]);

    #pragma unroll
    for (int y = 0; y < 5; y++)
      #pragma unroll
      for (int x = 0; x < 5; x++)
        s[x + 5 * y] = B[x + 5 * y] ^ ((~B[((x + 1) % 5) + 5 * y]) & B[((x + 2) % 5) + 5 * y]);

    s[0] ^= RC[round];
  }
}

/* Kept out of line: with CUDA 13.1, inlining the sponge into the large decapsulation
 * kernels produced wrong digests at the default cicc -O3 (correct at -O1/-O2 and when
 * not inlined). All harness outputs are checked against libmceliece. */
__device__ __noinline__ static void crypto_xof_shake256(shake_u8 *h, long long hlen, const shake_u8 *m, long long mlen) {
  shake_u64 s[25];
  shake_u8 t[200];
  long long i;
  for (i = 0; i < 25; i++) s[i] = 0;
  while (mlen >= SHAKE_RATEBYTES) {
    for (i = 0; i < SHAKE_RATEBYTES / 8; i++) {
      s[i] ^= shake_L64(m + 8 * i);
    }
    shake_F(s);
    mlen -= SHAKE_RATEBYTES;
    m += SHAKE_RATEBYTES;
  }
  for (i = 0; i < SHAKE_RATEBYTES; i++) t[i] = 0;
  for (i = 0; i < mlen; i++) t[i] = m[i];
  t[mlen] = SHAKE_PADDING;
  t[SHAKE_RATEBYTES - 1] |= 128;
  for (i = 0; i < SHAKE_RATEBYTES / 8; i++) {
    s[i] ^= shake_L64(t + 8 * i);
  }
  while (hlen > 0) {
    shake_F(s);
    for (i = 0; i < SHAKE_SQUEEZEBYTES; i++) {
      if (i < hlen) {
        h[i] = s[i / 8] >> (8 * (i % 8));
      }
    }
    h += SHAKE_SQUEEZEBYTES;
    hlen -= SHAKE_SQUEEZEBYTES;
  }
}

__device__ static inline void shake(shake_u8 *out, long long outlen, const shake_u8 *in, long long inlen) {
  crypto_xof_shake256(out, outlen, in, inlen);
}

__device__ static inline void crypto_hash_32b(shake_u8 *out, const shake_u8 *in, long long inlen) {
  shake(out, 32, in, inlen);
}

#endif
