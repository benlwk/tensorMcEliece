#ifndef GF_H
#define GF_H

#include "gf_params.h"
#include <stdint.h>

typedef uint16_t gf;

__device__ static inline int32_t crypto_int32_zero_mask(int32_t x) {
  return (x == 0) ? -1 : 0;
}

__device__ static inline int32_t crypto_int32_bottombit_mask(int32_t x) {
  return -(int32_t)(x & 1);
}

__device__ static inline int64_t crypto_int64_bottombit_mask(int64_t x) {
  return -(int64_t)(x & 1);
}

__device__ static inline gf gf_iszero(gf a) {
  return crypto_int32_zero_mask(a) & GFMASK;
}

__device__ static inline gf gf_mul(gf in0, gf in1) {
  int i;
  uint32_t tmp;
  uint32_t t0 = in0;
  uint32_t t1 = in1;

  tmp = t0 & crypto_int32_bottombit_mask(t1);

  for (i = 1; i < GFBITS; i++) {
    t0 <<= 1;
    t1 >>= 1;
    tmp ^= t0 & crypto_int32_bottombit_mask(t1);
  }

  uint32_t t = tmp & 0x7FC000;
  tmp ^= t >> 9;
  tmp ^= t >> 12;

  t = tmp & 0x3000;
  tmp ^= t >> 9;
  tmp ^= t >> 12;

  return tmp & GFMASK;
}

__device__ static inline uint64_t gf_mul2(gf a, gf b0, gf b1) {
  int i;
  uint64_t tmp = 0;
  uint64_t t0 = a;
  uint64_t t1 = b1;
  t1 = (t1 << 32) | b0;

  for (i = 0; i < GFBITS; i++) {
    tmp ^= t1 & crypto_int64_bottombit_mask(t0);
    t1 <<= 1;
    t0 >>= 1;
  }

  uint64_t t = tmp & 0x007FC000007FC000ULL;
  tmp ^= (t >> 9) ^ (t >> 12);

  t = tmp & 0x0000300000003000ULL;
  tmp ^= (t >> 9) ^ (t >> 12);

  return tmp & 0x00000FFF00000FFFULL;
}

__device__ static inline gf gf_sq(gf in) {
  const uint32_t B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
  uint32_t x = in;
  uint32_t t;

  x = (x | (x << 8)) & B[3];
  x = (x | (x << 4)) & B[2];
  x = (x | (x << 2)) & B[1];
  x = (x | (x << 1)) & B[0];

  t = x & 0x7FC000;
  x ^= t >> 9;
  x ^= t >> 12;

  t = x & 0x3000;
  x ^= t >> 9;
  x ^= t >> 12;

  return x & GFMASK;
}

__device__ static inline gf gf_inv(gf in) {
  gf tmp_11;
  gf tmp_1111;
  gf out = in;

  out = gf_sq(out);
  tmp_11 = gf_mul(out, in);

  out = gf_sq(tmp_11);
  out = gf_sq(out);
  tmp_1111 = gf_mul(out, tmp_11);

  out = gf_sq(tmp_1111);
  out = gf_sq(out);
  out = gf_sq(out);
  out = gf_sq(out);
  out = gf_mul(out, tmp_1111);

  out = gf_sq(out);
  out = gf_sq(out);
  out = gf_mul(out, tmp_11);

  out = gf_sq(out);
  out = gf_mul(out, in);

  return gf_sq(out);
}

__device__ static inline void GF_mul(gf *out, const gf *in0, const gf *in1) {
  int i, j;

  gf prod[128];

  for (i = 0; i < SYS_T * 2 - 1; i++) {
    prod[i] = 0;
  }

  for (i = 0; i < SYS_T; i++) {
    for (j = 0; j < SYS_T; j++) {
      prod[i + j] ^= gf_mul(in0[i], in1[j]);
    }
  }

  for (i = (SYS_T - 1) * 2; i >= SYS_T; i--) {
    prod[i - SYS_T + 3] ^= prod[i];
    prod[i - SYS_T + 1] ^= prod[i];
    prod[i - SYS_T + 0] ^= gf_mul(prod[i], (gf)2);
  }

  for (i = 0; i < SYS_T; i++) {
    out[i] = prod[i];
  }
}

#endif
