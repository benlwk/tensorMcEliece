#ifndef ENCAP_DEVICE_H
#define ENCAP_DEVICE_H

#include <string.h>
#include <stdint.h>

#include "params.h"
#include "gf.h"
#include "util.h"
#include "crypto_sort.h"
#include "shake256.h"

typedef int16_t crypto_int16;
typedef uint16_t crypto_uint16;
typedef uint32_t crypto_uint32;
typedef uint64_t crypto_uint64;

__device__ static inline uint16_t crypto_uint16_smaller_mask(uint16_t t, uint16_t u) {
  return (t < u) ? -1 : 0;
}

__device__ static inline uint16_t uint16_is_smaller_declassify(uint16_t t, uint16_t u) {
  return crypto_uint16_smaller_mask(t, u);
}

__device__ static inline uint16_t crypto_int16_zero_mask(int16_t x) {
  return (x == 0) ? -1 : 0;
}

__device__ static inline uint16_t crypto_int16_negative_mask(int16_t x) {
  return (x < 0) ? -1 : 0;
}

__device__ static inline void crypto_uint16_store(unsigned char *x, uint16_t u) {
  store_gf(x, u);
}

__device__ static inline void crypto_xof_bitwrite16(unsigned char *h, long long hlen, const unsigned char *m, long long mlen) {
  for (long long i = 0; i < hlen; i++) h[i] = 0;
  for (long long i = 0; i < mlen; i += 2) {
    uint16_t pos = load_gf(m + i);
    if ((pos >> 3) < hlen) {
      h[pos >> 3] |= (1 << (pos & 7));
    }
  }
}

__device__ static inline void gen_e(unsigned char *e, const unsigned char *rng_seed) {
  int i, j, count;
  union {
    uint16_t nums[SYS_T * 2];
    unsigned char bytes[SYS_T * 2 * sizeof(uint16_t)];
  } buf;

  uint16_t ind[SYS_T];
  crypto_int16 negifcollision;
  unsigned char indbytes[SYS_T * 2];
  unsigned char seed[33] = {64};

  memcpy(seed + 1, rng_seed, 32);

  while (1) {
    shake(buf.bytes, sizeof(buf), seed, 33);

    memcpy(seed + 1, buf.bytes, 32);

    for (i = 0; i < SYS_T * 2; i++) {
      buf.nums[i] = load_gf(buf.bytes + i * 2);
    }

    count = 0;
    for (i = 0; i < SYS_T * 2 && count < SYS_T; i++) {
      if (uint16_is_smaller_declassify(buf.nums[i], SYS_N)) {
        ind[count++] = buf.nums[i];
      }
    }

    if (count < SYS_T) {
      continue;
    }

    crypto_sort_int16((int16_t *)ind, SYS_T);

    negifcollision = 0;
    for (i = 1; i < SYS_T; i++) {
      negifcollision |= (ind[i - 1] ^ ind[i]) - 1;
    }

    negifcollision = crypto_int16_negative_mask(negifcollision);
    if (!negifcollision) {
      break;
    }
  }

  for (j = 0; j < SYS_T; j++) {
    crypto_uint16_store(indbytes + 2 * j, ind[j]);
  }

  crypto_xof_bitwrite16(e, SYS_N / 8, indbytes, 2 * SYS_T);
}

#endif