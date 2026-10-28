#ifndef SHAKE256_H
#define SHAKE256_H

#include <stdint.h>

#define SHAKE_RATEBYTES 136
#define SHAKE_SQUEEZEBYTES 136
#define SHAKE_PADDING 31

typedef unsigned char shake_u8;
typedef uint64_t shake_u64;

__device__ static inline shake_u64 shake_ROL(shake_u64 a, shake_u8 n) {
  return (a << n) | (a >> (64 - n));
}

__device__ static inline shake_u64 shake_L64(const shake_u8 *x) {
  shake_u64 r = 0;
  for (int i = 0; i < 8; i++) {
    r |= ((shake_u64)x[i]) << (8 * i);
  }
  return r;
}

__device__ static inline void shake_F(shake_u64 *s) {
  shake_u8 x, y, j, R = 1, r, n;
  shake_u64 t, B[5], Y;
  for (n = 0; n < 24; n++) {
    for (x = 0; x < 5; x++) {
      B[x] = 0;
      for (y = 0; y < 5; y++) {
        B[x] ^= s[x + 5 * y];
      }
    }
    for (x = 0; x < 5; x++) {
      t = B[(x + 4) % 5] ^ shake_ROL(B[(x + 1) % 5], 1);
      for (y = 0; y < 5; y++) {
        s[x + 5 * y] ^= t;
      }
    }
    t = s[1];
    y = r = 0;
    x = 1;
    for (j = 0; j < 24; j++) {
      r += j + 1;
      Y = 2 * x + 3 * y;
      x = y;
      y = Y % 5;
      Y = s[x + 5 * y];
      s[x + 5 * y] = shake_ROL(t, r % 64);
      t = Y;
    }
    for (y = 0; y < 5; y++) {
      for (x = 0; x < 5; x++) {
        B[x] = s[x + 5 * y];
      }
      for (x = 0; x < 5; x++) {
        s[x + 5 * y] = B[x] ^ (~B[(x + 1) % 5] & B[(x + 2) % 5]);
      }
    }
    for (y = 0; y < 7; y++) {
      R = (R << 1) ^ (113 * (R >> 7));
      if (R & 2) {
        *s ^= 1ULL << ((1 << y) - 1);
      }
    }
  }
}

__device__ static inline void crypto_xof_shake256(shake_u8 *h, long long hlen, const shake_u8 *m, long long mlen) {
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
