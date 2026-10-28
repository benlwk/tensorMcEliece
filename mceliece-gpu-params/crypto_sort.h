#ifndef CRYPTO_SORT_H
#define CRYPTO_SORT_H

#include <stdint.h>

__device__ static inline void crypto_int64_minmax(int64_t *p, int64_t *q) {
  int64_t x = *p;
  int64_t y = *q;
  int64_t r = y ^ x;
  int64_t z = y - x;
  z ^= r & (z ^ y);
  z = (z < 0) ? -1LL : 0LL;
  z &= r;
  x ^= z;
  y ^= z;
  *p = x;
  *q = y;
}

__device__ static inline void crypto_int32_minmax(int32_t *p, int32_t *q) {
  int32_t x = *p;
  int32_t y = *q;
  int32_t r = y ^ x;
  int32_t z = y - x;
  z ^= r & (z ^ y);
  z = (z < 0) ? -1 : 0;
  z &= r;
  x ^= z;
  y ^= z;
  *p = x;
  *q = y;
}

__device__ static inline void crypto_int16_minmax(int16_t *p, int16_t *q) {
  int16_t x = *p;
  int16_t y = *q;
  int16_t r = y ^ x;
  int16_t z = y - x;
  z ^= r & (z ^ y);
  z = (z < 0) ? -1 : 0;
  z &= r;
  x ^= z;
  y ^= z;
  *p = x;
  *q = y;
}

__device__ static inline void crypto_sort_int64(int64_t *x, long long n) {
  long long top, p, q, r, i, j;

  if (n < 2) return;
  top = 1;
  while (top < n - top) top += top;

  for (p = top; p >= 1; p >>= 1) {
    i = 0;
    while (i + 2 * p <= n) {
      for (j = i; j < i + p; ++j)
        crypto_int64_minmax(&x[j], &x[j+p]);
      i += 2 * p;
    }
    for (j = i; j < n - p; ++j)
      crypto_int64_minmax(&x[j], &x[j+p]);

    i = 0;
    j = 0;
    for (q = top; q > p; q >>= 1) {
      if (j != i) {
        for (;;) {
          if (j == n - q) goto done_64;
          int64_t a = x[j + p];
          for (r = q; r > p; r >>= 1)
            crypto_int64_minmax(&a, &x[j + r]);
          x[j + p] = a;
          ++j;
          if (j == i + p) {
            i += 2 * p;
            break;
          }
        }
      }
      while (i + p <= n - q) {
        for (j = i; j < i + p; ++j) {
          int64_t a = x[j + p];
          for (r = q; r > p; r >>= 1)
            crypto_int64_minmax(&a, &x[j+r]);
          x[j + p] = a;
        }
        i += 2 * p;
      }
      j = i;
      while (j < n - q) {
        int64_t a = x[j + p];
        for (r = q; r > p; r >>= 1)
          crypto_int64_minmax(&a, &x[j+r]);
        x[j + p] = a;
        ++j;
      }
      done_64: ;
    }
  }
}

__device__ static inline void crypto_sort_int32(int32_t *x, long long n) {
  long long top, p, q, r, i, j;

  if (n < 2) return;
  top = 1;
  while (top < n - top) top += top;

  for (p = top; p >= 1; p >>= 1) {
    i = 0;
    while (i + 2 * p <= n) {
      for (j = i; j < i + p; ++j)
        crypto_int32_minmax(&x[j], &x[j+p]);
      i += 2 * p;
    }
    for (j = i; j < n - p; ++j)
      crypto_int32_minmax(&x[j], &x[j+p]);

    i = 0;
    j = 0;
    for (q = top; q > p; q >>= 1) {
      if (j != i) {
        for (;;) {
          if (j == n - q) goto done_32;
          int32_t a = x[j + p];
          for (r = q; r > p; r >>= 1)
            crypto_int32_minmax(&a, &x[j + r]);
          x[j + p] = a;
          ++j;
          if (j == i + p) {
            i += 2 * p;
            break;
          }
        }
      }
      while (i + p <= n - q) {
        for (j = i; j < i + p; ++j) {
          int32_t a = x[j + p];
          for (r = q; r > p; r >>= 1)
            crypto_int32_minmax(&a, &x[j+r]);
          x[j + p] = a;
        }
        i += 2 * p;
      }
      j = i;
      while (j < n - q) {
        int32_t a = x[j + p];
        for (r = q; r > p; r >>= 1)
          crypto_int32_minmax(&a, &x[j+r]);
        x[j + p] = a;
        ++j;
      }
      done_32: ;
    }
  }
}

__device__ static inline void crypto_sort_int16(int16_t *x, long long n) {
  long long top, p, q, r, i, j;

  if (n < 2) return;
  top = 1;
  while (top < n - top) top += top;

  for (p = top; p >= 1; p >>= 1) {
    i = 0;
    while (i + 2 * p <= n) {
      for (j = i; j < i + p; ++j)
        crypto_int16_minmax(&x[j], &x[j+p]);
      i += 2 * p;
    }
    for (j = i; j < n - p; ++j)
      crypto_int16_minmax(&x[j], &x[j+p]);

    i = 0;
    j = 0;
    for (q = top; q > p; q >>= 1) {
      if (j != i) {
        for (;;) {
          if (j == n - q) goto done_16;
          int16_t a = x[j + p];
          for (r = q; r > p; r >>= 1)
            crypto_int16_minmax(&a, &x[j + r]);
          x[j + p] = a;
          ++j;
          if (j == i + p) {
            i += 2 * p;
            break;
          }
        }
      }
      while (i + p <= n - q) {
        for (j = i; j < i + p; ++j) {
          int16_t a = x[j + p];
          for (r = q; r > p; r >>= 1)
            crypto_int16_minmax(&a, &x[j+r]);
          x[j + p] = a;
        }
        i += 2 * p;
      }
      j = i;
      while (j < n - q) {
        int16_t a = x[j + p];
        for (r = q; r > p; r >>= 1)
          crypto_int16_minmax(&a, &x[j+r]);
        x[j + p] = a;
        ++j;
      }
      done_16: ;
    }
  }
}

#endif
