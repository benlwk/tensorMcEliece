/* Device implementations of the libmceliece "cryptoint" helpers used by the
 * vec reference code. Semantics follow libmceliece-20250507 (portable C path). */
#ifndef REF_PRELUDE_CUH
#define REF_PRELUDE_CUH

#include <stdint.h>
#include <string.h>
#include "shake256.h"
#include "crypto_sort.h"

typedef int8_t   crypto_int8;
typedef int16_t  crypto_int16;
typedef int32_t  crypto_int32;
typedef int64_t  crypto_int64;
typedef uint8_t  crypto_uint8;
typedef uint16_t crypto_uint16;
typedef uint32_t crypto_uint32;
typedef uint64_t crypto_uint64;

#define REF_INT_HELPERS(N, T)                                                                         \
  __device__ static inline T crypto_int##N##_negative_mask(T x) { return x < 0 ? (T)-1 : (T)0; }     \
  __device__ static inline T crypto_int##N##_zero_mask(T x) { return x == 0 ? (T)-1 : (T)0; }        \
  __device__ static inline T crypto_int##N##_nonzero_mask(T x) { return x != 0 ? (T)-1 : (T)0; }     \
  __device__ static inline T crypto_int##N##_zero_01(T x) { return x == 0 ? (T)1 : (T)0; }           \
  __device__ static inline T crypto_int##N##_bottombit_01(T x) { return (T)(x & 1); }                \
  __device__ static inline T crypto_int##N##_bottombit_mask(T x) { return (T)-(T)(x & 1); }          \
  __device__ static inline T crypto_int##N##_bitmod_01(T x, T s) {                                   \
    return (T)(((uint##N##_t)x >> ((uint##N##_t)s & (N - 1))) & 1); }                                \
  __device__ static inline T crypto_int##N##_bitmod_mask(T x, T s) {                                 \
    return (T)-(T)crypto_int##N##_bitmod_01(x, s); }                                                 \
  __device__ static inline T crypto_int##N##_shlmod(T x, T s) {                                      \
    return (T)((uint##N##_t)x << ((uint##N##_t)s & (N - 1))); }                                      \
  __device__ static inline T crypto_int##N##_min(T a, T b) { return a < b ? a : b; }

#define REF_UINT_HELPERS(N, T)                                                                        \
  __device__ static inline T crypto_uint##N##_zero_mask(T x) { return x == 0 ? (T)-1 : (T)0; }       \
  __device__ static inline T crypto_uint##N##_nonzero_mask(T x) { return x != 0 ? (T)-1 : (T)0; }    \
  __device__ static inline T crypto_uint##N##_equal_mask(T a, T b) { return a == b ? (T)-1 : (T)0; } \
  __device__ static inline T crypto_uint##N##_smaller_mask(T a, T b) { return a < b ? (T)-1 : (T)0; }\
  __device__ static inline T crypto_uint##N##_leq_mask(T a, T b) { return a <= b ? (T)-1 : (T)0; }   \
  __device__ static inline T crypto_uint##N##_bottombit_01(T x) { return (T)(x & 1); }               \
  __device__ static inline T crypto_uint##N##_bitmod_mask(T x, T s) {                                \
    return (T)-(T)((x >> (s & (N - 1))) & 1); }

REF_INT_HELPERS(8, int8_t)
REF_INT_HELPERS(16, int16_t)
REF_INT_HELPERS(32, int32_t)
REF_INT_HELPERS(64, int64_t)
REF_UINT_HELPERS(16, uint16_t)
REF_UINT_HELPERS(32, uint32_t)
REF_UINT_HELPERS(64, uint64_t)

__device__ static inline uint64_t crypto_uint64_load(const unsigned char *x) {
  uint64_t r = 0;
  for (int i = 7; i >= 0; i--) r = (r << 8) | x[i];
  return r;
}
__device__ static inline uint32_t crypto_uint32_load(const unsigned char *x) {
  return (uint32_t)x[0] | ((uint32_t)x[1] << 8) | ((uint32_t)x[2] << 16) | ((uint32_t)x[3] << 24);
}
__device__ static inline void crypto_uint16_store(unsigned char *x, uint16_t u) {
  x[0] = (unsigned char)u;
  x[1] = (unsigned char)(u >> 8);
}

#define crypto_declassify(p, n) ((void)0)

__device__ static inline void crypto_xof_bitwrite16(unsigned char *h, long long hlen,
                                                   const unsigned char *m, long long mlen) {
  for (long long i = 0; i < hlen; i++) h[i] = 0;
  for (long long i = 0; i < mlen; i += 2) {
    uint16_t pos = (uint16_t)(m[i] | (m[i + 1] << 8));
    if ((pos >> 3) < hlen) h[pos >> 3] |= (unsigned char)(1 << (pos & 7));
  }
}

/* the reference sorts uint16_t index arrays with crypto_sort_int16 */
__device__ static inline void crypto_sort_int16(uint16_t *x, long long n) {
  crypto_sort_int16((int16_t *)x, n);
}

#endif
