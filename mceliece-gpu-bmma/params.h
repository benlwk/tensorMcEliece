#ifndef PARAMS_BMMA_348864_H
#define PARAMS_BMMA_348864_H

#include "gf_params.h"
#include <stdint.h>

#define SYS_N 3488
#define SYS_T 64

#define COND_BYTES ((1 << (GFBITS-4))*(2*GFBITS - 1))
#define IRR_BYTES (SYS_T * 2)

#define PK_NROWS (SYS_T*GFBITS)
#define PK_NCOLS (SYS_N - PK_NROWS)
#define PK_ROW_BYTES ((PK_NCOLS + 7)/8)

#define SYND_BYTES ((PK_NROWS + 7)/8)
#define CT_SIZE SYND_BYTES
#define KEY_SIZE 32
#define SK_SIZE (40 + IRR_BYTES + COND_BYTES + SYS_N/8)
#define PK_SIZE (PK_NROWS * PK_ROW_BYTES)

#endif
