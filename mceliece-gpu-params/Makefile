# Multi-parameter harness: one binary per Classic McEliece parameter set, generated from
# the libmceliece vec implementation (ref/<set>.cuh; requires sm_80+).
include ../common/config.mk

SETS = 348864 460896 6688128 6960119 8192128
BINS = $(SETS:%=mceliece-gpu-params-%)

all: $(BINS)

mceliece-gpu-params-%: main.cu tc_kernels.cuh cc_kernels.cuh ref/%.cuh ref_prelude.cuh shake256.h crypto_sort.h $(COMMON_DEPS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -diag-suppress 550 -DPARAM_ID=$* \
	  -DHAS_FFT_POWERS=$$(grep -q fft_powers ref/$*.cuh && echo 1 || echo 0) \
	  -DHAS_FFT_TR_POSTPROCESS=$$(grep -q 'void postprocess(vec out' ref/$*.cuh && echo 1 || echo 0) \
	  -include ref/$*.cuh main.cu -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)

.PHONY: all clean
