# Tensor Core syndrome GEMMs: 1-bit BMMA and the INT8 WMMA family (requires sm_80+).
include ../common/config.mk

all: mceliece-gpu-bmma

mceliece-gpu-bmma: main.cu $(COMMON_DEPS)
	$(NVCC) $(NVCCFLAGS) main.cu -o $@ $(LDFLAGS)

# Unit test of the mma.m8n8k128.and.popc instruction against a CPU popcount.
test_bmma: test_bmma.cu
	$(NVCC) $(NVCCFLAGS) test_bmma.cu -o $@

test: test_bmma
	./test_bmma

clean:
	rm -f mceliece-gpu-bmma test_bmma

.PHONY: all test clean
