# Shared build settings for all harnesses. Override on the command line, e.g.
#   make ARCH=sm_90 TRIALS=30 MCELIECE_PREFIX=/opt/mceliece
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

NVCC   ?= nvcc
# GPU architecture: "native" (the GPU in this machine, nvcc >= 11.6) or e.g. sm_80, sm_89, sm_90.
# Only a command-line ARCH= counts: some clusters export ARCH=x86_64 in the environment.
ifneq ($(filter environment environment-override,$(origin ARCH)),)
ARCH   := native
endif
ARCH   ?= native
# Timed repetitions per measurement (results are the mean over TRIALS).
TRIALS ?= 30

# libmceliece: use ./deps/install if build_deps.sh was run, else /usr/local.
LOCAL_PREFIX    := $(ROOT_DIR)/deps/install
MCELIECE_PREFIX ?= $(if $(wildcard $(LOCAL_PREFIX)/include/mceliece.h),$(LOCAL_PREFIX),/usr/local)

NVCCFLAGS = -O3 -arch=$(ARCH) -I. -I$(ROOT_DIR)/common -I$(MCELIECE_PREFIX)/include \
            -DN_TRIALS=$(TRIALS) -diag-suppress 177 $(EXTRA_NVCCFLAGS)
# DT_RPATH (not RUNPATH) so libmceliece's own dependency librandombytes is also found in the prefix.
LDFLAGS   = -L$(MCELIECE_PREFIX)/lib -lmceliece -Xlinker --disable-new-dtags -Xlinker -rpath -Xlinker $(MCELIECE_PREFIX)/lib

COMMON_DEPS = $(wildcard *.h) $(wildcard *.cuh) $(ROOT_DIR)/common/bench_common.h $(ROOT_DIR)/common/config.mk
