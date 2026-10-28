# TensorMcEliece

Batched GPU engines for the Classic McEliece KEM on NVIDIA GPUs. The engines use the
Tensor Cores (TCs) where they help and the CUDA cores where they do not. This repository
contains the code and the benchmark scripts behind the results of the paper
*TensorMcEliece: Exploring Tensor Cores for Classic McEliece on NVIDIA GPUs*.

- **Encapsulation:** the GF(2) syndrome as a batched binary matrix multiplication on the 1-bit TC
  instruction `mma.m8n8k128.and.popc`, plus a family of exact GF(2) engines on the INT8 TCs.
- **Decapsulation:** a TC pipeline that evaluates the fixed-key linear maps (front end `F`, AFFT `V`,
  re-encryption `R`) as 1-bit TC GEMMs, and a CUDA-core pipeline with a table-based front end and a
  warp-cooperative (T)AFFT in shared memory.
- All five parameter sets: mceliece348864, 460896, 6688128, 6960119 and 8192128.

Every harness checks its session keys against the `libmceliece` CPU reference.

## Requirements

- Linux (or WSL2) with an NVIDIA GPU of compute capability 8.0 or newer (Ampere, Ada, Hopper).
- CUDA toolkit with `nvcc` (tested with CUDA 12.8 and 13.x).
- `gcc`, `make`, `python3`.
- About 2 GB of free GPU memory.

`libmceliece-20250507`, `librandombytes` and `libcpucycles` are bundled in `deps/` and are built
locally, without root.

## Quick start

```sh
export PATH=/usr/local/cuda/bin:$PATH
make deps      # once: build the CPU reference libraries into ./deps/install (about 5 min)
make           # build all harnesses for the GPU in this machine
make bench     # run everything (same as: python3 benchmark_all.py)
```

Results go to `results/<gpu>_<timestamp>/`:

| File             | Content                                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `summary.txt`  | throughput tables for every engine in the PKE and KEM settings, with peaks, and PASSED/FAILED per harness |
| `summary.json` | the same numbers, machine-readable                                                                        |
| `raw/*.txt`    | full output of every binary                                                                               |
| `env.txt`      | GPU, driver, clocks and nvcc version at the start of the run                                              |

A full run takes about 15–40 minutes, depending on the GPU.

## Where the paper's results come from

| Harness                                           | Paper results                                                                                                                                                                                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mceliece-gpu-bmma/mceliece-gpu-bmma`           | McEliece348864 encapsulation: 1-bit TC and the full INT8 family (`bs1`, `bs2`, `bs2-w4`, `bs2-m4`, `bs2-blk3`)                                                                                                                                |
| `mceliece-gpu-params/mceliece-gpu-params-<set>` | One binary per parameter set. Encapsulation (1-bit TC, `bs2-m4`, `bs2-blk3`) and decapsulation (TC and CUDA-core pipelines) for N = 256 to 65536, the per-kernel breakdown at N = 8192, the peak-throughput table and the figures for all five sets |

To run only some harnesses:

```sh
python3 benchmark_all.py --only params           # all five parameter sets
python3 benchmark_all.py --only params-348864    # one parameter set
python3 benchmark_all.py --only bmma             # McEliece348864 encapsulation engines
```

Each binary can also be run on its own, e.g. `./mceliece-gpu-params/mceliece-gpu-params-348864`.

`make -C mceliece-gpu-bmma test` runs a unit test of the 1-bit TC instruction against a CPU popcount.

## Build options

Pass these on the `make` command line:

| Variable            | Default            | Meaning                                                              |
| ------------------- | ------------------ | -------------------------------------------------------------------- |
| `ARCH`            | `native`         | GPU architecture for `nvcc`, e.g. `sm_80`, `sm_89`, `sm_90`  |
| `TRIALS`          | `30`             | timed repetitions per measurement; every reported number is the mean |
| `MCELIECE_PREFIX` | `./deps/install` | where `mceliece.h` and `libmceliece.so` are installed            |

Example: `make clean && make ARCH=sm_90`.

## Evaluation settings

- **PKE setting** (as in Wu et al., IEEE TIFS 2025). The error vector is generated before timing.
  Decapsulation stops at the recovered error, plus the weight check and the hash.
- **KEM setting.** Encapsulation and decapsulation exactly as specified, with fixed-weight sampling
  timed and the re-encryption check in decapsulation.
- One key pair is generated and shared by all operations in a batch.

## Notes

- `mceliece-gpu-params/ref/<set>.cuh` is the `vec` implementation of `libmceliece-20250507`,
  converted to CUDA device code.
- Clocks are not locked by the scripts. For stable numbers, lock them with
  `sudo nvidia-smi -lgc <min>,<max>` and reset with `sudo nvidia-smi -rgc`.
- This code targets throughput on a dedicated GPU and does not claim side-channel resistance.

## Directory layout

```
common/                shared benchmark helpers and build settings (config.mk)
deps/                  bundled libmceliece, librandombytes, libcpucycles (source tarballs)
mceliece-gpu-bmma/     TC encapsulation engines, incl. the full INT8 family (McEliece348864)
mceliece-gpu-params/   encapsulation and decapsulation for all five parameter sets
  main.cu              benchmark and verification driver
  tc_kernels.cuh       Tensor Core kernels (syndrome GEMMs; F, V, R decapsulation GEMMs)
  cc_kernels.cuh       CUDA-core decapsulation (table front end, warp-cooperative (T)AFFT)
  ref/<set>.cuh        libmceliece vec code converted to CUDA, one file per parameter set
benchmark_all.py       runs every harness and writes summary.txt / summary.json
build_deps.sh          builds deps/ into deps/install
```
