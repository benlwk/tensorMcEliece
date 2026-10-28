#!/usr/bin/env bash
# Build libcpucycles, librandombytes and libmceliece from the bundled tarballs
# into ./deps/install (no root needed). The CUDA harnesses link against this
# prefix automatically; set MCELIECE_PREFIX to use a system install instead.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="$ROOT/deps"
PREFIX="${PREFIX:-$DEPS/install}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
SRC="$DEPS/src"
mkdir -p "$SRC" "$PREFIX"

export CPATH="$PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$PREFIX/bin:$PATH"

build() {
  local name="$1"; shift
  echo "=== building $name -> $PREFIX"
  rm -rf "$SRC/$name"
  tar -xzf "$DEPS/$name.tar.gz" -C "$SRC"
  ( cd "$SRC/$name" && ./configure --prefix="$PREFIX" "$@" && make -j"$JOBS" install )
}

build libcpucycles-20260105
build librandombytes-20240318
# libmceliece enables valgrind (constant-time test) support by default, which needs
# <valgrind/memcheck.h>; without it every pk_gen.c fails and the build stops with an
# AssertionError in scripts/selected. The benchmarks do not use it.
build libmceliece-20250507 --no-valgrind

test -f "$PREFIX/include/mceliece.h" && test -e "$PREFIX/lib/libmceliece.so"
echo "=== done: $PREFIX"
