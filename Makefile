# Top-level build for the TensorMcEliece benchmark suite.
#   make deps     build libcpucycles/librandombytes/libmceliece into ./deps/install
#   make          build all harnesses (a failing harness does not stop the others)
#   make bench    build, then run the full suite (python3 benchmark_all.py)
#   make dist     source-only tarball ../<this folder>.tar.gz for another machine
# Variables (passed through): ARCH=native|sm_80|sm_89|sm_90...  TRIALS=30  MCELIECE_PREFIX=...
SUBDIRS = mceliece-gpu-bmma mceliece-gpu-params

all:
	@failed=""; for d in $(SUBDIRS); do \
	  $(MAKE) -C $$d || failed="$$failed $$d"; \
	done; \
	if [ -n "$$failed" ]; then echo "*** build failed in:$$failed"; else echo "*** all harnesses built"; fi

deps:
	./build_deps.sh

bench: all
	python3 benchmark_all.py

clean:
	for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done

# Source-only tarball (no binaries, no locally built deps, no results) for another machine.
dist:
	cd .. && tar --exclude-vcs --exclude-from=$(notdir $(CURDIR))/.gitignore -czf $(notdir $(CURDIR)).tar.gz $(notdir $(CURDIR))
	@echo "wrote ../$(notdir $(CURDIR)).tar.gz"

.PHONY: all deps bench clean dist
