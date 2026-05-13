# Build a specific candidate: make CAND=v1_baseline
# Resulting shared object: _C_<CAND>.cpython-...so
#
# Example:
#   make CAND=v1_baseline
#   make all-clean

GPU := H100
CAND ?= v1_baseline

SRC := $(CAND).cu
EXT := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT := _C_$(CAND)$(EXT)

NVCC := nvcc

THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
THIS_DIR := $(dir $(THIS_MAKEFILE))
THUNDERKITTENS_ROOT := $(abspath $(THIS_DIR)/../../../..)

NVCCFLAGS += -std=c++20 -O3 --use_fast_math
NVCCFLAGS += -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -forward-unknown-to-host-compiler
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype -I.

NVCCFLAGS += -DNDEBUG
NVCCFLAGS += -lineinfo
NVCCFLAGS += -ftemplate-backtrace-limit=0

PYTHON_VERSION ?= $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INCLUDES ?= $(shell python3 -c "import sysconfig; print('-I', sysconfig.get_path('include'), sep='')")
PYTHON_LIBDIR ?= $(shell python3 -c "import sysconfig; print('-L', sysconfig.get_config_var('LIBDIR'), sep='')")
PYBIND_INCLUDES ?= $(shell python3 -m pybind11 --includes)
PYTORCH_INCLUDES ?= $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTORCH_LIBDIR ?= $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")

NVCCFLAGS += -shared -fPIC
NVCCFLAGS += -diag-suppress 3189
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H
NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C_$(CAND)
NVCCFLAGS += -DTORCH_COMPILE
NVCCFLAGS += $(PYTHON_INCLUDES) ${PYTHON_LIBDIR} $(PYBIND_INCLUDES)
NVCCFLAGS += ${PYTORCH_LIBDIR} $(PYTORCH_INCLUDES)
NVCCFLAGS += -lpython${PYTHON_VERSION}
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10

NVCCFLAGS += -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a

all: $(OUT)

$(OUT): $(SRC) common_wrapper.cuh
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(OUT)

ptx: $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) --ptx -o $(OUT).ptx

clean:
	rm -f $(OUT) $(OUT).ptx

# Build every vN_*.cu in the dir.
all-cand:
	@for f in v*.cu; do \
		name=$${f%.cu}; \
		echo "--- building $$name ---"; \
		$(MAKE) -f $(THIS_MAKEFILE) CAND=$$name || exit 1; \
	done

all-clean:
	rm -f _C_v*.so *.ptx

.PHONY: all clean ptx all-cand all-clean
