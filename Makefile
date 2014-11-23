CXX = g++
GCCVERSION = $(shell gcc --version | grep ^gcc | sed 's/^.* //g')

ifeq "$(GCCVERSION)" "4.9.1"
    CXX = g++-4.8
endif

CUDA_PATH=/usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)  
OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS =
# internal flags
NVCCFLAGS   := -m${OS_SIZE} ${ARCH_FLAGS}
CCFLAGS     :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))
ALL_CCFLAGS += $(pkg-config --libs --cflags opencv)
ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))


CFLAGS = -I. -I$(CUDA_PATH)/include  -std=c++11
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart 

all: build

build: termiteNest

kernel.o: kernel.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o: main.cpp
	$(CXX) $(CFLAGS)  -o  $@  -c $?

termiteNest: kernel.o main.o
	$(CXX) -L$(CUDA_PATH)/lib64 -lcudart -lcuda  $?  -o $@

clean:
	rm -f termiteNest  main.o kernel.o

clobber: clean

